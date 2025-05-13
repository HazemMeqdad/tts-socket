import os
import io
import numpy as np
from fastapi import FastAPI, Request, Response, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from typing import Optional, List
from loguru import logger

from melo.api import TTS
from melo.split_utils import split_sentence

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5800
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_MAX_TEXT_LENGTH = 200

app = FastAPI(
    title="MeloTTS API",
    description="API for text-to-speech using MeloTTS",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global state to store TTS models
tts_models = {}
default_language = "EN"
default_device = "auto"
default_speed = 1.0
supported_languages = ["EN", "ES", "FR", "ZH", "JP", "KR"]


class TTSRequest(BaseModel):
    text: str
    language: str = Field(
        default="EN", description="Language code (EN, ES, FR, ZH, JP, KR)"
    )
    speaker: str = Field(default="EN-Default", description="Speaker ID")
    speed: Optional[float] = Field(default=1.0, description="Speech speed multiplier")


class SplitTextRequest(BaseModel):
    text: str
    language: str = Field(
        default="EN", description="Language code (EN, ES, FR, ZH, JP, KR)"
    )
    max_length: Optional[int] = Field(
        default=DEFAULT_MAX_TEXT_LENGTH, description="Maximum length of each chunk"
    )


def load_model(language):
    """Load a TTS model for the specified language"""
    if language not in supported_languages:
        logger.warning(
            f"Unsupported language {language}, falling back to {default_language}"
        )
        language = default_language

    if language not in tts_models:
        logger.info(f"Loading model for language: {language}")
        model = TTS(language=language, device=default_device)
        # Warm-up
        logger.debug(f"Warming up {language} model")
        model.tts_to_file(
            "Warm-up",
            list(model.hps.data.spk2id.values())[0],
            output_path=None,
            quiet=True,
        )
        tts_models[language] = model
        logger.info(f"Model for {language} loaded successfully")
    return tts_models[language]


def get_speaker_id(model, speaker_name=None):
    """Get speaker ID from the model"""
    speaker_ids = model.hps.data.spk2id
    logger.debug(f"Available speakers: {list(speaker_ids.keys())}")
    if not speaker_name or speaker_name not in speaker_ids:
        fallback = speaker_ids.get("EN-Default") or list(speaker_ids.values())[0]
        logger.debug(
            f"Speaker '{speaker_name}' not found, using fallback speaker ID: {fallback}"
        )
        return fallback
    logger.debug(f"Using speaker ID for '{speaker_name}': {speaker_ids[speaker_name]}")
    return speaker_ids[speaker_name]


def split_text(text, language, max_length=DEFAULT_MAX_TEXT_LENGTH):
    """Split text into chunks for processing"""
    if not text:
        logger.warning("Empty text received, returning empty chunks list")
        return []

    logger.debug(f"Splitting text of length {len(text)} using language '{language}'")
    chunks = split_sentence(text, language_str=language)
    logger.debug(f"Initial split returned {len(chunks)} chunks")

    result = []
    for i, chunk in enumerate(chunks):
        if len(chunk) <= max_length:
            result.append(chunk)
        else:
            logger.debug(
                f"Chunk {i} exceeds max length ({len(chunk)} > {max_length}), splitting further"
            )
            import re

            sentences = re.split(r"([.!?])", chunk)
            current = ""
            for i in range(0, len(sentences), 2):
                sentence = sentences[i]
                if i + 1 < len(sentences):
                    sentence += sentences[i + 1]
                if len(current) + len(sentence) > max_length:
                    if current:
                        result.append(current.strip())
                    current = sentence
                else:
                    current += " " + sentence
            if current:
                result.append(current.strip())

    logger.debug(f"Final split: {len(result)} chunks: {[len(c) for c in result]}")
    return result


def generate_audio(text, language, speaker_name, speed=None):
    """Generate audio for the given text"""
    speed = speed or default_speed
    logger.debug(
        f"Generating audio for text: '{text[:50]}...' with language={language}, speaker={speaker_name}, speed={speed}"
    )

    try:
        model = load_model(language)
        speaker_id = get_speaker_id(model, speaker_name)

        logger.debug(f"Starting TTS generation with speaker_id={speaker_id}")
        audio_numpy = model.tts_to_file(
            text, speaker_id, output_path=None, speed=speed, quiet=True
        )

        logger.debug(f"TTS generation complete, audio shape: {audio_numpy.shape}")
        audio_bytes = (audio_numpy * 32767).astype(np.int16).tobytes()
        logger.debug(f"Audio converted to bytes, size: {len(audio_bytes)} bytes")
        return audio_bytes
    except Exception as e:
        logger.error(f"Error in generate_audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint showing API information"""
    return {
        "name": "MeloTTS API",
        "version": "1.0.0",
        "description": "Text-to-speech API using MeloTTS",
        "endpoints": {
            "POST /tts": "Generate speech from text",
            "POST /split": "Split text into manageable chunks",
            "GET /languages": "List supported languages",
            "GET /speakers/{language}": "List available speakers for a language",
        },
    }


@app.get("/languages")
async def get_languages():
    """Get list of supported languages"""
    return {"languages": supported_languages}


@app.get("/speakers/{language}")
async def get_speakers(language: str):
    """Get available speakers for a language"""
    if language not in supported_languages:
        raise HTTPException(
            status_code=404, detail=f"Language {language} not supported"
        )

    try:
        model = load_model(language)
        return {"speakers": list(model.hps.data.spk2id.keys())}
    except Exception as e:
        logger.error(f"Error getting speakers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get speakers: {str(e)}")


@app.post("/split")
async def split_text_endpoint(request: SplitTextRequest):
    """Split text into chunks for processing"""
    try:
        chunks = split_text(request.text, request.language, request.max_length)
        return {"chunks": chunks, "count": len(chunks)}
    except Exception as e:
        logger.error(f"Error splitting text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to split text: {str(e)}")


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Generate speech from text and return as audio file"""
    try:
        # Split text if needed
        text_chunks = split_text(request.text, request.language)

        if not text_chunks:
            raise HTTPException(status_code=400, detail="No valid text to process")

        # For single chunks, return directly
        if len(text_chunks) == 1:
            audio_data = generate_audio(
                text_chunks[0], request.language, request.speaker, request.speed
            )
            return StreamingResponse(
                io.BytesIO(audio_data),
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=speech.wav"},
            )

        # For multiple chunks, we need to concatenate the audio
        logger.info(f"Processing {len(text_chunks)} chunks")
        all_audio = []

        for i, chunk in enumerate(text_chunks):
            logger.debug(
                f"Processing chunk {i+1}/{len(text_chunks)}: '{chunk[:50]}...'"
            )
            audio_data = generate_audio(
                chunk, request.language, request.speaker, request.speed
            )
            all_audio.append(np.frombuffer(audio_data, dtype=np.int16))

        # Concatenate all audio chunks
        combined_audio = np.concatenate(all_audio)
        audio_bytes = combined_audio.tobytes()

        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"},
        )
    except Exception as e:
        logger.error(f"Error in text_to_speech: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


@app.post("/tts-legacy")
async def legacy_tts(request: Request):
    """Legacy format endpoint for compatibility with older clients"""
    body = await request.body()
    text = body.decode("utf-8", errors="ignore")

    # Parse the legacy format: LANGUAGE|SPEAKER|SPEED|TEXT
    parts = text.split("|", 3)
    if len(parts) < 2:
        raise HTTPException(
            status_code=400,
            detail="Invalid request format. Expected: LANGUAGE|SPEAKER|[SPEED|]TEXT",
        )

    language = parts[0]
    speaker = parts[1]

    if len(parts) >= 4:
        try:
            speed = float(parts[2])
            text = parts[3]
        except ValueError:
            speed = default_speed
            text = parts[2] + "|" + parts[3]
    else:
        speed = default_speed
        text = parts[2]

    # Create a standard request and process it
    std_request = TTSRequest(text=text, language=language, speaker=speaker, speed=speed)
    return await text_to_speech(std_request)


def start_server(
    host=DEFAULT_HOST,
    port=DEFAULT_PORT,
    sample_rate=DEFAULT_SAMPLE_RATE,
    language="EN",
    device="auto",
    speed=1.0,
):
    """Start the FastAPI server"""
    global default_language, default_device, default_speed

    default_language = language
    default_device = device
    default_speed = speed

    logger.info(f"Starting MeloTTS FastAPI server on {host}:{port}")
    logger.info(f"Default language: {language}, device: {device}")

    # Load the default language model at startup
    load_model(language)

    # Start the server
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    logger.info("Starting MeloTTS FastAPI server")
    start_server()
