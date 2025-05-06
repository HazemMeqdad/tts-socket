import threading
import socket
import numpy as np
import re
from loguru import logger

from melo.api import TTS
from melo.split_utils import split_sentence

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5000
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_MAX_TEXT_LENGTH = 200


class MeloTTSServer:
    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT, sample_rate=DEFAULT_SAMPLE_RATE,
                 language="EN", device="auto", speed=1.0):
        self.host = host
        self.port = port
        self.sample_rate = sample_rate
        self.language = language
        self.device = device
        self.speed = speed
        self.server_socket = None
        self.tts_models = {}
        self.supported_languages = ["EN", "ES", "FR", "ZH", "JP", "KR"]
        self._initialize_tts()

    def _initialize_tts(self):
        logger.info("Initializing MeloTTS models...")
        try:
            self._load_model(self.language)
        except Exception as e:
            logger.error(f"Failed to initialize TTS: {e}")
            raise

    def _load_model(self, language):
        if language not in self.supported_languages:
            logger.warning(f"Unsupported language {language}, falling back to {self.language}")
            language = self.language

        if language not in self.tts_models:
            logger.info(f"Loading model for language: {language}")
            model = TTS(language=language, device=self.device)
            # Warm-up
            model.tts_to_file("Warm-up", list(model.hps.data.spk2id.values())[0], output_path=None, quiet=True)
            self.tts_models[language] = model
        return self.tts_models[language]

    def _get_speaker_id(self, model, speaker_name=None):
        speaker_ids = model.hps.data.spk2id
        if not speaker_name or speaker_name not in speaker_ids:
            fallback = speaker_ids.get("EN-Default") or list(speaker_ids.values())[0]
            logger.debug(f"Using fallback speaker: {fallback}")
            return fallback
        return speaker_ids[speaker_name]

    def split_text(self, text, language, max_length=DEFAULT_MAX_TEXT_LENGTH):
        if not text:
            return []

        chunks = split_sentence(text, language_str=language)
        result = []
        for chunk in chunks:
            if len(chunk) <= max_length:
                result.append(chunk)
            else:
                sentences = re.split(r'([.!?])', chunk)
                current = ""
                for i in range(0, len(sentences), 2):
                    sentence = sentences[i]
                    if i+1 < len(sentences): sentence += sentences[i+1]
                    if len(current) + len(sentence) > max_length:
                        if current: result.append(current.strip())
                        current = sentence
                    else:
                        current += " " + sentence
                if current: result.append(current.strip())
        return result

    def generate_audio_stream(self, text, language, speaker_name, speed=None):
        speed = speed or self.speed
        model = self._load_model(language)
        speaker_id = self._get_speaker_id(model, speaker_name)
        audio_numpy = model.tts_to_file(text, speaker_id, output_path=None, speed=speed, quiet=True)
        return (audio_numpy * 32767).astype(np.int16).tobytes()

    def handle_client(self, client_socket):
        try:
            buffer = b""
            while True:
                data = client_socket.recv(1024)
                if not data:
                    break
                buffer += data

                if b"|" in buffer:
                    try:
                        request = buffer.decode("utf-8", errors="ignore")
                        parts = request.split("|", 3)
                        if len(parts) < 3:
                            logger.error("Invalid request format")
                            client_socket.sendall((0).to_bytes(4, byteorder="big"))
                            buffer = b""
                            continue

                        language, speaker = parts[0], parts[1]
                        speed = float(parts[2]) if len(parts) == 4 else self.speed
                        text = parts[3] if len(parts) == 4 else parts[2]

                        chunks = self.split_text(text, language)
                        logger.info(f"Processing {len(chunks)} chunks [{language}, speaker: {speaker}]")

                        for chunk in chunks:
                            try:
                                audio_data = self.generate_audio_stream(chunk, language, speaker, speed)
                                client_socket.sendall(len(audio_data).to_bytes(4, byteorder="big"))
                                client_socket.sendall(audio_data)
                            except Exception as e:
                                logger.error(f"Chunk error: {e}")
                                client_socket.sendall((0).to_bytes(4, byteorder="big"))

                        client_socket.sendall((0).to_bytes(4, byteorder="big"))

                    except Exception as e:
                        logger.error(f"Processing error: {e}")
                        client_socket.sendall((0).to_bytes(4, byteorder="big"))
                    buffer = b""
        finally:
            client_socket.close()
            logger.info("Client disconnected")

    def start(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            logger.info(f"Server listening on {self.host}:{self.port}")

            while True:
                client_socket, addr = self.server_socket.accept()
                logger.info(f"Connected: {addr}")
                threading.Thread(target=self.handle_client, args=(client_socket,), daemon=True).start()

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            if self.server_socket:
                self.server_socket.close()


if __name__ == "__main__":
    server = MeloTTSServer()
    server.start()
