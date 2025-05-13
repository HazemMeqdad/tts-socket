#!/usr/bin/env python3
"""
Simple client to test the MeloTTS FastAPI server.
This can be used to generate speech from text and save it to a WAV file.
"""

import argparse
import requests
import json
import time
import wave
import sys
import os
from loguru import logger

def list_languages(server_url):
    """Get a list of supported languages"""
    response = requests.get(f"{server_url}/languages")
    if response.status_code == 200:
        return response.json()["languages"]
    else:
        logger.error(f"Failed to get languages: {response.status_code} - {response.text}")
        return []

def list_speakers(server_url, language):
    """Get a list of available speakers for a language"""
    response = requests.get(f"{server_url}/speakers/{language}")
    if response.status_code == 200:
        return response.json()["speakers"]
    else:
        logger.error(f"Failed to get speakers: {response.status_code} - {response.text}")
        return []

def generate_speech(server_url, text, language, speaker, speed, output_file):
    """Generate speech from text and save to WAV file"""
    logger.info(f"Generating speech for text: '{text[:50]}...' ({len(text)} chars)")
    logger.info(f"Language: {language}, Speaker: {speaker}, Speed: {speed}")
    
    # Create the request data
    data = {
        "text": text,
        "language": language,
        "speaker": speaker,
        "speed": speed
    }
    
    # Log start time
    start_time = time.time()
    
    # Make the request
    try:
        response = requests.post(
            f"{server_url}/tts", 
            json=data,
            stream=True
        )
        
        if response.status_code == 200:
            # Save the audio to file
            logger.info(f"Response received, saving to {output_file}")
            with open(output_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            duration = time.time() - start_time
            logger.info(f"Speech generated successfully in {duration:.2f} seconds")
            return True
        else:
            logger.error(f"Failed to generate speech: {response.status_code}")
            logger.error(response.text)
            return False
            
    except Exception as e:
        logger.error(f"Error sending request: {str(e)}")
        return False

def generate_speech_legacy(server_url, text, language, speaker, speed, output_file):
    """Generate speech using the legacy format"""
    logger.info(f"Generating speech for text (legacy format): '{text[:50]}...' ({len(text)} chars)")
    logger.info(f"Language: {language}, Speaker: {speaker}, Speed: {speed}")
    
    # Format the text according to the legacy format
    formatted_text = f"{language}|{speaker}|{speed}|{text}"
    
    # Log start time
    start_time = time.time()
    
    # Make the request
    try:
        response = requests.post(
            f"{server_url}/tts-legacy", 
            data=formatted_text,
            stream=True
        )
        
        if response.status_code == 200:
            # Save the audio to file
            logger.info(f"Response received, saving to {output_file}")
            with open(output_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            duration = time.time() - start_time
            logger.info(f"Speech generated successfully in {duration:.2f} seconds")
            return True
        else:
            logger.error(f"Failed to generate speech: {response.status_code}")
            logger.error(response.text)
            return False
            
    except Exception as e:
        logger.error(f"Error sending request: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="MeloTTS FastAPI Client")
    parser.add_argument(
        "--server", default="http://localhost:5800", help="Server URL (default: http://localhost:5800)"
    )
    parser.add_argument(
        "--language", default="EN", help="Language code (default: EN)"
    )
    parser.add_argument(
        "--speaker", default="EN-Default", help="Speaker ID (default: EN-Default)"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Speech speed multiplier (default: 1.0)"
    )
    parser.add_argument(
        "--output", default="output.wav", help="Output WAV file (default: output.wav)"
    )
    parser.add_argument(
        "--text", help="Text to convert to speech"
    )
    parser.add_argument(
        "--file", help="Text file to convert to speech"
    )
    parser.add_argument(
        "--list-languages", action="store_true", help="List available languages"
    )
    parser.add_argument(
        "--list-speakers", action="store_true", help="List available speakers for the specified language"
    )
    
    args = parser.parse_args()
    
    # Check if server is alive
    try:
        response = requests.get(f"{args.server}/")
        if response.status_code != 200:
            logger.error(f"Server returned error: {response.status_code} - {response.text}")
            return
    except Exception as e:
        logger.error(f"Failed to connect to server at {args.server}: {str(e)}")
        return
    
    # List languages if requested
    if args.list_languages:
        languages = list_languages(args.server)
        if languages:
            print("Available languages:")
            for lang in languages:
                print(f"  {lang}")
        return
    
    # List speakers if requested
    if args.list_speakers:
        speakers = list_speakers(args.server, args.language)
        if speakers:
            print(f"Available speakers for {args.language}:")
            for speaker in speakers:
                print(f"  {speaker}")
        return
    
    # Get text from either command line or file
    if args.text:
        text = args.text
    elif args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            logger.error(f"Failed to read text file {args.file}: {str(e)}")
            return
    else:
        logger.error("Either --text or --file must be provided")
        return
    
    generate_speech(args.server, text, args.language, args.speaker, args.speed, args.output)

if __name__ == "__main__":
    main() 