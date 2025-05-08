#!/usr/bin/env python3
"""
Script to start the MeloTTS server.
Run this script in a separate terminal/process before using the MeloTTS client.
"""

import os
import sys
import argparse
from loguru import logger

# Add the parent directory to the path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from melotts_server import MeloTTSServer

def main():
    parser = argparse.ArgumentParser(description="Start the MeloTTS server")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host address to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=5800, help="Port to listen on (default: 5800)"
    )
    parser.add_argument(
        "--sample-rate", type=int, default=24000, help="Audio sample rate (default: 24000)"
    )
    parser.add_argument(
        "--language", default="EN", choices=["EN", "ES", "FR", "ZH", "JP", "KR"],
        help="Default language for TTS (default: EN)"
    )
    parser.add_argument(
        "--device", default="auto", help="Device to use for inference (auto, cpu, cuda, mps) (default: auto)"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Speech speed multiplier (default: 1.0)"
    )
    parser.add_argument(
        "--use-ssl", action="store_true", help="Enable SSL/TLS encryption"
    )
    parser.add_argument(
        "--cert-file", help="Path to SSL certificate file (required if --use-ssl is set)"
    )
    parser.add_argument(
        "--key-file", help="Path to SSL private key file (required if --use-ssl is set)"
    )
    
    args = parser.parse_args()
    
    if args.use_ssl and (not args.cert_file or not args.key_file):
        parser.error("--cert-file and --key-file are required when --use-ssl is set")
    
    logger.info("Starting MeloTTS server...")
    logger.info(f"Host: {args.host}, Port: {args.port}")
    logger.info(f"Sample rate: {args.sample_rate}")
    logger.info(f"Default language: {args.language}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Speed: {args.speed}")
    if args.use_ssl:
        logger.info(f"SSL enabled with cert: {args.cert_file}, key: {args.key_file}")
    
    # Create and start the server
    server = MeloTTSServer(
        host=args.host,
        port=args.port,
        sample_rate=args.sample_rate,
        language=args.language,
        device=args.device,
        speed=args.speed,
        use_ssl=args.use_ssl,
        cert_file=args.cert_file,
        key_file=args.key_file,
    )
    
    server.start()

if __name__ == "__main__":
    main() 