import threading
import socket
import numpy as np
import re
import base64
import hashlib
import ssl
from loguru import logger

from melo.api import TTS
from melo.split_utils import split_sentence

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5000
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_MAX_TEXT_LENGTH = 200

# WebSocket constants
WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
WEBSOCKET_RESPONSE = (
    "HTTP/1.1 101 Switching Protocols\r\n"
    "Upgrade: websocket\r\n"
    "Connection: Upgrade\r\n"
    "Sec-WebSocket-Accept: {}\r\n"
    "\r\n"
)


class MeloTTSServer:
    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT, sample_rate=DEFAULT_SAMPLE_RATE,
                 language="EN", device="auto", speed=1.0, use_ssl=False, cert_file=None, key_file=None):
        self.host = host
        self.port = port
        self.sample_rate = sample_rate
        self.language = language
        self.device = device
        self.speed = speed
        self.server_socket = None
        self.tts_models = {}
        self.supported_languages = ["EN", "ES", "FR", "ZH", "JP", "KR"]
        self.use_ssl = use_ssl
        self.cert_file = cert_file
        self.key_file = key_file
        logger.info(f"Initializing server with: host={host}, port={port}, language={language}, device={device}, speed={speed}, ssl={use_ssl}")
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
            logger.debug(f"Warming up {language} model")
            model.tts_to_file("Warm-up", list(model.hps.data.spk2id.values())[0], output_path=None, quiet=True)
            self.tts_models[language] = model
            logger.info(f"Model for {language} loaded successfully")
        return self.tts_models[language]

    def _get_speaker_id(self, model, speaker_name=None):
        speaker_ids = model.hps.data.spk2id
        logger.debug(f"Available speakers: {list(speaker_ids.keys())}")
        if not speaker_name or speaker_name not in speaker_ids:
            fallback = speaker_ids.get("EN-Default") or list(speaker_ids.values())[0]
            logger.debug(f"Speaker '{speaker_name}' not found, using fallback speaker ID: {fallback}")
            return fallback
        logger.debug(f"Using speaker ID for '{speaker_name}': {speaker_ids[speaker_name]}")
        return speaker_ids[speaker_name]

    def split_text(self, text, language, max_length=DEFAULT_MAX_TEXT_LENGTH):
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
                logger.debug(f"Chunk {i} exceeds max length ({len(chunk)} > {max_length}), splitting further")
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
        
        logger.debug(f"Final split: {len(result)} chunks: {[len(c) for c in result]}")
        return result

    def generate_audio_stream(self, text, language, speaker_name, speed=None):
        speed = speed or self.speed
        logger.debug(f"Generating audio for text: '{text[:50]}...' with language={language}, speaker={speaker_name}, speed={speed}")
        
        try:
            model = self._load_model(language)
            speaker_id = self._get_speaker_id(model, speaker_name)
            
            logger.debug(f"Starting TTS generation for chunk with speaker_id={speaker_id}")
            audio_numpy = model.tts_to_file(text, speaker_id, output_path=None, speed=speed, quiet=True)
            
            logger.debug(f"TTS generation complete, audio shape: {audio_numpy.shape}, min: {audio_numpy.min()}, max: {audio_numpy.max()}")
            audio_bytes = (audio_numpy * 32767).astype(np.int16).tobytes()
            logger.debug(f"Audio converted to bytes, size: {len(audio_bytes)} bytes")
            return audio_bytes
        except Exception as e:
            logger.error(f"Error in generate_audio_stream: {str(e)}")
            raise

    def _perform_websocket_handshake(self, client_socket, request):
        """Complete the WebSocket handshake if this is a WebSocket connection"""
        # Check if this is a WebSocket handshake request
        if not "Upgrade: websocket" in request or not "Sec-WebSocket-Key:" in request:
            return False
        
        logger.info("WebSocket handshake request detected")
        
        # Extract the Sec-WebSocket-Key header
        key_match = re.search(r'Sec-WebSocket-Key: ([A-Za-z0-9+/=]+)', request)
        if not key_match:
            logger.error("Invalid WebSocket handshake: missing Sec-WebSocket-Key")
            return False
        
        ws_key = key_match.group(1).strip()
        
        # Calculate the response key
        response_key = base64.b64encode(
            hashlib.sha1((ws_key + WEBSOCKET_GUID).encode('utf-8')).digest()
        ).decode('utf-8')
        
        # Send the handshake response
        handshake_response = WEBSOCKET_RESPONSE.format(response_key)
        client_socket.sendall(handshake_response.encode('utf-8'))
        logger.info("WebSocket handshake completed successfully")
        return True

    def _decode_websocket_frame(self, data):
        """Decode a WebSocket frame to extract the payload data"""
        if len(data) < 6:
            return None
        
        # Basic parsing of a WebSocket frame
        b1, b2 = data[0], data[1]
        
        fin = (b1 & 0x80) >> 7
        opcode = b1 & 0x0F
        
        # Check for control frames
        if opcode == 0x8:  # Close frame
            logger.debug("Received WebSocket Close frame")
            return None
        if opcode == 0x9:  # Ping frame
            logger.debug("Received WebSocket Ping frame")
            return None
        if opcode == 0xA:  # Pong frame
            logger.debug("Received WebSocket Pong frame")
            return None
        
        mask = (b2 & 0x80) >> 7
        payload_len = b2 & 0x7F
        
        if payload_len == 126:
            payload_len = int.from_bytes(data[2:4], byteorder='big')
            mask_start = 4
        elif payload_len == 127:
            payload_len = int.from_bytes(data[2:10], byteorder='big')
            mask_start = 10
        else:
            mask_start = 2
            
        if mask:
            # Extract the masking key
            mask_key = data[mask_start:mask_start+4]
            data_start = mask_start + 4
            
            # Extract the payload data
            payload_data = bytearray(data[data_start:data_start+payload_len])
            
            # Unmask the data
            for i in range(len(payload_data)):
                payload_data[i] ^= mask_key[i % 4]
            
            return bytes(payload_data)
        else:
            return data[mask_start:mask_start+payload_len]

    def _encode_websocket_frame(self, data, opcode=0x2):  # 0x2 is binary data
        """Encode data as a WebSocket frame"""
        # The first byte: FIN bit (1) + reserved bits (000) + opcode (4 bits)
        b1 = 0x80 | opcode  # 0x80 for FIN bit set to 1
        
        # The second byte: MASK bit (0 for server) + payload length
        length = len(data)
        if length < 126:
            header = bytes([b1, length])
        elif length < 65536:
            header = bytes([b1, 126]) + length.to_bytes(2, byteorder='big')
        else:
            header = bytes([b1, 127]) + length.to_bytes(8, byteorder='big')
            
        # Return frame
        return header + data

    def handle_client(self, client_socket):
        client_addr = client_socket.getpeername()
        logger.info(f"Starting to handle client: {client_addr}")
        
        # State variables
        is_websocket = False
        initial_buffer = b""
        buffer = b""
        
        try:
            # Initial phase: determine connection type
            initial_data = client_socket.recv(1024)
            if not initial_data:
                logger.info(f"Client {client_addr} disconnected immediately")
                return
            
            initial_buffer = initial_data
            
            # Check if this might be a WebSocket handshake
            if initial_data.startswith(b'GET ') and b'Upgrade: websocket' in initial_data:
                logger.info("Potential WebSocket connection detected")
                # Accumulate more data if needed
                while b'\r\n\r\n' not in initial_buffer and len(initial_buffer) < 4096:
                    more_data = client_socket.recv(1024)
                    if not more_data:
                        break
                    initial_buffer += more_data
                
                # Try to complete WebSocket handshake
                handshake_request = initial_buffer.decode('utf-8', errors='ignore')
                is_websocket = self._perform_websocket_handshake(client_socket, handshake_request)
                
                if is_websocket:
                    logger.info(f"Established WebSocket connection with {client_addr}")
                    buffer = b""  # Clear the buffer after handshake
                else:
                    # Not a valid WebSocket request, treat as raw socket data
                    logger.info("Not a valid WebSocket request, treating as raw socket data")
                    buffer = initial_buffer
            else:
                # Not a WebSocket request, treat as raw socket data
                logger.info("Regular socket connection (not WebSocket)")
                buffer = initial_buffer
            
            # Main processing loop for both WebSocket and raw socket communications
            while True:
                data = client_socket.recv(1024)
                if not data:
                    logger.info(f"Client {client_addr} disconnected - no more data")
                    break
                
                if is_websocket:
                    # Handle WebSocket frames
                    decoded_data = self._decode_websocket_frame(data)
                    if decoded_data is None:
                        # Control frame or incomplete frame
                        continue
                    
                    logger.debug(f"Received WebSocket data: {len(decoded_data)} bytes")
                    payload = decoded_data
                else:
                    # Handle raw socket data
                    payload = data
                
                buffer += payload
                logger.debug(f"Received {len(payload)} bytes, buffer size: {len(buffer)} bytes")
                logger.debug(f"Raw buffer content (first 100 bytes): {buffer[:100]}")
                
                # Process the accumulated data
                try:
                    request_str = buffer.decode("utf-8", errors="ignore")
                    logger.debug(f"Decoded buffer (first 100 chars): {request_str[:100]}")
                    
                    # Look for the TTS request pattern
                    tts_pattern = r"(EN|ES|FR|ZH|JP|KR)\|([\w-]+)\|(?:(\d+(?:\.\d+)?)\|)?(.*)"
                    match = re.search(tts_pattern, request_str)
                    
                    if match:
                        logger.info(f"Found valid TTS request pattern in the buffer")
                        language = match.group(1)
                        speaker = match.group(2)
                        speed_str = match.group(3)
                        text = match.group(4)
                        
                        speed = float(speed_str) if speed_str else self.speed
                        
                        logger.info(f"Processing request: language={language}, speaker={speaker}, speed={speed}, text_length={len(text)}")
                        logger.debug(f"Text content: '{text[:100]}...'")
                        
                        chunks = self.split_text(text, language)
                        logger.info(f"Processing {len(chunks)} chunks for language={language}, speaker={speaker}")
                        
                        for i, chunk in enumerate(chunks):
                            try:
                                logger.debug(f"Processing chunk {i+1}/{len(chunks)}: '{chunk[:50]}...' ({len(chunk)} chars)")
                                audio_data = self.generate_audio_stream(chunk, language, speaker, speed)
                                audio_size = len(audio_data)
                                logger.debug(f"Generated audio chunk {i+1}/{len(chunks)} with size: {audio_size} bytes")
                                
                                if is_websocket:
                                    # For WebSocket, we need to wrap the data in WebSocket frames
                                    # First send the length as a 4-byte big-endian integer wrapped in a WS frame
                                    size_bytes = audio_size.to_bytes(4, byteorder="big")
                                    client_socket.sendall(self._encode_websocket_frame(size_bytes))
                                    
                                    # Then send the actual audio data wrapped in a WS frame
                                    client_socket.sendall(self._encode_websocket_frame(audio_data))
                                    logger.debug(f"Sent WebSocket audio chunk {i+1}/{len(chunks)}")
                                else:
                                    # For raw socket, send data as before
                                    client_socket.sendall(audio_size.to_bytes(4, byteorder="big"))
                                    client_socket.sendall(audio_data)
                                    logger.debug(f"Sent raw socket audio chunk {i+1}/{len(chunks)}")
                            except Exception as e:
                                logger.error(f"Error processing chunk {i+1}/{len(chunks)}: {str(e)}")
                                
                                # Send zero length to signal error
                                if is_websocket:
                                    client_socket.sendall(self._encode_websocket_frame((0).to_bytes(4, byteorder="big")))
                                else:
                                    client_socket.sendall((0).to_bytes(4, byteorder="big"))
                        
                        logger.info(f"Finished processing all chunks, sending completion marker")
                        # Send final zero-length marker to signal completion
                        if is_websocket:
                            client_socket.sendall(self._encode_websocket_frame((0).to_bytes(4, byteorder="big")))
                        else:
                            client_socket.sendall((0).to_bytes(4, byteorder="big"))
                        
                        buffer = b""  # Clear buffer after processing
                    else:
                        # If no match is found but buffer is getting large, we might have a problem
                        if len(buffer) > 10240:  # 10KB threshold
                            logger.warning(f"Buffer exceeds 10KB without valid request pattern, clearing")
                            buffer = b""
                except Exception as e:
                    logger.error(f"Error processing request: {str(e)}", exc_info=True)
                    # Send error signal
                    if is_websocket:
                        client_socket.sendall(self._encode_websocket_frame((0).to_bytes(4, byteorder="big")))
                    else:
                        client_socket.sendall((0).to_bytes(4, byteorder="big"))
                    buffer = b""
        except Exception as e:
            logger.error(f"Unexpected error in handle_client: {str(e)}", exc_info=True)
        finally:
            client_socket.close()
            logger.info(f"Client {client_addr} connection closed")

    def start(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            logger.info(f"Server listening on {self.host}:{self.port}")

            while True:
                client_socket, addr = self.server_socket.accept()
                logger.info(f"New client connected: {addr}")
                
                if self.use_ssl:
                    try:
                        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                        if self.cert_file and self.key_file:
                            ssl_context.load_cert_chain(certfile=self.cert_file, keyfile=self.key_file)
                        client_socket = ssl_context.wrap_socket(client_socket, server_side=True)
                        logger.info(f"SSL handshake completed with: {addr}")
                    except ssl.SSLError as e:
                        logger.error(f"SSL handshake failed: {str(e)}")
                        client_socket.close()
                        continue
                
                threading.Thread(target=self.handle_client, args=(client_socket,), daemon=True).start()

        except KeyboardInterrupt:
            logger.info("Server shutdown initiated by keyboard interrupt")
        except Exception as e:
            logger.error(f"Server error: {str(e)}")
        finally:
            if self.server_socket:
                self.server_socket.close()
                logger.info("Server socket closed")


if __name__ == "__main__":
    logger.info("Starting MeloTTS Server")
    server = MeloTTSServer()
    server.start()
