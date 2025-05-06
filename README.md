# MeloTTS Server

This is a socket server implementation for MeloTTS that allows you to generate text-to-speech audio over a network connection.

## Features

- Multi-language support: English, Spanish, French, Chinese, Japanese, and Korean
- Multiple speaker support for English (American, British, Indian, Australian, Default)
- Adjustable speech speed
- Automatic text chunking for longer passages
- Socket-based client-server architecture for network use
- Lazy-loading of language models to save memory

## Prerequisites

- Python 3.7+
- MeloTTS package and its dependencies

## Installation

1. Install MeloTTS:
```
pip install melotts
```

2. Install server requirements:
```
pip install loguru
```

## Usage

### Starting the Server

Run the server with default settings:

```bash
python start_tts_server.py
```

Available command-line options:

- `--host`: Host address to bind to (default: 0.0.0.0)
- `--port`: Port to listen on (default: 5000)
- `--sample-rate`: Audio sample rate (default: 24000)
- `--language`: Default language (EN, ES, FR, ZH, JP, KR) (default: EN)
- `--device`: Device to use (auto, cpu, cuda, mps) (default: auto)
- `--speed`: Speech speed multiplier (default: 1.0)

Example with custom settings:

```bash
python start_tts_server.py --host 127.0.0.1 --port 8000 --language ES --device cuda --speed 1.2
```

### Client Protocol

To communicate with the server, clients should:

1. Connect to the server socket
2. Send a request with the format: `language|speaker|speed|text` (speed is optional)
3. Receive audio data in chunks

Example request format:
```
EN|EN-US|1.2|This is sample text to be converted to speech.
```

Or without speed parameter:
```
ZH|ZH|这是示例文本，将被转换为语音。
```

The server will respond with:
1. 4-byte length of audio data (big endian)
2. Audio data bytes
3. Repeats for each text chunk
4. End of transmission marker (4 bytes with value 0)

## Supported Languages and Speakers

- English (EN)
  - EN-Default (default English speaker)
  - EN-US (American accent)
  - EN-BR (British accent)
  - EN_INDIA (Indian accent)
  - EN-AU (Australian accent)
- Spanish (ES)
- French (FR)
- Chinese (ZH)
- Japanese (JP)
- Korean (KR)

For languages other than English, use the language code as the speaker ID.

## Error Handling

If an error occurs during processing, the server will send a length value of 0, indicating no audio data is available for that chunk. Clients should handle this appropriately.

## Performance Considerations

- The server loads models lazily to minimize memory usage
- Initial requests for a new language may take longer as models are loaded
- Subsequent requests for the same language will be faster
- Models remain loaded until the server is stopped 