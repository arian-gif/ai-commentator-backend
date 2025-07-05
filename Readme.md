# AI Sports Commentary Backend (FastAPI)

This backend API accepts a sports highlight video (with or without audio), extracts either a transcript or visual features, and uses Gemini (Google Generative AI) to generate dramatic sports commentary.

## Features
- Accepts video file uploads
- Uses OpenAI Whisper for audio transcription (if audio present)
- Uses PyTorch to analyze visual content (if no audio)
- Generates commentary using Gemini 2.0 Flash model

## Tech Stack
- Python
- FastAPI
- Whisper
- PyTorch (MobileNetV2)
- Google Generative AI API (Gemini)
- OpenCV

