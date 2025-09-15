# Audio/Video Transcription and Subtitle Tools

A collection of Python scripts for audio/video transcription, translation, and subtitle generation using Whisper JAX. These tools provide an efficient workflow for creating subtitles and transcriptions from media files.

## Features

- **Audio/Video Transcription**: Convert speech to text using state-of-the-art Whisper models
- **Real-time Translation**: Translate audio content to English automatically
- **Multiple Output Formats**: Generate `.txt`, `.srt`, and `.vtt` subtitle files
- **Video Subtitle Embedding**: Add subtitles directly to video files
- **Batch Processing**: Handle large files through intelligent audio slicing

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Scripts Overview](#scripts-overview)
- [Usage Examples](#usage-examples)
- [Workflow Guide](#workflow-guide)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Prerequisites

Before installation, ensure you have:

- **Python 3.10 or higher**
- **FFmpeg** installed and accessible via system PATH
- **CUDA-compatible GPU** (recommended for faster processing)
- **12GB+ RAM** for large Whisper models (optional: use smaller models for lower specs)

### Installing FFmpeg

**Windows:**
```bash
# Using chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt update && sudo apt install ffmpeg
```

## Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd audio-video-transcription-tools
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

**Required packages:**
```bash
pip install jax jaxlib whisper-jax pydub moviepy openai
```

## Scripts Overview

### 1. `whisper_api.py`
Transcribes audio files by slicing them into manageable chunks, ideal for long recordings or API-based processing.

**Key Features:**
- Supports multiple audio formats (`.mp3`, `.wav`, `.flac`, `.m4a`)
- Configurable slice duration for memory management
- Multiple output formats (`.txt`, `.srt`, `.vtt`)
- OpenAI API integration

**Parameters:**
- `file` - Path to input audio file
- `--name` - Base name for output files (optional)
- `--slice_minutes` - Duration of each slice in minutes (default: 10)
- `--text_dir` - Output directory for transcriptions (default: current directory)
- `--format` - Output format: `vtt`, `srt`, or `text` (default: vtt)

### 2. `whisper_transcribe.py`
Direct transcription and translation using Whisper JAX, optimized for speed and accuracy.

**Key Features:**
- Real-time transcription with timestamps
- Built-in translation to English
- Supports both audio and video files
- GPU acceleration with JAX

**Parameters:**
- `--audio` - Path to input audio or video file
- `--output` - Path for output `.vtt` file
- `--translate` - Enable translation to English (optional)
- `--model` - Whisper model size: `tiny`, `small`, `medium`, `large` (default: medium)

### 3. `add_subtitle_to_video.py`
Embeds subtitle files directly into video files, creating a new video with burned-in subtitles.

**Key Features:**
- Supports `.srt` subtitle files
- Customizable subtitle styling
- Preserves original video quality
- Multiple output formats

**Parameters:**
- `input_video` - Path to input video file
- `subtitle_file` - Path to `.srt` subtitle file
- `output_video` - Path for output video with subtitles

## Usage Examples

### Basic Transcription
```bash
# Transcribe a video file
python whisper_transcribe.py --audio "meeting.mp4" --output "meeting_subtitles.vtt"

# Transcribe and translate to English
python whisper_transcribe.py --audio "spanish_audio.wav" --output "translated.vtt" --translate
```

### Processing Long Audio Files
```bash
# Slice long audio into 15-minute chunks
python whisper_api.py "long_podcast.mp3" \
  --name "podcast_episode_01" \
  --slice_minutes 15 \
  --text_dir "./transcriptions" \
  --format srt
```

### Adding Subtitles to Video
```bash
# Embed subtitles into video
python add_subtitle_to_video.py "original_video.mp4" "subtitles.srt" "video_with_subs.mp4"
```

### Complete Workflow Example
```bash
# Step 1: Transcribe video
python whisper_transcribe.py --audio "interview.mp4" --output "interview.vtt"

# Step 2: Convert VTT to SRT (if needed)
# Manual conversion or use additional script

# Step 3: Add subtitles to video
python add_subtitle_to_video.py "interview.mp4" "interview.srt" "interview_final.mp4"
```

## Workflow Guide

### For Short Files (< 30 minutes)
1. Use `whisper_transcribe.py` for direct transcription
2. Use `add_subtitle_to_video.py` to embed subtitles

### For Long Files (> 30 minutes)
1. Use `whisper_api.py` with appropriate slice duration
2. Combine output files if needed
3. Use `add_subtitle_to_video.py` for final video

### For Translation Projects
1. Use `whisper_transcribe.py` with `--translate` flag
2. Review and edit generated subtitles
3. Embed using `add_subtitle_to_video.py`

## Output Format Examples

**VTT Format:**
```
WEBVTT

00:00:00.000 --> 00:00:03.000
Hello and welcome to this presentation.

00:00:03.000 --> 00:00:07.000
Today we'll be discussing machine learning.
```

**SRT Format:**
```
1
00:00:00,000 --> 00:00:03,000
Hello and welcome to this presentation.

2
00:00:03,000 --> 00:00:07,000
Today we'll be discussing machine learning.
```

## Troubleshooting

### Common Issues

**Memory Errors:**
- Use smaller Whisper models (`tiny`, `small`, `medium`)
- Reduce `--slice_minutes` parameter
- Close other applications to free RAM

**FFmpeg Not Found:**
- Ensure FFmpeg is installed and in system PATH
- Restart terminal after FFmpeg installation
- On Windows, add FFmpeg to environment variables

**CUDA/GPU Issues:**
- Install CUDA-compatible JAX version: `pip install jax[cuda]`
- Verify GPU availability: `nvidia-smi`
- Fall back to CPU processing if needed

**Audio Format Issues:**
- Convert unsupported formats using FFmpeg:
  ```bash
  ffmpeg -i input.format output.wav
  ```

**Permission Errors:**
- Ensure write permissions for output directories
- Run with appropriate user privileges
- Check file locks from other applications

### Performance Tips

- Use GPU acceleration when available
- Choose appropriate model size for your hardware
- Process shorter segments for better memory management
- Use SSD storage for faster I/O operations

## Model Recommendations

| Hardware | Recommended Model | Processing Time | Quality |
|----------|-------------------|-----------------|---------|
| 4GB RAM  | tiny             | ~0.1x realtime  | Good    |
| 8GB RAM  | small            | ~0.2x realtime  | Better  |
| 16GB RAM | medium           | ~0.4x realtime  | Great   |
| 32GB RAM | large            | ~0.6x realtime  | Best    |

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with detailed description

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the underlying speech recognition model
- [Whisper JAX](https://github.com/sanchit-gandhi/whisper-jax) for JAX implementation
- [FFmpeg](https://ffmpeg.org/) for media processing capabilities