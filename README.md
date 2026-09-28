# Podcast Clip Finder 🎙️🎬

An AI-driven CLI tool that analyzes long-form podcast videos and transcripts using **Google Gemini** to automatically discover, score, and extract high-retention, viral clip segments.

---

## Key Features

- 🧠 **AI-Powered Selection**: Uses Gemini (e.g. `gemini-2.0-flash`) to detect hooks, punchlines, controversial opinions, and self-contained stories.
- ⏱️ **Flexible Transcript Parsing**: Automatically detects and parses common timestamped transcript formats:
  - `[00:01:23] Speaker: text`
  - `[00:01:23 - 00:01:45] Speaker: text`
  - `00:01:23,123 --> 00:01:40,456` (SRT / SubRip)
  - `00:01:23.123 --> 00:01:40.456` (WebVTT)
  - YouTube transcript copy-pastes (`0:05 \n text`)
  - Whisper output (`[00:00.000 -> 00:05.000] text`)
- 🎙️ **Automatic Local Whisper Fallback**: If no transcript file is found, extracts audio and transcribes locally using `faster-whisper` or `openai-whisper`, saving the timestamped transcript alongside the video for instant reuse.
- ⚡ **Fast & Lossless Stream Copying**: Cuts video clips instantaneously without quality loss using FFmpeg stream copy (`-c copy`).
- 🎯 **Frame-Accurate Cutting Mode**: Optional `--precise` flag re-encodes with `libx264`/`aac` for exact cuts on non-keyframes.
- 📊 **Structured Output & Summaries**: Outputs sanitized clip video files and a complete `clips_summary.json` containing metadata, scores (1-10), reasons, and quotes.

---

## Project Structure

```
clipper/
├── videos/                # Put input videos (.mp4, .mkv) and transcripts (.txt, .srt) here
├── output/                # Destination directory for generated clips & clips_summary.json
├── clip_finder.py         # Main CLI orchestration script
├── transcript_parser.py   # Flexible transcript parser & normalizer
├── gemini_client.py       # Gemini API client with prompt & JSON validation
├── clipper.py             # FFmpeg video cutting logic & filename sanitizer
├── transcriber.py         # Fallback local Whisper transcription module
├── requirements.txt       # Python dependencies
├── .env.example           # Template for API keys
├── README.md              # Documentation & usage guide
├── run_tests.py           # Unit and integration test runner
└── tests/
    ├── sample_transcript.txt
    ├── test_transcript_parser.py
    ├── test_gemini_client.py
    ├── test_clipper.py
    └── test_clip_finder.py
```

---

## Prerequisites

### 1. Install FFmpeg
FFmpeg is required to extract audio and cut video clips:
- **Windows**:
  ```powershell
  winget install Gyan.FFmpeg
  # or
  choco install ffmpeg
  ```
- **macOS**:
  ```bash
  brew install ffmpeg
  ```
- **Linux (Ubuntu/Debian)**:
  ```bash
  sudo apt-get update && sudo apt-get install -y ffmpeg
  ```

### 2. Gemini API Key
Get an API key from [Google AI Studio](https://aistudio.google.com/).

---

## Installation & Setup

1. **Clone repository & enter directory**:
   ```bash
   cd clipper
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   *(Optional: If you need local transcription when no transcript `.txt` is provided, install `faster-whisper`)*:
   ```bash
   pip install faster-whisper
   ```

3. **Configure API Key**:
   Create a `.env` file from `.env.example`:
   ```bash
   cp .env.example .env
   ```
   Add your Gemini API key:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=gemini-2.0-flash
   ```

---

## Usage Guide

### Basic Usage
Run with a video file located in `videos/` (auto-detects a matching `videos/episode1.txt`, `.srt`, or `.vtt` file with the same base name):
```bash
python clip_finder.py videos/episode1.mp4
```

### Specify a Custom Transcript
```bash
python clip_finder.py videos/episode1.mp4 --transcript videos/custom_transcript.srt
```

### Custom Clip Length and Count
Extract 5 clips between 30 and 60 seconds each:
```bash
python clip_finder.py videos/episode1.mp4 --min-clip-len 30 --max-clip-len 60 --num-clips 5
```

### Frame-Accurate Precise Re-encoding
For exact cuts at non-keyframe boundaries:
```bash
python clip_finder.py videos/episode1.mp4 --precise
```

### Dry Run (Preview Mode)
Generate AI clip selections and `output/clips_summary.json` without running FFmpeg cuts:
```bash
python clip_finder.py videos/episode1.mp4 --dry-run
```

### Custom Output Directory & Model
```bash
python clip_finder.py videos/podcast.mp4 --output-dir ./my_clips --model gemini-2.0-flash
```

---

## CLI Options Reference

| Argument | Short | Default | Description |
|---|---|---|---|
| `video_path` | | *Required* | Path to input video file |
| `--transcript` | `-t` | Auto | Path to `.txt`, `.srt`, `.vtt` transcript |
| `--min-clip-len` | | `20` | Minimum clip duration in seconds |
| `--max-clip-len` | | `90` | Maximum clip duration in seconds |
| `--num-clips` | `-n` | `8` | Target number of clips |
| `--output-dir` | `-o` | `./output` | Destination folder for clips and summary |
| `--model` | | `gemini-2.0-flash` | Gemini model ID |
| `--api-key` | | Env | Gemini API key override |
| `--precise` | | `False` | Re-encode with libx264/aac for exact cuts |
| `--whisper-model`| | `base` | Whisper model (`tiny`, `base`, `small`, etc.) |
| `--device` | | `auto` | Whisper device (`auto`, `cpu`, `cuda`) |
| `--dry-run` | | `False` | Run analysis without cutting video files |
| `--verbose` | `-v` | `False` | Print full debug stack traces |

---

## Output Structure

Clips and summaries are written to the output directory:
```
output/
├── 01_Catastrophic_Malfunction_Demo.mp4
├── 02_Sim_To_Real_Gap.mp4
├── 03_Buy_A_Soldering_Iron.mp4
└── clips_summary.json
```

Sample `clips_summary.json`:
```json
{
  "source_video": "C:/projects/clipper/episode1.mp4",
  "transcript_file": "C:/projects/clipper/episode1.txt",
  "total_clips": 3,
  "parameters": {
    "min_clip_len": 20,
    "max_clip_len": 90,
    "num_clips": 3,
    "gemini_model": "gemini-2.5-flash",
    "precise": false,
    "dry_run": false
  },
  "clips": [
    {
      "index": 1,
      "title": "Catastrophic Malfunction Demo",
      "file_name": "01_Catastrophic_Malfunction_Demo.mp4",
      "file_path": "C:/projects/clipper/output/01_Catastrophic_Malfunction_Demo.mp4",
      "start_time": "00:00:38",
      "end_time": "00:01:15",
      "start_seconds": 38.0,
      "end_seconds": 75.0,
      "duration_seconds": 37.0,
      "score": 9,
      "reason": "Compelling turning point story with high emotional hook",
      "key_quote": "Every major breakthrough starts as a catastrophic malfunction",
      "summary": "Sarah recounts how an investor backed her after a live robot fire"
    }
  ]
}
```

---

## Running Tests

Execute the comprehensive test suite:
```bash
python run_tests.py
# or using pytest
pytest
```
# clipper-script
