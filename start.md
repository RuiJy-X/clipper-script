Agent Prompt: Podcast Clip Finder

Role: You are a Python developer building a CLI tool that extracts short, interesting clips from long-form podcast videos (~1 hour) using AI-driven transcript analysis.

Goal

Build a Python script clip_finder.py that:

Takes a path to a podcast video file as input.
Looks for a matching transcript .txt file in the same folder (same base filename, e.g. episode1.mp4 + episode1.txt).
If a transcript file exists, parses it (it will contain timestamps) instead of running any transcription model.
If no transcript exists, transcribes the audio locally (use faster-whisper or openai-whisper) and generates a timestamped transcript, saving it alongside the video for reuse next time.
Sends the timestamped transcript to the Gemini API and asks it to identify the most interesting, quotable, or self-contained clip-worthy segments (hooks, strong opinions, funny moments, insight-dense answers, emotional beats, complete stories/anecdotes).
Gemini should return structured JSON: a list of clips, each with start_time, end_time, title, reason (why it's a good clip), and a score (1–10).
Uses ffmpeg (via subprocess or ffmpeg-python) to cut each selected clip from the source video and save it into an output/ folder next to the source, named using the clip title/index.
Logs a summary (clip count, titles, scores, timestamps) to console and to a clips_summary.json in the output folder.
Technical requirements

CLI interface (using argparse):

python clip_finder.py <video_path> [--min-clip-len 20] [--max-clip-len 90] [--num-clips 8] [--output-dir ./output]

Transcript format assumption — support a flexible parser for common timestamp formats, e.g.:

[00:01:23] Speaker: text...
00:01:23 --> 00:01:40
text...

Normalize into a list of {start, end, text} segments internally.

Gemini integration:

Read the API key from an environment variable GEMINI_API_KEY (never hardcode it; support a .env file via python-dotenv).
Use the google-genai (or google-generativeai) SDK.
Since a 1-hour transcript may be long, chunk it if needed to fit context, but prefer sending the full transcript in one call to preserve cross-segment context for good clip selection (Gemini 1.5/2.x models have large context windows).
Prompt Gemini with clear instructions: identify N clips of roughly X–Y seconds each, self-contained (make sense without extra context), varied in topic, ranked by "shareability"/interest. Require the response as strict JSON matching a schema you define, and validate/parse it defensively (retry once if malformed).

Clip cutting:

Use ffmpeg -ss <start> -to <end> -i <video> -c copy <output> for speed (fallback to re-encode if stream copy produces bad cuts at non-keyframes — add a --precise flag that re-encodes with -c:v libx264 -c:a aac when exact cuts matter).
Sanitize clip titles into safe filenames.

Project structure:

clip_finder.py # CLI entry + orchestration
transcript_parser.py # parse existing .txt transcript
transcriber.py # fallback whisper-based transcription
gemini_client.py # builds prompt, calls Gemini, parses/validates JSON
clipper.py # ffmpeg cutting logic
requirements.txt
.env.example

Error handling: missing video file, missing ffmpeg binary, missing/invalid API key, malformed Gemini response, empty transcript — all should produce clear, actionable error messages, not stack traces.

Dependencies: google-genai, python-dotenv, ffmpeg-python (optional), faster-whisper (optional fallback), standard library argparse, json, re, subprocess.

Deliverables
Working script(s) as listed above.
requirements.txt.
A short README.md explaining setup (installing ffmpeg, setting GEMINI_API_KEY, transcript format expected) and usage examples.
Basic test with a short sample transcript (no video needed) verifying the Gemini prompt/parsing logic in isolation.
