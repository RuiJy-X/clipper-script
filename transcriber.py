"""
transcriber.py - Fallback Whisper-based local audio transcription.

Transcribes audio/video files locally using faster-whisper or openai-whisper
when a pre-existing transcript file is not found, saving the result alongside the video.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import List, Optional

from transcript_parser import ParsedTranscript, TranscriptSegment, seconds_to_timestamp, save_segments_to_file


class TranscriptionError(Exception):
    """Raised when audio extraction or transcription fails."""
    pass


def extract_audio_from_video(video_path: str, output_wav_path: str) -> None:
    """
    Extracts 16kHz mono audio WAV from video file using ffmpeg for Whisper.
    """
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_wav_path,
    ]
    try:
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if process.returncode != 0:
            raise TranscriptionError(
                f"Failed to extract audio using ffmpeg: {process.stderr.strip()}"
            )
    except FileNotFoundError:
        raise TranscriptionError(
            "ffmpeg is not found on your PATH. Please install ffmpeg to extract audio from video."
        )


def _transcribe_with_faster_whisper(
    audio_path: str,
    model_size: str = "base",
    device: str = "auto",
) -> List[TranscriptSegment]:
    """
    Transcribes audio using faster-whisper (CTranslate2 backend).
    """
    from faster_whisper import WhisperModel

    # Map device
    compute_type = "default"
    if device == "auto":
        import torch
        device_to_use = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device_to_use == "cuda" else "int8"
    else:
        device_to_use = device

    model = WhisperModel(model_size, device=device_to_use, compute_type=compute_type)
    segments_gen, info = model.transcribe(audio_path, beam_size=5, vad_filter=True)

    segments: List[TranscriptSegment] = []
    for seg in segments_gen:
        text = seg.text.strip()
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                start_seconds=seg.start,
                end_seconds=seg.end,
                start_time=seconds_to_timestamp(seg.start),
                end_time=seconds_to_timestamp(seg.end),
                speaker="",
                text=text,
            )
        )
    return segments


def _transcribe_with_openai_whisper(
    audio_path: str,
    model_size: str = "base",
    device: str = "auto",
) -> List[TranscriptSegment]:
    """
    Transcribes audio using openai-whisper.
    """
    import whisper

    if device == "auto":
        import torch
        device_to_use = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_to_use = device

    model = whisper.load_model(model_size, device=device_to_use)
    result = model.transcribe(audio_path, verbose=False)

    segments: List[TranscriptSegment] = []
    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()
        if not text:
            continue
        start_s = float(seg.get("start", 0.0))
        end_s = float(seg.get("end", 0.0))
        segments.append(
            TranscriptSegment(
                start_seconds=start_s,
                end_seconds=end_s,
                start_time=seconds_to_timestamp(start_s),
                end_time=seconds_to_timestamp(end_s),
                speaker="",
                text=text,
            )
        )
    return segments


def transcribe_video(
    video_path: str,
    output_transcript_path: Optional[str] = None,
    model_size: str = "base",
    device: str = "auto",
) -> ParsedTranscript:
    """
    Transcribes a video file and saves a timestamped transcript alongside it.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Determine output transcript path
    if not output_transcript_path:
        base_name, _ = os.path.splitext(video_path)
        output_transcript_path = f"{base_name}.txt"

    # Check for available backend
    has_faster_whisper = False
    has_openai_whisper = False

    try:
        import faster_whisper
        has_faster_whisper = True
    except ImportError:
        pass

    try:
        import whisper
        has_openai_whisper = True
    except ImportError:
        pass

    if not has_faster_whisper and not has_openai_whisper:
        raise TranscriptionError(
            f"No transcript found at '{output_transcript_path}', and no Whisper transcription library is installed.\n"
            "Please either:\n"
            "1. Provide a matching .txt transcript file in the same folder (e.g., episode.txt), or\n"
            "2. Install faster-whisper via: pip install faster-whisper"
        )

    # Extract audio to temp file
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_wav = os.path.join(tmp_dir, "extracted_audio.wav")
        extract_audio_from_video(video_path, temp_wav)

        if has_faster_whisper:
            segments = _transcribe_with_faster_whisper(temp_wav, model_size=model_size, device=device)
        else:
            segments = _transcribe_with_openai_whisper(temp_wav, model_size=model_size, device=device)

    if not segments:
        raise TranscriptionError("Transcription completed but produced no text.")

    # Save to disk
    save_segments_to_file(segments, output_transcript_path)

    return ParsedTranscript(segments=segments, source_file=output_transcript_path)
