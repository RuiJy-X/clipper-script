"""
clipper.py - FFmpeg video cutting and media management module.

Handles fast stream-copy cutting, precise frame-accurate re-encoding,
filename sanitization, and output directory management.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from gemini_client import ClipCandidate
from transcript_parser import seconds_to_timestamp, timestamp_to_seconds


class FFmpegNotFoundError(Exception):
    """Raised when ffmpeg binary is not found on PATH."""
    pass


class ClipCutError(Exception):
    """Raised when cutting a video clip fails."""
    pass


def check_ffmpeg() -> bool:
    """Check if ffmpeg executable is available in PATH."""
    return shutil.which("ffmpeg") is not None


def ensure_ffmpeg() -> str:
    """Returns ffmpeg path or raises clear error."""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise FFmpegNotFoundError(
            "ffmpeg was not found on your system PATH.\n"
            "Please install ffmpeg to enable video cutting:\n"
            "  - Windows: winget install Gyan.FFmpeg or choco install ffmpeg\n"
            "  - macOS: brew install ffmpeg\n"
            "  - Linux: sudo apt-get install ffmpeg"
        )
    return ffmpeg_path


def sanitize_filename(title: str, index: Optional[int] = None, max_length: int = 60) -> str:
    """
    Sanitizes a string for use as a safe, cross-platform filename.
    Removes illegal characters across Windows, macOS, and Linux: \ / : * ? " < > |
    """
    # Replace spaces and punctuation with underscores or hyphens
    clean = re.sub(r'[\\/*?:"<>|]', "", title)
    clean = re.sub(r"\s+", "_", clean)
    clean = re.sub(r"[^\w\-_.]", "", clean)
    clean = clean.strip("._")

    if not clean:
        clean = "clip"

    # Truncate length
    if len(clean) > max_length:
        clean = clean[:max_length].rstrip("._")

    if index is not None:
        return f"{index:02d}_{clean}.mp4"
    return f"{clean}.mp4"


def build_ffmpeg_cut_command(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str,
    precise: bool = False,
    ffmpeg_bin: str = "ffmpeg",
) -> List[str]:
    """
    Builds the ffmpeg argument list.
    - Fast stream-copy mode: uses stream copy (-c copy)
    - Precise mode: re-encodes video/audio for frame-accurate cuts (-c:v libx264 -c:a aac)
    """
    duration = max(0.1, end_sec - start_sec)
    start_ts = seconds_to_timestamp(start_sec, include_millis=True)
    
    if precise:
        # Re-encode with high quality and fast preset
        return [
            ffmpeg_bin,
            "-y",
            "-ss", start_ts,
            "-i", video_path,
            "-t", f"{duration:.3f}",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "20",
            "-c:a", "aac",
            "-b:a", "192k",
            "-avoid_negative_ts", "make_zero",
            output_path,
        ]
    else:
        # Fast stream-copy mode
        return [
            ffmpeg_bin,
            "-y",
            "-ss", start_ts,
            "-i", video_path,
            "-t", f"{duration:.3f}",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            output_path,
        ]


def cut_single_clip(
    video_path: str,
    start_seconds: float,
    end_seconds: float,
    output_path: str,
    precise: bool = False,
    ffmpeg_bin: Optional[str] = None,
) -> str:
    """
    Cuts a single segment from the video file.
    If fast stream copy fails, automatically retries in precise re-encode mode.
    """
    bin_path = ffmpeg_bin or ensure_ffmpeg()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    cmd = build_ffmpeg_cut_command(
        video_path=video_path,
        start_sec=start_seconds,
        end_sec=end_seconds,
        output_path=output_path,
        precise=precise,
        ffmpeg_bin=bin_path,
    )

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        # If fast stream copy failed and wasn't already precise, retry in precise mode
        if not precise:
            cmd_retry = build_ffmpeg_cut_command(
                video_path=video_path,
                start_sec=start_seconds,
                end_sec=end_seconds,
                output_path=output_path,
                precise=True,
                ffmpeg_bin=bin_path,
            )
            retry_res = subprocess.run(
                cmd_retry,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if retry_res.returncode == 0:
                return output_path
            
            raise ClipCutError(
                f"Failed to cut clip '{output_path}' with ffmpeg: {retry_res.stderr.strip()}"
            )
        else:
            raise ClipCutError(
                f"Failed to cut clip '{output_path}' with ffmpeg: {result.stderr.strip()}"
            )

    return output_path


def cut_all_clips(
    video_path: str,
    clips: List[ClipCandidate],
    output_dir: str,
    precise: bool = False,
    progress_callback: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Cuts all clips from the video and saves them to the output directory.
    Returns metadata list of cut clips.
    """
    ffmpeg_bin = ensure_ffmpeg()
    os.makedirs(output_dir, exist_ok=True)

    cut_results: List[Dict[str, Any]] = []

    for i, clip in enumerate(clips, start=1):
        filename = sanitize_filename(clip.title, index=i)
        clip_path = os.path.join(output_dir, filename)

        if progress_callback:
            progress_callback(i, len(clips), clip.title)

        cut_single_clip(
            video_path=video_path,
            start_seconds=clip.start_seconds,
            end_seconds=clip.end_seconds,
            output_path=clip_path,
            precise=precise,
            ffmpeg_bin=ffmpeg_bin,
        )

        clip_dict = clip.to_dict()
        clip_dict["index"] = i
        clip_dict["file_name"] = filename
        clip_dict["file_path"] = os.path.abspath(clip_path)
        cut_results.append(clip_dict)

    return cut_results
