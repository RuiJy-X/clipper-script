#!/usr/bin/env python3
"""
clip_finder.py - CLI tool that extracts short, interesting clips from long-form
podcast videos using AI-driven transcript analysis.

Usage:
    python clip_finder.py <video_path> [options]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from clipper import ClipCutError, FFmpegNotFoundError, check_ffmpeg, cut_all_clips, ensure_ffmpeg
from gemini_client import (
    ClipCandidate,
    GeminiAPIError,
    GeminiClipFinderClient,
    GeminiResponseParseError,
)
from transcriber import TranscriptionError, transcribe_video
from transcript_parser import (
    ParsedTranscript,
    parse_transcript_file,
    parse_transcript_text,
    seconds_to_timestamp,
)


def find_matching_transcript(video_path: str) -> Optional[str]:
    """
    Looks for a matching transcript file with the same base name in the same directory.
    Checks .txt, .srt, .vtt.
    """
    base_stem, _ = os.path.splitext(video_path)
    for ext in [".txt", ".srt", ".vtt"]:
        candidate = f"{base_stem}{ext}"
        if os.path.isfile(candidate):
            return candidate
    return None


def print_banner() -> None:
    banner = """
============================================================
              PODCAST CLIP FINDER (AI Powered)
============================================================
"""
    print(banner)


def format_summary_table(clips: List[ClipCandidate], output_paths: Optional[List[str]] = None) -> str:
    """
    Creates a clean, human-readable summary table for console output.
    """
    lines = []
    lines.append("-" * 75)
    lines.append(f"{'#':<3} | {'SCORE':<5} | {'TIMESTAMPS':<21} | {'DURATION':<8} | {'TITLE'}")
    lines.append("-" * 75)

    for i, clip in enumerate(clips, start=1):
        ts_range = f"{clip.start_time} -> {clip.end_time}"
        dur_str = f"{clip.duration:.1f}s"
        score_str = f"{clip.score}/10"
        title_truncated = clip.title[:32] if len(clip.title) > 32 else clip.title
        lines.append(f"{i:<3} | {score_str:<5} | {ts_range:<21} | {dur_str:<8} | {title_truncated}")
        if clip.reason:
            lines.append(f"    Reason: {clip.reason}")
        if clip.key_quote:
            lines.append(f"    Quote:  \"{clip.key_quote}\"")
        if output_paths and i - 1 < len(output_paths):
            lines.append(f"    File:   {os.path.basename(output_paths[i-1])}")
        lines.append("-" * 75)

    return "\n".join(lines)


def run_clip_finder(
    video_path: str,
    transcript_path: Optional[str] = None,
    min_clip_len: int = 20,
    max_clip_len: int = 90,
    num_clips: int = 8,
    output_dir: Optional[str] = None,
    gemini_model: str = "gemini-3.5-flash",
    api_key: Optional[str] = None,
    precise: bool = False,
    whisper_model: str = "base",
    whisper_device: str = "auto",
    dry_run: bool = False,
    verbose: bool = False,
) -> int:
    """
    Main orchestration logic.
    """
    video_path = os.path.abspath(video_path)

    if not os.path.isfile(video_path):
        print(f"Error: Video file not found at '{video_path}'", file=sys.stderr)
        return 1

    # Determine output directory
    if not output_dir:
        output_dir = "./output"
    output_dir = os.path.abspath(output_dir)

    print(f"[*] Target Video:      {video_path}")
    print(f"[*] Output Directory:  {output_dir}")
    print(f"[*] Target Clip Count: {num_clips} (length: {min_clip_len}s - {max_clip_len}s)")
    print(f"[*] AI Model:          {gemini_model}")
    if precise:
        print("[*] Cutting Mode:      Precise re-encoding (-c:v libx264 -c:a aac)")
    else:
        print("[*] Cutting Mode:      Fast stream copy (-c copy)")

    # Step 1: Locate or generate transcript
    print("\n[Step 1/4] Checking for transcript...")
    selected_transcript_file = transcript_path or find_matching_transcript(video_path)

    parsed_transcript: Optional[ParsedTranscript] = None

    if selected_transcript_file and os.path.isfile(selected_transcript_file):
        print(f"  -> Found existing transcript: {selected_transcript_file}")
        try:
            parsed_transcript = parse_transcript_file(selected_transcript_file)
            print(f"  -> Parsed {len(parsed_transcript.segments)} segments "
                  f"({parsed_transcript.word_count} words, {seconds_to_timestamp(parsed_transcript.total_duration)} duration).")
        except Exception as e:
            print(f"Error parsing transcript file '{selected_transcript_file}': {e}", file=sys.stderr)
            if verbose:
                raise
            return 1
    else:
        print("  -> No existing transcript found.")
        print(f"  -> Transcribing audio locally using Whisper ('{whisper_model}' model)...")
        try:
            parsed_transcript = transcribe_video(
                video_path=video_path,
                output_transcript_path=None,
                model_size=whisper_model,
                device=whisper_device,
            )
            print(f"  -> Transcription complete! Saved to {parsed_transcript.source_file}")
            print(f"  -> Generated {len(parsed_transcript.segments)} segments.")
        except TranscriptionError as e:
            print(f"\nTranscription Error:\n{e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Unexpected error during transcription: {e}", file=sys.stderr)
            if verbose:
                raise
            return 1

    if not parsed_transcript or parsed_transcript.is_empty():
        print("Error: Transcript contains no speech or segments.", file=sys.stderr)
        return 1

    # Step 2: AI Clip Selection with Gemini
    print("\n[Step 2/4] Analyzing transcript with Gemini API to find best clips...")
    try:
        gemini_client = GeminiClipFinderClient(api_key=api_key, model_name=gemini_model)
        clips = gemini_client.find_clips(
            transcript=parsed_transcript,
            min_clip_len=min_clip_len,
            max_clip_len=max_clip_len,
            num_clips=num_clips,
        )
    except GeminiAPIError as e:
        print(f"\nGemini API Error: {e}", file=sys.stderr)
        return 1
    except GeminiResponseParseError as e:
        print(f"\nFailed to parse Gemini response: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error querying Gemini API: {e}", file=sys.stderr)
        if verbose:
            raise
        return 1

    print(f"  -> Successfully identified {len(clips)} candidate clips!")

    # Step 3: Cut clips with ffmpeg (unless dry-run)
    os.makedirs(output_dir, exist_ok=True)
    cut_results = []
    output_filepaths = []

    if dry_run:
        print("\n[Step 3/4] [DRY RUN] Skipping video cutting as requested.")
        for i, clip in enumerate(clips, start=1):
            clip_dict = clip.to_dict()
            clip_dict["index"] = i
            clip_dict["file_name"] = f"{i:02d}_(dry_run).mp4"
            clip_dict["file_path"] = os.path.join(output_dir, clip_dict["file_name"])
            cut_results.append(clip_dict)
            output_filepaths.append(clip_dict["file_path"])
    else:
        print(f"\n[Step 3/4] Cutting {len(clips)} clips with FFmpeg into '{output_dir}'...")
        if not check_ffmpeg():
            print(
                "Error: ffmpeg is required to cut clips but was not found on your system PATH.\n"
                "Install ffmpeg or run with --dry-run to only get the summary.",
                file=sys.stderr,
            )
            return 1

        def progress(idx: int, total: int, title: str):
            print(f"  [{idx}/{total}] Cutting: \"{title}\"...")

        try:
            cut_results = cut_all_clips(
                video_path=video_path,
                clips=clips,
                output_dir=output_dir,
                precise=precise,
                progress_callback=progress,
            )
            output_filepaths = [r["file_path"] for r in cut_results]
        except (ClipCutError, FFmpegNotFoundError) as e:
            print(f"\nFFmpeg Error: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Unexpected error cutting clips: {e}", file=sys.stderr)
            if verbose:
                raise
            return 1

    # Step 4: Write summary JSON and display results
    print("\n[Step 4/4] Writing summary...")
    summary_path = os.path.join(output_dir, "clips_summary.json")
    summary_data = {
        "source_video": video_path,
        "transcript_file": parsed_transcript.source_file,
        "total_clips": len(cut_results),
        "parameters": {
            "min_clip_len": min_clip_len,
            "max_clip_len": max_clip_len,
            "num_clips": num_clips,
            "gemini_model": gemini_model,
            "precise": precise,
            "dry_run": dry_run,
        },
        "clips": cut_results,
    }

    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"  -> Saved summary to: {summary_path}")
    except Exception as e:
        print(f"Warning: Could not save summary JSON: {e}", file=sys.stderr)

    print("\n" + format_summary_table(clips, output_filepaths))
    print(f"\n[SUCCESS] Completed! All {len(clips)} clips processed and saved in:\n  {output_dir}\n")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Podcast Clip Finder - AI-driven highlight clip extractor for podcasts & long-form video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "video_path",
        help="Path to the podcast video file (e.g., episode1.mp4, podcast.mkv)",
    )
    parser.add_argument(
        "--transcript",
        "-t",
        default=None,
        help="Path to existing transcript file (.txt, .srt, .vtt). Default: auto-detect alongside video.",
    )
    parser.add_argument(
        "--min-clip-len",
        type=int,
        default=20,
        help="Minimum clip duration in seconds.",
    )
    parser.add_argument(
        "--max-clip-len",
        type=int,
        default=90,
        help="Maximum clip duration in seconds.",
    )
    parser.add_argument(
        "--num-clips",
        "-n",
        type=int,
        default=8,
        help="Target number of clips to find.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./output",
        help="Directory to save clips and clips_summary.json. Default: ./output",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.0-flash",
        help="Gemini model ID to use (e.g. gemini-2.0-flash, gemini-1.5-flash).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Gemini API Key (default: loaded from GEMINI_API_KEY env or .env file).",
    )
    parser.add_argument(
        "--precise",
        action="store_true",
        help="Re-encode clips using libx264/aac for frame-accurate cuts (slower, exact cuts).",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        help="Whisper model size if local transcription is needed (e.g., tiny, base, small, medium, large-v3).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run Whisper transcription on.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Identify clips and write summary JSON without executing ffmpeg video cuts.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output and error stack traces.",
    )

    return parser


def main() -> None:
    print_banner()
    parser = build_arg_parser()
    args = parser.parse_args()

    exit_code = run_clip_finder(
        video_path=args.video_path,
        transcript_path=args.transcript,
        min_clip_len=args.min_clip_len,
        max_clip_len=args.max_clip_len,
        num_clips=args.num_clips,
        output_dir=args.output_dir,
        gemini_model=args.model,
        api_key=args.api_key,
        precise=args.precise,
        whisper_model=args.whisper_model,
        whisper_device=args.device,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
