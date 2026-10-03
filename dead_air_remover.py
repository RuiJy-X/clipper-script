#!/usr/bin/env python3
"""
dead_air_remover.py
Cuts silent segments out of MP4/video files (video + audio stay in sync).

Supports both batch folder processing (default) and single video processing.

Usage:
    # Batch mode: processes all videos in 'saved/' and outputs to 'cleaned/'
    python dead_air_remover.py
    python dead_air_remover.py --input-dir saved --output-dir cleaned --suffix _cleaned

    # Single video mode:
    python dead_air_remover.py input.mp4 output.mp4
    python dead_air_remover.py input.mp4 output.mp4 --threshold -35 --min-silence 0.5 --padding 0.1
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path


SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".flv",
    ".wmv",
    ".m4v",
    ".ts",
}


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def get_duration(path):
    r = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ])
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed to get duration: {r.stderr.strip()}")
    output = r.stdout.strip()
    if not output:
        raise RuntimeError(f"ffprobe returned empty duration for {path}")
    return float(output)


def detect_silence(path, threshold_db, min_silence):
    """Return list of (silence_start, silence_end) tuples."""
    r = run([
        "ffmpeg", "-i", str(path), "-af",
        f"silencedetect=noise={threshold_db}dB:d={min_silence}",
        "-f", "null", "-",
    ])
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg silencedetect failed: {r.stderr.strip()}")

    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr)]

    # If the file ends in silence, ffmpeg reports a start with no end
    if len(starts) > len(ends):
        ends.append(get_duration(path))
    return list(zip(starts, ends))


def build_keep_segments(silences, duration, padding):
    """Invert silence ranges into speech ranges, keeping a little padding."""
    if duration <= 0:
        return []

    keep = []
    cursor = 0.0
    for s_start, s_end in silences:
        seg_start, seg_end = cursor, min(s_start + padding, duration)
        if seg_end - seg_start > 0.05:
            keep.append((seg_start, seg_end))
        cursor = max(s_end - padding, 0)
    if duration - cursor > 0.05:
        keep.append((cursor, duration))

    # Merge segments that overlap because of padding
    merged = []
    for s, e in keep:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def cut_and_join(path, segments, output):
    """Use a filter_complex trim/concat so video and audio stay in sync."""
    if not segments:
        raise ValueError("Cannot cut video: no valid segments provided.")

    v_parts, a_parts, labels = [], [], []
    for i, (s, e) in enumerate(segments):
        v_parts.append(f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[v{i}]")
        a_parts.append(f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{i}]")
        labels.append(f"[v{i}][a{i}]")

    filter_complex = ";".join(v_parts + a_parts) + \
        ";" + "".join(labels) + f"concat=n={len(segments)}:v=1:a=1[outv][outa]"

    # Write the filter to a file: long videos can exceed command-line length limits
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(filter_complex)
        script_path = f.name

    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(path),
            "-filter_complex_script", script_path,
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            str(output),
        ]
        r = subprocess.run(cmd)
        if r.returncode != 0:
            raise RuntimeError(f"FFmpeg failed while cutting the video (code {r.returncode}).")
    finally:
        Path(script_path).unlink(missing_ok=True)


def find_video_files(input_dir):
    """Return a sorted list of video Path objects found in input_dir."""
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        return []

    videos = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    ]
    return sorted(videos, key=lambda p: p.name.lower())


def process_video(input_path, output_path, threshold_db=-35.0, min_silence=0.5, padding=0.1):
    """
    Process a single video file, removing dead air.
    Returns a result dict containing processing statistics.
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    duration = get_duration(input_file)
    silences = detect_silence(input_file, threshold_db, min_silence)
    segments = build_keep_segments(silences, duration, padding)

    if not segments:
        return {
            "success": False,
            "input": input_file,
            "output": output_file,
            "duration": duration,
            "kept": 0.0,
            "cut": duration,
            "segments": 0,
            "silences": len(silences),
            "error": "No non-silent audio found. Try a lower --threshold (e.g. -45).",
        }

    kept = sum(e - s for s, e in segments)
    cut_and_join(input_file, segments, output_file)

    return {
        "success": True,
        "input": input_file,
        "output": output_file,
        "duration": duration,
        "kept": kept,
        "cut": max(0.0, duration - kept),
        "segments": len(segments),
        "silences": len(silences),
        "error": None,
    }


def process_batch(input_dir, output_dir, threshold_db=-35.0, min_silence=0.5, padding=0.1, suffix="_cleaned"):
    """
    Finds all videos in input_dir, processes each, and outputs to output_dir with suffix.
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    videos = find_video_files(in_dir)

    print("=" * 60)
    print(f" Dead Air Remover - Batch Mode")
    print(f" [*] Input folder:  {in_dir.resolve()}")
    print(f" [*] Output folder: {out_dir.resolve()}")
    print(f" [*] Found videos:  {len(videos)}")
    print("=" * 60)

    if not videos:
        print(f"\n[!] No supported video files found in '{in_dir}'.")
        print(f"    Supported formats: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}")
        return {"total": 0, "succeeded": 0, "failed": 0, "results": []}

    results = []
    for idx, video in enumerate(videos, start=1):
        out_name = f"{video.stem}{suffix}{video.suffix}"
        out_file = out_dir / out_name
        print(f"\n[{idx}/{len(videos)}] Processing: {video.name} -> {out_name}")

        try:
            res = process_video(
                input_path=video,
                output_path=out_file,
                threshold_db=threshold_db,
                min_silence=min_silence,
                padding=padding,
            )
            results.append(res)

            if res["success"]:
                print(
                    f"  [+] Done! Kept {res['kept']:.1f}s of {res['duration']:.1f}s "
                    f"({res['cut']:.1f}s cut across {res['segments']} segments)."
                )
                print(f"  [+] Saved to: {out_file}")
            else:
                print(f"  [-] Skipped: {res['error']}")
        except Exception as e:
            print(f"  [-] Failed: {e}")
            results.append({
                "success": False,
                "input": video,
                "output": out_file,
                "duration": 0.0,
                "kept": 0.0,
                "cut": 0.0,
                "segments": 0,
                "silences": 0,
                "error": str(e),
            })

    succeeded = sum(1 for r in results if r["success"])
    failed = len(results) - succeeded
    total_saved = sum(r["cut"] for r in results if r["success"])

    print("\n" + "=" * 60)
    print(" Batch Processing Complete")
    print(f" Total Videos: {len(videos)}")
    print(f" Succeeded:    {succeeded}")
    print(f" Failed:       {failed}")
    print(f" Total Trimmed: {total_saved:.1f}s")
    print("=" * 60)

    return {
        "total": len(videos),
        "succeeded": succeeded,
        "failed": failed,
        "total_trimmed": total_saved,
        "results": results,
    }


def main():
    p = argparse.ArgumentParser(description="Remove dead air from MP4 / video files.")
    p.add_argument("input", nargs="?", default=None,
                   help="Input video file or input directory (default: 'saved')")
    p.add_argument("output", nargs="?", default=None,
                   help="Output video file or output directory (default: 'cleaned')")
    p.add_argument("--input-dir", "-i", default=None,
                   help="Directory containing input videos (default: 'saved')")
    p.add_argument("--output-dir", "-o", default=None,
                   help="Directory to save processed videos (default: 'cleaned')")
    p.add_argument("--suffix", default="_cleaned",
                   help="Suffix to append to output video filenames in batch mode (default: '_cleaned')")
    p.add_argument("--threshold", type=float, default=-35,
                   help="Silence threshold in dB (default -35; lower = stricter)")
    p.add_argument("--min-silence", type=float, default=0.5,
                   help="Minimum silence length in seconds to cut (default 0.5)")
    p.add_argument("--padding", type=float, default=0.1,
                   help="Seconds of silence to keep around speech (default 0.1)")
    args = p.parse_args()

    # Determine execution mode:
    # 1. If a positional input is explicitly provided and points to a file: single file mode
    # 2. Otherwise: batch directory mode
    if args.input and Path(args.input).is_file():
        input_path = Path(args.input)
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f"{input_path.stem}{args.suffix}{input_path.suffix}"

        print(f"[*] Processing single file: {input_path} -> {output_path}")
        try:
            res = process_video(
                input_path,
                output_path,
                threshold_db=args.threshold,
                min_silence=args.min_silence,
                padding=args.padding,
            )
            if not res["success"]:
                sys.exit(f"[!] Error: {res['error']}")
            print(
                f"[+] Done! Kept {res['kept']:.1f}s of {res['duration']:.1f}s "
                f"({res['segments']} segments)."
            )
            print(f"[+] Output: {res['output']}")
        except Exception as e:
            sys.exit(f"[!] Execution failed: {e}")
        return

    # Batch mode resolution
    # Positional args take precedence if provided, otherwise --input-dir / --output-dir, defaulting to saved/cleaned
    in_dir = Path(args.input if args.input else (args.input_dir or "saved"))
    out_dir = Path(args.output if args.output else (args.output_dir or "cleaned"))

    if not in_dir.exists():
        in_dir.mkdir(parents=True, exist_ok=True)
        print(f"[*] Created input directory: '{in_dir.resolve()}'.")
        print(f"[*] Please place your video files into '{in_dir}' and run again.")
        return

    summary = process_batch(
        input_dir=in_dir,
        output_dir=out_dir,
        threshold_db=args.threshold,
        min_silence=args.min_silence,
        padding=args.padding,
        suffix=args.suffix,
    )
    if summary["total"] > 0 and summary["succeeded"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()