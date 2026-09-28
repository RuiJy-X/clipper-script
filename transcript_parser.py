"""
transcript_parser.py - Flexible transcript parser for podcast clip finder.

Supports multiple common timestamp and transcript formats:
1. Bracketed timestamps: [00:01:23] Speaker: text or (00:01:23) Speaker: text
2. Range brackets: [00:01:23 - 00:01:45] Speaker: text
3. SRT/VTT arrow ranges: 00:01:23,123 --> 00:01:40,456 or 00:01:23.123 --> 00:01:40.456
4. Whisper arrow format: [00:01.000 -> 00:05.000] text
5. YouTube multi-line format: 0:05 \\n text
6. Prefix timestamps: 01:23:45 Host: text
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class TranscriptSegment:
    start_seconds: float
    end_seconds: float
    start_time: str
    end_time: str
    text: str
    speaker: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)

    def to_formatted_line(self) -> str:
        speaker_prefix = f"{self.speaker}: " if self.speaker else ""
        return f"[{self.start_time} - {self.end_time}] {speaker_prefix}{self.text}"


@dataclass
class ParsedTranscript:
    segments: List[TranscriptSegment] = field(default_factory=list)
    source_file: Optional[str] = None

    @property
    def total_duration(self) -> float:
        if not self.segments:
            return 0.0
        return self.segments[-1].end_seconds

    @property
    def full_text(self) -> str:
        return "\n".join(seg.to_formatted_line() for seg in self.segments)

    @property
    def word_count(self) -> int:
        return sum(len(seg.text.split()) for seg in self.segments)

    def is_empty(self) -> bool:
        return len(self.segments) == 0 or all(not s.text.strip() for s in self.segments)


def timestamp_to_seconds(ts_str: str) -> float:
    """
    Converts a timestamp string to seconds float.
    Handles:
      - HH:MM:SS,mmm or HH:MM:SS.mmm
      - MM:SS,mmm or MM:SS.mmm
      - HH:MM:SS or MM:SS or SS
    """
    ts_str = ts_str.strip().replace(",", ".")
    # Match standard HH:MM:SS.mmm or MM:SS.mmm or HH:MM:SS or MM:SS
    parts = ts_str.split(":")
    if len(parts) == 3:
        try:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        except ValueError:
            pass
    elif len(parts) == 2:
        try:
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        except ValueError:
            pass
    elif len(parts) == 1:
        try:
            return float(parts[0])
        except ValueError:
            pass

    # Regex fallback for embedded numbers
    match = re.search(r"(?:(?:(\d{1,2}):)?(\d{1,2}):)?(\d{1,2}(?:\.\d+)?)", ts_str)
    if match:
        h, m, s = match.groups()
        hours = float(h) if h else 0.0
        minutes = float(m) if m else 0.0
        seconds = float(s) if s else 0.0
        return hours * 3600 + minutes * 60 + seconds

    raise ValueError(f"Unable to parse timestamp: '{ts_str}'")


def seconds_to_timestamp(seconds: float, include_millis: bool = False) -> str:
    """
    Converts seconds float to HH:MM:SS formatted string.
    """
    if seconds < 0:
        seconds = 0.0
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))

    if include_millis:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}.{millis:03d}"
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def _clean_text(text: str) -> str:
    """Clean up whitespace and HTML tags (e.g. from VTT/SRT)."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_srt_or_vtt(content: str) -> List[TranscriptSegment]:
    """
    Parses SubRip (.srt) or WebVTT (.vtt) format.
    """
    # Regex for 00:01:23,123 --> 00:01:40,456 or 00:01:23.123 --> 00:01:40.456
    pattern = re.compile(
        r"(?:(\d+)\s*\n)?"
        r"(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?)\s*-->\s*(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?)"
        r"(?:[^\n]*\n)([\s\S]*?)(?=(?:\n\s*\n\d+\s*\n|\n\s*\n\d{1,2}:\d{2}|$))",
        re.MULTILINE,
    )

    segments: List[TranscriptSegment] = []
    # Split content by double newline blocks first as fallback or use regex finditer
    blocks = re.split(r"\n\s*\n", content.strip())

    for block in blocks:
        block = block.strip()
        if not block or block == "WEBVTT":
            continue

        arrow_match = re.search(
            r"(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?)\s*-->\s*(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?)",
            block,
        )
        if not arrow_match:
            continue

        start_str, end_str = arrow_match.group(1), arrow_match.group(2)
        try:
            start_sec = timestamp_to_seconds(start_str)
            end_sec = timestamp_to_seconds(end_str)
        except ValueError:
            continue

        # Text is whatever comes after the arrow line
        lines = block.splitlines()
        text_lines = []
        found_arrow = False
        for line in lines:
            line_str = line.strip()
            if "-->" in line_str:
                found_arrow = True
                continue
            if found_arrow:
                text_lines.append(line_str)

        raw_text = " ".join(text_lines)
        raw_text = _clean_text(raw_text)
        if not raw_text:
            continue

        # Extract speaker if format is "Speaker: text"
        speaker = ""
        speaker_match = re.match(r"^([A-Za-z0-9_\s]{2,30}?):\s*(.*)$", raw_text)
        if speaker_match and not raw_text.startswith("http"):
            speaker = speaker_match.group(1).strip()
            raw_text = speaker_match.group(2).strip()

        segments.append(
            TranscriptSegment(
                start_seconds=start_sec,
                end_seconds=end_sec,
                start_time=seconds_to_timestamp(start_sec),
                end_time=seconds_to_timestamp(end_sec),
                speaker=speaker,
                text=raw_text,
            )
        )

    return segments


def parse_bracket_or_arrow_lines(lines: List[str]) -> List[TranscriptSegment]:
    """
    Parses line-by-line transcripts with bracketed timestamps:
    - [00:01:23] Speaker: text
    - [00:01:23 - 00:01:45] Speaker: text
    - (00:01:23) text
    - [00:00.000 -> 00:05.000] text (faster-whisper style)
    - 00:01:23 Speaker: text
    """
    # Regex patterns for line starts
    # 1. Range: [00:01:23 - 00:01:45] or [00:01:23 -> 00:01:45] or (00:01:23 - 00:01:45)
    range_pat = re.compile(
        r"^[\[\(](\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?)\s*(?:-|–|—|->|-->)\s*(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?)[\]\)]\s*(.*)$"
    )

    # 2. Single timestamp in brackets: [00:01:23] or (00:01:23)
    single_bracket_pat = re.compile(
        r"^[\[\(](\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?)[\]\)]\s*(.*)$"
    )

    # 3. Unbracketed timestamp at line start: 00:01:23 Speaker: text or 01:23 text
    unbracketed_pat = re.compile(
        r"^(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?)\s+(.*)$"
    )

    raw_items: List[Tuple[float, Optional[float], str, str]] = []  # (start_sec, end_sec, speaker, text)

    current_start: Optional[float] = None
    current_end: Optional[float] = None
    current_speaker = ""
    current_text_parts: List[str] = []

    def flush_current():
        nonlocal current_start, current_end, current_speaker, current_text_parts
        if current_start is not None and current_text_parts:
            combined_text = _clean_text(" ".join(current_text_parts))
            if combined_text:
                raw_items.append((current_start, current_end, current_speaker, combined_text))
        current_start = None
        current_end = None
        current_speaker = ""
        current_text_parts = []

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue

        # Skip SRT counter index numbers if any
        if re.match(r"^\d+$", line_str):
            continue

        # Try Range pattern
        m_range = range_pat.match(line_str)
        if m_range:
            flush_current()
            s_str, e_str, rest = m_range.group(1), m_range.group(2), m_range.group(3)
            try:
                s_sec = timestamp_to_seconds(s_str)
                e_sec = timestamp_to_seconds(e_str)
                current_start = s_sec
                current_end = e_sec
                
                # Check for speaker
                spk_match = re.match(r"^([A-Za-z0-9_\s]{2,30}?):\s*(.*)$", rest)
                if spk_match:
                    current_speaker = spk_match.group(1).strip()
                    current_text_parts.append(spk_match.group(2).strip())
                else:
                    current_text_parts.append(rest.strip())
                continue
            except ValueError:
                pass

        # Try Single bracket pattern
        m_single = single_bracket_pat.match(line_str)
        if m_single:
            flush_current()
            s_str, rest = m_single.group(1), m_single.group(2)
            try:
                s_sec = timestamp_to_seconds(s_str)
                current_start = s_sec
                current_end = None

                spk_match = re.match(r"^([A-Za-z0-9_\s]{2,30}?):\s*(.*)$", rest)
                if spk_match:
                    current_speaker = spk_match.group(1).strip()
                    current_text_parts.append(spk_match.group(2).strip())
                else:
                    current_text_parts.append(rest.strip())
                continue
            except ValueError:
                pass

        # Try Unbracketed timestamp pattern
        m_unbrk = unbracketed_pat.match(line_str)
        if m_unbrk:
            s_str, rest = m_unbrk.group(1), m_unbrk.group(2)
            # Only treat as timestamp if it parses cleanly and doesn't look like general text
            try:
                s_sec = timestamp_to_seconds(s_str)
                flush_current()
                current_start = s_sec
                current_end = None

                spk_match = re.match(r"^([A-Za-z0-9_\s]{2,30}?):\s*(.*)$", rest)
                if spk_match:
                    current_speaker = spk_match.group(1).strip()
                    current_text_parts.append(spk_match.group(2).strip())
                else:
                    current_text_parts.append(rest.strip())
                continue
            except ValueError:
                pass

        # If no pattern matched, append to current segment if active
        if current_start is not None:
            current_text_parts.append(line_str)

    flush_current()

    # Post-process: Fill in missing end_seconds
    segments: List[TranscriptSegment] = []
    for i, (s_sec, e_sec, spk, txt) in enumerate(raw_items):
        if e_sec is None or e_sec <= s_sec:
            # Look at next segment start or estimate based on word count (~150 words per minute => 2.5 words per sec)
            if i + 1 < len(raw_items):
                next_start = raw_items[i + 1][0]
                if next_start > s_sec:
                    e_sec = next_start
                else:
                    e_sec = s_sec + max(3.0, len(txt.split()) / 2.5)
            else:
                e_sec = s_sec + max(4.0, len(txt.split()) / 2.5)

        segments.append(
            TranscriptSegment(
                start_seconds=s_sec,
                end_seconds=e_sec,
                start_time=seconds_to_timestamp(s_sec),
                end_time=seconds_to_timestamp(e_sec),
                speaker=spk,
                text=txt,
            )
        )

    return segments


def parse_youtube_multiline(lines: List[str]) -> List[TranscriptSegment]:
    """
    Parses YouTube transcript copy-paste format:
    0:05
    Welcome to the show!
    0:08
    Today we have a special guest.
    """
    segments: List[TranscriptSegment] = []
    ts_pattern = re.compile(r"^(\d{1,2}:\d{2}(?::\d{2})?)$")

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        m = ts_pattern.match(line)
        if m:
            start_str = m.group(1)
            try:
                start_sec = timestamp_to_seconds(start_str)
                text_parts = []
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if ts_pattern.match(next_line):
                        break
                    if next_line:
                        text_parts.append(next_line)
                    j += 1
                
                text_content = _clean_text(" ".join(text_parts))
                if text_content:
                    # Extract speaker if format is "Speaker: text" or ">> Speaker: text" or ">> text"
                    speaker = ""
                    clean_text = text_content
                    spk_match = re.match(r"^(?:>>\s*)?([A-Za-z0-9_\s]{2,30}?):\s*(.*)$", clean_text)
                    if spk_match and not clean_text.startswith("http"):
                        speaker = spk_match.group(1).strip()
                        clean_text = spk_match.group(2).strip()

                    segments.append(
                        TranscriptSegment(
                            start_seconds=start_sec,
                            end_seconds=start_sec + 5.0,  # placeholder, adjusted below
                            start_time=seconds_to_timestamp(start_sec),
                            end_time=seconds_to_timestamp(start_sec + 5.0),
                            speaker=speaker,
                            text=clean_text,
                        )
                    )
                i = j
                continue
            except ValueError:
                pass
        i += 1

    # Adjust end times
    for idx, seg in enumerate(segments):
        if idx + 1 < len(segments):
            next_start = segments[idx + 1].start_seconds
            if next_start > seg.start_seconds:
                seg.end_seconds = next_start
                seg.end_time = seconds_to_timestamp(next_start)
        else:
            # Estimate last segment duration based on word count (~2.5 words per sec)
            est_dur = max(3.0, len(seg.text.split()) / 2.5)
            seg.end_seconds = seg.start_seconds + est_dur
            seg.end_time = seconds_to_timestamp(seg.end_seconds)

    return segments


def parse_transcript_text(content: str, source_file: Optional[str] = None) -> ParsedTranscript:
    """
    Universal parser for transcript text content. Automatically tries strategies:
    1. SRT / WebVTT arrow parser
    2. Bracket / line-based parser
    3. Multi-line YouTube parser
    """
    content = content.strip()
    if not content:
        return ParsedTranscript(segments=[], source_file=source_file)

    # Strategy 1: Check for SRT / VTT arrow
    if "-->" in content:
        srt_segments = parse_srt_or_vtt(content)
        if srt_segments:
            return ParsedTranscript(segments=srt_segments, source_file=source_file)

    lines = content.splitlines()

    # Strategy 2: Bracket / line parser
    bracket_segments = parse_bracket_or_arrow_lines(lines)
    if bracket_segments and len(bracket_segments) >= 1:
        return ParsedTranscript(segments=bracket_segments, source_file=source_file)

    # Strategy 3: Multi-line YouTube timestamp parser
    yt_segments = parse_youtube_multiline(lines)
    if yt_segments and len(yt_segments) >= 1:
        return ParsedTranscript(segments=yt_segments, source_file=source_file)

    # If nothing matched timestamp patterns, treat as single un-timestamped block starting at 0
    clean = _clean_text(content)
    if clean:
        seg = TranscriptSegment(
            start_seconds=0.0,
            end_seconds=max(60.0, len(clean.split()) / 2.5),
            start_time="00:00:00",
            end_time=seconds_to_timestamp(max(60.0, len(clean.split()) / 2.5)),
            speaker="",
            text=clean,
        )
        return ParsedTranscript(segments=[seg], source_file=source_file)

    return ParsedTranscript(segments=[], source_file=source_file)


def parse_transcript_file(file_path: str) -> ParsedTranscript:
    """
    Reads and parses a transcript file from disk (.txt, .srt, .vtt).
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Transcript file not found: {file_path}")

    # Try utf-8 first, fallback to utf-8-sig or latin-1
    for encoding in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()
            return parse_transcript_text(content, source_file=file_path)
        except UnicodeDecodeError:
            continue

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    return parse_transcript_text(content, source_file=file_path)


def save_segments_to_file(segments: List[TranscriptSegment], output_path: str) -> None:
    """
    Saves a list of TranscriptSegments to a clean, formatted text file.
    Format: [HH:MM:SS - HH:MM:SS] Speaker: text
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(seg.to_formatted_line() + "\n")
