"""
gemini_client.py - Gemini AI client for podcast transcript analysis and clip selection.

Builds structured prompts, requests clip candidates from Gemini API,
and parses/validates the structured JSON output.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

# Suppress non-critical Google SDK deprecation and grpc quantum warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="google")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from transcript_parser import ParsedTranscript, seconds_to_timestamp, timestamp_to_seconds


class GeminiAPIError(Exception):
    """Raised when Gemini API fails or returns an error."""
    pass


class GeminiResponseParseError(Exception):
    """Raised when Gemini response cannot be parsed as valid clip JSON."""
    pass


@dataclass
class ClipCandidate:
    start_time: str
    end_time: str
    start_seconds: float
    end_seconds: float
    title: str
    reason: str
    score: int
    key_quote: str = ""
    summary: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "start_seconds": round(self.start_seconds, 2),
            "end_seconds": round(self.end_seconds, 2),
            "duration_seconds": round(self.duration, 2),
            "score": self.score,
            "reason": self.reason,
            "key_quote": self.key_quote,
            "summary": self.summary,
        }


def _clean_json_string(text: str) -> str:
    """
    Extracts JSON array or object from raw LLM output and fixes common JSON quirks.
    """
    text = text.strip()

    # Match ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    # Find first '[' or '{' and last ']' or '}'
    start_bracket = text.find("[")
    start_brace = text.find("{")

    if start_bracket != -1 and (start_brace == -1 or start_bracket < start_brace):
        end_bracket = text.rfind("]")
        if end_bracket != -1:
            text = text[start_bracket : end_bracket + 1]
    elif start_brace != -1:
        end_brace = text.rfind("}")
        if end_brace != -1:
            text = text[start_brace : end_brace + 1]

    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([\]}])", r"\1", text)

    return text


def build_clip_prompt(
    transcript_text: str,
    min_clip_len: int = 20,
    max_clip_len: int = 90,
    num_clips: int = 8,
) -> str:
    """
    Constructs an optimized prompt for Gemini to find the most viral/insightful clips.
    """
    return f"""You are a world-class viral video editor and podcast producer.
Your job is to analyze the following timestamped podcast transcript and select the top {num_clips} most engaging, high-retention, and shareable clips.

### CLIP CRITERIA:
1. **Hook & Retention**: Starts with a compelling question, bold statement, shocking story, or captivating insight that hooks the viewer in the first 3 seconds.
2. **Self-Contained**: The clip must make complete sense on its own without needing prior context from the rest of the episode. It must have a clear beginning, middle, and punchline/resolution.
3. **Clip Duration**: Every clip MUST be between {min_clip_len} and {max_clip_len} seconds long (approx. {min_clip_len}s - {max_clip_len}s).
4. **High Value / Emotion**: Prioritize moments with strong opinions, humor, intense emotion, rare insights, mind-blowing facts, or hilarious banter.
5. **Accurate Timestamps**: The `start_time` and `end_time` MUST match the timestamps provided in the transcript (format `HH:MM:SS` or `MM:SS`).
6. **Diversity**: Pick clips from across different moments of the episode covering different interesting topics. Do not overlap clip timestamps.
7. **Relevance**:  The main objective is to cut ANNA AI's (The main topic of the podcast) launch content into clips that get US parents signing up for a free trial.

### What works

- **One specific thing that slipped.** The permission slip in the inbox. The soccer time that changed in the class chat. The birthday gift both parents thought the other bought. Specific beats general every time
- **The mental load, named.** Parents know that feeling and have no word for it. Say it plainly
- **Talking while driving.** Anna is hands-free. That is a moment every parent recognises
- **Your own captions and voiceover are encouraged.** Building on top of the footage is fine and usually performs better than a straight re-post

### TRANSCRIPT:
\"\"\"
{transcript_text}
\"\"\"

### OUTPUT FORMAT:
Respond with ONLY a valid JSON array of objects. No additional markdown or conversational text.
Schema:
[
  {{
    "title": "Short, catchy, clickable title (3-7 words, no clickbait lies)",
    "start_time": "HH:MM:SS",
    "end_time": "HH:MM:SS",
    "score": 9,
    "reason": "Why this moment works well as a standalone clip (hook, punchline, insight)",
    "key_quote": "A memorable 1-sentence quote or punchline from this segment",
    "summary": "1 sentence summarizing what happens in this clip"
  }}
]
"""


class GeminiClipFinderClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self._sdk_type = None

        if not self.api_key:
            raise GeminiAPIError(
                "Gemini API key not found. Please set GEMINI_API_KEY in your environment "
                "or .env file, or pass --api-key <KEY>."
            )

    def _get_candidate_models(self) -> List[str]:
        """Returns a prioritized list of models to try in case of 404 not found."""
        candidates = [self.model_name]
        fallbacks = [
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro",
            "gemini-2.5-flash",
            "gemini-2.0-flash-lite",
        ]
        for fb in fallbacks:
            if fb not in candidates:
                candidates.append(fb)
        return candidates

    def _call_gemini_api(self, prompt: str) -> str:
        """
        Calls Gemini API using either modern google-genai or legacy google-generativeai SDK
        with automatic fallback across model IDs.
        """
        candidate_models = self._get_candidate_models()
        last_error = None

        # Strategy 1: Try modern google-genai SDK (recommended)
        has_google_genai = False
        try:
            from google import genai
            from google.genai import types
            has_google_genai = True

            client = genai.Client(api_key=self.api_key)
            for model_id in candidate_models:
                try:
                    response = client.models.generate_content(
                        model=model_id,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.2,
                        ),
                    )
                    self._sdk_type = f"google-genai ({model_id})"
                    if response and response.text:
                        return response.text
                except Exception as e:
                    err_msg = str(e)
                    if "API_KEY_INVALID" in err_msg or "PERMISSION_DENIED" in err_msg:
                        raise GeminiAPIError(f"Gemini API authentication failed: {err_msg}")
                    
                    # If model not found (404), try next candidate model
                    if "404" in err_msg or "NOT_FOUND" in err_msg or "not found" in err_msg.lower():
                        last_error = e
                        continue
                    
                    last_error = e
        except ImportError:
            pass

        # Strategy 2: Try legacy google-generativeai SDK as fallback
        try:
            import google.generativeai as legacy_genai

            legacy_genai.configure(api_key=self.api_key)
            for model_id in candidate_models:
                try:
                    gen_model = legacy_genai.GenerativeModel(
                        model_name=model_id,
                        generation_config={"response_mime_type": "application/json", "temperature": 0.2},
                    )
                    response = gen_model.generate_content(prompt)
                    self._sdk_type = f"google-generativeai ({model_id})"
                    if response and response.text:
                        return response.text
                except Exception as e:
                    err_msg = str(e)
                    if "API_KEY_INVALID" in err_msg or "PERMISSION_DENIED" in err_msg:
                        raise GeminiAPIError(f"Gemini API authentication failed: {err_msg}")
                    if "404" in err_msg or "NOT_FOUND" in err_msg or "not found" in err_msg.lower():
                        last_error = e
                        continue
                    last_error = e
        except ImportError:
            pass

        if not has_google_genai and "legacy_genai" not in locals():
            raise GeminiAPIError(
                "Neither 'google-genai' nor 'google-generativeai' is installed. "
                "Please run: pip install google-genai"
            )

        if last_error:
            raise GeminiAPIError(f"Gemini API call failed: {last_error}")

        raise GeminiAPIError("Failed to generate content from Gemini API.")

    def parse_clips_response(
        self,
        raw_text: str,
        min_clip_len: int = 15,
        max_clip_len: int = 120,
    ) -> List[ClipCandidate]:
        """
        Parses raw LLM text into a validated list of ClipCandidate objects.
        """
        cleaned_json = _clean_json_string(raw_text)
        try:
            parsed = json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            raise GeminiResponseParseError(
                f"Failed to parse Gemini response as JSON: {e}\nRaw output:\n{raw_text[:500]}"
            )

        # Allow either list of clips or dict with "clips" key
        if isinstance(parsed, dict):
            if "clips" in parsed and isinstance(parsed["clips"], list):
                parsed = parsed["clips"]
            elif "results" in parsed and isinstance(parsed["results"], list):
                parsed = parsed["results"]
            else:
                # Single clip object wrapped in dict
                parsed = [parsed]

        if not isinstance(parsed, list):
            raise GeminiResponseParseError(f"Expected JSON list of clips, got {type(parsed).__name__}")

        candidates: List[ClipCandidate] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue

            title = str(item.get("title", "Untitled Clip")).strip()
            reason = str(item.get("reason", "")).strip()
            key_quote = str(item.get("key_quote", "")).strip()
            summary = str(item.get("summary", "")).strip()

            # Score
            raw_score = item.get("score", 7)
            try:
                score = int(round(float(raw_score)))
                score = max(1, min(10, score))
            except (ValueError, TypeError):
                score = 7

            # Timestamps
            start_val = item.get("start_time") or item.get("start") or item.get("start_seconds")
            end_val = item.get("end_time") or item.get("end") or item.get("end_seconds")

            if start_val is None or end_val is None:
                continue

            try:
                if isinstance(start_val, (int, float)):
                    start_sec = float(start_val)
                    start_str = seconds_to_timestamp(start_sec)
                else:
                    start_str = str(start_val).strip()
                    start_sec = timestamp_to_seconds(start_str)

                if isinstance(end_val, (int, float)):
                    end_sec = float(end_val)
                    end_str = seconds_to_timestamp(end_sec)
                else:
                    end_str = str(end_val).strip()
                    end_sec = timestamp_to_seconds(end_str)

            except (ValueError, TypeError):
                continue

            # Ensure valid range
            if end_sec <= start_sec:
                continue

            duration = end_sec - start_sec
            # Allow some tolerance around min/max
            if duration < (min_clip_len * 0.7) or duration > (max_clip_len * 1.4):
                if duration < 5.0:
                    continue

            candidates.append(
                ClipCandidate(
                    start_time=start_str,
                    end_time=end_str,
                    start_seconds=start_sec,
                    end_seconds=end_sec,
                    title=title,
                    reason=reason,
                    score=score,
                    key_quote=key_quote,
                    summary=summary,
                )
            )

        # Sort clips by score descending
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def find_clips(
        self,
        transcript: ParsedTranscript,
        min_clip_len: int = 20,
        max_clip_len: int = 90,
        num_clips: int = 8,
    ) -> List[ClipCandidate]:
        """
        Sends formatted transcript to Gemini, retrieves and parses clip selections.
        Includes automatic retry if first attempt yields malformed output.
        """
        if transcript.is_empty():
            raise GeminiAPIError("Transcript is empty. Cannot extract clips.")

        transcript_text = transcript.full_text
        prompt = build_clip_prompt(
            transcript_text=transcript_text,
            min_clip_len=min_clip_len,
            max_clip_len=max_clip_len,
            num_clips=num_clips,
        )

        # First attempt
        raw_response = self._call_gemini_api(prompt)
        try:
            clips = self.parse_clips_response(
                raw_response,
                min_clip_len=min_clip_len,
                max_clip_len=max_clip_len,
            )
            if clips:
                return clips[:num_clips]
        except GeminiResponseParseError:
            pass

        # Retry once with stricter formatting instruction
        retry_prompt = (
            prompt
            + "\n\nCRITICAL: Your previous response could not be parsed. "
            "Return ONLY raw JSON with no markdown backticks, starting with '[' and ending with ']'. "
            "Ensure every object has 'start_time', 'end_time', 'title', 'reason', and 'score'."
        )

        retry_response = self._call_gemini_api(retry_prompt)
        clips = self.parse_clips_response(
            retry_response,
            min_clip_len=min_clip_len,
            max_clip_len=max_clip_len,
        )

        if not clips:
            raise GeminiResponseParseError(
                "Gemini did not return any valid clips after retry. "
                f"Raw response:\n{retry_response[:400]}"
            )

        return clips[:num_clips]
