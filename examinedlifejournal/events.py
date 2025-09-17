from __future__ import annotations

"""Append-only, non-sensitive metadata event logger (JSON Lines).

This module provides a minimal, centralised event logger that writes one JSON
object per line to a single file within the selected sessions directory.

Privacy constraints:
- No transcripts, prompts, questions, or summaries are ever logged.
- Only metadata like filenames, durations, model names, counts, and statuses.
"""

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import threading
from pathlib import Path
from typing import Any, Mapping

_LOCK = threading.Lock()
_EVENT_LOG_PATH: Path | None = None


REDACT_KEYS = {
    "transcript",
    "question",
    "prompt",
    "summary",
    "content",
    "transcript_markdown",
    "current_transcript",
}


def init_event_logger(base_dir: Path) -> None:
    """Initialise the event log file under the given base directory.

    The file is append-only and shared across all sessions within this directory.
    """
    global _EVENT_LOG_PATH
    base_dir.mkdir(parents=True, exist_ok=True)
    _EVENT_LOG_PATH = base_dir / "events.log"
    # Ensure the file exists so external tools can tail it immediately
    _EVENT_LOG_PATH.touch(exist_ok=True)


def get_event_log_path() -> Path | None:
    return _EVENT_LOG_PATH


def log_event(event: str, metadata: Mapping[str, Any] | None = None) -> None:
    """Write a single metadata event as JSON to the shared events.log file.

    If the logger is not initialised, this is a no-op.
    """
    if _EVENT_LOG_PATH is None:
        return

    payload: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
    }

    if metadata:
        safe_meta = _sanitize_metadata(metadata)
        payload.update(safe_meta)

    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    with _LOCK:
        with _EVENT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _sanitize_metadata(meta: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a JSON-safe mapping with sensitive keys removed and values serialised.

    - Drops any keys that appear in REDACT_KEYS.
    - Converts Paths to strings.
    - Converts dataclasses to dicts via asdict.
    - Converts Exceptions to their class name and string message.
    - Best-effort conversion for other non-serialisable types.
    """
    out: dict[str, Any] = {}
    for key, value in meta.items():
        if key in REDACT_KEYS:
            continue

        out[key] = _to_json_safe(value)

    return out


def _to_json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(v) for v in value]

    if is_dataclass(value):
        try:
            return _to_json_safe(asdict(value))
        except Exception:
            # Fallback if dataclass contains non-serialisable fields
            return {"type": value.__class__.__name__}

    if isinstance(value, BaseException):
        return {
            "type": value.__class__.__name__,
            "message": str(value),
        }

    # Fallback to string representation
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)
