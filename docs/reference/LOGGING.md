# Logging

## Introduction
This document describes logging in Healthy Self Journal. It covers the structured, append‑only event log used for telemetry and troubleshooting, and the lightweight use of Python’s standard logging for developer diagnostics.

## See also
- `../reference/FILE_FORMATS_ORGANISATION.md` – Where `events.log` lives and the overall session artefacts layout.
- `../reference/PRIVACY.md` – What metadata is recorded and what is explicitly never logged.
- `../reference/WEB_INTERFACE.md` – Web upload flow and where web events are emitted.
- `../../healthyselfjournal/events.py` – Canonical event logging API and sanitisation logic.
- `../../healthyselfjournal/cli_journal.py` – Initialises the event logger for CLI sessions.
- `../../healthyselfjournal/session.py` – Session lifecycle emits `session.*` events.
- `../../healthyselfjournal/tts.py` – TTS events (`tts.request`, `tts.response`, `tts.error`).
- `../../tests/test_web_app.py` – Examples of capturing events in tests by monkey‑patching `log_event`.

## Principles and key decisions
- **Privacy by design**: Events record metadata only (timestamps, filenames, durations, model identifiers, statuses). They never contain transcripts, prompts, questions, summaries, or free text.
- **JSON Lines, append‑only**: One JSON object per line in `events.log`. Easy to tail, parse, and post‑process.
- **Single shared file per sessions directory**: Events for all sessions within the selected `--sessions-dir` write to `sessions/events.log`.
- **Low coupling**: Application logic emits high‑level events; the transport is a simple file write guarded by a lock.
- **Developer logs are separate**: Python `logging` is used sparingly for stderr diagnostics and exceptions; the event log is the primary operational signal for users.

## Overview

There are two complementary logging streams:

1) Structured Event Log (recommended)
   - File: `sessions/events.log` (under your `--sessions-dir`).
   - Format: JSON Lines (UTF‑8), keys include `ts` (ISO‑8601 UTC) and `event` plus metadata.
   - API: `healthyselfjournal.events.init_event_logger(base_dir)` and `healthyselfjournal.events.log_event(event, metadata)`.

2) Python Logger (developer diagnostics)
   - Modules obtain a logger via `logging.getLogger(__name__)` and use it for warnings/errors.
   - No global handler configuration is installed by the app; messages typically go to stderr at WARNING+ depending on your environment or integration.

## Event log in detail

### Location
- The event log is created at `<sessions-dir>/events.log`.
- CLI sessions call `init_event_logger(--sessions-dir)` on startup. This ensures the file exists so external tools can tail it immediately.

### Format
- Each line is a compact JSON object with at least:
  - `ts`: ISO‑8601 timestamp in UTC
  - `event`: event name (e.g., `session.start`, `web.upload.received`)
  - Additional sanitized metadata fields

### Sanitisation and redaction
To preserve privacy, `log_event()` drops sensitive keys before writing. Keys currently redacted in `events.py`:

```12:33:healthyselfjournal/events.py
REDACT_KEYS = {
    "transcript",
    "question",
    "prompt",
    "summary",
    "content",
    "transcript_markdown",
    "current_transcript",
}
```

Other protections in `events._sanitize_metadata()` and `_to_json_safe()`:
- Converts `Path` to strings
- Serializes dataclasses via `asdict`
- Serializes exceptions as `{type, message}`
- Best‑effort conversion for non‑serializable objects

### Typical events
- CLI lifecycle: `cli.start`, `cli.error`, `cli.end`, `cli.cancelled`
- Session lifecycle: `session.start`, `session.resume`, `session.exchange.recorded`, `session.summary.updated`, `session.complete`, `session.exchange.discarded_short`
- Web UI: `web.session.started`, `web.session.resumed`, `web.upload.received`, `web.upload.processed`, `web.static.assets_missing`, `web.session.evicted`, `web.sessions.active`
- Audio/TTS: `tts.request`, `tts.response`, `tts.error`, `tts.skip`

Example lines (wrapped for readability):

```json
{"ts":"2025-09-19T16:45:12.345678+00:00","event":"cli.start","sessions_dir":"/path/to/sessions","model_llm":"anthropic:claude-sonnet-4:20250514:thinking"}
{"ts":"2025-09-19T16:45:13.012345+00:00","event":"session.exchange.recorded","session_id":"250919_1645","response_index":1,"wav":"mic_0001.wav","duration_seconds":8.52,"stt_backend":"cloud-openai","stt_model":"gpt-4o-transcribe"}
{"ts":"2025-09-19T16:45:18.000000+00:00","event":"session.summary.updated","session_id":"250919_1645","model":"anthropic:claude-sonnet-4:20250514:thinking"}
{"ts":"2025-09-19T16:45:20.000000+00:00","event":"cli.end","transcript_file":"250919_1645.md","session_id":"250919_1645"}
```

### Using the API

Minimal usage (for integrators or scripts):

```python
from pathlib import Path
from healthyselfjournal.events import init_event_logger, log_event

init_event_logger(Path("/path/to/sessions"))
log_event("my.custom.event", {"status": "ok", "duration_seconds": 1.23})
```

Notes:
- If `init_event_logger()` has not been called, `log_event()` is a no‑op.
- The CLI initialises this automatically; the web server currently does not (see transition notes below).

### Tail and filter
- Tail in real time:
  - `tail -f sessions/events.log`
- Pretty‑print with `jq`:
  - `jq . sessions/events.log | less`
- Filter by event type:
  - `jq 'select(.event=="session.exchange.recorded")' sessions/events.log`
- Count by event type:
  - `jq -r '.event' sessions/events.log | sort | uniq -c | sort -nr`

### Rotation and housekeeping
- There is no built‑in rotation. You can safely truncate or delete `events.log`; it will be recreated on the next write.
- The file contains metadata only; deleting it does not affect your session transcripts or audio.

## Python logging (stderr diagnostics)

- Modules use `logging.getLogger(__name__)` for warnings and exceptions, primarily during error handling or background tasks.
- The application does not install a global logging configuration. If you embed HSJ, you may configure logging yourself, e.g.:

```python
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
```

- In the CLI, error conditions are also surfaced to the console via Rich; the event log remains the canonical record of operational state.

## Documenting systems in transition

Current state
- CLI: Calls `init_event_logger(--sessions-dir)`; structured events are written to `sessions/events.log`.
- Web: Emits `log_event(...)` throughout request handlers, but does not currently initialise the event logger. If you run only the web server first, events will be dropped unless an external caller initialises the logger.

Target state
- Initialise event logging automatically for the web server using the configured `sessions_dir` when the app starts.

Migration status
- Pending: Add `init_event_logger(config.sessions_dir)` during web app startup.

## Troubleshooting
- No `events.log` file appears
  - Ensure the CLI was run (it initialises the logger), or call `init_event_logger(Path(--sessions-dir))` in your entrypoint.
- Seeing `tts.error` entries
  - Confirm `OPENAI_API_KEY` is set when TTS or cloud STT is enabled; see `AUDIO_SPEECH_GENERATION.md`.
- Web uploads succeed but no events are recorded
  - See transition note above; initialise the event logger for the web server.

## Quality checklist
- Cross‑references link to canonical sources (`events.py`, privacy docs, file formats).
- No content duplication of lower‑level details beyond what’s needed for orientation.
- Examples reflect current event names and usage patterns.
- Transitional state is clearly marked with current/target/migration.


