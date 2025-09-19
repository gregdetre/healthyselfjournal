# File Formats and Organisation

## Overview

Session transcripts remain flat, with per-session assets stored in a sibling subdirectory named after the transcript stem.

## See also

- `ARCHITECTURE.md` - Storage layer design and data flow patterns
- `CONVERSATION_SUMMARIES.md` - Frontmatter content
- `PRODUCT_VISION_FEATURES.md` - Persistence requirements
- `RESILIENCE.md` - Failure handling, placeholders, and recovery flow
- `../conversations/250117a_journaling_app_ui_technical_decisions.md` - Format decisions

## Directory Structure

Base directory (default `./sessions/`) contains (minute-level session IDs; no seconds):
- `yyMMdd_HHmm.md` — Transcript and dialogue for the session
- `yyMMdd_HHmm/` — Folder containing all session assets:
  - CLI captures: `yyMMdd_HHmm_XX.wav` (and optional `yyMMdd_HHmm_XX.mp3` when `ffmpeg` is present)
  - Web captures: `browser-XXX.webm` (no transcoding; recorded as `audio/webm;codecs=opus`)
  - `*.stt.json` — Raw transcription payload written beside each clip regardless of source
  - Frontmatter records the canonical filename under the `wav` key (so web entries look like `{wav: "browser-001.webm", mp3: null, duration_seconds: 1.5, voiced_seconds: 1.1}`)

Note: Extremely short, low‑voiced takes may be auto‑discarded. In those cases no `.wav`, `.mp3`, or `.stt.json` is kept.

By default, large `.wav` files are automatically deleted once both the `.mp3` and `.stt.json` exist. This saves disk space while retaining a compressed audio copy and the raw transcription payload. To keep WAVs, pass `--keep-wav` on the CLI or set `CONFIG.delete_wav_when_safe=False`.

## Markdown Format

```markdown
---
summary: LLM-generated session summary
---

## AI Q

```llm-question
First question from LLM (may span multiple lines)
```

User's transcribed response here

## AI Q

```llm-question
Follow-up question (may span multiple lines)
```

Next response...
```

## File Persistence

- Audio segments saved immediately after each recording stop
- Transcript saved after each Whisper transcription (skipped for auto‑discarded takes)
- Summary updated after each Q&A exchange
- MP3 conversion runs in the background when `ffmpeg` is present; WAV files remain canonical
- Frontmatter (`audio_file`, `duration_seconds`, etc.) is only mutated via `SessionManager` helpers; both CLI and web uploads share the same code path.
- Pending entries set `pending: true` and `pending_reason` until reconcile backfills the transcript. `voiced_seconds` is stored alongside `duration_seconds` for debugging.

## Pending transcriptions

When transcription fails, the dialogue body captures the AI prompt and inserts a placeholder response:

```
(transcription pending – segment yyMMdd_HHmm_03.wav)
<!-- hsj:pending segment="yyMMdd_HHmm_03.wav" -->
```

Frontmatter adds/updates the matching `audio_file` item with `pending: true`, the error class in `pending_reason`, and the recorded duration/voiced seconds. A companion sentinel `<segment>.stt.error.txt` stores a timestamped error summary.

`healthyselfjournal reconcile` replaces the placeholder with the real transcript, writes the `.stt.json` payload atomically, removes the sentinel, and clears `pending` metadata. See `RESILIENCE.md` for the full recovery flow and failure-mode catalogue.

## Event Log Schema

Events recorded in `sessions/events.log` are emitted through `healthyselfjournal.events.log_event`. Payloads include:
- `ui`: source of the interaction (`cli` or `web`)
- `session_id`: current session identifier
- `response_index`: sequential index when applicable
- Additional context depending on the event (`segment_label`, durations, backend/model identifiers)
