# File Formats and Organisation

## Overview

Session transcripts remain flat, with per-session assets stored in a sibling subdirectory named after the transcript stem.

## See also

- `ARCHITECTURE.md` - Storage layer design and data flow patterns
- `CONVERSATION_SUMMARIES.md` - Frontmatter content
- `PRODUCT_VISION_FEATURES.md` - Persistence requirements
- `../conversations/250117a_journaling_app_ui_technical_decisions.md` - Format decisions

## Directory Structure

Base directory (default `./sessions/`) contains (minute-level session IDs; no seconds):
- `yyMMdd_HHmm.md` — Transcript and dialogue for the session
- `yyMMdd_HHmm/` — Folder containing all session assets:
  - CLI captures: `yyMMdd_HHmm_XX.wav` (and optional `yyMMdd_HHmm_XX.mp3` when `ffmpeg` is present)
  - Web captures: `browser-XXX.webm` (no transcoding; recorded as `audio/webm;codecs=opus`)
  - `*.stt.json` — Raw transcription payload written beside each clip regardless of source
  - Frontmatter records the canonical filename under the `wav` key (so web entries look like `{wav: "browser-001.webm", mp3: null, duration_seconds: 1.5}`)

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

## Event Log Schema

Events recorded in `sessions/events.log` are emitted through `healthyselfjournal.events.log_event`. Payloads include:
- `ui`: source of the interaction (`cli` or `web`)
- `session_id`: current session identifier
- `response_index`: sequential index when applicable
- Additional context depending on the event (`segment_label`, durations, backend/model identifiers)
