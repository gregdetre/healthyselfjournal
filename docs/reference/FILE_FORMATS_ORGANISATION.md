# File Formats and Organisation

## Overview

Session transcripts remain flat, with per-session assets stored in a sibling subdirectory named after the transcript stem.

## See also

- `CONVERSATION_SUMMARIES.md` - Frontmatter content
- `PRODUCT_VISION_FEATURES.md` - Persistence requirements
- `../conversations/250117a_journaling_app_ui_technical_decisions.md` - Format decisions

## Directory Structure

Base directory (default `./sessions/`) contains:
- `yyMMdd_HHmm.md` — Transcript and dialogue for the session
- `yyMMdd_HHmm/` — Folder containing all session assets:
  - `yyMMdd_HHmm_XX.wav` — Audio segment for response `XX`
  - `yyMMdd_HHmm_XX.mp3` — Optional MP3 sibling when `ffmpeg` is available
  - `yyMMdd_HHmm_XX.stt.json` — Raw Whisper transcription payload for that segment

Note: Extremely short, low‑voiced takes may be auto‑discarded. In those cases no `.wav`, `.mp3`, or `.stt.json` is kept.

## Markdown Format

```markdown
---
summary: LLM-generated session summary
---

## AI Q: First question from LLM

User's transcribed response here

## AI Q: Follow-up question

Next response...
```

## File Persistence

- Audio segments saved immediately after each recording stop
- Transcript saved after each Whisper transcription (skipped for auto‑discarded takes)
- Summary updated after each Q&A exchange
- MP3 conversion runs in the background when `ffmpeg` is present; WAV files remain canonical
