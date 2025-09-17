# File Formats and Organisation

## Overview

Flat directory structure with paired audio and markdown files per session.

## See also

- `CONVERSATION_SUMMARIES.md` - Frontmatter content
- `PRODUCT_VISION_FEATURES.md` - Persistence requirements
- `../conversations/250117a_journaling_app_ui_technical_decisions.md` - Format decisions

## Directory Structure

Flat directory containing:
- `yyMMdd_HHmm.mp3` - Audio recording
- `yyMMdd_HHmm.md` - Transcript and dialogue

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

- Audio saved immediately after recording
- Transcript saved after each Whisper transcription
- Summary updated after each Q&A exchange