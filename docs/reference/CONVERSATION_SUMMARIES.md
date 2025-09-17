# Conversation Summaries

## Overview

LLM-generated summaries stored in frontmatter, regenerated after each Q&A within a session for crash resilience.

## See also

- `FILE_FORMATS_ORGANISATION.md` - Where summaries are stored
- `LLM_PROMPT_TEMPLATES.md` - How summaries provide context
- `../conversations/250916a_journaling_app_dialogue_design.md` - Continuity rationale

## Summary Generation

- Generated/regenerated after each question-answer exchange
- Stored in `summary` frontmatter field
- Captures conversation arc and key themes

## Context Usage

- Recent summaries included in LLM prompts
- Enables pattern detection across sessions
- Creates "knows you" feeling

## Backfill Process

Planned utility to generate missing summaries from existing transcripts.