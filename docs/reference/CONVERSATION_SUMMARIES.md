# Conversation Summaries

## Overview

LLM-generated summaries stored in frontmatter. For resilience, summaries are refreshed throughout the session; to reduce latency, regeneration now runs in the background.

## See also

- `FILE_FORMATS_ORGANISATION.md` - Where summaries are stored
- `LLM_PROMPT_TEMPLATES.md` - How summaries provide context
- `../conversations/250916a_journaling_app_dialogue_design.md` - Continuity rationale

## Summary Generation

- Scheduled after each question-answer exchange, executed in a background worker
- Stored in `summary` frontmatter field
- Captures conversation arc and key themes
- May briefly lag behind the most recent exchange while the background task runs

## Context Usage

- Recent summaries included in LLM prompts
- Enables pattern detection across sessions
- Creates "knows you" feeling

## Concurrency & Safety

- All transcript writes (frontmatter/body/summary) are serialized with an in-process lock
- Background worker snapshots the transcript body for the LLM call and reloads before write to avoid clobbering concurrent updates
- On session completion, the worker is shut down gracefully to flush pending writes

## Backfill Process

Planned utility to generate missing summaries from existing transcripts.