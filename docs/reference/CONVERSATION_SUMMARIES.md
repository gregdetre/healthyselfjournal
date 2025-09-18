# Conversation Summaries

Short LLM-generated summaries are stored in markdown frontmatter for each session. They are refreshed during a session (background worker) and used to provide context for future prompts.

## See also

- `FILE_FORMATS_ORGANISATION.md` – Where summaries are stored and related keys.
- `LLM_PROMPT_TEMPLATES.md` – Summary prompt template and variables.
- `COMMAND_LINE_INTERFACE.md` – CLI commands for listing/backfilling summaries.
- `../conversations/250916a_journaling_app_dialogue_design.md` – Why continuity and summaries matter.

## Summary generation (current state)

- Triggered after each exchange via a background worker to reduce latency.
- Persisted under `summary` in frontmatter (single-line normalized on write).
- Captures arc, themes, and suggested next steps.
- May briefly lag behind the latest exchange while the background task runs.

## Context usage

- Recent summaries feed into future prompts for continuity.
- Enables pattern detection across sessions; improves personalization.

## Concurrency and safety

- All writes (frontmatter/body/summary) serialized via in-process lock.
- Background worker snapshots body for LLM call; reloads before write to merge.
- Graceful shutdown flushes pending tasks on session completion.

## Backfill

Use the CLI to list and (re)generate summaries for existing session markdown files:

```bash
# Show sessions missing summaries (default)
uv run healthyselfjournal summaries list --sessions-dir ./sessions

# Show all sessions with status
uv run healthyselfjournal summaries list --sessions-dir ./sessions --all

# Backfill only missing summaries (default)
uv run healthyselfjournal summaries backfill --sessions-dir ./sessions

# Regenerate all summaries (overwrite existing)
uv run healthyselfjournal summaries backfill --sessions-dir ./sessions --all
```

Notes:
- Uses the same prompt and recent-history context as the live flow.
- Requires `ANTHROPIC_API_KEY` only when the chosen `--llm-model` uses the Anthropic provider; local `ollama:*` models run offline (ensure the Ollama daemon is available).
