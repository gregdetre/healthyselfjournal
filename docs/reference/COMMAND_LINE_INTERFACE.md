# Command Line Interface

## Overview

Auto-start voice recording interface with visual feedback and keyboard controls.

### Launching

The journaling loop is started from the project root:

```bash
uv run examinedlifejournal journal [--sessions-dir PATH]
```

Files default to `./sessions/`; pass `--sessions-dir` to override for archival or testing.

Tip: during a session, you can say "give me a question" to instantly get a question from the built‑in question bank (bypasses the LLM for speed/robustness).

#### Speech-to-text options

- `--stt-backend`: choose between `cloud-openai`, `local-mlx`, `local-faster`, `local-whispercpp`, or `auto-private` (local-first probe).
- `--stt-model`: preset (`default`, `accuracy`, `fast`) or explicit model id/path.
- `--stt-compute`: optional precision override for local backends (e.g. `int8_float16`). Ignored when unsupported.
- `--stt-formatting`: `sentences` (default heuristic splitter) or `raw` (unaltered backend output).

Environment variables:

- `journal`: requires `ANTHROPIC_API_KEY` for dialogue/summaries. `OPENAI_API_KEY` is only required when `--stt-backend cloud-openai` is selected.
- `reconcile`: requires `OPENAI_API_KEY` only when `--stt-backend cloud-openai` is selected. Local backends do not need API keys.

### Summaries Utilities

Minimal commands for working with summaries stored in session frontmatter:

```bash
# List (default shows only missing)
uv run examinedlifejournal summaries list [--sessions-dir PATH] [--missing-only/--all]

# Backfill (default only missing; use --all to regenerate all)
uv run examinedlifejournal summaries backfill [--sessions-dir PATH] [--llm-model SPEC] [--missing-only/--all] [--limit N]
```

- `--missing-only/--all` defaults to missing-only for both commands.
- Backfill requires `ANTHROPIC_API_KEY`.

See also (details and rationale):
- `CONVERSATION_SUMMARIES.md` – Why summaries exist, how they’re generated, and safety considerations.
- `FILE_FORMATS_ORGANISATION.md` – Where summaries live in frontmatter and related fields.
- `LLM_PROMPT_TEMPLATES.md` – Prompt template used for summary generation.

## See also

- `RECORDING_CONTROLS.md` – Detailed key mappings and recording flow for capture.
- `PRODUCT_VISION_FEATURES.md` – How the CLI supports the broader product vision.
- `../conversations/250917a_journaling_app_ui_technical_decisions.md` – Rationale behind CLI/UI choices and trade-offs.
- `CONVERSATION_SUMMARIES.md` – Summary lifecycle and backfill rationale.

## Visual Feedback

Unicode block volume meter using Python `rich` library:
```
Recording started… [████████░░░░░░░░] Press any key to stop (ESC cancels, Q quits)
```

## Display Mode

- V1: All-at-once response display
- Future: Streaming word-by-word for conversational feel
