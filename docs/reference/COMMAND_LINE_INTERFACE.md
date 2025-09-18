# Command Line Interface

## Overview

Auto-start voice recording interface with visual feedback and keyboard controls.

### Launching

The journaling loop is started from the project root:

```bash
uv run examinedlifejournal journal [--sessions-dir PATH] [--stream-llm/--no-stream-llm] [--voice-mode/--no-voice-mode] [--tts-model SPEC] [--tts-voice NAME] [--tts-format FORMAT]
```

Files default to `./sessions/`; pass `--sessions-dir` to override for archival or testing.

Tip: during a session, you can say "give me a question" to instantly get a question from the built‑in question bank (bypasses the LLM for speed/robustness).

Getting started:

- First-time users should run the setup wizard:
  ```bash
  examinedlifejournal init
  # or
  uvx examinedlifejournal -- init
  ```
- See `INIT_FLOW.md` for the init wizard flow and configuration details.

#### Speech-to-text options

- `--stt-backend`: choose between `cloud-openai`, `local-mlx`, `local-faster`, `local-whispercpp`, or `auto-private` (local-first probe).
- `--stt-model`: preset (`default`, `accuracy`, `fast`) or explicit model id/path.
- `--stt-compute`: optional precision override for local backends (e.g. `int8_float16`). Ignored when unsupported.
- `--stt-formatting`: `sentences` (default heuristic splitter) or `raw` (unaltered backend output).

Environment variables:

- `journal`: requires `ANTHROPIC_API_KEY` for dialogue/summaries. `OPENAI_API_KEY` is required when `--stt-backend cloud-openai` or when `--voice-mode` is enabled (OpenAI TTS).
- `reconcile`: requires `OPENAI_API_KEY` only when `--stt-backend cloud-openai` is selected. Local backends do not need API keys.

### Summaries Utilities

Minimal commands for working with summaries stored in session frontmatter:

```bash
# List (default shows only missing)
uv run examinedlifejournal summaries list [--sessions-dir PATH] [--missing-only/--all]

# Backfill (default only missing; use --all to regenerate all)
uv run examinedlifejournal summaries backfill [--sessions-dir PATH] [--llm-model SPEC] [--missing-only/--all] [--limit N]

# Regenerate a single file's summary
uv run examinedlifejournal summaries regenerate [--sessions-dir PATH] [--llm-model SPEC] yyMMdd_HHmm[.md]
```

- `--missing-only/--all` defaults to missing-only for both commands.
- Backfill requires `ANTHROPIC_API_KEY`.

### Merge sessions

Merge two sessions, keeping the earlier one. Moves assets, appends later Q&A to earlier, updates frontmatter, and regenerates the summary by default.

```bash
uv run examinedlifejournal merge [--sessions-dir PATH] [--llm-model SPEC] [--regenerate/--no-regenerate] [--dry-run] [--ignore-missing] yyMMdd_HHmm[.md] yyMMdd_HHmm[.md]
```

Notes:
- Asset filename collisions are avoided by suffixing with `_N` when needed.
- The later session folder is removed if empty after moving.
- Summary regeneration requires `ANTHROPIC_API_KEY`.
 - After a successful merge, the later `.md` file is deleted.
 - Frontmatter `audio_file` becomes MP3-centric (e.g., `{wav: null, mp3: <file>, duration_seconds: <float>}`).
 - If the later assets folder is missing, run again with `--ignore-missing` to proceed.

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

- **Streaming**: Default streams the next question word-by-word (`--stream-llm`).
- **Speech**: Enable `--voice-mode` to speak the assistant's questions out loud using OpenAI TTS. When speech is enabled, streaming display is automatically disabled for clarity.
- Disable streaming manually with `--no-stream-llm` to show all-at-once.

### Speech options

- `--voice-mode/--no-voice-mode`: convenience switch that enables speech with default settings.
- `--tts-model`: TTS model (default `gpt-4o-mini-tts`).
- `--tts-voice`: TTS voice (default `shimmer`).
- `--tts-format`: audio format for playback (default `wav`).

Examples:
```bash
# One-flag voice mode with defaults (shimmer, gpt-4o-mini-tts, wav)
uv run examinedlifejournal journal --voice-mode

# Explicit control
uv run examinedlifejournal journal --speak-llm --tts-voice shimmer --tts-model gpt-4o-mini-tts --tts-format wav
```

Notes:
- macOS uses `afplay` for local playback. If unavailable, `ffplay` is attempted.
- Only assistant questions are spoken; summaries and status messages remain text-only.
 - While a question is being spoken, press ENTER to skip the voice playback immediately.
