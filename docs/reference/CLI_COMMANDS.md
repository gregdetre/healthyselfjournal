# CLI Commands

## Overview
Single source for command discovery. See linked pages for detailed flags.

## Primary commands

- `journal cli` – Terminal-based recording. See `CLI_RECORDING_INTERFACE.md`.
- `journal web` – Launch local browser interface. See `WEB_RECORDING_INTERFACE.md`.
- `journal list --pending` – Show sessions with outstanding transcription segments and the reconcile hint.
- `session list` – Show sessions with summary snippets. See `SESSIONS.md`.
- `summarise ...` – List/backfill/regenerate summaries.
- `reconcile` – Backfill missing STT for saved WAV/webm/ogg files, replace markdown placeholders, and remove error sentinels.
- `merge` – Merge two sessions into the earlier one.
- `init` – Setup wizard for first-time configuration.
- `diagnose` – Diagnostics for mic/STT, local/cloud LLM, and TTS.

### Local LLM bootstrap

- `init local-llm --url <gguf_url> [--sha256 <checksum>] [--model <filename>]`
  - Downloads a `.gguf` model into the managed directory:
    `~/Library/Application Support/HealthySelfJournal/models/llama/` on macOS.
  - If you set `[llm].local_model_url` and `local_model_sha256` in `user_config.toml`, you can omit flags.
  - Example:

```bash
uv run --active healthyselfjournal init local-llm \
  --url https://huggingface.co/.../llama-3.1-8b-instruct-q4_k_m.gguf \
  --sha256 <expected_sha256>
```

Related:
- `diagnose local llm` will suggest the command above if the model file is missing.

## Structure

Each command lives in its own `cli_*.py` module for clarity:
- `cli_journal_cli.py` – journaling CLI sub-app
- `cli_journal_web.py` – journaling web sub-app
- `cli_session.py` – session utilities
- `cli_summarise.py` – summaries utilities
- `cli_diagnose.py` – diagnostics sub-app (mic/local/cloud)
- `cli_reconcile.py`, `cli_merge.py`, `cli_init.py` – other commands

## Examples

```bash
# Start CLI journaling
uvx healthyselfjournal -- journal cli --voice-mode

# Start web interface on a different port, resume latest session
uvx healthyselfjournal -- journal web --port 8888 --resume

# List sessions in a custom directory (first 200 chars)
uvx healthyselfjournal -- session list --sessions-dir ./sessions --nchars 200

# Summaries
uvx healthyselfjournal -- summarise list --missing-only
uvx healthyselfjournal -- summarise backfill --limit 10
uvx healthyselfjournal -- summarise regenerate 250918_0119.md
```

## See also

- `CLI_RECORDING_INTERFACE.md`
- `WEB_RECORDING_INTERFACE.md`
- `SESSIONS.md`
