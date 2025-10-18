# CLI Commands

## Overview
Single source for command discovery. See linked pages for detailed flags.

## Primary commands

- `version` – Print the installed package version.
- `journal cli` – Terminal-based recording. See `CLI_RECORDING_INTERFACE.md`.
- `journal web` – Launch local browser interface. See `WEB_RECORDING_INTERFACE.md`.
- `journal desktop` – Launch PyWebView desktop shell. See `DESKTOP_APP_PYWEBVIEW.md`.
- `sessions list` – Show sessions with summary snippets. See `SESSIONS.md`.
- `sessions summaries` – Show which sessions are missing summaries (moved from summarise list).
- `fix stt` – Backfill missing STT for saved WAV/webm/ogg files, replace markdown placeholders, and remove error sentinels.
- `fix backfill` – Generate summaries where missing.
- `fix regenerate <file>` – Regenerate a summary for a specific session file.
- `sessions merge` – Merge two sessions into the earlier one.
- `init` – Setup wizard for first-time configuration.
- `diagnose` – Diagnostics for mic/STT, local/cloud LLM, and TTS.
  - `diagnose desktop` – Probe desktop web shell routes (`/setup`, `/`) to catch template/context errors.

### Insight

- `insight list` – List existing insights files (newest first).
- `insight generate` – Generate one or more reflective insights using the two-range default.

Examples:

```bash
uv run --active healthyselfjournal insight list --sessions-dir ./sessions
uv run --active healthyselfjournal insight generate --sessions-dir ./sessions --llm-model anthropic:claude-sonnet-4:20250514:thinking
# Let the model decide the number (default)
uv run --active healthyselfjournal insight generate --sessions-dir ./sessions
# Explicit count
uv run --active healthyselfjournal insight generate --sessions-dir ./sessions --count 3
```

### Local LLM bootstrap

- `init local-llm --url <gguf_url> [--sha256 <checksum>] [--model <filename>]`
  - Downloads a `.gguf` model into the managed directory:
    `~/Library/Application Support/HealthySelfJournal/models/llama/` on macOS.
  - If you set `[llm].local_model_url` and `local_model_sha256` in `user_config.toml`, you can omit flags.
  - If `--url` is omitted in a TTY, the CLI will offer to paste a URL interactively,
    optionally accept a SHA-256, and can save these to `user_config.toml` for reuse.
  - You can also resolve from Hugging Face and auto-fetch SHA-256 with:
    `--hf-repo <repo_id> --hf-file <filename> [--hf-revision <rev>]`.
  - Example:

```bash
uv run --active healthyselfjournal init local-llm \
  --url https://huggingface.co/.../llama-3.1-8b-instruct-q4_k_m.gguf \
  --sha256 <expected_sha256>
```

```bash
# Resolve from Hugging Face (auto-resolves URL and sha256)
uv run --active healthyselfjournal init local-llm \
  --model llama-3.1-8b-instruct-q4_k_m.gguf \
  --hf-repo TheBloke/Llama-3.1-8B-Instruct-GGUF \
  --hf-file llama-3.1-8b-instruct-q4_k_m.gguf \
  --hf-revision main
```

Related:
- `diagnose local llm` will suggest the command above if the model file is missing.

## Structure

Each command lives in its own `cli_*.py` module for clarity:
- `cli_journal_cli.py` – journaling CLI sub-app
- `cli_journal_web.py` – journaling web sub-app
- `cli_journal_desktop.py` – journaling desktop sub-app (PyWebView)
- `cli_session.py` – session utilities (also includes `session merge` and `session summaries`)
- `cli_summarise.py` – legacy summaries utilities (re-exported under `fix`)
- `cli_diagnose.py` – diagnostics sub-app (mic/local/cloud)
- `cli_reconcile.py`, `cli_init.py` – other commands (merge lives under `cli_session.py`)

## Examples

```bash
# Show version
uvx healthyselfjournal -- version

# Start CLI journaling
uvx healthyselfjournal -- journal cli --voice-mode

# Start web interface on a different port, resume latest session
uvx healthyselfjournal -- journal web --port 8888 --resume

# List sessions in a custom directory (first 200 chars)
uvx healthyselfjournal -- sessions list --sessions-dir ./sessions --nchars 200

# Summaries
uvx healthyselfjournal -- sessions summaries --missing-only
uvx healthyselfjournal -- fix backfill --limit 10
uvx healthyselfjournal -- fix regenerate 250918_0119.md

# Desktop diagnostics
uvx healthyselfjournal -- diagnose desktop
```

## See also

- `CLI_RECORDING_INTERFACE.md`
- `WEB_RECORDING_INTERFACE.md`
- `SESSIONS.md`
