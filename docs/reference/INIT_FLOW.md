## Init Flow (First‑Run Setup Wizard)

### Introduction

This document describes the initialization flow for non‑technical users. The `init` wizard collects API keys, lets users choose between Cloud and Privacy modes, configures the sessions directory, optionally runs a smoke test, and persists configuration so `healthyselfjournal journal cli` works out of the box.

### See also

- `../reference/CLI_COMMANDS.md` – how to run commands (`journal cli`, `journal web`, `reconcile`, `summarise/*`).
- `../reference/SETUP_DEV.md` – development/venv setup; context for env variables and uv workflow.
- `../reference/libraries/QUESTIONARY.md` – prompt library usage patterns and tips.
- `../../healthyselfjournal/cli.py` – Typer CLI including `init` and auto‑init in `journal cli`.
- `../../healthyselfjournal/__init__.py` – `.env`/`.env.local` autoloading at import time.
- `../../healthyselfjournal/config.py` – defaults and env‑driven configuration (e.g., `SESSIONS_DIR`, `STT_*`).
- `../../healthyselfjournal/transcription.py` – STT backends and selection logic.
- `../../healthyselfjournal/audio.py` – recording utilities used by the smoke test.
- `../planning/250917c_publish_to_pypi.md` – packaging goals and user install paths.

### Principles, key decisions

- Optimize for a successful first run with minimal friction; offer sensible defaults.
- Prefer Cloud mode initially for best accuracy/latency; Privacy mode is available but early‑stage.
- Never block recording due to STT/LLM errors; degrade gracefully and allow backfill.
- Persist user choices in simple `.env.local` so non‑technical users can edit later.
- Autoload env from `.env` then `.env.local`, without overriding explicitly set OS env.
- Keep the default sessions directory at `./sessions` (CWD) to match `uvx` usage.

### Current state

- `healthyselfjournal init` launches an interactive Questionary wizard.
- `healthyselfjournal journal cli` auto‑runs the wizard if critical prerequisites are missing.
- Configuration is written to `.env.local` in the current working directory and applied to the current process immediately.
- Optional smoke test records a 1‑second WAV and, in Cloud mode, attempts a tiny transcription call.

### Auto‑init detection

Auto‑init triggers at the start of `journal cli` when running in a TTY if either condition holds:

- `ANTHROPIC_API_KEY` is missing (required for the default Anthropic LLM), or
- STT backend resolves to `cloud-openai` and `OPENAI_API_KEY` is missing.

If stdin is not a TTY, the CLI aborts with instructions to run `healthyselfjournal init`.

Notes:
- `.env` and `.env.local` are autoloaded on package import; values set there satisfy detection.
- Users who prefer shell‑exported env vars are fully supported; autoload never overrides existing OS env.

### Wizard steps (Questionary)

1) Mode selection
   - Cloud (recommended): uses OpenAI for STT and Anthropic for questions.
   - Privacy mode (experimental): attempts on‑device STT (`auto-private`).

2) Keys
   - Anthropic API key (required for Cloud mode; optional if you intend to use a local `ollama:*` model): the wizard opens `https://console.anthropic.com/settings/keys` in your browser; after you create a key and copy it, press Enter. It will try to read your clipboard automatically; if not present, you can paste into a masked prompt. The key is validated via a lightweight models listing call.
   - OpenAI API key (required for Cloud mode; optional/unused for Privacy mode): similarly, the wizard opens `https://platform.openai.com/api-keys`, attempts clipboard capture, falls back to masked paste, and validates with a lightweight models listing call.

3) Sessions directory
   - Prompted with default `./sessions` (resolved from current working directory).
   - Directory is created if missing.

4) Optional smoke test
   - Records a ~1s WAV (microphone permission may be prompted by the OS).
   - In Cloud mode, performs a minimal transcription call and reports success (non‑fatal if it fails).

5) Persistence and confirmation
   - Writes/merges to `./.env.local` with: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` (if provided), `STT_BACKEND`, `STT_MODEL`, `STT_COMPUTE`, `STT_FORMATTING`, `SPEAK_LLM`, `SESSIONS_DIR`.
   - Updates `os.environ` in‑process so the current run can proceed immediately.
   - Prints a summary panel and next steps.

### Configuration and precedence

- Autoload order at import: `.env` then `.env.local`; existing OS env always wins.
- Runtime precedence: CLI flags > OS env > `.env.local` > `.env` > code defaults.
- Relevant env variables supported by `config.py`:
  - `SESSIONS_DIR` (or `RECORDINGS_DIR`) → default sessions path.
  - `LLM_MODEL` → LLM provider/model string (supports `provider:model:version[:thinking]`).
  - `STT_BACKEND`, `STT_MODEL`, `STT_COMPUTE`, `STT_FORMATTING` → STT selection.
  - `SPEAK_LLM`, `TTS_MODEL`, `TTS_VOICE`, `TTS_FORMAT` → voice output options.

### Cloud vs Privacy modes

- Cloud mode (default recommendation)
  - Pros: higher accuracy, simpler setup, lower latency.
  - Requires: `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`.

- Privacy mode (experimental)
  - Tries local backends via `auto-private` (prefers `mlx-whisper` on Apple Silicon, then `faster-whisper`, then `whispercpp`).
  - If no local backend is available, running `journal cli` with `auto-private` will raise a helpful error; users can switch to Cloud mode or install local extras.
  - Combine with `--llm-model ollama:...` (and a running Ollama daemon) to keep dialogue offline.

### Smoke test details

- Microphone test uses `sounddevice` and `soundfile` to capture and write ~1 second of audio to the selected sessions directory (temporary file).
- Cloud transcription test constructs an OpenAI transcription backend and attempts a minimal request; token use is minimal and failures are non‑fatal.

### Common flows

- First run via `uvx healthyselfjournal` → auto‑init launches → user selects Cloud → keys entered → sessions at `./sessions` → optional smoke test → run `journal cli`.
- Switching to Privacy mode later: edit `.env.local` (`STT_BACKEND=auto-private`) or re‑run `healthyselfjournal init` and choose Privacy.
- No keys yet: run `init`, skip Cloud, select Privacy; if no local backends present, users can still record audio with `journal cli`, then backfill later after enabling Cloud or installing local STT.

### Troubleshooting

- “Environment variable X is required”: run `healthyselfjournal init` or export the variable before running `journal cli`.
- macOS microphone permissions: grant access in System Settings → Privacy & Security → Microphone.
- `.env.local` not picked up: ensure you’re running commands from the directory containing `.env.local`, or export variables in your shell profile.
- Local STT unavailable in Privacy mode: install `mlx-whisper` (Apple Silicon), or `faster-whisper`, or switch to Cloud.

### Planned future work

- Add `doctor` command to run comprehensive diagnostics (keys, audio, ffmpeg, local backends).
- Add `demo` command for an offline, no‑keys quick tour.
- Package extras to simplify local STT installs (e.g., `[local-macos]`, `[local]`).
