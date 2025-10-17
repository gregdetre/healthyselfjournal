# User Setup & Quickstart

## Introduction
This guide helps end users install and run Healthy Self Journal with minimal effort. It focuses on PyPI installation via `uvx` or `pip`, required keys for cloud features, optional local/offline setup, and the most useful commands.

## See also
- `CLI_COMMANDS.md` – CLI reference and flags
- `PRIVACY.md` – What data is stored, when network calls occur, and how to stay offline
- `AUDIO_VOICE_RECOGNITION_WHISPER.md` – STT backends and requirements
- `OLLAMA_GEMMA_DEPLOYMENT_GUIDE.md` – Local LLM for fully offline text generation
- `DESKTOP_APP_PYWEBVIEW.md` – Desktop Setup/Preferences, endpoints, and settings persistence
- `CONFIGURATION.md` – Runtime configuration, env vars, and precedence rules

## Prerequisites
- Python 3.10+
- Optional: `ffmpeg` on PATH (for background MP3 conversion)
- Audio notes:
  - Windows: usually works out of the box. If mic access is blocked, enable Microphone permission for your terminal.
  - macOS: grant Microphone permission to your terminal (System Settings → Privacy & Security → Microphone).
  - Linux: ensure PortAudio/libsndfile are available (e.g., `sudo apt install portaudio19-dev libsndfile1`).

## Install

Option A: run without installing, using uvx (recommended)
```bash
uvx healthyselfjournal -- --help
```

Option B: install with pip
```bash
pip install healthyselfjournal
healthyselfjournal --help
```

## First run: init wizard
Run the interactive first‑run setup to collect keys and preferences:
```bash
uvx healthyselfjournal -- init
```
The wizard helps you choose between Cloud (recommended) and Privacy (local/offline) modes, gathers API keys when needed, sets your sessions directory, and writes `.env.local` so future runs work without extra flags. By default it writes to your current directory; pass `--xdg` to save under `~/.config/healthyselfjournal/.env.local`.
On desktop, a built‑in Setup wizard appears on first launch and saves keys under `~/.config/healthyselfjournal/.env.local` and preferences under `~/.config/healthyselfjournal/settings.toml`.

## Keys and modes
- Cloud (default): highest accuracy and responsiveness
  - Requires: `ANTHROPIC_API_KEY` (LLM) and `OPENAI_API_KEY` (STT)
- Privacy (local/offline): avoids sending data to cloud providers
  - Choose local STT via `--stt-backend` (see below) and an `ollama:*` `--llm-model`

## Daily usage
Start a journaling session:
```bash
uvx healthyselfjournal -- journal cli
```

Handy flags:
- `--resume` – continue the most recent session
- `--sessions-dir PATH` – keep sessions in a different folder (e.g., encrypted location)

Recording behavior:
- Recording starts immediately
- Press any key to stop
- `ESC` cancels the current take (discarded)
- `Q` saves the take, transcribes it, then ends the session

## Staying offline
To avoid cloud calls entirely:
1) Use a local STT backend
```bash
healthyselfjournal journal cli --stt-backend local-mlx        # Apple Silicon
healthyselfjournal journal cli --stt-backend local-faster     # Portable CPU/GPU
healthyselfjournal journal cli --stt-backend local-whispercpp --stt-model /path/to/model.gguf
```
2) Use a local LLM via Ollama (daemon must be running)
```bash
healthyselfjournal journal cli --llm-model ollama:gemma3:27b-instruct-q4_K_M
```
3) Ensure no cloud keys are set in your environment

See `PRIVACY.md` for details about what leaves your machine and how to control it.

## Troubleshooting
- “Environment variable X is required”
  - Run `healthyselfjournal init` or export the variable before running `journal cli`.
- Microphone permission errors
  - Grant access in your OS privacy settings and try again.
- Local STT not available
  - Install the corresponding package and retry; see `AUDIO_VOICE_RECOGNITION_WHISPER.md`.

## Where your data lives
- Default sessions directory is `./sessions` under your current folder
- Each response is saved immediately as `.wav` (and `.mp3` when `ffmpeg` is available) and appended to a session `.md` file with YAML frontmatter that stores summaries and metadata

## Next steps
- Explore CLI flags in `CLI_COMMANDS.md`
- Read `PRIVACY.md` to understand cloud vs local tradeoffs
- For a browser UI, see `WEB_RECORDING_INTERFACE.md`

## Desktop app (optional)

Launch the PyWebView desktop app:
```bash
uvx healthyselfjournal -- desktop --resume --voice-mode
```
- Preferences lets you choose the Sessions folder, toggle resuming the last session, and turn Voice mode on/off.
- Desktop settings are saved to `~/.config/healthyselfjournal/settings.toml` and override defaults; runtime precedence is CLI flags > OS env > Desktop settings > project `.env.local` > code defaults.
- The first desktop run shows a Setup wizard to collect keys and the sessions folder; you can re‑run it from Preferences.

