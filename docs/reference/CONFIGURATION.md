# Configuration

## Introduction

This document explains how configuration is loaded, what can be customized via environment variables and CLI flags, and how to define a user-specific vocabulary for more accurate transcriptions.

## See also

- `ARCHITECTURE.md` – Configuration system hierarchy and dependency management
- `DOCUMENTATION_ORGANISATION.md` – where to find related docs
- `CLI_COMMANDS.md` – all CLI flags and commands
- `INIT_FLOW.md` – first-run setup wizard and persistence
- `AUDIO_VOICE_RECOGNITION_WHISPER.md` – STT backends and tuning
- `WEB_RECORDING_INTERFACE.md` – server options that mirror CLI
- `../../healthyselfjournal/config.py` – config defaults and loader
- `../../healthyselfjournal/transcription.py` – STT backends and vocabulary integration

## Configuration sources and precedence

At import time, the package loads environment variables from `.env` then `.env.local` and also from XDG (`~/.config/healthyselfjournal/.env` and `.env.local`) for desktop users, without overriding existing OS environment variables. The CLI init wizard writes to the current working directory by default; pass `--xdg` to persist under XDG.

Runtime precedence (highest to lowest):

1. CLI flags (e.g., `--stt-backend`, `--language`)
2. OS environment variables
3. Desktop settings (XDG `settings.toml`) when CLI flags are left at defaults
4. `.env.local` (including XDG variant for desktop)
5. `.env`
6. Code defaults (`healthyselfjournal/config.py`)

- Notes:
- Desktop app (`healthyselfjournal journal desktop`) applies a small set of user preferences from XDG `settings.toml` (sessions folder, resume, voice) when the corresponding CLI options are left at their defaults. Explicit CLI flags and OS env still take precedence.
- `.env.local` can exist in the project, CWD, or XDG config directory for desktop users; OS env variables always win over file-based values.

Relevant environment variables include:

- Sessions and paths
  - `SESSIONS_DIR` or `RECORDINGS_DIR` – default sessions directory
  - `WEB_UPLOAD_MAX_BYTES` – maximum accepted upload size (bytes) for the web interface
- STT (speech-to-text)
  - `STT_BACKEND` – `cloud-openai`, `local-mlx`, `local-faster`, `local-whispercpp`, or `auto-private`
  - `STT_MODEL` – preset or explicit model id/path
  - `STT_COMPUTE` – precision for local backends (e.g., `int8_float16`)
  - `STT_FORMATTING` – `sentences` (default) or `raw`
- LLM
  - `LLM_MODEL` – provider:model[:version][:thinking]
- Optional TTS
  - `SPEAK_LLM`, `TTS_MODEL`, `TTS_VOICE`, `TTS_FORMAT`

### Desktop settings (XDG)

File: `~/.config/healthyselfjournal/settings.toml`

Keys:
- `sessions_dir` – override default sessions directory for the desktop app
- `resume_on_launch` – `true`/`false`
- `voice_enabled` – `true`/`false`
- `mode` – `cloud` or `private` (influences STT defaults and Setup UI)

These settings are applied when launching the desktop app if the corresponding CLI flags are left at defaults. Change them via Preferences in the desktop UI.

## User-specific vocabulary (vocabulary-only)

Define a short list of names/terms that frequently occur in your journaling (e.g., people, products, places). This improves accuracy by providing a concise “initial prompt” to STT backends that support it.

### File: user_config.toml

Search order (first found wins):

1. `HSJ_USER_CONFIG` environment variable (absolute path)
2. Project root `user_config.toml`
3. Current working directory `user_config.toml`
4. XDG path: `~/.config/healthyselfjournal/user_config.toml`

This file is ignored by git by default.

Example:

```toml
[vocabulary]
terms = [
  "StartupName",
  "Partner Name",
  "Product X",
]
```

Notes:

- Keep the list short and focused; very long prompts may be truncated or ignored.
- No correction mappings are applied; this feature is vocabulary-only by design.

### How it’s used

- OpenAI STT: sends a short `prompt` constructed from `terms`.
- faster‑whisper: passes `initial_prompt`.
- whisper.cpp: attempts `initial_prompt` when supported; otherwise ignored.

## Examples

CLI with overrides:

```bash
uvx healthyselfjournal -- journal \
  --stt-backend cloud-openai \
  --stt-formatting sentences \
  --language en
```

Environment in `.env.local`:

```env
STT_BACKEND=cloud-openai
STT_MODEL=default
STT_FORMATTING=sentences
SESSIONS_DIR=./sessions
```

## Troubleshooting

- Vocabulary doesn’t seem to apply
  - Ensure `user_config.toml` is discoverable and valid TOML
  - Keep `terms` concise; then try the mic check (`--mic-check`) to preview output
- Local backends ignore prompts
  - Some binaries don’t support hints; in that case the list is safely ignored

## Maintenance

- Update this doc when adding new configuration keys or locations
- Cross-reference new options from `COMMAND_LINE_INTERFACE.md` and `INIT_FLOW.md`
