# CLI Recording Interface

## Overview

Voice-first recording from your terminal with visual feedback and keyboard controls.

## Launching

Run from the project root (or installed environment). Choose a mode explicitly:

```bash
uvx healthyselfjournal -- journal cli cloud \
  [--sessions-dir PATH] \
  [--llm-model SPEC] \
  [--stt-backend BACKEND] [--stt-model MODEL] [--stt-compute COMPUTE] [--stt-formatting MODE] \
  [--opening-question TEXT] [--language LANG] [--resume] \
  [--delete-wav-when-safe/--keep-wav] \
  [--stream-llm/--no-stream-llm] \
  [--voice-mode/--no-voice-mode] [--tts-model SPEC] [--tts-voice NAME] [--tts-format FORMAT] \
  [--llm-questions-debug/--no-llm-questions-debug] \
  [--mic-check/--no-mic-check]
```

Privacy mode variant:
```bash
uvx healthyselfjournal -- journal cli private \
  [--sessions-dir PATH] \
  [--llm-model SPEC] \
  [--stt-backend BACKEND] [--stt-model MODEL] [--stt-compute COMPUTE] [--stt-formatting MODE] \
  [--opening-question TEXT] [--language LANG] [--resume] \
  [--delete-wav-when-safe/--keep-wav] \
  [--stream-llm/--no-stream-llm] \
  [--voice-mode/--no-voice-mode] [--tts-model SPEC] [--tts-voice NAME] [--tts-format FORMAT] \
  [--llm-questions-debug/--no-llm-questions-debug] \
  [--mic-check/--no-mic-check]
```

Files default to `./sessions/`; pass `--sessions-dir` to override for archival or testing.

Tip: during a session, you can say "give me a question" to instantly get a question selected from built‑in examples embedded in the prompt.

### Speech options

- `--voice-mode/--no-voice-mode`: convenience switch that enables speech with default settings.
- `--tts-model`: TTS model (default `gpt-4o-mini-tts`).
- `--tts-voice`: TTS voice (default `shimmer`).
- `--tts-format`: audio format for playback (default `wav`).

Examples:
```bash
# One-flag voice mode with defaults (shimmer, gpt-4o-mini-tts, wav)
uvx healthyselfjournal -- journal cli --voice-mode

# Explicit control
uvx healthyselfjournal -- journal cli --voice-mode --tts-voice shimmer --tts-model gpt-4o-mini-tts --tts-format wav
```

Notes:
- macOS uses `afplay` for local playback. If unavailable, `ffplay` is attempted.
- Only assistant questions are spoken; summaries and status messages remain text-only.
- While a question is being spoken, press ENTER to skip the voice playback immediately.

### Mic check

- `--mic-check/--no-mic-check`: disabled by default. When enabled (including when `--resume` is used), records a fixed 3 second sample so you can verify your mic level and transcription quality. The temporary recording is transcribed and shown, then discarded. Press ENTER to continue, ESC to try again, or `q` to quit.

### STT and LLM options

- `--stt-backend`: choose between `cloud-openai`, `local-mlx`, `local-faster`, `local-whispercpp`, or `auto-private`.
- `--stt-model`: preset (`default`, `accuracy`, `fast`) or explicit model id/path.
- `--stt-compute`: optional precision override for local backends (e.g. `int8_float16`).
- `--stt-formatting`: `sentences` (default) or `raw`.
- `--llm-model` accepts `provider:model[:version][:thinking]`.

Environment variables:

- Provide `ANTHROPIC_API_KEY` when using `anthropic:*` models for dialogue/summaries. Switching to `ollama:*` models keeps the loop fully local—no cloud keys required—just ensure the Ollama service is running (override host via `OLLAMA_BASE_URL`).
- `OPENAI_API_KEY` is necessary whenever `--stt-backend cloud-openai` is selected or `--voice-mode` enables OpenAI TTS.

## Recording Flow and Controls

See `RECORDING_CONTROLS.md` for the full flow, keyboard shortcuts, and short‑answer gating rules. The CLI and web experiences share the same flow and thresholds.

## See also

- `CLI_COMMANDS.md` – Command index and quick reference
- `WEB_RECORDING_INTERFACE.md` – Run the browser-based interface (`journal web`)
- `CONFIGURATION.md` – Precedence and environment variables
- `FILE_FORMATS_ORGANISATION.md` – What gets saved when
- `PRIVACY.md` – Privacy modes and what leaves your machine

