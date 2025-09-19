# Web Recording Interface

## Introduction
The web interface brings the Healthy Self Journal experience into the browser while reusing the CLI pipeline for transcription, dialogue, and storage.

## Running

```bash
uvx healthyselfjournal -- journal web \
  [--sessions-dir PATH] \
  [--resume] \
  [--host HOST] \
  [--port PORT] \
  [--reload/--no-reload] \
  [--kill-existing] \
  [--open-browser/--no-open-browser] \
  [--voice-mode/--no-voice-mode] [--tts-model SPEC] [--tts-voice NAME] [--tts-format FORMAT]
```

- Defaults bind to `127.0.0.1:8765`. Open `http://127.0.0.1:8765` in a modern Chromium-based browser.
- `--sessions-dir` shares the same storage layout as the CLI; recordings appear under `./sessions/<session-id>/browser-*.webm`.
- `--resume` resumes the most recent existing session instead of starting a new one.
- `--reload` enables autoreload for static assets and server changes during development.
- `--open-browser` opens your default browser after the server becomes ready.
- `--kill-existing` attempts to free the chosen port by terminating existing listeners before starting.
- The web UI streams audio from the browser, uploads `webm/opus` clips, and reuses the same transcription/LLM pipeline as the CLI. When `--voice-mode` is enabled, the server synthesises the next question and the browser plays it.

## Architecture and flow

See `WEB_ARCHITECTURE.md` for a deeper architecture overview. Runtime flow mirrors the CLI: capture → upload → transcribe → append → schedule summary → next question. Keyboard shortcuts and short‑answer gating match the CLI.

## Prerequisites

- Python deps: `uv sync --active` (after updating `pyproject.toml`).
- Front-end tooling: `npm install` then `npm run build` after modifying `static/ts` sources.

## Testing

- `tests/test_web_app.py` covers upload flow and TTS endpoint behaviour with stubs.

## See also

- `CLI_RECORDING_INTERFACE.md` – Terminal-based recording
- `CLI_COMMANDS.md` – Command index and quick reference
- `FILE_FORMATS_ORGANISATION.md` – Session layout and artefacts

