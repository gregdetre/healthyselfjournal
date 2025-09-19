# Command Line Interface

## Overview

Auto-start voice recording interface with visual feedback and keyboard controls.

### Launching

The journaling loop is started from the project root:

```bash
uvx healthyselfjournal -- journal \
  [--sessions-dir PATH] \
  [--llm-model SPEC] \
  [--stt-backend BACKEND] [--stt-model MODEL] [--stt-compute COMPUTE] [--stt-formatting MODE] \
  [--opening-question TEXT] [--language LANG] [--resume] \
  [--delete-wav-when-safe/--keep-wav] \
  [--stream-llm/--no-stream-llm] \
  [--voice-mode/--no-voice-mode] [--tts-model SPEC] [--tts-voice NAME] [--tts-format FORMAT] \
  [--llm-questions-debug/--no-llm-questions-debug] \
  [--mic-check/--no-mic-check]

# List existing sessions with summary snippets
uvx healthyselfjournal -- journal list [--sessions-dir PATH] [--nchars N]
```

Files default to `./sessions/`; pass `--sessions-dir` to override for archival or testing.

Tip: during a session, you can say "give me a question" to instantly get a question selected from built‑in examples embedded in the prompt.
#### Listing sessions

Show each session by `.md` filename stem with a summary snippet from frontmatter. Use `--nchars` to limit characters (full summary when omitted):

```bash
uv run healthyselfjournal journal list --sessions-dir ./sessions --nchars 200
```

### Web interface

Serve the browser-based recording experience with FastHTML:

```bash
uvx healthyselfjournal -- web \
  [--sessions-dir PATH] \
  [--resume] \
  [--host HOST] \
  [--port PORT] \
  [--reload/--no-reload] \
  [--kill-existing] \
  [--open-browser/--no-open-browser] \
  [--voice-mode/--no-voice-mode] [--tts-model SPEC] [--tts-voice NAME] [--tts-format FORMAT]
```

- Defaults bind to `127.0.0.1:8765`. Open <http://127.0.0.1:8765> in a modern Chromium-based browser.
- `--sessions-dir` shares the same storage layout as the CLI; recordings appear under `./sessions/<session-id>/browser-*.webm`.
- `--resume` resumes the most recent existing session instead of starting a new one (mirrors CLI behaviour).
- `--reload` enables autoreload for static assets and server changes during development.
- `--open-browser` is enabled by default; it opens your default browser after the server becomes ready. Disable with `--no-open-browser`.
- `--kill-existing` attempts to free the chosen port by terminating existing listeners before starting (best-effort; uses `lsof`/`fuser` when available).
- The web UI streams audio from the browser, uploads `webm/opus` clips, and reuses the same transcription/LLM pipeline as the CLI. When `--voice-mode` is enabled, the server synthesises the next question and the browser plays it.
- Architecture and troubleshooting details live in `WEB_INTERFACE.md`.


LLM selection:

- `--llm-model` accepts `provider:model[:version][:thinking]`.
  - Cloud default: `anthropic:claude-opus-4-1:20250805:thinking` (the `:thinking` suffix is optional and available on Anthropic models only).
  - Private/local: `ollama:gemma3:27b-instruct-q4_K_M` (requires the Ollama daemon running on your machine; set `OLLAMA_BASE_URL` to override the host if needed).

Getting started:

- First-time users should run the setup wizard:
  ```bash
  uvx healthyselfjournal -- init
  ```
- See `INIT_FLOW.md` for the init wizard flow and configuration details.

#### Speech-to-text options

- `--stt-backend`: choose between `cloud-openai`, `local-mlx`, `local-faster`, `local-whispercpp`, or `auto-private` (local-first probe).
- `--stt-model`: preset (`default`, `accuracy`, `fast`) or explicit model id/path.
- `--stt-compute`: optional precision override for local backends (e.g. `int8_float16`). Ignored when unsupported.
- `--stt-formatting`: `sentences` (default heuristic splitter) or `raw` (unaltered backend output).

Other useful options:

- `--opening-question`: override the initial question shown at session start.
- `--language`: primary language for transcription and LLM guidance (default `en`).
- `--resume`: resume the most recent session in the sessions directory.
- `--delete-wav-when-safe/--keep-wav`: delete WAV files once MP3 and STT JSON exist (default delete).
- `--llm-questions-debug/--no-llm-questions-debug`: append a brief techniques postscript to LLM questions (off by default).

#### Mic check

- `--mic-check/--no-mic-check`: disabled by default. When enabled (including when `--resume` is used), records a fixed 3 second sample so you can verify your mic level and transcription quality. The temporary recording is transcribed and shown, then discarded. Press ENTER to continue, ESC to try again, or `q` to quit.

Environment variables:

- `journal`: supply `ANTHROPIC_API_KEY` when using `anthropic:*` models for dialogue/summaries. Switching to `ollama:*` models keeps the loop fully local—no cloud keys required—just ensure the Ollama service is running (override the host via `OLLAMA_BASE_URL` if it is not on the default `http://localhost:11434`). `OPENAI_API_KEY` remains necessary whenever `--stt-backend cloud-openai` is selected or `--voice-mode` enables OpenAI TTS.
- `reconcile`: requires `OPENAI_API_KEY` only when `--stt-backend cloud-openai` is selected. Local backends do not need API keys.

### Reconcile recordings

Backfill missing transcriptions for saved WAV files in `--sessions-dir`.

```bash
uvx healthyselfjournal -- reconcile \
  [--sessions-dir PATH] \
  [--stt-backend BACKEND] [--stt-model MODEL] [--stt-compute COMPUTE] \
  [--language LANG] [--limit N]
```

Notes:
- Uses the same STT backends as `journal`. Provide `OPENAI_API_KEY` only when using `--stt-backend cloud-openai`.
- Honors `--delete-wav-when-safe`: WAVs are deleted after MP3 + STT exist when enabled.

### Summaries Utilities

Minimal commands for working with summaries stored in session frontmatter:

```bash
# List (default shows only missing)
uvx healthyselfjournal -- summaries list [--sessions-dir PATH] [--missing-only/--all]

# Backfill (default only missing; use --all to regenerate all)
uvx healthyselfjournal -- summaries backfill [--sessions-dir PATH] [--llm-model SPEC] [--missing-only/--all] [--limit N]

# Regenerate a single file's summary
uvx healthyselfjournal -- summaries regenerate [--sessions-dir PATH] [--llm-model SPEC] yyMMdd_HHmm[.md]
```

- `--missing-only/--all` defaults to missing-only for both commands.
- Summary generation honours the same `--llm-model` provider syntax. Provide `ANTHROPIC_API_KEY` only when targeting `anthropic:*` models; local `ollama:*` runs stay offline (ensure the Ollama daemon is available).

### Merge sessions

Merge two sessions, keeping the earlier one. Moves assets, appends later Q&A to earlier, updates frontmatter, and regenerates the summary by default.

```bash
uvx healthyselfjournal -- merge [--sessions-dir PATH] [--llm-model SPEC] [--regenerate/--no-regenerate] [--dry-run] [--ignore-missing] yyMMdd_HHmm[.md] yyMMdd_HHmm[.md]
```

Notes:
- Asset filename collisions are avoided by suffixing with `_N` when needed.
- The later session folder is removed if empty after moving.
- Summary regeneration requires `ANTHROPIC_API_KEY` only when the chosen `--llm-model` uses the Anthropic provider. Local `ollama:*` options avoid cloud calls.
 - After a successful merge, the later `.md` file is deleted.
 - Frontmatter `audio_file` becomes MP3-centric (e.g., `{wav: null, mp3: <file>, duration_seconds: <float>}`).
 - If the later assets folder is missing, run again with `--ignore-missing` to proceed.

See also (details and rationale):
- `CONVERSATION_SUMMARIES.md` – Why summaries exist, how they’re generated, and safety considerations.
- `FILE_FORMATS_ORGANISATION.md` – Where summaries live in frontmatter and related fields.
- `LLM_PROMPT_TEMPLATES.md` – Prompt template used for summary generation.

## See also

- `ARCHITECTURE.md` – CLI interface layer and audio pipeline architecture
- `RECORDING_CONTROLS.md` – Detailed key mappings and recording flow for capture.
- `PRODUCT_VISION_FEATURES.md` – How the CLI supports the broader product vision.
- `PRIVACY.md` – Privacy modes, offline configuration, and what leaves your machine.
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
uvx healthyselfjournal -- journal --voice-mode

# Explicit control
uvx healthyselfjournal -- journal --voice-mode --tts-voice shimmer --tts-model gpt-4o-mini-tts --tts-format wav
```

Notes:
- macOS uses `afplay` for local playback. If unavailable, `ffplay` is attempted.
- Only assistant questions are spoken; summaries and status messages remain text-only.
 - While a question is being spoken, press ENTER to skip the voice playback immediately.
