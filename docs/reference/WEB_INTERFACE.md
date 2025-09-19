# Web Interface

## Introduction
The web interface brings the Healthy Self Journal experience into the browser while reusing the CLI pipeline for transcription, dialogue, and storage. This document covers the architecture, runtime flow, and operational guidance for the FastHTML-powered implementation.

## See also
- `docs/planning/250918b_web_interface_planning.md` – Original implementation stages and acceptance criteria that guided this build.
- `docs/reference/COMMAND_LINE_INTERFACE.md` – CLI usage reference, including the web entry command and shared options.
- `docs/reference/FILE_FORMATS_ORGANISATION.md` – Canonical description of session artefacts (`browser-XXX.webm`, frontmatter updates, `.stt.json`).
- `docs/reference/BACKGROUND_PROCESSING.md` – How transcript summaries are scheduled; the web upload path uses the same executor model.
- `docs/reference/SETUP.md` – Environment and asset build steps (`npm run build`) required before running the server.
- `docs/reference/libraries/FASTHTML.md` – Notes and caveats for the framework used to serve the pages.
- `docs/reference/AUDIO_SPEECH_GENERATION.md` – Design and configuration for speaking assistant questions out loud.

## Principles and key decisions
- **Single pipeline**: Browsers record locally, upload once, and the server reuses existing transcription/LLM/session code—no divergent storage formats.
- **FastHTML first**: HTML is rendered server-side via FastHTML/Starlette; JavaScript is limited to recording and minimal UI state.
- **Opus uploads**: Client audio remains `audio/webm;codecs=opus`; if a backend cannot accept Opus, the error surface is explicit rather than silently transcoding.
- **Local-first**: Server binds to `127.0.0.1:8765` by default and writes into the same `--sessions-dir` tree as the CLI, enabling hybrid usage.
- **UX parity**: The TypeScript recorder mirrors CLI behaviours—level meter, SPACE pause/resume, ESC cancel, short/quiet auto-discard, sequential questioning.
- **Optional voice mode**: When enabled, the server synthesises the next question using the existing TTS machinery and the browser plays it back.
- **Incremental enhancements**: Advanced feedback (SSE progress, device pickers) are deferred to keep V1 lightweight and maintainable.

## Architecture overview

### Server layer (Python)
- Entry point: `healthyselfjournal/cli_web.py` exposes `uv run healthyselfjournal web` with `--sessions-dir`, `--host`, `--port`, `--reload` and optional voice/TTS options.
- Entry point: `healthyselfjournal/cli_web.py` exposes `uv run healthyselfjournal web` with `--sessions-dir`, `--resume`, `--host`, `--port`, `--reload` and optional voice/TTS options.
- Application builder: `healthyselfjournal/web/app.py` constructs a FastHTML app on demand, mounts `/static/`, and maintains per-session state (`WebSessionState`).
- Compatibility shim: `_get_fast_html_class()` patches `fastcore.xml.ft` when necessary so FastHTML initialises correctly with current `fastcore` releases. With FastHTML 0.12+, routes should return plain strings/objects and FastHTML wraps responses; avoid manually returning `HTMLResponse` from handlers.
- Routes:
- `GET /` – Starts or resumes a session then redirects to the pretty URL `GET /journal/{sessions_dir_basename}/{session_id}/`.
  - `GET /journal/{sessions_dir}/{session_id}/` – Renders the main recording page (UI shell) for that session.
  - `POST /session/{id}/upload` – Accepts `FormData` (`audio`, `mime`, `duration_ms`, `voiced_ms`, `question`), persists the blob as `browser-XXX.webm`, and funnels processing through `SessionManager.process_uploaded_exchange()`.
  - `POST /session/{id}/tts` – When voice mode is enabled, synthesises speech for the provided text (`{"text": "..."}`) and returns `audio/*` bytes.
- Processing steps (reuse existing modules):
  1. Persist upload to session audio directory.
  2. Transcribe via configured backend (`transcription.create_transcription_backend`).
  3. Append transcript/metadata using `storage.append_exchange_body` and `_update_frontmatter_after_exchange`.
  4. Log events (`web.upload.received`, `web.upload.processed`).
  5. Schedule summary regeneration on the shared background executor.
  6. Generate the next question via `llm.generate_followup_question`.
- Error responses return JSON with `status="error"` and reason keys (`unknown_session`, `processing_failed`, etc.) to keep the client-side handler simple. Returning plain dicts is supported in 0.12 (FastHTML emits JSON); for custom headers/streaming we still use Starlette `Response`.

### Client layer (TypeScript + CSS)
- Source: `healthyselfjournal/static/ts/app.ts` compiled to `static/js/app.js` via `npm run build`; styling lives in `static/css/app.css`.
- Responsibilities:
  - Request microphone permission, manage a single `MediaRecorder`, and keep recordings running across tab focus changes.
  - Render a lightweight RMS bar using an `AnalyserNode`; update CSS custom property `--meter-level`.
  - Mirror CLI affordances: click-to-start/stop, SPACE pause/resume, ESC cancel, auto-discard segments shorter than `CONFIG.short_answer_duration_seconds` or with low voiced time.
  - Submit uploads with metadata; display spinner text while awaiting the server.
  - Render incremental history (question/answer pairs) and update the displayed next question without a full page reload.
  - If voice mode is active, request TTS audio for the next question and play it via an `HTMLAudioElement`.
- Build/watch: `npm run build` performs a one-off compile; `npm run watch` keeps the TS compiler live during development.

### Storage and session layout
- Sessions are created in the configured `--sessions-dir`; filenames follow the CLI convention (`yyMMdd_HHmm.md` plus `yyMMdd_HHmm/`).
- Browser clips are stored as `browser-XXX.webm`; each has a companion `browser-XXX.stt.json` with raw transcription payload.
- Frontmatter updates mirror CLI fields: `audio_file` entries gain `{wav: null, mp3: null, duration_seconds: <float>}` for web segments, and total duration is recomputed after every exchange.
- Historic questions, transcripts, and summaries remain in the markdown body/frontmatter; no web-specific divergence.

## Runtime flow
1. User runs `uv run healthyselfjournal web [options]` (see next section). If `--resume` is provided, the server resumes the most recent session instead of starting a new one.
2. The root page loads and initialises or resumes a session, then redirects to `/journal/{sessions_dir_basename}/{session_id}/` which shows the opening question.
3. On “Start recording”, the client captures audio, visualises levels, and tracks voiced time.
4. When recording stops, the clip is discarded if it fails the short-answer thresholds; otherwise it uploads to `/session/{id}/upload`.
5. The server transcribes, updates storage, schedules summarisation, and generates the next question.
6. The client receives the response, renders transcript+metadata in the history list, and updates the question prompt. If voice mode is enabled, the client fetches `POST /session/{id}/tts` for the new question and plays the audio.
7. After the question is displayed (and any TTS playback has finished), recording auto-starts to mirror the CLI flow.
8. Steps 3–7 repeat until the user closes the tab or the session completes via CLI/web.

## Keyboard shortcuts
- **ENTER**: Start/stop recording (same as clicking the button)
- **SPACE**: Pause/resume recording (paused audio is not saved; meter hidden while paused)
- **ESC**: Cancel the current take (discard; nothing is saved or uploaded)
  - When voice mode is on, ENTER also skips/halts question playback and starts recording immediately.

## Debug/config quick reference
<details>
<summary>Debug/config and defaults (click to expand)</summary>

- **Server CLI defaults**
  - **host**: `127.0.0.1`
  - **port**: `8765`
  - **reload**: `false`
  - **sessions-dir**: `CONFIG.recordings_dir` (see resolution below)
  - **kill-existing**: `false` (optional; frees the port before starting)

- **Sessions directory resolution**
  - **SESSIONS_DIR** or **RECORDINGS_DIR** env var if set
  - Otherwise: `<project>/sessions` (current working directory at launch)

- **Short-answer gating (client + server)**
  - **CONFIG.short_answer_duration_seconds**: `1.2` seconds
  - **CONFIG.short_answer_voiced_seconds**: `0.6` seconds
  - **CONFIG.voice_rms_dbfs_threshold**: `-40.0` dBFS

- **Transcription (STT)**
  - **STT_BACKEND**: default `cloud-openai` (also supports `local-mlx`, `local-faster`, `local-whispercpp`, `auto-private`)
  - **STT_MODEL**: default `default` (mapped per-backend, e.g. `gpt-4o-transcribe` for cloud)
  - **STT_COMPUTE**: default `auto` (used by local backends; e.g. `int8_float16` for faster-whisper)
  - **STT_FORMATTING**: default `sentences` (`sentences` or `raw`)

- **LLM (questions/summaries)**
  - **LLM_MODEL**: default `anthropic:claude-opus-4-1:20250805:thinking`
  - Provider inference: `anthropic` or `ollama` based on the prefix before the first `:`
  - Temperatures/tokens via `CONFIG`: question `0.5`, summary `0.4`, max tokens ~ `1200` each

- **Local LLM (Ollama)**
  - **OLLAMA_BASE_URL**: default `http://localhost:11434`
  - **OLLAMA_TIMEOUT_SECONDS**: default `30.0`
  - **OLLAMA_NUM_CTX**: default `8192`

- **Optional TTS of questions**
  - **SPEAK_LLM**: default `false`
  - **TTS_MODEL**: default `gpt-4o-mini-tts`
  - **TTS_VOICE**: default `shimmer`
  - **TTS_FORMAT**: default `wav`

- **Static assets**
  - Served from `/static/` (package path `healthyselfjournal/static`)
  - Ensure `static/js/app.js` is built from `static/ts/app.ts` (`npm run build`)

- **HTTP endpoints**
  - `GET /` → starts or resumes a session, then 307-redirects to the pretty URL
  - `GET /journal/{sessions_dir}/{session_id}/` → serves the HTML shell for a specific session
  - `POST /session/{id}/upload` → accepts `FormData` fields: `audio`, `mime`, `duration_ms`, `voiced_ms`, `question`

- **Logs and artefacts**
  - Event log: `sessions/events.log`
  - Uploaded clips: `browser-XXX.webm` (+ `browser-XXX.stt.json`)

</details>

## Running the web server
- **Prerequisites**:
  - Python deps: `uv sync --active` (after updating `pyproject.toml`).
  - Front-end tooling: `npm install` (installs TypeScript) and `npm run build` after modifying `static/ts` sources.
- **Command**:
  ```bash
  uv run healthyselfjournal web \
    [--sessions-dir PATH] \
    [--resume] \
    [--host 127.0.0.1] \
    [--port 8765] \
    [--reload/--no-reload] \
    [--kill-existing] \
    [--open-browser/--no-open-browser] \
    [--voice-mode/--no-voice-mode] \
    [--tts-model MODEL] \
    [--tts-voice VOICE] \
    [--tts-format FORMAT]
  ```
- **Access**: open `http://127.0.0.1:8765` in a Chromium-based browser (Safari works on recent versions but lacks explicit optimiser testing). The CLI opens your default browser automatically once the server is ready; disable with `--no-open-browser`.
- **Shared storage**: running CLI and web simultaneously is supported; both operate on timestamped filenames to avoid collisions.
- **Static assets**: `/static/css/app.css` and `/static/js/app.js` are served directly from the package; ensure the JS build artefact is up to date before packaging.
- **FastHTML 0.12 return types**: Return strings/objects or element trees from route handlers; avoid explicitly constructing `HTMLResponse`/`JSONResponse` for typical cases to prevent double-wrapping errors.
- **Voice mode**: When `--voice-mode` is enabled, the server synthesises the next question on demand and the browser plays it. Requires `OPENAI_API_KEY` for the default OpenAI TTS backend; see `AUDIO_SPEECH_GENERATION.md`.

## Testing
- Unit/integration coverage: `tests/test_web_app.py` uses `starlette.testclient` with stubbed transcription/LLM backends to exercise upload->persist->response flow. TTS is tested via a stubbed synthesiser.
- Run with `PYTHONPATH=. pytest tests/test_web_app.py` (or include `tests/test_storage.py` for regression coverage).
- Tests skip automatically if `starlette` is missing; a `uv sync` after adding dependencies is sufficient.

## Troubleshooting
- **MediaRecorder unsupported**: Firefox/Safari variants lacking Opus support will fail early; inform users to switch browsers or add future polyfills.
- **Permission errors**: If microphone access is denied, the UI surfaces a blocking status message; resolve by resetting browser permissions.
- **Upload failures**: JSON errors include a `detail` field; transcription or LLM failures bubble up so issues (API keys, backend availability) mirror CLI troubleshooting steps.
- **TypeScript build issues**: Ensure `npm install` has run; the `tsconfig.json` targets ES2020 modules using Node resolution, so stale node_modules or incompatible tooling will surface compiler errors.
- **Voice playback issues**: Ensure `OPENAI_API_KEY` is set when `--voice-mode` is enabled (default backend is OpenAI). Some browsers restrict autoplay; first playback should occur after a user gesture (record/stop) to satisfy policies. If audio is silent, check `sessions/events.log` for `tts.error` entries.
