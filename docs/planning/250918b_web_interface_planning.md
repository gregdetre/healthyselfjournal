# Web Interface V1 Planning (FastHTML + minimal TypeScript)

## Goal, context

Create a web interface for the voice-first reflective journaling app that closely mirrors the CLI experience while reusing the same core Python machinery and storage. The web app will use FastHTML for server-side rendering and a small TypeScript module for in-browser audio recording via the MediaRecorder/Web Audio APIs. Audio is recorded client-side, uploaded to the server, and processed by the existing transcription and dialogue pipeline. Both interfaces share the same `./sessions/` directory and configuration.

Motivations:
- Reduce friction by enabling voice journaling in the browser while preserving privacy-first, local-first options
- Reuse existing STT/LLM/session machinery and frontmatter format to keep a single source of truth
- Keep architecture simple enough to later bundle as Electron/Tauri/PyWebView without major refactors


## References

- `docs/conversations/250918c_web_interface_architecture_decisions.md` – Agreed decisions for the web approach
- `docs/reference/PRODUCT_VISION_FEATURES.md` – Overall product vision and core features
- `docs/reference/COMMAND_LINE_INTERFACE.md` – CLI behavior that the web should mirror
- `docs/reference/FILE_FORMATS_ORGANISATION.md` – Session file layout and frontmatter schema
- `docs/reference/DIALOGUE_FLOW.md` – Conversation loop and prompt structure
- `docs/reference/libraries/FASTHTML.md` – Notes and critical issues for FastHTML usage
- `healthyselfjournal/session.py` – Session orchestration
- `healthyselfjournal/transcription.py` – STT backends abstraction
- `healthyselfjournal/audio.py` – Audio utilities and formats
- `healthyselfjournal/llm.py` – LLM provider abstraction and prompt rendering
- `healthyselfjournal/storage.py` – Session persistence, paths, and utilities
- `docs/reference/AUDIO_VOICE_RECOGNITION_WHISPER.md` – Whisper/STT considerations


## Principles, key decisions

- Framework: FastHTML for server-side pages; minimal TypeScript only for audio recording (compiled with `tsc`)
- Pipeline: Client-side recording → upload to server → server runs STT/LLM via existing modules
- Storage: Single shared `./sessions/` directory for both CLI and web
- UX parity: Match CLI controls/flow unless there’s a strong reason to differ
- Loading UI: Simple spinner and “Transcribing…” first; enhance later (SSE/WebSockets)
- Focus behavior: Continue recording when tab loses focus
- User model: Single-user, local-only for now; avoid auth/user management
- Concurrency: Allow simultaneous CLI and web; timestamped session IDs avoid collisions
- Audio format: Record and upload `audio/webm; codecs=opus` from the browser; no server-side transcoding. STT must accept `webm/opus` directly.
- JavaScript scope: Keep JS minimal (UI + recording only). All processing remains in Python.
- Port and bind: Default to `127.0.0.1:8765`.
- Static assets: Serve at `/static/` path from `healthyselfjournal/static/`.
- No clip cap: Do not enforce a maximum duration in the browser.
- No hidden fallbacks: If STT cannot accept `webm/opus`, fail clearly (do not transcode or silently switch providers).


## Open questions and concerns (sounding-board)

Top questions (please decide):
1) STT backend choice(s) for web path: Confirm we will target STT providers that accept `audio/webm; codecs=opus` end-to-end (e.g., OpenAI Whisper API). Is this acceptable for Web V1 with no transcoding path?
2) Safari support policy: Is it acceptable to target Chrome primarily and mark Safari as “best effort” (no WAV polyfill or alternate encoder in V1)?

Secondary questions (lower priority):
- Phosphor icons integration: web font vs SVG sprites vs inline SVG? Minimal first: inline the spinner SVG to avoid extra asset plumbing.
- Safari support: `MediaRecorder` coverage is newer on Safari; acceptable to require relatively modern Safari, or add a WAV encoder fallback later?
- Max clip size: enforce a client-side cap (e.g., 10 minutes) to avoid huge uploads? CLI parity suggests keeping soft boundaries.
- Later desktop bundling target: Electron vs Tauri vs PyWebView – useful to keep assets under `healthyselfjournal/static/` either way.


## Stages & actions

### Stage: Repository prep and dependencies
- [ ] Add FastHTML dependency and minimal server entrypoint (no JS yet)
- [ ] Decide and document default port (e.g., 8765) and `--sessions-dir` CLI flag for the web entrypoint
- [ ] Create `healthyselfjournal/static/` directories for `js/`, `css/`, and optional icons
- [ ] Add TypeScript tooling: `tsconfig.json`, `package.json` (scripts: `build`, `watch`) – no separate webserver; assets served statically by FastHTML
- [ ] Serve static files at `/static/` mapped to `healthyselfjournal/static/`
- [ ] Bind to `127.0.0.1:8765` by default; make port configurable

Acceptance criteria:
- Running `uv run --active healthyselfjournal journal web` starts a local server and serves a “Hello” page at `http://127.0.0.1:8765`
- `/static/` serves assets from `healthyselfjournal/static/`

### Stage: Server skeleton (FastHTML)
- [ ] Create minimal FastHTML app: index page with record button, level meter container, and placeholder status text
- [ ] Implement upload route: POST audio blob with metadata (mime type, duration)
- [ ] Add `--sessions-dir` support and wire to `storage.py` for path resolution
- [ ] Return JSON including `session_id`, `segment_id`, and processing status

Acceptance criteria:
- Can POST a dummy file to `/upload` and receive JSON response with persisted file path

### Stage: Client audio recorder (TypeScript)
- [ ] Implement basic recording with `MediaDevices.getUserMedia({ audio: true })` + `MediaRecorder`
- [ ] Compute a visual meter using Web Audio `AnalyserNode` and display a simple bar
- [ ] Mirror CLI controls: click to start, click to stop; ESC to cancel (discard buffer)
- [ ] On stop, upload blob to server with form data; show spinner while awaiting server response
- [ ] Handle focus loss: continue recording (no auto-pause)

Acceptance criteria:
- In a modern Chromium browser, can record, stop, and see an upload succeed with a visible spinner during upload

### Stage: Server-side processing integration
- [ ] Save uploaded audio under `sessions/<timestamp>/browser-<n>.webm` (or mime-based extension)
- [ ] Invoke `transcription.py` with configured backend; persist raw STT `.stt.json` next to audio
- [ ] Update or create the session markdown (`.md`) with transcript and frontmatter using `session.py`/`storage.py`

Acceptance criteria:
- Uploading a real recording creates a new session folder with audio (`.webm`), transcript, raw STT JSON, and a markdown file with frontmatter (no transcoding step)

### Stage: Dialogue loop and UI feedback
- [ ] Call `llm.py` to generate next question based on transcript and context rules
- [ ] Show simple spinner “Transcribing…” / “Thinking…” states using a Phosphor spinner icon or inline SVG
- [ ] Display the AI’s next question on the page; provide a “Record response” button to continue the loop
- [ ] Maintain session continuity across multiple uploads within the same browser session

Acceptance criteria:
- After first upload, the page shows an AI-generated follow-up question; subsequent recordings append to the same session’s markdown

### Stage: Parity with CLI behavior
- [ ] Implement cancel/discard thresholds to auto-discard very short/empty clips (same thresholds as CLI)
- [ ] Ensure ESC cancels recording cleanly (no upload)
- [ ] Keep session summaries updated in frontmatter after each Q&A
- [ ] Reuse the same prompt templates from `prompts/*.jinja`

Acceptance criteria:
- Short accidental blips are auto-discarded; summaries are present/updated in markdown; prompts match CLI behavior

### Stage: Testing, health checks, and resilience
- [ ] Add an integration test that POSTs a small test audio sample and verifies session artifacts
- [ ] Run existing offline tests: `pytest tests/test_storage.py` (and others) to ensure web changes don’t regress core behavior
- [ ] Fast failure paths: clear errors for missing mic permissions, oversized blobs, STT/LLM network issues
- [ ] Log processing events to `sessions/events.log`

Acceptance criteria:
- Automated test passes creating a session via upload; no regressions in existing tests locally

### Stage: Documentation and developer ergonomics
- [ ] Update `docs/reference/COMMAND_LINE_INTERFACE.md` with web entrypoint details
- [ ] Add `docs/reference/FILE_FORMATS_ORGANISATION.md` note about web-originated audio using `webm/opus` (no conversion)
- [ ] Add a brief `docs/reference/BACKGROUND_PROCESSING.md` note on web uploads and conversion
- [ ] Document TypeScript build commands in `docs/reference/SETUP.md`

Acceptance criteria:
- Docs include how to run the web server, build TS, and where files go

### Stage: Optional enhancements (later)
- [ ] Progressive feedback via SSE for STT and LLM phases
- [ ] Device picker UI and input level calibration
- [ ] Recordings list sidebar for current session
- [ ] "New session"/"Resume session" routing


## Risks and mitigations

- STT provider format mismatch: Web V1 assumes `webm/opus` is accepted by the configured STT backend. Mitigation: choose/verify provider early and add automated test with a small `webm/opus` sample.
- Safari/older browsers: `MediaRecorder` support may lag; document minimum browser versions; consider a WAV encoder fallback later if needed
- Latency and large uploads: begin with simple whole-clip upload; later consider chunking or streaming only if needed
- Local HTTP exposure: bind to `127.0.0.1` by default; document how to change; avoid auth complexity in single-user mode
- Concurrency with CLI: rely on timestamped session IDs; ensure web segments append to the correct session


## Definition of Done (Web V1)

- Start server via `uv run --active healthyselfjournal journal web [--sessions-dir PATH]`
- Record in browser; stop; upload; server transcribes and writes markdown with frontmatter
- Show simple loading states and render the next AI question
- Mirror CLI cancel/discard behavior and session summary updates
- Tests: at least one integration test for upload→session creation; no regressions in existing tests


## Notes for future desktop bundling

- Keep static assets under `healthyselfjournal/static/` to ease packaging
- Avoid runtime requirements for external JS servers; pure `tsc` builds suffice
- Keep network calls configurable to toggle local vs cloud STT/LLM as in CLI


## Appendix

Implementation hints:
- Use `FormData` with a single `Blob` field (e.g., `audio`) and JSON fields (`mime`, `duration_ms`)
- Name files `browser-001.webm`, `browser-002.webm` within the session to preserve order
- Level meter: `AnalyserNode.getByteTimeDomainData` mapped to a simple bar is sufficient for V1
- Consider a simple router path structure: `/` (home), `/upload` (POST), `/session/<id>` (view)

