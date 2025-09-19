### Goal, context

- Build a signed, notarized macOS desktop app for the voice-first journaling system using PyWebView, minimizing process boundaries and IPC while retaining FastHTML for the UI layer.
- Prioritize early risk surfacing on Apple Silicon macOS. We do not currently have Windows/Linux machines; we will still plan for them but only validate on macOS now.
- Keep models (Whisper/LLM) out of the app bundle. Provide first-run download, integrity checks, and GPU capability detection.


### References

- `docs/conversations/250919b_desktop_bundling_electron_tauri_pywebview.md` — Bundling options analysis; recommendation: PyWebView first.
- `docs/reference/ARCHITECTURE.md` — System architecture and module boundaries.
- `docs/reference/PRODUCT_VISION_FEATURES.md` — Product goals and constraints.
- `docs/reference/COMMAND_LINE_INTERFACE.md` — Current CLI flows; informs which flows to expose in desktop.
- `docs/reference/RECORDING_CONTROLS.md` — Expected UX around recording.
- `docs/reference/FILE_FORMATS_ORGANISATION.md` — Session file layout, summaries, audio artifacts.
- `docs/reference/PRIVACY.md` — Local-first, privacy posture; constraints for network usage.
- Code modules: `healthyselfjournal/web/app.py` (FastHTML app), `audio.py`, `transcription.py`, `session.py`, `storage.py`, `llm.py`, `tts.py`.
- External docs: PyWebView (media permissions, JS bridge, packaging), PyInstaller, faster-whisper/ctranslate2, llama-cpp-python, Apple notarization (notarytool), WebKit media capture entitlements.


### Principles, key decisions

- Choose PyWebView to minimize IPC and packaging complexity for a Python-heavy AI stack.
- Keep UI responsive by moving heavy AI work (STT/LLM) into worker processes via `multiprocessing` (not threads) to avoid GIL/CPU contention and UI stalls.
- Prefer a local loopback HTTP server (`127.0.0.1`) for the FastHTML UI over file:// to reuse existing web app with minimal divergence, locked down via allowlists and CSP.
- Models are downloaded/stored under `platformdirs.user_data_dir(app)` with versioned subfolders and checksums; not in the signed bundle.
- Production builds disable debug features, remote content, eval, and restrict the JS bridge surface area. Apply strict CSP.
- macOS as the initial target: support microphone capture (getUserMedia/MediaRecorder) in WKWebView with required entitlements and permission prompts.
- Revisit Tauri/Electron only if “kill criteria” are met (see below).
 - Local LLM engine choice: ship local mode with `llama-cpp-python`; keep Ollama as an optional dev path. Do not block PyWebView POC on LLM engine swap.


### Stages & actions

#### Stage: Prepare local environment and baseline checks (macOS, Apple Silicon)
- [x] Ensure external venv active (`/Users/greg/.venvs/experim__healthyselfjournal`) and project sync
  - [x] `uv sync --active`
  - [x] Run unit tests (offline subset): `pytest tests/test_storage.py -q`
  - Acceptance: Sync completes; baseline tests pass locally.
- [x] Install PyWebView and PyInstaller in the venv
  - [ ] `uv run --active pip install pywebview pyinstaller`
  - [x] Declared both packages in `pyproject.toml`; `uv sync --active` will install on next sync.
  - Acceptance: Packages installed; `python -c "import webview; print(webview.__version__)"` works.

#### Stage: Mic capture POC in PyWebView (highest risk early)
- [x] Create a minimal PyWebView runner that points at the existing FastHTML app on `http://127.0.0.1:<port>`
  - [x] Expose only needed JS bridge functions (quit + devtools) and set `enable_media_stream=True` (`healthyselfjournal/desktop/app.py`).
  - [x] Configure a strict CSP and disable remote URLs via middleware (`healthyselfjournal/web/app.py`).
  - Acceptance: A window opens and serves the current FastHTML UI.
  - Progress: Desktop shell exposed via `healthyselfjournal desktop`; manual WKWebView mic verification still pending.
- [ ] Verify `getUserMedia` + `MediaRecorder` inside WKWebView
  - [x] Add Info.plist entries: `NSMicrophoneUsageDescription` (and if needed `NSSpeechRecognitionUsageDescription`).
  - [ ] Confirm permission prompt appears and recording meter updates in real time.
  - [ ] Save short test recording and check it persists under the current session.
  - Acceptance: Recording works end-to-end (start/stop/meter/saved WAV), no UI freezes.
- [ ] If mic capture fails, stop and discuss
  - Kill criteria: WKWebView cannot capture mic reliably even with entitlements and `enable_media_stream=True`.

#### Stage: Offline STT integration via faster-whisper (macOS)
- [x] Add a `multiprocessing` transcription worker invoked from the UI flow
  - [x] Parent process remains responsive while worker handles audio segments.
  - [x] Stream partial STT results to UI through the HTTP app (Server-Sent Events or polling) to avoid JS bridge complexity.
  - Acceptance: UI remains responsive; partial transcripts appear for longer recordings.
- [x] Model management (first run)
  - [x] Create a model manager that downloads faster-whisper models into `platformdirs` location, with resume and checksum.
  - [x] Detect Metal vs CPU and select a compatible model backend (prefer CPU first; Metal optional).
  - Acceptance: On clean machine, first-run downloads and then transcribes offline without errors.

#### Stage: Local LLM adapter and llama-cpp-python (parallel; do not block)
- [x] Introduce an LLM adapter in `llm.py` with modes: `cloud` and `local_llama` (default: `cloud` for POC)
  - [x] Config flag in `user_config.toml` or env var to switch modes; add a `cloud_off` option to force offline.
  - [x] Keep Ollama optional as a dev-only mode; do not include in packaged build by default.
  - Acceptance: App runs with `cloud` mode unchanged; switching to `local_llama` routes calls locally.
- [ ] Bring up `llama-cpp-python` locally (dev-run only)
  - [ ] Install `llama-cpp-python` CPU wheel first; verify inference with a small quantized model (e.g., Q4_K_M).
  - [x] Manage model paths under `platformdirs` (not bundled); add checksums and resume.
  - [ ] Optional: test Metal wheel; fall back to CPU if unstable.
  - Acceptance: Dialogue loop works offline end-to-end with local LLM in dev.
- [ ] Defer packaging of local LLM until after PyWebView mic POC passes
  - [ ] Draft PyInstaller collection rules for `llama_cpp` binaries in one-folder build.
  - Acceptance: Packaging plan documented; not a blocker for mic POC.

#### Stage: Packaging with PyInstaller (one-folder) for macOS
- [x] Add PyInstaller spec file for the PyWebView runner
  - [x] Include binary/data collection for `ctranslate2` and `faster_whisper` (collect dynamic libs, data).
  - [x] Exclude models from bundle; verify runtime model directory use.
  - Acceptance: Dist folder contains an app that launches successfully.
- [ ] Run packaged app and repeat the mic + STT flow
  - [ ] Verify permissions prompts still work; recording and transcription succeed.
  - Acceptance: Packaged app functions equivalently to dev run.
- [ ] If packaging misses binary deps, stop and discuss
  - Risk indicators: missing `libctranslate2`/`FFmpeg`-related symbols, crashes on import.

#### Stage: Signing and notarization (macOS)
- [ ] Configure hardened runtime and microphone entitlement; sign the app with Developer ID
  - [ ] Add entitlements.plist (mic access) and sign the `.app`.
  - Acceptance: `codesign --verify --deep --strict` passes locally.
- [ ] Notarize with `notarytool` and staple
  - [ ] Submit to Apple; after success, staple ticket.
  - Acceptance: Gatekeeper allows app to run on a clean mac without warnings.

#### Stage: Security, privacy, and settings
- [x] Enforce local-only content and strict CSP
  - [x] No remote JS or mixed content; block `eval` via middleware + regression test (`tests/test_web_app_security.py`).
  - Acceptance: CSP violations are zero in console during typical use.
- [x] Add explicit setting to disable network LLM usage (cloud off switch)
  - [ ] Document storage locations and export policy in UI.
  - Acceptance: With cloud off, no outbound requests are made (verified via proxy or logs).

#### Stage: Distribution and updates (macOS first)
- [ ] Package distribution as DMG/ZIP with branding
  - Acceptance: App mounts/installs and runs.
- [ ] Decide on auto-update strategy for PyWebView app
  - [ ] Evaluate simple “check updates” with GitHub Releases; plan future auto-update.
  - Acceptance: Manual update path documented and tested.

#### Stage: Cross-platform planning placeholders (defer execution)
- [ ] Windows plan: WebView2 bootstrap, microphone permissions, code signing (EV optional), MSI/EXE packaging.
- [ ] Linux plan: GTK/Qt choice, GStreamer/PipeWire dependencies, AppImage/Flatpak packaging.
- [ ] Define minimum acceptance criteria to run on both platforms later.

#### Stage: External critique and review
- [ ] Commit initial POC planning and implementation; seek critique from another assistant or peer.
- [ ] Integrate feedback; update this plan and references.


### Risks and kill criteria

- Mic capture in WKWebView is not reliable even with entitlements and `enable_media_stream=True`.
- Packaging burdens for `faster-whisper`/`ctranslate2` binary deps remain brittle despite PyInstaller hooks.
- Long-running STT/LLM workloads still impact UI responsiveness or cause memory pressure in-process.
- Notarization complications prevent shipping within reasonable effort.
 - If `llama-cpp-python` packaging remains brittle, consider a temporary dev-only local LLM mode and/or an Ollama sidecar fallback; do not block PyWebView viability on this.

If any of the above persist after one focused iteration, reconsider Tauri (sidecar) or Electron (bundle-everything) with clear trade-offs.


### Measurements and acceptance definition of done (macOS initial)

- Cold start time (first window visible): target < 2.5s packaged.
- Recording latency (start to meter activity): < 300ms.
- STT throughput (real-time factor): ≥ 0.8x for small models on Apple Silicon CPU; Metal optional.
- App size (one-folder): target < 250MB without models.
- Signed + notarized app runs on a clean mac without warnings.


### Appendix: packaging notes (macOS)

- PyWebView: create window with media streams enabled; do not expose broad JS bridge; prefer HTTP localhost origin for CSP.
- PyInstaller: start with one-folder; use `--collect-datas/--collect-binaries` for `ctranslate2` and `faster_whisper`; validate with `otool -L` on bundled libs.
- Models: store under `~/Library/Application Support/HealthySelfJournal/models/<name>/<version>`; manage with checksums and resume-capable downloads.
- Notarization: use `notarytool` with API key; staple tickets; ensure Info.plist contains microphone usage description.


## Appendix - How optimistic are you that this will work for the privacy (i.e., local AI case) on an Apple Silicon Mac?

### Short answer
- High confidence: 8/10 that a fully local, privacy‑preserving macOS build with PyWebView will work.

### Why I’m optimistic
- **Local STT**: faster‑whisper on Apple Silicon (CPU) is reliable and fast enough for journaling; packaging is manageable with PyInstaller if we collect ctranslate2 binaries. Fallback: `whisper.cpp` with Metal if needed.
- **Local LLM**: `llama-cpp-python` with Metal is mature on macOS; feasible to run small/medium models locally with acceptable latency for a dialogue loop.
- **Privacy controls**: PyWebView can run strictly local (localhost origin or file://), enforce a strict CSP, and ship with a “cloud off” switch so no network calls are made.
- **macOS platform fit**: WKWebView supports `getUserMedia` with the right entitlements; code signing/notarization is well-trodden.

### Main risks and mitigations
- **WKWebView MediaRecorder quirks**: If `getUserMedia`/MediaRecorder misbehave, we can capture PCM via WebAudio and stream to Python; or handle mic capture natively via Python and bypass MediaRecorder.
- **Packaging binary deps**: ctranslate2 and llama binaries can be finicky. Mitigate with one‑folder PyInstaller builds and explicit `--collect-binaries/--collect-datas`; fallback to `whisper.cpp` if faster‑whisper proves stubborn.
- **Performance/memory**: Run STT/LLM in separate processes to keep UI responsive; default to small models and CPU first, detect/enable Metal when available.
- **Notarization entitlements**: Add mic usage description and hardened runtime entitlements; this is procedural work rather than a blocker.

If we hit the two big blockers (mic capture in WKWebView, or bundling ctranslate2/llama in a signed app), we have viable fallbacks. Overall, the local‑only privacy case looks very achievable on your Apple Silicon Mac.
