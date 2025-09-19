# Desktop App (PyWebView)

The desktop experience wraps the existing FastHTML web UI in a PyWebView shell. It keeps all application logic in Python while providing a signed-friendly path for macOS bundling and a foundation for cross‑platform distribution.

## Introduction

This document describes the Desktop PyWebView app: what it does, how it is structured, how to run it, and how it will work when complete. It is forward‑looking but grounded in the current implementation.

## See also

- `../planning/250919c_pywebview_desktop_bundling_plan.md` – Implementation plan, risks, and acceptance criteria for the desktop app.
- `ARCHITECTURE.md` – System architecture and module boundaries that the desktop app reuses.
- `COMMAND_LINE_INTERFACE.md` – Index of CLI docs; desktop command is exposed via the main CLI.
- `WEB_RECORDING_INTERFACE.md` – Details of the shared web UI the desktop shell embeds.
- `DIALOGUE_FLOW.md` – Conversation loop expectations and UX.
- `FILE_FORMATS_ORGANISATION.md` – Where sessions, audio, and summaries are stored on disk.
- `AUDIO_VOICE_RECOGNITION_WHISPER.md` – STT backends and performance guidance.
- `../../healthyselfjournal/desktop/app.py` – Desktop runtime: window creation, background server, JS bridge, Apply & Restart.
- `../../healthyselfjournal/cli_desktop.py` – Typer command that launches the desktop experience.
- `../../healthyselfjournal/web/app.py` – FastHTML app with strict security headers for desktop and web.
- `../../healthyselfjournal/desktop/settings.py` – Desktop settings persistence (XDG TOML) used by Preferences and Setup.

## Principles and key decisions

- Use PyWebView to minimise IPC and keep the AI stack entirely in Python.
- Reuse the FastHTML web UI via a localhost loopback server to avoid UI divergence.
- Keep models out of the signed bundle; manage downloads at first‑run under platformdirs.
- Enforce a strict Content Security Policy and "Permissions‑Policy" that only grants microphone.
- Keep the JS bridge minimal: only window control and development helpers.
- Move STT/LLM workloads into separate processes to keep the UI responsive.

## Architecture and runtime

High‑level flow:

1) CLI entrypoint (`healthyselfjournal desktop`) builds a FastHTML app and starts a background `uvicorn` server on `127.0.0.1:<port>`.
2) PyWebView creates a window pointing at the loopback URL. `webview.settings.enable_media_stream = True` enables mic capture in WKWebView.
3) A tiny JS bridge exposes `quit()`, `toggle_devtools()` (when enabled), `pick_sessions_dir()` (native folder picker), and `apply_and_restart()` to restart the background server after saving settings. No arbitrary Python invocation is allowed.
4) The FastHTML app serves the journaling UI and endpoints:
   - `GET /` boots/resumes a session and redirects to `/journal/<sessions_dir>/<session_id>/`.
   - `POST /session/{id}/upload` accepts MediaRecorder uploads (Opus WEBM/OGG), persists segments, triggers STT, schedules summaries, and produces the next question.
   - `POST /session/{id}/tts` synthesizes assistant text into audio when voice mode is on.
   - `POST/GET /session/{id}/reveal` reveals the session markdown in Finder (macOS).
   - `POST /reveal/sessions` reveals the configured sessions directory in Finder.
   - `GET /settings` renders Preferences; `POST /settings/save` persists desktop settings.
   - `GET /setup` renders first‑run Setup; `POST /setup/save` persists initial mode/keys and sessions path.
5) Security middleware applies CSP and related headers on every response.

Worker processes:

- Transcription worker process handles longer audio segments to keep the UI thread unblocked. Partial transcript updates are exposed via a job status endpoint the client polls.
- Optional local LLM worker via `llama-cpp-python` for offline operation (configurable mode).

## Commands and options

```bash
uv sync --active
uv run --active healthyselfjournal desktop \
  --sessions-dir ./sessions \
  --port 0 \
  --host 127.0.0.1 \
  --resume \
  --voice-mode \
  --tts-model "${TTS_MODEL}" \
  --tts-voice "${TTS_VOICE}" \
  --tts-format wav \
  --title "Healthy Self Journal" \
  --width 1280 --height 860 \
  --fullscreen \
  --debug \
  --devtools \
  --server-timeout 15 \
  --confirm-close
```

Notes:

- `--port 0` selects a free ephemeral port; pass a number to pin it.
- `--resume` opens the most recent session if one exists.
- `--voice-mode` enables server‑side TTS for assistant prompts; TTS options can be set via flags or `user_config.toml`.
- `--debug` prints WKWebView console logs to stdout (development only).
- `--devtools` opens the embedded browser devtools (development only).

## Behaviour

- The desktop command starts a background HTTP server and opens a WKWebView window.
- Recording uses `getUserMedia` + `MediaRecorder` in the embedded WebView; uploads are persisted immediately to the session folder.
- STT is performed via the configured backend; short/quiet responses can be automatically discarded. A background worker processes segments and streams partial results via polling `/session/{id}/jobs/{job_id}`.
- Each successful upload returns 201 Created and updates the running total duration; the next question is returned once STT completes.
- When enabled, TTS synthesizes the assistant prompt server‑side and the browser plays it.
- Desktop voice authority: the Preferences toggle (Voice mode) overrides any `SPEAK_LLM` default for the desktop session.
- Closing the window stops the background server and exits cleanly.

## Security posture

- Strict CSP (default‑src 'self', no remote content; `blob:` allowed only where needed) and related headers are applied on all responses.
- `Permissions-Policy: microphone=(self)`; camera/geolocation disabled.
- `frame-ancestors 'none'` to prevent embedding; `X-Frame-Options: DENY`.
- Localhost origin only; no external network calls are needed for the core loop.
- Minimal JS bridge surface area; no eval or remote code loading.

## Current vs target state

Current state (implemented):

- Desktop runner (`desktop/app.py`) that starts the FastHTML server and embeds PyWebView.
- Microphone streams enabled; strict security headers applied.
- JS bridge with `quit()` and optional `toggle_devtools()`.
- CLI command `healthyselfjournal desktop` with window/server options.

- Desktop settings persistence in XDG config (`~/.config/healthyselfjournal/settings.toml`) with: sessions folder, resume on launch, voice mode, and mode (cloud/local). Precedence: CLI flags > OS env > Desktop settings > project `.env.local` > defaults.
- Preferences UI (`/settings`) with Sessions folder picker, Resume on launch, Voice mode; Apply & Restart restarts the background server to apply changes.
- First‑run Setup wizard (`/setup`) for desktop: choose mode (Cloud/Privacy), enter keys, choose sessions folder; writes XDG `.env.local` for keys and desktop settings TOML.
- Reveal Sessions Folder action (menu/button) opens the configured sessions directory in Finder.

- Multiprocessing transcription worker (`workers/transcription_worker.py`) integrated with the web app; partial transcripts are polled by the client.
- Model manager for faster‑whisper/ctranslate2 assets under platformdirs with checksum validation.
- LLM adapter with `cloud` and `local` modes and a `cloud_off` switch for privacy‑first usage.

Target state (planned; tracked in planning doc):

- Optional local LLM worker process and richer desktop settings (init wizard, cloud‑off toggle in UI).
- Packaged app via PyInstaller (one‑folder), with signing/notarisation on macOS.
- Desktop UX niceties: menu items, About panel, cloud‑off switch in settings.

Migration status:

- POC is functional in development. Mic permission prompts in packaged builds and model download manager are next.

## Gotchas and limitations

- WKWebView `getUserMedia` requires mic permission; packaged builds need Info.plist usage string and entitlement.
- Only Opus WEBM/OGG uploads are accepted in the web interface; alternate formats are rejected.
- If `static/js/app.js` is missing, the UI loads with limited functionality.
- Long uploads or heavyweight STT/LLM may stall if not moved to worker processes.
- The Setup wizard is only shown when running the desktop shell and no desktop settings file exists; it’s not shown in plain web runs.

## Troubleshooting

- Blank window or 404: ensure the background server started; check console for the selected port and try `http://127.0.0.1:<port>/`.
- No microphone prompt: check macOS System Settings → Privacy & Security → Microphone; for packaged builds, ensure Info.plist contains `NSMicrophoneUsageDescription`.
- Upload rejected: verify MediaRecorder produced Opus webm/ogg and that file size is below `web_upload_max_bytes`.
- No TTS audio: ensure `--voice-mode` is enabled and TTS config resolves; check server logs for `TTS_FAILED`.
- Next question empty: inspect logs for `QUESTION_FAILED`; verify LLM configuration.

## Manual testing (development run)

1) Environment
   - Activate the preferred venv and sync: `uv sync --active`.
   - Optional: set a temp sessions dir for testing, e.g. `./experim_sessions`.

2) Launch desktop (dev)
   - Run: `uv run --active healthyselfjournal desktop --sessions-dir ./experim_sessions --port 0 --voice-mode --debug`.
   - Expect: a window appears, console shows selected port, CSP headers applied, and no external network requests.

3) Microphone + recording
   - On first recording, expect a macOS mic permission prompt. Allow it.
   - Record a short answer; the meter should move and the upload should complete.
   - Verify that a new `browser-###.webm` (or `.ogg`) is written under `./experim_sessions/<session_id>/`.

4) STT + next question
   - After upload, expect a transcript in the UI and the next question to appear.
   - Check the session markdown in `./experim_sessions/<session_id>.md` for the new exchange and updated duration.

5) TTS (voice mode)
   - If `--voice-mode` is on, expect the assistant prompt to be synthesized and played.
   - If not, inspect the network tab (devtools) for a successful `/session/{id}/tts` response.

6) Resume and reveal
   - Close and relaunch with `--resume` and confirm the latest session opens.
   - Click the "Reveal in Finder" action (or call `/session/{id}/reveal`) and verify Finder highlights the markdown file (macOS).

7) Close down
   - Close the window; the background server should stop. No lingering process on the chosen port.

Acceptance for dev run: window opens < 3s; recording works end‑to‑end; transcript + next question update; TTS returns audio when enabled; no CSP violations.

## Packaging & distribution (cross‑platform)

Overview:

- Build native artifacts with PyInstaller per‑OS. PyInstaller is not a cross‑compiler; use native machines or CI runners for each target OS.
- macOS can build Apple Silicon (arm64) and Intel (x86_64) binaries. Prefer a universal2 build when feasible; otherwise ship two builds.
- Use GitHub Actions matrix to produce artifacts for macOS (arm64 + Intel), Windows, and Linux from a single tag or manual run.

What gets packaged:

- One‑folder bundles that include the Python runtime and app code. Models are not bundled; they are managed/downloaded at runtime under platformdirs.
- The FastHTML UI and static assets are included from `healthyselfjournal/static/`.

macOS notes:

- Ensure `Info.plist` contains `NSMicrophoneUsageDescription` and the app is signed with the hardened runtime and appropriate microphone entitlement for distribution.
- For development builds, unsigned artifacts are fine; for distribution, sign and notarize after PyInstaller emits the app bundle.
- You can build per‑arch on native runners (macOS‑14 arm64, macOS‑13 x86_64) or attempt a `universal2` build when your Python and native deps support it.

Windows notes:

- Build on Windows runners to produce an installer‑free one‑folder directory or wrap with an installer later if desired. Code signing is optional but recommended.

Linux notes:

- Build on a reasonably old glibc baseline for wider compatibility (GitHub `ubuntu-latest` is acceptable for initial releases). No signing required.

Local packaging (developer machine):

```bash
# ensure environment is synced
uv sync --active

# package using the project spec
uv run --active pyinstaller packaging/HealthySelfJournal.spec

# resulting bundles land under ./dist/
```

CI packaging (GitHub Actions):

- A CI workflow for desktop builds is not included yet. When adding one, use a matrix to build per‑OS and per‑arch, and invoke the PyInstaller spec. Update this section with the workflow path once added.

References:

- PyInstaller FAQ (cross‑compiling not supported) – pyinstaller.org
- pywebview Packaging guide – pywebview.org

Post‑package, repeat the manual testing steps to verify parity with dev.
