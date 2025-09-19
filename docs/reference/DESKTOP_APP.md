# Desktop App (PyWebView)

The desktop experience wraps the existing FastHTML web UI in a PyWebView shell. It keeps
all application logic in Python while providing a signed-friendly path for macOS
bundling.

## Command

```bash
uv run --active healthyselfjournal desktop \
  --sessions-dir ./sessions \
  --port 0 \
  --voice-mode
```

Key switches:

- `--port 0` chooses a free ephemeral port; override with a specific port if needed.
- `--debug` surfaces console logs from the embedded WebView (development only).
- `--devtools` opens the WKWebView devtools; only expose in trusted environments.
- `--voice-mode` keeps parity with the browser experience, enabling server-side TTS.

## Behaviour

- The CLI spins up the FastHTML server in the background and points the PyWebView
  window at `http://127.0.0.1:<port>/`.
- Microphone capture is enabled by default (`webview.settings.enable_media_stream = True`).
- Closing the window tears down the embedded server cleanly.
- A minimal JS bridge exposes `quit()` (for menu buttons) and, when `--devtools`
  is used, a `toggle_devtools()` helper.

## Security posture

- New middleware applies a strict Content Security Policy and related headers to
  every response, limiting resources to `self` plus the `blob:` URLs needed for
  in-memory audio clips.
- `Permissions-Policy` grants microphone access only and disables camera and
  geolocation.
- `frame-ancestors 'none'` hardens the UI against clickjacking when reused outside
  the desktop shell.

## Next steps

- Wire transcription/LLM workloads into multiprocessing workers to keep the UI responsive.
- Prepare a PyInstaller spec and Info.plist entitlements for signing/notarisation.
- Add UX glue (menu items, about panel) once the native shell stabilises.
