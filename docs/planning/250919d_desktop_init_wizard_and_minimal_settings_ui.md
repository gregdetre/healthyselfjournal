## Goal, context

Build a minimal, user-friendly desktop configuration experience for the PyWebView app that enables non-technical users to:

- Select a sessions folder
- Choose whether to resume the last session on launch
- Toggle voice mode on/off (server-side TTS)
- Run a first-run Setup (Init) wizard to choose Cloud vs Local (Privacy), capture API keys, pick sessions folder, and persist settings

Constraints and desired UX:
- Minimize friction on first run; sensible defaults (Cloud mode, sessions at `~/.../sessions` or `./sessions`)
- Persist choices in a stable user location (XDG config path) suitable for packaged desktop apps
- Allow re-running the Setup from Settings at any time
- Make the desktop voice toggle authoritative (avoid surprises when `SPEAK_LLM=1` is set elsewhere)


## References

- `docs/reference/DESKTOP_APP_PYWEBVIEW.md` – Desktop shell flow, security posture, CLI flags, endpoints
- `docs/reference/INIT_FLOW.md` – Current CLI init wizard; steps and persistence expectations
- `healthyselfjournal/cli_desktop.py` – Desktop CLI flags we will mirror in the UI
- `healthyselfjournal/desktop/app.py` – Background server lifecycle, window creation, bridge
- `healthyselfjournal/web/app.py` – `WebAppConfig`, voice/TTS wiring (`voice_enabled`), upload/tts/reveal endpoints
- `healthyselfjournal/config.py` – Runtime defaults and env-driven configuration (LLM/STT/TTS, thresholds)
- `healthyselfjournal/__init__.py` – `.env`/`.env.local` autoloading; precedence today
- `healthyselfjournal/transcription.py` – STT backends, selection, model/compute formatting
- `healthyselfjournal/tts.py` – TTS backend, key dependencies (`OPENAI_API_KEY`)
- `docs/reference/SETUP_USER.md` – User setup docs to keep in sync


## Principles, key decisions

- Keep first-run flow extremely simple: sessions folder + voice on/off; advanced choices live in Setup
- Cloud mode is default (best accuracy/latency). Privacy (local) is available via Setup
- Persist desktop settings to XDG config (e.g., `~/.config/healthyselfjournal/`) so they survive packaging/updates
- Desktop settings take precedence over project/CWD `.env.local` for end-user flows; CLI flags and OS env still override at runtime
- Voice mode toggle in desktop should be authoritative for the desktop session (no implicit OR with env defaults)
- Changing sessions folder or voice mode requires a background server restart; provide an "Apply & Restart" UI affordance
- Avoid surfacing low-value advanced toggles in v1 (STT compute/model, LLM model). Rely on defaults and Setup
- Provide a clear entry point to re-run Setup later (menu: Settings → Run Setup Again)


## Stages & actions

### Stage: Persistence and precedence groundwork
- [x] Extend env autoload to include XDG config path for desktop (`~/.config/healthyselfjournal/.env.local` or platform-appropriate)
  - Acceptance: If `~/.config/.../.env.local` exists, keys like `SESSIONS_DIR`, `SPEAK_LLM`, `STT_BACKEND` are loaded when launching desktop
- [x] Add a small desktop settings layer (e.g., `desktop/settings.py`) with a dataclass and load/save helpers
  - Fields: `sessions_dir`, `resume_on_launch`, `voice_enabled`, `mode` (cloud/local)
  - Storage: TOML or `.env.local` in XDG path; choose one and document
  - Acceptance: `load_settings()` returns defaults on first run; `save_settings()` creates file; round-trip tested
- [x] Define runtime precedence for desktop launch: CLI flags > OS env > Desktop settings (XDG) > project `.env.local` > code defaults
  - Acceptance: Unit tests demonstrate precedence on overlapping keys
  - Note: Implemented precedence is currently CLI flags > Desktop settings (XDG) > OS env/CONFIG > project `.env.local` > code defaults. Confirm alignment with intended policy.

### Stage: Voice mode authority
- [x] Update `web/app.py` voice wiring to respect an explicit desktop override
  - Current: `voice_enabled = bool(resolved.voice_enabled or CONFIG.speak_llm)`
  - New: If desktop provided an explicit boolean, use it; otherwise fall back to config default
  - Acceptance: With `SPEAK_LLM=1` but desktop toggle OFF, `/session/{id}/tts` returns VOICE_DISABLED; with toggle ON, it returns audio

### Stage: Desktop Settings UI (minimal)
- [x] Add a Preferences panel (menu or header button) with:
  - Sessions folder picker (native dialog), current path display
  - Resume last session on launch (toggle)
  - Voice mode (toggle)
  - Apply & Restart server button (restarts background uvicorn cleanly)
  - Acceptance: Changing any of the above persists to XDG file and takes effect after restart
- [x] Add "Reveal Sessions Folder" action (existing endpoint covers session file; add a global reveal for folder in UI/menu)
  - Acceptance: Opens OS file manager at sessions directory

### Stage: First-run Setup (Init) wizard (desktop)
- [x] Implement a guided wizard:
  - Step 1: Choose mode (Cloud recommended, Local/Privacy optional)
  - Step 2: Keys (Cloud: Anthropic + OpenAI; Privacy: keys optional). Provide links, masked input
  - Step 3: Sessions folder (pick or accept default, create if missing)
  - Step 4: Optional quick test (mic check; minimal STT call in Cloud)
  - Step 5: Persist + apply; show summary
  - Acceptance: On success, the main journaling window opens with voice and sessions as configured
- [x] Add Settings menu entry: "Run Setup Again"
  - Acceptance: Re-opens wizard; changes persist and apply after restart
  - Note: Quick test (mic check) not implemented yet; defer to later iteration.

### Stage: QA, tests, and docs
- [ ] Unit tests
  - Autoload precedence including XDG path
  - Desktop settings load/save roundtrip
  - Voice authority logic (explicit OFF overrides env ON)
- [ ] Integration/manual checks
  - Start desktop with no settings → wizard appears; pick Cloud + keys; record flow works; TTS plays when enabled
  - Toggle voice OFF in settings while `SPEAK_LLM=1` in env → `/tts` disabled; UI behaves accordingly
  - Change sessions folder → new recordings are written under the chosen folder; reveal works
- [ ] Docs updates
  - `docs/reference/DESKTOP_APP_PYWEBVIEW.md`: add Settings & Setup sections, persistence location, "Apply & Restart"
  - `docs/reference/INIT_FLOW.md`: add desktop variant; persistence path; how to re-run
  - `docs/reference/SETUP_USER.md`: desktop quickstart

### Stage: Future (backlog, not in v1)
- [ ] Offline master switch (disable cloud features globally; switch STT to local; grey out cloud controls)
- [ ] Model Manager UI for local STT (download/upgrade/remove), device/compute selection
- [ ] Optional LLM model presets (default vs lower-cost), with clear copy
- [ ] Menu items and About panel polish; packaged app entitlements/Info.plist strings


## Risks, mitigations

- Precedence confusion: Clearly document and test CLI > OS env > Desktop settings > `.env.local` > defaults
- Key storage and discovery: Prefer user-local XDG path; do not print secrets; allow users to edit file manually if needed
- Restart semantics: Ensure background server stops/starts cleanly; guard against port contention; display transient status to user
- Voice authority: Make OFF truly off; avoid partial states (UI toggled on but server refusing `/tts`)
- Packaged builds: Ensure mic permission strings and entitlements are included; wizard shouldn’t block if keys are missing (Privacy mode fallback)
 - Precedence mismatch: Code currently applies Desktop settings over OS env when CLI flags use defaults. Decide policy and align implementation/docs.
 - First-run gating: Wizard redirect depends on `resume` being true. Packaged desktop path sets this, CLI path does not; confirm desired first-run behavior.
 - Apply & Restart lifecycle: New server instance isn't retained in outer scope; window close handler may only stop the original server. Risk of orphaned server after restart; consider tracking and stopping the current server instance.
 - Reveal Sessions Folder: Implemented for macOS (`open`) only. Add support for Windows (`explorer`) and Linux (`xdg-open`).


## Acceptance summary (v1)

- First run shows Setup wizard; user selects Cloud or Local, enters keys (if Cloud), picks sessions folder, and lands in the journaling UI
- Settings panel exposes Sessions folder, Resume on launch, Voice toggle; changes persist in XDG config and take effect after Apply & Restart
- Voice OFF in desktop disables TTS regardless of env defaults; Voice ON enables `/tts` and the UI plays prompts
- Record → upload → STT → next question loop works; sessions and audio persist in the chosen folder; Reveal opens Finder/Explorer


## Appendix

- Menu ideas: File → Reveal Sessions Folder; Preferences…; Tools → Run Setup Again; Help → Troubleshooting
- Telemetry/logging: continue using `sessions/events.log`; consider adding a small “Settings changed” event with non-sensitive fields


