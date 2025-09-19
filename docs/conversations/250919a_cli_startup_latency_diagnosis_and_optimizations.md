# CLI Startup Latency – Diagnosis and Optimizations - 2025-09-19

---
Date: 2025-09-19
Duration: ~10 minutes
Type: Problem-solving
Status: Active
Related Docs: `docs/reference/COMMAND_LINE_INTERFACE.md`, `docs/reference/DIALOGUE_FLOW.md`, `docs/reference/FILE_FORMATS_ORGANISATION.md`
---

## Context & Goals
Investigate why the `journal` CLI takes a couple of seconds to show the first question and identify changes to reduce perceived and actual startup latency.

## Key Background
- User: "Why is it so slow to start up? it takes at least a couple of seconds from running the `journal` CLI command before it displays the first question…"
- Product goal emphasizes low friction and fast time-to-first‑prompt for voice-first journaling.

## Main Discussion
### Root Causes Identified
- Heavy imports at process start:
  - `healthyselfjournal/audio.py` imports `numpy`, `sounddevice`, and `soundfile` at module import time; these can be slow on macOS in fresh processes.
- Runtime dependency check imports:
  - `_verify_runtime_deps_for_command` in `healthyselfjournal/cli.py` imports `readchar`, `sounddevice`, `soundfile`, and `numpy` to verify presence, duplicating import cost before any UI prints.
- History scan and YAML parsing on session start:
  - `SessionManager.start()` calls `load_recent_summaries()` which scans/reads prior `.md` files and YAML frontmatter; cost grows with number of sessions.
- Minimal but synchronous file I/O:
  - Event logger init touches `sessions/events.log`; new session file and directory are created before first question.
- Normal backend resolution and env checks:
  - STT selection and TTS/env checks are quick but still precede the first UI panel in the current flow.

### Why the first question appears late
The intro panel and opening question are printed only after dependency verification, STT backend resolution, event logger init, and `SessionManager.start()` (directory creation + history load + transcript write). Import + I/O + history scan dominate the 1–3s.

### Optimizations Proposed
- Defer heavy imports (actual latency reduction):
  - Move `numpy`, `sounddevice`, `soundfile` imports inside functions in `audio.py` (e.g., `record_response`, `create_input_stream`).
- Make dependency checks non‑importing (actual latency reduction):
  - Replace import attempts with `importlib.util.find_spec()` checks in `_verify_runtime_deps_for_command` to avoid loading heavy modules twice.
- Print intro earlier (perceived latency reduction):
  - Show the “session starting” panel and opening question before history loading and before constructing audio/transcription backends, then continue initialization.
- Lighten or defer history loading (actual latency reduction on large history):
  - Lower `CONFIG.max_recent_summaries` and/or token budget for startup; optionally defer loading summaries until after first recording.
- Operational tip:
  - Launch from an already‑activated venv (e.g., `uv run --active healthyselfjournal journal`) to avoid slower interpreter cold starts.

## Alternatives Considered
- Caching Python module imports across invocations (out of scope for CLI; requires daemon/process reuse).
- Precomputing a compact summaries index to avoid scanning `.md` files (possible future improvement; adds complexity).

## Decisions Made
- Diagnosis accepted. Implementation deferred until requested.

## Open Questions
- Target time‑to‑first‑question on your machine (e.g., <500ms vs <1s)?
- Preference for perceived vs absolute latency improvements (intro‑first vs deeper defers)?
- Is reducing history context on startup acceptable if it significantly speeds the first prompt?

## Next Steps
- If approved, implement:
  - Lazy imports in `audio.py`.
  - Switch deps check to `find_spec` in `cli.py`.
  - Move intro panel display earlier in the `journal` flow.
  - Optionally defer `load_recent_summaries()` or lower budgets.

## Sources & References
- `healthyselfjournal/cli.py` – dependency verification and command wiring.
- `healthyselfjournal/cli_journal_cli.py` – startup flow, session start/resume UI, event logger init.
- `healthyselfjournal/session.py` – session start, history load, transcript write.
- `healthyselfjournal/history.py` – `load_recent_summaries()` scanning/parsing.
- `healthyselfjournal/audio.py` – heavy imports and capture loop.
- `docs/reference/COMMAND_LINE_INTERFACE.md` – user-facing flow and options.

## Related Work
- `docs/planning/250918a_core_refactors_and_architecture_improvements.md` – relevant to startup and modularization.
