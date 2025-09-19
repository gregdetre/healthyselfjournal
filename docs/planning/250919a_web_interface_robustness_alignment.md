# Web interface robustness and CLI-flow alignment plan

## Goal, context

The command-line journaling flow is robust and well-tested. The new FastHTML web interface reuses the same core pipeline, but a few gaps make it feel more brittle and less aligned with the CLI. This plan prioritizes small, high-value changes that:

- Improve reliability of uploads and processing
- Align client behavior with CLI gating and keyboard controls
- Ensure persisted state (indexes, frontmatter) stays consistent
- Harden error surfaces and developer ergonomics

This document is a concrete, staged plan to reach parity with the CLI experience while keeping the web code small and maintainable.


## References

- healthyselfjournal/web/app.py — FastHTML app, routes, session boot/resume, TTS
  - Upload handling, state management, and integration with `SessionManager`
- healthyselfjournal/static/ts/app.ts — Browser recorder, keyboard handling, uploads, TTS playback
  - Client heuristics for short-answer gating and UX flow
- healthyselfjournal/web/templates/session_shell.html.jinja — HTML shell and data-* config bridge
  - Exposes server-side config to client via dataset; includes debug section
- healthyselfjournal/session.py — Session orchestration, resume logic, persistence
  - `SessionManager.process_uploaded_exchange`, `_update_frontmatter_after_exchange`
- healthyselfjournal/audio.py — CLI capture, short-answer guard, voiced detection
  - dBFS threshold (`CONFIG.voice_rms_dbfs_threshold`) used by CLI; reference logic
- healthyselfjournal/storage.py — Frontmatter and markdown persistence helpers
  - `append_exchange_body` and frontmatter layout used by both flows
- tests/test_web_app.py — Upload/TTS/resume tests using Starlette `TestClient`
  - Stubs backends, validates persistence and next question flow
- docs/reference/WEB_INTERFACE.md — Architecture and runtime flow for the web UI
  - Declares intended behavior; some small mismatches with current code
- docs/diagrams/250919a_cli_ui_flow.mermaid — Baseline CLI flow diagram
  - Aids in mapping parity points for the web flow


## Principles, key decisions

- Single pipeline: Web uploads should go through the same session/persistence/transcription logic as CLI without format divergence.
- Fail-soft UX: Client should remain responsive even when TTS/transcription fails; surface clear statuses without blocking.
- Trust but verify: Client may apply early “short-answer” gating, but the server should enforce it to protect cost and consistency.
- Minimal surface area: Keep the number of routes and client states small; reuse existing helpers; prefer small deltas to fast-follow features.
- Parity-first: Align keyboard controls, short-answer gating semantics, response indexing, and resume behavior with CLI.
- Observability: Log events with enough context to diagnose issues quickly.


## Stages & actions

### Stage: Align client voiced detection with CONFIG (dBFS threshold) (high impact, low effort)

Rationale: CLI voiced gating uses `CONFIG.voice_rms_dbfs_threshold` in dBFS; client uses a fixed amplitude heuristic (`voicedThreshold = 0.07`). This mismatch causes web vs CLI variance in short-answer decisions.

- [ ] Provide `data-voice-rms-dbfs-threshold` on `<body>` in `session_shell.html.jinja` using `CONFIG.voice_rms_dbfs_threshold` (already passed through to template context).
- [ ] In `app.ts`, compute window RMS as today, then dBFS = `20*log10(rms + 1e-10)` and compare against the configured threshold; remove the 0.07 amplitude magic value.
- [ ] Keep current UX and meter visuals unchanged; only change the voiced-seconds accumulation rule.
- [ ] Manual verification across a few voice levels; ensure voiced seconds track reasonably vs CLI.

Acceptance criteria:
- Client voiced accumulation flips in/out close to CLI behavior for the same audio.
- No regression to meter rendering performance.


### Stage: Implement quit-after (“Q”) parity (high impact, low effort)

Rationale: The CLI supports “Q” to end after the current response. Web should match to reduce cognitive load.

- [ ] Client: In `app.ts`, listen for `q`/`Q` during recording. When pressed, set a `quitAfter` flag.
- [ ] On upload, include `quit_after` in form data.
- [ ] Server: Parse `quit_after` (truthy) in `upload()` and set `AudioCaptureResult.quit_after=True` when building the capture object.
- [ ] After successful upload, if server response indicates `quit_after`, the client must NOT auto-start the next recording and should set status to “Session complete (quit-after).”
- [ ] Logging: Ensure `session.exchange.recorded` logs `quit_after` for web, as it does for CLI.
- [ ] Tests: Extend web upload test to submit `quit_after=1` and ensure client-facing payload includes `quit_after` and that no next auto-start occurs (unit-test server response; client behavior can be exercised manually or in a Playwright follow-up if added later).

Acceptance criteria:
- Pressing Q ends the loop after the upload completes, mirroring CLI.
- Logged events include `quit_after: true`.


### Stage: Fix response_index drift for web segments (high impact, low effort)

Rationale: `response_index` increments in `process_uploaded_exchange`, while filename numbering is derived in the upload route by probing `browser-XXX.webm`. If a collision occurs and `next_index` advances, `response_index` can become inconsistent with the stored filename and logs.

- [ ] After selecting the final `browser-XXX` index (after collision avoidance) in `web/app.py`, set `state.manager.state.response_index` to that index before calling `process_uploaded_exchange`.
- [ ] Alternatively, pass the chosen index and have `SessionManager` set `response_index`, but prefer the small server-side update to avoid API changes.
- [ ] Tests: Upload multiple times and simulate an existing `browser-001.webm` to force a collision. Verify:
  - [ ] Persisted filename increments as expected.
  - [ ] `session.exchange.recorded` event includes the matching index.

Acceptance criteria:
- `response_index` matches the actual segment index of the persisted webm file.
- Logs and frontmatter reflect consistent segment ordering.


### Stage: Remove or verify the redundant client “question” field (high impact, low effort)

Rationale: The client sends `question` but the server uses `state.current_question`. Drift risks confusion.

- [ ] Simplest: Remove `question` from the client form submission (`app.ts`).
- [ ] Or, if retained, validate equality server-side and log a warning if mismatched.
- [ ] Tests: None required beyond existing flows; optional assertion that server ignores mismatched client `question`.

Acceptance criteria:
- Only a single source of truth for the asked question; no silent mismatch risks.


### Stage: Deduplicate TTS MIME mapping logic (high impact, low effort)

Rationale: MIME mapping for TTS is implemented twice: once for route return and once for template. Reduce duplication to prevent divergence.

- [ ] Create a small helper (e.g., `_tts_format_to_mime(fmt: str) -> str`) in `web/app.py` and reuse in both `tts()` and `_render_session_shell()`.
- [ ] Tests: Keep existing TTS test and extend to cover at least two formats (e.g., wav, mp3) with patched synth function.

Acceptance criteria:
- One authoritative mapping function; tests pass for multiple formats.


### Stage: Centralize media type and extension handling (high impact, low effort)

Rationale: Upload route infers extensions and accepts MIME types locally. Centralizing avoids drift and makes hardening easier.

- [ ] Move `_extension_for_mime` and an allowlist (e.g., `is_supported_media_type`) into `utils/audio_utils.py`.
- [ ] Reuse these helpers in `web/app.py` upload path (and any future ingestion).
- [ ] Tests: Unit-test mapping and allowlist with common/edge MIME strings.

Acceptance criteria:
- A single source of truth for media-type checks and extension mapping; web route uses it.


### Stage: Canonical error catalog and mapping (high impact, low effort)

Rationale: JSON error codes and CLI messages should align for diagnostics and tests.

- [ ] Create `healthyselfjournal/errors.py` with constants for error codes (e.g., `UNKNOWN_SESSION`, `MISSING_AUDIO`, `INVALID_PAYLOAD`, `AUDIO_FORMAT_UNSUPPORTED`, `PROCESSING_FAILED`).
- [ ] Use these constants in `web/app.py` responses and (optionally) CLI error surfaces/logs.
- [ ] Tests: Assert returned codes match constants in web tests.

Acceptance criteria:
- Consistent, documented error codes across web and CLI contexts.


### Stage: TTS options resolution in one place (high impact, low effort)

Rationale: Model/voice/format resolution is spread between config and web. Centralize to avoid divergence and simplify future CLI voice mode.

- [ ] Add `resolve_tts_options(overrides: dict | None) -> TTSOptions` in `tts.py` that merges CONFIG with overrides.
- [ ] Use in `web/app.py` when constructing `app.state.tts_options` and as a fallback inside `/tts` route.
- [ ] Tests: Extend TTS test to override model/voice/format and assert headers reflect the chosen format.

Acceptance criteria:
- One resolution path for TTS options; easier to share with future CLI voice mode.


### Stage: Event schema consistency and origin tagging (medium impact, low effort)

Rationale: Events are already useful; standardize fields to simplify analysis.

- [ ] Document a minimal event schema (in-code docstring or `docs/reference/FILE_FORMATS_ORGANISATION.md`): include `ui` (`cli`|`web`), `session_id`, `response_index`, segment label/name, and relevant model/backend fields.
- [ ] Ensure web and CLI events include `ui` consistently; fill any gaps.
- [ ] Tests: Light assertion in unit tests that certain events contain required keys.

Acceptance criteria:
- Events carry consistent key fields across CLI and web; easier to trace flows.


### Stage: Test fixtures reuse for stubbed backends (medium impact, low effort)

Rationale: Web tests define local stubs; centralizing promotes coverage reuse and consistency.

- [ ] Move STT/LLM/TTS stubs used by web tests to `tests/stubs.py`.
- [ ] Import stubs in `tests/test_web_app.py` and any CLI tests needing them.
- [ ] Optionally parametrize stubbed model ids for broader coverage.

Acceptance criteria:
- Shared stubs reduce duplication; tests stay green.


### Stage: Storage/frontmatter writer as single source of truth (medium impact, low effort)

Rationale: Reinforce the invariant that only `SessionManager` updates frontmatter/body.

- [ ] Add a short note to `WEB_INTERFACE.md` and/or `storage.py` docstrings stating that frontmatter and transcript body are only mutated via `SessionManager` methods.
- [ ] Quick grep to ensure no direct frontmatter writes in web code paths.

Acceptance criteria:
- Documented invariant; codebase adheres to it.


### Stage: Session layout and naming helpers (medium impact, low effort)

Rationale: Segment naming (`yyMMdd_HHmm_NN.wav` vs `browser-XXX.webm`) and collision handling live in separate spots.

- [ ] Add `utils/session_layout.py` with helpers like `next_cli_segment_name(session_id, dir)` and `next_web_segment_name(dir)` handling collision loops.
- [ ] Use these helpers in CLI and web paths to reduce duplication.
- [ ] Tests: Create temporary dirs with seeded files and assert next names are as expected.

Acceptance criteria:
- Name sequencing/collision logic is centralized; both UIs rely on it.


### Stage: Extract short-answer gating helper (shared; no behavior change for web) (medium impact)

Rationale: Consolidate semantics while keeping web behavior unchanged for now.

- [ ] Extract a small helper in `audio.py` or `utils/audio_utils.py`, e.g., `should_discard_short_answer(duration_s, voiced_s, cfg) -> bool`, mirroring CLI.
- [ ] Use in CLI capture flow (replacing inline check) to unify semantics.
- [ ] Optionally log the server’s evaluation in web upload (debug-only) without enforcing it yet.

Acceptance criteria:
- One implementation of the guard semantics; web can adopt later without rework.


### Stage: Shared session indexing utilities (complements resume; medium impact)

Rationale: Resume and upload should agree on the current/next index across both naming families.

- [ ] Add `session_utils.get_max_recorded_index(session_dir, session_id)` that scans for both `*_NN.wav` and `browser-*.webm`.
- [ ] Use in `SessionManager.resume()` and optionally in web upload to set `response_index` to the chosen web index.
- [ ] Tests: Seed both patterns and verify correct max index is returned.

Acceptance criteria:
- A shared index source prevents drift between resume logic and upload logic.


### Stage: Optional unified voiced analysis for non-WAV inputs (optional; medium impact)

Rationale: For future parity, support estimating voiced seconds from WEBM via optional decode.

- [ ] Add a feature-flagged utility that decodes WEBM/OGG to PCM (via ffmpeg if available) and estimates voiced seconds using the same dBFS logic as CLI.
- [ ] Keep disabled by default to avoid latency; expose a troubleshooting toggle.
- [ ] Tests: Skip by default; add a conditional test if ffmpeg present in CI/dev.

Acceptance criteria:
- Path exists to evaluate voiced time server-side when needed, without burdening the hot path by default.


### Stage: Better error surfaces when STT backend can’t accept Opus (high impact, low effort)

Rationale: If a local backend rejects `webm/opus`, errors are generic. Provide actionable guidance.

- [ ] In `upload()` `processing_failed` path, detect known audio-format errors (e.g., exceptions or STDERR signatures if exposed) and return `{error:"audio_format_unsupported", detail:"...", hint:"Try cloud-openai or a backend supporting WEBM/Opus"}` with 415/422.
- [ ] Log `web.upload.stt_format_error` with backend id and model.
- [ ] Docs: Add a troubleshooting note in `WEB_INTERFACE.md` about Opus support and suggested remedies.

Acceptance criteria:
- Users see specific error reasons and actionable hints; logs capture backend/context.


### Stage: Resume indexing should consider browser segments (medium impact)

Rationale: `SessionManager.resume()` currently infers `response_index` from WAV patterns and frontmatter entries; it ignores `browser-*.webm`. Web-only sessions resumed via `--resume` may report zero prior responses.

- [ ] Extend `resume()` to detect `browser-*.webm` in the session audio directory and compute the max index across both patterns.
- [ ] Update the “existing responses” log to reflect web segments.
- [ ] Tests: Create a temp session with `browser-001.webm` and resume → expect `response_index == 1`.

Acceptance criteria:
- Resumed sessions reflect the correct next index when only web uploads exist.


### Stage: Frontmatter doc vs implementation mismatch (medium impact)

Rationale: `WEB_INTERFACE.md` suggests web entries store `{wav: null}`, but the code stores the actual filename under the `wav` key (and no mp3). Prefer updating docs to reality for simplicity.

- [ ] Update `WEB_INTERFACE.md` Storage layout section to clarify that the `wav` key contains the audio filename regardless of file type (e.g., `.webm`), and `mp3` is null for web clips.
- [ ] Optional: Consider renaming to `file` in a future format change; out of scope for this stage.

Acceptance criteria:
- Docs match current behavior; no code change needed now.


### Stage: Upload validation hardening (medium impact)

Rationale: Current server accepts any uploaded blob; add basic validation to reduce misuse.

- [ ] Restrict MIME types to `audio/webm`, `audio/webm;codecs=opus`, and optionally `audio/ogg`.
- [ ] Enforce a practical max content size; return `413` for oversized payloads (configurable; default generous).
- [ ] Return `415` for unsupported media types; include a hint in JSON.
- [ ] Tests: Submit unsupported MIME and oversized payload to get correct status codes.

Acceptance criteria:
- Non-conforming uploads are rejected with clear status and message; valid uploads continue working.


### Stage: Tests to cover missing routes and new behaviors (medium impact)

- [ ] Add a test for `GET /session/{id}/reveal` (currently only POST path is covered indirectly): ensure it mirrors POST semantics or returns 404/501 appropriately.
- [ ] Add tests for server-side short-answer discard behavior.
- [ ] Add tests for resume with `browser-*.webm`.
- [ ] Extend TTS tests to cover multiple formats using stub synth.

Acceptance criteria:
- All newly added behaviors are under test; `pytest tests/test_web_app.py` passes locally/offline.


### Stage: TTS UX status feedback (medium impact)

Rationale: Client swallows TTS errors silently. Add non-blocking feedback.

- [ ] In `app.ts` `playTts()`, on failure set a transient status: “TTS unavailable; continuing silently.” Suppress console noise.
- [ ] Ensure ENTER press during TTS still cancels playback and starts recording immediately (already implemented; verify no regressions).

Acceptance criteria:
- Users get gentle feedback when TTS fails; no flow interruption.


### Stage: Session map lifecycle (lower impact)

Rationale: `app.state.sessions` grows without eviction. Locally this is small, but a simple policy avoids unbounded growth.

- [ ] Implement a small LRU or “only one active session” map with last-access timestamps.
- [ ] Provide a debug-only route or log entry to list active sessions on server start/stop.

Acceptance criteria:
- Session state does not grow unbounded during development.


### Stage: Static asset readiness feedback (lower impact)

Rationale: If `static/js/app.js` is missing/outdated, the page loads without functionality.

- [ ] On `index()`, best-effort check for `static/js/app.js` file presence; if missing, render a debug banner in the shell (“Run npm install && npm run build”).
- [ ] Keep FastHTML return semantics simple (strings only).

Acceptance criteria:
- Clear developer hint when JS build artifact is missing.


### Stage: FastHTML/fastcore compatibility shim hygiene (lower impact)

Rationale: `_get_fast_html_class()` monkey-patches `fastcore.xml.ft`. It works but is brittle across versions.

- [ ] Pin tested versions in `pyproject.toml` (e.g., FastHTML >= 0.12) and gate the patch behind version checks.
- [ ] Note in `WEB_INTERFACE.md` that route handlers should return strings/objects and let FastHTML wrap.

Acceptance criteria:
- Reduced risk of future framework upgrades breaking startup.


## Upfront prep and workflow notes

- Use the project venv as preferred: `/Users/greg/.venvs/experim__healthyselfjournal` and `uv run --active`.
- Front-end: `npm install` then `npm run watch` during development; commit only built changes when packaging.
- Tests (offline):
  - Minimal: `pytest tests/test_storage.py`
  - Web: `PYTHONPATH=. pytest tests/test_web_app.py`
- Run web during dev: `uv run --active healthyselfjournal journal web --reload --open-browser`


## Rollout and regression strategy

- Stage-by-stage small PRs; each ends with passing tests and manual smoke test in the browser (Chromium).
- For behavior changes that affect persistence, ensure backward compatibility of frontmatter and file layout; do not rename keys in this pass.
- Keep logs high signal; verify `sessions/events.log` reflects new events.


## Success criteria

- Web uploads are consistently gated server-side, reducing spurious STT runs.
- Client’s short-answer semantics align closely with CLI.
- Keyboard parity includes quit-after; auto-loop behaves as expected.
- Response indexes and resume behavior are consistent for web sessions.
- Errors are clearer and more actionable, especially around Opus support.
- Tests cover the newly added paths and behaviors.


## Appendix

- Consider adding a follow-up mermaid diagram for the web flow once parity is achieved, reusing colors and legends from `docs/diagrams/250919a_cli_ui_flow.mermaid`.
- Future enhancements (out of scope): SSE progress updates, device picker, richer history rendering, and streaming question deltas.


