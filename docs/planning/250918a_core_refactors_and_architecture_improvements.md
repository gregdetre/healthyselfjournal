## Goal, context

Refactor oversized functions and reduce duplication in the core journaling flow to improve readability, testability, and robustness without changing user-facing behavior. Prioritise small, high-value changes first (utility extraction and duplication removal), then tackle larger modularizations (`cli_journal.journal`, `audio.record_response`).

Context highlights:
- Voice-first CLI app with real-time recording, STT backends, and LLM-driven dialogue.
- Some long functions combine UI, I/O, streaming, and control logic.
- A few near-duplicate helpers (duration formatting, safe WAV deletion, Anthropic request assembly, key handling).
- Global `CONFIG` read/mutate patterns make some behaviors implicit.

Non-goals for this effort:
- Changing the dialogue prompts or STT/LLM behavior.
- Feature additions; focus is code health and internal maintainability.


## References

Code (most relevant first):
- `examinedlifejournal/cli_journal.py` — CLI entry for journaling loop; setup, backend resolution, panels, TTS, loop, streaming, finalize.
- `examinedlifejournal/audio.py` — Recording/meter/controls, short-answer guard, post-processing, MP3 conversion.
- `examinedlifejournal/session.py` — Session state, exchange recording, persistence, next-question generation, summaries.
- `examinedlifejournal/transcription.py` — Backend selection, OpenAI/faster-whisper/MLX/whispercpp.
- `examinedlifejournal/llm.py` — Prompt rendering, Anthropic/Ollama calls, streaming, thinking mode handling.
- `examinedlifejournal/storage.py` — Markdown + frontmatter I/O (non-atomic), append helpers.
- `examinedlifejournal/prompts/question.prompt.md.jinja` — Embedded example-questions behavior.

Docs:
- `docs/reference/COMMAND_LINE_INTERFACE.md` — CLI behaviors/options.
- `docs/reference/DIALOGUE_FLOW.md` — Flow across capture → STT → LLM → summary.
- `docs/reference/FILE_FORMATS_ORGANISATION.md` — Session and audio artifact layout.
- `docs/reference/PRODUCT_VISION_FEATURES.md` — Guides tradeoffs (simplicity, low friction).
- `docs/reference/AUDIO_VOICE_RECOGNITION_WHISPER.md` — STT references.
- `gjdutils/docs/instructions/WRITE_PLANNING_DOC.md` — Planning structure followed here.


## Principles, key decisions

- Prefer incremental, behavior-preserving edits with tests between stages.
- Extract utilities for duplicated logic; avoid cross-module hidden coupling.
- Keep UI text and control semantics unchanged unless explicitly noted.
- Minimise global state reliance; pass explicit config where practical.
- Add atomicity and typed surfaces where they improve robustness with low risk.
- Maintain logging parity; centralize event shapes gradually.


## Progress
- Discovery complete for high-impact, low-effort refactors:
  - Duration helpers: `_format_duration` in `audio.py` and `session.py`; `_format_minutes_seconds` in `session.py`.
  - Safe WAV deletion: duplication in `audio._maybe_start_mp3_conversion` and `session._persist_raw_transcription`.
  - Anthropic request assembly: shared logic needed across `llm._call_anthropic` and `llm.stream_followup_question`.
  - Key handling: normalization logic in `audio._wait_for_stop` and `cli_journal._run_mic_check`.
- Completed implementations:
  - Added `examinedlifejournal/utils/time_utils.py` with `format_hh_mm_ss` and `format_mm_ss` and replaced call sites in `audio.py` and `session.py` (kept small shim functions for backward parity).
  - Added `examinedlifejournal/utils/audio_utils.py` with `maybe_delete_wav_when_safe` and updated both `audio._maybe_start_mp3_conversion` and `session._persist_raw_transcription` to delegate.
  - Ran minimal tests: `pytest tests/test_storage.py` passed; broader suite hit external `OPENAI_API_KEY` requirement under `gjdutils` (expected offline).


## Stages & actions

### Upfront preparatory actions
- [ ] Ensure environment is ready
  - [ ] Activate venv at `/Users/greg/.venvs/experim__examinedlifejournal` and run `uv sync --active`
  - [ ] Run minimal offline tests: `pytest tests/test_storage.py`
- [ ] Consider creating a feature branch `250918a_core_refactors` for this multi-stage work
  - [ ] If created, merge back into `main` at the end after all checks pass


### Stage: Extract shared duration formatting helpers
- [x] Create `examinedlifejournal/utils/time_utils.py` with:
  - [x] `format_hh_mm_ss(seconds: float) -> str` (used by saved messages)
  - [x] `format_mm_ss(seconds: float) -> str` (used for conversation duration)
- [x] Replace duplicates in `audio.py` and `session.py`
- [x] Add unit tests for edge cases (sub-second, 59→60 rollovers, hours)
- **Acceptance**:
  - [x] No references to private `_format_duration` remain; tests green


### Stage: Factor safe WAV deletion into a single helper
- [x] Create `examinedlifejournal/utils/audio_utils.py` with:
  - [x] `maybe_delete_wav_when_safe(wav_path: Path) -> None`
    - Checks for sibling `.mp3` and `.stt.json` before deletion
    - Emits the existing `audio.wav.deleted` event
- [x] Replace duplicated logic in `audio._maybe_start_mp3_conversion` and `session._persist_raw_transcription`
- [x] Add unit tests with temp dirs (safe delete vs missing artifacts)
- **Acceptance**:
  - [x] Single implementation; both call sites delegate; tests green


### Stage: Centralize Anthropic request assembly and thinking budget
- [ ] Add `_build_anthropic_kwargs(model, prompt, max_tokens, temperature, top_p, top_k, thinking_enabled)` in `llm.py`
- [ ] Use it from both `_call_anthropic` and `stream_followup_question`
- [ ] Add focused tests for budget calculation boundaries (>=1024, < max_tokens)
- **Acceptance**:
  - [ ] Logic exists once; streaming and non-streaming parity retained


### Stage: Standardize key handling
- [ ] Add `examinedlifejournal/utils/keys.py` with:
  - [ ] `read_one_key_normalized() -> Literal["ENTER","ESC","Q","SPACE","OTHER"]`
- [ ] Replace local normalization in `audio._wait_for_stop` and `cli_journal._run_mic_check`
- [ ] Ensure behavior parity for ESC sequences and Ctrl-C handling
- **Acceptance**:
  - [ ] Shared key normalization; manual smoke check for pause/cancel/quit paths


### Stage: Unify next-question request assembly
- [ ] In `session.py`, add private helper `_build_question_request(transcript: str) -> QuestionRequest`
- [ ] Reuse in `generate_next_question` and `generate_next_question_streaming`
- **Acceptance**:
  - [ ] Single request builder; both methods call it; tests pass


### Stage: Split `cli_journal.journal` into cohesive units
- [ ] Extract helpers:
  - [ ] `prepare_runtime_and_backends(ctx_opts) -> selection + flags`
  - [ ] `start_or_resume_session(manager, sessions_dir, opening_question) -> (state, question)`
  - [ ] `run_journaling_loop(manager, question, stream_llm)`
  - [ ] `finalize_or_cleanup(manager, state, sessions_dir)`
- [ ] Keep CLI surface identical; minimize churn in printed strings
- [ ] Unit-test the extracted pure parts (where feasible)
- **Acceptance**:
  - [ ] Function shrinks significantly; interactive flow unchanged in manual run


### Stage: Break up `audio.record_response` into testable subfunctions
- [ ] Extract:
  - [ ] `create_input_stream(sample_rate, callback)` context manager
  - [ ] `run_meter_loop(...) -> frames_written, voiced_seconds`
  - [ ] `apply_short_answer_guard(duration, voiced) -> bool`
  - [ ] `postprocess_and_convert(wav_path, sample_rate, convert_to_mp3)`
- [ ] Preserve timing, meter, and disposition semantics
- [ ] Add unit tests for short-answer guard and VAD proxy thresholds
- **Acceptance**:
  - [ ] Smaller, focused units; identical UX in manual tests


### Stage: Reduce reliance on global `CONFIG` in core paths (scoped)
- [ ] Thread explicit settings through `SessionConfig` where practical (e.g., thresholds for short-answer guard; delete-wav flag)
- [ ] Keep global reads only for CLI defaults and prompts
- **Acceptance**:
  - [ ] No behavior change; clearer data flow in signatures


### Stage: Atomic transcript writes
- [ ] Update `storage.write_transcript` to write to `.<name>.tmp` then `Path.replace()`
- [ ] Ensure file permissions/ownership preserved where possible
- [ ] Add tests to validate atomic replacement and content integrity
- **Acceptance**:
  - [ ] Atomicity assured; no flakiness in tests


### Stage: Typed event payloads (incremental)
- [ ] Introduce lightweight `TypedDict`/dataclasses in `events.py` for high-volume events: `audio.record.*`, `stt.*`, `llm.*`, `session.*`
- [ ] Replace bare dicts gradually in touched call sites
- **Acceptance**:
  - [ ] Key events annotated; mypy (if run) can validate shapes


### Stage: Documentation & CLI messaging polish
- [ ] Update `docs/reference/COMMAND_LINE_INTERFACE.md` if control tips change wording
- [ ] Verify the in-app tip mentioning “embedded examples” aligns with current prompts (since `question_bank.py` was removed)
- [ ] Update `docs/reference/FILE_FORMATS_ORGANISATION.md` to reflect safe deletion behavior and post-processing
- **Acceptance**:
  - [ ] Docs consistent with behavior; wording parity maintained


### Stage: Health checks and wrap-up
- [ ] Run minimal offline tests: `pytest tests/test_storage.py`
- [ ] Run broader suite (with API keys set) if available
- [ ] Lint/type checks as configured; fix any regressions
- [ ] If a feature branch was used, merge back to `main`
- [ ] Summarize changes and learnings in commit messages per `gjdutils/docs/instructions/GIT_COMMIT_CHANGES.md`


## Risks, mitigations
- Hidden coupling through `CONFIG` might cause subtle behavior shifts — mitigate by adding focused tests and keeping defaults unchanged.
- Interactive flows are harder to test — rely on small, pure helper units + manual smoke tests.
- Event payload shape changes could break downstream consumers — keep event names stable; add new fields conservatively.


## Notes & future ideas (out of scope here)
- Consider a small state machine for `record_response` (Paused/Recording/Stopping) to further clarify control transitions.
- Explore a plugin-style STT provider registry for simpler backend additions.
- Optional richer VAD (we intentionally keep current lightweight proxy).
