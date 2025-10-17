## Goal, context

- Build resilience around transcription failures while keeping the default UX “detect and suggest” (no automatic background backfill by default).
- Ensure that when STT fails after a recording (CLI or Web):
  - The audio is already saved and discoverable for backfill.
  - The user is clearly informed how to fix it (exact command suggested).
  - The session markdown remains coherent. We will capture the asked question and a placeholder transcript so later backfill can replace it cleanly.
- Extend the `reconcile` tool to backfill across both CLI and Web recordings and optionally update markdown when placeholders are present.

### Current behaviour (summary)

- CLI: Audio is saved, STT is attempted synchronously. On failure, the session continues, and the user sees a hint to run `healthyselfjournal reconcile --sessions-dir '<dir>'`. Startup/finalization also display a pending count. Reconcile currently scans only `*.wav`, writes `<seg>.stt.json`, and may delete WAV when safe. It does not update markdown.
- Web: Upload saves `browser-XXX.webm`/`.ogg`, then STT is attempted synchronously. Failures return an error to the client; no reconcile is triggered; reconcile ignores web media.
- Raw STT is persisted per segment as `<segment>.stt.json` when transcription succeeds.


## References

- Code
  - `healthyselfjournal/cli_journal_cli.py` — CLI loop; startup/finish detect-and-suggest; failure message with reconcile hint; `_count_missing_stt()` counts only `.wav`.
  - `healthyselfjournal/session.py` — High-level session API; `record_exchange` (CLI capture path); `_transcribe_and_store()` persists raw STT and appends to markdown; `process_uploaded_exchange()` (web path).
  - `healthyselfjournal/cli_reconcile.py` — Scans `*.wav`, writes `<seg>.stt.json` atomically, optional safe WAV deletion; does not touch markdown.
  - `healthyselfjournal/web/app.py` — Upload pipeline; returns error on STT failure; no pending banner; no reconcile integration.
  - `healthyselfjournal/static/js/app.js` — Client recorder; shows a generic upload failure message.
  - `healthyselfjournal/utils/audio_utils.py` — Helpers incl. extension inferencing and WAV deletion when safe.
- Evergreen docs
  - `docs/reference/FILE_FORMATS_ORGANISATION.md` — Canonical session artefacts; `.stt.json` alongside audio.
  - `docs/reference/WEB_ARCHITECTURE.md` — Web storage layout (`browser-XXX.webm` + `.stt.json`).
  - `docs/reference/AUDIO_VOICE_RECOGNITION_WHISPER.md` — STT backends and event logging.
  - `docs/reference/RECORDING_CONTROLS.md` — Discard/quit semantics.


## Principles, key decisions

- Default remains “detect and suggest” (no auto reconcile). Keep user control and transparency.
- Prefer “placeholder-then-replace” for markdown backfill:
  - On STT failure, append the AI question and a short “(transcription pending)” placeholder linked to the saved audio.
  - Reconcile later replaces the placeholder with formatted transcript and records the `.stt.json` atomically.
- Be idempotent and safe:
  - Atomic writes for JSON, never corrupt markdown; serialize body edits within `SessionManager` IO locks.
  - Keep existing WAV deletion policy (only after both MP3 and STT exist) and extend it uniformly.
- Cross-UI parity: CLI and Web surface the same suggestion and use the same reconcile tool.
- Strong breadcrumbs: log structured events on failure/success to aid user and debugging.


## Stages & actions

### Stage: Expand reconcile to cover web media and keep atomic writes
- [ ] Scan `*.wav`, `*.webm`, and `*.ogg` under `--sessions-dir`.
- [ ] Skip if sibling `<seg>.stt.json` exists.
- [ ] On success, write `<seg>.stt.json` atomically (write `*.partial`, then rename).
- [ ] Respect `CONFIG.delete_wav_when_safe` for WAV deletion when both MP3 and STT exist.
- [ ] Events: `reconcile.started`, `reconcile.error`, `reconcile.completed` with counts.
- Acceptance:
  - [ ] Web-only sessions get `.stt.json` files for `browser-*.webm/ogg`.
  - [ ] Existing CLI behaviour unchanged; atomic writes verified in tests.

### Stage: Surface “pending” counts consistently (detect and suggest)
- [ ] CLI: keep startup/resume/finalize pending hints; unify wording; ensure count includes web media.
- [ ] Consider a future `session pending` view; for now rely on `fix stt` output.
- [ ] Web: compute pending count server-side and render a passive banner in the session page if `> 0` with exact suggested command.
- [ ] Web: on upload failure, include the suggested command in error detail (server) so client can display a helpful message.
- Acceptance:
  - [ ] Opening/closing CLI sessions prints correct counts across `.wav/.webm/.ogg`.
  - [ ] Web page shows banner when any segment lacks `.stt.json`.

### Stage: Placeholder insertion on STT failure (CLI)
- [ ] On exception from `manager.record_exchange(...)`, append a placeholder exchange to markdown:
  - Format: keep the AI question; for the user block put “(transcription pending)”.
  - Update frontmatter `audio_file` with the saved audio segment and durations so the link is preserved.
  - Emit `session.exchange.pending` with `session_id`, `response_index`, `wav`, `duration_seconds`, and error type.
- [ ] Keep the existing loop behaviour (re-ask the same question) and the reconcile hint.
- Acceptance:
  - [ ] When STT fails, the session markdown shows the AI question with a placeholder user reply.
  - [ ] Frontmatter and events reflect the segment.

### Stage: Placeholder insertion on STT failure (Web)
- [ ] In upload exception path (server), after saving audio, append the same placeholder exchange and log `session.exchange.pending`.
- [ ] Continue returning an error to client; client shows status with suggested reconcile command.
- Acceptance:
  - [ ] Upload failure still persists audio and adds placeholder to markdown.
  - [ ] Events show `session.exchange.pending` for web.

### Stage: Reconcile markdown backfill (replace placeholders)
- [ ] When a placeholder user block is detected, replace it with formatted transcript from `.stt.json` (or freshly transcribed text), preserving the original AI question.
- [ ] If no placeholder exists (legacy audio without captured question), leave markdown untouched for now and only write `.stt.json` (future enhancement may add minimal transcript-only append logic).
- [ ] Remove any failure sentinel (`<seg>.stt.error.txt`) on success.
- Acceptance:
  - [ ] Placeholders are replaced correctly and idempotently on reruns.
  - [ ] Legacy files are not modified except for JSON creation.

### Stage: Error breadcrumbs and safeguards
- [ ] On STT failure, write `<seg>.stt.error.txt` with error class/message; remove after successful backfill.
- [ ] Make `_persist_raw_transcription` atomic (mirror reconcile’s `.partial` then rename).
- Acceptance:
  - [ ] Error sentinel exists after failure and is removed after backfill.
  - [ ] All STT JSON writes are atomic.

### Stage: Tests
- [ ] Reconcile scans `.wav/.webm/.ogg`, writes JSON atomically, respects delete policy.
- [ ] CLI pending counts include web media; messages match.
- [ ] Web banner renders when pending > 0 (template variable driven); server computes count.
- [ ] Placeholder creation on CLI and Web failure; events logged; no duplicate indices.
- [ ] Placeholder replacement via reconcile; idempotent on reruns; does not touch non-placeholder text.
- [ ] Error sentinel lifecycle (created on failure, removed on success).

### Stage: Docs updates
- [ ] Update `FILE_FORMATS_ORGANISATION.md` to document placeholder semantics and error sentinel.
- [ ] Update `WEB_ARCHITECTURE.md` to note pending-banner and reconcile coverage for `.webm/.ogg`.
- [ ] Update `AUDIO_VOICE_RECOGNITION_WHISPER.md` to mention broadened reconcile and atomic writes.
- [ ] Update `COMMAND_LINE_INTERFACE.md` to remove `journal list` mention and reflect `fix stt`.

### Stage: Create RESILIENCE.md (reference doc)
- [ ] Add `docs/reference/RESILIENCE.md` capturing:
  - Failure modes (STT/network/backend unavailability, device/sample-rate issues).
  - What gets saved when (audio, placeholders, `.stt.json`, sentinels) and why it’s safe.
  - How to recover (detect-and-suggest flow; exact reconcile command; optional flags).
  - Event taxonomy (`stt.retry`, `stt.failed`, `session.exchange.pending`, `reconcile.*`).
  - Operational tips (keeping WAVs, privacy, offline backends, logs location).
- Acceptance:
  - [ ] New doc linked from `PRODUCT_VISION_FEATURES.md`, `WEB_ARCHITECTURE.md`, and `FILE_FORMATS_ORGANISATION.md`.


## Non-goals (for now)

- Automatic background reconcile by default (will remain opt-in/explicit if we add it later).
- Creating markdown entries for legacy audio when the original AI question is unknown (we only backfill JSON now).
- Changing the underlying markdown schema or renaming `wav` → `file` in frontmatter (could be a future cleanup).


## Risks and mitigations

- Race conditions on concurrent writes: continue using the `SessionManager` IO lock for body mutations; keep JSON atomic; keep reconcile single-threaded or small bounded concurrency.
- Mis-replacements in markdown: make placeholder markers unambiguous and include the file segment label to target replacements precisely.
- User confusion on web failures: keep the error simple and provide the exact reconcile command; also keep the passive banner as a consistent hint.
- Performance of directory scans: scope to the configured `--sessions-dir`, filter extensions, and short-circuit when counts reach zero.


## Acceptance criteria (end-to-end)

- On STT failure (CLI or Web): audio persists, a placeholder entry is added under the right AI question, and `session.exchange.pending` is logged.
- The UI (CLI/web) shows the exact suggested command:
  - `uv run --active healthyselfjournal reconcile --sessions-dir '<dir>'`
- Reconcile processes both CLI and Web media, writes `.stt.json` atomically, and replaces placeholders with formatted text.
- Error sentinels are created on failure and removed after success.
- CLI pending counts and the web banner reflect outstanding segments across all supported media types.
- All new tests pass; existing behaviours are unchanged for the happy path.


## Example commands

```bash
uv run --active healthyselfjournal reconcile --sessions-dir '/Users/greg/Dropbox/dev/experim/healthyselfjournal/sessions'

# Optional: backfill pending segments
uv run --active healthyselfjournal fix stt --sessions-dir '/Users/greg/Dropbox/dev/experim/healthyselfjournal/sessions'
```


