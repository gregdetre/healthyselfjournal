# Pending Banner Standardisation - 2025-09-20

## Context & Goals
We saw misleading pending transcription counts (e.g., "444 recording(s) still pending..."), caused by scanning the entire sessions directory. Goal: make hints actionable by scoping them to the active session and keep behavior consistent across CLI/web/desktop.

## Key Background
- "Pending" is defined as media files `.wav/.webm/.ogg` missing a sibling `.stt.json`.
- Short/silent detection is centralized via `should_discard_short_answer(...)` and used in CLI and web; reconcile also analyzes with the same thresholds.
- Web UI already shows per-session pending; CLI showed global counts.

## Decisions Made
- Standardize CLI pending hints to per-session counts using `count_pending_for_session(...)`.
- Keep thresholds and reconcile behavior unchanged; add a brief tip for clearing shorts: `--min-duration 0.6 --too-short mark`.
- Do not add legacy stub mode now; reconcile can backfill or mark/skip shorts when desired.

## Implementation Notes
- CLI now calls `count_pending_for_session(sessions_dir, state.session_id)` at start/resume and finalize.
- Hint copy: mentions "in this session" and includes an optional tip for short/noise cleanup while keeping the canonical command from `reconcile_command_for_dir(...)`.

## Alternatives Considered
- Global scan with filtering of short clips in the banner: deferred to reconcile to avoid duplicating logic.
- Adding `fix stt --mark-legacy` to stub old segments without STT: not implemented yet to keep risk low.

## Next Steps
- If legacy noise persists, consider an explicit `--mark-legacy` mode as an opt-in reconcile path.
- Optionally cap banner to newest N segments in-session if needed for UX.

## Related Work
- Code: `healthyselfjournal/cli_journal_cli.py`
- Utilities: `healthyselfjournal/utils/pending.py`, `healthyselfjournal/utils/audio_utils.py`
- Web: `healthyselfjournal/web/app.py` (already per-session)
