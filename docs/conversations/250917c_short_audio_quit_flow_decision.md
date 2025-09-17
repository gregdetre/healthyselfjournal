# Short‑Audio Quit Flow Decision - 2025-09-17

---
Date: 2025-09-17
Duration: ~10 minutes
Type: Decision-making
Status: Resolved
Related Docs: `docs/reference/RECORDING_CONTROLS.md`, `docs/reference/COMMAND_LINE_INTERFACE.md`, `docs/reference/FILE_FORMATS_ORGANISATION.md`
---

## Context & Goals
The user pressed `q` immediately after seeing an AI question, which captured ~1s of audio and produced a spurious transcription (e.g., "arrecife"). Goal: avoid junk recordings/transcriptions while preserving legitimate one‑word answers.

## Key Background
"As soon as I saw the question I hit Q, so there was probably only about a second's worth of audio... having just a second's worth of audio seems like it's probably always going to be a sign that the user doesn't want to engage."

## Decision
Implement automatic short‑answer discard gating based on duration and voiced time during capture. If exceedingly short and low‑voiced, the take is discarded: no files saved/transcribed; a brief notice is shown. If `Q` was pressed, end the session cleanly after handling the discard.

## Implementation Summary
- Config thresholds added: `short_answer_duration_seconds` (1.2s), `short_answer_voiced_seconds` (0.6s), `voice_rms_dbfs_threshold` (−40 dBFS).
- `audio.py`: tracks `voiced_seconds` via RMS threshold; returns `discarded_short_answer=True` when both duration and voiced time fall under thresholds. Skips MP3 conversion and deletes WAV.
- `session.py`: when `discarded_short_answer`, skip transcription and treat as no exchange; propagate `quit_requested` if Q was pressed.
- `cli.py`: clearer messaging for cancelled vs short‑discarded vs quit.
- Docs updated: recording controls, CLI example, file formats (note on discards).
- Tests updated: new fields in `AudioCaptureResult`; unit test for discard flow and quit propagation.

## Alternatives Considered
- Prompt the user to confirm discarding short takes. Rejected to keep flow frictionless.
- Require a dedicated key for immediate quit (Shift+Q). Deferred; current behavior suffices with gating.

## Open Questions
- Future: combine VAD with ASR confidence/token count for even safer gating.

## Next Steps
- Observe real usage to tune thresholds.
- Optionally expose thresholds via CLI flags.

