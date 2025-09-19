# Transcription Resilience

## Overview

The journaling flow keeps audio safe even when speech-to-text (STT) fails. Instead of blocking the user or silently retrying forever, the app records the failure, stores enough breadcrumbs to recover later, and surfaces the exact `reconcile` command everywhere (CLI banner, web banner, error prompts).

See this document alongside `FILE_FORMATS_ORGANISATION.md`, `CLI_COMMANDS.md`, and `WEB_ARCHITECTURE.md` for end-to-end context.

## Failure handling (CLI & web)

When `SessionManager` cannot obtain a transcript after a recording:

1. **Audio is already persisted** next to the session markdown (`yyMMdd_HHmm/segment.wav`, `browser-XXX.webm`, etc.).
2. **Markdown placeholder** is appended beneath the AI question:
   ```
   (transcription pending – segment yyMMdd_HHmm_03.wav)
   <!-- hsj:pending segment="yyMMdd_HHmm_03.wav" -->
   ```
3. **Frontmatter** updates the matching `audio_file` entry with:
   - `pending: true`
   - `pending_reason`: exception class (e.g., `TimeoutError`)
   - `duration_seconds`, `voiced_seconds`, and if present the `mp3` filename
4. **Sentinel** `<segment>.stt.error.txt` stores a timestamp, error type, and message.
5. **Event log** emits `session.exchange.pending` with session id, response index, source (`cli`/`web`), durations, and error details.
6. **UI hint** shows `uv run --active healthyselfjournal reconcile --sessions-dir '<path>'` so recovery is explicit.

The CLI re-asks the question immediately; the web client reports the error while keeping the session active. No audio is discarded.

## Reconcile flow

`healthyselfjournal reconcile` now scans WAV, WEBM, and OGG recordings under `--sessions-dir`. For each missing `.stt.json` it:

1. Transcribes the audio (respecting `--limit`, `--min-duration`, `--too-short`).
2. Writes the raw payload atomically to `<segment>.stt.json`.
3. Replaces markdown placeholders with formatted transcripts when present.
4. Clears `pending`/`pending_reason` metadata in frontmatter.
5. Removes `<segment>.stt.error.txt`.
6. Optionally deletes the original WAV via `CONFIG.delete_wav_when_safe` once MP3 + STT exist.

Events emitted: `reconcile.started`, `reconcile.placeholder_replaced`, `reconcile.error`, and `reconcile.completed` (with initial/remaining counts).

## Sentinel files

Sentinel files live beside the audio (`segment.stt.error.txt`) and contain:

```
timestamp: 2025-09-19T12:34:56
error_type: TimeoutError
message: Whisper backend timed out after 30s
```

They mark clips that need attention and are removed automatically when `reconcile` succeeds. You can safely delete them once the placeholder is replaced.

## Event taxonomy

Key events involved in resilience:

- `session.exchange.pending` – placeholder recorded; includes audio metadata and error type.
- `session.exchange.recorded` – successful transcript (clears pending state).
- `reconcile.started` / `reconcile.completed` – batch runs, pending counts, backend info.
- `reconcile.placeholder_replaced` – specific segment unblocked.
- `reconcile.error` – failures during backfill; cross-check sentinel files.
- `audio.wav.deleted` – WAV removed after MP3+STT exist (honours `delete_wav_when_safe`).

## Operational tips

- **Keep the command handy**: `uv run --active healthyselfjournal reconcile --sessions-dir '<path>'` is shown in CLI banners, web banner, and error messages.
- **Short/quiet clips**: Reconciling with `--too-short mark` records a stub `.stt.json` and removes the sentinel without forcing transcription.
- **Offline mode**: Local backends let you reconcile without network access once models are installed.
- **Inspecting raw data**: Use the `.stt.json` for backend payloads or the `.stt.error.txt` sentinel for failure context.
- **Retaining WAV**: Disable auto-delete via `--keep-wav` (CLI) or `CONFIG.delete_wav_when_safe = False` if you want lossless archives.

## Related documents

- `FILE_FORMATS_ORGANISATION.md` – Markdown placeholder format, audio metadata, sentinel details.
- `CLI_COMMANDS.md` – `journal list --pending` and `reconcile` options.
- `WEB_ARCHITECTURE.md` – Pending banner and web error messaging.
- `AUDIO_VOICE_RECOGNITION_WHISPER.md` – Backend configuration, atomic write notes.
