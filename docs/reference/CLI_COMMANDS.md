# CLI Commands

## Overview
Single source for command discovery. See linked pages for detailed flags.

## Primary commands

- `journal cli` – Terminal-based recording. See `CLI_RECORDING_INTERFACE.md`.
- `journal web` – Launch local browser interface. See `WEB_RECORDING_INTERFACE.md`.
- `journal list --pending` – Show sessions with outstanding transcription segments and the reconcile hint.
- `session list` – Show sessions with summary snippets. See `SESSIONS.md`.
- `summarise ...` – List/backfill/regenerate summaries.
- `reconcile` – Backfill missing STT for saved WAV/webm/ogg files, replace markdown placeholders, and remove error sentinels.
- `merge` – Merge two sessions into the earlier one.
- `init` – Setup wizard for first-time configuration.

## Examples

```bash
# Start CLI journaling
uvx healthyselfjournal -- journal cli --voice-mode

# Start web interface on a different port, resume latest session
uvx healthyselfjournal -- journal web --port 8888 --resume

# List sessions in a custom directory (first 200 chars)
uvx healthyselfjournal -- session list --sessions-dir ./sessions --nchars 200

# Summaries
uvx healthyselfjournal -- summarise list --missing-only
uvx healthyselfjournal -- summarise backfill --limit 10
uvx healthyselfjournal -- summarise regenerate 250918_0119.md
```

## See also

- `CLI_RECORDING_INTERFACE.md`
- `WEB_RECORDING_INTERFACE.md`
- `SESSIONS.md`
