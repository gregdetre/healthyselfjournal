# Sessions

## Overview
Tools to inspect and manage sessions created by CLI and web interfaces.

## Listing sessions

```bash
uvx healthyselfjournal -- session list [--sessions-dir PATH] [--nchars N]
```

- Shows each session by `.md` filename stem with a summary snippet from frontmatter. Use `--nchars` to limit characters (full summary when omitted).

## Related utilities

- `summarise list|backfill|regenerate` – Manage summaries stored in frontmatter.
- `reconcile` – Backfill STT JSON for recordings missing transcriptions.
- `merge` – Merge two sessions, move assets, append Q&A, regenerate summary.

See `FILE_FORMATS_ORGANISATION.md` for layout details of markdown and audio artefacts.

