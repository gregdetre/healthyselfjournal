# Background Processing

## Introduction

Non-blocking tasks improve responsiveness by moving work off the interactive loop. This document explains our background processing approach for summaries and outlines patterns for safely extending it.

## See also

- `CONVERSATION_SUMMARIES.md` — What summaries are and how they’re used
- `FILE_FORMATS_ORGANISATION.md` — Where transcript data and frontmatter live
- `DIALOGUE_FLOW.md` — The interactive Q&A loop

## Principles, key decisions

- Keep the interactive loop snappy; avoid blocking on non-essential work
- Serialize all file writes within the process using a single lock
- Snapshot inputs for background LLM calls; reload the document before writing results
- Prefer a single-thread worker (executor with `max_workers=1`) per process for simplicity
- Gracefully shut down workers on session completion to flush pending tasks

## Current state

- Summary regeneration is scheduled after each exchange and executed in a background worker
- Writes are protected by an in-process lock to prevent clobbering
- The summary may lag briefly behind the latest exchange; resilience preserved

## Patterns and how-tos

### Serialize writes

- Use a single `threading.Lock` to guard calls that update transcript frontmatter/body.
- Always reload the transcript immediately before writing results computed in the background.

### Schedule background work

- Use a `ThreadPoolExecutor(max_workers=1)` to run tasks.
- Capture a snapshot of required inputs (e.g., transcript body, recent summaries) before dispatch.

### Shutdown

- On session end, call `shutdown(wait=True)` on the executor to ensure all tasks complete.

## Gotchas

- Avoid reading and writing the same `TranscriptDocument` instance across threads; reload under the lock before write.
- If future background features are added, reuse the same lock to keep serialization simple.

## Planned future work

- Debounced/coalesced scheduling for bursts of updates
- Optional metrics around background task queueing and latency


