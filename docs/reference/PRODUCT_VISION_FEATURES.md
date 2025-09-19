# Voice-Based Reflective Journaling App

## Vision

A command-line journaling app using voice input to lower friction and dialogue-based questioning to maintain engagement while avoiding common pitfalls identified in research.

## See also

- `ARCHITECTURE.md` - System architecture, components, and data flow
- `CLI_COMMANDS.md` - Recording controls and visual feedback
- `DIALOGUE_FLOW.md` - Question sequencing and session management
- `FILE_FORMATS_ORGANISATION.md` - Storage structure for audio and transcripts
- `RESILIENCE.md` - Detect-and-suggest transcription resilience, placeholders, reconcile flow
- `LLM_PROMPT_TEMPLATES.md` - Jinja templates for adaptive questioning
- `PRIVACY.md` - Privacy, local-first data handling, and network boundaries
- `../conversations/250117a_journaling_app_ui_technical_decisions.md` - Technical decisions
- `../conversations/250916a_journaling_app_dialogue_design.md` - Dialogue design rationale
- `../research/JOURNALLING_SCIENTIFIC_EVIDENCE_RESEARCH.md` - Evidence base

## Core Features

- **Voice-first input** via Whisper for stream-of-consciousness expression
- **Text output** from Claude LLM for reflective dialogue
- **Multiple daily sessions** with persistent context across conversations
- **Hybrid adaptive questioning** - Socratic, motivational interviewing, validation based on context


## Current Implementation

- Voice recording with real-time meter and keyboard controls (press any key to stop; ESC cancels; Q saves then quits)
- Immediate WAV persistence; optional background MP3 conversion when `ffmpeg` is available
- OpenAI Whisper STT with retries; raw `.stt.json` responses persisted per segment
- Continuous dialogue loop with Claude; Jinja templates; embedded example questions ("give me a question")
- Recent session summaries loaded with a budget heuristic and included in prompts
- Summary regeneration runs in the background after each exchange; stored in frontmatter
- Resume the most recent session with `--resume`
- Append-only metadata event log at `sessions/events.log`
- Short accidental takes auto-discarded based on duration and voiced-time thresholds
- Detect-and-suggest transcription resilience with markdown placeholders, error sentinels, and CLI/web reconcile hints (see `RESILIENCE.md`)

## Next Steps

- Time‑based break nudge (research‑aligned)
  - Gentle reminder around 20 minutes (configurable; defaults to 20)
  - Non-blocking notice in UI; never auto‑terminate

- Question quality and style
  - Tighter, single‑focus follow‑ups; slightly lower temperature for clarity
  - Small prompt refinements for validation/rumination redirection
  - Keep “give me a question” via embedded examples for variety

- Summary brevity and usefulness
  - Stricter brevity guidance in the prompt; reduce max tokens
  - Heuristic: very short sessions → very short summaries

- Multi‑backend speech recognition (cloud + local)
  - Abstract STT backend with pluggable providers
  - Default: OpenAI (cloud). Optional: local MLX (macOS/Apple Silicon), faster‑whisper (CPU)
  - CLI flags: `--stt-backend`, `--stt-model`, `--language`; graceful fallback
  - OS/machine specific guidance; offline mode when local backends selected
  - Align/reference docs to reflect cloud default and optional local paths

- Audio device selection and diagnostics
  - `--list-devices` to enumerate; `--input-device` to select
  - Log selected device name/rate; basic troubleshooting tips

- Resilience and maintenance
  - Backfill utility to generate missing summaries for older sessions
  - E2E and CLI integration tests; device enumeration tests; short‑take discard tests
  - Optional streaming display of LLM responses (fallback to all‑at‑once)
