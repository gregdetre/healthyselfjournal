# Voice-Based Reflective Journaling App

## Vision

A command-line journaling app using voice input to lower friction and dialogue-based questioning to maintain engagement while avoiding common pitfalls identified in research.

We prioritize evidence-based design and the long-term wellbeing of users and society over engagement.

## See also

- `README.md`
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
- `../research/AUTONOMY_SUPPORT_MI_SDT_FOR_JOURNALING.md` - MI/SDT autonomy‑support guidance
- `../research/ANTI_SYCOPHANCY_AND_PARASOCIAL_RISK_GUARDRAILS.md` - Guardrails to avoid sycophancy/parasocial risks

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

Make it easier to setup and run the first time. Especially improve the local-first functionality, make it more robust, rely on defaults.

Consider how to monitor and respond to potentially dangerous/crisis situations, e.g. suicidal thoughts.

In a separate repo, we have a working mobile app in alpha, but we're struggling to get the Dropbox/iCloud sync working...

