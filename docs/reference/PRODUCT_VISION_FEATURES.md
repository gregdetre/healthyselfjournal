# Voice-Based Reflective Journaling App

## Vision

A command-line journaling app using voice input to lower friction and dialogue-based questioning to maintain engagement while avoiding common pitfalls identified in research.

## See also

- `COMMAND_LINE_INTERFACE.md` - Recording controls and visual feedback
- `DIALOGUE_FLOW.md` - Question sequencing and session management
- `FILE_FORMATS_ORGANISATION.md` - Storage structure for audio and transcripts
- `LLM_PROMPT_TEMPLATES.md` - Jinja templates for adaptive questioning
- `../conversations/250117a_journaling_app_ui_technical_decisions.md` - Technical decisions
- `../conversations/250916a_journaling_app_dialogue_design.md` - Dialogue design rationale
- `../research/JOURNALLING_SCIENTIFIC_EVIDENCE_RESEARCH.md` - Evidence base

## Core Features

- **Voice-first input** via Whisper for stream-of-consciousness expression
- **Text output** from Claude LLM for reflective dialogue
- **Multiple daily sessions** with persistent context across conversations
- **Hybrid adaptive questioning** - Socratic, motivational interviewing, validation based on context

## Key Decisions

- Python command-line implementation for V1
- Anthropic Claude for dialogue generation
- Immediate persistence to prevent data loss
- Transparent mode switching to avoid performative positivity
