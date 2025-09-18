# Dialogue Flow

## Overview

Continuous Q&A loop with LLM-generated follow-up questions after each response.

## See also

- `RECORDING_CONTROLS.md` - How to end sessions
- `OPENING_QUESTIONS.md` - How sessions start
- `LLM_PROMPT_TEMPLATES.md` - Question generation logic
- `../conversations/250916a_journaling_app_dialogue_design.md` - Adaptive model details

## Session Flow

1. Start with default opener or custom question
2. User records voice response
3. Transcribe and save
4. LLM generates contextual follow-up question (streamed by default for lower perceived latency)
5. Display question as text (progressively when streaming is enabled)
6. Loop until user presses Q to quit

## Question Generation

LLM sees:
- Current session transcript
- Recent session summaries
- Embedded example questions in prompt for variety (referenced when user says "give me a question")

## Session Limits

- Research suggests 15-20 minutes to prevent rumination
- Future: Gentle nudge after time/exchange threshold