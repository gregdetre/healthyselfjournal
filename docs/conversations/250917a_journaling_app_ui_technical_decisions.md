---
Date: 2025-01-17
Duration: ~20 minutes
Type: Decision-making, Technical Design
Status: Active
Related Docs:
- `docs/reference/PRODUCT_VISION_FEATURES.md`
- `docs/research/JOURNALLING_SCIENTIFIC_EVIDENCE_RESEARCH.md`
- `docs/conversations/250916a_journaling_app_dialogue_design.md`
---

# Journaling App UI/Technical Decisions - January 17, 2025

## Context & Goals

Building on the January 16 dialogue design conversation, this session focused on concrete technical and UI decisions for implementing the voice-based journaling app. The goal was to move from conceptual design to specific implementation choices for V1.

## Key Background

User's core requirement: "this is going to be on the command line. So I was thinking ideally that it would be recording until I press a key."

Additional context from user:
- "Let's not worry about the issues that happen when we have loads and loads of summaries yet. Let's just get to a V1 quickly, and make it engaging and helpful."
- Planning to use Anthropic Claude for the conversational LLM
- Priority on preventing data loss through immediate persistence

## Main Discussion

### Recording Interface

User decided on press-to-record with keypress to stop, providing natural control without timing pressure. For visual feedback: "Minimal indicator would be ok. It would be really nice to have some basic visual feedback so that I know for sure it's recording. Best of all would be some kind of waveform so I get a sense of volume, but that might not be easy to do on command line."

Solution identified: Unicode block volume meter using Python's `rich` library for real-time updates.

### Response Delivery

Starting with "all at once" display for simplicity, with intention to add streaming "word by word" later. User confirmed preference for Claude API over OpenAI.

### Session Flow

User specified: "I think after each recording from me, the LLM should come up with another question." Session termination through keyboard controls rather than voice commands.

### Data Persistence

Critical requirement from user: "To avoid any kind of loss (e.g. if there's a powercut), it should always record the audio to an mp3 file, and then record the transcription to a markdown file, so that nothing's ever lost."

### Session Controls

User decided on dual-key system:
- ESC to cancel transcribing ("oops, ignore that")
- Q to transcribe-and-quit ("transcribe this last thought and we're done")

### File Organization

User specified flat directory structure: "Yeah, let's do everything in a flat directory. Use `yyMMdd_HHmm` as the filename timestamp, common to both .md and .mp3 next to each other"

Markdown format: "Each question from the LLM will be a `##` H2 heading prepended with `AI Q: ` and then the question, and then the user response will be the text in that heading."

### Summary Generation

User requirement: "Generate/regenerate the summary after each question and answer, i.e. update/overwrite it each time. Maybe we'll also need a kind of backfill process just in case any of them are missing."

### Prompt System

User specified: "We'll probably use Jinja for our prompt templates, so we can include variables and if statements."

Default same opener with flexibility: "So we'll write the prompt to default to starting with the same opener. Then the user can always say, 'Give me a question.' So we'll probably need to include a bank of example questions that the LLM can use for inspiration (along with previous summaries, etc)."

### Context Management

User confirmed: "The LLM should always see the session so far and some recent summaries."

## Decisions Made

### Technical Stack
- **Voice Recording**: Command-line with press-to-start, keypress-to-stop
- **Visual Feedback**: Unicode block volume meter via Python `rich` library
- **Transcription**: Whisper API
- **LLM**: Anthropic Claude (not OpenAI)
- **Prompt Templates**: Jinja2 for flexibility
- **Response Display**: All-at-once initially, streaming later

### File Structure
- **Directory**: Flat structure, all files in one directory
- **Naming**: `yyMMdd_HHmm` format for both .mp3 and .md pairs
- **Markdown Format**:
  - H2 headings with "AI Q: " prefix for questions
  - User responses as text under each heading
  - Summary in frontmatter, regenerated after each Q&A

### User Controls
- **Recording**: Press any key to start, press any key to stop
- **Session End**: Q to transcribe-and-quit, ESC to cancel without transcribing
- **Default Flow**: Continuous Q&A until user explicitly ends

### Context Strategy
- Current session + recent summaries always included
- Summary regenerated after each exchange for crash resilience
- Backfill process for missing summaries

## Open Questions

- Specific number of recent summaries to include in context
- Exact format for question bank and variety mechanisms
- Implementation of backfill process for missing summaries
- Audio format details (mp3 encoding parameters)

## Sources & References

### Previous Conversations
- **Dialogue Design Discussion**: `docs/conversations/250916a_journaling_app_dialogue_design.md` - Established hybrid adaptive model and questioning approach
- **Research Summary**: `docs/research/JOURNALLING_SCIENTIFIC_EVIDENCE_RESEARCH.md` - Evidence base for design decisions
- **Product Vision**: `docs/reference/PRODUCT_VISION_FEATURES.md` - Original concept

### Technical Resources
- Python `rich` library for CLI visual feedback
- Whisper API for transcription
- Anthropic Claude API for dialogue
- Jinja2 for prompt templating