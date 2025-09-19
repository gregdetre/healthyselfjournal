# Naming the Voice Journaling App - January 18, 2025

## Context & Goals
Exploring a better name for "examinedlifejournal" - seeking something more memorable, shorter, and aligned with the app's evidence-based wellbeing focus.

## Key Background
The app is a command-line, voice-first reflective journaling tool that:
- Uses voice input via Whisper for stream-of-consciousness expression
- Provides text output from Claude LLM for reflective dialogue
- Employs hybrid adaptive questioning (Socratic, motivational interviewing, validation)
- Is grounded in research and evidence-based approaches to wellbeing

User emphasized: "I'm interested as well in the fact that it's evidence-based. It's about wellbeing."

## Main Discussion

### Initial Naming Directions
Started by identifying key aspects to emphasize:
- Voice/speaking component (oral, spoken, voiced, echo, whisper)
- Reflection/introspection (mirror, reflect, ponder, muse)
- Dialogue nature (converse, exchange, prompt, query)
- Daily practice (ritual, habit, flow)

Explored categories including simple & direct names (Reflect, Voiced, Muse), compound concepts (VoiceReflect, SpeakThink, WhisperLog), and metaphorical options (Echo, Mirror, Resonance).

### Evidence-Based Wellbeing Pivot
When the evidence-based wellbeing angle was introduced, new directions emerged:
- Clinical/research-inspired: WellVoice, VoiceWell, SpeakWell, WellSpoken
- Evidence/science hints: ValidVoice, ProvenPath, EvidenceFlow
- Wellbeing-forward: Flourish, Thrive, Bloom, Balance
- Subtle acronyms: VERA (Voice-Enabled Reflective Analysis), WISE (Wellbeing through Introspective Speech Expression)

## Decision Made
User selected: **"Well Spoken Journal"**

"I like Well Spoken Journal. I think one advantage of that is we'd have a chance of it being unique as a combination of words"

The name works on multiple levels:
- "Well Spoken" creates a dual meaning: articulate/eloquent AND wellness through speaking
- "Journal" grounds it as a journaling practice
- The three-word combination offers uniqueness in search and branding

## Implementation Considerations
For the codebase/package name:
- `wellspokenjournal` (all lowercase, Python convention)
- `well-spoken-journal` (hyphenated)
- `wsjournal` (abbreviated to avoid WSJ/Wall Street Journal conflict)

CLI command options:
- `wsj` (short but potential conflict)
- `wellspoken` (clear, unique)
- `wspeak` (action-oriented)
- `wsjournal` (safest)

Renaming would involve:
- Package name in `pyproject.toml`
- Module directory name
- CLI entry point
- Import statements throughout
- Documentation references
- Git repository name (if changing)

## Next Steps
- Decide on exact package naming convention
- Choose CLI command name
- Plan systematic renaming across codebase

## Related Work
- See `docs/reference/PRODUCT_VISION_FEATURES.md` for product vision
- Original implementation: `examinedlifejournal` package

UPDATE: we ended up going with `healthyselfjournal` (in part because it's both `healthy self` vs `heal thyself`).
