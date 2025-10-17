# Healthy Self Journal

**Speak your thoughts. Get better questions. Build healthier patterns.**

A voice-based journaling tool that makes self-reflection as easy as thinking out loud. No typing, no blank page anxiety – just speak naturally and receive thoughtful, evidence-based questions that guide you toward clarity without getting stuck in unhelpful thought loops.

## What makes this different

🎙️ **Voice-first**: Start journaling instantly by speaking – no typing, no friction
🧠 **Wise, helpful questions**: Evidence-based prompts adapted from cognitive behavioral therapy, psychology research, mindfulness practice, and famous coaches.
🔄 **Keeps you moving**: Gentle redirection when you're spiraling; deeper exploration when you're onto something
📊 **Builds on your history**: Each session connects to previous ones for continuity and growth
🔒 **Privacy choice**: Use private/local LLM+transcription, or Anthropic + OpenAI, as you prefer.

## Quick start

```bash
# Recommended: run without installing (uvx)
uvx healthyselfjournal -- init
uvx healthyselfjournal -- journal cli
```

That's it. Recording starts immediately. Press any key to stop and get your next question.

## Who this is for

If you've tried journaling but found it hard to maintain, or if you want to process your thoughts without getting stuck in rumination, this tool is designed for you. It combines the ease of voice recording with the structure of guided reflection, using techniques validated by research to help you develop healthier thinking patterns.

## How it works

1. **Start speaking** – Recording begins immediately when you launch
2. **Press any key to stop** – Your audio is transcribed automatically
3. **Get a thoughtful question** – Based on what you shared and your patterns
4. **Keep going or wrap up** – Continue as long as helpful, or press Q to end with a summary

### Example session

```
You: "I'm torn between applying for the new role and doubling down on my current
     project. I'm worried I'll disappoint people either way."

AI: "I can hear the weight of wanting to honor both paths and the people involved.
    When you imagine disappointing people - what's the hardest part about that for you?"

You: "I guess I hate feeling like I've let someone down who believed in me.
     Especially my current team. They took a chance on me when I didn't have much
     experience in this area. And we've built something together, you know? But then
     this other opportunity... it's exactly the kind of challenge I've been wanting.
     More leadership, bigger scope. I keep thinking about five years from now - will
     I regret not trying? But also these are real relationships, not just work..."

AI: "There's something powerful in how you describe building something together with
    your team. What does it mean to you that they 'took a chance' on you?"

[conversation continues...]
```

## Installation & Setup

### Requirements
- Python 3.10 or newer
- Optional: an API key from OpenAI (for voice transcription) and Anthropic (for questions)
- Optional: ffmpeg for audio compression

### Install

```bash
# Option 1: Run without installing (recommended)
uvx healthyselfjournal -- init

# Option 2: Install with pip
pip install healthyselfjournal
```

### First-time setup

The setup wizard will help you:
- Add your API keys securely
- Choose between Cloud mode (recommended) or Privacy mode (fully offline)
- Pick where to save your journal sessions

```bash
# Recommended
uvx healthyselfjournal -- init

# Or if installed via pip
healthyselfjournal init
```

## Daily use

```bash
# Start a new session
healthyselfjournal journal cli

# Continue your last session
healthyselfjournal journal cli --resume

# Use the web interface instead
healthyselfjournal journal web
```

### Insights (v1)

Generate reflective insights based on your past summaries and recent transcripts, saved under `sessions/insights/`:

```bash
# List existing insights
healthyselfjournal insights list --sessions-dir ./sessions

# Generate multiple insights in a single file
healthyselfjournal insights generate --sessions-dir ./sessions --count 3
```

### Controls
- **Any key**: Stop recording and get your next question
- **ESC**: Cancel the current recording (discard it)
- **Q**: Save and quit after this response

### Privacy options

**Cloud mode** (default): Uses OpenAI for transcription and Anthropic Claude for questions. Best accuracy and response quality.

**Privacy mode**: Everything stays on your device. Requires [Ollama](https://ollama.ai) for local AI and choosing a local transcription option. See `docs/reference/PRIVACY.md` for details.

## Where your journal lives

Your sessions are saved as markdown files with audio recordings:
```
sessions/
├── 250919_143022.md          # Today's afternoon session
├── 250919_143022/
│   ├── 250919_143022_01.wav  # Your voice recordings
│   └── 250919_143022_02.wav
└── events.log                 # Activity log
```

You own all your data. Export it, back it up, or delete it anytime.

## The research behind it

This tool is built on decades of evidence-based psychological research, integrating over 30 documented therapeutic and coaching frameworks:

### Core Therapeutic Foundations
- **Cognitive Behavioral Therapy (CBT)**: Socratic questioning to identify and reframe thought patterns (meta-analyses show d=0.73 effect size)
- **Motivational Interviewing**: Amplifying "change talk" and intrinsic motivation (70+ RCTs supporting effectiveness)
- **Clean Language (David Grove)**: Using your exact words and metaphors to maintain authenticity and avoid therapist contamination
- **Explanatory Style (Seligman's 3 P's)**: Challenging permanence, pervasiveness, and personalization in negative thinking

### Anti-Rumination & Safety Features
- **Structured vs. Destructive Rumination**: Evidence-based detection of maladaptive thought loops
- **Self-Distancing Techniques**: Third-person perspective and temporal distancing (strong neurological evidence)
- **Concrete vs. Abstract Processing**: Redirecting to specific, actionable thoughts when stuck
- **Session Timing Optimization**: 15-20 minute sweet spot to prevent rumination (based on expressive writing research)

### Narrative & Meaning-Making
- **Redemptive Narrative Construction (McAdams)**: Guiding from contamination to growth narratives
- **Implementation Intentions**: "When-then" planning for 2-3x better habit formation
- **Cognitive-Emotional Integration**: Balanced processing outperforms emotion-only expression

### Mindfulness & Contemplative Practices
- **Plum Village Tradition**: Mindful reflection and present-moment awareness
- **Beginning Anew Practice**: Four-part framework for relationship and self-compassion
- **Body Awareness Integration**: Somatic grounding when caught in mental loops

### Coaching Methodologies
- **GROW Model**: Goal-Reality-Options-Will framework with strong evidence base
- **Solution-Focused Brief Therapy**: Future-oriented questions emphasizing strengths
- **Values Clarification (ACT)**: Connecting actions to core personal values

### Expert Practitioner Wisdom
Questions inspired by renowned coaches and researchers:
- Tim Ferriss' fear-setting and simplification frameworks
- Jerry Colonna's radical self-inquiry
- Martha Beck's body compass methodology
- Tony Robbins' reframing techniques
- Arthur Brooks' failure integration

### Cultural & Individual Adaptation
- **Cultural Sensitivity**: Avoiding Western-centric assumptions about gratitude and individual achievement
- **Personalization**: Adapting to user patterns, chronotype, and emotional states
- **Developmental Considerations**: Age-appropriate approaches based on psychological development

The system continuously analyzes your responses for emotional intensity, thought patterns, topic persistence, exhaustion signals, and readiness for change, adapting its questioning strategy based on session phase and your current needs.

For an overview of all 30+ research areas and methodologies, see `docs/research/RESEARCH_TOPICS.md` and `docs/reference/SCIENTIFIC_RESEARCH_EVIDENCE_PRINCIPLES.md`.

## Advanced options

### Desktop app
```bash
healthyselfjournal journal desktop --voice-mode
```

### Different AI models
```bash
# Use a local model (requires Ollama)
healthyselfjournal journal cli --llm-model ollama:gemma3:27b-instruct-q4_K_M
```

### Custom session location
```bash
healthyselfjournal journal cli --sessions-dir ~/Documents/journal
```

## Support & Documentation

- **Issues or questions**: [GitHub Issues](https://github.com/anthropics/healthyselfjournal/issues)
- **Full documentation**: See the `docs/` folder
- **Contributing**: Contributions welcome! See `CONTRIBUTING.md`

## Technical details

For developers and technical users:
- Built with Python, using FastHTML for the web interface and PyWebView for desktop
- Transcription via OpenAI Whisper API (or local alternatives)
- Questions generated by Anthropic Claude (or local Ollama models)
- Either everything runs locally (your data never leaves your device), or choose cloud services
- See `docs/reference/ARCHITECTURE.md` for system design
- See `AGENTS.md` for development setup
