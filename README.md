# Healthyself Journal

**Speak your thoughts. Get good questions. Build healthier patterns.**

A voice-based journaling and self-reflection tool. Just ramble for a while out loud about what's on your mind, and receive a warm, thoughtful, evidence-based question each time in response to help you think things through.

Note: this beta version runs in the command-line and relies on OpenAI & Anthropic models.

**Table of Contents**:
- [What makes this different?](#what-makes-this-different)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Installation \& Setup](#installation--setup)
- [Daily use](#daily-use)
- [Where your journal lives](#where-your-journal-lives)
- [The research behind it](#the-research-behind-it)
- [Support \& Documentation](#support--documentation)
- [AI-first development](#ai-first-development)
- [Technical details](#technical-details)
- [Advanced options](#advanced-options)


## What makes this different?

- 🎙️ **Voice-first**: Start journaling instantly by speaking – no typing, no friction.
- 🧠 **Wise, helpful questions**: Evidence-based prompts adapted from cognitive behavioral therapy, psychology research, mindfulness practice, and famous coaches.
- 🔄 **Keeps you moving**: Gentle redirection when you're spiraling; deeper exploration when you're onto something.
- 📊 **Builds on your history**: Each session connects to previous ones for continuity and growth.
- 🔒 **Privacy choice**: Use Anthropic + OpenAI OR private/local LLM+transcription (in beta), as you prefer. Healthyself doesn't have any backend, storage, or analytics of its own outside your machine whatsoever.
- 🛡️ **Aims to be safe**: Anti-sycophancy, rumination pivots, and clear boundaries; see [`docs/reference/SAFEGUARDING.md`](docs/reference/SAFEGUARDING.md).
- **Open source** ([`LICENSE`](LICENSE)). So you can see & modify the prompts and research for yourself.
- This is assisted reflection. It's not trying to be or replace human conversation or therapy.

## Quick start

```bash
# Recommended: run without installing (uvx)
uvx healthyselfjournal@latest -- init
uvx healthyselfjournal@latest -- journal cli
```

That's it. Recording starts immediately. Press ENTER to stop and get your next question.

Tip for non-technical users: we're using a standard, widely-used tool for installing things. You may need to first install `uv` (which provides `uvx`) and try again:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```


## How it works

1. **Start speaking** – Recording begins immediately when you launch
2. **Press any key to stop** – Your audio is transcribed automatically
3. **Get a thoughtful question** (out loud if you want) – Based on what you shared and your patterns
4. **Keep going or wrap up** – Continue as many times as helpful, or press Q to end with a summary


### Example session

> 🧑 You: "I'm torn between applying for the new role and doubling down on my current project. I'm worried I'll disappoint people either way."
>
> 🤖 AI: "It sounds like you're caught between two paths, and the weight of others' expectations is making the choice harder. What would this decision look like if you set aside the fear of disappointing people and focused on what each option offers you?"
>
> [conversation continues...]

You can see the [full questioning prompt](https://github.com/gregdetre/healthyselfjournal/blob/main/healthyselfjournal/prompts/question.prompt.md.jinja), and the research on which it's based below.

For example, behind the scenes, that LLM response-question above draws on the following approaches: *Socratic questioning + values exploration + gentle challenge to personalization (Three P's pattern)*. Key signals: *moderate emotional intensity, future-focused worry, binary framing, emphasis on others' reactions over personal needs/values*.


## Installation & Setup

### Requirements
- Python 3.10+
- Microphone access (grant permission to your terminal/app)
- Optional cloud keys:
  - `OPENAI_API_KEY` (speech-to-text when using `--stt-backend cloud-openai`)
  - `ANTHROPIC_API_KEY` (LLM questions/summaries; default cloud provider)
- Optional local/offline (in beta):
  - Ollama running (for local LLM via `--llm-model ollama:<model>`)
  - One STT backend installed: `mlx-whisper` (Apple Silicon), `faster-whisper`, or `whispercpp` + `.gguf` model
- Optional: `ffmpeg` on PATH for background MP3 conversion
- Linux only: install audio libs (e.g., `sudo apt install portaudio19-dev libsndfile1`)

See [`docs/reference/AUDIO_VOICE_RECOGNITION_WHISPER.md`](docs/reference/AUDIO_VOICE_RECOGNITION_WHISPER.md) and [`docs/reference/PRIVACY.md`](docs/reference/PRIVACY.md) for details.

### Install

```bash
# Option 1: Run without installing (recommended)
uvx healthyselfjournal@latest -- init

# Option 2: Install with pip (latest release)
pip install -U "healthyselfjournal"
```

### First-time setup

The setup wizard will help you:
- Add your API keys securely
- Choose between Cloud mode (recommended) or Privacy mode (fully offline, but still in beta)
- Pick where to save your journal sessions

```bash
# Recommended
uvx healthyselfjournal@latest -- init
```


## Daily use

```bash
# Start a new session
uvx healthyselfjournal@latest -- journal cli

# Continue your last session
uvx healthyselfjournal@latest -- journal cli --resume
```

Use `--sessions-dir` to decide where to store them - otherwise it defaults to `./sessions/`.


### Insights (beta)

Generate high-level reflective insights every so often spanning recent conversations, saved under `sessions/insights/`:

```bash
# List existing insights
uvx healthyselfjournal -- insight list --sessions-dir ./sessions

# Generate multiple insights in a single file
uvx healthyselfjournal -- insight generate --sessions-dir ./sessions --count 3
```

Background and design: see [`docs/research/INSIGHTS_RESEARCH_OVERVIEW.md`](docs/research/INSIGHTS_RESEARCH_OVERVIEW.md).

### Controls
- **ENTER**: Stop recording and get your next question
- **ESC**: Cancel the current recording (discard it)
- **Q**: Save and quit after this response, and generate summary
- **Ctrl-c**: Quit immediately

See [`docs/reference/RECORDING_CONTROLS.md`](docs/reference/RECORDING_CONTROLS.md) for full control details.

### Privacy options

**Cloud mode** (default): Uses OpenAI for transcription/voice and Anthropic Claude for questions. Need to provide your own keys. Best accuracy and response quality. But Moloch gets to hear your darkest secrets.

**Privacy mode**: Everything stays on your device. Requires [Ollama](https://ollama.ai) for local AI and choosing a local transcription option. See [`docs/reference/PRIVACY.md`](docs/reference/PRIVACY.md) for details. More fiddly to setup, requires downloading multi-gigabyte models, requires recent-ish computer, maybe slower, and not quite such good questions.

Local speech-to-text (STT) options (install only what you need):
```bash
# Apple Silicon (no native builds): MLX Whisper CLI
pipx install mlx-whisper
uvx -p 3.12 healthyselfjournal@latest -- journal cli --stt-backend local-mlx --stt-model large-v2

# Cross‑platform, no FFmpeg build: whisper.cpp
pipx install whispercpp
uvx -p 3.12 healthyselfjournal@latest -- journal cli --stt-backend local-whispercpp --stt-model ~/models/ggml-base.en.gguf

# Faster‑whisper (requires FFmpeg + pkg-config for PyAV)
brew install pkg-config ffmpeg
uvx -p 3.12 healthyselfjournal@latest -- journal cli --stt-backend local-faster --stt-model large-v2
```


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

You own all your data. Export it, back it up, or delete it anytime. Open it in any text editor. It doesn't get sent to any storage or analytics, aside from the cloud AI providers if you choose to use them.

See [`docs/reference/FILE_FORMATS_ORGANISATION.md`](docs/reference/FILE_FORMATS_ORGANISATION.md) for detailed file formats and directory organisation.


## The research behind it

Healthyself journal is built primarily on evidence-based psychological research (and a few other tips that we have found helpful), integrating a few dozen documented therapeutic and coaching frameworks:

### Core Therapeutic Foundations
- **Cognitive Behavioral Therapy (CBT)**: [Socratic questioning](docs/research/SOCRATIC_QUESTIONING_TECHNIQUES.md) to identify and reframe thought patterns (meta-analyses show d=0.73 effect size)
- **Motivational Interviewing**: Amplifying "change talk" and intrinsic motivation ([evidence and autonomy support](docs/research/AUTONOMY_SUPPORT_MI_SDT_FOR_JOURNALING.md))
- **Explanatory Style (Seligman's 3 P's)**: Challenging permanence, pervasiveness, and personalization in negative thinking ([overview](docs/research/EXPLANATORY_STYLE_THREE_PS.md))
- **Clean Language (David Grove)**: Using your exact words and metaphors to maintain authenticity and avoid therapist contamination ([techniques](docs/research/CLEAN_LANGUAGE_TECHNIQUES.md)).

### Anti-Rumination & Safety Features
- **Structured vs. Destructive Rumination**: Evidence-based detection of maladaptive thought loops ([guide](docs/research/STRUCTURED_REFLECTION_VS_RUMINATION.md))
- **Self-Distancing Techniques**: Third-person perspective and temporal distancing ([evidence](docs/research/SELF_DISTANCING_TECHNIQUES.md))
- **Concrete vs. Abstract Processing**: Redirecting to specific, actionable thoughts when stuck
- **Session Timing Optimization**: 15-20 minute sweet spot to prevent rumination ([summary](docs/research/OPTIMAL_SESSION_TIMING.md))

### Narrative & Meaning-Making
- **Redemptive Narrative Construction (McAdams)**: Guiding from contamination to growth narratives ([overview](docs/research/REDEMPTIVE_NARRATIVE_CONSTRUCTION.md))
- **Implementation Intentions**: "When-then" planning for 2-3x better habit formation ([evidence](docs/research/IMPLEMENTATION_INTENTIONS_HABITS.md))
- **Cognitive-Emotional Integration**: Balanced processing outperforms emotion-only expression ([summary](docs/research/COGNITIVE_EMOTIONAL_INTEGRATION.md))

### Mindfulness & Contemplative Practices
- **Plum Village Tradition**: Mindful reflection and present-moment awareness ([practice](docs/research/MINDFUL_REFLECTION_PLUM_VILLAGE.md))
- **Beginning Anew Practice**: Four-part framework for relationship and self-compassion ([guide](docs/research/BEGINNING_ANEW_PRACTICE.md))
- **Body Awareness Integration**: Somatic grounding when caught in mental loops

### Coaching Methodologies
- **GROW Model**: Goal-Reality-Options-Will framework with strong evidence base ([coaching frameworks evidence](docs/research/COACHING_FRAMEWORKS_EVIDENCE.md))
- **Solution-Focused Brief Therapy**: Future-oriented questions emphasizing strengths ([coaching frameworks evidence](docs/research/COACHING_FRAMEWORKS_EVIDENCE.md))
- **Values Clarification (ACT)**: Connecting actions to core personal values ([values questions](docs/research/VALUES_SELF_DISCOVERY_QUESTIONS.md))

### Expert Practitioner Wisdom
Questions inspired by renowned coaches and researchers ([famous coach questions](docs/research/FAMOUS_COACH_QUESTIONS.md)):
- Tim Ferriss' fear-setting and simplification frameworks
- Jerry Colonna's radical self-inquiry
- Martha Beck's body compass methodology
- Tony Robbins' reframing techniques
- Arthur Brooks' failure integration

### Cultural & Individual Adaptation
- **Cultural Sensitivity**: Avoiding Western-centric assumptions about gratitude and individual achievement (see [`GRATITUDE_PRACTICE_OPTIMIZATION.md`](docs/research/GRATITUDE_PRACTICE_OPTIMIZATION.md))
- **Personalization**: Adapting to user patterns, chronotype, and emotional states
- **Developmental Considerations**: Age-appropriate approaches based on psychological development

The system continuously analyzes your responses for emotional intensity, thought patterns, topic persistence, exhaustion signals, and readiness for change, adapting its questioning strategy based on session phase and your current needs.

For an overview of all 30+ research areas and methodologies, see [`docs/research/RESEARCH_TOPICS.md`](docs/research/RESEARCH_TOPICS.md) and [`docs/reference/SCIENTIFIC_RESEARCH_EVIDENCE_PRINCIPLES.md`](docs/reference/SCIENTIFIC_RESEARCH_EVIDENCE_PRINCIPLES.md).


## Support & Documentation

- **Issues or questions**: [GitHub Issues](https://github.com/gregdetre/healthyselfjournal/issues)
- **Full documentation**: See the [`docs/`](docs/) folder
- **Contributing**: Contributions welcome! See [`CONTRIBUTING.md`](CONTRIBUTING.md)

### For more info on command-line options
```bash
healthyselfjournal --help

healthyselfjournal journal cli --help
```

See also: [`docs/reference/COMMAND_LINE_INTERFACE.md`](docs/reference/COMMAND_LINE_INTERFACE.md) and [`docs/reference/CLI_COMMANDS.md`](docs/reference/CLI_COMMANDS.md).


## AI-first development

This app was entirely written by AI, with human guidance, following the approaches in:

- [AI-first development principles](https://www.makingdatamistakes.com/ai-first-development/)
- [gjdutils docs](https://github.com/gregdetre/gjdutils/tree/main/docs)


## Technical details

For developers and technical users:
- Built with Python, using Typer for the CLI.
- OPTIONAL FastHTML for the web interface and PyWebView for desktop (ALPHA: these don't work very well)
- Transcription via OpenAI Whisper API (or local alternatives)
- Questions generated by Anthropic Claude (or local Ollama models)
- See [`docs/reference/ARCHITECTURE.md`](docs/reference/ARCHITECTURE.md) for system design
- See [`docs/reference/SETUP_DEV.md`](docs/reference/SETUP_DEV.md) for development setup


## Advanced options

### Desktop app (in alpha, probably won't work)
```bash
healthyselfjournal journal desktop --voice-mode
```
