# Examined Life Journal

Voice-first reflective journaling for the command line. Speak freely; your words are captured, transcribed with Whisper, and met with concise, evidence‑informed follow‑up questions from Claude to keep you moving forward without drifting into rumination.

See also: `AGENTS.md`

## Why this exists (vision)

- Lower friction to start: voice-first input encourages natural expression
- Keep momentum: adaptive dialogue vs static prompts
- Avoid harmful patterns: gentle redirection away from unproductive rumination
- Build continuity: multiple daily sessions with brief summaries for context

Core feature set (implemented today):
- Live audio recording with real-time meter and simple controls (any key to stop; `ESC` cancels; `Q` saves then quits)
- Immediate WAV persistence; background MP3 conversion when `ffmpeg` is available
- OpenAI Whisper STT with retries; raw `.stt.json` stored per take
- Claude-generated follow‑ups using Jinja prompts with embedded example questions
- Recent session summaries loaded under a budget and fed into prompts
- Background summary regeneration written to YAML frontmatter
- Resume the latest session with `--resume`; change location with `--sessions-dir`

## Snapshot: What it feels like

```text
(experim__examinedlifejournal) experim/examinedlifejournal git:(main) ✗  uv run --active examinedlifejournal journal --resume
╭───────────────────────────────── Examined Life Journal ──────────────────────────────────╮
│ Recording starts immediately.                                                            │
│ Press any key to stop. ESC cancels the current take; Q saves then ends after this entry. │
╰──────────────────────────────────────────────────────────────────────────────────────────╯
AI: I can hear you weighing a few options. Before we try to solve anything, what's the
one part of this that matters most to you right now?
Recording started. Press any key to stop (ESC cancels, Q quits after this response).
Recording  [██░░░░░░░░░░░░░░] Press any key to stop (ESC cancels, Q quits)
Saved WAV → 250917_101234_01.wav (1:48); MP3 conversion queued.
╭──────────────────────────────────────────────────── You ────────────────────────────────────────────────────╮
│ I’m torn between applying for the new role and doubling down on my current project. I’m worried I’ll        │
│ disappoint people either way.                                                                               │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────────────────────────────── Next Question ───────────────────────────────────────────────╮
│ It sounds important to honor both commitment and growth. In the next week, what would a small step look     │
│ like that tests the new role idea without burning bridges?                                                  │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
AI: You mentioned wanting to grow without letting people down. What’s a 30‑minute experiment you could try
this week to explore the new role while keeping current work healthy?
Recording started. Press any key to stop (ESC cancels, Q quits after this response).
Recording  [████░░░░░░░░░░░░] Press any key to stop (ESC cancels, Q quits)
Saved WAV → 250917_101234_02.wav (0:56); MP3 conversion queued.
╭──────────────────────────────────────────────────── You ────────────────────────────────────────────────────╮
│ I could draft a short proposal for the role and ask for feedback from one mentor. That feels doable.        │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Quit requested. Ending session after summary update.
╭──────── Session Complete ─────────╮
│ Session saved to 250917_101234.md │
╰───────────────────────────────────╯
```

## Prerequisites

- Python 3.12+
- `ffmpeg` on `PATH` (optional, for background MP3 conversion)
- Environment variables (set according to the backends you use):
  - `OPENAI_API_KEY` – required for OpenAI speech-to-text or TTS features.
  - `ANTHROPIC_API_KEY` – required only when using `anthropic:*` models for questions/summaries (cloud default).
  - `OLLAMA_BASE_URL` – optional override when running local `ollama:*` models (defaults to `http://localhost:11434`).

## Setup

Prefer the external virtualenv workflow in `docs/reference/SETUP.md`:

```bash
source /Users/greg/.venvs/experim__examinedlifejournal/bin/activate
uv sync --active
```

## Usage

### Getting started (PyPI users)

Use either pip or uvx:

```bash
# pip
pip install examinedlifejournal
examinedlifejournal init
examinedlifejournal journal

# or uvx (no install needed)
uvx examinedlifejournal -- init
uvx examinedlifejournal -- journal
```

To run the dialogue loop fully offline, install Ollama + Gemma (see `docs/reference/OLLAMA_GEMMA_DEPLOYMENT_GUIDE.md`), ensure the daemon is running, then start the CLI with:

```bash
examinedlifejournal journal --llm-model ollama:gemma3:27b-instruct-q4_K_M
```

Notes:
- The init wizard helps you add keys and pick Cloud vs Privacy mode.
- Default sessions directory is `./sessions` in your current folder.

1. Activate the project virtualenv:
   ```bash
   source /Users/greg/.venvs/experim__examinedlifejournal/bin/activate
   ```
2. Export keys:
   ```bash
   export OPENAI_API_KEY=sk-...
   export ANTHROPIC_API_KEY=ant-...
   ```
3. Start a session:
   ```bash
   uv run examinedlifejournal journal
   ```

Handy flags:
- `--resume` – continue the most recent session
- `--sessions-dir PATH` – store audio and markdown elsewhere

Key behavior during recording:
- Recording starts immediately
- Press any key to stop
- `ESC` cancels the take (audio discarded)
- `Q` saves the take, transcribes it, then ends the session

By default, sessions are saved under `./sessions/`. Each response is written immediately to `sessions/yyMMdd_HHmm_XX.wav` (and `.mp3` when `ffmpeg` is available) and appended to a matching markdown file with YAML frontmatter containing summaries and metadata.

## Research & methodology

This project is explicitly research‑informed. See:
- `docs/reference/SCIENTIFIC_RESEARCH_EVIDENCE_PRINCIPLES.md` – search strategy, quality standards, implementation focus (effect sizes, RCTs, cultural nuance)
- `docs/research/POTENTIAL_RESEARCH_TOPICS.md` – completed topics and prioritized pipeline
- `docs/reference/DIALOGUE_FLOW.md` – conversation sequencing and safety considerations

Highlights of the approach:
- Emphasis on meta‑analyses and RCTs (2019–2025 for digital interventions)
- Guardrails to avoid maladaptive rumination; preference for concrete, time‑bounded prompts
- Implementation‑ready guidance that maps directly to CLI and prompt templates

## Prompting and question design

Follow‑up questions are generated using `examinedlifejournal/prompts/question.prompt.md.jinja`:
- Analyzes emotional intensity, thought patterns, topic persistence, exhaustion, and change talk
- Adapts strategy (validation, redirection, Socratic deepening, or implementation planning)
- Uses clean‑language techniques (user’s exact words), single‑focus questions, and brevity

If the model cannot confidently select an approach, it can select from embedded example questions for safe variety.

## Storage, events, and formats

- File layout and metadata: `docs/reference/FILE_FORMATS_ORGANISATION.md`
- Append‑only event log: `sessions/events.log`
- Audio and transcripts live per session under `./sessions/`

## Testing

Targeted tests can be run without network access:

```bash
PYTHONPATH=. pytest tests/test_storage.py
```

Running the full suite requires valid API keys exported in the environment.

## See also

- CLI usage and controls: `docs/reference/COMMAND_LINE_INTERFACE.md`
- Dialogue flow: `docs/reference/DIALOGUE_FLOW.md`
- Prompt templates: `docs/reference/LLM_PROMPT_TEMPLATES.md`
- Product vision & features: `docs/reference/PRODUCT_VISION_FEATURES.md`
- Whisper/STT notes: `docs/reference/AUDIO_VOICE_RECOGNITION_WHISPER.md`
