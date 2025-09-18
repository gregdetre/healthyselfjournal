# AGENTS

Short guidelines for agents/tools working in this repo.

## Product vision & goals

A command-line, voice-first reflective journaling app that lowers friction (speak, don’t type) and uses dialogue-based questioning to sustain engagement while avoiding common pitfalls identified in research.

See: `docs/reference/PRODUCT_VISION_FEATURES.md`

- Core features:
  - Voice-first input (Whisper/STT) for stream-of-consciousness expression
  - Text output from Claude for reflective dialogue
  - Multiple daily sessions with persistent context
  - Hybrid adaptive questioning (Socratic, MI, validation) via prompt templates

- Current implementation highlights:
  - Real-time recording meter and keyboard controls; immediate WAV persistence; optional background MP3
  - OpenAI Whisper STT with retries; raw `.stt.json` persisted per segment
  - Continuous dialogue loop with Claude; Jinja templates; embedded example questions for variety
  - Summaries regenerated in the background and stored in session frontmatter

## Setup

- See `docs/reference/SETUP.md` for uv + external venv workflow
- Preferred venv: `/Users/greg/.venvs/experim__examinedlifejournal` (source it first)
- `gjdutils` is a local editable dep via `[tool.uv.sources]`
- `ffmpeg` on PATH enables background MP3 conversion (optional)

## Run

- Activate venv, then:
  - `uv sync --active`
  - `uv run --active examinedlifejournal journal [--sessions-dir PATH]`

## Tests

- Minimal, offline: `pytest tests/test_storage.py`
- Full suite requires API keys in env
- Tests live in `tests/`

## Logs & saved files

- Event log: `sessions/events.log` (metadata-only; see `examinedlifejournal/events.py`)
- Session outputs under `./sessions/` (markdown + per-session audio dir)
- Details: `docs/reference/FILE_FORMATS_ORGANISATION.md`

## Key modules

- Core: `session.py`, `audio.py`, `transcription.py`, `llm.py`
- Persistence & context: `storage.py`, `history.py`, `events.py`
- CLI & prompts: `cli.py`, `prompts/*.jinja`, `config.py`

## Key reference docs

- CLI usage: `docs/reference/COMMAND_LINE_INTERFACE.md`
- Recording controls: `docs/reference/RECORDING_CONTROLS.md`
- Dialogue flow: `docs/reference/DIALOGUE_FLOW.md`
- Prompt templates: `docs/reference/LLM_PROMPT_TEMPLATES.md`
- File formats: `docs/reference/FILE_FORMATS_ORGANISATION.md`
- Whisper/STT: `docs/reference/AUDIO_VOICE_RECOGNITION_WHISPER.md`
- see others in `docs/reference/`

## Planning & research

- Implementation plan: `docs/planning/250917a_voice_journaling_app_v1.md`
- Decisions & context: `docs/conversations/`
- Evidence base: `docs/research/`

## Tips

- When using the external venv, pass `--active` to uv project commands
- Use `--sessions-dir` to target a temp/test directory during development
- Git commits: see `gjdutils/docs/instructions/GIT_COMMIT_CHANGES.md`. To avoid interference, chain unstage/add/commit with reset first: `git reset && git add <paths> && git commit -m "type: subject"`
