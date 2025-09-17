# Examined Life Journal

Voice-first journaling CLI combining live audio capture, Whisper transcription, and Claude-powered follow-up questions.

see: `AGENTS.md`


## Prerequisites

- Python 3.12+
- `ffmpeg` on `PATH` (optional, for background MP3 conversion)
- Environment variables:
  - `OPENAI_API_KEY` – used for Whisper transcription
  - `ANTHROPIC_API_KEY` – used for question/summary generation

Install dependencies with [uv](https://docs.astral.sh/uv/) or your preferred tool:

```bash
uv sync  # or install via `uv pip install -r pyproject.toml`
```

The project expects the virtualenv described in `docs/reference/SETUP.md` (`/Users/greg/.venvs/experim__examinedlifejournal`). Activate it before running CLI commands.

## Usage

1. Activate the project virtualenv (`source /Users/greg/.venvs/experim__examinedlifejournal/bin/activate`).
2. Export the required keys:
   ```bash
   export OPENAI_API_KEY=sk-...
   export ANTHROPIC_API_KEY=ant-...
   ```
3. Launch the journaling loop:
   ```bash
   uv run examinedlifejournal journal
   ```

Key behavior during recording:
- Recording starts immediately
- Press any key to stop
- `ESC` cancels the take (audio discarded)
- `Q` saves the take, transcribes it, then ends the session

By default, every session is saved under `./sessions/` in the working directory. Each response is written immediately to `sessions/yyMMdd_HHmm_XX.wav` (and `.mp3` when `ffmpeg` is available) and appended to a matching markdown file with YAML frontmatter containing summaries and metadata. Pass `--sessions-dir PATH` to store files elsewhere.

## Testing

Targeted tests can be run without network access:

```bash
PYTHONPATH=. pytest tests/test_storage.py
```

Running the full suite requires valid API keys exported in the environment.
