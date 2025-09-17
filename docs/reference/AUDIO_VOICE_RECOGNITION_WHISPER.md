## Audio Voice Recognition with Whisper (Local)

### Introduction
This document describes how we run OpenAI Whisper locally for audio transcription. We prioritize accuracy, using the `large-v2` model by default. For Apple Silicon acceleration we use the MLX backend (GPU via `mlx-whisper`); CPU inference is available via `faster-whisper`. It covers setup, CLI usage, known gotchas, and troubleshooting.

### See also
- `../reference/SETUP.md` — project setup, venv, and `uv` usage.
- `../../AGENTS.md` — quick pointers for agents/tools (preferred venv, `uv --active`).
- `../../main.py` — simple CLI entrypoint that loads Whisper `large-v2` and transcribes a file.
 - `../../whisper_record.py` — live mic capture CLI with MLX (GPU) or faster‑whisper (CPU) backends.
- External:
  - OpenAI Whisper: https://github.com/openai/whisper — reference implementation and docs.
  - PyTorch MPS notes: https://pytorch.org/docs/stable/notes/mps.html — Apple Metal backend details.
  - faster-whisper (optional): https://github.com/guillaumekln/faster-whisper — faster CPU inference with near-parity accuracy.
  - whisper.cpp (optional): https://github.com/ggerganov/whisper.cpp — ultra‑lean C++/ggml/gguf implementation.

### Principles, key decisions
- Favor transcription accuracy over speed; default model is `large-v2` (particularly strong for English).
- Default backend on Apple Silicon is MLX (GPU via `mlx-whisper`); CPU is available via `faster-whisper`.
- No diarization or word‑level timestamps in scope here; plain text output only.
- Use system `ffmpeg` for audio handling.

### Overview
At a high level, the CLI in `main.py`:
- Detects device: `cpu` by default (or `mps` when explicitly requested).
- Loads `large-v2` via `whisper.load_model(...)` (from cache at `~/.cache/whisper` or auto‑downloads on first use).
- Transcribes the input audio/video file and prints plain text to stdout.

Live mic capture in `whisper_record.py`:
- Records from the system mic with `ffmpeg` (until RETURN or for `--duration` seconds)
- Transcribes using the selected backend: `mlx` (GPU) or `faster` (CPU)
- Defaults: `--backend mlx`, `--model-name large-v2`, language auto‑detect (override with `--language en`)

### Setup
1) Ensure `ffmpeg` is installed (macOS/Homebrew):
```bash
brew install ffmpeg
```
2) Use the preferred external venv and sync deps with `uv`:
```bash
source /Users/greg/.venvs/experim__examinedlifejournal/bin/activate
uv sync --active
```

### Usage
Live mic capture (GPU, MLX backend):
```bash
python whisper_record.py --duration 3                   # defaults to MLX + large-v2, language auto
python whisper_record.py --duration 3 --language en     # force English
```

CPU (faster-whisper) alternative:
```bash
python whisper_record.py --duration 3 --backend faster --model-name large-v2 --compute-type int8_float16
```

Basic transcription (CPU, safest):
```bash
python main.py --device cpu /path/to/audio.m4a
```
Try Apple Metal (may be faster, can be flaky depending on torch):
```bash
python main.py --device mps /path/to/audio.m4a
```
Notes:
- Supported formats include `.m4a`, `.mp3`, `.wav`, etc. `ffmpeg` handles conversion internally.
- Model files cache under `~/.cache/whisper/`; if `large-v2.pt` is present, it will be reused.

### Examples
Create a quick test file on macOS and transcribe:
```bash
say -v Samantha "This is a short Whisper test." -o test.aiff
ffmpeg -y -i test.aiff -ar 16000 -ac 1 test.wav
python main.py --device cpu test.wav
```

### Gotchas
- PyTorch MPS op gaps: Some torch versions on macOS fail with missing MPS sparse ops (e.g., sparse COO). Prefer MLX (GPU) or faster‑whisper (CPU) instead of MPS.
- FP16 on CPU is unsupported; we keep FP16 disabled on MPS too for stability.
- First run may download models; ensure adequate disk space and network access or pre‑seed the cache.
- MLX first run: models are fetched from `mlx-community/whisper-<model>` and compiled; this can take minutes. Subsequent runs are much faster. The `whisper_record.py` CLI streams progress.

### Troubleshooting
- "whisper not found": ensure the venv is active and `uv sync --active` has completed.
- `ffmpeg` missing: install via Homebrew (`brew install ffmpeg`).
- MPS errors: rerun with `--device cpu`; consider updating PyTorch to a newer version later.
- Slow on CPU: acceptable trade‑off for stability; consider trying `--device mps` or evaluating `faster-whisper` separately.

### Planned improvements
- Optional `faster-whisper` path for speed with near‑parity accuracy on CPU.
- Streaming mic capture command (e.g., `whisper_record.py`) for quick dictation workflows.
- Automated local benchmark script comparing CPU vs MPS on short samples.


