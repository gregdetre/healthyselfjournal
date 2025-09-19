## Speech-To-Text Backends

This project now exposes a modular speech-to-text (STT) layer with switchable backends. A single CLI flag drives the selection and model preset at runtime, making it easy to mix a cloud default with local/offline setups when privacy or cost require it.

### Available backends

| Backend id | Default model | Dependency | Notes |
| --- | --- | --- | --- |
| `cloud-openai` | `gpt-4o-transcribe` | `openai` SDK | Default: highest accuracy, low latency, requires API key |
| `local-mlx` | `large-v2` | [`mlx-whisper`](https://github.com/apple/mlx-examples/tree/main/whisper) CLI | Apple Silicon only; uses MLX + Metal. Model names map to `mlx-community/whisper-<model>` |
| `local-faster` | `large-v2` | [`faster-whisper`](https://github.com/guillaumekln/faster-whisper) | Portable CPU/GPU (CUDA/CPU). We default to `compute=int8_float16` |
| `local-whispercpp` | `large-v2` | [`whispercpp`](https://github.com/aarnphm/whispercpp.py) + GGUF file | Ultra-lightweight fallback; you must supply a local GGUF path |
| `auto-private` | local-first | probes environment | Picks the best available local backend (MLX → faster-whisper → whisper.cpp) or errors with setup guidance |

Model presets (`default`, `accuracy`, `fast`) resolve differently per backend. For example, `default` selects `large-v2` for local backends but sticks with `gpt-4o-transcribe` in the cloud; `fast` prefers `small`/`base` tiers. Supplying an explicit model name (e.g. `large-v3`) bypasses the preset.

### CLI usage

```
uv run --active healthyselfjournal journal cli \
  --stt-backend cloud-openai \
  --stt-model default \
  --stt-formatting sentences
```

Key options:

- `--stt-backend`: one of `cloud-openai`, `local-mlx`, `local-faster`, `local-whispercpp`, `auto-private`.
- `--stt-model`: preset (`default`, `accuracy`, `fast`) or backend-specific identifier (`large-v2`, `gpt-4o-mini-transcribe`, path to GGUF, ...).
- `--stt-compute`: optional precision override for local backends (e.g. `int8_float16`). Ignored by cloud/MLX.
- `--stt-formatting`: `sentences` (default) adds fast sentence-per-line formatting; `raw` keeps the provider output as-is.

`auto-private` prints the chosen backend and why. If no local backend is available, it errs with actionable instructions instead of falling back silently to the cloud.

### Dependency setup

> Assumes you are using the shared external venv described in `docs/reference/SETUP.md`.

**OpenAI (cloud)**
```
uv sync --active
export OPENAI_API_KEY=...
```

**MLX Whisper (Apple Silicon)**
```
uv add mlx-whisper
# Optional: prefetch large-v2
mlx_whisper --model mlx-community/whisper-large-v2 --download-only
```

**faster-whisper (portable CPU/GPU)**
```
uv add faster-whisper
# Optional CUDA build: set CTRANSLATE2_CUDA=1 before uv sync
```

**whisper.cpp bindings**
```
uv add whispercpp
# Supply a GGUF path via --stt-model (/path/to/ggml-large-v2.gguf)
```

`ffmpeg` remains optional but recommended for background MP3 conversion.

Environment requirements:

- `journal`: requires `ANTHROPIC_API_KEY` when the LLM provider is `anthropic:*` (default cloud mode). Local `ollama:*` models keep dialogue/summaries offline. `OPENAI_API_KEY` is still needed only when `--stt-backend cloud-openai` is selected.
- `reconcile`: requires `OPENAI_API_KEY` only when `--stt-backend cloud-openai` is selected. Local backends run fully offline if their dependencies are installed.

### Microphone input handling and sample‑rate fallback

The CLI records via `sounddevice`/PortAudio. If your input device changes mid-session (for example, turning off Bluetooth headphones), the app will now:

- Retry the default input device at the requested sample rate
- Retry the device’s default sample rate
- Iterate other available input devices with their default rates

When the input sample rate differs from the requested one (e.g., 48000 Hz rather than 16000 Hz), a brief yellow notice appears: “Input device uses 48000 Hz; adjusting.” The WAV is recorded at the effective rate.

### Formatting behaviour

Client-side formatting runs only when `--stt-formatting` is `sentences`. The heuristic remains O(n), preserves ellipses, and splits on `.`, `!`, `?` followed by an uppercase/digit. Switching to `--stt-formatting raw` keeps the backend text untouched (aside from `.strip()`).

### Event logging & persistence

Every transcription logs start/success/failure events with backend, model, compute (when relevant), and text length. Raw responses persist alongside each WAV as `<segment>.stt.json`. Frontmatter now records:

- `model_stt` — resolved model name/path
- `stt_backend`, `stt_compute`, `stt_formatting`
- Requested backend/model/compute (useful when auto selection differs)
- Any `stt_warnings` emitted during selection

### Troubleshooting

- `STT configuration error`: dependency missing. Install the package named in the message and rerun.
- `auto-private` errors immediately when no local backend is available—install `mlx-whisper`, `faster-whisper`, or `whispercpp`.
- whisper.cpp requires an explicit GGUF path; presets only work for cloud/faster/MLX.
- To inspect raw backend output, open the `.stt.json` that sits beside each WAV file.

- PortAudio/macOS input device changed (e.g., Bluetooth headset turned off mid-session):
  - Symptoms: console warnings like `PaMacCore (AUHAL) ... '!obj'`, `err='-10851' Audio Unit: Invalid Property Value`, followed by `sounddevice.PortAudioError: Error opening InputStream: Internal PortAudio error [PaErrorCode -9986]`.
  - Behaviour: the app automatically retries devices/sample rates. If it recovers, you may see a notice such as “Input device uses 48000 Hz; adjusting.”
  - If it still fails: fully disconnect the headset or select a working input in System Settings → Sound → Input, then rerun `uv run --active healthyselfjournal journal cli`. Also close any apps that might be holding the microphone.

### Next steps

See `docs/reference/BACKEND_BENCHMARKS.md` for current WER/xRT numbers and capture commands for running the benchmark locally.
