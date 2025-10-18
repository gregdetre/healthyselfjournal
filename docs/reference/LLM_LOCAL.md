## Local LLM Reference (stub)

This page helps you choose a local GGUF model for offline question generation, using your machine as a yardstick. It also points to a small and a larger alternative so you can trade speed for quality.

### Yardstick hardware

- Apple M3 MacBook Air, 24 GB unified memory
- Backend: `llama-cpp-python` (GGUF)

On this machine, an 8B model in `Q4_K_M` quantization is comfortable, with responsive inference for short “follow‑up questions.”

### Recommended starting point (debugging)

- Phi‑3 Mini Instruct, `Q4_K_M` GGUF
  - Very small and fast; ideal for verifying the local pipeline (download, load, and prompt) and for quick debugging.

Example (non‑interactive, resolves from Hugging Face):

```bash
uv run --active healthyselfjournal init local-llm \
  --model phi-3-mini-instruct-q4_k_m.gguf \
  --hf-repo TheBloke/phi-3-mini-4k-instruct-GGUF \
  --hf-file phi-3-mini-4k-instruct.Q4_K_M.gguf \
  --hf-revision main
```

Then test question generation end‑to‑end:

```bash
uv run --active healthyselfjournal diagnose local llm \
  --prompt "Quick check: ask me one reflective follow‑up." --max-tokens 64
```

### Mid‑size alternative

- Llama 3.1 8B Instruct, `Q4_K_M` GGUF
  - Good quality/speed balance on M‑series with ≥16 GB RAM; suitable for day‑to‑day local use.

Example:

```bash
uv run --active healthyselfjournal init local-llm \
  --model llama-3.1-8b-instruct-q4_k_m.gguf \
  --hf-repo TheBloke/Llama-3.1-8B-Instruct-GGUF \
  --hf-file llama-3.1-8b-instruct-q4_k_m.gguf \
  --hf-revision main
```

### Slightly smaller alternative

- Mistral 7B Instruct v0.3, `Q4_K_M` GGUF
  - A bit lighter than 8B; often runs a touch faster with still solid outputs.

Example:

```bash
uv run --active healthyselfjournal init local-llm \
  --model mistral-7b-instruct-v0.3-q4_k_m.gguf \
  --hf-repo TheBloke/Mistral-7B-Instruct-v0.3-GGUF \
  --hf-file mistral-7b-instruct-v0.3.Q4_K_M.gguf \
  --hf-revision main
```

### Notes

- Models are stored under the managed directory (macOS): `~/Library/Application Support/HealthySelfJournal/models/llama/`
- If you have a Hugging Face token, set one of: `HUGGING_FACE_HUB_TOKEN`, `HF_TOKEN`, or `HUGGING_FACE_TOKEN`.
- Tuning (env or `user_config.toml`):
  - `LLM_LOCAL_CONTEXT` (e.g., 4096 → 8192)
  - `LLM_LOCAL_THREADS` (0=auto, or set to CPU cores)
  - `LLM_LOCAL_GPU_LAYERS` (start at 0; increase gradually if desired)
  - If local LLM support is missing, install llama.cpp bindings: `uv pip install llama-cpp-python`

See also: `CLI_COMMANDS.md` (diagnostics), `AUDIO_VOICE_RECOGNITION_WHISPER.md` (STT options), and `ARCHITECTURE.md` for flow.

## Local LLM (llama.cpp / GGUF) – Yardstick and quickstart

This app supports a fully local question generator via `llama-cpp-python` loading GGUF models. Use this as a practical yardstick to choose a model that fits your Mac.

### Yardstick (example system)

- Apple M3 MacBook Air, 24 GB unified memory
- Comfortable default: Llama 3.1 8B Instruct, `Q4_K_M` quantization
- Lighter/faster alternative: Phi-3 Mini Instruct, `Q4_K_M`
- Larger/stronger alternative: Mistral 7B Instruct, `Q4_K_M`

Notes:
- `Q4_K_M` is a good quality/speed balance for Apple Silicon with llama.cpp.
- Start with context 4k–8k, threads auto (`LLM_LOCAL_THREADS=0`), GPU layers 0. Adjust later.

### Setup (managed download)

Use the non-interactive Hugging Face resolver to place the model under the managed path (`~/Library/Application Support/HealthySelfJournal/models/llama/`). Pick one.

Llama 3.1 8B Instruct (`Q4_K_M`):

```bash
uv run --active healthyselfjournal init local-llm \
  --model llama-3.1-8b-instruct-q4_k_m.gguf \
  --hf-repo TheBloke/Llama-3.1-8B-Instruct-GGUF \
  --hf-file llama-3.1-8b-instruct-q4_k_m.gguf \
  --hf-revision main
```

Mistral 7B Instruct v0.3 (`Q4_K_M`):

```bash
uv run --active healthyselfjournal init local-llm \
  --model mistral-7b-instruct-v0.3-q4_k_m.gguf \
  --hf-repo TheBloke/Mistral-7B-Instruct-v0.3-GGUF \
  --hf-file mistral-7b-instruct-v0.3.Q4_K_M.gguf \
  --hf-revision main
```

Phi-3 Mini Instruct (`Q4_K_M`) – small/fast for debugging:

```bash
uv run --active healthyselfjournal init local-llm \
  --model phi-3-mini-instruct-q4_k_m.gguf \
  --hf-repo TheBloke/phi-3-mini-4k-instruct-GGUF \
  --hf-file phi-3-mini-4k-instruct.Q4_K_M.gguf \
  --hf-revision main
```

### Test the local question generator

```bash
uv run --active healthyselfjournal diagnose local llm \
  --prompt "Quick check: ask me one reflective follow-up." \
  --max-tokens 64
```

If you see a "Model Missing" panel, run one of the init commands above first.

### Quick tips for better local results

- Prefer a coherent instruct model: Mistral 7B Instruct or Llama 3.1 8B. Use Phi‑3 Mini mainly to verify downloads/loading; its short outputs can be choppy.
- Lower temperature for crisp questions: set `LLM_TEMPERATURE_QUESTION=0.2` (0.0–0.3 works well). Example:

```bash
LLM_LOCAL_MODEL=Mistral-7B-Instruct-v0.3-Q4_K_M.gguf \
LLM_TEMPERATURE_QUESTION=0.2 \
uv run --active healthyselfjournal diagnose local llm \
  --prompt "Quick check: ask me one reflective follow-up." --max-tokens 64
```

- Keep outputs short: use `--max-tokens 32..128` for follow‑ups; larger budgets can drift.
- Resolver hiccups: if `--hf-repo/--hf-file` 404s, pass a direct `--url` (optionally `--sha256`), or place the `.gguf` under the managed path above. For gated repos, set a HF token (`HUGGING_FACE_HUB_TOKEN`, `HF_TOKEN`, or `HUGGING_FACE_TOKEN`).
- Runtime tuning defaults are fine: `LLM_LOCAL_THREADS=0`, `LLM_LOCAL_GPU_LAYERS=0`, `LLM_LOCAL_CONTEXT=4096`. Increase GPU layers gradually only if stable.
- Prompting: explicitly ask for a single reflective follow‑up (e.g., “Ask me one gentle, reflective follow‑up question.”) to improve adherence.

### Configuration (optional)

Set via environment or `user_config.toml`:

- `LLM_MODE=local` – force local LLM mode in journaling
- `LLM_LOCAL_MODEL` – filename under the managed llama directory
- `LLM_LOCAL_CONTEXT=4096` – context length (start 4096; adjust later)
- `LLM_LOCAL_THREADS=0` – auto threads (or set to CPU core count)
- `LLM_LOCAL_GPU_LAYERS=0` – start at 0; can experiment higher

### See also

- `docs/reference/CLI_COMMANDS.md` – `diagnose local llm`, `init local-llm`
- `healthyselfjournal/llm_local.py` – llama.cpp loader
- `healthyselfjournal/model_manager.py` – model storage locations

