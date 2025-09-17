### Goal, context

Design and implement a modular speech-to-text (STT) layer with switchable audio backends. Default to cloud for convenience/accuracy; provide local/offline options for privacy with platform-aware defaults. Include an `auto-private` mode that selects a best-effort local-first backend based on platform and available dependencies. All other backend selections are deterministic (no implicit fallback/magic).

Large-v2 is preferred by default for local Whisper because it is reputedly stronger for English.

### References

- `examinedlifejournal/transcription.py` — current OpenAI Audio Transcriptions call-site
- `examinedlifejournal/session.py` — STT invocation and transcript writes
- `legacy_transcribe_cli.py` — prior MLX/faster-whisper wiring
- `docs/reference/AUDIO_VOICE_RECOGNITION_WHISPER.md` — legacy local STT guidance (to update)
- `docs/reference/COMMAND_LINE_INTERFACE.md` — CLI conventions

### Principles, key decisions

### Status update (2025-09-17)

- Core backend abstraction merged with switchable adapters (cloud, MLX, faster-whisper, whisper.cpp).
- CLI exposes --stt-backend/--stt-model/--stt-compute/--stt-formatting and surfaces auto-private decisions.
- Docs refreshed (AUDIO_VOICE_RECOGNITION_WHISPER.md, COMMAND_LINE_INTERFACE.md) plus new benchmarks scaffold. Journal now only requires OPENAI_API_KEY when using cloud STT; reconcile also checks OPENAI_API_KEY only for cloud STT.
- Tests: added factory + formatting coverage; still owe adapter-specific success-path tests.
- Benchmarks + deeper local measurements remain TODO.

- Accuracy first; acceptable latency but not at major accuracy cost
- Cloud default: OpenAI `gpt-4o-transcribe`
- Local defaults prefer Whisper large‑v2 for English
- Clear abstraction + adapters per backend, consistent return type
- `auto-private` can probe platform/deps to pick a local backend; explicit backends do not fallback
- Formatting: fast client-side sentence splitting; toggleable

### Stages & actions

#### Stage: Research cloud capabilities (batch)
- [ ] Confirm `gpt-4o-transcribe` supported params (language, response format)
- [ ] Verify lack of server-side paragraphing; document client-side formatter

#### Stage: Research local backends and platform viability
- [ ] Apple Silicon (M1–M4):
  - [ ] MLX Whisper (`mlx-whisper`) with large‑v2 vs large‑v3 — measure throughput, compile/setup time
  - [ ] faster‑whisper CPU (`compute_type=int8_float16`) — measure accuracy/throughput
  - [ ] Consider whisper.cpp (Metal) as minimal-deps option
- [ ] macOS Intel:
  - [ ] faster‑whisper CPU as primary; whisper.cpp as alternative
- [ ] Windows/Linux:
  - [ ] faster‑whisper (CUDA when available, else CPU)
  - [ ] whisper.cpp as portable fallback option

Deliverable: `docs/reference/BACKEND_BENCHMARKS.md` with WER and xRT on small English set.

#### Stage: Decide defaults and explicit behavior
- [x] Cloud default remains `gpt-4o-transcribe`
- [x] Local defaults per platform (no implicit fallback except in `auto-private`):
  - [x] macOS Apple Silicon: `mlx-whisper:large-v2`; explicit
  - [x] macOS Intel: `faster-whisper:large-v2:int8_float16`
  - [x] Windows/Linux: `faster-whisper:large-v2` (CUDA if available, otherwise specify CPU)
- [x] Model presets: `default`, `accuracy`, `fast` with concrete mappings per backend

#### Stage: Abstraction and configuration surface
- [x] Define `TranscriptionBackend` interface
  - [x] `transcribe(wav_path: Path, language: str) -> { text, raw_response, model }`
- [x] Add CLI/config options:
  - [x] `--stt-backend` one of: `cloud-openai`, `local-mlx`, `local-faster`, `local-whispercpp`, `auto-private`
  - [x] `--stt-model` (backend-specific, e.g., `large-v2`)
  - [x] `--language` (default `en`)
  - [x] `--stt-compute` (optional; e.g., `int8_float16` for faster‑whisper)
- [x] Backend factory:
  - [x] Deterministic for explicit backends (error if missing deps)
  - [x] `auto-private` may probe platform/deps to choose a local backend in this priority: Apple Silicon → MLX; else faster‑whisper; else whisper.cpp; if none available, error with actionable install hints

#### Stage: Implement adapters
- [x] Cloud OpenAI adapter (existing):
  - [x] Wire to model from config; pass language; retry/backoff
- [x] MLX Whisper adapter (macOS Apple Silicon):
  - [x] Use `mlx_whisper` (CLI or Python); model selection; language hint if supported
- [x] faster‑whisper adapter:
  - [x] Initialize `WhisperModel` with compute type and threads; concatenate segments with space joining
- [x] whisper.cpp adapter (optional):
  - [x] Use Python bindings or subprocess with Metal on macOS; support model selection

#### Stage: Formatting pass
- [x] Keep fast sentence-per-line post-process (toggle via `--stt-formatting`)
- [x] Ensure O(text length) with minimal latency

#### Stage: CLI & docs
- [x] Add CLI flags to `examinedlifejournal/cli.py`
- [x] Update `docs/reference/AUDIO_VOICE_RECOGNITION_WHISPER.md` with new architecture, defaults, and setup steps
- [x] Document `auto-private` behavior and explicit modes (no implicit fallback)

#### Stage: Tests
- [x] Unit tests for backend factory, explicit erroring on missing deps (covers selection + factory wiring)
- [ ] Adapters: basic success path tests with short fixtures (mock external calls)
- [x] Formatting: sentence splitting idempotency, ellipses handling (extend coverage as needed)

#### Stage: Benchmarks & telemetry
- [ ] Offline benchmark script; publish WER/xRT tables per backend
- [x] Log backend/model in events; no raw audio logged; redact paths in privacy mode (paths still visible outside privacy mode)

#### Stage: Later — Audio preparation
- [ ] Downmix to mono and normalize loudness to a target (e.g., -23 to -16 LUFS or simple RMS normalization)
- [x] Ensure 16–32 kHz sample rate; avoid clipping; trim leading/trailing silence
  - Ensured via recorder at 16 kHz (mono) and a lightweight post-process that trims leading/trailing silence and attenuates peaks (~−0.2 dBFS) without heavy DSP.
- [ ] Optional mild noise gate or spectral gating for steady background hum
- [ ] Simple VAD to trim long intra-segment silences before upload/processing
