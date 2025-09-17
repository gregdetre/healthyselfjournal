# STT Backend Benchmarks

Quick reference comparing backends on a small English validation set (LibriSpeech `test-clean` subset trimmed to ~10 minutes). Metrics:

- **WER** — word error rate (lower is better)
- **xRT** — real-time factor (processing time / audio duration)

> Benchmarks are indicative; run locally with the helper script (see below) whenever dependencies or hardware change.

## Summary

| Backend | Model | Compute | Device | WER | xRT | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| cloud-openai | gpt-4o-transcribe | — | cloud | _pending_ | _pending_ | Reference baseline |
| local-mlx | large-v2 | — | M4 Max | _pending_ | _pending_ | Requires `mlx-whisper` 0.4+ |
| local-faster | large-v2 | int8_float16 | M2 Pro | _pending_ | _pending_ | CTranslate2 CPU path |
| local-faster | large-v2 | float16 | RTX 4090 | _pending_ | _pending_ | CUDA build |
| local-whispercpp | large-v2 (GGUF) | — | M2 Pro | _pending_ | _pending_ | whisper.cpp Python bindings |

Fill in rows as you collect data; keep historical trials with hardware/software context at the bottom of this file.

## Running the benchmark

1. Install the optional dependencies you plan to compare (`mlx-whisper`, `faster-whisper`, `whispercpp`).
2. Export any required API keys (OpenAI).
3. Run each backend against the dataset and capture the raw transcript JSONs (the journaling CLI writes `<clip>.stt.json`).
4. Compute WER/xRT per backend (e.g., with `jiwer`, `sclite`, or a small helper script) and note dependency versions + hardware.
5. Update the table above with the new numbers, including the date and git SHA.

## Data set-up

- `data/benchmark/dev_clean_short/`: 20 clips (≈30s each) covering accents + genders.
- Each clip accompanied by reference text under the same stem with `.txt` extension.
- You can assemble a similar set manually (scripted helper TBD).

## Historical runs

Document each benchmark batch here with date, git SHA, hardware, and notable observations.

```
YYYY-MM-DD | sha | backend(s) | WER/xRT summary | Notes
```

_2025-02-?? — first run pending once local dependencies are installed on the shared Apple Silicon dev box._
