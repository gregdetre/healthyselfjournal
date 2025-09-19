# Privacy and Data Practices

This document explains what data the voice‑first journaling app collects, where it is stored, when (and if) it leaves your machine, and how you control it. The app is designed local‑first and privacy‑preserving by default; network calls happen only for specific features you enable.

## See also

- `CLI_COMMANDS.md` – How to run the app, flags that affect privacy (e.g., STT, voice mode).
- `FILE_FORMATS_ORGANISATION.md` – Exact on‑disk structure for sessions, audio, transcripts, and summaries.
- `DIALOGUE_FLOW.md` – What conversational context is shared with the LLM and why.
- `CONVERSATION_SUMMARIES.md` – Summary lifecycle, where summaries are stored in frontmatter.
- `AUDIO_VOICE_RECOGNITION_WHISPER.md` – STT design, retries, and backend behavior.
- `PRODUCT_VISION_FEATURES.md` – Local‑first vision and how privacy supports it.
- `OLLAMA_GEMMA_DEPLOYMENT_GUIDE.md` – Running a local LLM to avoid sending text to cloud providers.

## Principles and key decisions

- Local‑first by design: your recordings, transcripts, and notes are stored on your disk in human‑readable formats.
- Explicit network use: audio leaves your device only when you choose a cloud STT backend; text leaves your device only when using a cloud LLM or TTS.
- No telemetry: the app does not send analytics, usage metrics, or crash reports.
- Minimum necessary data: prompts include only recent, relevant context using a budget heuristic; raw audio is never sent to LLMs.
- You control retention: files persist until you delete or archive them; no auto‑upload.
- Transparency and portability: everything is Markdown, WAV/MP3, and simple JSON.

## What data the app handles

The app may create and store the following data locally under your sessions directory (defaults to `./sessions/`):

- Audio recordings: primary capture in WAV; optional MP3 generated in background if `ffmpeg` is available.
- Transcripts and STT metadata: JSON responses per segment (e.g., `.stt.json`) and aggregated text.
- Dialogue content: your questions/answers and the assistant’s follow‑ups in the per‑session `.md` file.
- Summaries: short structured summaries stored in the session Markdown frontmatter.
- Event metadata log: `sessions/events.log` holds timestamps and action metadata only; no transcript or audio content.
- Configuration: local flags and environment variables are used at runtime but not stored in session files.

See the exact layout in `FILE_FORMATS_ORGANISATION.md`.

## What can leave your machine (and when)

Nothing leaves your machine unless you enable a cloud provider for a specific feature. Each feature can be configured to use local backends where available.

- Speech‑to‑Text (STT)
  - Cloud (default today): `--stt-backend cloud-openai` sends audio segments to OpenAI Whisper for transcription.
  - Local: `--stt-backend local-mlx`, `local-faster`, or `local-whispercpp` keeps all audio on‑device.

- LLM for dialogue and summaries
  - Cloud (current default): prompts and recent context are sent to Anthropic (Claude) for question generation and to generate summaries.
  - Local (optional): configure a local LLM (e.g., via Ollama) to keep all text processing on‑device. See `OLLAMA_GEMMA_DEPLOYMENT_GUIDE.md`.

Note on local LLM model setup: when you choose the CLI option to resolve a model from Hugging Face, the app makes a metadata request to the Hugging Face API to obtain a file’s SHA‑256 (LFS OID) and a stable download URL. This request does not include your recordings, transcripts, or prompts—only model metadata is fetched to verify file integrity.

- Text‑to‑Speech (TTS) for voice mode
  - Cloud: enabling `--voice-mode` uses OpenAI TTS to synthesize the assistant’s question audio from text.
  - Local: if/when a local TTS backend is configured, no text is sent to a cloud provider.

Notes:
- The app never sends raw audio to LLMs; only to STT when you choose a cloud STT backend.
- Prompt context is trimmed using heuristics (see `DIALOGUE_FLOW.md`), and includes recent summaries and selective excerpts as needed.

## Current state vs target state

**Current State**
- STT default favors OpenAI cloud for accuracy and simplicity.
- Dialogue and summaries default to a cloud LLM (Anthropic Claude).
- TTS (if enabled) uses OpenAI.

**Target State**
- First‑class local options across STT, LLM, and TTS to enable fully offline operation.
- Simple CLI switches to opt into local backends everywhere.

**Migration Status**
- STT supports multiple local backends today; see `CLI_COMMANDS.md` for flags.
- Local LLM path documented in `OLLAMA_GEMMA_DEPLOYMENT_GUIDE.md`.
- Local TTS evaluation is planned.

## Controlling network use

- STT: choose a local backend with `--stt-backend local-mlx` (Apple Silicon), `local-faster`, or `local-whispercpp`. You can also specify `--stt-model` and `--language`.
- LLM: follow `OLLAMA_GEMMA_DEPLOYMENT_GUIDE.md` to use a local model; otherwise set your cloud model via environment and config.
- TTS: disable voice output by omitting `--voice-mode`, or select a local TTS once available.
- Offline mode: use only local backends; the app will not attempt cloud calls without the corresponding API keys set.

### Verifying privacy with diagnostics

Use the built-in diagnostics to confirm that local paths are used and cloud calls are blocked when privacy is enabled:

```bash
# Diagnostics help (groups show help by default when no subcommand)
uv run --active healthyselfjournal diagnose local

# Individual checks
uv run --active healthyselfjournal diagnose local stt --no-audio
uv run --active healthyselfjournal diagnose local llm --fail-on-missing-model
uv run --active healthyselfjournal diagnose local privacy

# Mic-only (interactive)
uv run --active healthyselfjournal diagnose mic --seconds 1.0 --stt-backend auto-private
```

Notes:
- Diagnostics write `events.log` to a temp sandbox directory by default, not your real sessions.
- Local LLM requires a `.gguf` model file; see `OLLAMA_GEMMA_DEPLOYMENT_GUIDE.md` or point `[llm].local_model` at your file.

## On‑disk storage, retention, and deletion

- Location: sessions are stored under `./sessions/` by default. Use `--sessions-dir PATH` to choose another location (e.g., an encrypted volume).
- Retention: the app does not auto‑delete your recordings or transcripts. Short accidental takes can be auto‑discarded based on duration/voiced‑time thresholds (see `PRODUCT_VISION_FEATURES.md`).
- Deletion: remove a session by deleting its Markdown file and the corresponding assets folder. Merges update frontmatter and move assets safely (see `CLI_COMMANDS.md`).
- Portability: session files are plain text and standard audio formats; you can move or back them up with normal file tools.

## Security considerations

- Local files are not encrypted by the app. For sensitive journals, store `--sessions-dir` on an encrypted disk (e.g., FileVault, encrypted external drive) or inside an encrypted folder.
- API keys are read from environment variables (e.g., `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) and not written to session files. Avoid committing them to source control.
- The event log (`sessions/events.log`) contains metadata only. It is safe to delete if desired.
- The app does no automatic PII redaction. Be mindful of what you record; you control what is stored and for how long.

## Third‑party services (when enabled)

When you opt into cloud features, your data is processed by the corresponding provider under their terms:

- OpenAI (STT, TTS): processes audio or text you submit for transcription or speech synthesis. See their privacy and data‑use policies.
- Anthropic (LLM): processes prompt text and context to generate assistant questions and summaries. See their privacy and data‑use policies.
- Local LLM via Ollama: runs on your machine and does not send data externally by default. Review model cards for any model‑specific considerations.

Refer to provider documentation for details on retention, training use, and regional controls. Choose providers and models that align with your privacy requirements.

## Frequently asked questions

- Do you send my entire journal history to the LLM?
  - No. Prompts include only recent, relevant context using a budget heuristic, plus short summaries from prior sessions when helpful. See `DIALOGUE_FLOW.md`.

- Are my raw audio files ever sent to the LLM?
  - No. Audio is only sent to an STT provider when you choose a cloud STT backend.

- Is there telemetry or analytics?
  - No. The app does not collect analytics, usage metrics, or crash reports.

- Can I run fully offline?
  - Yes, by selecting local STT, a local LLM (via Ollama), and (when available) local TTS. Without API keys set, cloud calls will not occur.

- How do I remove all my data?
  - Delete the `sessions/` directory (or your custom `--sessions-dir`). Optionally delete `sessions/events.log`.

## Troubleshooting and tips

- To verify no cloud calls will be made, avoid setting `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` and select local backends.
- Store your sessions directory on an encrypted volume to add at‑rest protection.
- Use `--sessions-dir` to separate personal vs. test journals.

## Maintenance

This document is evergreen. It should be reviewed after any change to storage, STT/LLM/TTS backends, or CLI flags that affect privacy.

- After feature changes: update affected sections and cross‑references.
- During housekeeping: confirm links and defaults remain accurate.
