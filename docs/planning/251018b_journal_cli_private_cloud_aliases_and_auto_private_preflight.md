## Goal, context

Add two user-friendly CLI aliases that minimize flags and "just work" for the common cases:

- `healthyselfjournal journal cli private` – privacy-first defaults; prefer fully local STT and local LLM; great guidance and optional auto-fix when deps are missing.
- `healthyselfjournal journal cli cloud` – cloud-first defaults; enforce presence of required keys with concise guidance.

We already have `auto-private` STT backend selection and clear key checks. This work reuses those, adds an Ollama/local preflight, streams better guidance, and optionally offers one-click fixes (Y/n) when running interactively.


## References

- `healthyselfjournal/cli_journal_cli.py` – main CLI journaling command, options, preflight, panels.
- `healthyselfjournal/cli.py` – Typer app wiring; where to hang new subcommands.
- `healthyselfjournal/transcription.py` – `auto-private` backend selection and guidance text.
- `healthyselfjournal/llm.py` – cloud-off enforcement guardrails for LLM providers.
- `healthyselfjournal/config.py` – defaults, `LLM_CLOUD_OFF`, provider model parsing, precedence rules.
- `healthyselfjournal/cli_diagnose.py` – privacy diagnostics panels; tone and checks to reuse.
- `docs/reference/CLI_RECORDING_INTERFACE.md` – current flags and expectations for STT/LLM.
- `docs/reference/PRIVACY.md` – network boundaries; how to stay offline.
- `README.md`, `docs/reference/SETUP_USER.md` – user entry points; add the two new commands.


## Principles, key decisions

- Keep existing `journal cli` fully backward-compatible.
- Aliases are thin: they set sensible defaults, then delegate to the existing `journal` implementation.
- Fewer flags, better guidance: show one actionable next step per failure; advanced details follow.
- Interactive auto-fix only in a TTY, gated by a clear Y/n prompt; never persist env silently.
- Respect user overrides: explicit flags after the alias override alias defaults.
- Avoid cloud traffic in `private` by setting process-local `cloud_off` and disabling cloud TTS.
- Use `ollama:qwen2.5:7b-instruct` as the default local LLM question-generation model.


## Stages & actions

### Add alias commands (v1)
- [ ] Add Typer subcommands under `journal`:
  - `journal cli private`
  - `journal cli cloud`
- [ ] Map to defaults and delegate:
  - `private`:
    - Effective defaults: `--stt-backend auto-private`, `LLM_CLOUD_OFF=1` for this process.
    - `--llm-model`: if not already `ollama:*`, set a sensible default (proposal: `ollama:gemma3:27b-instruct-q4_K_M`).
    - TTS: default off unless/when we add a local TTS backend; keep voice disabled to avoid cloud.
  - `cloud`:
    - Effective defaults: `--stt-backend cloud-openai`, `LLM_CLOUD_OFF=0` for this process.
    - Use default cloud LLM (Anthropic Claude) unless user overrides.
- [ ] Preserve precedence: explicit flags > env > alias defaults.
- [ ] Log an event tag (e.g., `cli.mode_alias=private|cloud`).

### Preflight checks & guidance (v1.1)
- [ ] Reuse `resolve_backend_selection()` for STT; show improved guidance when `auto-private` fails.
- [ ] Ollama preflight when provider is or will be `ollama:*`:
  - `which('ollama')` present?
  - Daemon reachable at `OLLAMA_BASE_URL` (`/api/tags`)?
  - Model available (`/api/tags` contains tag)? If missing, propose `ollama pull <model>`.
- [ ] Compress guidance into one panel with a single best next step; include a short “details” section.

### Interactive auto-fix (v1.2)
- [ ] Only when interactive TTY; present "Fix now? [Y/n]" for safe operations:
  - Start Ollama daemon (platform-specific guidance; e.g., `open -a Ollama` on macOS, or show `ollama serve`).
  - `ollama pull <model>` if missing.
  - On Apple Silicon, offer: `pipx install mlx-whisper` (MLX Whisper). Otherwise link to faster‑whisper/whisper.cpp steps.
- [ ] On decline, exit with a concise summary of the next manual command to run.

### Docs & help (v1.3)
- [ ] README/SETUP_USER: add short sections for `journal cli private` and `journal cli cloud`.
- [ ] Update LLM_LOCAL.md to reference "ollama:qwen2.5:7b-instruct" as our default model
  - [ ] Update AGENTS.md and README.md and any other relevant docs to signpost to LLM_LOCAL.md for the default/auto local model (rather than referencing it explicitly, so we only have to update the docs in one place if we change our mind)
- [ ] CLI `--help`: ensure the `journal` group help surfaces both aliases with 1-liners.
- [ ] Reference docs: `CLI_RECORDING_INTERFACE.md`, `PRIVACY.md` mention the aliases and what they imply.

### Tests (v1.4)
- [ ] Unit tests: alias argument mapping → effective values, precedence with explicit flags.
- [ ] Preflight unit tests: simulate missing Ollama/daemon/model and verify guidance text.
- [ ] Interactive auto-fix: simulate TTY/non-TTY; ensure no side effects without confirmation.

### Optional enhancements (later)
- [ ] Local TTS backend; enable voice in `private` when available.
- [ ] `journal cli private --voice-mode` gates local TTS only; remains cloud-free.
- [ ] Non-interactive `--yes` flag to auto-apply safe fixes in CI/dev automation.


## Acceptance criteria

- Running `journal cli private` on Apple Silicon with MLX installed and Ollama running starts recording immediately without additional flags, never touching the network for STT/LLM.
- Running `journal cli cloud` fails fast with clear key guidance when required keys are absent; succeeds when keys are present.
- When Ollama is missing/not running/model absent, the `private` path shows a single clear next step and optionally performs it on Y.
- Explicit user flags override the alias defaults.
- No regressions for `journal cli` or existing flags; tests cover alias mapping and preflights.



## Notes

- The alias design intentionally avoids persisting settings; users can still use `init` to save long-lived prefs.
- `auto-private` guidance already prints MLX/faster/whisper.cpp options; we’ll enrich it and unify with the alias path so the same helpful content appears.
- Offer to run `ollama pull` (and run it, e.g. with `brew services` or `open` or similar) behind a Y/n prompt, but only when interactive. (Perhaps we'd look for whether there's a `brew` already installed and use that, falling back to something else if not? Use your judgment)


