# Voice Journaling App V1 Implementation

## Goal, context

Build a command-line journaling app with voice input (Whisper) and text output (Claude) that maintains engagement through adaptive questioning while avoiding common pitfalls identified in research. Priority on quick V1 with core features working end-to-end.

Core requirements:
- Voice recording with visual feedback until keypress
- Immediate persistence (audio + transcript) to prevent data loss
- Continuous dialogue with LLM-generated follow-up questions
- Session summaries maintained in frontmatter for context continuity
- ESC to cancel, Q to quit session

## References

- `docs/reference/PRODUCT_VISION_FEATURES.md` - Core product vision and features
- `docs/reference/COMMAND_LINE_INTERFACE.md` - CLI visual feedback specifications
- `docs/reference/FILE_FORMATS_ORGANISATION.md` - File structure and naming conventions
- `docs/reference/DIALOGUE_FLOW.md` - Session flow and question generation
- `docs/conversations/250917a_journaling_app_ui_technical_decisions.md` - Technical decisions from planning session
- `docs/research/JOURNALLING_SCIENTIFIC_EVIDENCE_RESEARCH.md` - Evidence base for design decisions

## Principles, key decisions

- Start with all-at-once response display (streaming later)
- Use flat directory structure with paired .mp3/.md files
- Regenerate summary after each Q&A for crash resilience
- Default to same opener question with override capability
- Include current session + recent summaries in context
- Transparent mode switching based on emotional context


## Decisions and constraints (V1)

- Audio capture: Use `sounddevice` for cross-platform input and `soundfile` to persist lossless WAV (mono, 16 kHz, 16-bit PCM) immediately after stop. Real-time volume meter computed via NumPy RMS over recent frames. Key handling via `readchar` for portable, non-blocking keypresses. Visuals via `rich`.
- Encoding: If `ffmpeg` is available on PATH, convert WAV to MP3 in a background step post-save; otherwise keep WAV. Never block end-to-end flow on MP3 conversion. Store both filenames in frontmatter when available.
- Transcription: Use OpenAI Speech-to-Text (Whisper-based) with a configurable model. Persist raw API response alongside transcript for debugging. Implement 3 retries with exponential backoff and jitter; on repeated failure, queue the audio for later transcription and proceed in text-less mode only if explicitly requested (default: do not proceed).
- LLM dialogue: Default model `anthropic:claude-sonnet-4:20250514` (format `provider:model:version[:thinking]`), allow override. On rate limit, backoff and optionally fall back to a smaller model. All prompts templated with Jinja2 and rendered via `gjdutils.strings.jinja_render`.
- Context window: Include summaries from the current session plus up to the last 8 session summaries, truncated to ~1,500 tokens total budget for historical summaries. Prefer shorter, high-signal summaries (target 120–180 words each). Hard cap the full prompt to an application-level token budget.
- Session boundaries: Soft reminder at ~20 minutes to consider a short break. No forced termination; user decides.
- File layout: Flat directory with paired audio `.wav`/`.mp3` and `.md` transcript in the same folder, named `yyMMdd_HHmm.*`.
- Frontmatter schema (minimum): `created_at`, `audio_file`, `transcript_file`, `duration_seconds`, `summary`, `recent_summary_refs` (list of filenames), `model_llm`, `model_stt`, `app_version`.
- Core dependencies (V1): `rich`, `anthropic`, `openai`, `jinja2`, `pyyaml`, `sounddevice`, `soundfile`, `numpy`, `readchar`. External: `ffmpeg` (optional, recommended).

### Prompt templates convention

- Each prompt lives in a `.prompt.md.jinja` file adjacent to the module/function that calls it.
- Render prompts using `gjdutils.strings.jinja_render`; validate required variables; prefer strict undefined.
- Keep prompts small, composable, and include variables for `recent_summaries`, `current_transcript`, and `opening_question`.

## Open Questions/Concerns

**Audio Library Selection**
- Need cross-platform audio recording library (pyaudio? sounddevice?)
- MP3 encoding options (pydub? direct ffmpeg?)
- Real-time volume monitoring implementation

Decision (V1): Use `sounddevice` + `soundfile` for capture/persistence; compute RMS levels with NumPy for the meter. Prefer direct `ffmpeg` subprocess for MP3 conversion when present; skip conversion gracefully if unavailable.

**Context Window Management**
- How many recent summaries to include initially? (Start with last 10?)
- Maximum token count before truncation needed?

Decision (V1): Include up to 50 most recent summaries, but enforce a ~5000-token budget for historical context; truncate older/longer summaries first. Implement a conservative token estimator to stay under the prompt budget.

**Error Handling**
- Whisper API failures - fallback strategy?
- Claude API rate limits - retry logic?
- Audio device unavailable - graceful degradation?

Decision (V1): Add 3 retries with exponential backoff and jitter for STT/LLM calls. On persistent STT failure, queue for later and keep files locally; on LLM rate limits, backoff then optionally fall back to a smaller model. If audio device selection fails, fall back to default device or exit with a clear message.

**Session Boundaries**
- Should we implement 15-20 minute warning based on research?
- Pattern detection threshold for suggesting breaks?

Decision (V1): Soft reminder at ~20 minutes based on research; no automated break detection in V1 beyond elapsed time.

## Stages & actions

### Stage: Environment setup and dependencies
- [x] Install core dependencies: `rich`, `anthropic`, `openai`, `jinja2`, `pyyaml`, `sounddevice`, `soundfile`, `numpy`, `readchar`
- [ ] Ensure `ffmpeg` available on PATH (optional for MP3 conversion)
- [ ] Verify audio device enumeration and default selection with `sounddevice`
- [x] Verify real-time volume monitoring capability (RMS computation)
- [x] Create basic project structure and .gitignore

### Stage: Configuration and model selection

- [x] Create `config.py` to centralize parameters:
  - `MAX_RECENT_SUMMARIES` (default 50)
  - `MAX_HISTORY_TOKENS` (default 5000)
  - `SESSION_BREAK_MINUTES` (default 20)
  - `MODEL_LLM` (string: `provider:model:version[:thinking]`, default `anthropic:claude-sonnet-4:20250514`)
  - `MODEL_STT` (e.g., OpenAI Whisper model string)
  - `PROMPT_BUDGET_TOKENS` (app-level hard cap)
  - `FFMPEG_PATH` (optional override)
  - `RETRY_MAX_ATTEMPTS`, `RETRY_BACKOFF_BASE_MS`
- [ ] Read `/Users/greg/dev/spideryarn/reading/docs/reference/LLM_MODEL_CONFIGURATION.md` and align with the model string scheme.
- [x] Add Ollama configuration knobs (`OLLAMA_BASE_URL`, timeout, context) and provider-aware env handling for local models.

### Stage: Basic audio recording with visual feedback
- [ ] Write test for audio recording functionality
- [x] Implement press-any-key to start recording
- [x] Create volume meter visualization using rich
  - Unicode blocks showing real-time audio levels
  - "Recording... [████████░░░░░░░░] Press any key to stop"
- [x] Implement press-any-key to stop recording
- [x] Save audio immediately to WAV with yyMMdd_HHmm timestamp; MP3 conversion in background when available
- [ ] Test recording on different audio devices
  - Use `readchar` for non-blocking keypress handling

### Stage: Whisper transcription integration
- [ ] Write test for transcription pipeline
- [x] Set up OpenAI API client with Whisper
- [x] Implement audio file upload to Whisper API
- [x] Handle transcription response and errors
- [x] Save initial transcript to markdown file
- [ ] Test with various audio qualities and accents
  - Persist raw STT response permanently; implement 3x retry with backoff

### Stage: Basic LLM dialogue with Claude
- [ ] Write test for Claude API integration
- [x] Set up Anthropic Claude API client
- [x] Create basic Jinja2 prompt template
  - Include current transcript
  - Add default opening question
  - Store prompt in `.prompt.md.jinja` next to caller; render via `gjdutils.strings.jinja_render`
- [x] Implement question generation after transcription
- [x] Display AI response as text output
- [x] Update markdown with Q&A format
  - "## AI Q: " prefix for questions
  - User response as section content
  - Enforce prompt token budget; include recent summaries per context rules

### Stage: Session management and controls
- [ ] Write tests for session control flow
- [x] Implement ESC key detection (cancel recording)
- [x] Implement Q key detection (transcribe and quit)
- [x] Add continuous dialogue loop
  - Record → Transcribe → Generate question → Display → Loop
- [x] Handle session termination gracefully
- [x] Ensure all files saved before exit

### Stage: Summary generation and frontmatter
- [ ] Write tests for summary generation
- [x] Create summary prompt template for Claude
  - Store template in `.prompt.md.jinja`; render via `gjdutils.strings.jinja_render`
- [x] Generate initial summary after first Q&A
- [x] Update summary after each subsequent Q&A
- [x] Implement frontmatter read/write with pyyaml
  - Parse existing frontmatter
  - Update summary field
  - Preserve other metadata if present
- [ ] Test summary quality and relevance
  - Frontmatter keys (minimum): `created_at`, `audio_file`, `transcript_file`, `duration_seconds`, `summary`, `recent_summary_refs`, `model_llm`, `model_stt`, `app_version`

### Stage: Context management with previous sessions
- [ ] Write tests for context loading
- [x] Implement file discovery for recent sessions
  - Sort by timestamp in filename
  - Load last N session files
- [x] Extract summaries from frontmatter
- [x] Update Jinja2 template to include context
  - Add recent_summaries variable
  - Format for LLM consumption
- [ ] Test pattern detection across sessions
  - Enforce 50 summaries max and ~5000-token budget for history

### Stage: Question variety and adaptation
- [x] Embed initial example questions in prompt
  - Concrete/specific questions
  - Open/exploratory questions
  - Pattern-interrupting questions
- [x] Implement "Give me a question" detection
- [x] Add embedded example selection logic to template
- [ ] Test question variety over multiple sessions
- [ ] Document question categories for future expansion

### Stage: Error handling and resilience
- [x] Add try/catch for audio device errors
- [x] Implement Whisper API retry logic
- [x] Add Claude API rate limit handling
- [x] Create fallback for network failures
  - Save audio/transcript locally
  - Queue for later processing
- [ ] Test crash recovery scenarios
- [x] Add logging for debugging
  - Validate graceful behavior when `ffmpeg` is missing (keep WAV only)

### Stage: Integration testing and polish
- [ ] Use subagent to run full end-to-end test session
- [ ] Test multiple sessions in single day
- [x] Verify file naming and organization
- [ ] Test with various session lengths
- [ ] Check memory/resource usage over time
- [ ] Run linting and type checking
- [x] Update documentation with usage instructions
  - Add `TESTING.md` with pytest guidance and minimal high-level tests approach

### Stage: Streaming LLM responses (late-stage)

- [ ] Research and implement streaming responses for Anthropic
  - Search the web for best practices and CLI-compatible approaches for Anthropic streaming
  - Prototype a terminal-friendly streaming renderer (spinner, incremental lines)
  - Fall back to all-at-once if streaming errors occur
  - Simulate process kill mid-session to verify immediate persistence guarantees

### Stage: Finalize V1
- [x] Create simple CLI entry point script
- [x] Write README with setup instructions
- [x] Document API key configuration
- [ ] Update any other documents as needed
- [ ] Add example session transcript
- [ ] Final test of complete flow
- [ ] Git commit with comprehensive message
- [ ] Move planning doc to planning/finished/
  - Verify Definition of Done is met

## Risks and mitigations (V1)

- External dependencies unavailable (e.g., `ffmpeg`): Proceed with WAV-only flow; surface informational notice; offer CLI flag to suppress warning.
- API rate limits and network issues: Implement retries with backoff and jitter; structured error messages; queue work for later processing.
- Data loss on crash: Immediate WAV persistence on stop; transcript and frontmatter updates happen atomically after successful STT/LLM steps; defensive file writes.
- Token overflows and runaway costs: Enforce conservative prompt budgets; truncate or summarize history aggressively when needed.
- Device inconsistencies: Detect and display selected input device; fall back to system default; document troubleshooting.

## Definition of Done (V1)

- Record, stop, persist audio (WAV) with visual meter and keypress controls.
- Transcribe via Whisper API with retries; store transcript and raw response; update paired markdown.
- Generate at least one follow-up question via Claude; display response; write Q&A to markdown.
- Update frontmatter summary after each Q&A; include recent summaries per policy without exceeding the budget.
- Handle ESC cancel and Q quit reliably; ensure files are saved on exit.
- End-to-end stop-to-question latency for a 60s clip under typical network conditions ≤ 25s.

## Out of scope for V1

- (Removed) Streaming LLM responses (addressed as a late-stage above)
- Automatic voice activity detection
- Multiple voice profiles
- Automated pattern detection beyond simple history inclusion
- Mobile or GUI apps

## Appendix

### Audio Library Considerations
Need to research:
- pyaudio: Well-established but requires PortAudio
- sounddevice: More modern, NumPy-based
- python-soundcard: Cross-platform, pure Python

### MP3 Encoding Options
- pydub + ffmpeg: Most flexible but requires external dependency
- audioread: Simpler but limited encoding options
- Direct ffmpeg subprocess: Most control but complex

### Future Enhancements (post-V1)
- Voice activity detection
- Multiple voice profiles
- Export/backup functionality
- Session analytics dashboard
- Mobile app companion
