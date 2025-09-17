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
- `docs/conversations/250117a_journaling_app_ui_technical_decisions.md` - Technical decisions from planning session
- `docs/research/JOURNALLING_SCIENTIFIC_EVIDENCE_RESEARCH.md` - Evidence base for design decisions

## Principles, key decisions

- Start with all-at-once response display (streaming later)
- Use flat directory structure with paired .mp3/.md files
- Regenerate summary after each Q&A for crash resilience
- Default to same opener question with override capability
- Include current session + recent summaries in context
- Transparent mode switching based on emotional context

## Open Questions/Concerns

**Audio Library Selection**
- Need cross-platform audio recording library (pyaudio? sounddevice?)
- MP3 encoding options (pydub? direct ffmpeg?)
- Real-time volume monitoring implementation

**Context Window Management**
- How many recent summaries to include initially? (Start with last 10?)
- Maximum token count before truncation needed?

**Error Handling**
- Whisper API failures - fallback strategy?
- Claude API rate limits - retry logic?
- Audio device unavailable - graceful degradation?

**Session Boundaries**
- Should we implement 15-20 minute warning based on research?
- Pattern detection threshold for suggesting breaks?

## Stages & actions

### Stage: Environment setup and dependencies
- [ ] Install core dependencies (rich, anthropic, openai, jinja2, pyyaml)
- [ ] Research and install audio recording library (pyaudio vs sounddevice)
  - Test cross-platform compatibility
  - Verify real-time volume monitoring capability
- [ ] Install MP3 encoding solution (pydub + ffmpeg vs alternatives)
- [ ] Create basic project structure and .gitignore

### Stage: Basic audio recording with visual feedback
- [ ] Write test for audio recording functionality
- [ ] Implement press-any-key to start recording
- [ ] Create volume meter visualization using rich
  - Unicode blocks showing real-time audio levels
  - "Recording... [████████░░░░░░░░] Press any key to stop"
- [ ] Implement press-any-key to stop recording
- [ ] Save audio to MP3 with yyMMdd_HHmm timestamp
- [ ] Test recording on different audio devices

### Stage: Whisper transcription integration
- [ ] Write test for transcription pipeline
- [ ] Set up OpenAI API client with Whisper
- [ ] Implement audio file upload to Whisper API
- [ ] Handle transcription response and errors
- [ ] Save initial transcript to markdown file
- [ ] Test with various audio qualities and accents

### Stage: Basic LLM dialogue with Claude
- [ ] Write test for Claude API integration
- [ ] Set up Anthropic Claude API client
- [ ] Create basic Jinja2 prompt template
  - Include current transcript
  - Add default opening question
- [ ] Implement question generation after transcription
- [ ] Display AI response as text output
- [ ] Update markdown with Q&A format
  - "## AI Q: " prefix for questions
  - User response as section content

### Stage: Session management and controls
- [ ] Write tests for session control flow
- [ ] Implement ESC key detection (cancel recording)
- [ ] Implement Q key detection (transcribe and quit)
- [ ] Add continuous dialogue loop
  - Record → Transcribe → Generate question → Display → Loop
- [ ] Handle session termination gracefully
- [ ] Ensure all files saved before exit

### Stage: Summary generation and frontmatter
- [ ] Write tests for summary generation
- [ ] Create summary prompt template for Claude
- [ ] Generate initial summary after first Q&A
- [ ] Update summary after each subsequent Q&A
- [ ] Implement frontmatter read/write with pyyaml
  - Parse existing frontmatter
  - Update summary field
  - Preserve other metadata if present
- [ ] Test summary quality and relevance

### Stage: Context management with previous sessions
- [ ] Write tests for context loading
- [ ] Implement file discovery for recent sessions
  - Sort by timestamp in filename
  - Load last N session files
- [ ] Extract summaries from frontmatter
- [ ] Update Jinja2 template to include context
  - Add recent_summaries variable
  - Format for LLM consumption
- [ ] Test pattern detection across sessions

### Stage: Question variety and adaptation
- [ ] Create initial question bank
  - Concrete/specific questions
  - Open/exploratory questions
  - Pattern-interrupting questions
- [ ] Implement "Give me a question" detection
- [ ] Add question selection logic to template
- [ ] Test question variety over multiple sessions
- [ ] Document question categories for future expansion

### Stage: Error handling and resilience
- [ ] Add try/catch for audio device errors
- [ ] Implement Whisper API retry logic
- [ ] Add Claude API rate limit handling
- [ ] Create fallback for network failures
  - Save audio/transcript locally
  - Queue for later processing
- [ ] Test crash recovery scenarios
- [ ] Add logging for debugging

### Stage: Integration testing and polish
- [ ] Use subagent to run full end-to-end test session
- [ ] Test multiple sessions in single day
- [ ] Verify file naming and organization
- [ ] Test with various session lengths
- [ ] Check memory/resource usage over time
- [ ] Run linting and type checking
- [ ] Update documentation with usage instructions

### Stage: Finalize V1
- [ ] Create simple CLI entry point script
- [ ] Write README with setup instructions
- [ ] Document API key configuration
- [ ] Add example session transcript
- [ ] Final test of complete flow
- [ ] Git commit with comprehensive message
- [ ] Move planning doc to planning/finished/

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
- Streaming LLM responses
- Voice activity detection
- Multiple voice profiles
- Export/backup functionality
- Session analytics dashboard
- Mobile app companion