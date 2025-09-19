# Architecture

Architectural overview of the HealthySelfJournal voice journaling application, covering the layered design, component relationships, and data flow patterns.

## See also

- `PRODUCT_VISION_FEATURES.md` - Product goals and feature specifications
- `DIALOGUE_FLOW.md` - Detailed conversation flow and state management
- `FILE_FORMATS_ORGANISATION.md` - Storage structure and data formats
- `healthyselfjournal/session.py` - Central session orchestration implementation
- `docs/planning/250917a_voice_journaling_app_v1.md` - Original architectural decisions

## Principles, key decisions

- **Voice-first design**: Minimize friction with speech input, keyboard-only controls
- **Privacy by default**: Local processing options, minimal telemetry, user-controlled data
- **Graceful degradation**: Fallback backends when dependencies or APIs unavailable
- **Dual interface**: Shared core logic between CLI and web interfaces
- **Provider agnostic**: Swappable backends for STT, TTS, and LLM services

## Architectural Layers

### 1. Interface Layer

**Components**: `cli.py`, `cli_journal_web.py`, `web/app.py`

The interface layer provides two distinct user experiences:
- **CLI**: Typer-based command-line interface with real-time audio meter
- **Web**: FastHTML server with browser-based recording and TTS

Both interfaces delegate to the same session management core, ensuring consistent behavior.

### 2. Session Orchestration

**Core**: `session.py` - `SessionManager`, `SessionState`, `Exchange`

The SessionManager orchestrates the complete dialogue lifecycle:
1. Initializes storage with markdown frontmatter
2. Manages conversation state and exchange history
3. Coordinates the audio → transcription → LLM → storage pipeline
4. Handles background summary generation
5. Tracks metrics (duration, words, exchanges)

State management follows an immutable pattern with explicit state transitions.

### 3. Service Layer

**Audio Pipeline** (`audio.py`):
- Real-time capture with sounddevice
- Visual RMS meter display
- Keyboard controls (SPACE/Q/ESC)
- Auto-discard for short/silent takes
- Background MP3 conversion via ffmpeg

**Transcription** (`transcription.py`):
- Multiple backend support (OpenAI, MLX, faster-whisper, whisper.cpp)
- Automatic fallback to local processing
- Retry mechanisms with exponential backoff
- Raw response persistence to `.stt.json`

**LLM Integration** (`llm.py`):
- Anthropic Claude for question generation
- Jinja2 template-based prompting
- Context management with recent summaries
- Streaming and non-streaming modes

**Text-to-Speech** (`tts.py`):
- OpenAI TTS for web voice mode
- Interrupt handling for natural conversation

### 4. Storage Layer

**Document Storage** (`storage.py`):
- Markdown files with YAML frontmatter
- Atomic writes with file locking
- Structured `TranscriptDocument` abstraction

**Historical Context** (`history.py`):
- Cross-session summary loading
- Token budget management
- Recent conversation retrieval

**Event Logging** (`events.py`):
- Append-only JSON Lines format
- Metadata-only telemetry (no content)
- Performance and usage metrics

## Data Flow

### Primary Session Flow

```
User Speech → Audio Capture → WAV File
                ↓
         Transcription API
                ↓
          Text + Timestamp
                ↓
         Append to Markdown
                ↓
         Generate Question (LLM)
                ↓
         Display/Speak Question
                ↓
         [Loop to User Speech]

    [Background: Generate Summary → Update Frontmatter]
    [Background: Convert WAV → MP3]
```

### Storage Organization

```
sessions/
├── events.log                    # Telemetry log
├── 241201_1430.md               # Session transcript
└── 241201_1430/                 # Audio assets
    ├── 241201_1430_001.wav      # Raw recording
    ├── 241201_1430_001.mp3      # Compressed
    └── 241201_1430_001.stt.json # Raw transcript
```

## Configuration System

**Hierarchy** (`config.py`):
1. Environment variables (highest priority)
2. Project `user_config.toml`
3. Working directory config
4. XDG config directories

**Key Configurations**:
- API keys and model selection
- Audio thresholds and timeouts
- User vocabulary for STT
- Prompt template customization

## Dependency Management

### Core Dependencies
- `anthropic` - LLM provider
- `openai` - STT/TTS provider
- `sounddevice` - Audio I/O
- `typer`/`rich` - CLI framework
- `fasthtml` - Web framework
- `jinja2` - Template engine

### Optional Components
- `ffmpeg` - Audio processing (graceful fallback)
- `readchar` - Keyboard input (fallback to simple input)
- Local STT models (MLX, faster-whisper)

## Web Interface Architecture

The web interface (`web/app.py`) implements:
- Session-per-tab with in-memory state
- RESTful endpoints for audio/TTS
- Browser MediaRecorder integration
- Server-side rendering with progressive enhancement
- Static asset serving via FastHTML

Frontend JavaScript handles:
- Real-time recording with VAD
- TTS playback and interruption
- Audio blob upload to server

## Extensibility Points

1. **Backend Providers**: Add new STT/TTS/LLM implementations
2. **Prompt Templates**: Customize questioning strategies via Jinja2
3. **Storage Formats**: Implement alternative storage backends
4. **Event Handlers**: Hook into session lifecycle events
5. **Audio Processing**: Add custom audio filters/effects

## Performance Characteristics

- **Audio**: Real-time capture with ~100ms latency
- **Transcription**: 1-5s for cloud, 2-10s for local models
- **LLM Response**: 1-3s for question generation
- **Storage**: Atomic writes, concurrent-safe
- **Memory**: ~50MB base, +10MB per active session

## Security Considerations

- API keys stored in environment/config files only
- No sensitive data in telemetry logs
- Local processing options for privacy
- User-controlled data retention
- No external analytics or tracking

## Future Architecture Considerations

**Current State**: Monolithic application with pluggable backends
**Potential Evolution**:
- Service extraction for multi-user deployment
- Queue-based background processing
- Database-backed storage for scale
- WebSocket for real-time updates

See `docs/planning/` for specific enhancement proposals.