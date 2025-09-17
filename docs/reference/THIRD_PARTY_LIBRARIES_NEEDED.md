# Third-Party Libraries Needed

## Overview

Required external libraries and functionality for the voice journaling app V1.

## See also

- `PRODUCT_VISION_FEATURES.md` - Core features requiring these libraries
- `COMMAND_LINE_INTERFACE.md` - CLI visual requirements
- `../planning/250117a_voice_journaling_app_v1.md` - Implementation plan

## Audio Recording

**Functionality needed:**
- Real-time audio capture from system microphone
- Cross-platform support (macOS, Windows, Linux)
- Volume level monitoring for visual feedback
- Non-blocking recording control

**Library options to evaluate:**
- pyaudio - Requires PortAudio system dependency
- sounddevice - NumPy-based, more modern API
- python-soundcard - Pure Python, no system dependencies

## Audio Encoding

**Functionality needed:**
- Save audio buffer to MP3 format
- Configurable bitrate and quality
- Efficient encoding without blocking

**Library options to evaluate:**
- pydub + ffmpeg - Full-featured but requires ffmpeg installation
- audioread - Simpler API, limited options
- Direct ffmpeg subprocess - Most control, complex implementation


## CLI Interface

Confirmed:
- Typer
- Rich


## API Clients

**Confirmed libraries:**
- **openai** - Official OpenAI client for Whisper API
- **anthropic** - Official Anthropic client for Claude API
  - Future: streaming response support

## Template Engine

**Confirmed library:**
- **jinja2** - Prompt template rendering
  - Variable substitution
  - Conditional logic
  - Template inheritance

## File Processing

**Required functionality:**
- **pyyaml** or **python-frontmatter** - Parse/update markdown frontmatter
- Standard library sufficient for markdown text manipulation

## Development Dependencies

**Testing and quality:**
- pytest - Test framework
- mypy - Type checking
- ruff or black - Code formatting
- pre-commit - Git hooks for quality checks
