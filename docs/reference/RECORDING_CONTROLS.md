# Recording Controls

## Overview

Keyboard controls for voice recording and session management.

## See also

- `COMMAND_LINE_INTERFACE.md` - Visual feedback during recording
- `DIALOGUE_FLOW.md` - Session flow after recording
- `FILE_FORMATS_ORGANISATION.md` - What gets saved when

## Recording Flow

1. Press any key to start recording
2. Visual volume meter shows recording active
3. Press any key to stop recording
4. Whisper transcribes to text
5. LLM generates response question

## Session Controls

- **ESC**: Cancel current recording (don't transcribe)
- **Q**: Transcribe current recording and quit session

## Error Prevention

- Audio saved immediately to .mp3
- Transcript saved after each transcription
- Summary updated after each exchange