# Recording Controls

## Overview

Keyboard controls for voice recording and session management.

## See also

- `COMMAND_LINE_INTERFACE.md` - Visual feedback during recording
- `DIALOGUE_FLOW.md` - Session flow after recording
- `FILE_FORMATS_ORGANISATION.md` - What gets saved when

## Recording Flow

1. Recording starts immediately on command launch
2. Visual volume meter shows recording active
3. Press SPACE to pause/resume; press any other key to stop recording (when paused, the volume meter is hidden and a "Paused" label is shown)
4. If the captured answer is extremely short and low‑voiced, it is discarded automatically (no files saved, no transcription)
5. Otherwise, Whisper transcribes to text
6. LLM generates response question

## Session Controls

- **ESC**: Cancel current recording (don't save or transcribe)
- **SPACE**: Pause/resume the recording (paused audio is not saved; meter hidden while paused)
- **Q**: Quit after this response
  - If pressed before any substantial speech: the take is auto‑discarded and the session ends cleanly
  - If pressed after a normal answer: the take is saved/transcribed, then the session ends

## Error Prevention

- Short accidental takes are auto‑discarded
- Audio saved to .wav; optional .mp3 conversion queued when available
- Transcript saved after each transcription
- Summary updated after each exchange