# Command Line Interface

## Overview

Auto-start voice recording interface with visual feedback and keyboard controls.

### Launching

The journaling loop is started from the project root:

```bash
uv run examinedlifejournal journal [--sessions-dir PATH]
```

Files default to `./sessions/`; pass `--sessions-dir` to override for archival or testing.

## See also

- `RECORDING_CONTROLS.md` - Detailed key mappings and recording flow
- `PRODUCT_VISION_FEATURES.md` - Overall product vision
- `../conversations/250117a_journaling_app_ui_technical_decisions.md` - UI decisions

## Visual Feedback

Unicode block volume meter using Python `rich` library:
```
Recording started… [████████░░░░░░░░] Press any key to stop (ESC cancels, Q quits)
```

## Display Mode

- V1: All-at-once response display
- Future: Streaming word-by-word for conversational feel
