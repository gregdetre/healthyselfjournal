# Command Line Interface

## Overview

Press-to-record voice interface with visual feedback and keyboard controls.

## See also

- `RECORDING_CONTROLS.md` - Detailed key mappings and recording flow
- `PRODUCT_VISION_FEATURES.md` - Overall product vision
- `../conversations/250117a_journaling_app_ui_technical_decisions.md` - UI decisions

## Visual Feedback

Unicode block volume meter using Python `rich` library:
```
Recording... [████████░░░░░░░░] Press any key to stop
```

## Display Mode

- V1: All-at-once response display
- Future: Streaming word-by-word for conversational feel