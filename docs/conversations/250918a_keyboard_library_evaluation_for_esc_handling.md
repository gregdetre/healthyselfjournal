---
Date: 2025-09-18
Duration: ~15 minutes
Type: Problem-solving
Status: Resolved
Related Docs: docs/reference/RECORDING_CONTROLS.md
---

# Keyboard Library Evaluation for ESC Key Handling - 250918

## Context & Goals

The user reported: "We've been having trouble with capturing the ESC key reliably" in the examinedlifejournal application's recording controls. They requested evaluation of Python CLI libraries for keyboard shortcuts, following their third-party library selection criteria.

## Key Background

Current implementation uses `readchar` (v4.0.6+) for keyboard input in `examinedlifejournal/audio.py`. The ESC key is meant to cancel recording without discarding audio, but there are reliability issues with detecting ESC keypresses across different terminals.

## Main Discussion

### Current Implementation Analysis

The application currently uses `readchar` with fallback handling:
- Primary keyboard library: `readchar>=4.0.6`
- Graceful degradation if readchar import fails
- ESC key detection attempts multiple approaches: `getattr(readchar.key, "ESC", "\x1b")`
- Also handles Ctrl-C, SPACE for pause/resume, Q for quit

### Root Cause: Terminal Escape Sequences

Research revealed the fundamental issue with ESC key detection:
- **Unix/Linux terminals report special keys as escape sequences** starting with `\x1b` (ESC character)
- **readchar's `readkey()` hangs on ESC** because it anticipates more characters (expecting an escape sequence)
- This is "unfortunately a common Unix problem, with no truly reliable solution" according to multiple sources
- The ambiguity between standalone ESC and beginning of escape sequence makes detection inherently difficult

### Libraries Evaluated

**1. keyboard**
- Pure Python, zero dependencies
- **Deal-breaker**: Requires root/sudo on macOS/Linux
- "Acts like a keylogger on Windows"

**2. pynput**
- Cross-platform with simple API
- **Deal-breaker**: Requires root on macOS, "not trusted on macOS"

**3. blessed**
- Active since 2014, well-maintained (3.2k GitHub stars)
- Cross-platform: Windows, Mac, Linux, BSD
- **Key advantage**: Better escape sequence handling via `Terminal.inkey()` with timeout parameter
- No root privileges required
- Rich ecosystem with extensive documentation

**4. getkey**
- Based on getch-like unbuffered reading
- **Issues**: Not actively maintained, broken pip install on Windows

**5. prompt-toolkit**
- Powerful for full REPL applications
- Overkill for simple keypress detection

**6. curses**
- Traditional Unix solution
- **Limitation**: Not cross-platform (no Windows support)

## Decisions Made

### Primary Recommendation: blessed

The user was advised to adopt **blessed** as the primary keyboard input library with the following implementation approach:

```python
from blessed import Terminal

term = Terminal()
with term.cbreak():  # Raw mode, no echo
    key = term.inkey(timeout=0.1)  # Timeout helps distinguish ESC from escape sequences

    if key.is_sequence:
        if key.name == 'KEY_ESCAPE':
            # Handle ESC
```

**Key insight**: "Use blessed's timeout feature to handle ESC reliably (0.05-0.1s timeout distinguishes ESC from escape sequences)"

### Migration Strategy

1. Add blessed to dependencies alongside readchar
2. Try blessed first, fall back to readchar if import fails
3. Use blessed's timeout feature (0.05-0.1s) to distinguish ESC from escape sequences
4. Maintain backward compatibility with existing readchar implementation

## Open Questions

- Optimal timeout value for ESC detection (0.05s vs 0.1s) may need tuning based on user testing
- Whether to eventually deprecate readchar once blessed proves stable

## Next Steps

- Add blessed to project dependencies
- Implement blessed with readchar fallback
- Test ESC key detection across different terminal environments

## Sources & References

**Library Documentation:**
- **blessed** ([PyPI](https://pypi.org/project/blessed/)) - Terminal library with elegant keyboard handling
- **readchar** ([PyPI](https://pypi.org/project/readchar/)) - Current implementation
- **keyboard** ([PyPI](https://pypi.org/project/keyboard/)) - Requires root privileges

**Research & Issues:**
- **ESC key problem in readchar** ([GitHub Issue](https://github.com/magmax/python-readchar/issues/2)) - Core issue with escape sequences
- **Cross-platform keyboard input discussion** ([Python.org](https://discuss.python.org/t/cross-platform-keyboard-input/51979)) - Community perspectives

**Selection Criteria:**
- `gjdutils/docs/instructions/THIRD_PARTY_LIBRARY_SELECTION.md` - Library evaluation framework used

## Related Work

- `docs/reference/RECORDING_CONTROLS.md` - Current recording control documentation
- `examinedlifejournal/audio.py` - Implementation requiring keyboard input improvements