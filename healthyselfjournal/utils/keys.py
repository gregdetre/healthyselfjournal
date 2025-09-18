from __future__ import annotations

from typing import Literal


KeyName = Literal["ENTER", "ESC", "Q", "SPACE", "OTHER"]


def read_one_key_normalized() -> KeyName:
    """Read one keypress and normalize to a small set of names.

    Behavior:
    - ENTER: enter/return (\n, \r)
    - ESC: escape key or any ESC-prefixed sequence
    - Q: 'q' or 'Q'
    - SPACE: space bar
    - OTHER: any other key

    Falls back to input() if readchar is unavailable.
    """
    try:
        import readchar  # type: ignore
    except Exception:
        try:
            s = input()
        except KeyboardInterrupt:
            return "ESC"  # treat Ctrl-C as ESC/cancel in fallback
        # Map empty line to ENTER; otherwise OTHER (so callers stop)
        return "ENTER" if not s else "OTHER"

    key = readchar.readkey()
    if isinstance(key, (bytes, bytearray)):
        try:
            key = key.decode("utf-8", "ignore")
        except Exception:
            key = str(key)

    # Ctrl-C (interrupt)
    if key == "\x03" or key == getattr(readchar.key, "CTRL_C", None):
        return "ESC"

    # ENTER/Return
    if key in ("\n", "\r", getattr(readchar.key, "ENTER", "\n")):
        return "ENTER"

    # ESC or any ESC-prefixed sequence
    if key == getattr(readchar.key, "ESC", "\x1b") or (
        isinstance(key, str) and key.startswith("\x1b")
    ):
        return "ESC"

    # Space bar
    if key == getattr(readchar.key, "SPACE", " ") or key == " ":
        return "SPACE"

    # Q or q
    if str(key).lower() == "q":
        return "Q"

    return "OTHER"
