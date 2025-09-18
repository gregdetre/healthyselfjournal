from __future__ import annotations

"""Time and duration formatting utilities.

These helpers standardize human-friendly duration strings across the app.
"""


def format_hh_mm_ss(total_seconds: float) -> str:
    """Return H:MM:SS (hours omitted when zero) for a given duration in seconds.

    - Rounds to nearest second.
    - Clamps at zero on negative input.
    """
    seconds = int(round(max(0.0, float(total_seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_mm_ss(total_seconds: float) -> str:
    """Return M:SS for a given duration in seconds.

    - Rounds to nearest second.
    - Clamps at zero on negative input.
    """
    seconds = int(round(max(0.0, float(total_seconds))))
    minutes, secs = divmod(seconds, 60)
    return f"{minutes}:{secs:02d}"
