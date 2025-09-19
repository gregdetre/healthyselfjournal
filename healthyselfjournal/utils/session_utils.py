from __future__ import annotations

"""Utilities for inspecting stored session artefacts."""

from pathlib import Path
import re


_BROWSER_PATTERN = re.compile(r"browser-(\d+)")


def get_max_recorded_index(session_dir: Path, session_id: str) -> int:
    """Return the highest recorded response index in ``session_dir``.

    Scans for both CLI-style ``{session_id}_NN`` files and web ``browser-XXX``
    segments regardless of extension.
    """

    cli_pattern = re.compile(rf"{re.escape(session_id)}_(\d+)")
    max_index = 0

    if not session_dir.exists():
        return 0

    for child in session_dir.iterdir():
        if not child.is_file():
            continue
        stem = child.name.split(".", 1)[0]

        cli_match = cli_pattern.search(stem)
        if cli_match:
            max_index = max(max_index, int(cli_match.group(1)))
            continue

        web_match = _BROWSER_PATTERN.search(stem)
        if web_match:
            max_index = max(max_index, int(web_match.group(1)))

    return max_index
