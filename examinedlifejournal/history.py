from __future__ import annotations

"""Utilities for loading recent session summaries."""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import List

from .storage import load_transcript

_LOGGER = logging.getLogger(__name__)
from .events import log_event


@dataclass
class HistoricalSummary:
    filename: str
    summary: str


def load_recent_summaries(
    sessions_dir: Path,
    current_filename: str,
    limit: int,
    max_estimated_tokens: int,
) -> List[HistoricalSummary]:
    """Load summaries from prior markdown files while respecting a token budget."""

    summaries: List[HistoricalSummary] = []
    total_tokens = 0

    markdown_files = sorted(
        (p for p in sessions_dir.glob("*.md") if p.name != current_filename),
        reverse=True,
    )

    for path in markdown_files:
        if len(summaries) >= limit:
            break
        doc = load_transcript(path)
        summary = doc.frontmatter.data.get("summary")
        if not summary:
            continue
        est_tokens = _estimate_tokens(summary)
        if total_tokens + est_tokens > max_estimated_tokens and summaries:
            break
        total_tokens += est_tokens
        summaries.append(HistoricalSummary(filename=path.name, summary=summary))

    summaries.reverse()  # Oldest first for chronological context
    try:
        log_event(
            "history.summaries.loaded",
            {
                "files_scanned": len(markdown_files),
                "returned": len(summaries),
                "token_budget": max_estimated_tokens,
                "token_estimate": total_tokens,
            },
        )
    except Exception:
        # Never let logging failure break history loading
        pass
    return summaries


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Simple heuristic: 1 token ≈ 4 characters
    return max(1, int(len(text) / 4))
