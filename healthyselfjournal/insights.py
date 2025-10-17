from __future__ import annotations

"""Insights range selection and file writing utilities (v1).

This module implements the two-range default described in docs/reference/INSIGHTS.md:
- Historical summaries from the beginning up to the last insights output
- Recent full transcripts since the last insights output

It also provides a writer that persists a single-insight markdown file with
frontmatter provenance under [SESSIONS-DIR]/insights/yyMMdd_HHmm_insights.md.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

from .storage import load_transcript
from .events import log_event


@dataclass
class InsightsInputs:
    historical_summaries: List[str]
    recent_transcripts: List[str]
    prior_insights_excerpt: str | None
    range_text: str
    source_sessions: List[str]
    prior_insights_refs: List[str]


def _list_insights_files(sessions_dir: Path) -> List[Path]:
    insights_dir = sessions_dir / "insights"
    if not insights_dir.exists():
        return []
    return sorted(insights_dir.glob("*_insights.md"), reverse=True)


def _read_last_insights_excerpt(
    sessions_dir: Path, max_chars: int = 2000
) -> tuple[str | None, list[str]]:
    files = _list_insights_files(sessions_dir)
    if not files:
        return None, []
    latest = files[0]
    text = latest.read_text(encoding="utf-8")
    excerpt = text.strip()[-max_chars:]
    refs = [f.name for f in files[:3]]  # include a few recent refs for provenance
    return excerpt, refs


def _parse_stamp_from_name(name: str) -> str | None:
    # Expect format yyMMdd_HHmm.md
    try:
        stem = Path(name).stem
        # if name already includes suffix like _insights, strip it
        stamp = stem.split("_")[0] + "_" + stem.split("_")[1]
        # Validate it parses
        datetime.strptime(stamp, "%y%m%d_%H%M")
        return stamp
    except Exception:
        return None


def select_ranges(sessions_dir: Path, *, max_words: int = 100_000) -> InsightsInputs:
    """Return selected historical summaries and recent transcripts per v1 rules.

    - If prior insights exist, treat their timestamp as the cutoff.
    - Historical summaries include all sessions strictly before cutoff.
    - Recent transcripts include full bodies for sessions on/after cutoff.
    - Apply a rough word cap by trimming recent transcripts first if needed.
    """

    excerpt, prior_refs = _read_last_insights_excerpt(sessions_dir)
    cutoff_ts: datetime | None = None
    if prior_refs:
        # Use latest insights timestamp as cutoff (start of recent window)
        latest_name = prior_refs[0]
        try:
            stamp = latest_name.split("_insights")[0]
            cutoff_ts = datetime.strptime(stamp, "%y%m%d_%H%M")
        except Exception:
            cutoff_ts = None

    markdown_files = sorted((p for p in sessions_dir.glob("*.md")))
    historical_summaries: list[str] = []
    recent_transcripts: list[str] = []
    source_sessions: list[str] = []

    total_words = 0

    for path in markdown_files:
        doc = load_transcript(path)
        # Skip non-session insight files if any ended up at root
        if "insights" in path.name:
            continue

        # Determine the session timestamp
        stamp = _parse_stamp_from_name(path.name)
        session_ts: datetime | None = None
        if stamp:
            try:
                session_ts = datetime.strptime(stamp, "%y%m%d_%H%M")
            except Exception:
                session_ts = None

        # Classify into historical or recent window
        in_recent_window = False
        if cutoff_ts is None:
            in_recent_window = True
        elif session_ts is not None and session_ts >= cutoff_ts:
            in_recent_window = True

        if in_recent_window:
            # Prefer transcript body; if empty, fall back to summary text
            body_text = (doc.body or "").strip()
            text = (
                body_text
                if body_text
                else str(doc.frontmatter.data.get("summary") or "")
            )
            if not text:
                continue
            words = max(1, len(text.split()))
            if total_words + words > max_words and recent_transcripts:
                # Stop adding more detailed transcripts to respect budget
                break
            recent_transcripts.append(text)
            source_sessions.append(path.name)
            total_words += words
        else:
            summary = str(doc.frontmatter.data.get("summary") or "").strip()
            if summary:
                historical_summaries.append(summary)
                source_sessions.append(path.name)

    # Build human-readable range text
    since = None
    until = None
    if markdown_files:
        try:
            stamps = [
                s for s in (_parse_stamp_from_name(p.name) for p in markdown_files) if s
            ]
            if stamps:
                since = stamps[0][:6]  # yyMMdd
                until = stamps[-1][:6]
        except Exception:
            pass

    range_parts: list[str] = []
    if since and until:
        range_parts.append(f"{since} → {until}")
    if prior_refs:
        range_parts.append("since last insights")
    range_text = ", ".join(range_parts) if range_parts else "selected sessions"

    try:
        log_event(
            "insights.range.selected",
            {
                "historical_summaries": len(historical_summaries),
                "recent_transcripts": len(recent_transcripts),
                "prior_refs": prior_refs,
                "words_estimate": total_words,
            },
        )
    except Exception:
        pass

    return InsightsInputs(
        historical_summaries=historical_summaries,
        recent_transcripts=recent_transcripts,
        prior_insights_excerpt=excerpt,
        range_text=range_text,
        source_sessions=source_sessions,
        prior_insights_refs=prior_refs,
    )


def write_insight_output(
    sessions_dir: Path,
    *,
    content_markdown: str | Sequence[str],
    model_llm: str,
    source_range_text: str,
    source_sessions: Sequence[str],
    prior_insights_refs: Sequence[str],
    guidelines_version: str = "1.0",
) -> Path:
    insights_dir = sessions_dir / "insights"
    insights_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%y%m%d_%H%M")
    out_path = insights_dir / f"{stamp}_insights.md"

    fm_lines: list[str] = [
        "---",
        f'generated_at: "{datetime.utcnow().isoformat()}Z"',
        f'model_llm: "{model_llm}"',
        "source_range:",
        f'  text: "{source_range_text}"',
        "source_sessions:",
    ]
    for name in source_sessions:
        fm_lines.append(f'  - "{name}"')
    fm_lines.append("prior_insights_refs:")
    for ref in prior_insights_refs:
        fm_lines.append(f'  - "{ref}"')
    fm_lines.append(f'guidelines_version: "{guidelines_version}"')
    fm_lines.append("---")

    items: list[str]
    if isinstance(content_markdown, (list, tuple)):
        items = [str(x).strip() for x in content_markdown if str(x).strip()]
    else:
        items = [str(content_markdown).strip()] if str(content_markdown).strip() else []

    title = "# Insight — " + source_range_text if len(items) == 1 else "# Insights — " + source_range_text
    joined = "\n\n---\n\n".join(items) if items else ""

    body_lines: list[str] = [
        "",
        title,
        "",
        joined,
        "",
        "---",
        "",
        "*These observations are generated from your journaling sessions to help you notice patterns and connections. They are not therapeutic advice or diagnosis.*",
        "",
    ]

    text = "\n".join(fm_lines + body_lines)
    out_path.write_text(text, encoding="utf-8")
    try:
        log_event(
            "insights.output.written",
            {"file": out_path.name, "bytes": len(text)},
        )
    except Exception:
        pass
    return out_path
