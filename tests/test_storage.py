from __future__ import annotations

from healthyselfjournal.history import load_recent_summaries
from healthyselfjournal.storage import (
    Frontmatter,
    TranscriptDocument,
    append_exchange_body,
    load_transcript,
    write_transcript,
)


def test_transcript_round_trip(tmp_path):
    path = tmp_path / "250101_1200.md"
    frontmatter = Frontmatter(data={"created_at": "2025-01-01T12:00:00Z"})
    doc = TranscriptDocument(frontmatter=frontmatter, body="")
    write_transcript(path, doc)

    append_exchange_body(path, "What is on your mind?", "Testing response")
    loaded = load_transcript(path)

    body = loaded.body.strip()
    assert body.startswith("## AI Q")
    # Ensure fenced llm-question block exists with the question text
    assert "```llm-question" in body
    assert "What is on your mind?" in body
    assert loaded.frontmatter.data["created_at"] == "2025-01-01T12:00:00Z"
    assert loaded.frontmatter.data["transcript_file"] == path.name


def test_recent_summary_loader_respects_limits(tmp_path):
    base = tmp_path
    summaries = [
        "Focused on gratitude today.",
        "Explored sources of stress at work.",
        "Reflected on relationships and support.",
    ]
    for idx, summary in enumerate(summaries, start=1):
        md_path = base / f"250101_120{idx}.md"
        fm = Frontmatter(data={"summary": summary})
        write_transcript(md_path, TranscriptDocument(frontmatter=fm, body=""))

    result = load_recent_summaries(
        base,
        current_filename="999999_9999.md",
        limit=2,
        max_estimated_tokens=100,
    )

    assert len(result) == 2
    assert result[0].summary == summaries[1]
    assert result[1].summary == summaries[2]
