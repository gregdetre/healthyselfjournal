from __future__ import annotations

"""Typer sub-app for insights generation and listing (v1)."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import CONFIG
from .events import log_event
from .insights import select_ranges, write_insight_output
from .llm import generate_insight, get_model_provider, InsightsRequest


console = Console()


def build_app() -> typer.Typer:
    app = typer.Typer(
        add_completion=False,
        no_args_is_help=True,
        context_settings={"help_option_names": ["-h", "--help"]},
        help="Generate and list reflective insights.",
    )

    @app.command("list")
    def list_insights(
        sessions_dir: Path = typer.Option(
            CONFIG.recordings_dir,
            "--sessions-dir",
            help="Directory where session markdown/audio files are stored.",
        ),
    ) -> None:
        """List existing insights files (newest first)."""

        insights_dir = sessions_dir / "insights"
        files = []
        if insights_dir.exists():
            files = sorted(insights_dir.glob("*_insights.md"), reverse=True)

        if not files:
            console.print(
                "[yellow]No insights found.[/] Run: healthyselfjournal insight generate"
            )
            return

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Filename")
        table.add_column("Title/First line")
        for path in files:
            # Grab first non-empty line after frontmatter
            title = ""
            try:
                raw = path.read_text(encoding="utf-8").splitlines()
                # Strip frontmatter if present
                i = 0
                if i < len(raw) and raw[i].strip() == "---":
                    i += 1
                    while i < len(raw) and raw[i].strip() != "---":
                        i += 1
                    i += 1  # skip closing ---
                # Find first non-empty line
                while i < len(raw) and not raw[i].strip():
                    i += 1
                title = raw[i].strip() if i < len(raw) else ""
            except Exception:
                title = ""
            table.add_row(path.name, title)

        console.print(table)

    @app.command("generate")
    def generate(
        sessions_dir: Path = typer.Option(
            CONFIG.recordings_dir,
            "--sessions-dir",
            help="Directory where session markdown/audio files are stored.",
        ),
        llm_model: str = typer.Option(
            CONFIG.model_llm,
            "--llm-model",
            help="LLM model string: provider:model:version[:thinking]",
        ),
        max_words: int = typer.Option(
            100_000,
            "--max-words",
            help="Rough cap for input word budget (applies to recent transcripts).",
        ),
        count: int | None = typer.Option(
            None,
            "--count",
            help="Number of insights to generate in this run (>=1). If omitted, the model may choose.",
        ),
    ) -> None:
        """Generate one or more insights and write them to the insights folder."""

        # Trigger provider resolution early for error clarity, though generation enforces cloud_off
        _ = get_model_provider(llm_model)

        inputs = select_ranges(sessions_dir, max_words=max_words)
        # Print a brief range summary
        n = 0 if count is None else max(1, int(count))
        console.print(
            (
                f"Generating auto insight count from: [cyan]{inputs.range_text}[/]"
                if n == 0
                else f"Generating {n} insight(s) from: [cyan]{inputs.range_text}[/]"
            )
            + "\n"
            f"Historical summaries: {len(inputs.historical_summaries)}; "
            f"Recent transcripts: {len(inputs.recent_transcripts)}"
        )

        guidelines = "Use tentative, descriptive language; ground observations in quotes; end with an open question."

        # Request insights once; if a count was provided, ask the model for exactly that many
        requested_count = None if n == 0 else n
        resp = generate_insight(
            InsightsRequest(
                historical_summaries=inputs.historical_summaries,
                recent_transcripts=inputs.recent_transcripts,
                prior_insights_excerpt=inputs.prior_insights_excerpt,
                range_text=inputs.range_text,
                guidelines=guidelines,
                model=llm_model,
                count=requested_count,
            )
        )

        # Split the model response into individual insights using robust separators
        insights: list[str] = []
        text = resp.insight_markdown.strip()
        if text:
            parts = [p.strip() for p in text.split("\n---\n") if p.strip()]
            if len(parts) <= 1:
                # Fallback to double-newline split if horizontal rule not used
                parts = [p.strip() for p in text.split("\n\n") if p.strip()]

            if requested_count is None:
                insights = parts if parts else ([text] if text else [])
            else:
                if parts:
                    insights = parts[:requested_count]
                else:
                    # Model returned a single block; treat as one insight
                    insights = [text]

        path = write_insight_output(
            sessions_dir,
            content_markdown=insights,
            model_llm=llm_model,
            source_range_text=inputs.range_text,
            source_sessions=inputs.source_sessions,
            prior_insights_refs=inputs.prior_insights_refs,
        )

        log_event(
            "insights.generate.completed",
            {"file": path.name, "model": llm_model, "count": len(insights)},
        )
        console.print(f"[green]Wrote:[/] {path}")

    return app
