from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .config import CONFIG
from .storage import load_transcript


console = Console()


def build_app() -> typer.Typer:
    """Build the Typer sub-app for session utilities."""

    app = typer.Typer(
        add_completion=False,
        no_args_is_help=True,
        context_settings={"help_option_names": ["-h", "--help"]},
        help="Session utilities (list, future: new, show, etc.).",
    )

    @app.command("list")
    def list_sessions(
        sessions_dir: Path = typer.Option(
            CONFIG.recordings_dir,
            "--sessions-dir",
            help="Directory where session markdown/audio files are stored.",
        ),
        nchars: int | None = typer.Option(
            None,
            "--nchars",
            help="Limit summary snippet to N characters (None = full summary).",
        ),
    ) -> None:
        """List sessions by filename stem with a summary snippet from frontmatter."""

        markdown_files = sorted((p for p in sessions_dir.glob("*.md")))
        if not markdown_files:
            console.print("[yellow]No session markdown files found.[/]")
            return

        for path in markdown_files:
            try:
                doc = load_transcript(path)
                summary_raw = doc.frontmatter.data.get("summary")
                summary_text = summary_raw if isinstance(summary_raw, str) else ""
                normalized = " ".join(summary_text.split())
                if nchars is not None and nchars > 0:
                    snippet = normalized[:nchars]
                else:
                    snippet = normalized
                body = Text(snippet) if snippet else Text("(no summary)", style="dim")
                console.print(
                    Panel.fit(
                        body,
                        title=path.stem,
                        border_style="cyan",
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive surface
                console.print(
                    Panel.fit(
                        Text(f"error reading - {exc}", style="red"),
                        title=path.name,
                        border_style="red",
                    )
                )

    return app


