from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from .config import CONFIG
from .events import log_event
from .history import load_recent_summaries
from .llm import SummaryRequest, generate_summary, get_model_provider
from .storage import load_transcript, write_transcript


console = Console()


def build_app() -> typer.Typer:
    app = typer.Typer(
        add_completion=False,
        no_args_is_help=True,
        context_settings={"help_option_names": ["-h", "--help"]},
        help="Utilities for listing and backfilling session summaries.",
    )

    @app.command("list")
    def summaries_list(
        sessions_dir: Path = typer.Option(
            CONFIG.recordings_dir,
            "--sessions-dir",
            help="Directory where session markdown/audio files are stored.",
        ),
        missing_only: bool = typer.Option(
            True,
            "--missing-only/--all",
            help="Only show sessions without summaries (default). Use --all to show all.",
        ),
    ) -> None:
        """List session markdown files and whether they have summaries."""

        markdown_files = sorted((p for p in sessions_dir.glob("*.md")))
        if not markdown_files:
            console.print("[yellow]No session markdown files found.[/]")
            return

        shown = 0
        for path in markdown_files:
            try:
                doc = load_transcript(path)
                summary = (doc.frontmatter.data.get("summary") or "").strip()
                has_summary = bool(summary)
                if missing_only and has_summary:
                    continue
                status = (
                    "missing" if not has_summary else f"present ({len(summary)} chars)"
                )
                console.print(f"{path.name}: {status}")
                shown += 1
            except Exception as exc:  # pragma: no cover - defensive surface
                console.print(f"[red]{path.name}: error reading - {exc}[/]")

        if shown == 0 and missing_only:
            console.print("[green]All sessions have summaries.[/]")

    @app.command("backfill")
    def summaries_backfill(
        sessions_dir: Path = typer.Option(
            CONFIG.recordings_dir,
            "--sessions-dir",
            help="Directory where session markdown/audio files are stored.",
        ),
        llm_model: str = typer.Option(
            CONFIG.model_llm,
            "--llm-model",
            help="LLM model string: provider:model:version[:thinking] (e.g., anthropic:claude-sonnet-4:20250514:thinking)",
        ),
        missing_only: bool = typer.Option(
            True,
            "--missing-only/--all",
            help="Only process sessions without summaries (default). Use --all to regenerate all.",
        ),
        limit: int = typer.Option(
            0,
            "--limit",
            help="Maximum number of files to backfill (0 = no limit).",
        ),
    ) -> None:
        """Generate summaries for any sessions missing them, in place."""

        provider = get_model_provider(llm_model)
        if provider == "anthropic" and not _has_env("ANTHROPIC_API_KEY"):
            console.print("[red]Environment variable ANTHROPIC_API_KEY is required.[/]")
            raise typer.Exit(code=2)

        markdown_files = sorted((p for p in sessions_dir.glob("*.md")))
        if not markdown_files:
            console.print("[yellow]No session markdown files found.[/]")
            return

        updated = 0
        skipped = 0

        for path in markdown_files:
            try:
                doc = load_transcript(path)
                existing = (doc.frontmatter.data.get("summary") or "").strip()
                if missing_only and existing:
                    skipped += 1
                    continue

                recents = load_recent_summaries(
                    sessions_dir,
                    current_filename=path.name,
                    limit=CONFIG.max_recent_summaries,
                    max_estimated_tokens=CONFIG.max_history_tokens,
                )
                history_text = [item.summary for item in recents]

                response = generate_summary(
                    SummaryRequest(
                        transcript_markdown=doc.body,
                        recent_summaries=history_text,
                        model=llm_model,
                    )
                )

                latest = load_transcript(path)
                latest.frontmatter.data["summary"] = response.summary_markdown
                write_transcript(path, latest)

                log_event(
                    "summaries.backfill.updated",
                    {"file": path.name, "model": response.model},
                )
                action_word = "Backfilled" if not existing else "Regenerated"
                console.print(f"[green]{action_word}:[/] {path.name}")
                updated += 1

                if limit and updated >= limit:
                    break

            except Exception as exc:  # pragma: no cover - defensive surface
                log_event(
                    "summaries.backfill.error",
                    {"file": path.name, "error": str(exc)},
                )
                console.print(f"[red]Failed to backfill {path.name}:[/] {exc}")

        console.print(
            f"Completed. Updated {updated} file(s); skipped {skipped} with existing summaries."
        )

    @app.command("regenerate")
    def summaries_regenerate(
        sessions_dir: Path = typer.Option(
            CONFIG.recordings_dir,
            "--sessions-dir",
            help="Directory where session markdown/audio files are stored.",
        ),
        filename: str = typer.Argument(
            ..., help="Target session markdown file or stamp (yyMMdd_HHmm[.md])."
        ),
        llm_model: str = typer.Option(
            CONFIG.model_llm,
            "--llm-model",
            help="LLM model string: provider:model:version[:thinking]",
        ),
    ) -> None:
        """Regenerate summary for a single session file (overwrite existing)."""

        def _resolve_path(sdir: Path, name: str) -> Path:
            p = Path(name)
            if p.suffix == "":
                p = p.with_suffix(".md")
            if p.name and p.parent == Path("."):
                p = sdir / p.name
            return p

        provider = get_model_provider(llm_model)
        if provider == "anthropic" and not _has_env("ANTHROPIC_API_KEY"):
            console.print("[red]Environment variable ANTHROPIC_API_KEY is required.[/]")
            raise typer.Exit(code=2)

        target = _resolve_path(sessions_dir, filename)
        if not target.exists():
            console.print(f"[red]File not found:[/] {target}")
            raise typer.Exit(code=2)

        doc = load_transcript(target)
        recents = load_recent_summaries(
            sessions_dir,
            current_filename=target.name,
            limit=CONFIG.max_recent_summaries,
            max_estimated_tokens=CONFIG.max_history_tokens,
        )
        history_text = [item.summary for item in recents]
        response = generate_summary(
            SummaryRequest(
                transcript_markdown=doc.body,
                recent_summaries=history_text,
                model=llm_model,
            )
        )
        latest = load_transcript(target)
        latest.frontmatter.data["summary"] = response.summary_markdown
        write_transcript(target, latest)
        log_event(
            "summaries.regenerate.updated",
            {"file": target.name, "model": response.model},
        )
        console.print(f"[green]Regenerated:[/] {target.name}")

    return app


def _has_env(var_name: str) -> bool:
    import os

    return bool(os.environ.get(var_name))
