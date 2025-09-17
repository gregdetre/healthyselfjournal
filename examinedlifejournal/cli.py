from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from . import __version__
from .config import CONFIG
from .storage import load_transcript, write_transcript
from .session import SessionConfig, SessionManager
from .events import init_event_logger, log_event, get_event_log_path
from .history import load_recent_summaries
from .llm import SummaryRequest, generate_summary

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

console = Console()

# Sub-app for summaries utilities
summaries_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Utilities for listing and backfilling session summaries.",
)
app.add_typer(summaries_app, name="summaries")


@app.command()
def journal(
    sessions_dir: Path = typer.Option(
        CONFIG.recordings_dir,
        "--sessions-dir",
        help="Directory where session markdown/audio files are stored.",
    ),
    llm_model: str = typer.Option(
        CONFIG.model_llm,
        "--llm-model",
        help="LLM model spec (provider:model:version).",
    ),
    stt_model: str = typer.Option(
        CONFIG.model_stt,
        "--stt-model",
        help="Whisper/STT model identifier for OpenAI.",
    ),
    opening_question: str = typer.Option(
        CONFIG.opening_question,
        "--opening-question",
        help="Initial question used to start each session.",
    ),
    language: str = typer.Option(
        "en",
        "--language",
        help="Primary language for transcription and LLM guidance.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume the most recent session in the sessions directory.",
    ),
    delete_wav_when_safe: bool = typer.Option(
        True,
        "--delete-wav-when-safe/--keep-wav",
        help="Delete WAV after MP3+STT exist (saves disk).",
    ),
) -> None:
    """Run the interactive voice journaling session."""

    _require_env("OPENAI_API_KEY")
    _require_env("ANTHROPIC_API_KEY")

    # Initialize append-only metadata event logger
    init_event_logger(sessions_dir)
    log_event(
        "cli.start",
        {
            "sessions_dir": sessions_dir,
            "model_llm": llm_model,
            "model_stt": stt_model,
            "language": language,
            "events_log": str(get_event_log_path() or ""),
            "app_version": __version__,
            "resume": resume,
        },
    )

    # Propagate config flag
    from .config import CONFIG as _CFG

    _CFG.delete_wav_when_safe = bool(delete_wav_when_safe)

    session_cfg = SessionConfig(
        base_dir=sessions_dir,
        llm_model=llm_model,
        stt_model=stt_model,
        opening_question=opening_question,
        language=language,
    )
    manager = SessionManager(session_cfg)

    # Determine whether to start new or resume recent session
    if resume:
        markdown_files = sorted((p for p in sessions_dir.glob("*.md")), reverse=True)
        if not markdown_files:
            console.print(
                Panel.fit(
                    "No prior sessions found. Starting a new session.",
                    title="Examined Life Journal",
                    border_style="magenta",
                )
            )
            state = manager.start()
            question = opening_question
        else:
            latest_md = markdown_files[0]
            state = manager.resume(latest_md)
            doc = load_transcript(state.markdown_path)
            if doc.body.strip():
                try:
                    next_q = manager.generate_next_question(doc.body)
                    question = next_q.question
                except Exception as exc:
                    console.print(f"[red]Question generation failed:[/] {exc}")
                    question = opening_question
            else:
                question = opening_question
            console.print(
                Panel.fit(
                    f"Resuming session {state.session_id}. Recording starts immediately.\n"
                    "Press any key to stop. ESC cancels the current take; Q saves then ends after this entry.",
                    title="Examined Life Journal",
                    border_style="magenta",
                )
            )
    else:
        console.print(
            Panel.fit(
                "Voice journaling session starting. Recording starts immediately.\n"
                "Press any key to stop. ESC cancels the current take; Q saves then ends after this entry.",
                title="Examined Life Journal",
                border_style="magenta",
            )
        )
        state = manager.start()
        question = opening_question

    try:
        while True:
            console.print(f"[bold magenta]AI:[/] {question}")

            try:
                exchange = manager.record_exchange(question, console)
            except Exception as exc:  # pragma: no cover - runtime error surface
                console.print(f"[red]Error during recording/transcription:[/] {exc}")
                log_event(
                    "cli.error",
                    {
                        "where": "record_exchange",
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                    },
                )
                break

            if exchange is None:
                # Could be cancelled or discarded short answer; message accordingly
                if manager.state and manager.state.quit_requested:
                    console.print(
                        "[cyan]Quit requested. Ending session after summary update.[/]"
                    )
                    break
                # Provide clearer feedback depending on disposition flags
                if manager.state and getattr(manager.state, "last_cancelled", False):
                    console.print(
                        "[yellow]Cancelled. Take discarded. Re-asking the same question...[/]"
                    )
                    # Reset flag to avoid stale messaging on next loop
                    manager.state.last_cancelled = False
                elif manager.state and getattr(
                    manager.state, "last_discarded_short", False
                ):
                    console.print(
                        "[yellow]Very short/quiet; take discarded. Re-asking the same question...[/]"
                    )
                    manager.state.last_discarded_short = False
                else:
                    console.print(
                        "[yellow]No usable answer captured (cancelled or very short). Re-asking...[/]"
                    )
                continue

            console.print(
                Panel.fit(exchange.transcript, title="You", border_style="green")
            )

            try:
                # Background scheduling to reduce latency
                manager.schedule_summary_regeneration()
            except Exception as exc:
                console.print(f"[red]Summary scheduling failed:[/] {exc}")
                log_event(
                    "cli.error",
                    {
                        "where": "schedule_summary_regeneration",
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                    },
                )

            transcript_doc = load_transcript(state.markdown_path)

            if exchange.audio.quit_after:
                console.print(
                    "[cyan]Quit requested. Ending session after summary update.[/]"
                )
                break

            try:
                next_question = manager.generate_next_question(transcript_doc.body)
            except Exception as exc:
                console.print(f"[red]Question generation failed:[/] {exc}")
                log_event(
                    "cli.error",
                    {
                        "where": "generate_next_question",
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                    },
                )
                break

            question = next_question.question
            console.print(
                Panel.fit(question, title="Next Question", border_style="cyan")
            )

    except KeyboardInterrupt:
        console.print("[red]Session interrupted by user.[/]")
    finally:
        manager.complete()
        log_event(
            "cli.end",
            {
                "transcript_file": state.markdown_path.name,
                "session_id": state.session_id,
            },
        )
        console.print(
            Panel.fit(
                f"Session saved to {state.markdown_path.name}",
                title="Session Complete",
                border_style="magenta",
            )
        )


@app.command()
def legacy_transcribe() -> None:
    """Temporary bridge to the legacy ffmpeg-based transcription CLI."""
    console.print(
        "[yellow]The legacy transcription workflow has moved to `legacy_transcribe_cli.py`."
    )
    console.print(
        "Run `python legacy_transcribe_cli.py --help` for the previous ffmpeg interface."
    )


def _require_env(var_name: str) -> None:
    if not os.environ.get(var_name):
        console.print(f"[red]Environment variable {var_name} is required.[/]")
        raise typer.Exit(code=2)


@summaries_app.command("list")
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
            status = "missing" if not has_summary else f"present ({len(summary)} chars)"
            console.print(f"{path.name}: {status}")
            shown += 1
        except Exception as exc:  # pragma: no cover - defensive surface
            console.print(f"[red]{path.name}: error reading - {exc}[/]")

    if shown == 0 and missing_only:
        console.print("[green]All sessions have summaries.[/]")


@summaries_app.command("backfill")
def summaries_backfill(
    sessions_dir: Path = typer.Option(
        CONFIG.recordings_dir,
        "--sessions-dir",
        help="Directory where session markdown/audio files are stored.",
    ),
    llm_model: str = typer.Option(
        CONFIG.model_llm,
        "--llm-model",
        help="LLM model spec (provider:model:version).",
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

    _require_env("ANTHROPIC_API_KEY")

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

            # Reload to merge with any external edits between read and write
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
