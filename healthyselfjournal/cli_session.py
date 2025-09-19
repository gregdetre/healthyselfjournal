from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import os
from typing import Iterable, MutableMapping
import contextlib
import wave

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .config import CONFIG
from .events import log_event
from .history import load_recent_summaries
from .llm import SummaryRequest, generate_summary, get_model_provider
from .storage import load_transcript, write_transcript


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

    @app.command("summaries")
    def summaries_status(
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

    # --- Merge command (moved from cli_merge.py) ---

    app.command("merge")(merge)

    return app


# ---- Implementation moved from cli_merge.py ----


@dataclass
class _SessionPaths:
    stamp: str
    markdown_path: Path
    assets_dir: Path


def _coerce_stamp_or_filename(value: str) -> str:
    """Accept bare stamp (yyMMdd_HHmm) or a markdown filename and return the stamp."""
    p = Path(value)
    name = p.name
    if name.endswith(".md"):
        name = Path(name).stem
    # Basic validation
    try:
        # Accept minute-level (yyMMdd_HHmm) or second-level (yyMMdd_HHmmss)
        try:
            datetime.strptime(name, "%y%m%d_%H%M")
        except Exception:
            datetime.strptime(name, "%y%m%d_%H%M%S")
    except Exception as exc:
        raise typer.BadParameter(
            f"Invalid stamp '{value}'. Expected format yyMMdd_HHmm (e.g., 250918_0119)."
        ) from exc
    return name


def _parse_stamp_datetime(stamp: str) -> datetime:
    """Parse a session stamp supporting minute- or second-level precision."""
    try:
        return datetime.strptime(stamp, "%y%m%d_%H%M")
    except Exception:
        return datetime.strptime(stamp, "%y%m%d_%H%M%S")


def _paths_for_stamp(sessions_dir: Path, stamp: str) -> _SessionPaths:
    return _SessionPaths(
        stamp=stamp,
        markdown_path=sessions_dir / f"{stamp}.md",
        assets_dir=sessions_dir / stamp,
    )


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    idx = 1
    while True:
        candidate = path.with_name(f"{stem}_{idx}{suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def _merge_bodies(body_earlier: str, body_later: str) -> str:
    be = (body_earlier or "").rstrip()
    bl = (body_later or "").strip()
    if not bl:
        return be + ("\n\n" if be else "")
    # If the entire later body is already contained (idempotency guard), skip append
    if bl and bl in be:
        return be + ("\n\n" if be else "")
    blocks: list[str] = []
    if be:
        blocks.append(be)
        blocks.append("")
        blocks.append("")
    blocks.append(bl)
    return ("\n".join(blocks)).rstrip() + "\n\n"


def _merge_audio_lists(
    existing: Iterable[MutableMapping], incoming: Iterable[MutableMapping]
) -> list[MutableMapping]:
    existing_list = list(existing or [])
    incoming_list = list(incoming or [])
    by_wav: dict[str, MutableMapping] = {
        str(item.get("wav")): dict(item) for item in existing_list if item.get("wav")
    }
    for seg in incoming_list:
        key = str(seg.get("wav")) if seg.get("wav") is not None else None
        if key is None:
            # Fallback: try mp3 as a unique key
            key = f"mp3:{seg.get('mp3')}"
        if key is not None:
            by_wav[key] = dict(seg)
    merged_in_order = [
        *[
            item
            for item in existing_list
            if str(item.get("wav"))
            in {
                k.split(":", 1)[-1] if k.startswith("mp3:") else k
                for k in by_wav.keys()
            }
        ],
        *[
            seg
            for seg in incoming_list
            if str(seg.get("wav")) not in {str(i.get("wav")) for i in existing_list}
        ],
    ]
    return merged_in_order


def _sum_duration_seconds(segments: Iterable[MutableMapping]) -> float:
    total = 0.0
    for item in segments or []:
        try:
            total += float(item.get("duration_seconds", 0.0) or 0.0)
        except Exception:
            continue
    return round(total, 2)


def _has_env(var_name: str) -> bool:
    return bool(os.environ.get(var_name))


def merge(
    earlier_or_a: str = typer.Argument(
        ..., help="Earlier stamp or filename (yyMMdd_HHmm[.md])."
    ),
    later_or_b: str = typer.Argument(
        ..., help="Later stamp or filename (yyMMdd_HHmm[.md])."
    ),
    sessions_dir: Path = typer.Option(
        CONFIG.recordings_dir,
        "--sessions-dir",
        help="Directory where session markdown/audio files are stored.",
    ),
    llm_model: str = typer.Option(
        CONFIG.model_llm,
        "--llm-model",
        help="LLM model for summary regeneration (if enabled).",
    ),
    regenerate: bool = typer.Option(
        True,
        "--regenerate/--no-regenerate",
        help="Regenerate the merged session's summary (default on).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run/--no-dry-run",
        help="Print actions without changing files.",
    ),
    ignore_missing: bool = typer.Option(
        False,
        "--ignore-missing/--no-ignore-missing",
        help="Proceed even if the later session's assets folder is missing.",
    ),
) -> None:
    """
    Merge two sessions, keeping the earlier one.

    This command consolidates two sessions identified by datetime-stamp (yyMMdd_HHmm)
    or filename. The earlier session is kept as the target; the later session is
    merged into it. The following steps occur:

    1) Move assets: All files from the later session's subfolder are moved into the
       earlier session's subfolder. Filename collisions are resolved by suffixing
       with `_N` (e.g., `_1`, `_2`). The later subfolder is removed if empty.
    2) Merge transcript bodies: The later `.md` body (Questions + Answers) is
       appended to the earlier `.md` body with clear spacing. If the entire later
       body already exists verbatim in the earlier body, duplication is skipped.
    3) Update frontmatter: The `audio_file` list is rebuilt to be MP3-centric
       (keys preserved; `wav` explicitly set to None, `mp3` set when available), and
       `duration_seconds` is recomputed from the merged list. `transcript_file` is
       set to the kept filename.
    4) Regenerate summary (default): If `--regenerate` is on and
       `ANTHROPIC_API_KEY` is set, a fresh summary is generated using the usual
       recent-history context and written to frontmatter.
    5) Delete later markdown: After a successful merge, the later `.md` file is
       deleted. This keeps the sessions directory tidy and avoids confusion.

    Notes and behavior:
    - Renumbering: Segment indices in filenames are not renumbered; only collisions
      are suffixed when moving. Ordering is preserved by the transcript body.
    - Duplicates: If partial overlaps exist between bodies, duplication may occur.
      This is considered acceptable and not automatically deduplicated beyond the
      full-body containment check.
    - MP3-first: Since WAVs are often deleted after safe conditions are met, the
      merged frontmatter intentionally prefers MP3 references going forward.
    - Missing assets: If the later session's assets folder is missing, the merge
      fails unless `--ignore-missing` is provided.
    - Dry run: Use `--dry-run` to preview actions without making changes.

    Examples:
      - Merge by stamps (keeps earlier):
        healthyselfjournal session merge 250918_0119 250918_085316

      - Merge explicit files and skip regeneration (preview only):
        healthyselfjournal session merge --no-regenerate --dry-run \
          sessions/250917_041214.md sessions/250918_0119.md
    """

    a_stamp = _coerce_stamp_or_filename(earlier_or_a)
    b_stamp = _coerce_stamp_or_filename(later_or_b)

    dt_a = _parse_stamp_datetime(a_stamp)
    dt_b = _parse_stamp_datetime(b_stamp)

    keep_stamp, drop_stamp = (a_stamp, b_stamp) if dt_a <= dt_b else (b_stamp, a_stamp)
    keep = _paths_for_stamp(sessions_dir, keep_stamp)
    drop = _paths_for_stamp(sessions_dir, drop_stamp)

    if not keep.markdown_path.exists():
        console.print(f"[red]Missing earlier transcript:[/] {keep.markdown_path}")
        raise typer.Exit(code=2)
    if not drop.markdown_path.exists():
        console.print(f"[red]Missing later transcript:[/] {drop.markdown_path}")
        raise typer.Exit(code=2)

    console.print(
        f"Merging [bold]{drop.stamp}[/] into [bold]{keep.stamp}[/] under {sessions_dir}"
    )

    log_event(
        "merge.start",
        {"keep": keep.markdown_path.name, "drop": drop.markdown_path.name},
    )

    # Preflight: ensure assets folders exist unless ignoring missing
    missing: list[str] = []
    if not keep.assets_dir.exists():
        missing.append(f"keep assets: {keep.assets_dir}")
    if not drop.assets_dir.exists():
        missing.append(f"later assets: {drop.assets_dir}")
    if missing and not ignore_missing:
        console.print(
            "[red]Missing required assets folder(s):[/] " + "; ".join(missing)
        )
        console.print("Re-run with --ignore-missing to proceed anyway.")
        raise typer.Exit(code=2)
    elif missing and ignore_missing:
        for item in missing:
            console.print(
                f"[yellow]Missing {item}; continuing due to --ignore-missing.[/]"
            )

    # 1) Move asset files from later subfolder into earlier subfolder
    moved_files: list[Path] = []
    if not drop.assets_dir.exists():
        if ignore_missing:
            console.print(
                f"[yellow]Later assets folder not found; continuing due to --ignore-missing:[/] {drop.assets_dir}"
            )
        else:
            console.print(f"[red]Assets folder not found:[/] {drop.assets_dir}")
            raise typer.Exit(code=2)
    else:
        keep.assets_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in drop.assets_dir.iterdir() if p.is_file())
        if not files:
            console.print(f"[yellow]No files found in {drop.assets_dir} to move.[/]")
        for src in files:
            dst = keep.assets_dir / src.name
            if dst.exists():
                dst = _next_available_path(dst)
            console.print(f"Move: {src.name} → {dst.relative_to(sessions_dir)}")
            if not dry_run:
                src.rename(dst)
                moved_files.append(dst)
        # Attempt to remove the now-empty directory
        try:
            remaining = (
                list(drop.assets_dir.iterdir()) if drop.assets_dir.exists() else []
            )
            if not dry_run and len(remaining) == 0:
                drop.assets_dir.rmdir()
                console.print(f"[green]Removed empty folder:[/] {drop.assets_dir.name}")
                log_event(
                    "merge.assets_dir.removed",
                    {"dir": drop.assets_dir.name},
                )
            elif not dry_run and remaining:
                console.print(
                    f"[yellow]Folder not empty, not removed:[/] {drop.assets_dir.name} ({len(remaining)} entries)"
                )
        except Exception:
            # Non-fatal
            pass

    # 2) Merge markdown bodies and frontmatter
    keep_doc = load_transcript(keep.markdown_path)
    drop_doc = load_transcript(drop.markdown_path)

    merged_body = _merge_bodies(keep_doc.body, drop_doc.body)

    # Build drop-side audio list: prefer frontmatter; if empty, derive from moved files
    keep_audio_list = keep_doc.frontmatter.data.get("audio_file") or []

    drop_audio_list = drop_doc.frontmatter.data.get("audio_file") or []
    if not drop_audio_list:
        derived_targets: list[Path] = []
        if moved_files:
            derived_targets = moved_files
        else:
            # Derive by scanning keep assets for files that originated from drop stamp
            if keep.assets_dir.exists():
                derived_targets = sorted(
                    p
                    for p in keep.assets_dir.iterdir()
                    if p.is_file() and p.name.startswith(f"{drop.stamp}_")
                )
        # Reduce to unique base stems and prefer mp3 entry
        by_stem: dict[str, dict] = {}
        for f in derived_targets:
            stem = f.stem.split(".")[0]
            rec = by_stem.setdefault(
                stem,
                {"wav": None, "mp3": None, "duration_seconds": 0.0},
            )
            if f.suffix.lower() == ".mp3":
                rec["mp3"] = f.name
            elif f.suffix.lower() == ".wav":
                rec["wav"] = f.name
                # Try to compute duration from WAV
                try:
                    with contextlib.closing(wave.open(str(f), "rb")) as wf:
                        frames = wf.getnframes()
                        rate = wf.getframerate()
                        if rate:
                            rec["duration_seconds"] = round(frames / float(rate), 2)
                except Exception:
                    pass
        drop_audio_list = [v for _, v in sorted(by_stem.items())]

    merged_audio_list = _merge_audio_lists(
        keep_audio_list,
        drop_audio_list,
    )
    merged_duration = _sum_duration_seconds(merged_audio_list)

    # Prefer MP3-centric entries going forward; keep structure but null out WAV
    mp3_centric_list = [
        {
            "wav": None,
            "mp3": seg.get("mp3"),
            "duration_seconds": float(seg.get("duration_seconds", 0.0) or 0.0),
        }
        for seg in merged_audio_list
    ]

    # Update keep_doc frontmatter
    keep_doc.body = merged_body
    fm = keep_doc.frontmatter.data
    fm["audio_file"] = mp3_centric_list
    fm["duration_seconds"] = merged_duration
    fm["transcript_file"] = keep.markdown_path.name
    # Preserve keep summary for now; will optionally regenerate below

    console.print(
        f"Update: {keep.markdown_path.name} — {len(merged_audio_list)} segments, total {merged_duration:.2f}s"
    )
    if not dry_run:
        write_transcript(keep.markdown_path, keep_doc)
        log_event(
            "merge.transcript.updated",
            {
                "file": keep.markdown_path.name,
                "segments": len(merged_audio_list),
                "duration_seconds": merged_duration,
            },
        )

    # 3) Optionally regenerate summary for the merged transcript
    provider = get_model_provider(llm_model)

    if regenerate:
        if provider == "anthropic" and not _has_env("ANTHROPIC_API_KEY"):
            console.print(
                "[yellow]Skipping summary regeneration: ANTHROPIC_API_KEY not set.[/]"
            )
        else:
            try:
                recents = load_recent_summaries(
                    sessions_dir,
                    current_filename=keep.markdown_path.name,
                    limit=CONFIG.max_recent_summaries,
                    max_estimated_tokens=CONFIG.max_history_tokens,
                )
                history_text = [item.summary for item in recents]
                if not dry_run:
                    response = generate_summary(
                        SummaryRequest(
                            transcript_markdown=keep_doc.body,
                            recent_summaries=history_text,
                            model=llm_model,
                        )
                    )
                    latest = load_transcript(keep.markdown_path)
                    latest.frontmatter.data["summary"] = response.summary_markdown
                    write_transcript(keep.markdown_path, latest)
                    log_event(
                        "merge.summary.regenerated",
                        {"file": keep.markdown_path.name, "model": response.model},
                    )
                    console.print(
                        f"[green]Summary regenerated:[/] {keep.markdown_path.name}"
                    )
            except Exception as exc:  # pragma: no cover - defensive surface
                log_event(
                    "merge.summary.error",
                    {"file": keep.markdown_path.name, "error": str(exc)},
                )
                console.print(
                    f"[yellow]Summary regeneration failed:[/] {exc}. Continuing without blocking merge."
                )

    # 4) Remove the merged (later) markdown file now that content is extracted
    if not dry_run:
        try:
            drop.markdown_path.unlink(missing_ok=True)
            console.print(
                f"[green]Removed merged markdown:[/] {drop.markdown_path.name}"
            )
            log_event(
                "merge.source_markdown.deleted",
                {"file": drop.markdown_path.name},
            )
        except Exception as exc:
            console.print(
                f"[yellow]Could not delete source markdown:[/] {drop.markdown_path.name} — {exc}"
            )

    console.print(
        f"[green]Merge complete.[/] Kept {keep.markdown_path.name}; review {drop.markdown_path.name} as needed."
    )
    log_event(
        "merge.complete",
        {"kept": keep.markdown_path.name, "dropped": drop.markdown_path.name},
    )
