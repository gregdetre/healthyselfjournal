from __future__ import annotations

"""Shared interactive microphone check utility."""

from dataclasses import dataclass
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.text import Text

from .audio import record_response
from .transcription import (
    BackendSelection,
    apply_transcript_formatting,
    create_transcription_backend,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rich.console import Console


@dataclass
class MicCheckOptions:
    language: str = "en"
    stt_formatting: str = "sentences"
    seconds: float = 3.0
    sample_rate: int = 16_000


def run_interactive_mic_check(
    selection: BackendSelection,
    *,
    console: "Console",
    language: str,
    stt_formatting: str,
    seconds: float = 3.0,
    sample_rate: int = 16_000,
) -> None:
    """Run an interactive mic check loop until user accepts or quits.

    Behavior:
    - Records a short clip to a temp directory and transcribes with the given STT selection
    - Displays the transcript (formatted) and simple backend info
    - Prompts the user: ENTER to accept and continue; ESC to try again; q to quit
    - Deletes all temporary artifacts regardless of outcome
    - Raises typer.Exit(code=0) if the user chooses to quit
    """

    try:
        import typer  # type: ignore
    except Exception:  # pragma: no cover - fallback import guard
        typer = None  # type: ignore

    # Construct a transcription backend matching current selection
    backend = create_transcription_backend(selection)

    while True:
        console.print(
            Panel.fit(
                "Mic check: speak a few words. We'll record briefly and show the transcript.\n"
                "Press ENTER to continue, ESC to try again, or q to quit.",
                title="Mic Check",
                border_style="magenta",
            )
        )

        tmp_dir = Path(tempfile.mkdtemp(prefix="elj_miccheck_"))
        try:
            capture = record_response(
                tmp_dir,
                base_filename="miccheck",
                console=console,
                sample_rate=sample_rate,
                ffmpeg_path=None,
                print_saved_message=False,
                convert_to_mp3=False,
                max_seconds=float(seconds),
                enforce_short_answer_guard=False,
            )

            if capture.cancelled:
                # If user pressed ESC during capture, just retry automatically
                continue

            # Transcribe and show formatted transcript without persisting
            transcription = backend.transcribe(capture.wav_path, language=language)
            try:
                formatted = apply_transcript_formatting(
                    transcription.text, stt_formatting
                )
            except Exception:
                formatted = transcription.text.strip()

            console.print()
            console.print(
                Panel.fit(
                    formatted or "(no speech detected)",
                    title="Mic Check Transcript",
                    border_style="green",
                )
            )
            console.print(
                Text(
                    f"Backend: {selection.backend_id}  Model: {selection.model}",
                    style="dim",
                )
            )
        finally:
            # Remove artifacts regardless of outcome
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

        # Await user decision
        console.print(
            Text(
                "Press ENTER to continue, ESC to try again, or q to quit…",
                style="cyan",
            )
        )

        # Use shared key normalization utility
        try:
            from .utils.keys import read_one_key_normalized
        except Exception:
            read_one_key_normalized = None  # type: ignore

        if read_one_key_normalized is None:
            # Fallback: on raw input, treat non-empty as retry, 'q' to quit
            try:
                response = input()
            except KeyboardInterrupt:
                if typer is not None:
                    raise typer.Exit(code=0)
                return
            if response.strip().lower() == "q":
                if typer is not None:
                    raise typer.Exit(code=0)
                return
            if response.strip():
                continue
            return
        else:
            try:
                key_name = read_one_key_normalized()
                if key_name == "ENTER":
                    return
                if key_name == "ESC":
                    continue
                if key_name == "Q":
                    if typer is not None:
                        raise typer.Exit(code=0)
                    return
                # Any other key: accept and proceed
                return
            except KeyboardInterrupt:
                if typer is not None:
                    raise typer.Exit(code=0)
                return
