from __future__ import annotations

from pathlib import Path

import typer

from .config import CONFIG
from .cli_reconcile import reconcile as reconcile_cmd
from .cli_summarise import build_app as build_summaries_app  # type: ignore


def build_app() -> typer.Typer:
    app = typer.Typer(
        add_completion=False,
        no_args_is_help=True,
        context_settings={"help_option_names": ["-h", "--help"]},
        help="Repair utilities: STT backfill and summary regeneration.",
    )

    @app.command("stt")
    def fix_stt(
        sessions_dir: Path = typer.Option(
            CONFIG.recordings_dir,
            "--sessions-dir",
            help="Directory where session markdown/audio files are stored.",
        ),
        stt_backend: str = typer.Option(
            CONFIG.stt_backend,
            "--stt-backend",
            help=(
                "Transcription backend: cloud-openai, local-mlx, local-faster, "
                "local-whispercpp, or auto-private."
            ),
        ),
        stt_model: str = typer.Option(
            CONFIG.model_stt,
            "--stt-model",
            help="Model preset or identifier for the selected backend.",
        ),
        stt_compute: str = typer.Option(
            CONFIG.stt_compute or "auto",
            "--stt-compute",
            help="Optional compute precision override for local backends (e.g., int8_float16).",
        ),
        language: str = typer.Option(
            "en",
            "--language",
            help="Primary language for transcription.",
        ),
        limit: int = typer.Option(
            0,
            "--limit",
            help="Maximum number of recordings to process (0 = no limit).",
        ),
        min_duration: float = typer.Option(
            0.1,
            "--min-duration",
            help="Minimum duration (seconds) to attempt transcription.",
        ),
        too_short_action: str = typer.Option(
            "skip",
            "--too-short",
            help="Action for recordings under thresholds: skip, mark, or delete.",
        ),
    ) -> None:
        """Alias of `reconcile` under the `fix` group."""

        reconcile_cmd(
            sessions_dir=sessions_dir,
            stt_backend=stt_backend,
            stt_model=stt_model,
            stt_compute=stt_compute,
            language=language,
            limit=limit,
            min_duration=min_duration,
            too_short_action=too_short_action,
        )

    # Mount summaries app under fix to avoid relying on Typer internals
    _summaries_app = build_summaries_app()

    @_summaries_app.callback()
    def _summaries_banner() -> None:
        pass

    app.add_typer(_summaries_app, name="summaries")

    return app
