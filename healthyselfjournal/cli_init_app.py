from __future__ import annotations

"""Init sub-commands: interactive setup wizard and local LLM bootstrap."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .cli_init import run_init_wizard
from .config import CONFIG
from .model_manager import get_model_manager


console = Console()


def build_app() -> typer.Typer:
    app = typer.Typer(
        add_completion=False,
        no_args_is_help=False,
        context_settings={"help_option_names": ["-h", "--help"]},
        help="Setup commands (wizard, local LLM)",
    )

    @app.callback()
    def _default(ctx: typer.Context) -> None:
        # Run the full setup wizard only when invoked as `healthyselfjournal init`
        # without any subcommand (e.g., not when running `init local-llm`).
        if getattr(ctx, "invoked_subcommand", None):
            return
        run_init_wizard()

    @app.command("local-llm")
    def init_local_llm(
        model: str = typer.Option(
            CONFIG.llm_local_model,
            "--model",
            help="Model filename to place under the managed llama models dir.",
        ),
        url: Optional[str] = typer.Option(
            CONFIG.llm_local_model_url,
            "--url",
            help="Download URL for the gguf file. If omitted, uses user_config.toml if set.",
        ),
        sha256: Optional[str] = typer.Option(
            CONFIG.llm_local_model_sha256,
            "--sha256",
            help="Optional SHA-256 checksum for verification.",
        ),
        force: bool = typer.Option(
            False,
            "--force/--no-force",
            help="Re-download even if the file already exists.",
        ),
    ) -> None:
        """Download and register the local LLM gguf model file.

        The model is stored under the platform-managed directory
        (e.g., ~/Library/Application Support/HealthySelfJournal/models/llama/ on macOS).
        """

        manager = get_model_manager()
        target = manager.llama_model_path(model)

        if target.exists() and not force:
            console.print(
                Panel.fit(
                    Text(
                        f"Model already present:\n{target}",
                        style="green",
                    ),
                    title="Local LLM",
                    border_style="green",
                )
            )
            return

        if not url:
            console.print(
                Panel.fit(
                    Text(
                        "No download URL provided. Set [llm].local_model_url in user_config.toml, "
                        "or supply --url here. Optionally add --sha256 for integrity verification.",
                        style="yellow",
                    ),
                    title="Missing URL",
                    border_style="yellow",
                )
            )
            raise typer.Exit(code=2)

        # Ensure parent dirs exist and perform download/verification
        try:
            path = manager.ensure_llama_model(model, url=url, sha256=sha256)
        except Exception as exc:
            console.print(
                Panel.fit(
                    Text(f"Download failed: {exc}", style="red"),
                    title="Local LLM",
                    border_style="red",
                )
            )
            raise typer.Exit(code=2)

        console.print(
            Panel.fit(
                Text(
                    f"Downloaded and registered:\n{path}",
                    style="green",
                ),
                title="Local LLM",
                border_style="green",
            )
        )

    return app
