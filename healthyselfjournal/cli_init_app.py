from __future__ import annotations

"""Init sub-commands: interactive setup wizard and local LLM bootstrap."""

from pathlib import Path
from typing import Optional, Tuple
import os
import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import httpx

from .cli_init import run_init_wizard
from .config import CONFIG
from .model_manager import get_model_manager


console = Console()


def build_app() -> typer.Typer:
    app = typer.Typer(
        add_completion=False,
        no_args_is_help=False,
        invoke_without_command=True,
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

    @app.command("wizard")
    def wizard_command() -> None:
        """Run the interactive setup wizard."""
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
        hf_repo_id: Optional[str] = typer.Option(
            None,
            "--hf-repo",
            help="Hugging Face repo id to resolve from (e.g., TheBloke/Model-GGUF).",
        ),
        hf_filename: Optional[str] = typer.Option(
            None,
            "--hf-file",
            help="Filename within the HF repo (e.g., llama-3.1-8b-instruct-q4_k_m.gguf).",
        ),
        hf_revision: str = typer.Option(
            "main",
            "--hf-revision",
            help="Revision/tag/commit to resolve (default: main).",
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

        def _hf_resolve(
            repo_id: str, filename: str, revision: str = "main"
        ) -> Tuple[str, str]:
            api_url = f"https://huggingface.co/api/models/{repo_id}/revision/{revision}"
            # Allow private/gated repos when a token is provided
            headers = {}
            token = (
                os.environ.get("HUGGING_FACE_HUB_TOKEN")
                or os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGING_FACE_TOKEN")
            )
            if token:
                headers["Authorization"] = f"Bearer {token}"
            try:
                with httpx.Client(timeout=30.0, headers=headers) as client:
                    resp = client.get(api_url, follow_redirects=True)
                    resp.raise_for_status()
                    data = resp.json()
            except Exception as exc:  # pragma: no cover - network/HTTP errors
                raise RuntimeError(f"Hugging Face metadata fetch failed: {exc}")

            siblings = data.get("siblings") or []
            found_sha: Optional[str] = None
            for s in siblings:
                if s.get("rfilename") == filename:
                    lfs = s.get("lfs") or {}
                    oid = lfs.get("oid") or ""
                    if isinstance(oid, str) and oid.startswith("sha256:"):
                        found_sha = oid.split(":", 1)[1]
                        break
            if not found_sha:
                raise RuntimeError(
                    "File not found in repo or missing SHA-256 metadata."
                )
            download_url = (
                f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
            )
            return found_sha, download_url

        # Non-interactive HF resolve path when flags are provided
        if not url and hf_repo_id and hf_filename:
            try:
                resolved_sha, resolved_url = _hf_resolve(
                    hf_repo_id, hf_filename, hf_revision
                )
                url = resolved_url
                sha256 = sha256 or resolved_sha
                console.print(
                    Panel.fit(
                        Text(
                            f"Resolved from Hugging Face:\nrepo: {hf_repo_id}\nfile: {hf_filename}\nrev: {hf_revision}\nsha256: {resolved_sha}",
                            style="cyan",
                        ),
                        title="Hugging Face",
                        border_style="cyan",
                    )
                )
            except Exception as exc:
                console.print(
                    Panel.fit(
                        Text(f"Hugging Face resolve failed: {exc}", style="red"),
                        title="Local LLM",
                        border_style="red",
                    )
                )
                raise typer.Exit(code=2)

        if not url:
            # Friendly guidance panel with actionable next steps
            msg_lines = [
                "No download URL was provided.",
                "",
                "What would you like to do?",
                f"- A) Paste a direct .gguf URL now to download into:\n  {target}",
                "- B) Place your .gguf file at the path above and re-run this command.",
                "- C) Add settings to user_config.toml so you can omit flags next time.",
                "- D) Resolve from Hugging Face by repo/file (fetches SHA-256 automatically).",
                "- E) Run: healthyselfjournal diagnose local llm (for checks and guidance)",
                "",
                "Tip: You can also pass --url here, and optionally --sha256 for integrity.",
                "",
                "Example URL (replace with your preferred model):",
                "https://huggingface.co/.../llama-3.1-8b-instruct-q4_k_m.gguf",
            ]
            console.print(
                Panel.fit(
                    Text("\n".join(msg_lines), style="yellow"),
                    title="Missing URL",
                    border_style="yellow",
                )
            )

            # Offer interactive prompt when running in a TTY
            if sys.stdin.isatty():
                pasted_url = typer.prompt(
                    "Paste .gguf download URL (press ENTER to use Hugging Face resolver)",
                    default="",
                    show_default=False,
                ).strip()

                pasted_sha: Optional[str] = None
                if pasted_url:
                    url = pasted_url
                    pasted_sha = (
                        typer.prompt(
                            "Optional expected SHA-256 (press ENTER to skip)",
                            default="",
                            show_default=False,
                        ).strip()
                        or None
                    )
                else:
                    # Resolve from Hugging Face interactively
                    repo_id = typer.prompt(
                        "Hugging Face repo id (e.g., TheBloke/Model-GGUF)",
                        default="",
                        show_default=False,
                    ).strip()
                    if not repo_id:
                        raise typer.Exit(code=2)
                    default_file = model if model.endswith(".gguf") else ""
                    filename = typer.prompt(
                        "Filename within the repo",
                        default=default_file,
                        show_default=bool(default_file),
                    ).strip()
                    if not filename:
                        raise typer.Exit(code=2)
                    revision = (
                        typer.prompt(
                            "Revision (tag/branch/commit)",
                            default="main",
                            show_default=True,
                        ).strip()
                        or "main"
                    )
                    try:
                        resolved_sha, resolved_url = _hf_resolve(
                            repo_id, filename, revision
                        )
                    except Exception as exc:
                        console.print(
                            Panel.fit(
                                Text(
                                    f"Hugging Face resolve failed: {exc}", style="red"
                                ),
                                title="Local LLM",
                                border_style="red",
                            )
                        )
                        raise typer.Exit(code=2)
                    url = resolved_url
                    pasted_sha = resolved_sha
                    console.print(
                        Panel.fit(
                            Text(
                                f"Resolved from Hugging Face:\nrepo: {repo_id}\nfile: {filename}\nrev: {revision}\nsha256: {resolved_sha}",
                                style="cyan",
                            ),
                            title="Hugging Face",
                            border_style="cyan",
                        )
                    )

                # Offer to persist settings for reuse
                try:
                    persist = typer.confirm(
                        "Save these settings to user_config.toml for next time?",
                        default=False,
                    )
                except Exception:
                    persist = False

                if persist:
                    # Choose a user_config.toml path: honor HSJ_USER_CONFIG or use XDG
                    target_cfg = os.environ.get("HSJ_USER_CONFIG")
                    if target_cfg:
                        cfg_path = Path(target_cfg).expanduser().resolve()
                    else:
                        xdg = os.environ.get("XDG_CONFIG_HOME") or str(
                            Path.home() / ".config"
                        )
                        cfg_path = Path(xdg) / "healthyselfjournal" / "user_config.toml"

                    cfg_path.parent.mkdir(parents=True, exist_ok=True)

                    snippet_lines = [
                        "[llm]",
                        f'local_model = "{model}"',
                        f'local_model_url = "{url}"',
                    ]
                    if pasted_sha:
                        snippet_lines.append(f'local_model_sha256 = "{pasted_sha}"')
                    snippet = "\n".join(snippet_lines) + "\n"

                    if cfg_path.exists():
                        # Do not overwrite existing config; show a snippet to add
                        console.print(
                            Panel.fit(
                                Text(
                                    "Add the following to your existing user_config.toml:\n\n"
                                    + snippet,
                                    style="cyan",
                                ),
                                title=f"Edit {cfg_path}",
                                border_style="cyan",
                            )
                        )
                    else:
                        try:
                            cfg_path.write_text(snippet, encoding="utf-8")
                            console.print(
                                Panel.fit(
                                    Text(
                                        f"Created user_config.toml at:\n{cfg_path}\n\n"
                                        + "You can edit this file to adjust local LLM settings.",
                                        style="green",
                                    ),
                                    title="Saved Settings",
                                    border_style="green",
                                )
                            )
                        except Exception as exc:
                            console.print(
                                Panel.fit(
                                    Text(
                                        f"Could not write user_config.toml: {exc}",
                                        style="red",
                                    ),
                                    title="Warning",
                                    border_style="red",
                                )
                            )
                # Continue with provided url/sha
                sha256 = pasted_sha
            else:
                # Non-interactive: exit after showing guidance
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

    @app.command("remove-local-llm")
    def remove_local_llm(
        model: str = typer.Argument(
            ..., help="Model filename under the managed llama dir to delete"
        ),
        yes: bool = typer.Option(
            False, "--yes", help="Do not prompt for confirmation."
        ),
    ) -> None:
        """Delete a managed local LLM gguf file and its metadata entry.

        Example:
          healthyselfjournal init remove-local-llm phi-3-mini-4k-instruct-q4_k_m.gguf
        """
        manager = get_model_manager()
        path = manager.llama_model_path(model)
        if not path.exists():
            console.print(
                Panel.fit(
                    Text(f"Not found: {path}", style="yellow"),
                    title="Local LLM",
                    border_style="yellow",
                )
            )
            # Still attempt metadata cleanup
            removed = manager.delete_llama_model(model)
            raise typer.Exit(code=0 if removed else 2)

        if not yes:
            try:
                confirmed = typer.confirm(f"Delete {path}?", default=False)
            except Exception:
                confirmed = False
            if not confirmed:
                raise typer.Exit(code=1)

        removed = manager.delete_llama_model(model)
        if removed:
            console.print(
                Panel.fit(
                    Text(f"Removed: {path}", style="green"),
                    title="Local LLM",
                    border_style="green",
                )
            )
        else:
            console.print(
                Panel.fit(
                    Text("Nothing removed (file/metadata not present)", style="yellow"),
                    title="Local LLM",
                    border_style="yellow",
                )
            )

    return app
