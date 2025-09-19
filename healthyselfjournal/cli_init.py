from __future__ import annotations

import os
from pathlib import Path
import platform
import subprocess
import webbrowser

import typer
from rich.console import Console
from rich.panel import Panel

from .transcription import (
    resolve_backend_selection,
)
from .mic_check import run_interactive_mic_check


console = Console()


def needs_init(stt_backend_value: str | None) -> bool:
    """Return True if essential config appears missing for a good first run.

    - Anthropic key is required for LLM questions.
    - If cloud STT is configured, require OpenAI key as well.
    """

    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not has_anthropic:
        return True

    backend = (stt_backend_value or "cloud-openai").strip().lower()
    if backend == "cloud-openai" and not os.environ.get("OPENAI_API_KEY"):
        return True

    return False


def init() -> None:
    """Interactive setup wizard to configure keys, backend, and storage."""

    run_init_wizard()


def run_init_wizard() -> None:
    """Run the interactive Questionary-based setup flow.

    Writes results to .env.local in the current working directory and updates
    the current process environment so the session can continue immediately.
    """

    try:
        import questionary
        from questionary import Choice
    except Exception as exc:  # pragma: no cover - defensive
        console.print(
            f"[red]Questionary is required for setup but failed to import:[/] {exc}"
        )
        raise typer.Exit(code=2)

    console.print(
        Panel.fit(
            "Welcome! This wizard will configure your API keys and recording options.\n\n"
            "- Cloud mode (recommended): best accuracy/latency; requires OpenAI + Anthropic keys.\n"
            "- Privacy mode: on-device STT if available (experimental).",
            title="Healthy Self Journal – Setup",
            border_style="magenta",
        )
    )

    mode = questionary.select(
        "Choose setup mode:",
        choices=[
            Choice(title="Cloud (recommended)", value="cloud"),
            Choice(title="Privacy mode (experimental, on-device)", value="private"),
        ],
        default="cloud",
    ).ask()
    if mode is None:
        raise typer.Abort()

    anthropic_key = _prompt_anthropic_key()

    openai_key: str | None = None
    stt_backend_choice = "cloud-openai" if mode == "cloud" else "auto-private"

    if mode == "cloud":
        openai_key = _prompt_openai_key()

    default_sessions = str((Path.cwd() / "sessions").resolve())
    sessions_dir_str = questionary.path(
        "Where should sessions be saved?",
        only_directories=True,
        default=default_sessions,
    ).ask()
    if not sessions_dir_str:
        raise typer.Abort()

    sessions_dir = Path(sessions_dir_str).expanduser().resolve()
    try:
        sessions_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        console.print(f"[red]Failed to create sessions directory:[/] {exc}")
        raise typer.Abort()

    run_test = questionary.confirm(
        "Run a quick microphone/transcription test now?", default=True
    ).ask()
    if run_test is None:
        raise typer.Abort()

    updates: dict[str, str] = {
        "ANTHROPIC_API_KEY": anthropic_key,
        "STT_BACKEND": stt_backend_choice,
        "STT_MODEL": os.environ.get("STT_MODEL", "default"),
        "STT_COMPUTE": os.environ.get("STT_COMPUTE", "auto"),
        "STT_FORMATTING": os.environ.get("STT_FORMATTING", "sentences"),
        "SPEAK_LLM": os.environ.get("SPEAK_LLM", "0"),
        "SESSIONS_DIR": str(sessions_dir),
    }
    if openai_key:
        updates["OPENAI_API_KEY"] = openai_key

    _update_env_local(Path.cwd() / ".env.local", updates)
    for k, v in updates.items():
        os.environ[k] = v

    console.print(
        Panel.fit(
            f"Configuration saved to .env.local in {Path.cwd()}\nSessions directory: {sessions_dir}",
            title="Setup Complete",
            border_style="green",
        )
    )

    if bool(run_test):
        try:
            # Resolve selection from wizard choices and run interactive mic check
            selection = resolve_backend_selection(
                stt_backend_choice,
                os.environ.get("STT_MODEL", "default"),
                os.environ.get("STT_COMPUTE", "auto"),
            )
            run_interactive_mic_check(
                selection,
                console=console,
                language="en",
                stt_formatting=os.environ.get("STT_FORMATTING", "sentences"),
                seconds=3.0,
            )
        except Exception as exc:  # pragma: no cover - runtime surface
            console.print(
                f"[yellow]Mic check encountered an issue; you can continue:[/] {exc}"
            )

    # Friendly next-step hint
    console.print("[bold]Next:[/] run [cyan]healthyselfjournal journal[/]")


def _smoke_test_setup(sessions_dir: Path, stt_backend_value: str) -> None:
    """Deprecated: replaced by interactive mic check in the wizard."""
    console.print(
        "[yellow]Note:[/] The setup wizard now runs an interactive mic check instead of the old smoke test."
    )


def _update_env_local(env_path: Path, updates: dict[str, str]) -> None:
    """Merge key=value updates into the given .env.local file atomically."""

    try:
        from dotenv import dotenv_values
    except Exception:
        existing: dict[str, str] = {}
    else:
        raw = dotenv_values(str(env_path)) if env_path.exists() else {}
        existing = {k: v for k, v in raw.items() if v is not None}

    merged: dict[str, str] = {**existing, **updates}

    def _fmt(v: str) -> str:
        needs_quotes = any(ch in v for ch in [" ", "#", '"'])
        if needs_quotes:
            escaped = v.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        return v

    lines = [f"{k}={_fmt(v)}\n" for k, v in sorted(merged.items())]
    tmp = env_path.with_suffix(env_path.suffix + ".partial")
    tmp.write_text("".join(lines), encoding="utf-8")
    tmp.replace(env_path)


def _prompt_anthropic_key() -> str:
    """Open Anthropic keys page, attempt clipboard capture, fallback to masked input, validate."""
    try:
        import questionary
    except Exception:
        # Fallback to plain input if Questionary failed (should not happen here)
        questionary = None  # type: ignore

    url = "https://console.anthropic.com/settings/api-keys"
    try:
        webbrowser.open(url)
    except Exception:
        pass
    console.print(
        Panel.fit(
            "We opened the Anthropic Console in your browser. Create a key, copy it, then return here.",
            title="Anthropic API Key",
            border_style="cyan",
        )
    )

    while True:
        if questionary is not None:
            # Prompt just to pause until the user copies the key
            _ = questionary.text("Press Enter after copying your key").ask()
        # Best-effort clipboard read
        key = _try_read_clipboard()
        if not key or not _looks_like_anthropic_key(key):
            if questionary is not None:
                key = questionary.password("Paste your Anthropic API key").ask()
            else:
                key = input("Anthropic API key: ")

        if not key:
            console.print("[red]Anthropic key is required to continue.[/]")
            raise typer.Abort()

        if _validate_anthropic_key(key):
            return key

        console.print("[red]Could not validate Anthropic key.[/]")
        if questionary is not None:
            retry = questionary.confirm(
                "Try entering the key again?", default=True
            ).ask()
        else:
            retry = True
        if not retry:
            raise typer.Abort()


def _prompt_openai_key() -> str:
    """Open OpenAI keys page, attempt clipboard capture, fallback to masked input, validate."""
    try:
        import questionary
    except Exception:
        questionary = None  # type: ignore

    url = "https://platform.openai.com/api-keys"
    try:
        webbrowser.open(url)
    except Exception:
        pass
    console.print(
        Panel.fit(
            "We opened the OpenAI API Keys page in your browser. Create a key, copy it, then return here.",
            title="OpenAI API Key",
            border_style="cyan",
        )
    )

    while True:
        if questionary is not None:
            _ = questionary.text("Press Enter after copying your key").ask()
        key = _try_read_clipboard()
        if not key or not _looks_like_openai_key(key):
            if questionary is not None:
                key = questionary.password("Paste your OpenAI API key").ask()
            else:
                key = input("OpenAI API key: ")

        if not key:
            console.print("[red]OpenAI key is required for Cloud mode.[/]")
            raise typer.Abort()

        if _validate_openai_key(key):
            return key

        console.print("[red]Could not validate OpenAI key.[/]")
        if questionary is not None:
            retry = questionary.confirm(
                "Try entering the key again?", default=True
            ).ask()
        else:
            retry = True
        if not retry:
            raise typer.Abort()


def _try_read_clipboard() -> str | None:
    """Return clipboard text if available; best-effort and cross-platform."""
    # Try pyperclip if present
    try:
        import pyperclip  # type: ignore
    except Exception:
        pyperclip = None  # type: ignore
    if pyperclip is not None:
        try:
            text = pyperclip.paste()  # type: ignore[attr-defined]
            if isinstance(text, str) and text.strip():
                return text.strip()
        except Exception:
            pass

    system = platform.system()
    try:
        if system == "Darwin":
            out = subprocess.check_output(["pbpaste"], text=True).strip()
            return out or None
        if system == "Windows":
            out = subprocess.check_output(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "[Console]::OutputEncoding=[Text.UTF8Encoding]::UTF8; Get-Clipboard",
                ],
                text=True,
            ).strip()
            return out or None
        # Linux / other: try common clipboards
        for cmd in (
            ["xclip", "-selection", "clipboard", "-o"],
            ["xsel", "-b", "-o"],
            ["wl-paste", "-n"],
        ):
            try:
                out = subprocess.check_output(cmd, text=True).strip()
                if out:
                    return out
            except Exception:
                continue
    except Exception:
        pass
    return None


def _looks_like_anthropic_key(value: str) -> bool:
    v = value.strip()
    return v.startswith("sk-ant-") or v.startswith("sk-")


def _looks_like_openai_key(value: str) -> bool:
    v = value.strip()
    return v.startswith("sk-")


def _validate_anthropic_key(key: str) -> bool:
    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=key)
        # Cheap, non-billable endpoint
        _ = list(client.models.list())
        return True
    except Exception:
        return False


def _validate_openai_key(key: str) -> bool:
    try:
        from openai import OpenAI

        client = OpenAI(api_key=key)
        _ = list(client.models.list())
        return True
    except Exception:
        return False
