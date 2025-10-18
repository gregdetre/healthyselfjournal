from __future__ import annotations

"""Executable entry point for the packaged desktop application."""

from pathlib import Path

from platformdirs import user_data_dir

from ..config import CONFIG
from .settings import load_settings
from ..web.app import WebAppConfig
from .app import DesktopConfig, run_desktop_app


def _default_sessions_dir() -> Path:
    base = Path(user_data_dir("HealthySelfJournal", "Experim"))
    return base / "sessions"


def main() -> None:
    # Load desktop settings for precedence: Desktop settings > project .env.local defaults
    ds, _ = load_settings()

    configured_dir = Path(CONFIG.recordings_dir).expanduser()
    default_dir = Path.cwd() / "sessions"
    try:
        if configured_dir.resolve() == default_dir.resolve():
            sessions_dir = _default_sessions_dir()
        else:
            sessions_dir = configured_dir
    except Exception:
        sessions_dir = configured_dir
    sessions_dir = sessions_dir.expanduser()

    # Apply desktop settings for resume and voice if provided
    resume_on_launch = (
        True if ds.resume_on_launch is None else bool(ds.resume_on_launch)
    )
    voice_enabled = (
        bool(CONFIG.speak_llm) if ds.voice_enabled is None else bool(ds.voice_enabled)
    )

    web_cfg = WebAppConfig(
        sessions_dir=sessions_dir if ds.sessions_dir is None else ds.sessions_dir,
        resume=resume_on_launch,
        host="127.0.0.1",
        port=0,
        reload=False,
        voice_enabled=voice_enabled,
        tts_model=CONFIG.tts_model,
        tts_voice=CONFIG.tts_voice,
        tts_format=CONFIG.tts_format,
        desktop_setup=True,
    )
    desktop_cfg = DesktopConfig(
        web=web_cfg,
        window_title="Healthyself Journal",
        window_width=1280,
        window_height=860,
        fullscreen=False,
        debug=False,
        open_devtools=False,
        confirm_close=True,
    )
    run_desktop_app(desktop_cfg)


if __name__ == "__main__":
    main()
