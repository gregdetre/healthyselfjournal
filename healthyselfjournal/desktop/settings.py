from __future__ import annotations

"""Desktop settings persistence using the user's XDG config directory.

This module centralises where we read/write end-user desktop preferences that
should persist across packaged upgrades. We intentionally keep this small and
focused on the handful of knobs exposed in the desktop UI.

Storage location:
- macOS/Linux: ~/.config/healthyselfjournal/settings.toml (via platformdirs)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from platformdirs import user_config_dir


try:  # Python 3.11+
    import tomllib as _toml_r
except Exception:  # pragma: no cover - fallback
    _toml_r = None  # type: ignore

try:
    import tomli_w as _toml_w  # type: ignore
except Exception:  # pragma: no cover - fallback simple writer
    _toml_w = None  # type: ignore


def _xdg_config_dir() -> Path:
    base = Path(user_config_dir("healthyselfjournal", "experim"))
    return base


def settings_path() -> Path:
    """Return the canonical TOML path for desktop settings."""

    return _xdg_config_dir() / "settings.toml"


@dataclass(slots=True)
class DesktopSettings:
    """Persisted desktop settings.

    Fields are optional to allow distinguishing between "unset" and
    explicit True/False or string values. This is useful for precedence logic
    where CLI flags or environment variables may override these values.
    """

    sessions_dir: Path | None = None
    resume_on_launch: bool | None = None
    voice_enabled: bool | None = None
    # cloud or local
    mode: str | None = None


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "on"}:
            return True
        if v in {"0", "false", "no", "off"}:
            return False
    return None


def _to_toml_dict(settings: DesktopSettings) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if settings.sessions_dir is not None:
        data["sessions_dir"] = str(settings.sessions_dir)
    if settings.resume_on_launch is not None:
        data["resume_on_launch"] = bool(settings.resume_on_launch)
    if settings.voice_enabled is not None:
        data["voice_enabled"] = bool(settings.voice_enabled)
    if settings.mode is not None:
        data["mode"] = str(settings.mode)
    return data


def _from_toml_dict(data: Dict[str, Any]) -> DesktopSettings:
    sessions_dir_val = data.get("sessions_dir")
    sessions_dir: Path | None = None
    if isinstance(sessions_dir_val, str) and sessions_dir_val.strip():
        sessions_dir = Path(sessions_dir_val).expanduser()

    resume = _parse_bool(data.get("resume_on_launch"))
    voice = _parse_bool(data.get("voice_enabled"))
    mode_val = data.get("mode")
    mode: str | None = (
        str(mode_val).strip().lower() if isinstance(mode_val, str) else None
    )

    return DesktopSettings(
        sessions_dir=sessions_dir,
        resume_on_launch=resume,
        voice_enabled=voice,
        mode=mode,
    )


def load_settings() -> Tuple[DesktopSettings, Path | None]:
    """Load settings from TOML file, returning (settings, path_used).

    If the file does not exist or cannot be parsed, returns default settings
    with all fields None and a None path.
    """

    path = settings_path()
    if not path.exists():
        return DesktopSettings(), None
    if _toml_r is None:  # pragma: no cover - missing tomllib
        return DesktopSettings(), None
    try:
        text = path.read_text(encoding="utf-8")
        data = _toml_r.loads(text)  # type: ignore[attr-defined]
        if not isinstance(data, dict):
            return DesktopSettings(), None
        return _from_toml_dict(data), path
    except Exception:
        return DesktopSettings(), None


def save_settings(settings: DesktopSettings) -> Path:
    """Write settings TOML to XDG config path and return the path.

    Creates parent directory if needed. Uses tomli_w if available, otherwise
    writes a minimal TOML by hand.
    """

    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _to_toml_dict(settings)
    try:
        if _toml_w is not None:  # type: ignore
            text = _toml_w.dumps(data)  # type: ignore[attr-defined]
        else:  # pragma: no cover - simple fallback
            # Minimal TOML writer for our flat dict
            lines = []
            for k, v in data.items():
                if isinstance(v, bool):
                    lines.append(f"{k} = {str(v).lower()}")
                else:
                    lines.append(
                        f"{k} = \"{str(v).replace('\\', '\\\\').replace('\"', '\\\"')}\""
                    )
            text = "\n".join(lines) + "\n"
        path.write_text(text, encoding="utf-8")
        return path
    except Exception as exc:  # pragma: no cover - IO failure path
        # Best-effort: attempt to write a minimal file per-key to still persist
        try:
            path.write_text("", encoding="utf-8")
        except Exception:
            pass
        raise exc
