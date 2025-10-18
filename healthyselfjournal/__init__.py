"""Core package for the Healthyself Journal voice journaling app."""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "__version__",
]

# Resolve package version from installed metadata; fall back for editable/dev
try:  # Python stdlib importlib.metadata
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
except Exception:  # pragma: no cover - extremely defensive
    PackageNotFoundError = Exception  # type: ignore[assignment]

    def _pkg_version(_: str) -> str:  # type: ignore[no-redef]
        return "0.0.0+local"


try:
    __version__ = _pkg_version("healthyselfjournal")
except PackageNotFoundError:
    __version__ = "0.0.0+local"


def _autoload_env() -> None:
    """Load environment variables from .env and .env.local if present.

    Precedence: existing OS environment > .env.local > .env
    """
    try:
        from dotenv import dotenv_values, find_dotenv  # type: ignore
    except Exception:
        # python-dotenv not installed; skip silent
        return

    # Candidate locations: project root (package parent), current working dir,
    # and user XDG config directory for desktop installs
    package_root = Path(__file__).resolve().parents[1]

    # Gather paths explicitly to control precedence
    paths_in_order: list[str] = []
    # Project root first (preferred when running package from elsewhere)
    prj_env_local = package_root / ".env.local"
    prj_env = package_root / ".env"
    if prj_env.exists():
        paths_in_order.append(str(prj_env))
    if prj_env_local.exists():
        paths_in_order.append(str(prj_env_local))

    # Also consider CWD for ad-hoc local runs
    cwd_env = find_dotenv(".env", usecwd=True)
    cwd_env_local = find_dotenv(".env.local", usecwd=True)
    if cwd_env:
        paths_in_order.append(cwd_env)
    if cwd_env_local:
        paths_in_order.append(cwd_env_local)

    # Also consider XDG config for desktop users
    try:
        from platformdirs import user_config_dir  # type: ignore
    except Exception:
        user_config_dir = None  # type: ignore

    if user_config_dir is not None:
        xdg_dir = Path(user_config_dir("healthyselfjournal", "experim"))
        xdg_env = xdg_dir / ".env"
        xdg_env_local = xdg_dir / ".env.local"
        if xdg_env.exists():
            paths_in_order.append(str(xdg_env))
        if xdg_env_local.exists():
            paths_in_order.append(str(xdg_env_local))

    # Merge with precedence: later entries override earlier ones, but never override OS env
    merged: dict[str, str] = {}
    for path in paths_in_order:
        for k, v in dotenv_values(path).items():
            if v is not None:
                merged[k] = v

    for key, value in merged.items():
        if key not in os.environ:
            os.environ[key] = value


_autoload_env()
