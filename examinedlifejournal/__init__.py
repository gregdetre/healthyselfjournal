"""Core package for the Examined Life Journal voice journaling app."""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "__version__",
]

__version__ = "0.1.0"


def _autoload_env() -> None:
    """Load environment variables from .env and .env.local if present.

    Precedence: existing OS environment > .env.local > .env
    """
    try:
        from dotenv import dotenv_values, find_dotenv  # type: ignore
    except Exception:
        # python-dotenv not installed; skip silent
        return

    # Candidate locations: project root (package parent) and current working dir
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
