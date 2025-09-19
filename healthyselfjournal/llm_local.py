from __future__ import annotations

"""Local LLM helpers backed by llama-cpp-python."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from .config import CONFIG
from .model_manager import get_model_manager


class LocalLLMNotAvailableError(RuntimeError):
    """Raised when local LLM support is requested but unavailable."""


def _resolve_model_path() -> Path:
    manager = get_model_manager()
    return manager.ensure_llama_model(
        CONFIG.llm_local_model,
        url=CONFIG.llm_local_model_url,
        sha256=CONFIG.llm_local_model_sha256,
    )


@lru_cache(maxsize=1)
def _load_llama() -> Any:
    try:
        from llama_cpp import Llama
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise LocalLLMNotAvailableError(
            "llama-cpp-python is not installed. Install it to enable local LLM mode."
        ) from exc

    model_path = _resolve_model_path()
    n_threads = CONFIG.llm_local_threads if CONFIG.llm_local_threads > 0 else None
    ctx = max(256, int(CONFIG.llm_local_context))
    gpu_layers = max(0, int(CONFIG.llm_local_gpu_layers))

    return Llama(
        model_path=str(model_path),
        n_ctx=ctx,
        n_gpu_layers=gpu_layers,
        n_threads=n_threads,
        logits_all=False,
        seed=None,
        embedding=False,
    )


def generate_local_completion(
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: Optional[float],
    top_k: Optional[int],
) -> str:
    """Generate text from the local llama model."""

    llama = _load_llama()
    params: dict[str, Any] = {
        "max_tokens": max(1, max_tokens),
        "temperature": max(0.0, temperature),
        "repeat_penalty": 1.05,
    }
    if top_p is not None:
        params["top_p"] = max(0.0, min(float(top_p), 1.0))
    if top_k is not None and top_k > 0:
        params["top_k"] = int(top_k)

    result = llama(prompt, **params)
    choices = result.get("choices") or []
    if not choices:
        return ""
    text = choices[0].get("text") or ""
    return str(text).strip()


def reset_local_cache() -> None:
    """Clear cached llama instance (for testing)."""

    _load_llama.cache_clear()
