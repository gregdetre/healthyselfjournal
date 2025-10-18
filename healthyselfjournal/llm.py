from __future__ import annotations

"""LLM prompt orchestration for the journaling dialogue."""

from dataclasses import dataclass
import logging
import random
import time
import platform
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence, Callable, Any

from anthropic import (
    APIStatusError,
    RateLimitError,
    Anthropic,
    AnthropicError,
    NotFoundError,
)
import httpx
from gjdutils.strings import jinja_render

_LOGGER = logging.getLogger(__name__)
from .events import log_event
from .config import CONFIG

SYSTEM_PROMPT = "You are a thoughtful journaling companion."


@dataclass
class QuestionRequest:
    """Inputs required to generate a follow-up question."""

    model: str
    current_transcript: str
    recent_summaries: Sequence[str]
    opening_question: str
    question_bank: Sequence[str]
    language: str
    conversation_duration: str
    max_tokens: int = 256
    llm_questions_debug: bool = False


@dataclass
class QuestionResponse:
    """Structured LLM response."""

    question: str
    model: str


@dataclass
class SummaryRequest:
    """Inputs for regenerating session summaries."""

    transcript_markdown: str
    recent_summaries: Sequence[str]
    model: str
    max_tokens: int = 1200


@dataclass
class SummaryResponse:
    summary_markdown: str
    model: str


@dataclass
class InsightsRequest:
    """Inputs for generating a single reflective insight.

    historical_summaries: Oldest→newest list of summaries that provide broad context.
    recent_transcripts: Full transcript bodies for the recent window since last insights.
    prior_insights_excerpt: Optional excerpt from the latest prior insights to reduce repetition.
    range_text: Human-readable description of the covered range (e.g., dates and counts).
    guidelines: Minimal guardrails and style guidance included in the prompt.
    """

    historical_summaries: Sequence[str]
    recent_transcripts: Sequence[str]
    prior_insights_excerpt: str | None
    range_text: str
    guidelines: str
    model: str
    count: int | None = None
    max_tokens: int = 1200


@dataclass
class InsightsResponse:
    insight_markdown: str
    model: str


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def get_active_llm_model_spec() -> str:
    """Return the currently active LLM model spec based on configuration."""

    mode = (CONFIG.llm_mode or "cloud").strip().lower()
    if mode in {"local", "local_llama"}:
        return f"local:{CONFIG.llm_local_model}"
    return CONFIG.model_llm


def generate_followup_question(request: QuestionRequest) -> QuestionResponse:
    provider, model_name, thinking_enabled = _split_model_spec(request.model)
    if thinking_enabled and provider != "anthropic":
        raise ValueError(f"Thinking mode is not supported for provider '{provider}'")
    rendered = _render_question_prompt(request)

    if CONFIG.llm_cloud_off and provider not in {"local", "ollama"}:
        raise RuntimeError(
            "Cloud LLM access is disabled (cloud_off); switch to local_llama mode."
        )

    if provider == "local":
        text = _call_local_llama(
            rendered,
            max_tokens=request.max_tokens,
            temperature=CONFIG.llm_temperature_question,
            top_p=CONFIG.llm_top_p,
            top_k=CONFIG.llm_top_k,
        )
    elif provider == "anthropic":
        text = _call_anthropic(
            model_name,
            rendered,
            max_tokens=request.max_tokens,
            temperature=CONFIG.llm_temperature_question,
            top_p=CONFIG.llm_top_p,
            top_k=CONFIG.llm_top_k,
            thinking_enabled=thinking_enabled,
            cache_question_prompt=True,
        )
    elif provider == "ollama":
        text = _call_ollama(
            model_name,
            rendered,
            max_tokens=request.max_tokens,
            temperature=CONFIG.llm_temperature_question,
            top_p=CONFIG.llm_top_p,
            top_k=CONFIG.llm_top_k,
        )
    else:
        raise ValueError(f"Unsupported provider '{provider}' for question generation")

    question = text.strip()
    if not question.endswith("?"):
        question = question.rstrip(".") + "?"

    response = QuestionResponse(question=question, model=request.model)
    log_event(
        "llm.question.success",
        {
            "provider": provider,
            "model": model_name,
            "max_tokens": request.max_tokens,
        },
    )
    return response


def stream_followup_question(
    request: QuestionRequest, on_delta: Callable[[str], None]
) -> QuestionResponse:
    """Stream a follow-up question from Anthropic and return the final response.

    Calls on_delta with incremental text chunks as they arrive. Falls back to the
    non-streaming call on errors and emits the full text via on_delta once.
    """
    provider, model_name, thinking_enabled = _split_model_spec(request.model)
    if thinking_enabled and provider != "anthropic":
        raise ValueError(f"Thinking mode is not supported for provider '{provider}'")

    rendered = _render_question_prompt(request)

    global _ANTHROPIC_CLIENT
    if _ANTHROPIC_CLIENT is None:
        _ANTHROPIC_CLIENT = Anthropic()

    if CONFIG.llm_cloud_off and provider not in {"local", "ollama"}:
        raise RuntimeError(
            "Cloud LLM access is disabled (cloud_off); switch to local_llama mode."
        )

    if provider == "local":
        text = _call_local_llama(
            rendered,
            max_tokens=request.max_tokens,
            temperature=CONFIG.llm_temperature_question,
            top_p=CONFIG.llm_top_p,
            top_k=CONFIG.llm_top_k,
        )
        try:
            on_delta(text)
        except Exception:
            pass
        question = text.strip()
        if not question.endswith("?"):
            question = question.rstrip(".") + "?"
        log_event(
            "llm.question.streaming.success",
            {
                "provider": provider,
                "model": model_name,
                "max_tokens": request.max_tokens,
            },
        )
        return QuestionResponse(question=question, model=request.model)

    if provider == "anthropic":
        # Stream; fail fast with helpful error messages on failure
        try:
            log_event(
                "llm.question.streaming.started",
                {
                    "provider": provider,
                    "model": model_name,
                    "max_tokens": request.max_tokens,
                },
            )
            stream_kwargs = _build_anthropic_kwargs(
                model=model_name,
                prompt=rendered,
                max_tokens=request.max_tokens,
                temperature=CONFIG.llm_temperature_question,
                top_p=CONFIG.llm_top_p,
                top_k=CONFIG.llm_top_k,
                thinking_enabled=thinking_enabled,
                cache_question_prompt=True,
            )
            with _ANTHROPIC_CLIENT.messages.stream(**stream_kwargs) as stream:
                for text in stream.text_stream:
                    try:
                        on_delta(text)
                    except Exception:  # pragma: no cover - defensive in callback
                        pass
                final_message = stream.get_final_message()
                text = "".join(
                    block.text
                    for block in final_message.content
                    if block.type == "text"
                ).strip()
        except Exception as exc:
            _LOGGER.warning("Streaming failed: %s", exc)
            log_event(
                "llm.question.streaming.failed",
                {
                    "provider": provider,
                    "model": model_name,
                    "error_type": exc.__class__.__name__,
                },
            )
            # Re-raise to surface a user-visible error
            raise
    elif provider == "ollama":
        log_event(
            "llm.question.streaming.started",
            {
                "provider": provider,
                "model": model_name,
                "max_tokens": request.max_tokens,
            },
        )
        try:
            text = _call_ollama(
                model_name,
                rendered,
                max_tokens=request.max_tokens,
                temperature=CONFIG.llm_temperature_question,
                top_p=CONFIG.llm_top_p,
                top_k=CONFIG.llm_top_k,
            )
        except Exception as exc:
            log_event(
                "llm.question.streaming.failed",
                {
                    "provider": provider,
                    "model": model_name,
                    "error_type": exc.__class__.__name__,
                },
            )
            raise
        try:
            on_delta(text)
        except Exception:  # pragma: no cover - defensive callback
            pass
    else:
        raise ValueError(f"Unsupported provider '{provider}' for question generation")

    question = text.strip()
    if not question.endswith("?"):
        question = question.rstrip(".") + "?"

    response = QuestionResponse(question=question, model=request.model)
    log_event(
        "llm.question.streaming.success",
        {
            "provider": provider,
            "model": model_name,
            "max_tokens": request.max_tokens,
        },
    )
    return response


def generate_insight(request: InsightsRequest) -> InsightsResponse:
    provider, model_name, thinking_enabled = _split_model_spec(request.model)
    if thinking_enabled and provider != "anthropic":
        raise ValueError(f"Thinking mode is not supported for provider '{provider}'")

    template = _load_prompt("insights.prompt.md.jinja")
    rendered = jinja_render(
        template,
        {
            "historical_summaries": list(request.historical_summaries),
            "recent_transcripts": list(request.recent_transcripts),
            "prior_insights_excerpt": request.prior_insights_excerpt or "",
            "range_text": request.range_text,
            "guidelines": request.guidelines,
            "insight_count": request.count,
        },
        filesystem_loader=PROMPTS_DIR,
    )

    if CONFIG.llm_cloud_off and provider not in {"local", "ollama"}:
        raise RuntimeError(
            "Cloud LLM access is disabled (cloud_off); switch to local_llama mode."
        )

    if provider == "local":
        text = _call_local_llama(
            rendered,
            max_tokens=request.max_tokens,
            temperature=CONFIG.llm_temperature_summary,
            top_p=CONFIG.llm_top_p,
            top_k=CONFIG.llm_top_k,
        )
    elif provider == "anthropic":
        text = _call_anthropic(
            model_name,
            rendered,
            max_tokens=request.max_tokens,
            temperature=CONFIG.llm_temperature_summary,
            top_p=CONFIG.llm_top_p,
            top_k=CONFIG.llm_top_k,
            thinking_enabled=thinking_enabled,
        )
    elif provider == "ollama":
        text = _call_ollama(
            model_name,
            rendered,
            max_tokens=request.max_tokens,
            temperature=CONFIG.llm_temperature_summary,
            top_p=CONFIG.llm_top_p,
            top_k=CONFIG.llm_top_k,
        )
    else:
        raise ValueError(f"Unsupported provider '{provider}' for insights generation")

    response = InsightsResponse(insight_markdown=text.strip(), model=request.model)
    log_event(
        "llm.insight.success",
        {
            "provider": provider,
            "model": model_name,
            "max_tokens": request.max_tokens,
        },
    )
    return response


def generate_summary(request: SummaryRequest) -> SummaryResponse:
    provider, model_name, thinking_enabled = _split_model_spec(request.model)
    if thinking_enabled and provider != "anthropic":
        raise ValueError(f"Thinking mode is not supported for provider '{provider}'")

    template = _load_prompt("summary.prompt.md.jinja")
    rendered = jinja_render(
        template,
        {
            "transcript_markdown": request.transcript_markdown,
            "recent_summaries": list(request.recent_summaries),
        },
        filesystem_loader=PROMPTS_DIR,
    )

    if CONFIG.llm_cloud_off and provider not in {"local", "ollama"}:
        raise RuntimeError(
            "Cloud LLM access is disabled (cloud_off); switch to local_llama mode."
        )

    if provider == "local":
        text = _call_local_llama(
            rendered,
            max_tokens=request.max_tokens,
            temperature=CONFIG.llm_temperature_summary,
            top_p=CONFIG.llm_top_p,
            top_k=CONFIG.llm_top_k,
        )
    elif provider == "anthropic":
        text = _call_anthropic(
            model_name,
            rendered,
            max_tokens=request.max_tokens,
            temperature=CONFIG.llm_temperature_summary,
            top_p=CONFIG.llm_top_p,
            top_k=CONFIG.llm_top_k,
            thinking_enabled=thinking_enabled,
        )
    elif provider == "ollama":
        text = _call_ollama(
            model_name,
            rendered,
            max_tokens=request.max_tokens,
            temperature=CONFIG.llm_temperature_summary,
            top_p=CONFIG.llm_top_p,
            top_k=CONFIG.llm_top_k,
        )
    else:
        raise ValueError(f"Unsupported provider '{provider}' for summaries")

    response = SummaryResponse(summary_markdown=text.strip(), model=request.model)
    log_event(
        "llm.summary.success",
        {
            "provider": provider,
            "model": model_name,
            "max_tokens": request.max_tokens,
        },
    )
    return response


def get_model_provider(spec: str) -> str:
    """Return the provider segment from a model spec string."""

    provider, _, _ = _split_model_spec(spec)
    return provider


def _render_question_prompt(request: QuestionRequest) -> str:
    template = _load_prompt("question.prompt.md.jinja")
    return jinja_render(
        template,
        {
            "recent_summaries": list(request.recent_summaries),
            "current_transcript": request.current_transcript,
            "language": request.language,
            "conversation_duration": request.conversation_duration,
            "llm_questions_debug": request.llm_questions_debug,
        },
        filesystem_loader=PROMPTS_DIR,
    )


def _call_local_llama(
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
) -> str:
    from .llm_local import (
        LocalLLMNotAvailableError,
        generate_local_completion,
    )

    try:
        return generate_local_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    except LocalLLMNotAvailableError as exc:
        raise RuntimeError(str(exc)) from exc


def _split_model_spec(spec: str) -> tuple[str, str, bool]:
    """Parse provider:model[:version][:thinking] → (provider, provider_model_id, thinking).

    Provider-specific normalization:
    - Anthropic: model id is "<model>-<version>" when version is provided.
      If version omitted, accept existing hyphenated ids, or use a sensible default
      for known aliases (e.g., claude-sonnet-4 → claude-sonnet-4-20250514).
      Thinking maps to the API "thinking" parameter, not the model id.
    """
    if ":" not in spec:
        provider = "anthropic"
        rest = spec
    else:
        provider, rest = spec.split(":", 1)

    provider = provider.strip().lower()
    if provider in {"local_llama", "local"}:
        provider = "local"

    # Split remaining segments: model[:version][:thinking]
    rest_parts = rest.split(":") if rest else []
    if not rest_parts:
        raise ValueError("Invalid model spec: missing model segment")

    base_model = rest_parts[0]

    if provider == "local":
        return provider, base_model, False
    version: str | None = None
    thinking_enabled = False

    if len(rest_parts) >= 2:
        # If the last segment is 'thinking', mark it and pop
        if rest_parts[-1] == "thinking":
            thinking_enabled = True
            rest_parts = rest_parts[:-1]
        # After removing thinking, a second segment is the version
        if len(rest_parts) >= 2:
            version = rest_parts[1]

    # Provider-specific normalization
    if provider == "anthropic":
        model_id = _normalize_anthropic_model(base_model, version)
    else:
        # For unknown providers, pass through the model name (and ignore version)
        model_id = base_model if version is None else f"{base_model}:{version}"

    return provider, model_id, thinking_enabled


def _normalize_anthropic_model(model_name: str, version: str | None) -> str:
    """Return canonical Anthropic model id for messages API.

    Accepts either plain ids like 'claude-3-7-sonnet-20250219' or tuple
    (model_name='claude-sonnet-4', version='20250514') and returns
    'claude-sonnet-4-20250514'. If version is None and model_name already
    includes a hyphenated date suffix, it's returned as-is. For known aliases
    without version, default to the stable version.
    """
    # If already looks like a full model id with date suffix, keep as-is
    if any(token.isdigit() and len(token) == 8 for token in model_name.split("-")):
        return model_name

    if version:
        return f"{model_name}-{version}"

    # No explicit version: map known aliases to a default stable version
    default_versions: dict[str, str] = {
        "claude-sonnet-4": "20250514",
        "sonnet-4": "20250514",
        "claude-sonnet": "20250514",
    }
    if model_name in default_versions:
        return f"{model_name}-{default_versions[model_name]}"
    # Otherwise pass through unchanged
    return model_name


def _strip_thinking_variant(model_name: str) -> str:
    """Remove the "-thinking-" infix from a model string.

    Example: "claude-3-7-sonnet-thinking-20250219" → "claude-3-7-sonnet-20250219".
    """
    if "-thinking-" in model_name:
        return model_name.replace("-thinking-", "-")
    return model_name


@lru_cache(maxsize=4)
def _load_prompt(filename: str) -> str:
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


_ANTHROPIC_CLIENT: Anthropic | None = None


def _format_ollama_connect_help(*, base_url: str, model: str) -> str:
    """Return contextual, platform-specific guidance for an Ollama connection failure.

    Designed to be appended to exception messages for semi-technical users.
    """
    system = platform.system().lower()
    common = [
        f"Could not reach Ollama at {base_url} (for model '{model}').",
        "Try the following:\n1) Start the Ollama server\n   - Foreground: ollama serve\n   - Verify: curl -s http://127.0.0.1:11434/api/version && echo",
        "2) If Ollama runs on another host/port, set OLLAMA_BASE_URL, e.g.\n   export OLLAMA_BASE_URL=http://192.168.1.10:11434",
    ]

    if system == "darwin":
        os_specific = [
            "macOS tips:",
            "- Launch as a background service (Homebrew): brew services start ollama",
            "- If using Docker for your app, use host.docker.internal: http://host.docker.internal:11434",
        ]
    elif system == "linux":
        os_specific = [
            "Linux tips:",
            "- If installed as a systemd service: sudo systemctl start ollama && sudo systemctl status ollama",
            "- Firewall: sudo ufw allow 11434/tcp (if UFW is enabled)",
            "- In Docker: use host.docker.internal (on macOS/Windows) or bind Ollama with OLLAMA_HOST=0.0.0.0:11434",
        ]
    elif system == "windows":
        os_specific = [
            "Windows tips:",
            "- Start via 'ollama serve' or the Ollama Desktop app",
            "- Verify with: curl http://127.0.0.1:11434/api/version",
            "- If your app runs in Docker, use http://host.docker.internal:11434",
        ]
    else:
        os_specific = []

    tail = [
        "If the model is missing, pull it first:",
        f"  ollama pull {model}",
        "Then test:",
        f"  ollama run {model} -p 'Hello'",
    ]

    return "\n".join(common + os_specific + tail)


def _call_ollama(
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float | None = None,
    top_k: int | None = None,
    max_retries: int = 3,
    backoff_base_seconds: float = 1.0,
) -> str:
    """Call the local Ollama chat endpoint and return the assistant text."""

    base_url = CONFIG.ollama_base_url.rstrip("/")
    url = f"{base_url}/api/chat"
    options: dict[str, Any] = {
        "temperature": temperature,
        "num_predict": max_tokens,
        "num_ctx": CONFIG.ollama_num_ctx,
    }
    if top_p is not None:
        options["top_p"] = top_p
    if top_k is not None:
        options["top_k"] = top_k
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": options,
    }

    last_error: Exception | None = None
    timeout = CONFIG.ollama_timeout_seconds

    for attempt in range(1, max_retries + 1):
        try:
            response = httpx.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            message = data.get("message") or {}
            content = message.get("content")
            if not content:
                raise ValueError("Ollama response missing assistant content")
            return str(content).strip()
        except httpx.HTTPStatusError as exc:
            # Compatibility fallback: older Ollama versions only support /api/generate
            if exc.response is not None and exc.response.status_code == 404:
                try:
                    _LOGGER.info(
                        "/api/chat not available (404); trying /api/generate compatibility path"
                    )
                    gen_url = f"{base_url}/api/generate"
                    gen_payload = {
                        "model": model,
                        "prompt": prompt,
                        "system": SYSTEM_PROMPT,
                        "stream": False,
                        "options": options,
                    }
                    gen_resp = httpx.post(gen_url, json=gen_payload, timeout=timeout)
                    gen_resp.raise_for_status()
                    gen_data = gen_resp.json()
                    # /api/generate returns { "response": "..." }
                    gen_text = gen_data.get("response")
                    if not gen_text:
                        # Some variants may still nest content under message
                        gen_text = (gen_data.get("message") or {}).get("content")
                    if not gen_text:
                        raise ValueError(
                            "Ollama generate response missing text content"
                        )
                    return str(gen_text).strip()
                except Exception as gen_exc:
                    last_error = gen_exc
                    _LOGGER.warning("Ollama generate fallback failed: %s", gen_exc)
                    log_event(
                        "llm.retry",
                        {
                            "provider": "ollama",
                            "model": model,
                            "attempt": attempt,
                            "error_type": gen_exc.__class__.__name__,
                        },
                    )
            else:
                last_error = exc
                _LOGGER.warning(
                    "Ollama call failed (attempt %s/%s): %s", attempt, max_retries, exc
                )
                log_event(
                    "llm.retry",
                    {
                        "provider": "ollama",
                        "model": model,
                        "attempt": attempt,
                        "error_type": exc.__class__.__name__,
                    },
                )
        except (httpx.RequestError, ValueError) as exc:
            last_error = exc
            _LOGGER.warning(
                "Ollama call failed (attempt %s/%s): %s", attempt, max_retries, exc
            )
            log_event(
                "llm.retry",
                {
                    "provider": "ollama",
                    "model": model,
                    "attempt": attempt,
                    "error_type": exc.__class__.__name__,
                },
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            last_error = exc
            _LOGGER.exception("Unexpected Ollama failure: %s", exc)
            log_event(
                "llm.error",
                {
                    "provider": "ollama",
                    "model": model,
                    "attempt": attempt,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
            )

        if attempt < max_retries:
            sleep_for = backoff_base_seconds * (2 ** (attempt - 1))
            jitter = random.uniform(0, sleep_for * 0.25)
            time.sleep(sleep_for + jitter)

    assert last_error is not None
    log_event(
        "llm.failed",
        {
            "provider": "ollama",
            "model": model,
            "attempts": max_retries,
            "error_type": last_error.__class__.__name__,
            "error": str(last_error),
        },
    )
    try:
        import httpx as _httpx
    except Exception:  # pragma: no cover - defensive import
        _httpx = None  # type: ignore

    # When connection is refused, append actionable guidance
    if _httpx is not None and isinstance(last_error, _httpx.RequestError):
        message = str(last_error)
        lowered = message.lower()
        if ("connection refused" in lowered) or (
            getattr(last_error, "__class__", None)
            and "connecterror" in last_error.__class__.__name__.lower()
        ):
            help_text = _format_ollama_connect_help(base_url=base_url, model=model)
            raise RuntimeError(f"{message}\n\n{help_text}") from last_error

    raise last_error


def _call_anthropic(
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float | None = None,
    top_k: int | None = None,
    max_retries: int = 3,
    backoff_base_seconds: float = 1.5,
    thinking_enabled: bool = False,
    cache_question_prompt: bool = False,
) -> str:
    global _ANTHROPIC_CLIENT
    if _ANTHROPIC_CLIENT is None:
        _ANTHROPIC_CLIENT = Anthropic()

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            create_kwargs = _build_anthropic_kwargs(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                thinking_enabled=thinking_enabled,
                cache_question_prompt=cache_question_prompt,
            )
            response = _ANTHROPIC_CLIENT.messages.create(**create_kwargs)
            return "".join(
                block.text for block in response.content if block.type == "text"
            ).strip()
        except NotFoundError as exc:
            last_error = exc
            _LOGGER.warning(
                "Anthropic call failed (attempt %s/%s): %s",
                attempt,
                max_retries,
                exc,
            )
            log_event(
                "llm.retry",
                {
                    "provider": "anthropic",
                    "model": model,
                    "attempt": attempt,
                    "error_type": exc.__class__.__name__,
                },
            )
        except (RateLimitError, APIStatusError, AnthropicError) as exc:
            last_error = exc
            _LOGGER.warning(
                "Anthropic call failed (attempt %s/%s): %s", attempt, max_retries, exc
            )
            log_event(
                "llm.retry",
                {
                    "provider": "anthropic",
                    "model": model,
                    "attempt": attempt,
                    "error_type": exc.__class__.__name__,
                },
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            last_error = exc
            _LOGGER.exception("Unexpected Anthropic failure: %s", exc)
            log_event(
                "llm.error",
                {
                    "provider": "anthropic",
                    "model": model,
                    "attempt": attempt,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
            )

        if attempt < max_retries:
            sleep_for = backoff_base_seconds * (2 ** (attempt - 1))
            jitter = random.uniform(0, sleep_for * 0.3)
            time.sleep(sleep_for + jitter)

    assert last_error is not None
    log_event(
        "llm.failed",
        {
            "provider": "anthropic",
            "model": model,
            "attempts": max_retries,
            "error_type": last_error.__class__.__name__,
            "error": str(last_error),
        },
    )
    raise last_error


def _build_anthropic_kwargs(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    thinking_enabled: bool,
    cache_question_prompt: bool = False,
) -> dict[str, Any]:
    """Construct kwargs for Anthropic messages API with consistent thinking behavior.

    - Applies system prompt
    - Normalizes temperature when thinking is enabled
    - Enforces Anthropic thinking budget constraints (>=1024 and < max_tokens)
    - Passes through top_p/top_k when provided
    """
    effective_temperature = 1.0 if thinking_enabled else temperature
    # Enforce Anthropic minimum budget requirement (>= 1024) when thinking is enabled
    if thinking_enabled:
        reserved_for_output = 1
        max_allowed_by_output = max(max_tokens - reserved_for_output, 0)
        effective_budget_tokens = max(
            1024, min(CONFIG.prompt_budget_tokens, max_allowed_by_output)
        )
    else:
        effective_budget_tokens = None

    # When caching question prompts, convert system and user content into content blocks
    # and mark the stable prefix (before the '## Context' section) as cacheable.
    if cache_question_prompt:
        # Split prompt at the first '## Context' heading, if present
        marker = "## Context"
        idx = prompt.find(marker)
        if idx != -1:
            stable_prefix = prompt[:idx].rstrip()
            dynamic_suffix = prompt[idx:]
        else:
            # If no marker, treat entire prompt as dynamic to be safe
            stable_prefix = ""
            dynamic_suffix = prompt

        system_blocks: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        user_content_blocks: list[dict[str, Any]] = []
        if stable_prefix:
            user_content_blocks.append(
                {
                    "type": "text",
                    "text": stable_prefix,
                    "cache_control": {"type": "ephemeral"},
                }
            )
        if dynamic_suffix:
            user_content_blocks.append({"type": "text", "text": dynamic_suffix})

        create_kwargs: dict[str, Any]
        create_kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": effective_temperature,
            "system": system_blocks,
            "messages": [
                {
                    "role": "user",
                    "content": user_content_blocks,
                }
            ],
            # Required beta header to enable prompt caching
            "extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"},
        }
    else:
        create_kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": effective_temperature,
            "system": SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }
    if thinking_enabled:
        create_kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": effective_budget_tokens,
        }
    if top_p is not None:
        create_kwargs["top_p"] = top_p
    if top_k is not None:
        create_kwargs["top_k"] = top_k

    return create_kwargs
