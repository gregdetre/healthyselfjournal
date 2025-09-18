from __future__ import annotations

"""LLM prompt orchestration for the journaling dialogue."""

from dataclasses import dataclass
import logging
import random
import time
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
from gjdutils.strings import jinja_render

_LOGGER = logging.getLogger(__name__)
from .events import log_event
from .config import CONFIG


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


@dataclass
class QuestionResponse:
    """Structured Anthropic response."""

    question: str
    model: str


@dataclass
class SummaryRequest:
    """Inputs for regenerating session summaries."""

    transcript_markdown: str
    recent_summaries: Sequence[str]
    model: str
    max_tokens: int = 512


@dataclass
class SummaryResponse:
    summary_markdown: str
    model: str


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def generate_followup_question(request: QuestionRequest) -> QuestionResponse:
    provider, model_name, thinking_enabled = _split_model_spec(request.model)
    if provider != "anthropic":
        raise ValueError(f"Unsupported provider '{provider}' for question generation")

    template = _load_prompt("question.prompt.md.jinja")
    # Shuffle the question bank to vary ordering in the prompt
    shuffled_question_bank = list(request.question_bank)
    random.shuffle(shuffled_question_bank)
    rendered = jinja_render(
        template,
        {
            "recent_summaries": list(request.recent_summaries),
            "current_transcript": request.current_transcript,
            "question_bank": shuffled_question_bank,
            "language": request.language,
            "conversation_duration": request.conversation_duration,
        },
        filesystem_loader=PROMPTS_DIR,
    )

    text = _call_anthropic(
        model_name,
        rendered,
        max_tokens=request.max_tokens,
        temperature=CONFIG.llm_temperature_question,
        top_p=CONFIG.llm_top_p,
        top_k=CONFIG.llm_top_k,
        thinking_enabled=thinking_enabled,
    )

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
    if provider != "anthropic":
        raise ValueError(f"Unsupported provider '{provider}' for question generation")

    template = _load_prompt("question.prompt.md.jinja")
    # Shuffle the question bank to vary ordering in the prompt
    shuffled_question_bank = list(request.question_bank)
    random.shuffle(shuffled_question_bank)
    rendered = jinja_render(
        template,
        {
            "recent_summaries": list(request.recent_summaries),
            "current_transcript": request.current_transcript,
            "question_bank": shuffled_question_bank,
            "language": request.language,
            "conversation_duration": request.conversation_duration,
        },
        filesystem_loader=PROMPTS_DIR,
    )

    global _ANTHROPIC_CLIENT
    if _ANTHROPIC_CLIENT is None:
        _ANTHROPIC_CLIENT = Anthropic()

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
        # Ensure thinking temperature and budget obey API constraints
        effective_temperature = 1.0 if thinking_enabled else 0.7
        # Enforce Anthropic minimum budget requirement (>= 1024) when thinking is enabled
        # and keep it strictly below max_tokens to avoid exhausting the output quota.
        # Also clamp by configured prompt budget.
        if thinking_enabled:
            # Reserve at least 1 token for output; budget must be >= 1024
            reserved_for_output = 1
            max_allowed_by_output = max(request.max_tokens - reserved_for_output, 0)
            effective_budget_tokens = max(
                1024, min(CONFIG.prompt_budget_tokens, max_allowed_by_output)
            )
        else:
            effective_budget_tokens = None

        stream_kwargs: dict[str, Any] = {
            "model": model_name,
            "max_tokens": request.max_tokens,
            "temperature": effective_temperature,
            "system": "You are a thoughtful journaling companion.",
            "messages": [
                {
                    "role": "user",
                    "content": rendered,
                }
            ],
        }
        if thinking_enabled:
            stream_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": effective_budget_tokens,
            }
        with _ANTHROPIC_CLIENT.messages.stream(**stream_kwargs) as stream:
            for text in stream.text_stream:
                try:
                    on_delta(text)
                except Exception:  # pragma: no cover - defensive in callback
                    pass
            final_message = stream.get_final_message()
            text = "".join(
                block.text for block in final_message.content if block.type == "text"
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


def generate_summary(request: SummaryRequest) -> SummaryResponse:
    provider, model_name, thinking_enabled = _split_model_spec(request.model)
    if provider != "anthropic":
        raise ValueError(f"Unsupported provider '{provider}' for summaries")

    template = _load_prompt("summary.prompt.md.jinja")
    rendered = jinja_render(
        template,
        {
            "transcript_markdown": request.transcript_markdown,
            "recent_summaries": list(request.recent_summaries),
        },
        filesystem_loader=PROMPTS_DIR,
    )

    text = _call_anthropic(
        model_name,
        rendered,
        max_tokens=request.max_tokens,
        temperature=CONFIG.llm_temperature_summary,
        top_p=CONFIG.llm_top_p,
        top_k=CONFIG.llm_top_k,
        thinking_enabled=thinking_enabled,
    )

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

    # Split remaining segments: model[:version][:thinking]
    rest_parts = rest.split(":") if rest else []
    if not rest_parts:
        raise ValueError("Invalid model spec: missing model segment")

    base_model = rest_parts[0]
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
) -> str:
    global _ANTHROPIC_CLIENT
    if _ANTHROPIC_CLIENT is None:
        _ANTHROPIC_CLIENT = Anthropic()

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
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
            create_kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": effective_temperature,
                "system": "You are a thoughtful journaling companion.",
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
