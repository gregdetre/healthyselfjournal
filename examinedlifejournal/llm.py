from __future__ import annotations

"""LLM prompt orchestration for the journaling dialogue."""

from dataclasses import dataclass
import logging
import random
import time
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence, Callable

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
        with _ANTHROPIC_CLIENT.messages.stream(
            model=model_name,
            max_tokens=request.max_tokens,
            temperature=0.7,
            system="You are a thoughtful journaling companion.",
            thinking={"type": "enabled"} if thinking_enabled else None,
            messages=[
                {
                    "role": "user",
                    "content": rendered,
                }
            ],
        ) as stream:
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
    parts = spec.split(":", 1)
    if len(parts) == 1:
        provider, model_name = "anthropic", parts[0]
    else:
        provider, model_name = parts[0], parts[1]

    # Normalize Anthropic model aliases and optional ":thinking" suffix
    if provider == "anthropic":
        model_name, thinking_enabled = _normalize_anthropic_model(model_name)
    else:
        thinking_enabled = False
    return provider, model_name, thinking_enabled


def _normalize_anthropic_model(model_name: str) -> tuple[str, bool]:
    """Return canonical Anthropic model id and if thinking is requested.

    Supports alias inputs and an optional trailing ":thinking" tag, which we
    turn into the Anthropic API "thinking" parameter rather than a model id.
    """
    want_thinking = False
    if model_name.endswith(":thinking"):
        want_thinking = True
        model_name = model_name[: -len(":thinking")]

    # Map aliases → canonical IDs (do NOT downgrade Sonnet 4 to 3.7)
    legacy_map: dict[str, str] = {
        "claude-sonnet-4": "claude-sonnet-4-20250514",
        "sonnet-4": "claude-sonnet-4-20250514",
        "claude-sonnet": "claude-sonnet-4-20250514",
    }
    normalized = legacy_map.get(model_name, model_name)
    return normalized, want_thinking


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
            create_kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": "You are a thoughtful journaling companion.",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            }
            if thinking_enabled:
                create_kwargs["thinking"] = {"type": "enabled"}
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
