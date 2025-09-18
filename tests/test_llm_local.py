from __future__ import annotations

import pytest

from healthyselfjournal import llm


def test_get_model_provider() -> None:
    assert llm.get_model_provider("anthropic:claude-sonnet-4") == "anthropic"
    assert llm.get_model_provider("ollama:gemma3:27b-instruct-q4_K_M") == "ollama"


def test_generate_followup_question_with_ollama(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_call(model: str, prompt: str, **_: object) -> str:
        captured["model"] = model
        captured["prompt"] = prompt
        return "Reflect on your day"

    monkeypatch.setattr(llm, "_call_ollama", fake_call)

    request = llm.QuestionRequest(
        model="ollama:gemma3:27b-instruct-q4_K_M",
        current_transcript="I felt grounded after journaling.",
        recent_summaries=[],
        opening_question="What is present for you?",
        question_bank=["What feels most alive right now?"],
        language="en",
        conversation_duration="05:12",
        max_tokens=256,
    )

    response = llm.generate_followup_question(request)

    assert response.model == request.model
    assert response.question.endswith("?")
    assert captured["model"] == "gemma3:27b-instruct-q4_K_M"
    assert "grounded" in captured["prompt"]


def test_stream_followup_question_with_ollama(monkeypatch) -> None:
    monkeypatch.setattr(llm, "_call_ollama", lambda *args, **kwargs: "Take a deep breath")
    seen: list[str] = []

    def on_delta(chunk: str) -> None:
        seen.append(chunk)

    request = llm.QuestionRequest(
        model="ollama:gemma3:12b",
        current_transcript="I need a gentle nudge today.",
        recent_summaries=[],
        opening_question="How do you want to begin?",
        question_bank=["What would feel supportive right now?"],
        language="en",
        conversation_duration="02:45",
        max_tokens=200,
    )

    response = llm.stream_followup_question(request, on_delta)

    assert response.question.endswith("?")
    assert seen == ["Take a deep breath"]


def test_generate_summary_with_ollama(monkeypatch) -> None:
    monkeypatch.setattr(llm, "_call_ollama", lambda *args, **kwargs: "A concise summary")

    request = llm.SummaryRequest(
        transcript_markdown="## Entry\n- Explored gratitude for the day",
        recent_summaries=[],
        model="ollama:gemma3:27b",
        max_tokens=512,
    )

    response = llm.generate_summary(request)

    assert response.summary_markdown == "A concise summary"
    assert response.model == request.model


def test_ollama_thinking_mode_not_supported(monkeypatch) -> None:
    monkeypatch.setattr(llm, "_call_ollama", lambda *args, **kwargs: "Should not run")

    request = llm.QuestionRequest(
        model="ollama:gemma3:27b:thinking",
        current_transcript="Testing unsupported mode",
        recent_summaries=[],
        opening_question="Start",
        question_bank=["Placeholder"],
        language="en",
        conversation_duration="00:10",
        max_tokens=128,
    )

    with pytest.raises(ValueError):
        llm.generate_followup_question(request)

