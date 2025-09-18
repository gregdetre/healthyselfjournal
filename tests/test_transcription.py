from __future__ import annotations

import pytest

from healthyselfjournal import transcription
from healthyselfjournal.transcription import (
    AUTO_PRIVATE_BACKEND,
    CLOUD_BACKEND,
    LOCAL_FASTER_BACKEND,
    LOCAL_MLX_BACKEND,
    BackendNotAvailableError,
    BackendSelection,
    OpenAITranscriptionBackend,
    apply_transcript_formatting,
    create_transcription_backend,
    resolve_backend_selection,
)


def test_resolve_backend_cloud_default():
    selection = resolve_backend_selection(CLOUD_BACKEND, "default", "auto")
    assert selection.backend_id == CLOUD_BACKEND
    assert selection.model == "gpt-4o-transcribe"
    assert selection.compute is None
    assert selection.warnings == []


def test_resolve_backend_local_faster_assigns_default_compute():
    selection = resolve_backend_selection(LOCAL_FASTER_BACKEND, "fast", "auto")
    assert selection.backend_id == LOCAL_FASTER_BACKEND
    assert selection.model == "small"
    assert selection.compute == "int8_float16"


def test_resolve_backend_with_unused_compute_emits_warning():
    selection = resolve_backend_selection(CLOUD_BACKEND, "accuracy", "int8")
    assert selection.compute is None
    assert selection.warnings != []


def test_auto_private_prefers_mlx(monkeypatch):
    monkeypatch.setattr(transcription.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(transcription.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(transcription, "_has_mlx_whisper", lambda: True)
    monkeypatch.setattr(transcription, "_has_faster_whisper", lambda: False)
    monkeypatch.setattr(transcription, "_has_whispercpp", lambda: False)

    selection = resolve_backend_selection(AUTO_PRIVATE_BACKEND, "default", "auto")
    assert selection.backend_id == LOCAL_MLX_BACKEND
    assert selection.model == "large-v2"
    assert selection.reason


def test_auto_private_raises_when_no_local_backends(monkeypatch):
    monkeypatch.setattr(transcription.platform, "system", lambda: "Linux")
    monkeypatch.setattr(transcription.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(transcription, "_has_mlx_whisper", lambda: False)
    monkeypatch.setattr(transcription, "_has_faster_whisper", lambda: False)
    monkeypatch.setattr(transcription, "_has_whispercpp", lambda: False)

    with pytest.raises(BackendNotAvailableError):
        resolve_backend_selection(AUTO_PRIVATE_BACKEND, "default", "auto")


def test_apply_transcript_formatting_modes():
    text = "Hello world. This is a test!"
    sentences = apply_transcript_formatting(text, "sentences")
    assert sentences.count("\n") == 1
    raw = apply_transcript_formatting(text, "raw")
    assert raw == text


def test_create_transcription_backend_cloud(monkeypatch):
    selection = BackendSelection(backend_id=CLOUD_BACKEND, model="gpt-4o-transcribe")
    backend = create_transcription_backend(selection)
    assert isinstance(backend, OpenAITranscriptionBackend)


def test_create_transcription_backend_delegates_to_faster(monkeypatch):
    captured: dict[str, tuple] = {}

    class FakeBackend:
        backend_id = LOCAL_FASTER_BACKEND

        def __init__(self, model: str, compute: str | None = None, **kwargs) -> None:
            captured["args"] = (model, compute, kwargs)

    monkeypatch.setattr(transcription, "FasterWhisperBackend", FakeBackend)

    selection = BackendSelection(
        backend_id=LOCAL_FASTER_BACKEND,
        model="large-v2",
        compute="int8",
    )
    backend = create_transcription_backend(selection)

    assert isinstance(backend, FakeBackend)
    assert captured["args"][0] == "large-v2"
    assert captured["args"][1] == "int8"
