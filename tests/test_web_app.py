from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("starlette")
pytest.importorskip("fasthtml")
pytest.importorskip("fastcore")

from starlette.testclient import TestClient

from healthyselfjournal.errors import (
    AUDIO_FORMAT_UNSUPPORTED,
    SHORT_ANSWER_DISCARDED,
    UPLOAD_TOO_LARGE,
    VOICE_DISABLED,
)
from healthyselfjournal.session import SessionManager
from healthyselfjournal.transcription import BackendSelection
from healthyselfjournal.web.app import WebAppConfig, build_app
from healthyselfjournal.tts import TTSOptions
from tests.stubs import StubTranscriptionBackend, stub_followup_question, stub_tts


def _extract_session_id(html: str) -> str:
    match = re.search(r"data-session-id=\"([^\"]+)\"", html)
    assert match, "session id not found in HTML"
    return match.group(1)

@pytest.fixture()
def web_app(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "healthyselfjournal.web.app.resolve_backend_selection",
        lambda *_, **__: BackendSelection(
            backend_id="stub",
            model="stub-model",
            compute=None,
        ),
    )
    monkeypatch.setattr(
        SessionManager, "schedule_summary_regeneration", lambda self: None
    )
    monkeypatch.setattr(
        "healthyselfjournal.session.create_transcription_backend",
        lambda *_, **__: StubTranscriptionBackend(),
    )
    monkeypatch.setattr(
        "healthyselfjournal.session.generate_followup_question",
        lambda _: stub_followup_question(),
    )

    config = WebAppConfig(
        sessions_dir=tmp_path,
        static_dir=tmp_path / "static",
    )
    return build_app(config)


def test_tts_endpoint_disabled_returns_error(tmp_path: Path, web_app):
    client = TestClient(web_app, follow_redirects=True)
    response = client.get("/")
    assert response.status_code == 200
    session_id = _extract_session_id(response.text)

    result = client.post(f"/session/{session_id}/tts", json={"text": "Hello"})
    assert result.status_code == 400
    data = result.json()
    assert data == {"status": "error", "error": VOICE_DISABLED}


@pytest.mark.parametrize(
    "tts_format, expected_type",
    [("wav", "audio/wav"), ("mp3", "audio/mpeg")],
)
def test_tts_endpoint_ok_when_enabled(tmp_path: Path, monkeypatch, tts_format: str, expected_type: str):
    config = WebAppConfig(
        sessions_dir=tmp_path,
        static_dir=tmp_path / "static",
        voice_enabled=True,
        tts_format=tts_format,
    )
    monkeypatch.setattr("healthyselfjournal.web.app.synthesize_text", stub_tts)

    app = build_app(config)
    client = TestClient(app, follow_redirects=True)
    response = client.get("/")
    session_id = _extract_session_id(response.text)

    result = client.post(f"/session/{session_id}/tts", json={"text": "Speak me"})
    assert result.status_code == 200
    assert result.headers.get("content-type", "").startswith(expected_type)
    assert result.content == b"FAKEAUDIOBYTES"


def test_landing_page_warns_when_static_assets_missing(web_app):
    client = TestClient(web_app, follow_redirects=True)
    response = client.get("/")
    assert 'Static assets missing.' in response.text
    assert 'data-voice-rms-dbfs-threshold="' in response.text


def test_upload_creates_session_artifacts_and_logs_ui(tmp_path: Path, monkeypatch, web_app):
    captured: list[tuple[str, dict]] = []

    def _capture(name: str, payload: dict) -> None:
        captured.append((name, payload))

    monkeypatch.setattr("healthyselfjournal.events.log_event", _capture)
    monkeypatch.setattr("healthyselfjournal.web.app.log_event", _capture)
    monkeypatch.setattr("healthyselfjournal.session.log_event", _capture)

    client = TestClient(web_app, follow_redirects=True)
    response = client.get("/")
    session_id = _extract_session_id(response.text)

    upload_path = f"/session/{session_id}/upload"
    payload = {
        "duration_ms": "1500",
        "voiced_ms": "900",
    }
    files = {"audio": ("clip.webm", b"faux-data", "audio/webm")}

    result = client.post(upload_path, data=payload, files=files)
    assert result.status_code == 201
    data = result.json()
    assert data["status"] == "ok"
    assert data["session_id"] == session_id
    assert data["segment_label"].startswith("browser-")
    assert data["total_duration_seconds"] == 1.5
    assert data["quit_after"] is False

    session_dir = tmp_path / session_id
    audio_path = session_dir / "browser-001.webm"
    markdown_path = tmp_path / f"{session_id}.md"
    assert audio_path.exists()
    assert markdown_path.exists()
    stt_payload = json.loads(audio_path.with_suffix(".stt.json").read_text(encoding="utf-8"))
    assert stt_payload["text"] == "stub transcript"

    recorded = [evt for evt in captured if evt[0] == "session.exchange.recorded"]
    assert recorded, "expected session.exchange.recorded event"
    assert recorded[-1][1]["ui"] == "web"


def test_upload_quit_after_flag_propagates(web_app):
    client = TestClient(web_app, follow_redirects=True)
    session_id = _extract_session_id(client.get("/").text)

    payload = {"duration_ms": "1500", "voiced_ms": "900", "quit_after": "1"}
    files = {"audio": ("clip.webm", b"faux-data", "audio/webm")}

    result = client.post(f"/session/{session_id}/upload", data=payload, files=files)
    assert result.status_code == 201
    data = result.json()
    assert data["quit_after"] is True


def test_upload_short_answer_discarded_returns_422(web_app):
    client = TestClient(web_app, follow_redirects=True)
    session_id = _extract_session_id(client.get("/").text)

    payload = {"duration_ms": "200", "voiced_ms": "100"}
    files = {"audio": ("clip.webm", b"tiny", "audio/webm")}

    result = client.post(f"/session/{session_id}/upload", data=payload, files=files)
    assert result.status_code == 422
    data = result.json()
    assert data["status"] == "error"
    assert data["error"] == SHORT_ANSWER_DISCARDED


def test_upload_rejects_unsupported_mime_type(web_app):
    client = TestClient(web_app, follow_redirects=True)
    session_id = _extract_session_id(client.get("/").text)
    payload = {"duration_ms": "500", "voiced_ms": "400"}
    files = {"audio": ("clip.bin", b"data", "video/mp4")}

    result = client.post(f"/session/{session_id}/upload", data=payload, files=files)
    assert result.status_code == 415
    data = result.json()
    assert data["error"] == AUDIO_FORMAT_UNSUPPORTED


def test_upload_enforces_size_limit(web_app, monkeypatch):
    monkeypatch.setattr("healthyselfjournal.config.CONFIG.web_upload_max_bytes", 8)
    client = TestClient(web_app, follow_redirects=True)
    session_id = _extract_session_id(client.get("/").text)

    payload = {"duration_ms": "500", "voiced_ms": "400"}
    files = {"audio": ("clip.webm", b"0123456789", "audio/webm")}

    result = client.post(f"/session/{session_id}/upload", data=payload, files=files)
    assert result.status_code == 413
    data = result.json()
    assert data["error"] == UPLOAD_TOO_LARGE


def test_reveal_endpoint_get_matches_post(tmp_path: Path, monkeypatch):
    config = WebAppConfig(sessions_dir=tmp_path, static_dir=tmp_path / "static")
    monkeypatch.setattr(
        "healthyselfjournal.web.app.resolve_backend_selection",
        lambda *_, **__: BackendSelection(backend_id="stub", model="stub-model", compute=None),
    )
    monkeypatch.setattr(
        "healthyselfjournal.session.create_transcription_backend",
        lambda *_, **__: StubTranscriptionBackend(),
    )
    monkeypatch.setattr(
        "healthyselfjournal.session.generate_followup_question",
        lambda _: stub_followup_question(),
    )
    monkeypatch.setattr("healthyselfjournal.web.app.sys.platform", "darwin", raising=False)
    called: list[list[str]] = []
    monkeypatch.setattr(
        "healthyselfjournal.web.app.subprocess.run",
        lambda args, check=False: called.append(args),
    )

    app = build_app(config)
    client = TestClient(app, follow_redirects=True)
    session_id = _extract_session_id(client.get("/").text)

    payload = {"duration_ms": "1000", "voiced_ms": "800"}
    files = {"audio": ("clip.webm", b"faux-data", "audio/webm")}
    assert client.post(f"/session/{session_id}/upload", data=payload, files=files).status_code == 201

    assert client.post(f"/session/{session_id}/reveal").status_code == 200
    assert client.get(f"/session/{session_id}/reveal").status_code == 200
    assert called, "expected subprocess.run to be invoked"


def test_resume_latest_session_when_enabled(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "healthyselfjournal.web.app.resolve_backend_selection",
        lambda *_, **__: BackendSelection(backend_id="stub", model="stub-model", compute=None),
    )
    monkeypatch.setattr(
        SessionManager, "schedule_summary_regeneration", lambda self: None
    )
    monkeypatch.setattr(
        "healthyselfjournal.session.create_transcription_backend",
        lambda *_, **__: StubTranscriptionBackend(),
    )
    monkeypatch.setattr(
        "healthyselfjournal.session.generate_followup_question",
        lambda _: stub_followup_question(),
    )

    client1 = TestClient(
        build_app(WebAppConfig(sessions_dir=tmp_path, static_dir=tmp_path / "static")),
        follow_redirects=True,
    )
    first_sid = _extract_session_id(client1.get("/").text)

    client2 = TestClient(
        build_app(
            WebAppConfig(
                sessions_dir=tmp_path,
                static_dir=tmp_path / "static",
                resume=True,
            )
        ),
        follow_redirects=True,
    )
    second_response = client2.get("/")
    resumed_sid = _extract_session_id(second_response.text)
    assert resumed_sid == first_sid


def test_setup_page_renders_without_surplus_context(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "healthyselfjournal.web.app.resolve_backend_selection",
        lambda *_, **__: BackendSelection(
            backend_id="stub",
            model="stub-model",
            compute=None,
        ),
    )
    monkeypatch.setattr(
        "healthyselfjournal.session.create_transcription_backend",
        lambda *_, **__: StubTranscriptionBackend(),
    )
    monkeypatch.setattr(
        "healthyselfjournal.session.generate_followup_question",
        lambda _: stub_followup_question(),
    )

    app = build_app(
        WebAppConfig(
            sessions_dir=tmp_path,
            static_dir=tmp_path / "static",
            desktop_setup=True,
        )
    )
    client = TestClient(app, follow_redirects=True)

    r = client.get("/setup")
    assert r.status_code == 200
    assert "Welcome" in r.text
