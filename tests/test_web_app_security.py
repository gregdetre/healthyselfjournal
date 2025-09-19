from __future__ import annotations

from pathlib import Path

from starlette.testclient import TestClient

from healthyselfjournal.web.app import WebAppConfig, build_app


def test_security_headers_applied(tmp_path: Path) -> None:
    config = WebAppConfig(sessions_dir=tmp_path, reload=False)
    app = build_app(config)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/static/css/app.css")

    headers = response.headers
    csp = headers.get("Content-Security-Policy", "")
    assert "default-src 'self'" in csp
    assert "media-src 'self' blob: data:" in csp
    assert headers.get("Permissions-Policy") == "camera=(), geolocation=(), microphone=(self)"
    assert headers.get("Cross-Origin-Opener-Policy") == "same-origin"
    assert headers.get("X-Frame-Options") == "DENY"
