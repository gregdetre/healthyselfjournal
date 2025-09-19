from __future__ import annotations

"""FastHTML application setup for the web journaling interface."""

from collections import OrderedDict
from dataclasses import dataclass, field
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, cast
from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
import sys
import subprocess
from starlette.staticfiles import StaticFiles

from ..audio import AudioCaptureResult
from ..config import CONFIG
from ..errors import (
    AUDIO_FORMAT_UNSUPPORTED,
    EMPTY_AUDIO,
    INACTIVE_SESSION,
    INVALID_PAYLOAD,
    MISSING_AUDIO,
    PROCESSING_FAILED,
    QUESTION_FAILED,
    SHORT_ANSWER_DISCARDED,
    UNKNOWN_SESSION,
    UPLOAD_TOO_LARGE,
    VOICE_DISABLED,
    MISSING_TEXT,
    TTS_FAILED,
    TTS_ERROR,
    REVEAL_FAILED,
    UNSUPPORTED_PLATFORM,
)
from ..events import log_event, init_event_logger, get_event_log_path
from ..session import PendingTranscriptionError, SessionConfig, SessionManager
from ..storage import load_transcript
from ..transcription import BackendNotAvailableError, resolve_backend_selection
from ..tts import TTSOptions, TTSError, resolve_tts_options, synthesize_text
from ..utils.audio_utils import (
    extension_for_media_type,
    is_supported_media_type,
    normalize_mime,
    should_discard_short_answer,
)
from ..utils.pending import count_pending_for_session, reconcile_command_for_dir
from ..utils.session_layout import build_segment_path, next_web_segment_name
from ..utils.time_utils import format_hh_mm_ss, format_minutes_text
from gjdutils.strings import jinja_render


_LOGGER = logging.getLogger(__name__)

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATIC_DIR = _PACKAGE_ROOT / "static"


def _error_response(
    error: str, status_code: int, detail: str | None = None
) -> JSONResponse:
    payload: dict[str, Any] = {"status": "error", "error": error}
    if detail:
        payload["detail"] = detail
    return JSONResponse(payload, status_code=status_code)


def _apply_security_headers(response: Response) -> Response:
    """Apply strict security headers suitable for the desktop shell."""

    csp_directives = [
        "default-src 'self'",
        "img-src 'self' data:",
        "style-src 'self' 'unsafe-inline'",
        "script-src 'self'",
        "font-src 'self'",
        "connect-src 'self'",
        "media-src 'self' blob: data:",
        "worker-src 'self' blob:",
        "frame-ancestors 'none'",
        "base-uri 'self'",
        "form-action 'self'",
        "object-src 'none'",
    ]
    headers = response.headers
    headers.setdefault("Content-Security-Policy", "; ".join(csp_directives))
    headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    headers.setdefault("Cross-Origin-Embedder-Policy", "require-corp")
    headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
    headers.setdefault("Referrer-Policy", "no-referrer")
    headers.setdefault("X-Frame-Options", "DENY")
    headers.setdefault("X-Content-Type-Options", "nosniff")
    headers.setdefault("Permissions-Policy", "camera=(), geolocation=(), microphone=(self)")
    return response


@lru_cache(maxsize=1)
def _get_fast_html_class():
    """Import FastHTML with compatibility patches for modern fastcore."""

    from fastcore import (
        xml as fast_xml,
    )  # Imported lazily to avoid hard dependency at module load

    original_ft = fast_xml.ft
    if not getattr(original_ft, "_hsj_returns_tuple", False):
        try:
            probe = original_ft("div")
        except Exception:  # pragma: no cover - defensive
            probe = None

        if isinstance(probe, fast_xml.FT):

            def compat_ft(tag: str, *c, **kwargs):
                ft_obj = original_ft(tag, *c, **kwargs)
                if isinstance(ft_obj, fast_xml.FT):
                    return ft_obj.tag, ft_obj.children, ft_obj.attrs
                return ft_obj

            compat_ft._hsj_returns_tuple = True  # type: ignore[attr-defined]
            fast_xml.ft = compat_ft  # type: ignore[assignment]
        else:
            setattr(original_ft, "_hsj_returns_tuple", True)  # type: ignore[attr-defined]

    from fasthtml import FastHTML  # Imported lazily so the patch above is in effect

    return FastHTML


@dataclass(slots=True)
class WebAppConfig:
    """Runtime configuration for the FastHTML web server."""

    sessions_dir: Path
    static_dir: Path = field(default=DEFAULT_STATIC_DIR)
    host: str = "127.0.0.1"
    port: int = 8765
    reload: bool = False
    # When enabled, GET / will resume the latest existing session if present
    resume: bool = False
    # Optional voice mode (browser playback of assistant questions)
    voice_enabled: bool = False
    tts_model: str | None = None
    tts_voice: str | None = None
    tts_format: str | None = None

    def resolved(self) -> "WebAppConfig":
        """Return a copy with absolute paths for filesystem access."""

        return WebAppConfig(
            sessions_dir=self.sessions_dir.expanduser().resolve(),
            static_dir=self.static_dir.expanduser().resolve(),
            host=self.host,
            port=self.port,
            reload=self.reload,
            resume=self.resume,
            voice_enabled=self.voice_enabled,
            tts_model=self.tts_model,
            tts_voice=self.tts_voice,
            tts_format=self.tts_format,
        )


@dataclass(slots=True)
class WebSessionState:
    """Book-keeping for an active web session."""

    manager: SessionManager
    current_question: str

    @property
    def session_id(self) -> str:
        state = self.manager.state
        if state is None:  # pragma: no cover - defensive guard
            raise RuntimeError("Session state not initialised")
        return state.session_id


def build_app(config: WebAppConfig) -> Any:
    """Construct and configure a FastHTML app instance."""

    resolved = config.resolved()
    resolved.sessions_dir.mkdir(parents=True, exist_ok=True)
    resolved.static_dir.mkdir(parents=True, exist_ok=True)

    # Ensure the append-only metadata event logger is initialised for the web server
    try:
        init_event_logger(resolved.sessions_dir)
        log_event(
            "web.server.start",
            {
                "ui": "web",
                "sessions_dir": str(resolved.sessions_dir),
                "host": resolved.host,
                "port": resolved.port,
                "reload": bool(resolved.reload),
                "events_log": str(get_event_log_path() or ""),
            },
        )
    except Exception:
        # Defensive: never fail app construction due to logging setup
        pass

    FastHTML = _get_fast_html_class()
    app = FastHTML()
    app.state.config = resolved
    app.state.sessions = OrderedDict()
    app.state.max_sessions = 4
    app.state.resume = bool(resolved.resume)
    # Configure voice mode and TTS options for this app instance
    voice_enabled = bool(resolved.voice_enabled or CONFIG.speak_llm)
    app.state.voice_enabled = voice_enabled
    if voice_enabled:
        overrides = {
            "model": resolved.tts_model,
            "voice": resolved.tts_voice,
            "audio_format": resolved.tts_format,
        }
        app.state.tts_options = resolve_tts_options(overrides)
    else:
        app.state.tts_options = None

    @app.middleware("http")  # type: ignore[attr-defined]
    async def _security_middleware(request: Request, call_next):
        response = await call_next(request)
        return _apply_security_headers(response)

    # Serve static files (JS, CSS, media) under /static/
    app.mount(
        "/static",
        StaticFiles(directory=str(resolved.static_dir), check_dir=False),
        name="static",
    )

    @app.route("/")
    def index():
        """Landing page that boots or resumes a session and redirects to pretty URL."""

        try:
            if bool(getattr(app.state, "resume", False)):
                state = _start_or_resume_session(app)
            else:
                state = _start_session(app)
        except Exception as exc:  # pragma: no cover - surface to browser
            _LOGGER.exception("Failed to start web session")
            return """
                <!doctype html>
                <html lang=\"en\">
                  <head>
                    <meta charset=\"utf-8\" />
                    <title>Healthy Self Journal (Web)</title>
                  </head>
                  <body>
                    <main style=\"max-width:600px;margin:3rem auto;font-family:system-ui\">
                      <h1>Healthy Self Journal</h1>
                      <p>Sorry, the web interface could not start: check your STT/LLM configuration.</p>
                    </main>
                  </body>
                </html>
                """

        # Redirect to pretty session URL: /journal/<sessions_dir_name>/<session_id>/
        resolved_cfg = getattr(app.state, "config", None)
        sessions_dir_path = (
            getattr(resolved_cfg, "sessions_dir", Path("sessions"))
            if resolved_cfg
            else Path("sessions")
        )
        sessions_dir_name = Path(str(sessions_dir_path)).name
        location = f"/journal/{sessions_dir_name}/{state.session_id}/"
        return Response(status_code=307, headers={"Location": location})

    @app.get("/journal/{sessions_dir}/{session_id}/")  # type: ignore[attr-defined]
    def journal_page(sessions_dir: str, session_id: str):
        """Render the main recording UI for a specific session id.

        - If the session is active in-memory, render immediately.
        - If not, and a matching markdown exists on disk, resume it.
        - If sessions_dir doesn't match current config's basename, redirect.
        """

        # Ensure sessions_dir in URL matches configured one; redirect if not
        resolved_cfg = getattr(app.state, "config", None)
        configured_sessions_dir = (
            getattr(resolved_cfg, "sessions_dir", Path("sessions"))
            if resolved_cfg
            else Path("sessions")
        )
        configured_name = Path(str(configured_sessions_dir)).name
        if sessions_dir != configured_name:
            return Response(
                status_code=307,
                headers={"Location": f"/journal/{configured_name}/{session_id}/"},
            )

        # Obtain or resume the requested session
        state = _touch_session(app, session_id)
        if state is None:
            state = _resume_specific_session(app, session_id)
            if state is None:
                return _error_response(UNKNOWN_SESSION, status_code=404)

        return _render_session_page(
            app, state, static_assets_ready=_static_assets_ready(app)
        )

    @app.post("/session/{session_id}/upload")  # type: ignore[attr-defined]
    async def upload(session_id: str, request: Request):
        state = _touch_session(app, session_id)
        if state is None:
            return _error_response(UNKNOWN_SESSION, status_code=404)

        form = await request.form()
        upload = form.get("audio")
        if upload is None:
            return _error_response(MISSING_AUDIO, status_code=400)
        if not isinstance(upload, UploadFile):
            return _error_response(INVALID_PAYLOAD, status_code=400)

        mime_form = form.get("mime")
        if isinstance(mime_form, bytes):
            try:
                mime_form = mime_form.decode()
            except Exception:
                mime_form = None
        mime = normalize_mime(mime_form) or normalize_mime(upload.content_type)
        if mime is None:
            mime = "audio/webm"

        if not is_supported_media_type(mime):
            return _error_response(
                AUDIO_FORMAT_UNSUPPORTED,
                status_code=415,
                detail="Only Opus WEBM/OGG uploads are supported in the web interface.",
            )

        blob = await upload.read()
        size_bytes = len(blob)
        if size_bytes == 0:
            return _error_response(EMPTY_AUDIO, status_code=400)
        if size_bytes > CONFIG.web_upload_max_bytes:
            return _error_response(
                UPLOAD_TOO_LARGE,
                status_code=413,
                detail=f"Upload exceeded {CONFIG.web_upload_max_bytes} bytes",
            )

        def _to_float(val: Any, default: float) -> float:
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, (str, bytes)):
                try:
                    s = val.decode() if isinstance(val, bytes) else val
                    return float(s)
                except Exception:
                    return default
            return default

        duration_ms = _to_float(form.get("duration_ms"), 0.0)
        voiced_ms = _to_float(form.get("voiced_ms"), duration_ms)

        def _truthy(val: Any) -> bool:
            if isinstance(val, (int, float)):
                return bool(val)
            if isinstance(val, bytes):
                try:
                    val = val.decode()
                except Exception:
                    return False
            if isinstance(val, str):
                return val.strip().lower() in {"1", "true", "yes", "on", "q"}
            return False

        quit_after = _truthy(form.get("quit_after"))

        session_state = state.manager.state
        if session_state is None:
            return _error_response(INACTIVE_SESSION, status_code=409)

        target_dir = session_state.audio_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        previous_index = session_state.response_index
        extension = extension_for_media_type(mime, upload.filename)
        index_hint = previous_index + 1
        index, segment_basename = next_web_segment_name(
            target_dir, start_index=index_hint
        )
        target_path = build_segment_path(target_dir, segment_basename, extension)
        session_state.response_index = max(index - 1, previous_index)

        target_path.write_bytes(blob)
        duration_seconds = max(duration_ms, 0.0) / 1000.0
        voiced_seconds = max(voiced_ms, 0.0) / 1000.0

        capture = AudioCaptureResult(
            wav_path=target_path,
            mp3_path=None,
            duration_seconds=duration_seconds,
            voiced_seconds=voiced_seconds,
            cancelled=False,
            quit_after=quit_after,
            discarded_short_answer=False,
        )

        log_event(
            "web.upload.received",
            {
                "ui": "web",
                "session_id": session_id,
                "filename": target_path.name,
                "content_type": mime,
                "bytes": size_bytes,
                "duration_seconds": round(duration_seconds, 2),
                "quit_after": quit_after,
            },
        )

        if should_discard_short_answer(duration_seconds, voiced_seconds, CONFIG):
            session_state.response_index = previous_index
            try:
                target_path.unlink(missing_ok=True)
            except Exception:
                pass
            log_event(
                "session.exchange.discarded_short",
                {
                    "ui": "web",
                    "session_id": session_id,
                    "response_index": previous_index + 1,
                    "duration_seconds": round(duration_seconds, 2),
                    "voiced_seconds": round(voiced_seconds, 2),
                },
            )
            return _error_response(
                SHORT_ANSWER_DISCARDED,
                status_code=422,
                detail="Response was too short or quiet; no transcript generated.",
            )

        try:
            exchange = state.manager.process_uploaded_exchange(
                state.current_question,
                capture,
                segment_label=target_path.name,
            )
        except PendingTranscriptionError as exc:
            _LOGGER.exception("STT pending; placeholder recorded")
            command = reconcile_command_for_dir(state.manager.config.base_dir)
            detail = (
                f"{exc.error}. Audio saved; run {command} to backfill."
            )
            return _error_response(
                PROCESSING_FAILED,
                status_code=503,
                detail=detail,
            )
        except BackendNotAvailableError as exc:
            session_state.response_index = previous_index
            _LOGGER.exception("STT backend rejected uploaded audio")
            return _error_response(
                AUDIO_FORMAT_UNSUPPORTED,
                status_code=415,
                detail=str(exc),
            )
        except Exception as exc:  # pragma: no cover - runtime path
            session_state.response_index = previous_index
            _LOGGER.exception("Failed to process uploaded exchange")
            detail = str(exc)
            if "opus" in detail.lower():
                return _error_response(
                    AUDIO_FORMAT_UNSUPPORTED,
                    status_code=415,
                    detail=detail,
                )
            return _error_response(PROCESSING_FAILED, status_code=500, detail=detail)

        summary_scheduled = True
        try:
            state.manager.schedule_summary_regeneration()
        except Exception as exc:  # pragma: no cover - best-effort logging
            summary_scheduled = False
            _LOGGER.exception("Summary scheduling failed: %s", exc)

        try:
            next_question = state.manager.generate_next_question(exchange.transcript)
            state.current_question = next_question.question
        except Exception as exc:  # pragma: no cover - runtime path
            _LOGGER.exception("Next question generation failed")
            return _error_response(QUESTION_FAILED, status_code=502, detail=str(exc))

        log_event(
            "web.upload.processed",
            {
                "ui": "web",
                "session_id": session_id,
                "segment_label": target_path.name,
                "response_index": (
                    state.manager.state.response_index if state.manager.state else None
                ),
                "transcript_chars": len(exchange.transcript),
                "next_question_chars": len(state.current_question or ""),
                "quit_after": quit_after,
            },
        )

        # Compute new cumulative total duration across exchanges
        st = state.manager.state
        try:
            total_seconds = (
                sum(e.audio.duration_seconds for e in st.exchanges) if st else 0.0
            )
        except Exception:
            total_seconds = 0.0

        response_payload = {
            "status": "ok",
            "session_id": session_id,
            "segment_label": target_path.name,
            "duration_seconds": round(exchange.audio.duration_seconds, 2),
            "total_duration_seconds": round(total_seconds, 2),
            "total_duration_hms": format_hh_mm_ss(total_seconds),
            "total_duration_minutes_text": format_minutes_text(total_seconds),
            "transcript": exchange.transcript,
            "next_question": state.current_question,
            "llm_model": getattr(next_question, "model", None),
            "summary_scheduled": summary_scheduled,
            "quit_after": quit_after,
        }
        return JSONResponse(response_payload, status_code=201)

    @app.post("/session/{session_id}/tts")  # type: ignore[attr-defined]
    async def tts(session_id: str, request: Request):
        """Synthesize TTS for a given text and return audio bytes.

        Expects JSON body: {"text": "..."}. Returns audio/* content.
        """

        # Validate session
        state = _touch_session(app, session_id)
        if state is None:
            return _error_response(UNKNOWN_SESSION, status_code=404)

        if not getattr(app.state, "voice_enabled", False):
            return _error_response(VOICE_DISABLED, status_code=400)

        try:
            try:
                payload = await request.json()
                if not isinstance(payload, dict):
                    raise ValueError("invalid json")
            except Exception:
                form = await request.form()
                payload = {"text": form.get("text")}

            text_val = payload.get("text")
            if isinstance(text_val, bytes):
                try:
                    text_val = text_val.decode()
                except Exception:
                    text_val = ""
            if not isinstance(text_val, str):
                text_val = ""
            text = text_val.strip()
            if not text:
                return _error_response(MISSING_TEXT, status_code=400)

            tts_opts: TTSOptions | None = getattr(app.state, "tts_options", None)
            if tts_opts is None:
                tts_opts = resolve_tts_options(None)

            audio_bytes = synthesize_text(text, tts_opts)
            content_type = _tts_format_to_mime(str(tts_opts.audio_format))

            headers = {"Cache-Control": "no-store"}
            return Response(
                content=audio_bytes, media_type=content_type, headers=headers
            )
        except TTSError as exc:
            _LOGGER.exception("TTS failed: %s", exc)
            return _error_response(TTS_FAILED, status_code=502, detail=str(exc))
        except Exception as exc:  # pragma: no cover - generic surfacing
            _LOGGER.exception("TTS endpoint error: %s", exc)
            return _error_response(TTS_ERROR, status_code=500, detail=str(exc))

    @app.post("/session/{session_id}/reveal")  # type: ignore[attr-defined]
    async def reveal(session_id: str):
        """Reveal the session's markdown file in the OS file manager.

        - macOS: uses `open -R`
        - Others: returns a 501 to indicate unsupported platform for now
        """

        state = _touch_session(app, session_id)
        if state is None:
            return _error_response(UNKNOWN_SESSION, status_code=404)

        try:
            st = state.manager.state
            if st is None:
                return _error_response(INACTIVE_SESSION, status_code=409)

            md_path = st.markdown_path
            if sys.platform == "darwin":
                try:
                    subprocess.run(["open", "-R", str(md_path)], check=False)
                    return JSONResponse({"status": "ok"}, status_code=200)
                except Exception as exc:  # pragma: no cover - runtime path
                    _LOGGER.exception("Reveal failed: %s", exc)
                    return _error_response(
                        REVEAL_FAILED, status_code=500, detail=str(exc)
                    )
            else:
                return _error_response(UNSUPPORTED_PLATFORM, status_code=501)
        except Exception as exc:  # pragma: no cover - generic surfacing
            _LOGGER.exception("Reveal endpoint error: %s", exc)
            return _error_response(REVEAL_FAILED, status_code=500, detail=str(exc))

    @app.get("/session/{session_id}/reveal")  # type: ignore[attr-defined]
    async def reveal_get(session_id: str):
        return await reveal(session_id)

    return app


def run_app(config: WebAppConfig) -> None:
    """Run the FastHTML development server."""

    app = build_app(config)
    # Run via uvicorn to support installed FastHTML versions without app.run()
    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port, reload=config.reload)


def _build_session_manager(app: Any) -> SessionManager:
    """Create a SessionManager using current config and resolved STT backend."""

    resolved: WebAppConfig = app.state.config

    try:
        selection = resolve_backend_selection(
            CONFIG.stt_backend,
            CONFIG.model_stt,
            CONFIG.stt_compute,
        )
    except (ValueError, BackendNotAvailableError) as exc:
        raise RuntimeError(f"Unable to configure STT backend: {exc}") from exc

    session_cfg = SessionConfig(
        base_dir=resolved.sessions_dir,
        llm_model=CONFIG.model_llm,
        stt_model=selection.model,
        stt_backend=selection.backend_id,
        stt_compute=selection.compute,
        opening_question=CONFIG.opening_question,
        language="en",
        stt_formatting=CONFIG.stt_formatting,
        stt_backend_requested=CONFIG.stt_backend,
        stt_model_requested=CONFIG.model_stt,
        stt_compute_requested=CONFIG.stt_compute,
        stt_auto_private_reason=selection.reason,
        stt_backend_selection=selection,
        stt_warnings=selection.warnings,
    )

    manager = SessionManager(session_cfg)
    return manager


def _render_session_page(
    app: Any, state: WebSessionState, *, static_assets_ready: bool
) -> str:
    """Render the HTML shell for the given session state."""

    # Resolve voice/TTS options for this app instance
    voice_enabled: bool = bool(getattr(app.state, "voice_enabled", False))
    tts_opts: TTSOptions | None = getattr(app.state, "tts_options", None)
    tts_format = tts_opts.audio_format if tts_opts else CONFIG.tts_format

    # Compute current total duration for display (frontmatter authoritative)
    try:
        doc = load_transcript(state.manager.state.markdown_path)  # type: ignore[arg-type]
        total_seconds = float(doc.frontmatter.data.get("duration_seconds", 0.0))
    except Exception:  # pragma: no cover - defensive fallback
        st = getattr(state.manager, "state", None)
        total_seconds = (
            sum(e.audio.duration_seconds for e in st.exchanges) if st else 0.0
        )
    total_hms = format_hh_mm_ss(total_seconds)
    total_minutes_text = format_minutes_text(total_seconds)

    base_dir = state.manager.config.base_dir
    try:
        pending_count = count_pending_for_session(base_dir, state.session_id)
    except Exception:  # pragma: no cover - defensive fallback
        pending_count = 0
    reconcile_cmd = reconcile_command_for_dir(base_dir)

    # Server/runtime context for debug panel
    resolved_cfg = getattr(app.state, "config", None)
    server_host = (
        getattr(resolved_cfg, "host", "127.0.0.1") if resolved_cfg else "127.0.0.1"
    )
    server_port = getattr(resolved_cfg, "port", 8765) if resolved_cfg else 8765
    server_reload = (
        bool(getattr(resolved_cfg, "reload", False)) if resolved_cfg else False
    )
    server_resume = bool(getattr(app.state, "resume", False))
    sessions_dir = (
        str(getattr(resolved_cfg, "sessions_dir", "sessions"))
        if resolved_cfg
        else "sessions"
    )

    # Session/LLM/STT context for debug panel
    app_version = state.manager.config.app_version
    llm_model = state.manager.config.llm_model
    stt_backend = state.manager.config.stt_backend
    stt_model = state.manager.config.stt_model
    stt_compute = state.manager.config.stt_compute or ""
    stt_formatting = state.manager.config.stt_formatting
    stt_auto_reason = state.manager.config.stt_auto_private_reason or ""
    stt_warnings = list(state.manager.config.stt_warnings or [])
    voice_rms_dbfs_threshold = CONFIG.voice_rms_dbfs_threshold

    if not static_assets_ready:
        log_event(
            "web.static.assets_missing",
            {
                "ui": "web",
                "static_dir": str(getattr(resolved_cfg, "static_dir", "")),
            },
        )

    body = _render_session_shell(
        state,
        voice_enabled=voice_enabled,
        tts_format=str(tts_format),
        total_seconds=total_seconds,
        total_hms=total_hms,
        total_minutes_text=total_minutes_text,
        server_host=server_host,
        server_port=server_port,
        server_reload=server_reload,
        server_resume=server_resume,
        sessions_dir=sessions_dir,
        app_version=app_version,
        llm_model=llm_model,
        stt_backend=stt_backend,
        stt_model=stt_model,
        stt_compute=stt_compute,
        stt_formatting=stt_formatting,
        stt_auto_reason=stt_auto_reason,
        stt_warnings=stt_warnings,
        voice_rms_dbfs_threshold=voice_rms_dbfs_threshold,
        static_assets_ready=static_assets_ready,
        pending_count=pending_count,
        reconcile_command=reconcile_cmd,
    )
    return body


def _sessions_map(app: Any) -> OrderedDict[str, WebSessionState]:
    return cast(OrderedDict[str, WebSessionState], app.state.sessions)


def _register_session(app: Any, session_id: str, web_state: WebSessionState) -> None:
    sessions = _sessions_map(app)
    sessions[session_id] = web_state
    sessions.move_to_end(session_id)
    _trim_sessions(app)
    _log_active_sessions(app)


def _touch_session(app: Any, session_id: str) -> WebSessionState | None:
    sessions = _sessions_map(app)
    state = sessions.get(session_id)
    if state is not None:
        sessions.move_to_end(session_id)
        _log_active_sessions(app)
    return state


def _trim_sessions(app: Any) -> None:
    sessions = _sessions_map(app)
    max_sessions = max(1, int(getattr(app.state, "max_sessions", 4)))
    while len(sessions) > max_sessions:
        evicted_id, _ = sessions.popitem(last=False)
        log_event("web.session.evicted", {"ui": "web", "session_id": evicted_id})


def _log_active_sessions(app: Any) -> None:
    try:
        session_ids = list(_sessions_map(app).keys())
        log_event("web.sessions.active", {"ui": "web", "session_ids": session_ids})
    except Exception:
        pass


def _static_assets_ready(app: Any) -> bool:
    resolved_cfg = getattr(app.state, "config", None)
    if resolved_cfg is None:
        return True
    try:
        static_dir = Path(getattr(resolved_cfg, "static_dir", ""))
    except Exception:
        return True
    candidate = static_dir / "js" / "app.js"
    try:
        return candidate.exists()
    except Exception:
        return True


def _tts_format_to_mime(fmt: str) -> str:
    mapping = {
        "wav": "audio/wav",
        "wave": "audio/wave",
        "mp3": "audio/mpeg",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
        "opus": "audio/ogg",
        "aac": "audio/aac",
        "pcm": "audio/pcm",
    }
    return mapping.get(fmt.lower(), "audio/wave")


def _resume_specific_session(app: Any, session_id: str) -> WebSessionState | None:
    """Resume a specific session by id if its markdown exists on disk."""

    manager = _build_session_manager(app)
    base_dir = manager.config.base_dir
    md_path = base_dir / f"{session_id}.md"
    if not md_path.exists():
        return None

    state = manager.resume(md_path)

    # Determine initial question: try to generate from existing body, else use opener
    try:
        from ..storage import load_transcript

        doc = load_transcript(state.markdown_path)
        if doc.body.strip():
            try:
                next_q = manager.generate_next_question(doc.body)
                current_question = next_q.question
            except Exception:
                current_question = manager.config.opening_question
        else:
            current_question = manager.config.opening_question
    except Exception:
        current_question = manager.config.opening_question

    web_state = WebSessionState(manager=manager, current_question=current_question)
    _register_session(app, state.session_id, web_state)
    return web_state


def _start_session(app: Any) -> WebSessionState:
    """Initialise a new journaling session for the web client."""

    manager = _build_session_manager(app)
    state = manager.start()
    current_question = manager.config.opening_question

    web_state = WebSessionState(manager=manager, current_question=current_question)
    _register_session(app, state.session_id, web_state)

    log_event(
        "web.session.started",
        {
            "ui": "web",
            "session_id": state.session_id,
            "markdown_path": state.markdown_path.name,
            "audio_dir": str(state.audio_dir),
        },
    )
    return web_state


def _start_or_resume_session(app: Any) -> WebSessionState:
    """Start a new session or resume the most recent existing session."""

    manager = _build_session_manager(app)
    base_dir = manager.config.base_dir
    markdown_files = sorted((p for p in base_dir.glob("*.md")), reverse=True)

    if not markdown_files:
        return _start_session(app)

    latest_md = markdown_files[0]
    state = manager.resume(latest_md)

    # Determine initial question: try to generate from existing body, else use opener
    try:
        from ..storage import load_transcript

        doc = load_transcript(state.markdown_path)
        if doc.body.strip():
            try:
                next_q = manager.generate_next_question(doc.body)
                current_question = next_q.question
            except Exception:
                current_question = manager.config.opening_question
        else:
            current_question = manager.config.opening_question
    except Exception:
        current_question = manager.config.opening_question

    web_state = WebSessionState(manager=manager, current_question=current_question)
    _register_session(app, state.session_id, web_state)

    log_event(
        "web.session.resumed",
        {
            "ui": "web",
            "session_id": state.session_id,
            "markdown_path": state.markdown_path.name,
            "audio_dir": str(state.audio_dir),
        },
    )
    return web_state


def _render_session_shell(
    state: WebSessionState,
    *,
    voice_enabled: bool,
    tts_format: str,
    total_seconds: float,
    total_hms: str,
    total_minutes_text: str,
    server_host: str,
    server_port: int,
    server_reload: bool,
    server_resume: bool,
    sessions_dir: str,
    app_version: str,
    llm_model: str,
    stt_backend: str,
    stt_model: str,
    stt_compute: str,
    stt_formatting: str,
    stt_auto_reason: str,
    stt_warnings: list[str],
    voice_rms_dbfs_threshold: float,
    static_assets_ready: bool,
) -> str:
    """Return the base HTML shell; dynamic behaviour handled client-side."""

    session_id = state.session_id
    question = state.current_question
    short_duration = CONFIG.short_answer_duration_seconds
    short_voiced = CONFIG.short_answer_voiced_seconds
    voice_attr = "true" if voice_enabled else "false"
    tts_mime = _tts_format_to_mime(tts_format)

    template_path = (
        Path(__file__).resolve().parent / "templates" / "session_shell.html.jinja"
    )
    template_str = template_path.read_text(encoding="utf-8")

    return jinja_render(
        template_str,
        {
            "session_id": session_id,
            "question": question,
            "short_duration": short_duration,
            "short_voiced": short_voiced,
            "voice_attr": voice_attr,
            "tts_mime": tts_mime,
            "total_seconds": round(float(total_seconds or 0.0), 2),
            "total_hms": total_hms,
            "total_minutes_text": total_minutes_text,
            "server_host": server_host,
            "server_port": server_port,
            "server_reload": server_reload,
            "server_resume": server_resume,
            "sessions_dir": sessions_dir,
            "app_version": app_version,
            "llm_model": llm_model,
            "stt_backend": stt_backend,
            "stt_model": stt_model,
            "stt_compute": stt_compute,
            "stt_formatting": stt_formatting,
            "stt_auto_reason": stt_auto_reason,
            "stt_warnings": stt_warnings,
            "voice_rms_dbfs_threshold": voice_rms_dbfs_threshold,
            "static_assets_ready": static_assets_ready,
        },
        filesystem_loader=template_path.parent,
    )
