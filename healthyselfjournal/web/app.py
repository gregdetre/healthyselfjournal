from __future__ import annotations

"""FastHTML application setup for the web journaling interface."""

from dataclasses import dataclass, field
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any
from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
import sys
import subprocess
from starlette.staticfiles import StaticFiles

from ..audio import AudioCaptureResult
from ..storage import load_transcript
from ..utils.time_utils import format_hh_mm_ss, format_minutes_text
from ..config import CONFIG
from ..events import log_event
from ..session import SessionConfig, SessionManager
from ..transcription import BackendNotAvailableError, resolve_backend_selection
from ..tts import TTSOptions, synthesize_text, TTSError
from gjdutils.strings import jinja_render


_LOGGER = logging.getLogger(__name__)

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATIC_DIR = _PACKAGE_ROOT / "static"


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

    FastHTML = _get_fast_html_class()
    app = FastHTML()
    app.state.config = resolved
    app.state.sessions = {}
    app.state.resume = bool(resolved.resume)
    # Configure voice mode and TTS options for this app instance
    voice_enabled = bool(resolved.voice_enabled or CONFIG.speak_llm)
    app.state.voice_enabled = voice_enabled
    app.state.tts_options = (
        TTSOptions(
            backend="openai",
            model=(resolved.tts_model or CONFIG.tts_model),
            voice=(resolved.tts_voice or CONFIG.tts_voice),
            audio_format=(resolved.tts_format or CONFIG.tts_format),  # type: ignore[arg-type]
        )
        if voice_enabled
        else None
    )

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
            getattr(resolved_cfg, "sessions_dir", Path("sessions")) if resolved_cfg else Path("sessions")
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
            getattr(resolved_cfg, "sessions_dir", Path("sessions")) if resolved_cfg else Path("sessions")
        )
        configured_name = Path(str(configured_sessions_dir)).name
        if sessions_dir != configured_name:
            return Response(
                status_code=307,
                headers={"Location": f"/journal/{configured_name}/{session_id}/"},
            )

        # Obtain or resume the requested session
        state: WebSessionState | None = app.state.sessions.get(session_id)
        if state is None:
            state = _resume_specific_session(app, session_id)
            if state is None:
                return JSONResponse(
                    {"status": "error", "error": "unknown_session"}, status_code=404
                )

        return _render_session_page(app, state)

    @app.post("/session/{session_id}/upload")  # type: ignore[attr-defined]
    async def upload(session_id: str, request: Request):
        state = app.state.sessions.get(session_id)
        if state is None:
            return JSONResponse(
                {"status": "error", "error": "unknown_session"}, status_code=404
            )

        form = await request.form()
        upload = form.get("audio")
        if upload is None:
            return JSONResponse(
                {"status": "error", "error": "missing_audio"}, status_code=400
            )
        if not isinstance(upload, UploadFile):
            return JSONResponse(
                {"status": "error", "error": "invalid_payload"}, status_code=400
            )

        # Metadata provided by the browser recorder
        mime_val = form.get("mime")
        if isinstance(mime_val, bytes):
            try:
                mime_val = mime_val.decode()
            except Exception:
                mime_val = None
        if not isinstance(mime_val, str):
            mime_val = None
        content_type = upload.content_type or "audio/webm"
        mime = (mime_val or content_type).lower()

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

        # Persist uploaded audio to the active session directory
        session_state = state.manager.state
        if session_state is None:
            return JSONResponse(
                {"status": "error", "error": "inactive_session"}, status_code=409
            )

        target_dir = session_state.audio_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        next_index = session_state.response_index + 1
        extension = _extension_for_mime(mime, upload.filename)
        segment_basename = _build_segment_basename(next_index)
        target_path = target_dir / f"{segment_basename}{extension}"
        while target_path.exists():  # Defensive: avoid accidental overwrite
            next_index += 1
            segment_basename = _build_segment_basename(next_index)
            target_path = target_dir / f"{segment_basename}{extension}"

        blob = await upload.read()
        if not blob:
            return JSONResponse(
                {"status": "error", "error": "empty_audio"}, status_code=400
            )

        target_path.write_bytes(blob)
        duration_seconds = max(duration_ms, 0.0) / 1000.0
        voiced_seconds = max(voiced_ms, 0.0) / 1000.0

        capture = AudioCaptureResult(
            wav_path=target_path,
            mp3_path=None,
            duration_seconds=duration_seconds,
            voiced_seconds=voiced_seconds,
            cancelled=False,
            quit_after=False,
            discarded_short_answer=False,
        )

        log_event(
            "web.upload.received",
            {
                "session_id": session_id,
                "filename": target_path.name,
                "content_type": mime,
                "bytes": len(blob),
                "duration_seconds": round(duration_seconds, 2),
            },
        )

        try:
            exchange = state.manager.process_uploaded_exchange(
                state.current_question,
                capture,
                segment_label=target_path.name,
            )
        except Exception as exc:  # pragma: no cover - runtime path
            _LOGGER.exception("Failed to process uploaded exchange")
            return JSONResponse(
                {
                    "status": "error",
                    "error": "processing_failed",
                    "detail": str(exc),
                },
                status_code=500,
            )

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
            return JSONResponse(
                {
                    "status": "error",
                    "error": "question_failed",
                    "detail": str(exc),
                },
                status_code=502,
            )

        log_event(
            "web.upload.processed",
            {
                "session_id": session_id,
                "segment_label": target_path.name,
                "transcript_chars": len(exchange.transcript),
                "next_question_chars": len(state.current_question or ""),
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
        }
        return JSONResponse(response_payload, status_code=201)

    @app.post("/session/{session_id}/tts")  # type: ignore[attr-defined]
    async def tts(session_id: str, request: Request):
        """Synthesize TTS for a given text and return audio bytes.

        Expects JSON body: {"text": "..."}. Returns audio/* content.
        """

        # Validate session
        state = app.state.sessions.get(session_id)
        if state is None:
            return JSONResponse(
                {"status": "error", "error": "unknown_session"}, status_code=404
            )

        if not getattr(app.state, "voice_enabled", False):
            return JSONResponse(
                {"status": "error", "error": "voice_disabled"}, status_code=400
            )

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
                return JSONResponse(
                    {"status": "error", "error": "missing_text"}, status_code=400
                )

            tts_opts: TTSOptions | None = getattr(app.state, "tts_options", None)
            if tts_opts is None:
                tts_opts = TTSOptions(
                    backend="openai",
                    model=CONFIG.tts_model,
                    voice=CONFIG.tts_voice,
                    audio_format=CONFIG.tts_format,  # type: ignore[arg-type]
                )

            audio_bytes = synthesize_text(text, tts_opts)
            # Map simple content-type from format
            fmt = tts_opts.audio_format
            content_type = (
                "audio/wav"
                if fmt == "wav"
                else (
                    "audio/mpeg"
                    if fmt == "mp3"
                    else (
                        "audio/flac"
                        if fmt == "flac"
                        else (
                            "audio/ogg"
                            if fmt in {"ogg", "opus"}
                            else "audio/aac" if fmt == "aac" else "audio/wave"
                        )
                    )
                )
            )

            headers = {"Cache-Control": "no-store"}
            return Response(
                content=audio_bytes, media_type=content_type, headers=headers
            )
        except TTSError as exc:
            _LOGGER.exception("TTS failed: %s", exc)
            return JSONResponse(
                {"status": "error", "error": "tts_failed", "detail": str(exc)},
                status_code=502,
            )
        except Exception as exc:  # pragma: no cover - generic surfacing
            _LOGGER.exception("TTS endpoint error: %s", exc)
            return JSONResponse(
                {"status": "error", "error": "tts_error", "detail": str(exc)},
                status_code=500,
            )

    @app.post("/session/{session_id}/reveal")  # type: ignore[attr-defined]
    async def reveal(session_id: str):
        """Reveal the session's markdown file in the OS file manager.

        - macOS: uses `open -R`
        - Others: returns a 501 to indicate unsupported platform for now
        """

        state = app.state.sessions.get(session_id)
        if state is None:
            return JSONResponse(
                {"status": "error", "error": "unknown_session"}, status_code=404
            )

        try:
            st = state.manager.state
            if st is None:
                return JSONResponse(
                    {"status": "error", "error": "inactive_session"}, status_code=409
                )

            md_path = st.markdown_path
            if sys.platform == "darwin":
                try:
                    subprocess.run(["open", "-R", str(md_path)], check=False)
                    return JSONResponse({"status": "ok"}, status_code=200)
                except Exception as exc:  # pragma: no cover - runtime path
                    _LOGGER.exception("Reveal failed: %s", exc)
                    return JSONResponse(
                        {
                            "status": "error",
                            "error": "reveal_failed",
                            "detail": str(exc),
                        },
                        status_code=500,
                    )
            else:
                return JSONResponse(
                    {"status": "error", "error": "unsupported_platform"},
                    status_code=501,
                )
        except Exception as exc:  # pragma: no cover - generic surfacing
            _LOGGER.exception("Reveal endpoint error: %s", exc)
            return JSONResponse(
                {"status": "error", "error": "reveal_error", "detail": str(exc)},
                status_code=500,
            )

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


def _render_session_page(app: Any, state: WebSessionState) -> str:
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
    )
    return body


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
    app.state.sessions[state.session_id] = web_state
    return web_state


def _start_session(app: Any) -> WebSessionState:
    """Initialise a new journaling session for the web client."""

    manager = _build_session_manager(app)
    state = manager.start()
    current_question = manager.config.opening_question

    web_state = WebSessionState(manager=manager, current_question=current_question)
    app.state.sessions[state.session_id] = web_state

    log_event(
        "web.session.started",
        {
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
    app.state.sessions[state.session_id] = web_state

    log_event(
        "web.session.resumed",
        {
            "session_id": state.session_id,
            "markdown_path": state.markdown_path.name,
            "audio_dir": str(state.audio_dir),
        },
    )
    return web_state


def _build_segment_basename(index: int) -> str:
    return f"browser-{index:03d}"


def _extension_for_mime(mime: str, filename: str | None) -> str:
    """Infer a file extension from the supplied MIME type/filename."""

    if mime.startswith("audio/webm"):
        return ".webm"
    if mime in {"audio/ogg", "application/ogg"}:
        return ".ogg"
    if mime in {"audio/mpeg", "audio/mp3"}:
        return ".mp3"
    if mime == "audio/wav" or mime == "audio/x-wav":
        return ".wav"
    if filename and "." in filename:
        return "." + filename.rsplit(".", 1)[-1]
    return ".bin"


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
) -> str:
    """Return the base HTML shell; dynamic behaviour handled client-side."""

    session_id = state.session_id
    question = state.current_question
    short_duration = CONFIG.short_answer_duration_seconds
    short_voiced = CONFIG.short_answer_voiced_seconds
    voice_attr = "true" if voice_enabled else "false"
    tts_mime = (
        "audio/wav"
        if tts_format == "wav"
        else (
            "audio/mpeg"
            if tts_format == "mp3"
            else (
                "audio/flac"
                if tts_format == "flac"
                else (
                    "audio/ogg"
                    if tts_format in {"ogg", "opus"}
                    else "audio/aac" if tts_format == "aac" else "audio/wave"
                )
            )
        )
    )

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
        },
        filesystem_loader=template_path.parent,
    )
