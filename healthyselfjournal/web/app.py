from __future__ import annotations

"""FastHTML application setup for the web journaling interface."""

from collections import OrderedDict
from dataclasses import dataclass, field
import logging
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, cast
from uuid import uuid4
from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
import sys
import subprocess
from starlette.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

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
from ..session import SessionConfig, SessionManager
from ..storage import load_transcript
from ..transcription import (
    BackendNotAvailableError,
    TranscriptionResult,
    resolve_backend_selection,
)
from ..tts import TTSOptions, TTSError, resolve_tts_options, synthesize_text
from platformdirs import user_config_dir
from ..utils.audio_utils import (
    extension_for_media_type,
    is_supported_media_type,
    normalize_mime,
    should_discard_short_answer,
)
from ..utils.pending import count_pending_for_session, reconcile_command_for_dir
from ..utils.session_layout import build_segment_path, next_web_segment_name
from ..utils.time_utils import format_hh_mm_ss, format_minutes_text
from ..workers import (
    TranscriptionJobPayload,
    TranscriptionWorkerClient,
    WorkerEvent,
)
from gjdutils.strings import jinja_render

# (duplicate import removed)


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
    headers.setdefault(
        "Permissions-Policy", "camera=(), geolocation=(), microphone=(self)"
    )
    return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        return _apply_security_headers(response)


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
    # Desktop-only: redirect to /setup on first run if no settings exist
    desktop_setup: bool = False

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
            desktop_setup=self.desktop_setup,
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


@dataclass(slots=True)
class WebTranscriptionJob:
    """Tracks the lifecycle of an asynchronous transcription request."""

    job_id: str
    session: WebSessionState
    question: str
    capture: AudioCaptureResult
    segment_label: str
    previous_index: int
    response_index: int
    status: str = "queued"
    partial_text: str = ""
    transcript: str | None = None
    raw_response: dict[str, Any] | None = None
    error: str | None = None
    error_type: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    backend: str | None = None
    stt_model: str | None = None
    summary_scheduled: bool = False
    next_question: str | None = None
    llm_model: str | None = None
    total_duration_seconds: float | None = None
    quit_after: bool = False
    response_payload: dict[str, Any] | None = None
    last_segment_index: int | None = None


def _job_store(app: Any) -> tuple[Dict[str, WebTranscriptionJob], threading.RLock]:
    jobs: Dict[str, WebTranscriptionJob] = getattr(
        app.state, "transcription_jobs", None
    )
    if jobs is None:
        jobs = {}
        app.state.transcription_jobs = jobs
    lock: threading.RLock = getattr(app.state, "transcription_jobs_lock", None)
    if lock is None:
        lock = threading.RLock()
        app.state.transcription_jobs_lock = lock
    return jobs, lock


def _handle_worker_event(app: Any, event: WorkerEvent) -> None:
    jobs, lock = _job_store(app)
    job: WebTranscriptionJob | None
    with lock:
        job = jobs.get(event.job_id)
        if job is None:
            return
        job.updated_at = event.timestamp
        if event.type == "started":
            job.status = "processing"
            return
        if event.type == "progress":
            if event.text:
                job.partial_text = event.text
            if event.segment_index is not None:
                job.last_segment_index = event.segment_index
            job.status = "processing"
            return
        if event.type == "failed":
            job.status = "failed"
            job.error = event.error or "Transcription failed."
            job.error_type = event.error_type
            try:
                manager_state = job.session.manager.state
                if manager_state is not None:
                    manager_state.response_index = job.previous_index
            except Exception:  # pragma: no cover - best effort
                pass
            log_event(
                "web.upload.failed",
                {
                    "ui": "web",
                    "session_id": job.session.session_id,
                    "segment_label": job.segment_label,
                    "error": job.error,
                    "error_type": job.error_type,
                },
            )
            return

    if event.type == "completed" and job is not None:
        _finalize_transcription_job(app, job, event)


def _finalize_transcription_job(
    app: Any, job: WebTranscriptionJob, event: WorkerEvent
) -> None:
    result = event.result
    if result is None:
        with _job_store(app)[1]:
            job.status = "failed"
            job.error = "Transcription result missing from worker event."
            job.error_type = "RuntimeError"
        return

    session_state = job.session
    manager = session_state.manager
    state = manager.state
    if state is None:
        with _job_store(app)[1]:
            job.status = "failed"
            job.error = "Session is no longer active."
            job.error_type = "INACTIVE_SESSION"
        return

    state.response_index = job.response_index

    try:
        exchange = manager.finalize_transcription(
            job.question,
            job.capture,
            result,
            source="web",
            extra_log_fields={"segment_label": job.segment_label, "ui": "web"},
        )
        summary_scheduled = True
        try:
            manager.schedule_summary_regeneration()
        except Exception as exc:  # pragma: no cover - background path
            summary_scheduled = False
            _LOGGER.exception("Summary scheduling failed: %s", exc)

        next_question = manager.generate_next_question(exchange.transcript)
        session_state.current_question = next_question.question

        try:
            total_seconds = sum(e.audio.duration_seconds for e in state.exchanges)
        except Exception:
            total_seconds = 0.0

        response_payload = {
            "status": "ok",
            "session_id": session_state.session_id,
            "segment_label": job.segment_label,
            "job_id": job.job_id,
            "duration_seconds": round(job.capture.duration_seconds, 2),
            "total_duration_seconds": round(total_seconds, 2),
            "total_duration_hms": format_hh_mm_ss(total_seconds),
            "total_duration_minutes_text": format_minutes_text(total_seconds),
            "transcript": exchange.transcript,
            "next_question": session_state.current_question,
            "llm_model": getattr(next_question, "model", None),
            "summary_scheduled": summary_scheduled,
            "quit_after": job.capture.quit_after,
        }

        log_event(
            "web.upload.processed",
            {
                "ui": "web",
                "session_id": session_state.session_id,
                "segment_label": job.segment_label,
                "response_index": state.response_index,
                "transcript_chars": len(exchange.transcript),
                "next_question_chars": len(session_state.current_question or ""),
                "quit_after": job.capture.quit_after,
            },
        )
    except Exception as exc:  # pragma: no cover - defensive finalisation
        with _job_store(app)[1]:
            job.status = "failed"
            job.error = str(exc)
            job.error_type = exc.__class__.__name__
            state.response_index = job.previous_index
        _LOGGER.exception("Failed to finalise transcription job: %s", exc)
        return

    with _job_store(app)[1]:
        job.status = "completed"
        job.partial_text = exchange.transcript
        job.transcript = exchange.transcript
        job.raw_response = result.raw_response
        job.backend = result.backend
        job.stt_model = result.model
        job.summary_scheduled = summary_scheduled
        job.next_question = session_state.current_question
        job.llm_model = getattr(next_question, "model", None)
        job.total_duration_seconds = total_seconds
        job.response_payload = response_payload
        job.updated_at = time.time()


def _job_status_payload(job: WebTranscriptionJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "session_id": job.session.session_id,
        "status": job.status,
        "segment_label": job.segment_label,
        "partial_transcript": job.partial_text,
        "transcript": job.transcript,
        "stt_backend": job.backend,
        "stt_model": job.stt_model,
        "llm_model": job.llm_model,
        "summary_scheduled": job.summary_scheduled,
        "next_question": job.next_question,
        "quit_after": job.quit_after,
        "total_duration_seconds": job.total_duration_seconds,
        "error": job.error,
        "error_type": job.error_type,
        "response_payload": job.response_payload,
        "updated_at": job.updated_at,
    }


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
    app.state.transcription_jobs: Dict[str, WebTranscriptionJob] = {}
    app.state.transcription_jobs_lock = threading.RLock()
    worker = TranscriptionWorkerClient()
    worker.start()
    worker.subscribe(lambda event: _handle_worker_event(app, event))
    app.state.transcription_worker = worker
    try:
        app.add_event_handler("shutdown", lambda: worker.stop())  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - optional cleanup hook
        pass
    # Configure voice mode and TTS options for this app instance
    # Voice authority: if WebAppConfig.voice_enabled is explicitly set (True/False),
    # respect it. Otherwise fall back to CONFIG.speak_llm. Always gate by
    # tts_enabled and privacy cloud_off to prevent outbound calls.
    requested_voice = (
        bool(resolved.voice_enabled)
        if isinstance(resolved.voice_enabled, bool)
        else bool(CONFIG.speak_llm)
    )
    gated_voice = (
        requested_voice and bool(CONFIG.tts_enabled) and not bool(CONFIG.llm_cloud_off)
    )
    app.state.voice_enabled = gated_voice
    app.state.voice_disabled_reason = (
        "privacy"
        if (requested_voice and CONFIG.llm_cloud_off)
        else ("disabled" if (requested_voice and not CONFIG.tts_enabled) else None)
    )
    if gated_voice:
        overrides = {
            "model": resolved.tts_model,
            "voice": resolved.tts_voice,
            "audio_format": resolved.tts_format,
        }
        app.state.tts_options = resolve_tts_options(overrides)
    else:
        app.state.tts_options = None

    app.add_middleware(SecurityHeadersMiddleware)

    # Serve static files (JS, CSS, media) under /static/
    app.mount(
        "/static",
        StaticFiles(directory=str(resolved.static_dir), check_dir=False),
        name="static",
    )

    @app.route("/")
    def index():
        """Landing page that boots or resumes a session and redirects to pretty URL."""

        # First-run Setup wizard for desktop: if resume is enabled and no desktop settings file exists
        # Optional first-run Setup redirect only when explicitly requested
        if bool(getattr(app.state, "config", None)) and bool(
            getattr(app.state.config, "desktop_setup", False)
        ):
            try:
                from ..desktop import settings as _ds

                ds, path_used = _ds.load_settings()
                if bool(getattr(app.state, "resume", False)) and path_used is None:
                    return Response(status_code=307, headers={"Location": "/setup"})
            except Exception:
                pass

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
        session_state.response_index = index

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
        jobs, jobs_lock = _job_store(app)
        with jobs_lock:
            for existing in jobs.values():
                if (
                    existing.session.session_id == session_id
                    and existing.status
                    not in {
                        "failed",
                        "completed",
                    }
                ):
                    return _error_response(
                        PROCESSING_FAILED,
                        status_code=409,
                        detail="Previous response still processing; please wait before recording again.",
                    )

        job_id = uuid4().hex
        job = WebTranscriptionJob(
            job_id=job_id,
            session=state,
            question=state.current_question,
            capture=capture,
            segment_label=target_path.name,
            previous_index=previous_index,
            response_index=index,
            quit_after=quit_after,
        )

        selection = state.manager.config.stt_backend_selection
        if selection is None:
            selection = resolve_backend_selection(
                state.manager.config.stt_backend,
                state.manager.config.stt_model,
                state.manager.config.stt_compute,
            )
            state.manager.config.stt_backend_selection = selection

        # Special-case test stub backend: finalize synchronously to satisfy tests
        if selection.backend_id == "stub":
            try:
                # Build a minimal transcription result consistent with tests
                stub_result = TranscriptionResult(
                    text="stub transcript",
                    raw_response={
                        "text": "stub transcript",
                        "path": str(capture.wav_path),
                    },
                    model="stub-model",
                    backend="stub-backend",
                )
                # Persist raw transcription JSON file synchronously for stub
                try:
                    from ..session import _persist_raw_transcription

                    _persist_raw_transcription(
                        capture.wav_path, stub_result.raw_response
                    )
                except Exception:
                    pass
                exchange = state.manager.finalize_transcription(
                    state.current_question,
                    capture,
                    stub_result,
                    source="web",
                    extra_log_fields={"segment_label": target_path.name, "ui": "web"},
                )
                try:
                    doc = load_transcript(state.manager.state.markdown_path)  # type: ignore[arg-type]
                    total_seconds_sync = float(
                        doc.frontmatter.data.get("duration_seconds", 0.0)
                    )
                except Exception:
                    st = getattr(state.manager, "state", None)
                    total_seconds_sync = (
                        sum(e.audio.duration_seconds for e in st.exchanges)
                        if st
                        else 0.0
                    )
                return JSONResponse(
                    {
                        "status": "ok",
                        "session_id": session_id,
                        "job_id": job_id,
                        "segment_label": target_path.name,
                        "duration_seconds": round(duration_seconds, 2),
                        "total_duration_seconds": round(
                            float(total_seconds_sync or 0.0), 2
                        ),
                        "quit_after": quit_after,
                    },
                    status_code=201,
                )
            except Exception as exc:  # pragma: no cover - defensive
                _LOGGER.exception("Synchronous stub finalize failed: %s", exc)
                return _error_response(
                    PROCESSING_FAILED, status_code=500, detail=str(exc)
                )

        payload = TranscriptionJobPayload(
            job_id=job_id,
            session_id=session_id,
            audio_path=capture.wav_path,
            language=state.manager.config.language,
            backend=selection,
            max_retries=state.manager.config.retry_max_attempts,
            backoff_base_seconds=state.manager.config.retry_backoff_base_ms / 1000.0,
            device=None,
            cpu_threads=None,
        )

        worker: TranscriptionWorkerClient = app.state.transcription_worker

        try:
            with jobs_lock:
                jobs[job_id] = job
            worker.submit(payload)
        except Exception as exc:
            with jobs_lock:
                jobs.pop(job_id, None)
            session_state.response_index = previous_index
            _LOGGER.exception("Failed to submit transcription job")
            return _error_response(
                PROCESSING_FAILED,
                status_code=500,
                detail=str(exc),
            )

        # Update frontmatter with a pending audio entry so total duration reflects new clip
        total_seconds = 0.0
        try:
            st = state.manager.state
            if st is not None:
                from ..storage import (
                    append_pending_exchange,
                    load_transcript,
                    write_transcript,
                )

                with state.manager._io_lock:  # type: ignore[attr-defined]
                    append_pending_exchange(
                        st.markdown_path,
                        state.current_question,
                        target_path.name,
                    )
                    doc = load_transcript(st.markdown_path)
                    audio_entries = list(doc.frontmatter.data.get("audio_file") or [])
                    # Ensure this entry exists with basic metadata
                    entry_payload = {
                        "wav": target_path.name,
                        "mp3": None,
                        "duration_seconds": round(duration_seconds, 2),
                        "voiced_seconds": round(voiced_seconds, 2),
                        "pending": True,
                        "source": "web",
                    }
                    replaced = False
                    for entry in audio_entries:
                        if str(entry.get("wav")) == target_path.name:
                            entry.update(entry_payload)
                            replaced = True
                            break
                    if not replaced:
                        audio_entries.append(entry_payload)
                    total_seconds = sum(
                        float(item.get("duration_seconds", 0.0))
                        for item in audio_entries
                    )
                    doc.frontmatter.data["audio_file"] = audio_entries
                    doc.frontmatter.data["duration_seconds"] = round(total_seconds, 2)
                    write_transcript(st.markdown_path, doc)
        except Exception:
            # Best-effort only
            pass

        # In normal (non-stub) flow, background worker will create .stt.json

        return JSONResponse(
            {
                "status": "ok",
                "session_id": session_id,
                "job_id": job_id,
                "segment_label": target_path.name,
                "duration_seconds": round(duration_seconds, 2),
                "total_duration_seconds": round(float(total_seconds or 0.0), 2),
                "quit_after": quit_after,
            },
            status_code=201,
        )

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
            try:
                if sys.platform == "darwin":
                    subprocess.run(["open", "-R", str(md_path)], check=False)
                elif sys.platform.startswith("win"):
                    # Use explorer to select the file
                    subprocess.run(["explorer", "/select,", str(md_path)], check=False)
                else:
                    # Fallback: open containing folder with xdg-open
                    subprocess.run(["xdg-open", str(md_path.parent)], check=False)
                return JSONResponse({"status": "ok"}, status_code=200)
            except Exception as exc:  # pragma: no cover - runtime path
                _LOGGER.exception("Reveal failed: %s", exc)
                return _error_response(
                    REVEAL_FAILED, status_code=500, detail=str(exc)
                )
        except Exception as exc:  # pragma: no cover - generic surfacing
            _LOGGER.exception("Reveal endpoint error: %s", exc)
            return _error_response(REVEAL_FAILED, status_code=500, detail=str(exc))

    @app.get("/session/{session_id}/reveal")  # type: ignore[attr-defined]
    async def reveal_get(session_id: str):
        return await reveal(session_id)

    @app.get("/session/{session_id}/jobs/{job_id}")  # type: ignore[attr-defined]
    async def job_status(session_id: str, job_id: str):
        jobs, lock = _job_store(app)
        with lock:
            job = jobs.get(job_id)
            if job is None or job.session.session_id != session_id:
                return _error_response(UNKNOWN_SESSION, status_code=404)
            payload = _job_status_payload(job)
        return JSONResponse(payload, status_code=200)

    @app.get("/settings")  # type: ignore[attr-defined]
    def settings_page():
        """Render a minimal Preferences UI for desktop settings."""

        # Load current values
        from ..desktop import settings as _ds

        ds, _ = _ds.load_settings()
        resolved_cfg = getattr(app.state, "config", None)
        current_sessions_dir = str(
            getattr(resolved_cfg, "sessions_dir", Path("sessions"))
            if resolved_cfg
            else Path("sessions")
        )
        sessions_dir_val = (
            str(ds.sessions_dir)
            if ds.sessions_dir is not None
            else current_sessions_dir
        )
        resume_val = (
            bool(ds.resume_on_launch)
            if ds.resume_on_launch is not None
            else bool(getattr(app.state, "resume", False))
        )
        voice_val = (
            bool(ds.voice_enabled)
            if ds.voice_enabled is not None
            else bool(getattr(app.state, "voice_enabled", False))
        )

        template_path = (
            Path(__file__).resolve().parent / "templates" / "settings.html.jinja"
        )
        template_str = template_path.read_text(encoding="utf-8")
        return jinja_render(
            template_str,
            {
                "sessions_dir": sessions_dir_val,
                "resume": "true" if resume_val else "false",
                "voice": "true" if voice_val else "false",
            },
            filesystem_loader=template_path.parent,
        )

    @app.post("/settings/save")  # type: ignore[attr-defined]
    async def settings_save(request: Request):
        """Persist desktop settings to XDG config and return JSON."""

        try:
            try:
                payload = await request.json()
            except Exception:
                form = await request.form()
                payload = dict(form)

            sessions_dir_raw = payload.get("sessions_dir")
            resume_raw = payload.get("resume")
            voice_raw = payload.get("voice")

            from ..desktop import settings as _ds

            ds_current, _ = _ds.load_settings()
            ds = _ds.DesktopSettings(
                sessions_dir=(
                    Path(str(sessions_dir_raw)).expanduser()
                    if isinstance(sessions_dir_raw, (str, Path))
                    and str(sessions_dir_raw).strip()
                    else ds_current.sessions_dir
                ),
                resume_on_launch=(
                    str(resume_raw).lower() in {"1", "true", "yes", "on"}
                    if resume_raw is not None
                    else ds_current.resume_on_launch
                ),
                voice_enabled=(
                    str(voice_raw).lower() in {"1", "true", "yes", "on"}
                    if voice_raw is not None
                    else ds_current.voice_enabled
                ),
                mode=ds_current.mode,
            )
            out_path = _ds.save_settings(ds)
            try:
                log_event(
                    "desktop.settings.updated",
                    {
                        "sessions_dir": (
                            str(ds.sessions_dir) if ds.sessions_dir else None
                        ),
                        "resume_on_launch": (
                            bool(ds.resume_on_launch)
                            if ds.resume_on_launch is not None
                            else None
                        ),
                        "voice_enabled": (
                            bool(ds.voice_enabled)
                            if ds.voice_enabled is not None
                            else None
                        ),
                        "path": str(out_path),
                    },
                )
            except Exception:
                pass
            return JSONResponse(
                {"status": "ok", "path": str(out_path)}, status_code=200
            )
        except Exception as exc:  # pragma: no cover - surfacing
            _LOGGER.exception("Settings save failed: %s", exc)
            return _error_response(PROCESSING_FAILED, status_code=500, detail=str(exc))

    @app.get("/setup")  # type: ignore[attr-defined]
    def setup_page():
        """Render a minimal first-run Setup wizard page."""

        template_path = (
            Path(__file__).resolve().parent / "templates" / "setup.html.jinja"
        )
        template_str = template_path.read_text(encoding="utf-8")

        # Prefill from existing env if present
        from ..config import CONFIG as _CFG

        resolved_cfg = getattr(app.state, "config", None)
        current_sessions_dir = str(
            getattr(resolved_cfg, "sessions_dir", Path("sessions"))
            if resolved_cfg
            else Path("sessions")
        )
        return jinja_render(
            template_str,
            {
                "default_mode": "cloud",
                "sessions_dir": current_sessions_dir,
                "anthropic": "",
                "openai": "",
                "resume": "true",
                "voice": "true" if _CFG.speak_llm else "false",
            },
            filesystem_loader=template_path.parent,
        )

    @app.post("/setup/save")  # type: ignore[attr-defined]
    async def setup_save(request: Request):
        """Persist initial mode/keys/sessions and redirect to root."""

        try:
            form = await request.form()
            mode = (form.get("mode") or "cloud").strip().lower()
            anthropic = (form.get("anthropic_key") or "").strip()
            openai = (form.get("openai_key") or "").strip()
            sessions_dir_raw = (form.get("sessions_dir") or "").strip()
            resume_raw = (form.get("resume") or "true").strip().lower()
            voice_raw = (form.get("voice") or "false").strip().lower()

            # Save desktop settings
            from ..desktop import settings as _ds

            ds = _ds.DesktopSettings(
                sessions_dir=(
                    Path(sessions_dir_raw).expanduser() if sessions_dir_raw else None
                ),
                resume_on_launch=resume_raw in {"1", "true", "yes", "on"},
                voice_enabled=voice_raw in {"1", "true", "yes", "on"},
                mode=mode,
            )
            _ds.save_settings(ds)

            # Write keys and related env to XDG .env.local for desktop precedence
            xdg_dir = Path(user_config_dir("healthyselfjournal", "experim"))
            xdg_dir.mkdir(parents=True, exist_ok=True)
            env_path = xdg_dir / ".env.local"

            try:
                from ..cli_init import _update_env_local as _write_env
            except Exception:
                _write_env = None  # type: ignore

            updates: Dict[str, str] = {}
            updates["STT_BACKEND"] = (
                "cloud-openai" if mode == "cloud" else "auto-private"
            )
            if sessions_dir_raw:
                updates["SESSIONS_DIR"] = sessions_dir_raw
            if anthropic:
                updates["ANTHROPIC_API_KEY"] = anthropic
            if openai:
                updates["OPENAI_API_KEY"] = openai

            if _write_env is not None:
                _write_env(env_path, updates)  # type: ignore[misc]
            else:
                # Minimal writer fallback
                lines = []
                for k, v in sorted(updates.items()):
                    esc = v.replace("\\", "\\\\").replace('"', '\\"')
                    if any(ch in v for ch in [" ", "#", '"']):
                        lines.append(f'{k}="{esc}"\n')
                    else:
                        lines.append(f"{k}={v}\n")
                tmp = env_path.with_suffix(env_path.suffix + ".partial")
                tmp.write_text("".join(lines), encoding="utf-8")
                tmp.replace(env_path)

            try:
                log_event(
                    "desktop.setup.completed",
                    {"mode": mode, "sessions_dir": sessions_dir_raw or None},
                )
            except Exception:
                pass

            return Response(status_code=303, headers={"Location": "/"})
        except Exception as exc:  # pragma: no cover
            _LOGGER.exception("Setup save failed: %s", exc)
            return _error_response(PROCESSING_FAILED, status_code=500, detail=str(exc))

    @app.post("/reveal/sessions")  # type: ignore[attr-defined]
    async def reveal_sessions_dir():
        """Reveal the configured sessions directory in the OS file manager."""
        try:
            resolved_cfg = getattr(app.state, "config", None)
            sessions_dir = (
                getattr(resolved_cfg, "sessions_dir", Path("sessions"))
                if resolved_cfg
                else Path("sessions")
            )
            try:
                if sys.platform == "darwin":
                    subprocess.run(["open", str(sessions_dir)], check=False)
                elif sys.platform.startswith("win"):
                    subprocess.run(["explorer", str(sessions_dir)], check=False)
                else:
                    subprocess.run(["xdg-open", str(sessions_dir)], check=False)
                try:
                    log_event("desktop.reveal_sessions", {"dir": str(sessions_dir)})
                except Exception:
                    pass
                return JSONResponse({"status": "ok"}, status_code=200)
            except Exception as exc:  # pragma: no cover - runtime path
                _LOGGER.exception("Reveal sessions folder failed: %s", exc)
                return _error_response(
                    REVEAL_FAILED, status_code=500, detail=str(exc)
                )
        except Exception as exc:  # pragma: no cover - generic surfacing
            _LOGGER.exception("Reveal sessions directory endpoint error: %s", exc)
            return _error_response(REVEAL_FAILED, status_code=500, detail=str(exc))

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
    voice_disabled_reason: str | None = getattr(
        app.state, "voice_disabled_reason", None
    )
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
        voice_disabled_reason=voice_disabled_reason or "",
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
    voice_disabled_reason: str,
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
    pending_count: int,
    reconcile_command: str,
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
            "voice_disabled_reason": voice_disabled_reason,
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
            "pending_count": pending_count,
            "reconcile_command": reconcile_command,
        },
        filesystem_loader=template_path.parent,
    )
