from __future__ import annotations

"""Session orchestration and state management."""

from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Sequence
from concurrent.futures import ThreadPoolExecutor

from . import __version__
from .audio import AudioCaptureResult, record_response
from .config import CONFIG
from .history import HistoricalSummary, load_recent_summaries
from .llm import (
    QuestionRequest,
    QuestionResponse,
    SummaryRequest,
    generate_followup_question,
    generate_summary,
)
from . import llm as llm_module
from .storage import (
    Frontmatter,
    TranscriptDocument,
    append_exchange_body,
    append_pending_exchange,
    load_transcript,
    write_transcript,
)
from .transcription import (
    BackendNotAvailableError,
    BackendSelection,
    TranscriptionBackend,
    TranscriptionResult,
    apply_transcript_formatting,
    create_transcription_backend,
    format_transcript_sentences,
)
from .events import log_event
from .utils.audio_utils import maybe_delete_wav_when_safe
from .utils.pending import remove_error_sentinel, write_error_sentinel
from .utils.session_layout import (
    build_segment_path,
    next_cli_segment_name,
)
from .utils.session_utils import get_max_recorded_index
from .utils.time_utils import format_hh_mm_ss, format_mm_ss
from rich.text import Text

if TYPE_CHECKING:
    from rich.console import Console


_LOGGER = logging.getLogger(__name__)


@dataclass
class Exchange:
    """Single question/answer pair."""

    question: str
    transcript: str
    audio: AudioCaptureResult
    transcription: TranscriptionResult
    followup_question: QuestionResponse | None = None
    discarded_short_answer: bool = False


@dataclass
class SessionState:
    """Mutable state stored for the duration of a journaling session."""

    session_id: str
    markdown_path: Path
    audio_dir: Path
    exchanges: List[Exchange] = field(default_factory=list)
    quit_requested: bool = False
    response_index: int = 0
    recent_history: List[HistoricalSummary] = field(default_factory=list)
    resumed: bool = False
    # UI feedback flags for last capture disposition
    last_cancelled: bool = False
    last_discarded_short: bool = False


@dataclass
class SessionConfig:
    base_dir: Path
    llm_model: str
    stt_model: str
    opening_question: str = CONFIG.opening_question
    stt_backend: str = CONFIG.stt_backend
    stt_compute: str | None = CONFIG.stt_compute
    max_history_tokens: int = CONFIG.max_history_tokens
    recent_summaries_limit: int = CONFIG.max_recent_summaries
    app_version: str = __version__
    retry_max_attempts: int = CONFIG.retry_max_attempts
    retry_backoff_base_ms: int = CONFIG.retry_backoff_base_ms
    ffmpeg_path: str | None = CONFIG.ffmpeg_path
    language: str = "en"
    stt_formatting: str = CONFIG.stt_formatting
    stt_backend_requested: str | None = None
    stt_model_requested: str | None = None
    stt_compute_requested: str | None = None
    stt_auto_private_reason: str | None = None
    stt_backend_selection: BackendSelection | None = None
    stt_warnings: List[str] = field(default_factory=list)
    llm_questions_debug: bool = False


class PendingTranscriptionError(RuntimeError):
    """Raised when audio capture succeeds but STT must be deferred."""

    def __init__(
        self,
        *,
        segment_label: str,
        audio_path: Path,
        source: str,
        error: Exception,
    ) -> None:
        super().__init__(str(error))
        self.segment_label = segment_label
        self.audio_path = audio_path
        self.source = source
        self.error = error
        self.error_type = error.__class__.__name__


class SessionManager:
    """High-level API consumed by the CLI layer."""

    def __init__(self, config: SessionConfig) -> None:
        self.config = config
        self.state: SessionState | None = None
        if config.stt_backend_selection is not None:
            self._backend_selection = config.stt_backend_selection
        else:
            self._backend_selection = BackendSelection(
                backend_id=config.stt_backend,
                model=config.stt_model,
                compute=config.stt_compute,
                requested_backend=config.stt_backend_requested,
                requested_model=config.stt_model_requested,
                requested_compute=config.stt_compute_requested,
                reason=config.stt_auto_private_reason,
                warnings=list(config.stt_warnings or []),
            )
        # Normalise resolved values back onto config for downstream consumers.
        self.config.stt_backend = self._backend_selection.backend_id
        self.config.stt_model = self._backend_selection.model
        self.config.stt_compute = self._backend_selection.compute
        self.config.stt_backend_selection = self._backend_selection
        self.config.stt_warnings = list(self._backend_selection.warnings)
        self._transcription_backend: TranscriptionBackend | None = None
        # Serialize all transcript writes (frontmatter/body/summary) within process
        self._io_lock = threading.Lock()
        # Single worker to run summary generation tasks in background
        self._summary_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="summary"
        )
        self._summary_shutdown = False

    def start(self) -> SessionState:
        base_dir = self.config.base_dir
        base_dir.mkdir(parents=True, exist_ok=True)

        session_id = datetime.now().strftime("%y%m%d_%H%M")
        markdown_path = base_dir / f"{session_id}.md"

        recent = load_recent_summaries(
            base_dir,
            current_filename=markdown_path.name,
            limit=self.config.recent_summaries_limit,
            max_estimated_tokens=self.config.max_history_tokens,
        )

        frontmatter_data = {
            "created_at": datetime.now().isoformat(),
            "transcript_file": markdown_path.name,
            "recent_summary_refs": [item.filename for item in recent],
            "model_llm": self.config.llm_model,
            "model_stt": self._backend_selection.model,
            "stt_backend": self._backend_selection.backend_id,
            "stt_compute": self._backend_selection.compute,
            "stt_formatting": self.config.stt_formatting,
            "stt_requested_backend": self.config.stt_backend_requested,
            "stt_requested_model": self.config.stt_model_requested,
            "stt_requested_compute": self.config.stt_compute_requested,
            "stt_auto_private_reason": self.config.stt_auto_private_reason,
            "stt_warnings": list(self._backend_selection.warnings),
            "app_version": self.config.app_version,
            "duration_seconds": 0.0,
            "audio_file": [],
            "summary": "",
        }
        frontmatter = Frontmatter(data=frontmatter_data)
        frontmatter.ensure_keys()
        write_transcript(
            markdown_path, TranscriptDocument(frontmatter=frontmatter, body="")
        )

        self.state = SessionState(
            session_id=session_id,
            markdown_path=markdown_path,
            audio_dir=base_dir / session_id,
            recent_history=recent,
        )
        # Ensure per-session assets directory exists
        self.state.audio_dir.mkdir(parents=True, exist_ok=True)
        log_event(
            "session.start",
            {
                "session_id": session_id,
                "transcript_file": markdown_path.name,
                "audio_dir": self.state.audio_dir,
                "llm_model": self.config.llm_model,
                "stt_backend": self._backend_selection.backend_id,
                "stt_model": self._backend_selection.model,
                "stt_compute": self._backend_selection.compute,
                "stt_formatting": self.config.stt_formatting,
                "stt_auto_reason": self.config.stt_auto_private_reason,
                "language": self.config.language,
                "recent_summaries_count": len(recent),
                "stt_warnings": self._backend_selection.warnings,
            },
        )
        for warning in self._backend_selection.warnings:
            _LOGGER.warning("STT backend warning: %s", warning)
        return self.state

    def _get_transcription_backend(self) -> TranscriptionBackend:
        if self._transcription_backend is None:
            try:
                self._transcription_backend = create_transcription_backend(
                    self._backend_selection,
                    max_retries=self.config.retry_max_attempts,
                    backoff_base_seconds=self.config.retry_backoff_base_ms / 1000,
                )
            except BackendNotAvailableError as exc:
                raise RuntimeError(
                    f"Unable to initialise transcription backend '{self._backend_selection.backend_id}': {exc}"
                ) from exc
        return self._transcription_backend

    def resume(self, markdown_path: Path) -> SessionState:
        """Resume an existing session from a transcript markdown file.

        - Preserves existing transcript and frontmatter
        - Sets response index to next available segment number
        - Loads recent summaries for LLM context
        """
        base_dir = self.config.base_dir
        base_dir.mkdir(parents=True, exist_ok=True)

        session_id = markdown_path.stem

        recent = load_recent_summaries(
            base_dir,
            current_filename=markdown_path.name,
            limit=self.config.recent_summaries_limit,
            max_estimated_tokens=self.config.max_history_tokens,
        )

        self.state = SessionState(
            session_id=session_id,
            markdown_path=markdown_path,
            audio_dir=base_dir / session_id,
            recent_history=recent,
            resumed=True,
        )
        self.state.audio_dir.mkdir(parents=True, exist_ok=True)

        # Determine next response index from existing frontmatter or files
        try:
            doc = load_transcript(markdown_path)
            audio_entries = doc.frontmatter.data.get("audio_file") or []
            indices: list[int] = []
            for entry in audio_entries:
                try:
                    wav_name = str(entry.get("wav", ""))
                    stem = Path(wav_name).name.split(".", 1)[0]
                    parts = stem.split("_")
                    if len(parts) >= 2 and parts[-1].isdigit():
                        indices.append(int(parts[-1]))
                except Exception:
                    continue

            fs_index = get_max_recorded_index(self.state.audio_dir, session_id)
            if fs_index:
                indices.append(fs_index)

            self.state.response_index = max(indices) if indices else 0
        except Exception:
            # Defensive: on any error, continue from zero
            self.state.response_index = 0

        log_event(
            "session.resume",
            {
                "session_id": session_id,
                "transcript_file": markdown_path.name,
                "audio_dir": self.state.audio_dir,
                "existing_responses": self.state.response_index,
            },
        )
        return self.state

    def record_exchange(self, question: str, console: "Console") -> Exchange | None:
        if self.state is None:
            raise RuntimeError("Session has not been started")

        previous_index = self.state.response_index
        next_index_hint = previous_index + 1
        index, segment_basename = next_cli_segment_name(
            self.state.session_id,
            self.state.audio_dir,
            start_index=next_index_hint,
        )
        target_wav = build_segment_path(self.state.audio_dir, segment_basename, ".wav")
        self.state.response_index = index

        capture = record_response(
            self.state.audio_dir,
            segment_basename,
            console=console,
            sample_rate=16_000,
            ffmpeg_path=self.config.ffmpeg_path,
            print_saved_message=False,
            target_wav_path=target_wav,
        )

        if capture.cancelled:
            # Mark disposition for clearer UI messaging
            self.state.last_cancelled = True
            self.state.last_discarded_short = False
            self.state.response_index = previous_index
            return None

        if capture.discarded_short_answer:
            # Treat as no exchange; do not transcribe, do not persist any body
            self.state.response_index = previous_index
            self.state.last_cancelled = False
            self.state.last_discarded_short = True
            log_event(
                "session.exchange.discarded_short",
                {
                    "session_id": self.state.session_id,
                    "response_index": self.state.response_index + 1,
                    "duration_seconds": round(capture.duration_seconds, 2),
                    "voiced_seconds": round(capture.voiced_seconds, 2),
                },
            )
            # If user pressed Q while discarding, mark quit flag on state and let caller handle
            self.state.quit_requested = capture.quit_after
            return None

        # Print a combined saved message with segment and session total durations
        try:
            doc = load_transcript(self.state.markdown_path)
            prior_total = float(doc.frontmatter.data.get("duration_seconds", 0.0))
        except Exception:
            prior_total = 0.0

        seg_formatted = format_hh_mm_ss(capture.duration_seconds)
        total_formatted = format_hh_mm_ss(prior_total + capture.duration_seconds)
        console.print(
            Text(
                f"Saved WAV → {capture.wav_path.name} ({seg_formatted}; total {total_formatted})",
                style="green",
            )
        )

        return self._transcribe_and_store(
            question,
            capture,
            source="cli",
            extra_log_fields={"ui": "cli"},
        )

    def process_uploaded_exchange(
        self,
        question: str,
        capture: AudioCaptureResult,
        *,
        segment_label: str | None = None,
    ) -> Exchange:
        """Persist an exchange sourced from an uploaded audio clip.

        The ``capture`` argument should reference audio already written to disk
        within the active session's media directory.
        """

        if self.state is None:
            raise RuntimeError("Session has not been started")

        self.state.response_index += 1
        extra: dict[str, Any] | None = None
        if segment_label:
            extra = {"segment_label": segment_label, "ui": "web"}
        return self._transcribe_and_store(
            question,
            capture,
            source="web",
            extra_log_fields=extra,
        )

    def _transcribe_and_store(
        self,
        question: str,
        capture: AudioCaptureResult,
        *,
        source: str,
        extra_log_fields: dict[str, Any] | None = None,
    ) -> Exchange:
        if self.state is None:
            raise RuntimeError("Session has not been started")

        backend = self._get_transcription_backend()
        try:
            transcription = backend.transcribe(
                capture.wav_path,
                language=self.config.language,
            )
        except Exception as exc:
            self._handle_transcription_failure(
                question=question,
                capture=capture,
                source=source,
                error=exc,
                extra_log_fields=extra_log_fields,
            )
            raise PendingTranscriptionError(
                segment_label=capture.wav_path.name,
                audio_path=capture.wav_path,
                source=source,
                error=exc,
            ) from exc

        return self.finalize_transcription(
            question,
            capture,
            transcription,
            source=source,
            extra_log_fields=extra_log_fields,
        )

    def finalize_transcription(
        self,
        question: str,
        capture: AudioCaptureResult,
        transcription: TranscriptionResult,
        *,
        source: str,
        extra_log_fields: dict[str, Any] | None = None,
    ) -> Exchange:
        """Persist exchange artifacts once transcription already completed."""

        if self.state is None:
            raise RuntimeError("Session has not been started")

        _persist_raw_transcription(capture.wav_path, transcription.raw_response)

        try:
            formatted_text = apply_transcript_formatting(
                transcription.text, self.config.stt_formatting
            )
        except ValueError:
            _LOGGER.warning(
                "Unsupported formatting mode '%s'; falling back to sentence splits",
                self.config.stt_formatting,
            )
            formatted_text = format_transcript_sentences(transcription.text)

        # Serialize body append to avoid racing with background summary writes
        with self._io_lock:
            append_exchange_body(
                self.state.markdown_path,
                question,
                formatted_text,
            )

        exchange = Exchange(
            question=question,
            transcript=formatted_text,
            audio=capture,
            transcription=transcription,
            discarded_short_answer=False,
        )
        self.state.exchanges.append(exchange)
        self.state.quit_requested = capture.quit_after
        # Successful capture resets disposition flags
        self.state.last_cancelled = False
        self.state.last_discarded_short = False

        self._update_frontmatter_after_exchange()

        log_payload = {
            "session_id": self.state.session_id,
            "response_index": self.state.response_index,
            "wav": exchange.audio.wav_path.name,
            "mp3": (exchange.audio.mp3_path.name if exchange.audio.mp3_path else None),
            "duration_seconds": round(exchange.audio.duration_seconds, 2),
            "stt_model": exchange.transcription.model,
            "stt_backend": exchange.transcription.backend,
            "quit_after": exchange.audio.quit_after,
            "cancelled": exchange.audio.cancelled,
            "source": source,
            "ui": source,
        }
        if extra_log_fields:
            log_payload.update(extra_log_fields)

        log_event("session.exchange.recorded", log_payload)
        return exchange

    def _handle_transcription_failure(
        self,
        *,
        question: str,
        capture: AudioCaptureResult,
        source: str,
        error: Exception,
        extra_log_fields: dict[str, Any] | None = None,
    ) -> None:
        if self.state is None:
            return

        segment_label = capture.wav_path.name

        with self._io_lock:
            append_pending_exchange(
                self.state.markdown_path,
                question,
                segment_label,
            )

            doc = load_transcript(self.state.markdown_path)
            audio_entries = list(doc.frontmatter.data.get("audio_file") or [])
            entry_payload = {
                "wav": segment_label,
                "mp3": (capture.mp3_path.name if capture.mp3_path else None),
                "duration_seconds": round(capture.duration_seconds, 2),
                "voiced_seconds": round(capture.voiced_seconds, 2),
                "pending": True,
                "pending_reason": error.__class__.__name__,
                "source": source,
            }

            replaced = False
            for entry in audio_entries:
                if str(entry.get("wav")) == segment_label:
                    entry.update(entry_payload)
                    replaced = True
                    break
            if not replaced:
                audio_entries.append(entry_payload)

            total_duration = sum(
                float(item.get("duration_seconds", 0.0)) for item in audio_entries
            )
            doc.frontmatter.data["audio_file"] = audio_entries
            doc.frontmatter.data["duration_seconds"] = round(total_duration, 2)
            write_transcript(self.state.markdown_path, doc)

        write_error_sentinel(capture.wav_path, error)

        payload = {
            "session_id": self.state.session_id,
            "response_index": self.state.response_index,
            "wav": segment_label,
            "mp3": capture.mp3_path.name if capture.mp3_path else None,
            "duration_seconds": round(capture.duration_seconds, 2),
            "voiced_seconds": round(capture.voiced_seconds, 2),
            "error_type": error.__class__.__name__,
            "error": str(error),
            "source": source,
            "ui": source,
        }
        if extra_log_fields:
            payload.update(extra_log_fields)

        log_event("session.exchange.pending", payload)

        self.state.quit_requested = capture.quit_after
        self.state.last_cancelled = False
        self.state.last_discarded_short = False

    def generate_next_question(self, transcript: str) -> QuestionResponse:
        if self.state is None:
            raise RuntimeError("Session has not been started")
        request = self._build_question_request(transcript)
        response = generate_followup_question(request)
        if self.state.exchanges:
            self.state.exchanges[-1].followup_question = response
        log_event(
            "session.next_question.generated",
            {
                "session_id": self.state.session_id if self.state else None,
                "model": response.model,
            },
        )
        return response

    def generate_next_question_streaming(
        self, transcript: str, on_delta: callable
    ) -> QuestionResponse:
        if self.state is None:
            raise RuntimeError("Session has not been started")
        request = self._build_question_request(transcript)
        response = llm_module.stream_followup_question(request, on_delta)
        if self.state.exchanges:
            self.state.exchanges[-1].followup_question = response
        log_event(
            "session.next_question.generated",
            {
                "session_id": self.state.session_id if self.state else None,
                "model": response.model,
            },
        )
        return response

    def _build_question_request(self, transcript: str) -> QuestionRequest:
        """Build a QuestionRequest with shared logic across generation modes.

        - Computes recent summaries text list
        - Derives total conversation duration so far as mm:ss
        - Uses configured LLM model and language
        """
        assert self.state is not None
        history_text = [item.summary for item in self.state.recent_history]
        try:
            doc = load_transcript(self.state.markdown_path)
            total_seconds = float(doc.frontmatter.data.get("duration_seconds", 0.0))
        except Exception:
            total_seconds = sum(e.audio.duration_seconds for e in self.state.exchanges)
        duration_mm_ss = format_mm_ss(total_seconds)

        return QuestionRequest(
            model=self.config.llm_model,
            current_transcript=transcript,
            recent_summaries=history_text,
            opening_question=self.config.opening_question,
            question_bank=[],
            language=self.config.language,
            conversation_duration=duration_mm_ss,
            max_tokens=CONFIG.llm_max_tokens_question,
            llm_questions_debug=self.config.llm_questions_debug,
        )

    def regenerate_summary(self) -> None:
        if self.state is None:
            raise RuntimeError("Session has not been started")

        # Snapshot transcript body under lock to avoid partial reads during writes
        with self._io_lock:
            doc = load_transcript(self.state.markdown_path)
            snapshot_body = doc.body
        history_text = [item.summary for item in self.state.recent_history]
        response = generate_summary(
            SummaryRequest(
                transcript_markdown=snapshot_body,
                recent_summaries=history_text,
                model=self.config.llm_model,
                max_tokens=CONFIG.llm_max_tokens_summary,
            )
        )
        # Write latest summary, reloading to merge with any concurrent updates
        with self._io_lock:
            latest = load_transcript(self.state.markdown_path)
            latest.frontmatter.data["summary"] = response.summary_markdown
            write_transcript(self.state.markdown_path, latest)
        log_event(
            "session.summary.updated",
            {
                "session_id": self.state.session_id,
                "model": response.model,
            },
        )

    def schedule_summary_regeneration(self) -> None:
        """Schedule background summary generation for current transcript.

        Safe to call after each exchange; coexists with other writers via _io_lock.
        """
        if self.state is None:
            return
        # Snapshot body under lock to avoid reading while writing
        with self._io_lock:
            doc = load_transcript(self.state.markdown_path)
            snapshot_body = doc.body
        history_text = [item.summary for item in self.state.recent_history]

        def _task() -> None:
            try:
                response = generate_summary(
                    SummaryRequest(
                        transcript_markdown=snapshot_body,
                        recent_summaries=history_text,
                        model=self.config.llm_model,
                        max_tokens=CONFIG.llm_max_tokens_summary,
                    )
                )
                with self._io_lock:
                    latest = load_transcript(self.state.markdown_path)
                    latest.frontmatter.data["summary"] = response.summary_markdown
                    write_transcript(self.state.markdown_path, latest)
                log_event(
                    "session.summary.updated",
                    {
                        "session_id": self.state.session_id,
                        "model": response.model,
                    },
                )
            except (
                Exception
            ):  # pragma: no cover - defensive logging in background thread
                _LOGGER.exception("Background summary generation failed")

        try:
            # May raise RuntimeError if executor has been shut down
            self._summary_executor.submit(_task)
        except Exception:  # pragma: no cover - defensive logging
            _LOGGER.exception("Failed to submit background summary task")

    def complete(self) -> None:
        if self.state is None:
            return
        self._update_frontmatter_after_exchange()
        # Flush background summary tasks before exiting
        try:
            if not self._summary_shutdown:
                self._summary_executor.shutdown(wait=True)
                self._summary_shutdown = True
        except Exception:  # pragma: no cover - defensive
            _LOGGER.exception("Failed to shutdown summary executor")
        log_event(
            "session.complete",
            {
                "session_id": self.state.session_id,
                "responses": len(self.state.exchanges),
                "transcript_file": self.state.markdown_path.name,
                "duration_seconds": sum(
                    e.audio.duration_seconds for e in self.state.exchanges
                ),
            },
        )

    def _update_frontmatter_after_exchange(self) -> None:
        if self.state is None:
            return

        with self._io_lock:
            doc = load_transcript(self.state.markdown_path)

            existing_list = list(doc.frontmatter.data.get("audio_file") or [])
            merged_map: dict[str, dict[str, Any]] = {
                str(item.get("wav")): dict(item)
                for item in existing_list
                if item.get("wav")
            }

            new_segments: list[dict[str, Any]] = []
            for exchange in self.state.exchanges:
                seg = {
                    "wav": exchange.audio.wav_path.name,
                    "mp3": (
                        exchange.audio.mp3_path.name
                        if exchange.audio.mp3_path
                        else None
                    ),
                    "duration_seconds": round(exchange.audio.duration_seconds, 2),
                }
                # Remove any stale pending markers once transcription succeeded
                merged_entry = dict(merged_map.get(seg["wav"], {}))
                merged_entry.update(seg)
                merged_entry.pop("pending", None)
                merged_entry.pop("pending_reason", None)
                merged_map[seg["wav"]] = merged_entry
                new_segments.append(seg)

            merged_list: list[dict[str, Any]] = []
            seen: set[str] = set()
            for entry in existing_list:
                wav = str(entry.get("wav"))
                if wav in merged_map:
                    merged_list.append(merged_map[wav])
                    seen.add(wav)
                else:
                    merged_list.append(entry)
            for seg in new_segments:
                wav = seg["wav"]
                if wav not in seen:
                    merged_list.append(merged_map[wav])
                    seen.add(wav)

            total_duration = sum(
                float(item.get("duration_seconds", 0.0)) for item in merged_list
            )
            audio_file_value = merged_list

            doc.frontmatter.data.update(
                {
                    "duration_seconds": round(total_duration, 2),
                    "audio_file": audio_file_value,
                    "recent_summary_refs": [
                        item.filename for item in self.state.recent_history
                    ],
                    "model_llm": self.config.llm_model,
                    "model_stt": self._backend_selection.model,
                    "stt_backend": self._backend_selection.backend_id,
                    "stt_compute": self._backend_selection.compute,
                    "stt_formatting": self.config.stt_formatting,
                    "app_version": self.config.app_version,
                    "transcript_file": self.state.markdown_path.name,
                }
            )
            write_transcript(self.state.markdown_path, doc)


def _persist_raw_transcription(wav_path: Path, payload: dict) -> None:
    output_path = wav_path.with_suffix(".stt.json")
    tmp_path = output_path.with_name(output_path.name + ".partial")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(output_path)
    remove_error_sentinel(wav_path)
    # Optional cleanup: delete WAV when safe (MP3 + STT present)
    try:
        from .config import CONFIG as _CFG

        if getattr(_CFG, "delete_wav_when_safe", False):
            maybe_delete_wav_when_safe(wav_path)
    except Exception:
        pass


def _format_duration(seconds: float) -> str:  # pragma: no cover - deprecated shim
    return format_hh_mm_ss(seconds)


def _format_minutes_seconds(
    seconds: float,
) -> str:  # pragma: no cover - deprecated shim
    return format_mm_ss(seconds)
