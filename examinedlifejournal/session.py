from __future__ import annotations

"""Session orchestration and state management."""

from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, List, Sequence
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
from .question_bank import QUESTION_BANK
from .storage import (
    Frontmatter,
    TranscriptDocument,
    append_exchange_body,
    load_transcript,
    write_transcript,
)
from .transcription import TranscriptionResult, transcribe_wav
from .events import log_event
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
    opening_question: str
    max_history_tokens: int = CONFIG.max_history_tokens
    recent_summaries_limit: int = CONFIG.max_recent_summaries
    app_version: str = __version__
    retry_max_attempts: int = CONFIG.retry_max_attempts
    retry_backoff_base_ms: int = CONFIG.retry_backoff_base_ms
    ffmpeg_path: str | None = CONFIG.ffmpeg_path
    language: str = "en"


class SessionManager:
    """High-level API consumed by the CLI layer."""

    def __init__(self, config: SessionConfig) -> None:
        self.config = config
        self.state: SessionState | None = None
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

        session_id = datetime.now().strftime("%y%m%d_%H%M%S")
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
            "model_stt": self.config.stt_model,
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
                "stt_model": self.config.stt_model,
                "language": self.config.language,
                "recent_summaries_count": len(recent),
            },
        )
        return self.state

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
                    # Expect pattern <session_id>_<NN>.wav
                    stem = Path(wav_name).stem
                    parts = stem.split("_")
                    if len(parts) >= 2 and parts[-1].isdigit():
                        indices.append(int(parts[-1]))
                except Exception:
                    continue
            if not indices:
                # Fallback: inspect filesystem for existing WAVs
                for wav_path in sorted(
                    self.state.audio_dir.glob(f"{session_id}_*.wav")
                ):
                    try:
                        stem = wav_path.stem
                        parts = stem.split("_")
                        if len(parts) >= 2 and parts[-1].isdigit():
                            indices.append(int(parts[-1]))
                    except Exception:
                        continue
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

        self.state.response_index += 1
        segment_basename = f"{self.state.session_id}_{self.state.response_index:02d}"

        capture = record_response(
            self.state.audio_dir,
            segment_basename,
            console=console,
            sample_rate=16_000,
            ffmpeg_path=self.config.ffmpeg_path,
            print_saved_message=False,
        )

        if capture.cancelled:
            # Mark disposition for clearer UI messaging
            self.state.last_cancelled = True
            self.state.last_discarded_short = False
            self.state.response_index -= 1
            return None

        if capture.discarded_short_answer:
            # Treat as no exchange; do not transcribe, do not persist any body
            self.state.response_index -= 1
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

        seg_formatted = _format_duration(capture.duration_seconds)
        total_formatted = _format_duration(prior_total + capture.duration_seconds)
        console.print(
            Text(
                f"Saved WAV → {capture.wav_path.name} ({seg_formatted}; total {total_formatted})",
                style="green",
            )
        )

        transcription = transcribe_wav(
            capture.wav_path,
            model=self.config.stt_model,
            language=self.config.language,
            max_retries=self.config.retry_max_attempts,
            backoff_base_seconds=self.config.retry_backoff_base_ms / 1000,
        )

        _persist_raw_transcription(capture.wav_path, transcription.raw_response)

        # Serialize body append to avoid racing with background summary writes
        with self._io_lock:
            append_exchange_body(
                self.state.markdown_path,
                question,
                transcription.text,
            )

        exchange = Exchange(
            question=question,
            transcript=transcription.text,
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
        log_event(
            "session.exchange.recorded",
            {
                "session_id": self.state.session_id,
                "response_index": self.state.response_index,
                "wav": exchange.audio.wav_path.name,
                "mp3": (
                    exchange.audio.mp3_path.name if exchange.audio.mp3_path else None
                ),
                "duration_seconds": round(exchange.audio.duration_seconds, 2),
                "stt_model": exchange.transcription.model,
                "quit_after": exchange.audio.quit_after,
                "cancelled": exchange.audio.cancelled,
            },
        )
        return exchange

    def generate_next_question(self, transcript: str) -> QuestionResponse:
        if self.state is None:
            raise RuntimeError("Session has not been started")

        history_text = [item.summary for item in self.state.recent_history]
        lowered = transcript.lower()
        if "give me a question" in lowered:
            import random

            chosen = random.choice(QUESTION_BANK)
            response = QuestionResponse(question=chosen, model="question-bank")
            if self.state.exchanges:
                self.state.exchanges[-1].followup_question = response
            return response

        request = QuestionRequest(
            model=self.config.llm_model,
            current_transcript=transcript,
            recent_summaries=history_text,
            opening_question=self.config.opening_question,
            question_bank=QUESTION_BANK,
            language=self.config.language,
        )
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

            # Build list of new segments from current process
            new_segments = [
                {
                    "wav": exchange.audio.wav_path.name,
                    "mp3": (
                        exchange.audio.mp3_path.name
                        if exchange.audio.mp3_path
                        else None
                    ),
                    "duration_seconds": round(exchange.audio.duration_seconds, 2),
                }
                for exchange in self.state.exchanges
            ]

            if self.state.resumed:
                # Merge with existing without duplicating entries
                existing_list = list(doc.frontmatter.data.get("audio_file") or [])
                by_wav: dict[str, dict] = {
                    str(item.get("wav")): item for item in existing_list
                }
                for seg in new_segments:
                    by_wav[seg["wav"]] = seg
                merged_list = [
                    # Keep original order, append any truly new ones in order recorded
                    *[item for item in existing_list if str(item.get("wav")) in by_wav],
                    *[
                        seg
                        for seg in new_segments
                        if seg["wav"] not in {str(i.get("wav")) for i in existing_list}
                    ],
                ]
                total_duration = sum(
                    float(item.get("duration_seconds", 0.0)) for item in merged_list
                )
                audio_file_value = merged_list
            else:
                total_duration = sum(
                    item.audio.duration_seconds for item in self.state.exchanges
                )
                audio_file_value = new_segments

            doc.frontmatter.data.update(
                {
                    "duration_seconds": round(total_duration, 2),
                    "audio_file": audio_file_value,
                    "recent_summary_refs": [
                        item.filename for item in self.state.recent_history
                    ],
                    "model_llm": self.config.llm_model,
                    "model_stt": self.config.stt_model,
                    "app_version": self.config.app_version,
                    "transcript_file": self.state.markdown_path.name,
                }
            )
            write_transcript(self.state.markdown_path, doc)


def _persist_raw_transcription(wav_path: Path, payload: dict) -> None:
    output_path = wav_path.with_suffix(".stt.json")
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    # Optional cleanup: delete WAV when safe (MP3 + STT present)
    try:
        from .config import CONFIG as _CFG

        if getattr(_CFG, "delete_wav_when_safe", False):
            mp3_path = wav_path.with_suffix(".mp3")
            if mp3_path.exists() and wav_path.exists():
                try:
                    wav_path.unlink(missing_ok=True)
                    log_event(
                        "audio.wav.deleted",
                        {
                            "wav": wav_path.name,
                            "reason": "safe_delete_after_mp3_and_stt",
                        },
                    )
                except Exception:
                    pass
    except Exception:
        pass


def _format_duration(seconds: float) -> str:
    total_seconds = int(round(max(0.0, float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
