from __future__ import annotations

"""Session orchestration and state management."""

from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Sequence

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
        )

        if capture.cancelled:
            self.state.response_index -= 1
            return None

        if capture.discarded_short_answer:
            # Treat as no exchange; do not transcribe, do not persist any body
            self.state.response_index -= 1
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

        transcription = transcribe_wav(
            capture.wav_path,
            model=self.config.stt_model,
            language=self.config.language,
            max_retries=self.config.retry_max_attempts,
            backoff_base_seconds=self.config.retry_backoff_base_ms / 1000,
        )

        _persist_raw_transcription(capture.wav_path, transcription.raw_response)

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

        doc = load_transcript(self.state.markdown_path)
        history_text = [item.summary for item in self.state.recent_history]
        response = generate_summary(
            SummaryRequest(
                transcript_markdown=doc.body,
                recent_summaries=history_text,
                model=self.config.llm_model,
            )
        )
        doc.frontmatter.data["summary"] = response.summary_markdown
        write_transcript(self.state.markdown_path, doc)
        log_event(
            "session.summary.updated",
            {
                "session_id": self.state.session_id,
                "model": response.model,
            },
        )

    def complete(self) -> None:
        if self.state is None:
            return
        self._update_frontmatter_after_exchange()
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

        doc = load_transcript(self.state.markdown_path)
        total_duration = sum(
            item.audio.duration_seconds for item in self.state.exchanges
        )
        audio_segments = [
            {
                "wav": exchange.audio.wav_path.name,
                "mp3": (
                    exchange.audio.mp3_path.name if exchange.audio.mp3_path else None
                ),
                "duration_seconds": round(exchange.audio.duration_seconds, 2),
            }
            for exchange in self.state.exchanges
        ]

        doc.frontmatter.data.update(
            {
                "duration_seconds": round(total_duration, 2),
                "audio_file": audio_segments,
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
