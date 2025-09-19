from __future__ import annotations

"""Multiprocessing transcription worker that reports incremental progress."""

from dataclasses import dataclass
import multiprocessing as mp
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Literal, Optional

from ..transcription import (
    BackendSelection,
    TranscriptionProgress,
    TranscriptionResult,
    create_transcription_backend,
)

WorkerEventType = Literal["started", "progress", "completed", "failed"]


@dataclass(slots=True)
class TranscriptionJobPayload:
    """Payload describing a transcription task for the worker."""

    job_id: str
    session_id: str
    audio_path: Path
    language: str | None
    backend: BackendSelection
    max_retries: int
    backoff_base_seconds: float
    device: str | None = None
    cpu_threads: int | None = None


@dataclass(slots=True)
class WorkerEvent:
    """Event emitted from the worker process for job lifecycle updates."""

    type: WorkerEventType
    job_id: str
    session_id: str
    timestamp: float
    text: str | None = None
    segment_index: int | None = None
    segment: dict | None = None
    result: TranscriptionResult | None = None
    error: str | None = None
    error_type: str | None = None


def _worker_main(
    command_queue: "mp.queues.Queue[TranscriptionJobPayload | None]",
    event_queue: "mp.queues.Queue[WorkerEvent]",
) -> None:
    """Entry point executed within the worker process."""

    backend_cache: dict[tuple[str, str, str | None], object] = {}

    while True:
        payload = command_queue.get()
        if payload is None:
            break

        assert isinstance(payload, TranscriptionJobPayload)
        job_id = payload.job_id
        event_queue.put(
            WorkerEvent(
                type="started",
                job_id=job_id,
                session_id=payload.session_id,
                timestamp=time.time(),
            )
        )

        key = (
            payload.backend.backend_id,
            payload.backend.model,
            payload.backend.compute,
        )
        backend = backend_cache.get(key)
        if backend is None:
            backend = create_transcription_backend(
                payload.backend,
                max_retries=payload.max_retries,
                backoff_base_seconds=payload.backoff_base_seconds,
                device=payload.device,
                cpu_threads=payload.cpu_threads,
            )
            backend_cache[key] = backend

        def _progress(update: TranscriptionProgress) -> None:
            event_queue.put(
                WorkerEvent(
                    type="progress",
                    job_id=job_id,
                    session_id=payload.session_id,
                    timestamp=time.time(),
                    text=update.text,
                    segment_index=update.segment_index,
                    segment=update.segment,
                )
            )

        try:
            # Some stub backends used in tests do not accept a progress callback
            try:
                result = backend.transcribe(
                    payload.audio_path,
                    language=payload.language,
                    progress=_progress,
                )
            except TypeError:
                result = backend.transcribe(
                    payload.audio_path,
                    language=payload.language,
                )
        except Exception as exc:  # pragma: no cover - runtime surface
            event_queue.put(
                WorkerEvent(
                    type="failed",
                    job_id=job_id,
                    session_id=payload.session_id,
                    timestamp=time.time(),
                    error=str(exc),
                    error_type=exc.__class__.__name__,
                )
            )
            continue

        event_queue.put(
            WorkerEvent(
                type="completed",
                job_id=job_id,
                session_id=payload.session_id,
                timestamp=time.time(),
                text=result.text,
                result=result,
            )
        )


class TranscriptionWorkerClient:
    """Client-side helper for submitting jobs and consuming worker events."""

    def __init__(self) -> None:
        self._ctx = mp.get_context("spawn")
        self._commands: mp.queues.Queue[TranscriptionJobPayload | None] = (
            self._ctx.Queue()
        )
        self._events: mp.queues.Queue[WorkerEvent] = self._ctx.Queue()
        self._process: mp.Process | None = None
        self._event_thread: threading.Thread | None = None
        self._handlers: list[Callable[[WorkerEvent], None]] = []
        self._stopping = threading.Event()

    def start(self) -> None:
        if self._process and self._process.is_alive():
            return
        self._process = self._ctx.Process(
            target=_worker_main,
            args=(self._commands, self._events),
            name="hsj-transcription-worker",
            daemon=True,
        )
        self._process.start()
        self._event_thread = threading.Thread(
            target=self._drain_events,
            name="hsj-transcription-events",
            daemon=True,
        )
        self._event_thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stopping.set()
        if self._process and self._process.is_alive():
            try:
                self._commands.put_nowait(None)
            except Exception:
                pass
            self._process.join(timeout=timeout)
        if self._event_thread and self._event_thread.is_alive():
            self._event_thread.join(timeout=timeout)

    def submit(self, job: TranscriptionJobPayload) -> None:
        if self._process is None or not self._process.is_alive():
            self.start()
        self._commands.put(job)

    def subscribe(self, handler: Callable[[WorkerEvent], None]) -> None:
        self._handlers.append(handler)

    def _drain_events(self) -> None:
        while not self._stopping.is_set():
            try:
                event = self._events.get(timeout=0.2)
            except queue.Empty:
                continue
            for handler in list(self._handlers):
                try:
                    handler(event)
                except Exception:  # pragma: no cover - defensive
                    continue

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.stop()
        except Exception:
            pass
