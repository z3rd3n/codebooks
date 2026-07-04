"""Tiny thread-based job manager (in-memory store).

Jobs run in one worker thread, figures sequential within a job, results
appended as they finish so the UI can show partial results while polling.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Callable

_LOCK = threading.Lock()
_JOBS: dict[str, "Job"] = {}


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"  # queued | running | done | error
    progress: float = 0.0
    message: str = ""
    results: list[dict] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "results": self.results,
        }


def get_job(job_id: str) -> Job | None:
    with _LOCK:
        return _JOBS.get(job_id)


def _run(job: Job, target: Callable[[Job], None]) -> None:
    job.status = "running"
    try:
        target(job)
        job.status = "done"
        job.progress = 1.0
    except Exception as exc:  # noqa: BLE001 - surfaced to the client
        job.status = "error"
        job.error = str(exc)
        job.message = f"failed: {exc}"


def start_job(kind: str, target: Callable[[Job], None]) -> str:
    job_id = uuid.uuid4().hex[:12]
    job = Job(id=job_id, kind=kind)
    with _LOCK:
        _JOBS[job_id] = job
    thread = threading.Thread(target=_run, args=(job, target), daemon=True)
    thread.start()
    return job_id
