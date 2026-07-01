"""Minimal in-process job registry for long-running engine work.

This deployment is a single uvicorn worker serving one operator, so an
in-process registry (module dict + threads) is the correct scale. Jobs report
an honest tri-state status: running, done, or failed with the real error.
Progress is populated only when the job body actually reports it; it is never
fabricated.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Callable

logger = logging.getLogger("kairos.jobs")

# Keep the most recent finished jobs so the operator can still read a result
# after completion; running jobs are never evicted.
_MAX_FINISHED = 50

_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}


def _evict_finished_locked() -> None:
    finished = [
        (job["finished_at"], job_id)
        for job_id, job in _JOBS.items()
        if job["status"] in ("done", "failed") and job["finished_at"] is not None
    ]
    if len(finished) <= _MAX_FINISHED:
        return
    finished.sort()
    for _, job_id in finished[: len(finished) - _MAX_FINISHED]:
        _JOBS.pop(job_id, None)


def running_job(name: str) -> str | None:
    """Return the id of a currently running job with this name, if any."""
    with _LOCK:
        for job_id, job in _JOBS.items():
            if job["name"] == name and job["status"] == "running":
                return job_id
    return None


def get(job_id: str) -> dict[str, Any] | None:
    """Return a copy of the job record, or None for an unknown id."""
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return None
        record = dict(job)
        if record["progress"] is not None:
            record["progress"] = dict(record["progress"])
        return record


def submit(name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    """Run fn in a background thread and return a job id.

    fn may accept a keyword argument named progress_cb; if it does, the caller
    should pass progress_cb=jobs-provided callback via functools/partial before
    submit, or simply accept the injected `_job_progress` kwarg below. To keep
    the contract explicit, submit injects nothing: the caller wires progress by
    closing over `report_progress(job_id, done, total)`.
    """
    job_id = uuid.uuid4().hex[:12]
    with _LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "name": name,
            "status": "running",
            "progress": None,
            "error": None,
            "started_at": time.time(),
            "finished_at": None,
            "result": None,
        }
        _evict_finished_locked()

    def _run() -> None:
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:  # honest failure: never report done on error
            logger.exception("job %s (%s) failed", job_id, name)
            with _LOCK:
                job = _JOBS.get(job_id)
                if job is not None:
                    job["status"] = "failed"
                    job["error"] = str(exc)
                    job["finished_at"] = time.time()
            return
        with _LOCK:
            job = _JOBS.get(job_id)
            if job is not None:
                job["status"] = "done"
                job["result"] = result
                job["finished_at"] = time.time()

    thread = threading.Thread(target=_run, name=f"kairos-job-{name}-{job_id}", daemon=True)
    thread.start()
    return job_id


def report_progress(job_id: str, done: int, total: int) -> None:
    """Record real progress for a running job. Only the job body calls this."""
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is not None and job["status"] == "running":
            job["progress"] = {"done": int(done), "total": int(total)}
