"""Shared parallel execution helpers."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import multiprocessing as mp
import os
import queue
import signal
import time
from typing import Optional

from .runtime_utils import get_runtime_logger

logger = get_runtime_logger(__name__)


def _normalize_n_jobs(n_jobs: Optional[int]) -> int:
    if n_jobs is None:
        return 1
    n_jobs = int(n_jobs)
    if n_jobs == 0:
        raise ValueError("n_jobs must be != 0.")
    if n_jobs < 0:
        cpu_count = os.cpu_count() or 1
        return max(1, cpu_count + 1 + n_jobs)
    return max(1, n_jobs)


def _parallel_map(
    func,
    jobs,
    max_workers: int,
    verbose: bool = False,
    timeout_seconds: Optional[float] = None,
    timeout_fallback_label: str = "parallel work",
    fallback_on_timeout: bool = True,
    deadline_monotonic: Optional[float] = None,
):
    if not jobs:
        return []
    if timeout_seconds is not None:
        return _managed_process_map(
            func,
            jobs,
            max_workers=max_workers,
            timeout_seconds=float(timeout_seconds),
            timeout_fallback_label=timeout_fallback_label,
            deadline_monotonic=deadline_monotonic,
        )
    if max_workers <= 1 or len(jobs) <= 1:
        return [func(job) for job in jobs]
    try:
        executor = ProcessPoolExecutor(max_workers=min(max_workers, len(jobs)))
        try:
            return list(executor.map(func, jobs))
        finally:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
    except Exception as exc:
        if not _should_fallback_to_threads(exc):
            raise
        if verbose:
            logger.warning("Process-based decode parallelism unavailable; falling back to threads.")
        with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs))) as executor:
            return list(executor.map(func, jobs))


def _should_fallback_to_threads(exc: Exception) -> bool:
    if isinstance(exc, (OSError, PermissionError, BrokenProcessPool)):
        return True
    if isinstance(exc, (AttributeError, TypeError, RuntimeError)):
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "pickle",
                "pickl",
                "serialize",
                "serializ",
                "start method",
                "bootstr",
                "daemonic",
            )
        )
    return False


def _managed_worker(result_queue, func, job_idx, job):
    try:
        try:
            os.setsid()
        except Exception:
            pass
        result_queue.put((job_idx, "ok", func(job)))
    except BaseException as exc:  # pragma: no cover - exercised through parent
        result_queue.put((job_idx, "err", repr(exc)))


def _terminate_process(process) -> None:
    if not process.is_alive():
        process.join(timeout=0)
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except Exception:
        process.terminate()
    process.join(timeout=0.25)
    if process.is_alive():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            process.kill()
        process.join(timeout=1.0)


def _terminate_processes(processes) -> None:
    alive = [process for process in processes if process.is_alive()]
    for process in alive:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except Exception:
            process.terminate()
    grace_deadline = time.monotonic() + 0.25
    for process in alive:
        process.join(timeout=max(0.0, grace_deadline - time.monotonic()))
    survivors = [process for process in alive if process.is_alive()]
    for process in survivors:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            process.kill()
    for process in survivors:
        process.join(timeout=1.0)


def _managed_process_map(
    func,
    jobs,
    *,
    max_workers: int,
    timeout_seconds: float,
    timeout_fallback_label: str,
    deadline_monotonic: Optional[float] = None,
):
    if timeout_seconds <= 0.0:
        raise TimeoutError("timeout_seconds must be > 0 when provided.")
    try:
        context = mp.get_context("fork")
    except ValueError as exc:
        raise RuntimeError(
            "Hard-timeout process execution requires the multiprocessing 'fork' context."
        ) from exc

    deadline = (
        time.monotonic() + timeout_seconds
        if deadline_monotonic is None
        else float(deadline_monotonic)
    )
    result_queue = context.Queue()
    results = [None] * len(jobs)
    active = {}
    next_job_idx = 0
    completed = 0

    def start_available_jobs() -> None:
        nonlocal next_job_idx
        worker_limit = max(1, min(int(max_workers), len(jobs)))
        while next_job_idx < len(jobs) and len(active) < worker_limit:
            process = context.Process(
                target=_managed_worker,
                args=(result_queue, func, next_job_idx, jobs[next_job_idx]),
                daemon=True,
            )
            process.start()
            active[next_job_idx] = process
            next_job_idx += 1

    try:
        start_available_jobs()
        while completed < len(jobs):
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(
                    f"Process-based {timeout_fallback_label} exceeded {timeout_seconds:.1f}s."
                )
            try:
                job_idx, status, payload = result_queue.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError(
                    f"Process-based {timeout_fallback_label} exceeded {timeout_seconds:.1f}s."
                ) from exc
            process = active.pop(job_idx)
            process.join(timeout=0.25)
            if process.is_alive():
                _terminate_process(process)
            if status == "err":
                raise RuntimeError(payload)
            results[job_idx] = payload
            completed += 1
            start_available_jobs()
        return results
    finally:
        _terminate_processes(active.values())
        result_queue.close()
