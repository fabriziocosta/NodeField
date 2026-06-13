"""Shared parallel execution helpers."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from concurrent.futures.process import BrokenProcessPool
import os
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
):
    if max_workers <= 1 or len(jobs) <= 1:
        return [func(job) for job in jobs]
    try:
        executor = ProcessPoolExecutor(max_workers=min(max_workers, len(jobs)))
        try:
            if timeout_seconds is None:
                return list(executor.map(func, jobs))
            futures = [executor.submit(func, job) for job in jobs]
            deadline = time.monotonic() + float(timeout_seconds)
            results = []
            for future in futures:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise FuturesTimeoutError()
                results.append(future.result(timeout=remaining))
            return results
        except FuturesTimeoutError:
            executor.shutdown(wait=False, cancel_futures=True)
            if not fallback_on_timeout:
                raise TimeoutError(
                    f"Process-based {timeout_fallback_label} exceeded {float(timeout_seconds):.1f}s."
                )
            if verbose:
                logger.warning(
                    "Process-based %s exceeded %.1fs; falling back to sequential execution.",
                    timeout_fallback_label,
                    float(timeout_seconds),
                )
            return [func(job) for job in jobs]
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
