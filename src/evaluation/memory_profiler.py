"""Memory Profiling and Leak Detection for MoE Router Calibration Pipeline.

Uses tracemalloc (Python allocator) and psutil (OS-level RSS) to detect
memory leaks, unbounded growth, and tensor accumulation during pipeline
execution.

Provides:
- MemoryProfiler: Context manager for tracking memory over code blocks.
- profile_memory: Decorator variant for function-level profiling.
- MemoryReport: Structured report of memory usage and assertions.
"""

import functools
import gc
import logging
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, cast

import torch

logger = logging.getLogger(__name__)

# Try to import psutil; it's optional but strongly recommended
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    logger.warning(
        "psutil not installed. RSS tracking will be unavailable. "
        "Install with: pip install psutil"
    )


F_co = TypeVar("F_co", bound=Callable[..., Any])


@dataclass
class MemorySnapshot:
    """Single point-in-time memory measurement.

    Attributes:
        rss_bytes: Resident Set Size in bytes (OS-level, via psutil).
            None if psutil is unavailable.
        tracemalloc_current_bytes: Current traced memory in bytes.
        tracemalloc_peak_bytes: Peak traced memory in bytes.
        tensor_count: Number of live torch.Tensor objects tracked by GC.
        gpu_allocated_bytes: GPU memory allocated (CUDA/MPS), 0 if N/A.
    """
    rss_bytes: Optional[int] = None
    tracemalloc_current_bytes: int = 0
    tracemalloc_peak_bytes: int = 0
    tensor_count: int = 0
    gpu_allocated_bytes: int = 0

    @property
    def rss_mb(self) -> Optional[float]:
        """RSS in megabytes, or None if unavailable."""
        if self.rss_bytes is None:
            return None
        return self.rss_bytes / (1024 * 1024)

    @property
    def tracemalloc_current_mb(self) -> float:
        """Current traced memory in megabytes."""
        return self.tracemalloc_current_bytes / (1024 * 1024)


def _count_live_tensors() -> int:
    """Count number of live torch.Tensor objects via garbage collector.

    Forces a GC collection first to get an accurate count.

    Returns:
        Number of torch.Tensor objects currently tracked by the GC.
    """
    gc.collect()
    count = 0
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor):
                count += 1
        except (ReferenceError, RuntimeError):
            # Weak references may be collected during iteration
            pass
    return count


def _take_snapshot() -> MemorySnapshot:
    """Capture a complete memory snapshot at this moment.

    Returns:
        MemorySnapshot with all available metrics populated.
    """
    snapshot = MemorySnapshot()

    # psutil RSS
    if _HAS_PSUTIL:
        process = psutil.Process()
        snapshot.rss_bytes = process.memory_info().rss

    # tracemalloc
    if tracemalloc.is_tracing():
        current, peak = tracemalloc.get_traced_memory()
        snapshot.tracemalloc_current_bytes = current
        snapshot.tracemalloc_peak_bytes = peak

    # Tensor count
    snapshot.tensor_count = _count_live_tensors()

    # GPU memory (CUDA)
    if torch.cuda.is_available():
        snapshot.gpu_allocated_bytes = torch.cuda.memory_allocated()

    return snapshot


@dataclass
class MemoryReport:
    """Summary report of memory profiling for a code block.

    Attributes:
        label: Human-readable label for the profiled block.
        start: Snapshot taken before execution.
        end: Snapshot taken after execution.
        rss_delta_bytes: Change in RSS (None if psutil unavailable).
        tracemalloc_delta_bytes: Change in traced memory.
        tensor_delta: Change in tensor count.
        peak_traced_mb: Peak traced memory during execution.
        passed_assertions: Whether all memory assertions passed.
        assertion_errors: List of assertion failure messages.
    """
    label: str = ""
    start: MemorySnapshot = field(default_factory=MemorySnapshot)
    end: MemorySnapshot = field(default_factory=MemorySnapshot)
    rss_delta_bytes: Optional[int] = None
    tracemalloc_delta_bytes: int = 0
    tensor_delta: int = 0
    peak_traced_mb: float = 0.0
    passed_assertions: bool = True
    assertion_errors: List[str] = field(default_factory=list)

    @property
    def rss_delta_mb(self) -> Optional[float]:
        """RSS delta in megabytes, or None if unavailable."""
        if self.rss_delta_bytes is None:
            return None
        return self.rss_delta_bytes / (1024 * 1024)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize report to JSON-compatible dictionary."""
        return {
            "label": self.label,
            "rss_delta_mb": self.rss_delta_mb,
            "tracemalloc_delta_bytes": self.tracemalloc_delta_bytes,
            "tensor_delta": self.tensor_delta,
            "peak_traced_mb": self.peak_traced_mb,
            "tensor_count_start": self.start.tensor_count,
            "tensor_count_end": self.end.tensor_count,
            "passed_assertions": self.passed_assertions,
            "assertion_errors": self.assertion_errors,
        }


class MemoryProfiler:
    """Context manager for memory profiling and leak detection.

    Tracks RSS, tracemalloc allocations, and tensor counts before and
    after a code block, then asserts bounded growth.

    Usage::

        profiler = MemoryProfiler(label="calibration_phase")
        with profiler:
            # ... code to profile ...
        report = profiler.report
        assert report.passed_assertions

    Or as a context manager yielding the report::

        with MemoryProfiler.profile("my_block") as report_holder:
            # ... code to profile ...
        report = report_holder.report

    Args:
        label: Human-readable label for the profiled block.
        max_rss_growth_mb: Maximum allowed RSS growth in MB. None = no check.
        max_tensor_delta: Maximum allowed growth in tensor count.
            Default 0 means no net new tensors should survive.
        assert_on_exit: If True, raises AssertionError on violations
            when exiting the context. If False, violations are only logged.
    """

    def __init__(
        self,
        label: str = "unnamed",
        max_rss_growth_mb: Optional[float] = None,
        max_tensor_delta: int = 0,
        assert_on_exit: bool = False,
    ) -> None:
        self.label = label
        self.max_rss_growth_mb = max_rss_growth_mb
        self.max_tensor_delta = max_tensor_delta
        self.assert_on_exit = assert_on_exit
        self._report: Optional[MemoryReport] = None
        self._tracemalloc_was_tracing: bool = False

    @property
    def report(self) -> MemoryReport:
        """Access the profiling report after exiting the context.

        Raises:
            RuntimeError: If accessed before the context manager exits.
        """
        if self._report is None:
            raise RuntimeError(
                "MemoryReport not yet available. "
                "Use within a 'with' block and access after exit."
            )
        return self._report

    def __enter__(self) -> "MemoryProfiler":
        """Start memory profiling."""
        # Start tracemalloc if not already running
        self._tracemalloc_was_tracing = tracemalloc.is_tracing()
        if not self._tracemalloc_was_tracing:
            tracemalloc.start()

        # Force GC before taking baseline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._start_snapshot = _take_snapshot()
        logger.info(
            "[MemoryProfiler:%s] START — RSS=%.1fMB, traced=%.1fMB, tensors=%d",
            self.label,
            self._start_snapshot.rss_mb or 0.0,
            self._start_snapshot.tracemalloc_current_mb,
            self._start_snapshot.tensor_count,
        )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Complete profiling and generate report."""
        # Force GC before final measurement
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        end_snapshot = _take_snapshot()

        # Build report
        report = MemoryReport(
            label=self.label,
            start=self._start_snapshot,
            end=end_snapshot,
        )

        # Compute deltas
        report.tensor_delta = end_snapshot.tensor_count - self._start_snapshot.tensor_count
        report.tracemalloc_delta_bytes = (
            end_snapshot.tracemalloc_current_bytes
            - self._start_snapshot.tracemalloc_current_bytes
        )

        if (
            self._start_snapshot.rss_bytes is not None
            and end_snapshot.rss_bytes is not None
        ):
            report.rss_delta_bytes = end_snapshot.rss_bytes - self._start_snapshot.rss_bytes

        report.peak_traced_mb = end_snapshot.tracemalloc_peak_bytes / (1024 * 1024)

        # Run assertions
        errors: List[str] = []

        if self.max_tensor_delta is not None and report.tensor_delta > self.max_tensor_delta:
            msg = (
                f"[{self.label}] Tensor count grew by {report.tensor_delta} "
                f"(max allowed: {self.max_tensor_delta}). "
                f"Start={self._start_snapshot.tensor_count}, "
                f"End={end_snapshot.tensor_count}."
            )
            errors.append(msg)
            logger.warning(msg)

        if self.max_rss_growth_mb is not None and report.rss_delta_mb is not None:
            if report.rss_delta_mb > self.max_rss_growth_mb:
                msg = (
                    f"[{self.label}] RSS grew by {report.rss_delta_mb:.1f}MB "
                    f"(max allowed: {self.max_rss_growth_mb:.1f}MB)."
                )
                errors.append(msg)
                logger.warning(msg)

        report.assertion_errors = errors
        report.passed_assertions = len(errors) == 0

        logger.info(
            "[MemoryProfiler:%s] END — RSS=%.1fMB (Δ=%s), traced=%.1fMB, "
            "tensors=%d (Δ=%d), passed=%s",
            self.label,
            end_snapshot.rss_mb or 0.0,
            f"{report.rss_delta_mb:.1f}MB" if report.rss_delta_mb is not None else "N/A",
            end_snapshot.tracemalloc_current_mb,
            end_snapshot.tensor_count,
            report.tensor_delta,
            report.passed_assertions,
        )

        self._report = report

        # Stop tracemalloc if we started it
        if not self._tracemalloc_was_tracing and tracemalloc.is_tracing():
            tracemalloc.stop()

        # Raise on assertion failure if configured
        if self.assert_on_exit and not report.passed_assertions:
            raise AssertionError(
                f"Memory profiler assertions failed for '{self.label}': "
                + "; ".join(errors)
            )


def profile_memory(
    label: Optional[str] = None,
    max_rss_growth_mb: Optional[float] = None,
    max_tensor_delta: int = 0,
    assert_on_exit: bool = False,
) -> Callable[[F_co], F_co]:
    """Decorator for memory-profiling a function.

    Wraps the target function in a MemoryProfiler context manager.
    The profiling report is attached to the function as
    ``func.memory_report`` after execution.

    Args:
        label: Label for the profile. Defaults to the function's __name__.
        max_rss_growth_mb: Maximum RSS growth allowed in MB.
        max_tensor_delta: Maximum tensor count growth. Default 0.
        assert_on_exit: Raise AssertionError on memory violations.

    Returns:
        Decorated function with memory profiling.

    Example::

        @profile_memory(label="train_step", max_tensor_delta=0)
        def train_step(model, batch):
            ...

        train_step(model, batch)
        print(train_step.memory_report.to_dict())
    """
    def decorator(func: F_co) -> F_co:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            profiler_label = label or func.__name__
            profiler = MemoryProfiler(
                label=profiler_label,
                max_rss_growth_mb=max_rss_growth_mb,
                max_tensor_delta=max_tensor_delta,
                assert_on_exit=assert_on_exit,
            )
            with profiler:
                result = func(*args, **kwargs)
            wrapper.memory_report = profiler.report  # type: ignore[attr-defined]
            return result
        return cast(F_co, wrapper)
    return decorator
