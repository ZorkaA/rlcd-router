"""Targeted Gating Evaluation module for MoE Router Calibration Pipeline.

Provides:
- Targeted ECE computation at abort (0.05) and mass cutoff (0.85) thresholds.
- Memory profiling and leak detection utilities.
"""

from src.evaluation.targeted_ece import (
    evaluate_targeted_ece,
    compute_targeted_ece_single,
    compute_cumulative_mass_stats,
)

from src.evaluation.memory_profiler import (
    MemoryProfiler,
    profile_memory,
    MemoryReport,
)

__all__ = [
    "evaluate_targeted_ece",
    "compute_targeted_ece_single",
    "compute_cumulative_mass_stats",
    "MemoryProfiler",
    "profile_memory",
    "MemoryReport",
]
