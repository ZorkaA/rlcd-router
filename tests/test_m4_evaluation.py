"""Unit tests for Milestone 4: Targeted ECE Evaluation, Memory Profiler, and Pipeline.

Tests cover:
- targeted_ece.py: evaluate_targeted_ece, compute_targeted_ece_single, cumulative mass stats
- memory_profiler.py: MemoryProfiler context manager, decorator, leak detection
- pipeline.py: CLI argument parsing, synthetic pipeline execution, report serialization
"""

import gc
import json
import sys
import tempfile
import tracemalloc
from pathlib import Path
from typing import Dict, Tuple
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

# Ensure project root is on sys.path
WORKSPACE_ROOT = Path(__file__).parent.parent.resolve()
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.config import (
    ABORT_THRESHOLD,
    MASS_CUTOFF_THRESHOLD,
    TARGETED_ECE_BANDWIDTH,
    TARGETED_ECE_THRESHOLDS,
    NUM_DEEP_LAYERS,
    NUM_EARLY_LAYERS,
    NUM_LATE_LAYERS,
    NUM_HORIZONS,
    NUM_EXPERTS,
    GRID_BUCKETS,
    GRID_HORIZONS,
    PipelineConfig,
)

from src.evaluation.targeted_ece import (
    evaluate_targeted_ece,
    compute_targeted_ece_single,
    compute_cumulative_mass_stats,
    _create_binary_ground_truth,
    _compute_overall_ece,
    _get_layer_bucket_slice,
    _get_horizon_index,
)

from src.evaluation.memory_profiler import (
    MemoryProfiler,
    MemoryReport,
    MemorySnapshot,
    profile_memory,
    _count_live_tensors,
    _take_snapshot,
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def standard_calibrated_probs():
    """Standard calibrated probabilities: (32, 3, 20, 60)."""
    torch.manual_seed(42)
    logits = torch.randn(32, 3, 20, 60, dtype=torch.float32)
    return F.softmax(logits, dim=-1)


@pytest.fixture
def standard_ground_truth_top4():
    """Standard ground truth top-4 indices: (32, 3, 20, 4)."""
    torch.manual_seed(42)
    logits = torch.randn(32, 3, 20, 60, dtype=torch.float32)
    return torch.topk(logits, k=4, dim=-1).indices


@pytest.fixture
def small_calibrated_probs():
    """Small calibrated probabilities: (8, 3, 20, 60)."""
    torch.manual_seed(123)
    logits = torch.randn(8, 3, 20, 60, dtype=torch.float32)
    return F.softmax(logits, dim=-1)


@pytest.fixture
def small_ground_truth_top4():
    """Small ground truth top-4 indices: (8, 3, 20, 4)."""
    torch.manual_seed(123)
    logits = torch.randn(8, 3, 20, 60, dtype=torch.float32)
    return torch.topk(logits, k=4, dim=-1).indices


# =========================================================================
# Tests: _create_binary_ground_truth
# =========================================================================

class TestCreateBinaryGroundTruth:
    """Tests for the binary ground-truth creation helper."""

    def test_basic_shape(self):
        """Binary ground truth has correct shape."""
        gt_top4 = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
        gt_binary = _create_binary_ground_truth(gt_top4, num_experts=10)
        assert gt_binary.shape == (2, 10)

    def test_correct_values(self):
        """Exactly 4 experts marked as 1.0 per sample."""
        gt_top4 = torch.tensor([[0, 3, 5, 9]])
        gt_binary = _create_binary_ground_truth(gt_top4, num_experts=10)
        assert gt_binary.sum().item() == 4
        assert gt_binary[0, 0].item() == 1.0
        assert gt_binary[0, 3].item() == 1.0
        assert gt_binary[0, 5].item() == 1.0
        assert gt_binary[0, 9].item() == 1.0
        assert gt_binary[0, 1].item() == 0.0

    def test_batched_nd(self):
        """Works with multi-dimensional leading shapes."""
        gt_top4 = torch.randint(0, 60, (4, 3, 20, 4))
        gt_binary = _create_binary_ground_truth(gt_top4, num_experts=60)
        assert gt_binary.shape == (4, 3, 20, 60)


# =========================================================================
# Tests: compute_targeted_ece_single
# =========================================================================

class TestComputeTargetedECESingle:
    """Tests for single-threshold targeted ECE computation."""

    def test_perfect_calibration(self):
        """Perfect calibration yields ECE = 0."""
        # Create predictions exactly at threshold 0.5 ± bandwidth
        N = 100
        probs = torch.full((N, 10), 0.1)  # 10 experts, each 0.1 prob
        # Ground truth: exactly 1 expert correct per sample -> accuracy = 0.1
        gt_binary = torch.zeros(N, 10)
        gt_binary[:, 0] = 1.0  # Expert 0 always correct

        # At threshold=0.1, all predictions fall in window
        # conf = 0.1, acc = 0.1 -> ECE = 0
        ece, count = compute_targeted_ece_single(probs, gt_binary, threshold=0.1, bandwidth=0.05)
        assert abs(ece) < 1e-6
        assert count == N * 10  # All predictions in window

    def test_empty_window(self):
        """No predictions in window yields (0.0, 0)."""
        probs = torch.full((10, 5), 0.5)
        gt_binary = torch.zeros(10, 5)
        ece, count = compute_targeted_ece_single(probs, gt_binary, threshold=0.01, bandwidth=0.005)
        assert ece == 0.0
        assert count == 0

    def test_ece_nonnegative(self, standard_calibrated_probs, standard_ground_truth_top4):
        """ECE is always non-negative."""
        probs_2d = standard_calibrated_probs.reshape(-1, 60)
        gt_2d = standard_ground_truth_top4.reshape(-1, 4)
        gt_binary = _create_binary_ground_truth(gt_2d, 60)
        for threshold in [0.01, 0.05, 0.1, 0.5, 0.85, 0.95]:
            ece, _ = compute_targeted_ece_single(probs_2d, gt_binary, threshold)
            assert ece >= 0.0, f"ECE negative at threshold {threshold}: {ece}"

    def test_ece_bounded_by_one(self, standard_calibrated_probs, standard_ground_truth_top4):
        """ECE is bounded by 1.0."""
        probs_2d = standard_calibrated_probs.reshape(-1, 60)
        gt_2d = standard_ground_truth_top4.reshape(-1, 4)
        gt_binary = _create_binary_ground_truth(gt_2d, 60)
        for threshold in [0.05, 0.85]:
            ece, _ = compute_targeted_ece_single(probs_2d, gt_binary, threshold)
            assert ece <= 1.0, f"ECE > 1.0 at threshold {threshold}: {ece}"

    def test_matches_oracle(self, standard_calibrated_probs, standard_ground_truth_top4):
        """Targeted ECE matches the oracle reference implementation from conftest."""
        from tests.conftest import MathematicalOracles

        # Take a single bucket/horizon slice
        probs_slice = standard_calibrated_probs[:, 0, :6, :]  # early, T+1
        gt_slice = standard_ground_truth_top4[:, 0, :6, :]

        probs_2d = probs_slice.reshape(-1, 60)
        gt_2d = gt_slice.reshape(-1, 4)

        gt_binary = _create_binary_ground_truth(gt_2d, 60)
        our_ece, our_count = compute_targeted_ece_single(
            probs_2d, gt_binary, threshold=0.05, bandwidth=0.025
        )

        oracle_ece, oracle_count = MathematicalOracles.targeted_ece(
            probs_2d, gt_2d, threshold=0.05, window=0.025
        )

        assert abs(our_ece - oracle_ece) < 1e-5, (
            f"ECE mismatch: ours={our_ece:.8f}, oracle={oracle_ece:.8f}"
        )
        assert our_count == oracle_count


# =========================================================================
# Tests: compute_cumulative_mass_stats
# =========================================================================

class TestCumulativeMassStats:
    """Tests for cumulative probability mass statistics."""

    def test_perfect_mass(self):
        """Perfect predictions give cumulative mass = 1.0."""
        N = 10
        probs = torch.zeros(N, 60)
        # Put all probability on top-4
        gt_top4 = torch.tensor([[0, 1, 2, 3]] * N)
        probs[:, 0] = 0.25
        probs[:, 1] = 0.25
        probs[:, 2] = 0.25
        probs[:, 3] = 0.25

        stats = compute_cumulative_mass_stats(probs, gt_top4)
        assert abs(stats["mean"] - 1.0) < 1e-5
        assert abs(stats["fraction_above_0.85"] - 1.0) < 1e-5

    def test_output_keys(self, standard_calibrated_probs, standard_ground_truth_top4):
        """Stats dict contains all required keys."""
        probs_2d = standard_calibrated_probs.reshape(-1, 60)
        gt_2d = standard_ground_truth_top4.reshape(-1, 4)
        stats = compute_cumulative_mass_stats(probs_2d, gt_2d)
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "fraction_above_0.85" in stats
        assert "num_samples" in stats

    def test_values_in_range(self, standard_calibrated_probs, standard_ground_truth_top4):
        """Mean mass is between 0 and 1, fraction is between 0 and 1."""
        probs_2d = standard_calibrated_probs.reshape(-1, 60)
        gt_2d = standard_ground_truth_top4.reshape(-1, 4)
        stats = compute_cumulative_mass_stats(probs_2d, gt_2d)
        assert 0.0 <= stats["mean"] <= 1.0
        assert 0.0 <= stats["fraction_above_0.85"] <= 1.0
        assert stats["min"] <= stats["mean"] <= stats["max"]

    def test_single_sample(self):
        """Works with a single sample."""
        probs = torch.softmax(torch.randn(1, 60), dim=-1)
        gt = torch.topk(probs, k=4, dim=-1).indices
        stats = compute_cumulative_mass_stats(probs, gt)
        assert stats["num_samples"] == 1
        # std should be 0 for single sample
        assert stats["std"] == 0.0


# =========================================================================
# Tests: evaluate_targeted_ece (integration)
# =========================================================================

class TestEvaluateTargetedECE:
    """Integration tests for the main evaluate_targeted_ece function."""

    def test_output_structure(self, standard_calibrated_probs, standard_ground_truth_top4):
        """Output dict has all required keys and structure."""
        result = evaluate_targeted_ece(
            standard_calibrated_probs, standard_ground_truth_top4
        )

        # Check top-level keys
        assert "targeted_ece_0.05" in result
        assert "targeted_ece_0.85" in result
        assert "cumulative_mass_0.85_stats" in result
        assert "overall_ece" in result

        # Check bucket × horizon keys for each threshold
        for threshold_key in ["targeted_ece_0.05", "targeted_ece_0.85"]:
            ece_dict = result[threshold_key]
            for bucket in GRID_BUCKETS:
                for horizon in GRID_HORIZONS:
                    assert (bucket, horizon) in ece_dict, (
                        f"Missing key ({bucket}, {horizon}) in {threshold_key}"
                    )

    def test_all_ece_nonnegative(self, standard_calibrated_probs, standard_ground_truth_top4):
        """All targeted ECE values are non-negative."""
        result = evaluate_targeted_ece(
            standard_calibrated_probs, standard_ground_truth_top4
        )
        for threshold_key in ["targeted_ece_0.05", "targeted_ece_0.85"]:
            for (bucket, horizon), val in result[threshold_key].items():
                assert val >= 0.0, f"Negative ECE at {threshold_key}[{bucket},{horizon}]={val}"

    def test_overall_ece_nonnegative(self, standard_calibrated_probs, standard_ground_truth_top4):
        """Overall ECE is non-negative."""
        result = evaluate_targeted_ece(
            standard_calibrated_probs, standard_ground_truth_top4
        )
        assert result["overall_ece"] >= 0.0

    def test_custom_thresholds(self, standard_calibrated_probs, standard_ground_truth_top4):
        """Custom threshold list works correctly."""
        result = evaluate_targeted_ece(
            standard_calibrated_probs,
            standard_ground_truth_top4,
            thresholds=[0.1, 0.5, 0.9],
        )
        assert "targeted_ece_0.1" in result
        assert "targeted_ece_0.5" in result
        assert "targeted_ece_0.9" in result
        assert "targeted_ece_0.05" not in result  # Default not included

    def test_input_validation_4d(self):
        """Raises ValueError for non-4D inputs."""
        with pytest.raises(ValueError, match="4D"):
            evaluate_targeted_ece(
                torch.randn(10, 60),
                torch.randint(0, 60, (10, 4)),
            )

    def test_deterministic(self, standard_calibrated_probs, standard_ground_truth_top4):
        """Same inputs produce identical results."""
        r1 = evaluate_targeted_ece(standard_calibrated_probs, standard_ground_truth_top4)
        r2 = evaluate_targeted_ece(standard_calibrated_probs, standard_ground_truth_top4)

        for threshold_key in ["targeted_ece_0.05", "targeted_ece_0.85"]:
            for key in r1[threshold_key]:
                assert r1[threshold_key][key] == r2[threshold_key][key]
        assert r1["overall_ece"] == r2["overall_ece"]

    def test_small_batch(self, small_calibrated_probs, small_ground_truth_top4):
        """Works with small batch sizes."""
        result = evaluate_targeted_ece(small_calibrated_probs, small_ground_truth_top4)
        assert "overall_ece" in result
        assert result["overall_ece"] >= 0.0


# =========================================================================
# Tests: _compute_overall_ece
# =========================================================================

class TestComputeOverallECE:
    """Tests for the global ECE computation."""

    def test_perfect_calibration(self):
        """Uniform probabilities with matching accuracy -> ECE ~ 0."""
        N = 1000
        probs = torch.full((N, 10), 0.1)
        gt_binary = torch.zeros(N, 10)
        # Set exactly 1 expert correct per sample (10% accuracy matches 0.1 conf)
        gt_binary[:, 0] = 1.0
        ece = _compute_overall_ece(probs, gt_binary)
        assert ece < 0.01, f"ECE too high for perfect calibration: {ece}"

    def test_empty_input(self):
        """Empty input yields 0."""
        ece = _compute_overall_ece(torch.tensor([]), torch.tensor([]))
        assert ece == 0.0

    def test_ece_bounded(self):
        """ECE is between 0 and 1."""
        torch.manual_seed(42)
        probs = F.softmax(torch.randn(100, 60), dim=-1)
        gt_binary = torch.zeros(100, 60)
        gt_binary[:, :4] = 1.0
        ece = _compute_overall_ece(probs, gt_binary)
        assert 0.0 <= ece <= 1.0


# =========================================================================
# Tests: Helper functions
# =========================================================================

class TestHelperFunctions:
    """Tests for layer bucket and horizon mapping helpers."""

    def test_early_bucket_slice(self):
        """Early bucket maps to first 6 layers."""
        s = _get_layer_bucket_slice("early")
        indices = list(range(20))[s]
        assert len(indices) == 6
        assert indices == [0, 1, 2, 3, 4, 5]

    def test_late_bucket_slice(self):
        """Late bucket maps to layers 6-19."""
        s = _get_layer_bucket_slice("late")
        indices = list(range(20))[s]
        assert len(indices) == 14
        assert indices == list(range(6, 20))

    def test_invalid_bucket(self):
        """Invalid bucket raises ValueError."""
        with pytest.raises(ValueError, match="Unknown layer bucket"):
            _get_layer_bucket_slice("middle")

    def test_horizon_indices(self):
        """Horizon labels map to correct indices."""
        assert _get_horizon_index("T+1") == 0
        assert _get_horizon_index("T+2") == 1
        assert _get_horizon_index("T+3") == 2

    def test_invalid_horizon(self):
        """Invalid horizon raises ValueError."""
        with pytest.raises(ValueError, match="Unknown horizon"):
            _get_horizon_index("T+4")


# =========================================================================
# Tests: MemoryProfiler
# =========================================================================

class TestMemoryProfiler:
    """Tests for the memory profiler context manager."""

    def test_basic_usage(self):
        """MemoryProfiler produces a valid report."""
        profiler = MemoryProfiler(label="test_basic")
        with profiler:
            x = torch.randn(100, 100)
        report = profiler.report
        assert isinstance(report, MemoryReport)
        assert report.label == "test_basic"

    def test_report_has_tensor_counts(self):
        """Report tracks tensor counts at start and end."""
        profiler = MemoryProfiler(label="test_tensors")
        with profiler:
            pass
        report = profiler.report
        assert report.start.tensor_count >= 0
        assert report.end.tensor_count >= 0

    def test_report_before_exit_raises(self):
        """Accessing report before exiting context raises RuntimeError."""
        profiler = MemoryProfiler(label="test_early")
        with pytest.raises(RuntimeError, match="not yet available"):
            _ = profiler.report

    def test_no_leak_passes(self):
        """No net tensor growth passes assertions."""
        profiler = MemoryProfiler(label="test_no_leak", max_tensor_delta=5)
        with profiler:
            _ = 1 + 1  # No tensor allocation
        assert profiler.report.passed_assertions

    def test_assert_on_exit(self):
        """assert_on_exit=True raises AssertionError on violations."""
        # We create a profiler that disallows any tensor growth
        # and then intentionally create tensors that survive
        profiler = MemoryProfiler(
            label="test_assert_exit",
            max_tensor_delta=-1,  # Require decrease
            assert_on_exit=True,
        )
        with pytest.raises(AssertionError, match="Memory profiler assertions failed"):
            with profiler:
                # Create a tensor that will persist
                pass  # Even existing tensors may cause delta > -1

    def test_report_to_dict(self):
        """to_dict() produces serializable output."""
        profiler = MemoryProfiler(label="test_dict")
        with profiler:
            pass
        d = profiler.report.to_dict()
        assert isinstance(d, dict)
        assert "label" in d
        assert "tensor_delta" in d
        assert "passed_assertions" in d
        # Verify JSON serializable
        json.dumps(d)

    def test_tracemalloc_restored(self):
        """tracemalloc state is restored after profiling."""
        was_tracing = tracemalloc.is_tracing()
        if was_tracing:
            tracemalloc.stop()

        assert not tracemalloc.is_tracing()
        profiler = MemoryProfiler(label="test_tracemalloc")
        with profiler:
            assert tracemalloc.is_tracing()
        # Should be stopped again since it wasn't running before
        assert not tracemalloc.is_tracing()


class TestMemorySnapshot:
    """Tests for MemorySnapshot dataclass."""

    def test_rss_mb_conversion(self):
        """RSS bytes convert to MB correctly."""
        snap = MemorySnapshot(rss_bytes=1024 * 1024 * 100)
        assert snap.rss_mb == 100.0

    def test_rss_mb_none(self):
        """None RSS returns None MB."""
        snap = MemorySnapshot()
        assert snap.rss_mb is None

    def test_tracemalloc_mb(self):
        """Traced memory bytes convert to MB correctly."""
        snap = MemorySnapshot(tracemalloc_current_bytes=1024 * 1024 * 50)
        assert snap.tracemalloc_current_mb == 50.0


class TestProfileMemoryDecorator:
    """Tests for the profile_memory decorator."""

    def test_basic_decorator(self):
        """Decorator attaches memory_report to function."""
        @profile_memory(label="dec_test")
        def my_func():
            return 42

        result = my_func()
        assert result == 42
        assert hasattr(my_func, "memory_report")
        assert isinstance(my_func.memory_report, MemoryReport)
        assert my_func.memory_report.label == "dec_test"

    def test_decorator_default_label(self):
        """Decorator uses function name as default label."""
        @profile_memory()
        def compute_something():
            return "done"

        compute_something()
        assert compute_something.memory_report.label == "compute_something"

    def test_decorator_preserves_args(self):
        """Decorator correctly passes arguments through."""
        @profile_memory()
        def add(a, b):
            return a + b

        assert add(3, 4) == 7


class TestCountLiveTensors:
    """Tests for tensor counting utility."""

    def test_count_returns_int(self):
        """_count_live_tensors returns an integer."""
        count = _count_live_tensors()
        assert isinstance(count, int)
        assert count >= 0

    def test_count_increases_with_allocation(self):
        """Creating tensors increases the count."""
        gc.collect()
        baseline = _count_live_tensors()
        tensors = [torch.randn(10) for _ in range(5)]
        count_after = _count_live_tensors()
        assert count_after >= baseline + 5
        del tensors
        gc.collect()


# =========================================================================
# Tests: Pipeline
# =========================================================================

class TestPipeline:
    """Tests for the end-to-end pipeline."""

    def test_build_parser(self):
        """Parser builds without error and has expected arguments."""
        from src.pipeline import build_parser
        parser = build_parser()
        args = parser.parse_args(["--synthetic", "--verbose"])
        assert args.synthetic is True
        assert args.verbose is True

    def test_parser_defaults(self):
        """Parser has sensible defaults."""
        from src.pipeline import build_parser
        parser = build_parser()
        args = parser.parse_args([])
        assert args.synthetic is False
        assert args.verbose is False
        assert args.device is None
        assert args.output is None

    def test_serialize_report(self):
        """Report serialization converts tuple keys to strings."""
        from src.pipeline import _serialize_report
        report = {
            "targeted_ece_0.05": {
                ("early", "T+1"): 0.01,
                ("late", "T+3"): 0.02,
            },
            "overall_ece": 0.05,
        }
        serialized = _serialize_report(report)
        assert "early|T+1" in serialized["targeted_ece_0.05"]
        assert "late|T+3" in serialized["targeted_ece_0.05"]
        assert serialized["overall_ece"] == 0.05
        # Verify JSON serializable
        json.dumps(serialized)

    def test_synthetic_pipeline_runs(self, tmp_path):
        """Synthetic pipeline runs end-to-end and produces a report."""
        from src.pipeline import run_pipeline
        output_path = tmp_path / "test_report.json"
        report = run_pipeline(
            synthetic=True,
            verbose=False,
            output_path=output_path,
            device="cpu",
        )

        assert "phases" in report
        assert "timings" in report
        assert "memory_profile" in report
        assert report["phases"]["data"]["mode"] == "synthetic"
        assert report["timings"]["total_seconds"] > 0

        # Verify JSON file was written
        assert output_path.exists()
        with open(output_path) as f:
            loaded = json.load(f)
        assert "phases" in loaded

    def test_synthetic_pipeline_evaluation_results(self, tmp_path):
        """Synthetic pipeline produces valid evaluation results."""
        from src.pipeline import run_pipeline
        output_path = tmp_path / "eval_report.json"
        report = run_pipeline(
            synthetic=True,
            verbose=False,
            output_path=output_path,
            device="cpu",
        )

        eval_phase = report["phases"]["evaluation"]
        assert "overall_ece" in eval_phase
        assert eval_phase["overall_ece"] >= 0.0

    def test_main_cli(self, tmp_path):
        """main() with argv works correctly."""
        from src.pipeline import main
        output_path = tmp_path / "cli_report.json"
        report = main(["--synthetic", "--device", "cpu", "--output", str(output_path)])
        assert output_path.exists()
        assert "phases" in report

    def test_pipeline_memory_profile(self, tmp_path):
        """Pipeline includes memory profiling data."""
        from src.pipeline import run_pipeline
        report = run_pipeline(
            synthetic=True,
            output_path=tmp_path / "mem_report.json",
            device="cpu",
        )
        mem = report["memory_profile"]
        assert "label" in mem
        assert "tensor_delta" in mem
        assert "passed_assertions" in mem

    def test_production_mode_requires_data(self):
        """Non-synthetic mode raises FileNotFoundError without data files."""
        from src.pipeline import run_pipeline
        config = PipelineConfig()
        config.extraction.output_dir = Path("/nonexistent/path")
        with pytest.raises(FileNotFoundError, match="not found"):
            run_pipeline(
                synthetic=False,
                device="cpu",
                config=config,
            )


# =========================================================================
# Tests: Edge cases
# =========================================================================

class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_single_sample_evaluation(self):
        """evaluate_targeted_ece works with N=1."""
        torch.manual_seed(99)
        probs = F.softmax(torch.randn(1, 3, 20, 60), dim=-1)
        gt = torch.topk(torch.randn(1, 3, 20, 60), k=4, dim=-1).indices
        result = evaluate_targeted_ece(probs, gt)
        assert "overall_ece" in result
        assert result["overall_ece"] >= 0.0

    def test_uniform_probabilities(self):
        """Uniform probabilities produce valid ECE."""
        probs = torch.full((10, 3, 20, 60), 1.0 / 60)
        gt = torch.randint(0, 60, (10, 3, 20, 4))
        result = evaluate_targeted_ece(probs, gt)
        assert result["overall_ece"] >= 0.0

    def test_concentrated_probabilities(self):
        """Highly concentrated probabilities (one-hot-ish) produce valid ECE."""
        logits = torch.zeros(10, 3, 20, 60)
        logits[:, :, :, 0] = 100.0  # Very peaked
        probs = F.softmax(logits, dim=-1)
        gt = torch.zeros(10, 3, 20, 4, dtype=torch.long)
        gt[:, :, :, 0] = 0
        gt[:, :, :, 1] = 1
        gt[:, :, :, 2] = 2
        gt[:, :, :, 3] = 3
        result = evaluate_targeted_ece(probs, gt)
        assert result["overall_ece"] >= 0.0

    def test_different_num_experts(self):
        """Works with non-standard number of experts."""
        probs = F.softmax(torch.randn(5, 3, 20, 16), dim=-1)
        gt = torch.randint(0, 16, (5, 3, 20, 4))
        result = evaluate_targeted_ece(probs, gt)
        assert "overall_ece" in result

    def test_bandwidth_zero(self):
        """Bandwidth=0 produces valid (possibly 0) results."""
        probs = F.softmax(torch.randn(5, 3, 20, 60), dim=-1)
        gt = torch.randint(0, 60, (5, 3, 20, 4))
        result = evaluate_targeted_ece(probs, gt, bandwidth=0.0)
        # With 0 bandwidth, very few or no predictions will match exactly
        assert result["overall_ece"] >= 0.0

    def test_memory_profiler_nested(self):
        """Nested memory profilers work correctly."""
        outer = MemoryProfiler(label="outer")
        inner = MemoryProfiler(label="inner")
        with outer:
            with inner:
                _ = torch.randn(50, 50)
            inner_report = inner.report
        outer_report = outer.report
        assert inner_report.label == "inner"
        assert outer_report.label == "outer"
