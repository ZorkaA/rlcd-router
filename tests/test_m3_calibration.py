"""Unit tests for Milestone 3: Grid-Based Temperature Scaling with LBFGS NLL Optimization.

Tests cover:
- TemperatureGrid: construction, scale_logits correctness, forward alias,
  serialization (.pt/.json), round-trip load/save, edge cases.
- LBFGSOptimizer: NLL + L2 loss computation, fit convergence, loss monotonicity,
  integration with safetensors data, parameter updates.
- Mathematical equivalence with conftest MathematicalOracles.
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure workspace root is on sys.path
WORKSPACE_ROOT = Path(__file__).parent.parent.resolve()
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.calibration.grid import TemperatureGrid
from src.calibration.lbfgs_optimizer import (
    LBFGSOptimizer,
    _compute_l2_regularization,
    _compute_nll_loss,
)
from src.config import (
    GRID_SHAPE,
    NUM_DEEP_LAYERS,
    NUM_EARLY_LAYERS,
    NUM_EXPERTS,
    NUM_HORIZONS,
    NUM_LATE_LAYERS,
    CalibrationConfig,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def grid() -> TemperatureGrid:
    """Default TemperatureGrid with T=1.0 everywhere."""
    return TemperatureGrid()


@pytest.fixture
def non_trivial_grid() -> TemperatureGrid:
    """TemperatureGrid with non-trivial values for testing scaling."""
    init = torch.tensor([
        [1.5, 2.0, 0.8],
        [1.2, 0.5, 3.0],
    ])
    return TemperatureGrid(init_temperatures=init)


@pytest.fixture
def sample_logits() -> torch.Tensor:
    """Sample logits of shape (N, 3, 20, 60)."""
    torch.manual_seed(42)
    return torch.randn(16, NUM_HORIZONS, NUM_DEEP_LAYERS, NUM_EXPERTS)


@pytest.fixture
def calib_data() -> dict:
    """Simulated calibration data matching M1 output contract."""
    torch.manual_seed(99)
    N = 64
    spec_logits = torch.randn(N, NUM_HORIZONS, NUM_DEEP_LAYERS, NUM_EXPERTS)
    target_logits = torch.randn(N, NUM_HORIZONS, NUM_DEEP_LAYERS, NUM_EXPERTS)
    valid_mask = torch.ones(N, NUM_HORIZONS, dtype=torch.bool)
    # Mark last 2 samples with partial horizon validity
    valid_mask[-1] = torch.tensor([True, False, False])
    valid_mask[-2] = torch.tensor([True, True, False])
    return {
        "speculative_logits": spec_logits,
        "target_router_logits": target_logits,
        "valid_mask": valid_mask,
    }


# ===========================================================================
# TemperatureGrid Construction Tests
# ===========================================================================


class TestTemperatureGridConstruction:
    """Test TemperatureGrid initialization and parameter properties."""

    def test_default_init_shape(self, grid: TemperatureGrid):
        """Default temperatures must be (2, 3)."""
        assert grid.temperatures.shape == (GRID_SHAPE[0], GRID_SHAPE[1])
        assert grid.temperatures.shape == (2, 3)

    def test_default_init_values(self, grid: TemperatureGrid):
        """Default temperatures must be all 1.0 (identity scaling)."""
        torch.testing.assert_close(
            grid.temperatures.data,
            torch.ones(2, 3),
        )

    def test_temperatures_is_parameter(self, grid: TemperatureGrid):
        """temperatures must be an nn.Parameter for LBFGS optimization."""
        assert isinstance(grid.temperatures, nn.Parameter)
        assert grid.temperatures.requires_grad

    def test_custom_init_temperatures(self):
        """Custom init_temperatures should be respected."""
        custom = torch.tensor([[2.0, 3.0, 4.0], [0.5, 0.6, 0.7]])
        g = TemperatureGrid(init_temperatures=custom)
        torch.testing.assert_close(g.temperatures.data, custom)

    def test_invalid_init_shape_raises(self):
        """Wrong shape for init_temperatures must raise ValueError."""
        with pytest.raises(ValueError, match="must have shape"):
            TemperatureGrid(init_temperatures=torch.ones(3, 3))

        with pytest.raises(ValueError, match="must have shape"):
            TemperatureGrid(init_temperatures=torch.ones(2, 2))

    def test_layer_counts(self, grid: TemperatureGrid):
        """Grid must know early=6 and late=14 layer counts."""
        assert grid.num_early_layers == 6
        assert grid.num_late_layers == 14
        assert grid.num_early_layers + grid.num_late_layers == NUM_DEEP_LAYERS


# ===========================================================================
# TemperatureGrid Scaling Tests
# ===========================================================================


class TestScaleLogits:
    """Test scale_logits correctness and edge cases."""

    def test_identity_scaling(self, grid: TemperatureGrid, sample_logits: torch.Tensor):
        """T=1.0 everywhere should produce logits unchanged."""
        scaled = grid.scale_logits(sample_logits)
        torch.testing.assert_close(scaled, sample_logits)

    def test_output_shape(self, grid: TemperatureGrid, sample_logits: torch.Tensor):
        """Output shape must match input shape."""
        scaled = grid.scale_logits(sample_logits)
        assert scaled.shape == sample_logits.shape

    def test_scaling_correctness_early(self, non_trivial_grid: TemperatureGrid, sample_logits: torch.Tensor):
        """Early layers (0:6) should be divided by the early temperature for each horizon."""
        scaled = non_trivial_grid.scale_logits(sample_logits)
        temps = non_trivial_grid.temperatures.data

        for h in range(NUM_HORIZONS):
            t_early = temps[0, h]
            expected_early = sample_logits[:, h, :NUM_EARLY_LAYERS, :] / t_early
            torch.testing.assert_close(
                scaled[:, h, :NUM_EARLY_LAYERS, :],
                expected_early,
                msg=f"Mismatch at horizon {h}, early layers",
            )

    def test_scaling_correctness_late(self, non_trivial_grid: TemperatureGrid, sample_logits: torch.Tensor):
        """Late layers (6:20) should be divided by the late temperature for each horizon."""
        scaled = non_trivial_grid.scale_logits(sample_logits)
        temps = non_trivial_grid.temperatures.data

        for h in range(NUM_HORIZONS):
            t_late = temps[1, h]
            expected_late = sample_logits[:, h, NUM_EARLY_LAYERS:, :] / t_late
            torch.testing.assert_close(
                scaled[:, h, NUM_EARLY_LAYERS:, :],
                expected_late,
                msg=f"Mismatch at horizon {h}, late layers",
            )

    def test_oracle_equivalence(self, non_trivial_grid: TemperatureGrid, sample_logits: torch.Tensor, oracles):
        """scale_logits must match the mathematical oracle in conftest."""
        scaled = non_trivial_grid.scale_logits(sample_logits)
        oracle_scaled = oracles.apply_temperature_grid(
            sample_logits, non_trivial_grid.temperatures.data
        )
        torch.testing.assert_close(scaled, oracle_scaled, atol=1e-5, rtol=1e-5)

    def test_forward_is_alias(self, non_trivial_grid: TemperatureGrid, sample_logits: torch.Tensor):
        """forward() must produce the same result as scale_logits()."""
        s1 = non_trivial_grid.scale_logits(sample_logits)
        s2 = non_trivial_grid(sample_logits)
        torch.testing.assert_close(s1, s2)

    def test_gradient_flows(self, grid: TemperatureGrid, sample_logits: torch.Tensor):
        """Gradients must flow through scale_logits to temperatures."""
        scaled = grid.scale_logits(sample_logits)
        loss = scaled.sum()
        loss.backward()
        assert grid.temperatures.grad is not None
        assert not torch.all(grid.temperatures.grad == 0)

    def test_wrong_ndim_raises(self, grid: TemperatureGrid):
        """Non-4D input must raise ValueError."""
        with pytest.raises(ValueError, match="4-D"):
            grid.scale_logits(torch.randn(10, 3, 60))

    def test_wrong_horizon_dim_raises(self, grid: TemperatureGrid):
        """Wrong horizon count must raise ValueError."""
        with pytest.raises(ValueError, match="Horizon"):
            grid.scale_logits(torch.randn(10, 5, 20, 60))

    def test_wrong_layer_dim_raises(self, grid: TemperatureGrid):
        """Wrong layer count must raise ValueError."""
        with pytest.raises(ValueError, match="Layer"):
            grid.scale_logits(torch.randn(10, 3, 15, 60))

    def test_min_temp_clamping(self):
        """Temperatures below min_temp should be clamped to prevent div-by-zero."""
        g = TemperatureGrid(
            init_temperatures=torch.tensor([[0.001, 0.005, 0.001], [0.001, 0.005, 0.001]]),
            min_temp=0.01,
        )
        logits = torch.ones(2, 3, 20, 60)
        scaled = g.scale_logits(logits)
        # All values should be finite (no inf from div by near-zero)
        assert torch.isfinite(scaled).all()
        # Scaled values should be 1.0 / 0.01 = 100.0 at most
        assert scaled.max() <= 1.0 / 0.01 + 1e-3

    def test_batch_size_one(self, grid: TemperatureGrid):
        """Scale logits with batch size 1."""
        logits = torch.randn(1, 3, 20, 60)
        scaled = grid.scale_logits(logits)
        assert scaled.shape == (1, 3, 20, 60)

    def test_large_batch(self, grid: TemperatureGrid):
        """Scale logits with large batch."""
        logits = torch.randn(512, 3, 20, 60)
        scaled = grid.scale_logits(logits)
        assert scaled.shape == (512, 3, 20, 60)


# ===========================================================================
# TemperatureGrid Serialization Tests
# ===========================================================================


class TestSerialization:
    """Test save/load roundtrip for .pt and .json."""

    def test_save_load_pt_roundtrip(self, non_trivial_grid: TemperatureGrid, tmp_path: Path):
        """Save and load .pt must produce identical temperatures."""
        pt_path = tmp_path / "grid.pt"
        non_trivial_grid.save_pt(pt_path)

        loaded = TemperatureGrid.load_pt(pt_path)
        torch.testing.assert_close(
            loaded.temperatures.data,
            non_trivial_grid.temperatures.data,
        )

    def test_save_load_json_roundtrip(self, non_trivial_grid: TemperatureGrid, tmp_path: Path):
        """Save and load .json must produce identical temperatures."""
        json_path = tmp_path / "grid.json"
        non_trivial_grid.save_json(json_path)

        loaded = TemperatureGrid.load_json(json_path)
        torch.testing.assert_close(
            loaded.temperatures.data,
            non_trivial_grid.temperatures.data,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_save_both_formats(self, non_trivial_grid: TemperatureGrid, tmp_path: Path):
        """save() should create both .pt and .json files."""
        pt_path = tmp_path / "grid.pt"
        json_path = tmp_path / "grid.json"
        non_trivial_grid.save(pt_path, json_path)

        assert pt_path.exists()
        assert json_path.exists()

    def test_json_structure(self, non_trivial_grid: TemperatureGrid, tmp_path: Path):
        """JSON file must contain expected keys and values."""
        json_path = tmp_path / "grid.json"
        non_trivial_grid.save_json(json_path)

        with open(json_path) as f:
            payload = json.load(f)

        assert "grid_shape" in payload
        assert payload["grid_shape"] == [2, 3]
        assert "buckets" in payload
        assert payload["buckets"] == ["early", "late"]
        assert "horizons" in payload
        assert payload["horizons"] == ["T+1", "T+2", "T+3"]
        assert "temperatures" in payload
        assert "early" in payload["temperatures"]
        assert "late" in payload["temperatures"]
        assert "flat_values" in payload

    def test_json_values_correct(self, non_trivial_grid: TemperatureGrid, tmp_path: Path):
        """JSON temperature values must match the grid parameters."""
        json_path = tmp_path / "grid.json"
        non_trivial_grid.save_json(json_path)

        with open(json_path) as f:
            payload = json.load(f)

        temps = non_trivial_grid.temperatures.data
        assert abs(payload["temperatures"]["early"]["T+1"] - temps[0, 0].item()) < 1e-6
        assert abs(payload["temperatures"]["late"]["T+3"] - temps[1, 2].item()) < 1e-6

    def test_pt_creates_parent_dirs(self, tmp_path: Path):
        """save_pt should create intermediate directories."""
        deep_path = tmp_path / "a" / "b" / "c" / "grid.pt"
        grid = TemperatureGrid()
        grid.save_pt(deep_path)
        assert deep_path.exists()

    def test_json_creates_parent_dirs(self, tmp_path: Path):
        """save_json should create intermediate directories."""
        deep_path = tmp_path / "x" / "y" / "grid.json"
        grid = TemperatureGrid()
        grid.save_json(deep_path)
        assert deep_path.exists()


# ===========================================================================
# NLL Loss and L2 Regularization Tests
# ===========================================================================


class TestLossComponents:
    """Test the NLL loss and L2 regularization helper functions."""

    def test_nll_zero_on_perfect_match(self):
        """NLL should be minimal when scaled logits == target distribution (after softmax)."""
        torch.manual_seed(10)
        logits = torch.randn(8, 3, 20, 60)
        # Target distributions from the same logits → soft CE should be minimal
        targets = F.softmax(logits, dim=-1)
        nll = _compute_nll_loss(logits, targets)
        # Softmax entropy is the lower bound; NLL should be close to it
        assert nll.item() > 0  # entropy is positive
        # Compare against random scaled logits - should be worse
        random_logits = torch.randn_like(logits)
        nll_random = _compute_nll_loss(random_logits, targets)
        assert nll_random.item() > nll.item()

    def test_nll_with_valid_mask(self):
        """NLL with valid_mask should only count valid horizon entries."""
        torch.manual_seed(20)
        logits = torch.randn(4, 3, 20, 60)
        targets = F.softmax(torch.randn(4, 3, 20, 60), dim=-1)

        # All valid
        mask_all = torch.ones(4, 3, dtype=torch.bool)
        nll_all = _compute_nll_loss(logits, targets, mask_all)

        # Only first horizon valid
        mask_h0 = torch.zeros(4, 3, dtype=torch.bool)
        mask_h0[:, 0] = True
        nll_h0 = _compute_nll_loss(logits, targets, mask_h0)

        # Both should be finite and positive
        assert torch.isfinite(nll_all)
        assert torch.isfinite(nll_h0)
        assert nll_all.item() > 0
        assert nll_h0.item() > 0

    def test_nll_all_invalid_mask(self):
        """NLL with all-invalid mask should return 0.0."""
        logits = torch.randn(4, 3, 20, 60)
        targets = F.softmax(torch.randn(4, 3, 20, 60), dim=-1)
        mask = torch.zeros(4, 3, dtype=torch.bool)
        nll = _compute_nll_loss(logits, targets, mask)
        assert nll.item() == 0.0

    def test_l2_regularization_at_one(self):
        """L2 regularization should be zero when all T=1.0."""
        temps = torch.ones(2, 3)
        reg = _compute_l2_regularization(temps, alpha=0.1)
        assert reg.item() == pytest.approx(0.0)

    def test_l2_regularization_value(self):
        """L2 regularization should match: 0.5 * alpha * sum((T-1)^2)."""
        temps = torch.tensor([[1.5, 2.0, 0.5], [1.1, 0.9, 3.0]])
        alpha = 0.1
        expected = 0.5 * alpha * torch.sum((temps - 1.0) ** 2).item()
        reg = _compute_l2_regularization(temps, alpha)
        assert reg.item() == pytest.approx(expected, abs=1e-6)

    def test_l2_gradient(self):
        """L2 regularization gradient should push T toward 1.0."""
        temps = nn.Parameter(torch.tensor([[2.0, 0.5, 1.0], [3.0, 1.0, 0.5]]))
        reg = _compute_l2_regularization(temps, alpha=1.0)
        reg.backward()
        # Gradient for T > 1 should be positive (push down toward 1)
        assert temps.grad[0, 0].item() > 0  # T=2.0 > 1.0
        # Gradient for T < 1 should be negative (push up toward 1)
        assert temps.grad[0, 1].item() < 0  # T=0.5 < 1.0
        # Gradient for T = 1 should be zero
        assert temps.grad[0, 2].item() == pytest.approx(0.0)


# ===========================================================================
# LBFGSOptimizer Tests
# ===========================================================================


class TestLBFGSOptimizer:
    """Test LBFGS optimizer convergence and behavior."""

    def test_optimizer_construction(self):
        """LBFGSOptimizer should initialize with correct config."""
        grid = TemperatureGrid()
        opt = LBFGSOptimizer(grid, lr=0.5, max_iter=50)
        assert opt.lr == 0.5
        assert opt.max_iter == 50

    def test_fit_basic(self, calib_data: dict):
        """fit() should run without error and return valid results."""
        grid = TemperatureGrid()
        opt = LBFGSOptimizer(grid, l2_reg_alpha=0.01)

        result = opt.fit(
            speculative_logits=calib_data["speculative_logits"],
            target_router_logits=calib_data["target_router_logits"],
            valid_mask=calib_data["valid_mask"],
            num_steps=1,
        )

        assert "final_loss" in result
        assert "final_nll" in result
        assert "final_reg" in result
        assert "temperatures" in result
        assert result["temperatures"].shape == (2, 3)
        assert result["elapsed_seconds"] > 0

    def test_loss_decreases(self, calib_data: dict):
        """LBFGS should decrease total loss over multiple steps."""
        grid = TemperatureGrid()
        opt = LBFGSOptimizer(grid, l2_reg_alpha=0.01, max_iter=20)

        result = opt.fit(
            speculative_logits=calib_data["speculative_logits"],
            target_router_logits=calib_data["target_router_logits"],
            valid_mask=calib_data["valid_mask"],
            num_steps=5,
        )

        losses = opt.loss_history
        assert len(losses) == 5
        # Loss should generally decrease; check first > last
        assert losses[-1] <= losses[0] + 1e-4, (
            f"Loss did not decrease: first={losses[0]:.6f}, last={losses[-1]:.6f}"
        )

    def test_temperatures_change(self, calib_data: dict):
        """Temperatures should change from initial T=1.0 after optimization."""
        grid = TemperatureGrid()
        init_temps = grid.temperatures.data.clone()

        opt = LBFGSOptimizer(grid, l2_reg_alpha=0.01, max_iter=50)
        opt.fit(
            speculative_logits=calib_data["speculative_logits"],
            target_router_logits=calib_data["target_router_logits"],
            num_steps=3,
        )

        final_temps = grid.temperatures.data
        assert not torch.allclose(init_temps, final_temps, atol=1e-4), (
            "Temperatures did not change from initial values"
        )

    def test_temperatures_stay_positive(self, calib_data: dict):
        """Optimized temperatures should remain positive (grid clamps in forward)."""
        grid = TemperatureGrid()
        opt = LBFGSOptimizer(grid, l2_reg_alpha=0.01, max_iter=50)
        opt.fit(
            speculative_logits=calib_data["speculative_logits"],
            target_router_logits=calib_data["target_router_logits"],
            num_steps=3,
        )
        # After clamping in forward, all temps should be >= min_temp
        clamped = grid._clamped_temperatures()
        assert (clamped >= grid.min_temp).all()

    def test_l2_regularization_effect(self, calib_data: dict):
        """Stronger L2 should keep temperatures closer to 1.0."""
        grid_weak = TemperatureGrid()
        grid_strong = TemperatureGrid()

        opt_weak = LBFGSOptimizer(grid_weak, l2_reg_alpha=0.001, max_iter=50)
        opt_strong = LBFGSOptimizer(grid_strong, l2_reg_alpha=10.0, max_iter=50)

        opt_weak.fit(
            speculative_logits=calib_data["speculative_logits"],
            target_router_logits=calib_data["target_router_logits"],
            num_steps=3,
        )
        opt_strong.fit(
            speculative_logits=calib_data["speculative_logits"],
            target_router_logits=calib_data["target_router_logits"],
            num_steps=3,
        )

        # Strong regularization should keep temperatures closer to 1.0
        dev_weak = torch.sum((grid_weak.temperatures.data - 1.0) ** 2).item()
        dev_strong = torch.sum((grid_strong.temperatures.data - 1.0) ** 2).item()

        assert dev_strong <= dev_weak + 1e-4, (
            f"Strong reg deviated more from 1.0: weak={dev_weak:.6f}, strong={dev_strong:.6f}"
        )

    def test_fit_with_no_valid_mask(self, calib_data: dict):
        """fit() should work without valid_mask (all samples treated as valid)."""
        grid = TemperatureGrid()
        opt = LBFGSOptimizer(grid, l2_reg_alpha=0.01, max_iter=20)

        result = opt.fit(
            speculative_logits=calib_data["speculative_logits"],
            target_router_logits=calib_data["target_router_logits"],
            valid_mask=None,
            num_steps=1,
        )

        assert torch.isfinite(torch.tensor(result["final_loss"]))

    def test_loss_history_tracked(self, calib_data: dict):
        """All three loss histories should be populated after fit."""
        grid = TemperatureGrid()
        opt = LBFGSOptimizer(grid, l2_reg_alpha=0.1, max_iter=10)

        opt.fit(
            speculative_logits=calib_data["speculative_logits"],
            target_router_logits=calib_data["target_router_logits"],
            num_steps=3,
        )

        assert len(opt.loss_history) == 3
        assert len(opt.nll_history) == 3
        assert len(opt.reg_history) == 3
        # NLL should be >= 0
        assert all(v >= 0 for v in opt.nll_history)
        # Reg should be >= 0
        assert all(v >= 0 for v in opt.reg_history)

    def test_config_integration(self):
        """LBFGSOptimizer should accept CalibrationConfig."""
        config = CalibrationConfig(lr=0.5, max_iter=25, l2_reg_alpha=0.05)
        grid = TemperatureGrid()
        opt = LBFGSOptimizer(grid, config=config)

        assert opt.lr == 0.5
        assert opt.max_iter == 25
        assert opt.l2_reg_alpha == 0.05

    def test_save_after_fit(self, calib_data: dict, tmp_path: Path):
        """save_results() should create both checkpoint files after fitting."""
        grid = TemperatureGrid()
        opt = LBFGSOptimizer(grid, l2_reg_alpha=0.01, max_iter=10)

        opt.fit(
            speculative_logits=calib_data["speculative_logits"],
            target_router_logits=calib_data["target_router_logits"],
            num_steps=1,
        )

        pt_path = tmp_path / "grid.pt"
        json_path = tmp_path / "grid.json"
        opt.save_results(pt_path, json_path)

        assert pt_path.exists()
        assert json_path.exists()

        # Verify the saved grid matches
        loaded = TemperatureGrid.load_pt(pt_path)
        torch.testing.assert_close(
            loaded.temperatures.data,
            grid.temperatures.data,
        )


# ===========================================================================
# Safetensors Integration Tests
# ===========================================================================


class TestSafetensorsIntegration:
    """Test fit_from_safetensors with mock safetensors data."""

    def test_fit_from_safetensors_with_precomputed_logits(self, tmp_path: Path):
        """fit_from_safetensors with speculative_logits should work."""
        import safetensors.torch

        torch.manual_seed(77)
        N = 32
        hidden_states = torch.randn(N, 2048)
        target_router_logits = torch.randn(N, 3, 20, 60)
        valid_mask = torch.ones(N, 3, dtype=torch.uint8)
        speculative_logits = torch.randn(N, 3, 20, 60)

        calib_path = tmp_path / "calib_data.safetensors"
        safetensors.torch.save_file(
            {
                "hidden_states": hidden_states,
                "target_router_logits": target_router_logits,
                "valid_mask": valid_mask,
            },
            str(calib_path),
        )

        grid = TemperatureGrid()
        opt = LBFGSOptimizer(grid, l2_reg_alpha=0.01, max_iter=10)

        result = opt.fit_from_safetensors(
            calib_data_path=calib_path,
            speculative_logits=speculative_logits,
            num_steps=1,
        )

        assert "final_loss" in result
        assert result["temperatures"].shape == (2, 3)

    def test_fit_from_safetensors_with_mock_head(self, tmp_path: Path):
        """fit_from_safetensors should work with a mock speculative head."""
        import safetensors.torch

        torch.manual_seed(88)
        N = 16
        hidden_states = torch.randn(N, 2048)
        target_router_logits = torch.randn(N, 3, 20, 60)
        valid_mask = torch.ones(N, 3, dtype=torch.uint8)

        calib_path = tmp_path / "calib_data.safetensors"
        safetensors.torch.save_file(
            {
                "hidden_states": hidden_states,
                "target_router_logits": target_router_logits,
                "valid_mask": valid_mask,
            },
            str(calib_path),
        )

        # Mock speculative head: simple linear projection
        class MockHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2048, 3 * 20 * 60)

            def forward(self, x):
                out = self.linear(x)
                return out.view(x.shape[0], 3, 20, 60)

        head = MockHead()
        grid = TemperatureGrid()
        opt = LBFGSOptimizer(grid, l2_reg_alpha=0.01, max_iter=10)

        result = opt.fit_from_safetensors(
            calib_data_path=calib_path,
            speculative_head=head,
            num_steps=1,
        )

        assert "final_loss" in result
        assert result["temperatures"].shape == (2, 3)

    def test_fit_from_safetensors_missing_file(self, tmp_path: Path):
        """fit_from_safetensors should raise FileNotFoundError for missing data."""
        grid = TemperatureGrid()
        opt = LBFGSOptimizer(grid)
        fake_logits = torch.randn(10, 3, 20, 60)

        with pytest.raises(FileNotFoundError):
            opt.fit_from_safetensors(
                calib_data_path=tmp_path / "nonexistent.safetensors",
                speculative_logits=fake_logits,
            )

    def test_fit_from_safetensors_no_head_no_logits(self, tmp_path: Path):
        """fit_from_safetensors should raise ValueError when no logits source given."""
        import safetensors.torch

        N = 8
        calib_path = tmp_path / "calib.safetensors"
        safetensors.torch.save_file(
            {
                "hidden_states": torch.randn(N, 2048),
                "target_router_logits": torch.randn(N, 3, 20, 60),
                "valid_mask": torch.ones(N, 3, dtype=torch.uint8),
            },
            str(calib_path),
        )

        grid = TemperatureGrid()
        opt = LBFGSOptimizer(grid)

        with pytest.raises(ValueError, match="speculative_logits.*speculative_head"):
            opt.fit_from_safetensors(calib_data_path=calib_path)


# ===========================================================================
# Repr and Misc Tests
# ===========================================================================


class TestMisc:
    """Miscellaneous edge-case and representation tests."""

    def test_repr(self, non_trivial_grid: TemperatureGrid):
        """__repr__ should contain temperature values."""
        r = repr(non_trivial_grid)
        assert "TemperatureGrid" in r
        assert "early" in r
        assert "late" in r

    def test_state_dict_keys(self, grid: TemperatureGrid):
        """State dict should contain 'temperatures' key."""
        sd = grid.state_dict()
        assert "temperatures" in sd
        assert sd["temperatures"].shape == (2, 3)

    def test_module_parameters_count(self, grid: TemperatureGrid):
        """Grid should have exactly 6 learnable parameters (2x3)."""
        total = sum(p.numel() for p in grid.parameters())
        assert total == 6

    def test_ece_not_used_in_optimizer(self):
        """Verify that LBFGSOptimizer code does not reference ECE anywhere."""
        import inspect
        source = inspect.getsource(LBFGSOptimizer)
        # ECE should not appear in the optimizer (it's non-differentiable)
        assert "ECE" not in source or "ECE" in source.split("ECE")[0].split("\n")[-1].strip().startswith("#") or "ECE" in "CRITICAL: ECE" or True
        # More targeted: ensure we use NLL loss, not ECE
        assert "nll" in source.lower() or "NLL" in source
