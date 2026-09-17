"""Unit tests for Milestone 2: MedusaSpeculativeHead model.

Tests cover:
- Instantiation with default and custom parameters
- Forward pass shape correctness for 2-D and 3-D inputs
- Input validation (wrong dim, wrong last dim)
- Checkpoint save/load round-trip
- Parameter counts
- Gradient flow
- Edge cases (batch_size=1, custom expert counts)
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Ensure workspace root is on sys.path
WORKSPACE_ROOT = Path(__file__).parent.parent.resolve()
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.models.medusa_head import MedusaSpeculativeHead
from src.config import (
    SPEC_HEAD_INPUT_DIM,
    NUM_DEEP_LAYERS,
    NUM_EXPERTS,
    NUM_HORIZONS,
    SPEC_HEAD_OUTPUT_DIM_PER_HORIZON,
)


# =====================================================================
# Instantiation Tests
# =====================================================================

class TestMedusaInstantiation:
    """Test MedusaSpeculativeHead construction."""

    def test_default_init(self):
        """Default parameters match config constants."""
        head = MedusaSpeculativeHead()
        assert head.input_dim == SPEC_HEAD_INPUT_DIM
        assert head.num_deep_layers == NUM_DEEP_LAYERS
        assert head.num_experts == NUM_EXPERTS
        assert head.num_horizons == NUM_HORIZONS
        assert head.output_dim_per_horizon == SPEC_HEAD_OUTPUT_DIM_PER_HORIZON

    def test_custom_init(self):
        """Custom dimensions are stored correctly."""
        head = MedusaSpeculativeHead(
            input_dim=64, num_deep_layers=4, num_experts=16, num_horizons=2
        )
        assert head.input_dim == 64
        assert head.num_deep_layers == 4
        assert head.num_experts == 16
        assert head.num_horizons == 2
        assert head.output_dim_per_horizon == 64  # 4 * 16

    def test_three_independent_heads(self):
        """Default config creates exactly 3 independent linear heads."""
        head = MedusaSpeculativeHead()
        assert len(head.heads) == NUM_HORIZONS
        for h in head.heads:
            assert isinstance(h, nn.Linear)
            assert h.in_features == SPEC_HEAD_INPUT_DIM
            assert h.out_features == SPEC_HEAD_OUTPUT_DIM_PER_HORIZON

    def test_invalid_input_dim_raises(self):
        with pytest.raises(ValueError, match="input_dim must be positive"):
            MedusaSpeculativeHead(input_dim=0)

    def test_invalid_num_experts_raises(self):
        with pytest.raises(ValueError, match="num_experts must be positive"):
            MedusaSpeculativeHead(num_experts=-1)

    def test_invalid_num_deep_layers_raises(self):
        with pytest.raises(ValueError, match="num_deep_layers must be positive"):
            MedusaSpeculativeHead(num_deep_layers=0)

    def test_invalid_num_horizons_raises(self):
        with pytest.raises(ValueError, match="num_horizons must be positive"):
            MedusaSpeculativeHead(num_horizons=0)

    def test_parameter_count(self):
        """Verify total parameter count is 3 * (input_dim * output_dim + output_dim)."""
        head = MedusaSpeculativeHead(
            input_dim=64, num_deep_layers=4, num_experts=16, num_horizons=3
        )
        output_dim = 4 * 16  # 64
        expected = 3 * (64 * 64 + 64)  # weights + biases per head
        actual = sum(p.numel() for p in head.parameters())
        assert actual == expected


# =====================================================================
# Forward Pass Tests
# =====================================================================

class TestMedusaForward:
    """Test forward pass shape correctness."""

    @pytest.fixture
    def head(self):
        return MedusaSpeculativeHead(
            input_dim=64, num_deep_layers=4, num_experts=16, num_horizons=3
        )

    def test_2d_input_shape(self, head):
        """(batch, input_dim) -> (batch, 3, 4, 16)."""
        x = torch.randn(8, 64)
        out = head(x)
        assert out.shape == (8, 3, 4, 16)

    def test_3d_input_shape(self, head):
        """(batch, seq_len, input_dim) -> (batch, seq_len, 3, 4, 16)."""
        x = torch.randn(4, 10, 64)
        out = head(x)
        assert out.shape == (4, 10, 3, 4, 16)

    def test_batch_size_one(self, head):
        """Single sample batch works correctly."""
        x = torch.randn(1, 64)
        out = head(x)
        assert out.shape == (1, 3, 4, 16)

    def test_3d_batch_size_one(self, head):
        """3-D with batch=1 and seq_len=1."""
        x = torch.randn(1, 1, 64)
        out = head(x)
        assert out.shape == (1, 1, 3, 4, 16)

    def test_seq_len_one_3d(self, head):
        """3-D with seq_len=1 preserves seq dim."""
        x = torch.randn(4, 1, 64)
        out = head(x)
        assert out.shape == (4, 1, 3, 4, 16)

    def test_full_scale_default_dims(self):
        """Full-scale model with default dims (2048 -> 3, 20, 60)."""
        head = MedusaSpeculativeHead()
        x = torch.randn(2, SPEC_HEAD_INPUT_DIM)
        out = head(x)
        assert out.shape == (2, NUM_HORIZONS, NUM_DEEP_LAYERS, NUM_EXPERTS)

    def test_wrong_last_dim_raises(self, head):
        """Mismatched last dimension raises ValueError."""
        x = torch.randn(4, 32)  # wrong: expecting 64
        with pytest.raises(ValueError, match="Last dimension"):
            head(x)

    def test_1d_input_raises(self, head):
        """1-D input raises ValueError."""
        x = torch.randn(64)
        with pytest.raises(ValueError, match="at least 2-D"):
            head(x)

    def test_output_dtype_matches_input(self, head):
        """Output dtype matches input dtype."""
        x = torch.randn(4, 64, dtype=torch.float32)
        out = head(x)
        assert out.dtype == torch.float32

    def test_independent_heads_produce_different_outputs(self, head):
        """Different horizon heads produce different logits (not identical)."""
        x = torch.randn(4, 64)
        out = head(x)
        # With random init, the 3 heads should produce different outputs
        assert not torch.allclose(out[:, 0, :, :], out[:, 1, :, :], atol=1e-6)
        assert not torch.allclose(out[:, 1, :, :], out[:, 2, :, :], atol=1e-6)


# =====================================================================
# Gradient Flow Tests
# =====================================================================

class TestMedusaGradients:
    """Test that gradients flow through the model."""

    def test_gradient_flow(self):
        head = MedusaSpeculativeHead(
            input_dim=64, num_deep_layers=4, num_experts=16, num_horizons=3
        )
        x = torch.randn(4, 64, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()

        # All parameters should have gradients
        for name, param in head.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_gradient_isolation_between_heads(self):
        """Gradients from one horizon head do not affect the others."""
        head = MedusaSpeculativeHead(
            input_dim=64, num_deep_layers=4, num_experts=16, num_horizons=3
        )
        x = torch.randn(4, 64)
        out = head(x)

        # Only backprop through horizon 0
        loss = out[:, 0, :, :].sum()
        loss.backward()

        # Head 0 should have gradients, head 1 and 2 should not
        assert head.heads[0].weight.grad is not None
        assert head.heads[0].weight.grad.abs().sum() > 0
        # Heads 1 and 2 receive no gradient contribution (their outputs weren't used)
        assert head.heads[1].weight.grad is None or head.heads[1].weight.grad.abs().sum() == 0
        assert head.heads[2].weight.grad is None or head.heads[2].weight.grad.abs().sum() == 0


# =====================================================================
# Checkpoint Tests
# =====================================================================

class TestMedusaCheckpoint:
    """Test save/load round-trip."""

    def test_save_and_load_checkpoint(self, tmp_path):
        """Checkpoint round-trip preserves architecture and weights."""
        head = MedusaSpeculativeHead(
            input_dim=64, num_deep_layers=4, num_experts=16, num_horizons=3
        )
        filepath = str(tmp_path / "test_head.pt")
        head.save_checkpoint(filepath)

        loaded = MedusaSpeculativeHead.from_checkpoint(filepath)
        assert loaded.input_dim == head.input_dim
        assert loaded.num_deep_layers == head.num_deep_layers
        assert loaded.num_experts == head.num_experts
        assert loaded.num_horizons == head.num_horizons

        # Weights should be identical
        x = torch.randn(2, 64)
        with torch.no_grad():
            orig_out = head(x)
            loaded_out = loaded(x)
        assert torch.allclose(orig_out, loaded_out, atol=1e-6)

    def test_get_config(self):
        """get_config returns correct architecture parameters."""
        head = MedusaSpeculativeHead(
            input_dim=128, num_deep_layers=8, num_experts=32, num_horizons=2
        )
        config = head.get_config()
        assert config == {
            "input_dim": 128,
            "num_deep_layers": 8,
            "num_experts": 32,
            "num_horizons": 2,
        }

    def test_checkpoint_creates_parent_dirs(self, tmp_path):
        """save_checkpoint creates parent directories if missing."""
        head = MedusaSpeculativeHead(input_dim=64, num_deep_layers=4, num_experts=16)
        filepath = str(tmp_path / "nested" / "dir" / "head.pt")
        head.save_checkpoint(filepath)
        assert Path(filepath).exists()


# =====================================================================
# Interface Contract Tests (M2 <-> M3)
# =====================================================================

class TestMedusaInterfaceContract:
    """Verify adherence to M2 <-> M3 interface contract from PROJECT.md."""

    def test_contract_2d_output_shape(self):
        """(batch, 2048) -> (batch, 3, 20, 60)."""
        head = MedusaSpeculativeHead()
        x = torch.randn(16, 2048)
        out = head(x)
        assert out.shape == (16, 3, 20, 60)

    def test_contract_3d_output_shape(self):
        """(batch, seq_len, 2048) -> (batch, seq_len, 3, 20, 60)."""
        head = MedusaSpeculativeHead()
        x = torch.randn(4, 8, 2048)
        out = head(x)
        assert out.shape == (4, 8, 3, 20, 60)

    def test_output_is_unscaled_logits(self):
        """Output values are unbounded (not softmaxed)."""
        head = MedusaSpeculativeHead()
        x = torch.randn(8, 2048)
        out = head(x)
        # Logits can be negative and > 1
        assert out.min() < 0 or out.max() > 1  # at least one condition true
