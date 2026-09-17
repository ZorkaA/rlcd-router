"""Unit tests for Milestone 2: MMCE Loss Functions and Trainer.

Tests cover:
- Soft cross-entropy loss computation and shape
- RKHS MMCE penalty computation
- Combined loss function
- CombinedCalibrationLoss module
- Valid mask handling
- Trainer: in-memory training loop, loss decreasing, checkpoint saving
- Trainer: safetensors file-based training
- Edge cases: zero-valid mask, single sample
- Gradient flow through all loss components
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure workspace root is on sys.path
WORKSPACE_ROOT = Path(__file__).parent.parent.resolve()
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.training.mmce_loss import (
    soft_cross_entropy_loss,
    rkhs_mmce_penalty,
    combined_loss,
    CombinedCalibrationLoss,
)
from src.training.trainer import Trainer
from src.models.medusa_head import MedusaSpeculativeHead
from src.config import TrainingConfig


# =====================================================================
# Fixture Helpers
# =====================================================================

@pytest.fixture
def small_logits():
    """Small-scale logits for fast testing: (B=8, H=3, L=4, E=16)."""
    torch.manual_seed(42)
    pred = torch.randn(8, 3, 4, 16)
    target = torch.randn(8, 3, 4, 16)
    return pred, target


@pytest.fixture
def small_valid_mask():
    """Valid mask for 8 samples, 3 horizons, with boundary invalidity."""
    mask = torch.ones(8, 3, dtype=torch.bool)
    mask[-1, :] = torch.tensor([True, False, False])
    mask[-2, :] = torch.tensor([True, True, False])
    return mask


@pytest.fixture
def training_data():
    """Small in-memory training data."""
    torch.manual_seed(42)
    N, D, H, L, E = 64, 64, 3, 4, 16
    hidden_states = torch.randn(N, D)
    target_router_logits = torch.randn(N, H, L, E)
    valid_mask = torch.ones(N, H, dtype=torch.bool)
    return hidden_states, target_router_logits, valid_mask


# =====================================================================
# Soft Cross-Entropy Loss Tests
# =====================================================================

class TestSoftCrossEntropy:
    """Test soft_cross_entropy_loss function."""

    def test_output_is_scalar(self, small_logits):
        pred, target = small_logits
        loss = soft_cross_entropy_loss(pred, target)
        assert loss.dim() == 0

    def test_loss_is_positive(self, small_logits):
        pred, target = small_logits
        loss = soft_cross_entropy_loss(pred, target)
        assert loss.item() > 0

    def test_perfect_prediction_gives_low_loss(self):
        """When pred == target, CE should be minimal (the entropy of the target distribution)."""
        target = torch.randn(4, 3, 4, 16)
        loss_same = soft_cross_entropy_loss(target.clone(), target)
        loss_diff = soft_cross_entropy_loss(torch.randn_like(target), target)
        # Perfect prediction should give lower loss than random
        assert loss_same.item() < loss_diff.item()

    def test_shape_mismatch_raises(self):
        pred = torch.randn(4, 3, 4, 16)
        target = torch.randn(4, 3, 4, 8)  # wrong expert dim
        with pytest.raises(ValueError, match="Shape mismatch"):
            soft_cross_entropy_loss(pred, target)

    def test_wrong_ndim_raises(self):
        pred = torch.randn(4, 16)
        target = torch.randn(4, 16)
        with pytest.raises(ValueError, match="4-D"):
            soft_cross_entropy_loss(pred, target)

    def test_valid_mask_reduces_contribution(self, small_logits, small_valid_mask):
        pred, target = small_logits
        loss_full = soft_cross_entropy_loss(pred, target, valid_mask=None)
        loss_masked = soft_cross_entropy_loss(pred, target, valid_mask=small_valid_mask)
        # Both should be finite, but may differ in value
        assert torch.isfinite(loss_full)
        assert torch.isfinite(loss_masked)

    def test_all_invalid_mask(self):
        """All-zero mask should produce zero loss."""
        pred = torch.randn(4, 3, 4, 16)
        target = torch.randn(4, 3, 4, 16)
        mask = torch.zeros(4, 3, dtype=torch.bool)
        loss = soft_cross_entropy_loss(pred, target, valid_mask=mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_gradient_flows(self, small_logits):
        pred, target = small_logits
        pred.requires_grad_(True)
        loss = soft_cross_entropy_loss(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0

    def test_matches_oracle_simple(self):
        """Verify soft CE matches manual computation for a simple case."""
        # Single sample, single horizon, single layer, 4 experts
        pred = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])   # (1,1,1,4)
        target = torch.tensor([[[[0.5, 1.0, 1.5, 2.0]]]])  # (1,1,1,4)

        loss = soft_cross_entropy_loss(pred, target)

        # Manual computation
        target_probs = F.softmax(target, dim=-1)
        log_probs = F.log_softmax(pred, dim=-1)
        expected = -torch.sum(target_probs * log_probs, dim=-1).mean()

        assert loss.item() == pytest.approx(expected.item(), abs=1e-5)


# =====================================================================
# RKHS MMCE Penalty Tests
# =====================================================================

class TestRKHSMMCE:
    """Test rkhs_mmce_penalty function."""

    def test_output_is_scalar(self, small_logits):
        pred, target = small_logits
        mmce = rkhs_mmce_penalty(pred, target)
        assert mmce.dim() == 0

    def test_mmce_is_non_negative(self, small_logits):
        pred, target = small_logits
        mmce = rkhs_mmce_penalty(pred, target)
        assert mmce.item() >= 0

    def test_mmce_is_finite(self, small_logits):
        pred, target = small_logits
        mmce = rkhs_mmce_penalty(pred, target)
        assert torch.isfinite(mmce)

    def test_shape_mismatch_raises(self):
        pred = torch.randn(4, 3, 4, 16)
        target = torch.randn(4, 3, 4, 8)
        with pytest.raises(ValueError, match="Shape mismatch"):
            rkhs_mmce_penalty(pred, target)

    def test_wrong_ndim_raises(self):
        pred = torch.randn(4, 16)
        target = torch.randn(4, 16)
        with pytest.raises(ValueError, match="4-D"):
            rkhs_mmce_penalty(pred, target)

    def test_with_valid_mask(self, small_logits, small_valid_mask):
        pred, target = small_logits
        mmce = rkhs_mmce_penalty(pred, target, valid_mask=small_valid_mask)
        assert torch.isfinite(mmce)
        assert mmce.item() >= 0

    def test_gradient_flows(self, small_logits):
        pred, target = small_logits
        pred.requires_grad_(True)
        mmce = rkhs_mmce_penalty(pred, target)
        mmce.backward()
        assert pred.grad is not None

    def test_sigma_parameter_affects_output(self, small_logits):
        pred, target = small_logits
        mmce_small_sigma = rkhs_mmce_penalty(pred, target, sigma=0.1)
        mmce_large_sigma = rkhs_mmce_penalty(pred, target, sigma=1.0)
        # Different sigmas should produce different MMCE values
        assert not torch.allclose(mmce_small_sigma, mmce_large_sigma, atol=1e-6)

    def test_perfect_calibration_gives_small_mmce(self):
        """When predictions match targets perfectly, MMCE should be small."""
        target = torch.randn(4, 3, 4, 16)
        mmce = rkhs_mmce_penalty(target.clone(), target)
        # With perfect predictions, MMCE should be close to sqrt(eps)
        assert mmce.item() < 0.1


# =====================================================================
# Combined Loss Tests
# =====================================================================

class TestCombinedLoss:
    """Test combined_loss and CombinedCalibrationLoss."""

    def test_returns_three_tensors(self, small_logits):
        pred, target = small_logits
        total, ce, mmce = combined_loss(pred, target)
        assert total.dim() == 0
        assert ce.dim() == 0
        assert mmce.dim() == 0

    def test_total_equals_ce_plus_lambda_mmce(self, small_logits):
        pred, target = small_logits
        lam = 0.5
        total, ce, mmce = combined_loss(pred, target, mmce_lambda=lam)
        expected = ce + lam * mmce
        assert total.item() == pytest.approx(expected.item(), abs=1e-5)

    def test_lambda_zero_gives_pure_ce(self, small_logits):
        pred, target = small_logits
        total, ce, mmce = combined_loss(pred, target, mmce_lambda=0.0)
        assert total.item() == pytest.approx(ce.item(), abs=1e-5)

    def test_module_wrapper(self, small_logits):
        pred, target = small_logits
        loss_fn = CombinedCalibrationLoss(mmce_lambda=1.0, mmce_sigma=0.2)
        total, ce, mmce = loss_fn(pred, target)
        assert torch.isfinite(total)
        assert torch.isfinite(ce)
        assert torch.isfinite(mmce)

    def test_valid_mask_propagated(self, small_logits, small_valid_mask):
        pred, target = small_logits
        total, ce, mmce = combined_loss(pred, target, valid_mask=small_valid_mask)
        assert torch.isfinite(total)

    def test_gradient_through_combined(self, small_logits):
        pred, target = small_logits
        pred.requires_grad_(True)
        total, ce, mmce = combined_loss(pred, target)
        total.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0


# =====================================================================
# Trainer Tests
# =====================================================================

class TestTrainer:
    """Test the end-to-end Trainer class."""

    def test_in_memory_training_runs(self, training_data):
        """Basic in-memory training completes without error."""
        hs, trl, vm = training_data
        config = TrainingConfig(
            input_dim=64,
            output_dim_per_horizon=4 * 16,
            num_horizons=3,
            learning_rate=1e-3,
            weight_decay=1e-4,
            batch_size=16,
            epochs=2,
            mmce_lambda=0.5,
            mmce_sigma=0.2,
            checkpoint_path=Path(tempfile.mkdtemp()) / "test_head.pt",
        )
        trainer = Trainer(
            config=config,
            device="cpu",
            hidden_states=hs,
            target_router_logits=trl,
            valid_mask=vm,
        )
        history = trainer.train()

        assert len(history) == 2
        for record in history:
            assert "epoch" in record
            assert "ce_loss" in record
            assert "mmce_loss" in record
            assert "total_loss" in record
            assert "time_s" in record
            assert record["ce_loss"] > 0
            assert record["total_loss"] > 0

    def test_loss_decreases(self, training_data):
        """Loss should decrease over multiple epochs (basic learning signal)."""
        hs, trl, vm = training_data
        config = TrainingConfig(
            input_dim=64,
            output_dim_per_horizon=4 * 16,
            num_horizons=3,
            learning_rate=1e-2,
            weight_decay=0.0,
            batch_size=64,
            epochs=10,
            mmce_lambda=0.0,  # Pure CE for cleaner signal
            mmce_sigma=0.2,
            checkpoint_path=Path(tempfile.mkdtemp()) / "test_head.pt",
        )
        trainer = Trainer(
            config=config,
            device="cpu",
            hidden_states=hs,
            target_router_logits=trl,
            valid_mask=vm,
        )
        history = trainer.train()

        # Total loss in last epoch should be less than first epoch
        assert history[-1]["total_loss"] < history[0]["total_loss"]

    def test_checkpoint_saved(self, training_data, tmp_path):
        """Training saves a valid checkpoint file."""
        hs, trl, vm = training_data
        ckpt_path = tmp_path / "test_ckpt.pt"
        config = TrainingConfig(
            input_dim=64,
            output_dim_per_horizon=4 * 16,
            num_horizons=3,
            learning_rate=1e-3,
            batch_size=32,
            epochs=1,
            checkpoint_path=ckpt_path,
        )
        trainer = Trainer(
            config=config,
            device="cpu",
            hidden_states=hs,
            target_router_logits=trl,
            valid_mask=vm,
        )
        trainer.train()

        assert ckpt_path.exists()

        # Load and verify checkpoint
        loaded = MedusaSpeculativeHead.from_checkpoint(str(ckpt_path))
        assert loaded.input_dim == 64
        assert loaded.num_deep_layers == 4
        assert loaded.num_experts == 16
        assert loaded.num_horizons == 3

    def test_get_model_after_train(self, training_data, tmp_path):
        """get_model() returns the trained model."""
        hs, trl, vm = training_data
        config = TrainingConfig(
            input_dim=64,
            output_dim_per_horizon=4 * 16,
            num_horizons=3,
            epochs=1,
            checkpoint_path=tmp_path / "head.pt",
        )
        trainer = Trainer(
            config=config,
            device="cpu",
            hidden_states=hs,
            target_router_logits=trl,
            valid_mask=vm,
        )
        trainer.train()
        model = trainer.get_model()
        assert isinstance(model, MedusaSpeculativeHead)

    def test_get_model_before_train_raises(self):
        """get_model() raises before train() is called."""
        trainer = Trainer(device="cpu")
        with pytest.raises(RuntimeError, match="not available"):
            trainer.get_model()

    def test_no_valid_mask_defaults_to_all_valid(self):
        """Training with valid_mask=None defaults to all valid."""
        torch.manual_seed(42)
        N, D, H, L, E = 32, 64, 3, 4, 16
        hs = torch.randn(N, D)
        trl = torch.randn(N, H, L, E)
        config = TrainingConfig(
            input_dim=64,
            output_dim_per_horizon=4 * 16,
            num_horizons=3,
            epochs=1,
            checkpoint_path=Path(tempfile.mkdtemp()) / "head.pt",
        )
        trainer = Trainer(
            config=config,
            device="cpu",
            hidden_states=hs,
            target_router_logits=trl,
            valid_mask=None,
        )
        history = trainer.train()
        assert len(history) == 1
        assert history[0]["ce_loss"] > 0

    def test_safetensors_training(self, tmp_path):
        """Training from a safetensors file works correctly."""
        torch.manual_seed(42)
        N, D, H, L, E = 32, 64, 3, 4, 16
        hs = torch.randn(N, D)
        trl = torch.randn(N, H, L, E)
        top4 = torch.topk(trl, k=4, dim=-1).indices
        vm = torch.ones(N, H, dtype=torch.bool)

        # Save to safetensors
        from src.data.dataset import save_dataset_safetensors
        sf_path = tmp_path / "train_data.safetensors"
        save_dataset_safetensors(
            {
                "hidden_states": hs,
                "target_router_logits": trl,
                "target_top4_indices": top4,
                "valid_mask": vm,
            },
            sf_path,
        )

        config = TrainingConfig(
            input_dim=64,
            output_dim_per_horizon=4 * 16,
            num_horizons=3,
            epochs=2,
            batch_size=16,
            checkpoint_path=tmp_path / "ckpt.pt",
        )
        trainer = Trainer(
            config=config,
            device="cpu",
            train_data_path=sf_path,
        )
        history = trainer.train()
        assert len(history) == 2
        assert (tmp_path / "ckpt.pt").exists()

    def test_missing_file_raises(self, tmp_path):
        """Training with non-existent file raises FileNotFoundError."""
        config = TrainingConfig(
            epochs=1,
            checkpoint_path=tmp_path / "head.pt",
        )
        trainer = Trainer(
            config=config,
            device="cpu",
            train_data_path=tmp_path / "nonexistent.safetensors",
        )
        with pytest.raises(FileNotFoundError):
            trainer.train()


# =====================================================================
# Edge Case Tests
# =====================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_sample(self):
        """Loss functions work with a single sample."""
        pred = torch.randn(1, 3, 4, 16)
        target = torch.randn(1, 3, 4, 16)
        total, ce, mmce = combined_loss(pred, target)
        assert torch.isfinite(total)

    def test_single_expert(self):
        """Loss functions work with a single expert."""
        pred = torch.randn(4, 3, 4, 1)
        target = torch.randn(4, 3, 4, 1)
        total, ce, mmce = combined_loss(pred, target)
        assert torch.isfinite(total)

    def test_single_horizon(self):
        """Loss functions work with a single horizon."""
        pred = torch.randn(4, 1, 4, 16)
        target = torch.randn(4, 1, 4, 16)
        total, ce, mmce = combined_loss(pred, target)
        assert torch.isfinite(total)

    def test_single_layer(self):
        """Loss functions work with a single deep layer."""
        pred = torch.randn(4, 3, 1, 16)
        target = torch.randn(4, 3, 1, 16)
        total, ce, mmce = combined_loss(pred, target)
        assert torch.isfinite(total)

    def test_large_logit_values(self):
        """Loss is stable with large logit magnitudes."""
        pred = torch.randn(4, 3, 4, 16) * 100
        target = torch.randn(4, 3, 4, 16) * 100
        total, ce, mmce = combined_loss(pred, target)
        assert torch.isfinite(total)

    def test_identical_logits(self):
        """Loss is finite when pred == target."""
        logits = torch.randn(4, 3, 4, 16)
        total, ce, mmce = combined_loss(logits.clone(), logits.clone())
        assert torch.isfinite(total)
        assert torch.isfinite(ce)
        assert torch.isfinite(mmce)
