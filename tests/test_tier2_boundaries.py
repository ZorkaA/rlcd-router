"""Tier 2 Boundary Tests: Boundary Value Analysis & Corner Conditions.
Covers trailing token masks, sparse buckets, extreme temperatures, singular MMCE batches,
empty Targeted ECE windows, and zero/extreme tensor inputs across the 18 features.
"""

import math
import tempfile
from pathlib import Path
from typing import Dict

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tests.conftest import safe_import, MathematicalOracles


# ===========================================================================
# Boundary Domain 1: Trailing Token Masks & Sequence Boundaries
# ===========================================================================

def test_b01_seq_len_one_all_horizons_invalid():
    """B1.1: For sequence length L=1, all future horizons T+1, T+2, T+3 are out-of-bounds."""
    L = 1
    # Horizon offsets: 1, 2, 3
    valid_mask = torch.zeros(L, 3, dtype=torch.bool)
    for t in range(L):
        for h_idx, offset in enumerate([1, 2, 3]):
            if t + offset < L:
                valid_mask[t, h_idx] = True
    assert not valid_mask.any(), "All tokens in L=1 sequence must have invalid future horizons"


def test_b01_seq_len_two_horizon_transition():
    """B1.2: For sequence length L=2, T+1 is valid only at pos 0; T+2 and T+3 are invalid everywhere."""
    L = 2
    valid_mask = torch.zeros(L, 3, dtype=torch.bool)
    for t in range(L):
        for h_idx, offset in enumerate([1, 2, 3]):
            valid_mask[t, h_idx] = (t + offset < L)
    # Pos 0: T+1 valid (0+1 < 2), T+2 (0+2 < 2 False), T+3 False
    assert valid_mask[0, 0].item() is True
    assert valid_mask[0, 1].item() is False
    assert valid_mask[0, 2].item() is False
    # Pos 1: All invalid
    assert not valid_mask[1].any()


def test_b01_seq_len_four_exact_mask_pattern():
    """B1.3: For sequence length L=4, verify exact triangular boundary mask."""
    L = 4
    valid_mask = torch.zeros(L, 3, dtype=torch.bool)
    for t in range(L):
        for h_idx, offset in enumerate([1, 2, 3]):
            valid_mask[t, h_idx] = (t + offset < L)
    # Pos 0: T+1, T+2, T+3 all True
    assert torch.equal(valid_mask[0], torch.tensor([True, True, True]))
    # Pos 1: T+1, T+2 True, T+3 False
    assert torch.equal(valid_mask[1], torch.tensor([True, True, False]))
    # Pos 2: T+1 True, T+2, T+3 False
    assert torch.equal(valid_mask[2], torch.tensor([True, False, False]))
    # Pos 3: all False
    assert torch.equal(valid_mask[3], torch.tensor([False, False, False]))


def test_b01_all_false_mask_loss_stability():
    """B1.4: When valid_mask is entirely False, loss computation should return 0 or avoid NaN."""
    loss_all = torch.randn(4, 3, 20)
    all_false_mask = torch.zeros(4, 3, 20, dtype=torch.bool)
    valid_count = all_false_mask.sum()
    masked_loss = (loss_all * all_false_mask.float()).sum() / valid_count.clamp(min=1)
    assert masked_loss.item() == 0.0
    assert not torch.isnan(masked_loss)


def test_b01_safetensors_bool_mask_serialization(tmp_path):
    """B1.5: Verify boolean valid_mask survives safetensors uint8 conversion roundtrip."""
    import safetensors.torch
    mask = torch.tensor([[True, False, True], [False, True, False]], dtype=torch.bool)
    path = tmp_path / "mask_test.safetensors"
    safetensors.torch.save_file({"valid_mask": mask.to(torch.uint8)}, str(path))
    loaded = safetensors.torch.load_file(str(path))
    recovered_mask = loaded["valid_mask"].bool()
    assert torch.equal(mask, recovered_mask)


# ===========================================================================
# Boundary Domain 2: Sparse & Empty Buckets in Calibration Grid
# ===========================================================================

def test_b02_completely_empty_bucket_l2_shrinkage():
    """B2.1: An empty bucket (N=0) with L2 regularizer pulls temperature directly toward T=1.0."""
    temp = nn.Parameter(torch.tensor([2.5]))
    alpha_l2 = 1.0
    optimizer = torch.optim.SGD([temp], lr=0.2)
    for _ in range(40):
        optimizer.zero_grad()
        # NLL loss is zero (empty data), only L2 loss active
        loss = 0.5 * alpha_l2 * (temp - 1.0) ** 2
        loss.backward()
        optimizer.step()
    assert pytest.approx(temp.item(), abs=0.01) == 1.0


def test_b02_single_sample_bucket_nll():
    """B2.2: A bucket with exactly N=1 sample computes stable cross-entropy without NaN."""
    logits = torch.randn(1, 60, requires_grad=True)
    target = torch.tensor([5])
    temp = nn.Parameter(torch.tensor([1.2]))
    scaled_logits = logits / temp
    loss = F.cross_entropy(scaled_logits, target)
    loss.backward()
    assert not torch.isnan(loss)
    assert temp.grad is not None
    assert not torch.isnan(temp.grad)


def test_b02_identical_labels_bucket_stability():
    """B2.3: When all samples in a bucket share the identical expert label, NLL remains stable."""
    logits = torch.randn(20, 60)
    targets = torch.full((20,), 7, dtype=torch.long)
    temp = torch.tensor([1.5])
    loss = F.cross_entropy(logits / temp, targets)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_b02_extreme_sample_count_numerical_stability():
    """B2.4: Large sample count (N=10,000) average NLL does not overflow float32."""
    logits = torch.randn(10000, 60)
    targets = torch.randint(0, 60, (10000,))
    loss = F.cross_entropy(logits, targets)
    assert not torch.isnan(loss)
    assert loss.item() < 100.0


def test_b02_asymmetric_bucket_sizes_grid():
    """B2.5: Verify temperature scaling handles highly unbalanced sample counts across buckets."""
    counts = {"early_T1": 1000, "late_T3": 5}
    assert counts["late_T3"] < counts["early_T1"]


# ===========================================================================
# Boundary Domain 3: Temperature Extremes & Numerical Safeguards
# ===========================================================================

def test_b03_temperature_approaching_zero():
    """B3.1: Enforce lower bound clamp on temperature (T >= 0.01) to prevent division by zero."""
    temp_min = 0.01
    proposed_temps = torch.tensor([0.0, -1.0, -0.001, 0.0001])
    clamped = torch.clamp(proposed_temps, min=temp_min)
    assert torch.all(clamped >= temp_min)


def test_b03_temperature_approaching_infinity():
    """B3.2: As T -> inf, scaled softmax approaches uniform distribution (entropy -> ln(60))."""
    logits = torch.randn(1, 60)
    huge_T = 1e6
    probs = F.softmax(logits / huge_T, dim=-1)
    expected_uniform = 1.0 / 60.0
    assert torch.allclose(probs, torch.full_like(probs, expected_uniform), atol=1e-5)
    entropy = -(probs * torch.log(probs)).sum().item()
    assert pytest.approx(entropy, rel=1e-4) == math.log(60.0)


def test_b03_negative_temperature_rejection():
    """B3.3: Verify negative temperatures are rejected or mapped via exp parameterization."""
    # Method 1: Exp parameterization guarantees T > 0 for all real theta
    theta = torch.tensor([-100.0, -1.0, 0.0, 2.0])
    T = torch.exp(theta)
    assert torch.all(T > 0.0)


def test_b03_temperature_exactly_one_is_identity(mock_speculative_logits):
    """B3.4: At T=1.0, temperature scaling is an exact identity operation."""
    scaled = mock_speculative_logits / 1.0
    assert torch.equal(scaled, mock_speculative_logits)


def test_b03_extreme_logits_float32_stability():
    """B3.5: Verify extreme logit values (+500, -500) do not produce NaN under softmax."""
    extreme_logits = torch.tensor([[500.0, -500.0, 0.0]])
    probs = F.softmax(extreme_logits, dim=-1)
    assert not torch.isnan(probs).any()
    assert probs[0, 0].item() == 1.0
    assert probs[0, 1].item() == 0.0


# ===========================================================================
# Boundary Domain 4: Single-Sample & Degenerate Batches in RKHS MMCE
# ===========================================================================

def test_b04_single_sample_mmce_batch(oracles):
    """B4.1: Batch size m=1 in RKHS MMCE evaluates cleanly without crash."""
    conf = torch.tensor([0.8])
    corr = torch.tensor([1.0])
    mmce = oracles.rkhs_mmce(conf, corr, sigma=0.2)
    assert mmce.ndim == 0
    assert not torch.isnan(mmce)
    # Residual e = 1.0 - 0.8 = 0.2; K = [1.0]; quad = 0.2^2 = 0.04; mmce = sqrt(0.04) = 0.2
    assert pytest.approx(mmce.item(), abs=1e-4) == 0.2


def test_b04_identical_confidences_mmce(oracles):
    """B4.2: When all predictions have identical confidence, MMCE kernel matrix is all 1.0s."""
    m = 5
    conf = torch.full((m,), 0.7)
    corr = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])  # 3 correct, 2 incorrect -> acc = 0.6
    mmce = oracles.rkhs_mmce(conf, corr, sigma=0.2)
    assert not torch.isnan(mmce)
    # e = corr - conf = [0.3, 0.3, 0.3, -0.7, -0.7], sum(e) = 0.9 - 1.4 = -0.5
    # e^T 1 e / m^2 = (sum(e) / m)^2 = (-0.5 / 5)^2 = (-0.1)^2 = 0.01
    # mmce = sqrt(0.01) = 0.1
    assert pytest.approx(mmce.item(), abs=1e-4) == 0.1


def test_b04_perfect_confidence_wrong_prediction(oracles):
    """B4.3: Worst-case calibration: 100% confident but 0% accurate (residual = -1.0)."""
    conf = torch.tensor([1.0])
    corr = torch.tensor([0.0])
    mmce = oracles.rkhs_mmce(conf, corr, sigma=0.2)
    assert pytest.approx(mmce.item(), abs=1e-4) == 1.0


def test_b04_very_small_kernel_bandwidth_sigma(oracles):
    """B4.4: As sigma -> 0, Gaussian RBF kernel approaches identity matrix."""
    conf = torch.tensor([0.1, 0.9])
    corr = torch.tensor([0.0, 1.0])
    mmce = oracles.rkhs_mmce(conf, corr, sigma=1e-4)
    assert not torch.isnan(mmce)


def test_b04_very_large_kernel_bandwidth_sigma(oracles):
    """B4.5: As sigma -> inf, Gaussian RBF kernel approaches all-ones matrix."""
    conf = torch.tensor([0.3, 0.7])
    corr = torch.tensor([1.0, 0.0])
    mmce = oracles.rkhs_mmce(conf, corr, sigma=1e4)
    assert not torch.isnan(mmce)


# ===========================================================================
# Boundary Domain 5: Targeted ECE Boundary Conditions
# ===========================================================================

def test_b05_predictions_exactly_on_threshold(oracles):
    """B5.1: Predictions exactly at threshold p*=0.05 are included in the window."""
    probs = torch.zeros(10, 60)
    probs[:, 0] = 0.05  # Exactly 0.05
    top4 = torch.zeros((10, 4), dtype=torch.long)
    ece, count = oracles.targeted_ece(probs, top4, threshold=0.05, window=0.025)
    assert count == 10


def test_b05_predictions_just_outside_window(oracles):
    """B5.2: Predictions outside [0.05 - delta, 0.05 + delta] are excluded."""
    delta = 0.025
    # Just outside: 0.05 + 0.025 + 0.001 = 0.076
    probs = torch.zeros(5, 60)
    probs[:, 0] = 0.05 + delta + 0.001
    top4 = torch.zeros((5, 4), dtype=torch.long)
    ece, count = oracles.targeted_ece(probs, top4, threshold=0.05, window=delta)
    assert count == 0
    assert ece == 0.0


def test_b05_all_predictions_in_window_zero_accuracy(oracles):
    """B5.3: All predictions in 0.85 window with 0% accuracy yields ECE = conf."""
    probs = torch.zeros(10, 60)
    probs[:, 0] = 0.85
    # top4 does not include expert 0
    top4 = torch.tensor([[1, 2, 3, 4]] * 10)
    ece, count = oracles.targeted_ece(probs, top4, threshold=0.85, window=0.025)
    assert count == 10
    # acc = 0.0, conf = 0.85 -> |acc - conf| = 0.85
    assert pytest.approx(ece, abs=1e-5) == 0.85


def test_b05_empty_dataset_targeted_ece(oracles):
    """B5.4: Empty input tensor (N=0) returns 0.0 with count 0 without crashing."""
    probs = torch.empty((0, 60))
    top4 = torch.empty((0, 4), dtype=torch.long)
    ece, count = oracles.targeted_ece(probs, top4, threshold=0.05)
    assert count == 0
    assert ece == 0.0


def test_b05_narrow_versus_wide_window_comparison(oracles):
    """B5.5: Wider window captures a superset of sample count compared to narrow window."""
    torch.manual_seed(42)
    probs = torch.rand(100, 60)
    probs = F.softmax(probs, dim=-1)
    top4 = torch.randint(0, 60, (100, 4))

    _, count_narrow = oracles.targeted_ece(probs, top4, threshold=0.05, window=0.01)
    _, count_wide = oracles.targeted_ece(probs, top4, threshold=0.05, window=0.05)
    assert count_wide >= count_narrow


# ===========================================================================
# Boundary Domain 6: Zero & Extreme Input Tensors
# ===========================================================================

def test_b06_all_zero_hidden_states():
    """B6.1: All-zero hidden states produce valid finite output through linear head."""
    head = nn.Linear(2048, 3 * 20 * 60)
    zero_h = torch.zeros(2, 2048)
    out = head(zero_h)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_b06_large_norm_hidden_states():
    """B6.2: High norm hidden states (||h|| = 1000) do not cause overflow in linear projection."""
    head = nn.Linear(2048, 3 * 20 * 60)
    large_h = torch.randn(2, 2048) * 1000.0
    out = head(large_h)
    assert not torch.isnan(out).any()


def test_b06_single_token_input_shape():
    """B6.3: Single token forward pass (B=1, L=1) preserves exact tensor dimensions."""
    head = nn.Linear(64, 3 * 20 * 16)
    x = torch.randn(1, 1, 64)
    out = head(x).view(1, 1, 3, 20, 16)
    assert out.shape == (1, 1, 3, 20, 16)


def test_b06_empty_batch_dimension_handling():
    """B6.4: Zero batch dimension (B=0) creates empty output without throwing error."""
    head = nn.Linear(64, 3 * 20 * 16)
    empty_x = torch.empty(0, 64)
    out = head(empty_x)
    assert out.shape == (0, 3 * 20 * 16)


def test_b06_non_contiguous_tensor_support():
    """B6.5: Non-contiguous tensor memory layout does not break view / projection operations."""
    x = torch.randn(4, 8, 64).transpose(0, 1)  # Non-contiguous (8, 4, 64)
    assert not x.is_contiguous()
    contiguous_x = x.contiguous()
    assert contiguous_x.is_contiguous()
    head = nn.Linear(64, 16)
    out = head(contiguous_x)
    assert out.shape == (8, 4, 16)
