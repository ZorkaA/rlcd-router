"""Tier 3 Pairwise Combinatorial Tests: Cross-Feature Interactions.
Covers layer buckets x horizons (2x3 grid), CE + MMCE loss formulations,
NLL + L2 regularization across sparsity regimes, and device/precision pairs.
"""

import math
from typing import Dict, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tests.conftest import safe_import, MathematicalOracles


# ===========================================================================
# Pairwise Set 1: Layer Buckets x Lookahead Horizons (2 x 3 Grid Cells)
# ===========================================================================

@pytest.mark.parametrize("layer_bucket,layer_idx_range", [
    ("early", range(0, 6)),    # Deep layers 0..5 (Layers 5-10)
    ("late", range(6, 20)),    # Deep layers 6..19 (Layers 11-24)
])
@pytest.mark.parametrize("horizon_idx,horizon_name", [
    (0, "T+1"),
    (1, "T+2"),
    (2, "T+3"),
])
def test_p01_grid_cell_scaling_isolation(
    oracles,
    mock_speculative_logits,
    layer_bucket,
    layer_idx_range,
    horizon_idx,
    horizon_name,
):
    """P1.1: Verify scaling logits for a specific cell (b, h) modifies only that slice."""
    b_idx = 0 if layer_bucket == "early" else 1
    # Initialize all temperatures to 1.0 except the selected cell
    temps = torch.ones(2, 3)
    test_temp = 2.0
    temps[b_idx, horizon_idx] = test_temp

    scaled = oracles.apply_temperature_grid(mock_speculative_logits, temps)

    # The selected slice should be scaled by test_temp
    selected_slice = scaled[:, horizon_idx, layer_idx_range, :]
    orig_slice = mock_speculative_logits[:, horizon_idx, layer_idx_range, :]
    assert torch.allclose(selected_slice, orig_slice / test_temp)

    # All other horizons should remain exactly unscaled
    for h in range(3):
        if h != horizon_idx:
            assert torch.equal(scaled[:, h], mock_speculative_logits[:, h])


@pytest.mark.parametrize("b_idx,b_name", [(0, "early"), (1, "late")])
@pytest.mark.parametrize("h_idx,h_name", [(0, "T+1"), (1, "T+2"), (2, "T+3")])
def test_p01_gradient_isolation_per_grid_cell(b_idx, b_name, h_idx, h_name):
    """P1.2: Verify loss gradient on T[b, h] propagates strictly from that cell's logits."""
    temps = nn.Parameter(torch.ones(2, 3))
    logits = torch.randn(4, 3, 20, 60)

    # Build scaled logits out-of-place to preserve autograd graph
    slices_h = []
    for h in range(3):
        early = logits[:, h, :6, :] / temps[0, h]
        late = logits[:, h, 6:, :] / temps[1, h]
        slices_h.append(torch.cat([early, late], dim=1))
    scaled = torch.stack(slices_h, dim=1)

    # Compute loss solely on (b_idx, h_idx) slice
    layer_slice = slice(0, 6) if b_idx == 0 else slice(6, 20)
    target_slice = scaled[:, h_idx, layer_slice, :]
    loss = target_slice.sum()
    loss.backward()

    # temps.grad must be non-zero at [b_idx, h_idx] and exactly zero elsewhere
    for r in range(2):
        for c in range(3):
            if r == b_idx and c == h_idx:
                assert temps.grad[r, c].item() != 0.0
            else:
                assert temps.grad[r, c].item() == 0.0


# ===========================================================================
# Pairwise Set 2: Cross-Entropy Loss x RKHS MMCE Penalty
# ===========================================================================

@pytest.mark.parametrize("lambda_mmce", [0.0, 0.1, 0.5, 1.0, 5.0])
@pytest.mark.parametrize("sigma", [0.05, 0.2, 1.0])
def test_p02_composite_loss_lambda_sigma_interaction(oracles, lambda_mmce, sigma):
    """P2.1: Verify total loss L = CE + lambda * MMCE across lambdas and kernel bandwidths."""
    torch.manual_seed(42)
    logits = torch.randn(20, 60)
    probs = F.softmax(logits, dim=-1)
    confs, preds = torch.max(probs, dim=-1)
    targets = torch.randint(0, 60, (20,))
    correctness = (preds == targets).float()

    ce_loss = F.cross_entropy(logits, targets)
    mmce_penalty = oracles.rkhs_mmce(confs, correctness, sigma=sigma)
    total_loss = ce_loss + lambda_mmce * mmce_penalty

    assert not torch.isnan(total_loss)
    assert not torch.isinf(total_loss)
    if lambda_mmce == 0.0:
        assert pytest.approx(total_loss.item(), rel=1e-5) == ce_loss.item()
    else:
        assert total_loss.item() >= ce_loss.item()


@pytest.mark.parametrize("target_type", ["hard_top1", "soft_distribution"])
@pytest.mark.parametrize("lambda_mmce", [0.0, 0.5])
def test_p02_hard_vs_soft_targets_with_mmce(oracles, target_type, lambda_mmce):
    """P2.2: Verify loss works with both hard top-1 class indices and soft routing probability targets."""
    torch.manual_seed(10)
    logits = torch.randn(16, 60, requires_grad=True)

    if target_type == "hard_top1":
        targets = torch.randint(0, 60, (16,))
        ce_loss = F.cross_entropy(logits, targets)
    else:
        soft_targets = F.softmax(torch.randn(16, 60), dim=-1)
        ce_loss = oracles.soft_cross_entropy(logits, soft_targets)

    confs = F.softmax(logits, dim=-1).max(dim=-1).values
    corr = torch.ones(16)
    mmce = oracles.rkhs_mmce(confs, corr, sigma=0.2)
    total_loss = ce_loss + lambda_mmce * mmce
    total_loss.backward()

    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


# ===========================================================================
# Pairwise Set 3: NLL Optimization x L2 Regularization x Sparsity Regimes
# ===========================================================================

@pytest.mark.parametrize("alpha_l2", [0.0, 0.01, 1.0, 10.0])
@pytest.mark.parametrize("sample_regime,num_samples", [
    ("sparse", 2),
    ("moderate", 25),
    ("dense", 200),
])
def test_p03_nll_l2_sparsity_interaction(alpha_l2, sample_regime, num_samples):
    """P3.1: Verify interaction between L2 regularization strength and data sparsity."""
    torch.manual_seed(42)
    logits = torch.randn(num_samples, 60) * 2.0
    targets = torch.randint(0, 60, (num_samples,))

    temp = nn.Parameter(torch.tensor([2.0]))
    optimizer = torch.optim.SGD([temp], lr=0.05)

    for _ in range(30):
        optimizer.zero_grad()
        nll = F.cross_entropy(logits / temp, targets)
        reg = 0.5 * alpha_l2 * (temp - 1.0) ** 2
        loss = nll + reg
        loss.backward()
        optimizer.step()

    # For sparse data + high alpha, temp should be pulled close to 1.0
    if sample_regime == "sparse" and alpha_l2 >= 10.0:
        assert pytest.approx(temp.item(), abs=0.25) == 1.0
    elif sample_regime == "sparse" and alpha_l2 >= 1.0:
        assert temp.item() < 2.0  # Demonstrates shrinkage toward 1.0
    # For dense data + zero alpha, temp fits the data
    elif sample_regime == "dense" and alpha_l2 == 0.0:
        assert temp.item() > 0.1


# ===========================================================================
# Pairwise Set 4: Device Placement x Data Precision
# ===========================================================================

@pytest.mark.parametrize("device_str", ["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_p04_device_dtype_linear_projection(device_str, dtype):
    """P4.1: Verify linear projection executes across supported device and dtype pairs."""
    device = torch.device(device_str)
    head = nn.Linear(64, 32).to(device=device, dtype=dtype)
    x = torch.randn(4, 64, device=device, dtype=dtype)
    out = head(x)
    assert out.device.type == device.type
    assert out.dtype == dtype
    assert out.shape == (4, 32)


# ===========================================================================
# Pairwise Set 5: Sequence Length x Batch Size
# ===========================================================================

@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [16, 64, 128])
def test_p05_batch_seqlen_tensor_shaping(batch_size, seq_len):
    """P5.1: Verify tensor shaping (B, L, 3, 20, 60) across varying batch sizes and sequence lengths."""
    d_model = 64
    horizons = 3
    num_deep_layers = 20
    num_experts = 60
    head = nn.Linear(d_model, horizons * num_deep_layers * num_experts)
    x = torch.randn(batch_size, seq_len, d_model)
    out = head(x).view(batch_size, seq_len, horizons, num_deep_layers, num_experts)
    assert out.shape == (batch_size, seq_len, 3, 20, 60)
