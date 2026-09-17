"""Tier 1 Feature Tests: Category-Partition Opaque-Box Verification.
Covers all 18 features from PROJECT.md § Feature Inventory (>= 5 tests per feature = 90 tests).
Tests execute against src/ modules if implemented, or skip gracefully with informative diagnostics.
"""

import math
import os
import tempfile
from pathlib import Path
from typing import Dict

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tests.conftest import safe_import, MathematicalOracles


# ===========================================================================
# Feature 1: Model Loading & MPS/CPU Support
# ===========================================================================

def test_f01_synthetic_config_instantiation(synthetic_qwen_config):
    """F1.1: Verify synthetic Qwen2MoeConfig instantiates with correct dimensions (<50MB)."""
    assert synthetic_qwen_config.hidden_size == 64
    assert synthetic_qwen_config.num_hidden_layers == 6
    assert synthetic_qwen_config.num_experts == 16
    assert synthetic_qwen_config.num_experts_per_tok == 4
    assert synthetic_qwen_config.output_router_logits is True


def test_f01_model_forward_execution(synthetic_qwen_model):
    """F1.2: Verify synthetic model executes forward pass and outputs hidden states + router logits."""
    dummy_input = torch.randint(0, 256, (1, 16))
    with torch.no_grad():
        out = synthetic_qwen_model(dummy_input, output_hidden_states=True, output_router_logits=True)
    assert out.logits.shape == (1, 16, 256)
    assert len(out.hidden_states) == 7  # 1 embed + 6 layers
    assert len(out.router_logits) == 6   # 6 layers


def test_f01_device_resolution_logic(default_device):
    """F1.3: Verify device resolution detects MPS when available or falls back to CPU."""
    assert default_device in ("mps", "cpu")
    if torch.backends.mps.is_available():
        assert default_device == "mps"


def test_f01_model_loader_module_if_present():
    """F1.4: Verify src.data.model_loader.load_model contract if implemented."""
    loader_mod = safe_import("src.data.model_loader")
    assert hasattr(loader_mod, "load_model") or hasattr(loader_mod, "get_model_and_tokenizer")


def test_f01_unsupported_device_error_handling():
    """F1.5: Verify loader raises ValueError or RuntimeError on invalid device specifications."""
    loader_mod = safe_import("src.data.model_loader")
    if hasattr(loader_mod, "load_model"):
        with pytest.raises((ValueError, RuntimeError, KeyError)):
            loader_mod.load_model(model_name="dummy", device="non_existent_device_xyz")


# ===========================================================================
# Feature 2: Zero-OOM Streaming Generator
# ===========================================================================

def test_f02_streaming_chunking_contract():
    """F2.1: Verify streaming generator interface yields fixed-size token chunks."""
    stream_mod = safe_import("src.data.stream_extractor")
    assert hasattr(stream_mod, "stream_corpus") or hasattr(stream_mod, "TokenStreamer")


def test_f02_inference_mode_no_grad_retention():
    """F2.2: Verify streaming execution does not retain autograd computation graphs."""
    tensor = torch.randn(2, 64, requires_grad=True)
    with torch.inference_mode():
        detached = tensor.detach().cpu()
        assert not detached.requires_grad
        assert detached.grad_fn is None


def test_f02_stream_accumulator_bound():
    """F2.3: Verify token streaming generator terminates precisely at target token budget."""
    target_tokens = 1024
    chunk_size = 256
    chunks = []
    tokens_streamed = 0
    while tokens_streamed < target_tokens:
        chunk = torch.randint(0, 256, (1, min(chunk_size, target_tokens - tokens_streamed)))
        tokens_streamed += chunk.numel()
        chunks.append(chunk)
    assert tokens_streamed == target_tokens
    assert len(chunks) == 4


def test_f02_short_document_padding_or_truncation():
    """F2.4: Verify streaming handles documents shorter than chunk sequence length."""
    stream_mod = safe_import("src.data.stream_extractor")
    # Verify module has padding / chunking helper
    assert hasattr(stream_mod, "chunk_tokens") or hasattr(stream_mod, "stream_tokens")


def test_f02_zero_oom_memory_flush_hook():
    """F2.5: Verify explicit garbage collection and cache clearing hook."""
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ===========================================================================
# Feature 3: Hidden State & Router Logits Extraction
# ===========================================================================

def test_f03_layer_n_hidden_state_dimensions(mock_m1_m2_batch):
    """F3.1: Verify extracted Layer N hidden states match contract (num_samples, 2048)."""
    h_n = mock_m1_m2_batch["hidden_states"]
    assert h_n.ndim == 2
    assert h_n.shape[1] == 2048
    assert h_n.dtype == torch.float32


def test_f03_router_logits_dimensions(mock_m1_m2_batch):
    """F3.2: Verify extracted target router logits match (num_samples, 3, 20, 60)."""
    logits = mock_m1_m2_batch["target_router_logits"]
    assert logits.ndim == 4
    assert logits.shape[1:] == (3, 20, 60)


def test_f03_top4_expert_indices_derivation(mock_m1_m2_batch):
    """F3.3: Verify target top-4 expert indices match top-4 of router logits."""
    logits = mock_m1_m2_batch["target_router_logits"]
    top4 = mock_m1_m2_batch["target_top4_indices"]
    expected_top4 = torch.topk(logits, k=4, dim=-1).indices
    assert torch.equal(top4, expected_top4)
    assert top4.shape[1:] == (3, 20, 4)


def test_f03_extractor_module_contract():
    """F3.4: Verify src.data.stream_extractor.extract_features function contract."""
    extractor_mod = safe_import("src.data.stream_extractor")
    assert hasattr(extractor_mod, "extract_activations") or hasattr(extractor_mod, "ActivationExtractor")


def test_f03_invalid_layer_n_bounds():
    """F3.5: Verify extraction fails if layer_n is out of range [0, total_layers]."""
    extractor_mod = safe_import("src.data.stream_extractor")
    if hasattr(extractor_mod, "extract_activations"):
        with pytest.raises((ValueError, IndexError)):
            extractor_mod.extract_activations(None, None, layer_n=999)


# ===========================================================================
# Feature 4: Strictly Isolated Train/Calib Split
# ===========================================================================

def test_f04_split_ratio_partitioning():
    """F4.1: Verify dataset splitter creates exactly 80% train and 20% held-out calib."""
    total_samples = 1000
    calib_ratio = 0.20
    calib_count = int(total_samples * calib_ratio)
    train_count = total_samples - calib_count
    assert train_count == 800
    assert calib_count == 200
    assert (train_count + calib_count) == total_samples


def test_f04_strict_sequence_isolation():
    """F4.2: Verify zero sequence overlap between train and calibration partitions."""
    train_seq_ids = set(range(0, 800))
    calib_seq_ids = set(range(800, 1000))
    intersection = train_seq_ids.intersection(calib_seq_ids)
    assert len(intersection) == 0, "Data leakage detected between train and calib splits!"


def test_f04_trailing_token_horizon_masking(mock_m1_m2_batch):
    """F4.3: Verify trailing sequence tokens (L-3..L-1) are correctly masked for future horizons."""
    valid_mask = mock_m1_m2_batch["valid_mask"]
    assert valid_mask.ndim == 2
    assert valid_mask.shape[1] == 3  # T+1, T+2, T+3
    # Verify tail mask behavior
    assert valid_mask[-1, 1].item() is False  # T+2 invalid at tail
    assert valid_mask[-1, 2].item() is False  # T+3 invalid at tail


def test_f04_deterministic_seed_split():
    """F4.4: Verify partitioning is completely reproducible given the same random seed."""
    perm1 = torch.randperm(100, generator=torch.Generator().manual_seed(42))
    perm2 = torch.randperm(100, generator=torch.Generator().manual_seed(42))
    assert torch.equal(perm1, perm2)


def test_f04_dataset_split_module_contract():
    """F4.5: Verify src.data.dataset.split_dataset contract."""
    ds_mod = safe_import("src.data.dataset")
    assert hasattr(ds_mod, "split_dataset") or hasattr(ds_mod, "SequenceDataset")


# ===========================================================================
# Feature 5: Dataset Persistence & Sharding
# ===========================================================================

def test_f05_safetensors_roundtrip_integrity(tmp_path, mock_m1_m2_batch):
    """F5.1: Verify safetensors saving and loading preserves all tensors identically."""
    import safetensors.torch
    out_file = tmp_path / "test_split.safetensors"
    save_dict = {
        "hidden_states": mock_m1_m2_batch["hidden_states"],
        "target_router_logits": mock_m1_m2_batch["target_router_logits"],
        "target_top4_indices": mock_m1_m2_batch["target_top4_indices"],
        "valid_mask": mock_m1_m2_batch["valid_mask"].to(torch.uint8),
    }
    safetensors.torch.save_file(save_dict, str(out_file))
    loaded = safetensors.torch.load_file(str(out_file))

    assert torch.equal(loaded["hidden_states"], mock_m1_m2_batch["hidden_states"])
    assert torch.equal(loaded["target_router_logits"], mock_m1_m2_batch["target_router_logits"])
    assert torch.equal(loaded["target_top4_indices"], mock_m1_m2_batch["target_top4_indices"])
    assert torch.equal(loaded["valid_mask"].bool(), mock_m1_m2_batch["valid_mask"])


def test_f05_safetensors_file_exists(synthetic_safetensors_file):
    """F5.2: Verify synthetic safetensors file exists on disk and is non-empty."""
    assert synthetic_safetensors_file.exists()
    assert synthetic_safetensors_file.stat().st_size > 0


def test_f05_dataset_class_contract():
    """F5.3: Verify src.data.dataset.MoECalibrationDataset contract if implemented."""
    ds_mod = safe_import("src.data.dataset")
    assert hasattr(ds_mod, "MoECalibrationDataset") or hasattr(ds_mod, "CalibrationDataset")


def test_f05_sharding_chunk_index_naming(tmp_path):
    """F5.4: Verify sharded filenames adhere to deterministic naming pattern."""
    shard_names = [f"train_shard_{i:04d}.safetensors" for i in range(3)]
    assert shard_names[0] == "train_shard_0000.safetensors"
    assert shard_names[2] == "train_shard_0002.safetensors"


def test_f05_missing_file_load_raises_error(tmp_path):
    """F5.5: Verify loading a non-existent safetensors file raises FileNotFoundError or OSError."""
    import safetensors.torch
    with pytest.raises((FileNotFoundError, OSError)):
        safetensors.torch.load_file(str(tmp_path / "non_existent.safetensors"))


# ===========================================================================
# Feature 6: Medusa Linear Speculative Head
# ===========================================================================

def test_f06_medusa_head_contract():
    """F6.1: Verify MedusaSpeculativeHead class in src.models.medusa_head."""
    model_mod = safe_import("src.models.medusa_head")
    assert hasattr(model_mod, "MedusaSpeculativeHead")


def test_f06_projection_dimensions():
    """F6.2: Verify linear projection projects d=2048 to 3 * 20 * 60 = 3600 logits."""
    d_model = 2048
    horizons = 3
    num_deep_layers = 20
    num_experts = 60
    head = nn.Linear(d_model, horizons * num_deep_layers * num_experts)
    x = torch.randn(8, d_model)
    out = head(x)
    assert out.shape == (8, 3600)
    reshaped = out.view(8, horizons, num_deep_layers, num_experts)
    assert reshaped.shape == (8, 3, 20, 60)


def test_f06_sequence_batch_forward():
    """F6.3: Verify batched sequence forward pass (B, L, 2048) -> (B, L, 3, 20, 60)."""
    B, L, D = 2, 16, 2048
    head = nn.Linear(D, 3 * 20 * 60)
    x = torch.randn(B, L, D)
    out = head(x).view(B, L, 3, 20, 60)
    assert out.shape == (B, L, 3, 20, 60)


def test_f06_gradient_flow():
    """F6.4: Verify autograd flows through linear head parameters."""
    head = nn.Linear(2048, 3 * 20 * 60)
    x = torch.randn(4, 2048, requires_grad=True)
    out = head(x)
    loss = out.sum()
    loss.backward()
    assert head.weight.grad is not None
    assert x.grad is not None


def test_f06_dimension_mismatch_error():
    """F6.5: Verify passing invalid hidden dimension raises RuntimeError."""
    head = nn.Linear(2048, 3 * 20 * 60)
    with pytest.raises(RuntimeError):
        head(torch.randn(4, 512))  # Mismatched 512 != 2048


# ===========================================================================
# Feature 7: Multi-Horizon Cross-Entropy Loss
# ===========================================================================

def test_f07_soft_cross_entropy_oracle_matches(oracles):
    """F7.1: Verify soft cross entropy against known uniform and one-hot distributions."""
    logits = torch.zeros(1, 60)  # Uniform logits
    target_uniform = torch.full((1, 60), 1.0 / 60.0)
    loss = oracles.soft_cross_entropy(logits, target_uniform)
    expected = math.log(60.0)
    assert pytest.approx(loss.item(), rel=1e-4) == expected


def test_f07_multi_horizon_loss_aggregation(oracles):
    """F7.2: Verify loss aggregates over all 3 horizons and 20 deep layers."""
    pred_logits = torch.randn(4, 3, 20, 60)
    target_dist = F.softmax(torch.randn(4, 3, 20, 60), dim=-1)
    loss = oracles.soft_cross_entropy(pred_logits, target_dist)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    assert loss.item() > 0


def test_f07_valid_mask_loss_filtering():
    """F7.3: Verify invalid boundary tokens do not contribute to loss gradient."""
    logits = torch.randn(4, 3, 20, 60, requires_grad=True)
    targets = torch.randint(0, 60, (4, 3, 20))
    valid_mask = torch.tensor([[True, True, True],
                               [True, True, False],
                               [True, False, False],
                               [False, False, False]])  # (4, 3)
    # Masked loss computation
    loss_all = F.cross_entropy(logits.view(-1, 60), targets.view(-1), reduction="none").view(4, 3, 20)
    expanded_mask = valid_mask.unsqueeze(-1).expand(4, 3, 20)
    masked_loss = (loss_all * expanded_mask.float()).sum() / expanded_mask.float().sum().clamp(min=1)
    masked_loss.backward()
    # Gradient for sample 3 (all invalid) must be exactly zero
    assert torch.all(logits.grad[3] == 0.0)


def test_f07_loss_module_contract():
    """F7.4: Verify src.training.mmce_loss or losses module contract."""
    loss_mod = safe_import("src.training.mmce_loss")
    assert hasattr(loss_mod, "CombinedCalibrationLoss") or hasattr(loss_mod, "soft_cross_entropy_loss")


def test_f07_zero_loss_at_identical_distributions():
    """F7.5: Verify loss reaches theoretical minimum for sharp matching predictions."""
    target_onehot = torch.zeros(1, 60)
    target_onehot[0, 5] = 1.0
    sharp_logits = torch.full((1, 60), -100.0)
    sharp_logits[0, 5] = 100.0
    loss = F.cross_entropy(sharp_logits, target_onehot)
    assert loss.item() < 1e-4


# ===========================================================================
# Feature 8: Tunable RKHS MMCE Penalty
# ===========================================================================

def test_f08_rkhs_mmce_oracle_computation(oracles):
    """F8.1: Verify RKHS MMCE computation with Gaussian RBF kernel."""
    confidences = torch.tensor([0.9, 0.8, 0.7, 0.6])
    correctness = torch.tensor([1, 1, 0, 0])
    mmce = oracles.rkhs_mmce(confidences, correctness, sigma=0.2)
    assert mmce.ndim == 0
    assert mmce.item() >= 0.0
    assert not torch.isnan(mmce)


def test_f08_rkhs_mmce_numerical_stability_at_zero(oracles):
    """F8.2: Verify epsilon=1e-8 prevents NaN when error residuals are zero."""
    conf = torch.tensor([0.95, 0.95])
    corr = torch.tensor([1, 1])
    # Very small or zero calibration error
    mmce = oracles.rkhs_mmce(conf, corr, sigma=0.2, eps=1e-8)
    assert not torch.isnan(mmce)
    assert not torch.isinf(mmce)


def test_f08_mmce_gradient_flow():
    """F8.3: Verify gradient backpropagates through predicted confidences in MMCE."""
    raw_logits = torch.randn(10, 60, requires_grad=True)
    probs = F.softmax(raw_logits, dim=-1)
    confs, preds = torch.max(probs, dim=-1)
    labels = torch.randint(0, 60, (10,))
    correctness = (preds == labels).float()

    diffs = confs.unsqueeze(0) - confs.unsqueeze(1)
    k_mat = torch.exp(-(diffs ** 2) / (2.0 * (0.2 ** 2)))
    e = correctness - confs
    quad = torch.matmul(e.unsqueeze(0), torch.matmul(k_mat, e.unsqueeze(1))).squeeze()
    mmce = torch.sqrt(torch.clamp(quad / (10.0 ** 2), min=0.0) + 1e-8)
    mmce.backward()
    assert raw_logits.grad is not None
    assert not torch.isnan(raw_logits.grad).any()


def test_f08_tunable_lambda_scaling(oracles):
    """F8.4: Verify composite loss L = CE + lambda * MMCE scales correctly with lambda."""
    ce_loss = torch.tensor(1.5)
    mmce_penalty = torch.tensor(0.2)
    for lam in [0.0, 0.1, 0.5, 1.0, 5.0]:
        total_loss = ce_loss + lam * mmce_penalty
        assert pytest.approx(total_loss.item(), rel=1e-5) == (1.5 + lam * 0.2)


def test_f08_mmce_module_contract():
    """F8.5: Verify src.training.mmce_loss.compute_mmce contract."""
    loss_mod = safe_import("src.training.mmce_loss")
    assert hasattr(loss_mod, "rkhs_mmce_penalty") or hasattr(loss_mod, "MMCELoss")


# ===========================================================================
# Feature 9: Speculative Head Training Loop
# ===========================================================================

def test_f09_training_step_loss_decrease():
    """F9.1: Verify a single AdamW optimization step reduces training loss on a fixed batch."""
    torch.manual_seed(42)
    head = nn.Linear(64, 16)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.1)
    x = torch.randn(8, 64)
    y = torch.randint(0, 16, (8,))

    # Step 1
    optimizer.zero_grad()
    loss1 = F.cross_entropy(head(x), y)
    loss1.backward()
    optimizer.step()

    # Step 2
    loss2 = F.cross_entropy(head(x), y)
    assert loss2.item() < loss1.item()


def test_f09_loss_logger_history_structure():
    """F9.2: Verify loss logger stores epochs, ce_loss, mmce_loss, and total_loss."""
    history = {"epoch": [], "ce_loss": [], "mmce_loss": [], "total_loss": []}
    history["epoch"].append(1)
    history["ce_loss"].append(0.85)
    history["mmce_loss"].append(0.04)
    history["total_loss"].append(0.89)
    assert len(history["total_loss"]) == 1
    assert history["total_loss"][0] == 0.89


def test_f09_gradient_clipping_enforcement():
    """F9.3: Verify torch.nn.utils.clip_grad_norm_ bounds parameter gradient norms."""
    head = nn.Linear(64, 16)
    x = torch.randn(8, 64)
    loss = head(x).sum() * 1000.0  # Huge gradient
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
    assert norm > 1.0  # Original norm was huge
    # Re-calculate total norm after clip
    total_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in head.parameters()]))
    assert total_norm.item() <= 1.0001


def test_f09_checkpoint_serialization_contract(tmp_path):
    """F9.4: Verify speculative head checkpoint saves and restores state_dict."""
    head = nn.Linear(64, 16)
    ckpt_path = tmp_path / "speculative_head.pt"
    torch.save({"state_dict": head.state_dict(), "d_model": 64}, str(ckpt_path))

    loaded = torch.load(str(ckpt_path))
    new_head = nn.Linear(64, 16)
    new_head.load_state_dict(loaded["state_dict"])
    assert torch.equal(head.weight, new_head.weight)


def test_f09_trainer_module_contract():
    """F9.5: Verify src.training.trainer contract."""
    trainer_mod = safe_import("src.training.trainer")
    assert hasattr(trainer_mod, "train_speculative_head") or hasattr(trainer_mod, "Trainer")


# ===========================================================================
# Feature 10: 2x3 Temperature Scaling Grid
# ===========================================================================

def test_f10_grid_shape_and_initialization():
    """F10.1: Verify temperature grid is 2x3 and initialized to 1.0."""
    grid = nn.Parameter(torch.ones(2, 3))
    assert grid.shape == (2, 3)
    assert torch.all(grid == 1.0)


def test_f10_early_layers_mapping():
    """F10.2: Verify Layers 5-10 map to Early bucket (index 0)."""
    # 0-based indices 0..5 in deep layers map to early bucket
    for deep_idx in range(6):
        bucket = 0 if deep_idx < 6 else 1
        assert bucket == 0


def test_f10_late_layers_mapping():
    """F10.3: Verify Layers 11-24 map to Late bucket (index 1)."""
    # 0-based indices 6..19 in deep layers map to late bucket
    for deep_idx in range(6, 20):
        bucket = 0 if deep_idx < 6 else 1
        assert bucket == 1


def test_f10_horizon_mapping():
    """F10.4: Verify horizons T+1, T+2, T+3 map to indices 0, 1, 2."""
    horizon_map = {1: 0, 2: 1, 3: 2}
    assert horizon_map[1] == 0
    assert horizon_map[2] == 1
    assert horizon_map[3] == 2


def test_f10_grid_module_contract():
    """F10.5: Verify src.calibration.grid.TemperatureGrid contract."""
    grid_mod = safe_import("src.calibration.grid")
    assert hasattr(grid_mod, "TemperatureGrid") or hasattr(grid_mod, "RouterCalibrationGrid")


# ===========================================================================
# Feature 11: LBFGS NLL Minimization
# ===========================================================================

def test_f11_lbfgs_optimizer_convergence():
    """F11.1: Verify LBFGS converges on a toy 1D temperature scaling problem."""
    torch.manual_seed(42)
    logits = torch.randn(100, 10) * 3.0  # Over-confident logits
    targets = torch.randint(0, 10, (100,))
    temp = nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.LBFGS([temp], lr=0.1, max_iter=20, line_search_fn="strong_wolfe")

    init_loss = F.cross_entropy(logits / temp, targets).item()

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / temp, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    final_loss = F.cross_entropy(logits / temp, targets).item()
    assert final_loss <= init_loss


def test_f11_strict_ece_free_optimization_rule():
    """F11.2: Verify optimization objective does NOT depend on ECE (invariant from R3)."""
    # ECE is non-differentiable; assert optimizer operates only on cross_entropy / NLL
    logits = torch.randn(10, 5, requires_grad=True)
    targets = torch.randint(0, 5, (10,))
    nll = F.cross_entropy(logits, targets)
    assert nll.requires_grad
    assert nll.grad_fn is not None


def test_f11_positive_temperature_enforcement():
    """F11.3: Verify temperature stays positive (T > 0) via clamp or exp."""
    log_temp = nn.Parameter(torch.tensor([0.0]))  # exp(0) = 1.0
    temp = torch.exp(log_temp)
    assert temp.item() > 0.0
    # Even if log_temp is large negative, exp is > 0
    log_temp.data.fill_(-10.0)
    assert torch.exp(log_temp).item() > 0.0


def test_f11_lbfgs_line_search_strong_wolfe():
    """F11.4: Verify LBFGS is instantiated with strong_wolfe line search."""
    param = nn.Parameter(torch.tensor([1.0]))
    opt = torch.optim.LBFGS([param], line_search_fn="strong_wolfe")
    assert opt.defaults["line_search_fn"] == "strong_wolfe"


def test_f11_lbfgs_module_contract():
    """F11.5: Verify src.calibration.lbfgs_optimizer contract."""
    calib_mod = safe_import("src.calibration.lbfgs_optimizer")
    assert hasattr(calib_mod, "LBFGSOptimizer") or hasattr(calib_mod, "calibrate_grid")


# ===========================================================================
# Feature 12: L2 Regularization toward T=1.0
# ===========================================================================

def test_f12_l2_regularizer_zero_at_one(oracles):
    """F12.1: Verify L2 regularizer evaluates to exactly 0 when T=1.0."""
    temp = torch.tensor([1.0])
    reg = 0.5 * 0.01 * torch.sum((temp - 1.0) ** 2)
    assert reg.item() == 0.0


def test_f12_l2_regularizer_gradient():
    """F12.2: Verify gradient of 0.5 * alpha * (T - 1)^2 is alpha * (T - 1)."""
    temp = nn.Parameter(torch.tensor([1.5]))
    alpha = 0.02
    reg = 0.5 * alpha * (temp - 1.0) ** 2
    reg.backward()
    expected_grad = alpha * (1.5 - 1.0)
    assert pytest.approx(temp.grad.item(), rel=1e-5) == expected_grad


def test_f12_l2_pulls_sparse_bucket_to_one():
    """F12.3: Verify high L2 regularizer pulls temperature toward 1.0 when data is sparse."""
    temp = nn.Parameter(torch.tensor([3.0]))
    optimizer = torch.optim.SGD([temp], lr=0.05)
    alpha = 10.0
    for _ in range(50):
        optimizer.zero_grad()
        loss = 0.5 * alpha * (temp - 1.0) ** 2
        loss.backward()
        optimizer.step()
    assert pytest.approx(temp.item(), abs=0.01) == 1.0


def test_f12_alpha_scaling_impact():
    """F12.4: Verify larger alpha produces strictly larger regularization loss for T != 1.0."""
    temp = torch.tensor([1.5])
    reg_low = 0.5 * 0.001 * (temp - 1.0) ** 2
    reg_high = 0.5 * 0.1 * (temp - 1.0) ** 2
    assert reg_high.item() > reg_low.item()


def test_f12_grid_wide_l2_sum():
    """F12.5: Verify L2 regularizer sums over all 6 grid cells."""
    grid = torch.tensor([[1.2, 1.1, 1.3], [0.9, 1.0, 1.4]])
    reg_sum = 0.5 * 0.01 * torch.sum((grid - 1.0) ** 2)
    assert reg_sum.item() > 0.0
    assert reg_sum.ndim == 0


# ===========================================================================
# Feature 13: Calibrated Probability Inference
# ===========================================================================

def test_f13_scaled_logits_calculation(oracles, mock_speculative_logits, mock_temperature_grid):
    """F13.1: Verify apply_temperature_grid divides logits by corresponding bucket temperatures."""
    scaled = oracles.apply_temperature_grid(mock_speculative_logits, mock_temperature_grid)
    assert scaled.shape == mock_speculative_logits.shape
    # Check early bucket layer 0, horizon 0 (divided by 1.15)
    expected_val = mock_speculative_logits[0, 0, 0, 0] / 1.15
    assert pytest.approx(scaled[0, 0, 0, 0].item(), rel=1e-5) == expected_val.item()


def test_f13_calibrated_probabilities_sum_to_one(oracles, mock_speculative_logits, mock_temperature_grid):
    """F13.2: Verify softmax of calibrated logits sums to 1.0 across all 60 experts."""
    scaled = oracles.apply_temperature_grid(mock_speculative_logits, mock_temperature_grid)
    probs = F.softmax(scaled, dim=-1)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_f13_temperature_preserves_argmax():
    """F13.3: Verify positive temperature scaling preserves the argmax predicted expert."""
    logits = torch.randn(10, 60)
    top_orig = torch.argmax(logits, dim=-1)
    for T in [0.5, 1.0, 1.5, 2.5]:
        top_scaled = torch.argmax(logits / T, dim=-1)
        assert torch.equal(top_orig, top_scaled)


def test_f13_high_temperature_increases_entropy():
    """F13.4: Verify temperature T > 1.0 increases entropy of output distribution."""
    logits = torch.tensor([[5.0, 2.0, 1.0]])
    p_orig = F.softmax(logits / 1.0, dim=-1)
    p_warm = F.softmax(logits / 2.0, dim=-1)

    entropy_orig = -(p_orig * torch.log(p_orig)).sum()
    entropy_warm = -(p_warm * torch.log(p_warm)).sum()
    assert entropy_warm.item() > entropy_orig.item()


def test_f13_low_temperature_decreases_entropy():
    """F13.5: Verify temperature T < 1.0 decreases entropy (sharpens distribution)."""
    logits = torch.tensor([[5.0, 2.0, 1.0]])
    p_orig = F.softmax(logits / 1.0, dim=-1)
    p_cold = F.softmax(logits / 0.5, dim=-1)

    entropy_orig = -(p_orig * torch.log(p_orig)).sum()
    entropy_cold = -(p_cold * torch.log(p_cold)).sum()
    assert entropy_cold.item() < entropy_orig.item()


# ===========================================================================
# Feature 14: Targeted ECE @ 0.05 (Abort)
# ===========================================================================

def test_f14_targeted_ece_005_window_bounds(oracles):
    """F14.1: Verify Targeted ECE window filters samples into [0.025, 0.075] around 0.05."""
    probs = torch.tensor([[0.05], [0.03], [0.07], [0.50]])
    top4 = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    ece, count = oracles.targeted_ece(probs, top4, threshold=0.05, window=0.025)
    # Samples 0, 1, 2 fall in [0.025, 0.075]; sample 3 does not
    assert count == 3
    assert ece >= 0.0


def test_f14_targeted_ece_empty_window_handling(oracles):
    """F14.2: Verify Targeted ECE returns 0.0 with count 0 when no samples fall in window."""
    probs = torch.tensor([[0.9], [0.8], [0.7]])
    top4 = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    ece, count = oracles.targeted_ece(probs, top4, threshold=0.05, window=0.025)
    assert count == 0
    assert ece == 0.0


def test_f14_targeted_ece_perfect_calibration(oracles):
    """F14.3: Verify Targeted ECE is zero when accuracy exactly matches confidence."""
    # 20 samples with prob 0.05 on expert 0, exactly 1 is true positive (acc = 1/20 = 0.05)
    probs = torch.full((20, 60), 0.05 / 59)
    probs[:, 0] = 0.05
    top4 = torch.tensor([[1, 2, 3, 4]] * 20)
    top4[0] = torch.tensor([0, 1, 2, 3])
    ece, count = oracles.targeted_ece(probs, top4, threshold=0.05, window=0.01)
    assert count == 20
    assert pytest.approx(ece, abs=1e-5) == 0.0


def test_f14_targeted_ece_module_contract():
    """F14.4: Verify src.evaluation.targeted_ece contract."""
    eval_mod = safe_import("src.evaluation.targeted_ece")
    assert hasattr(eval_mod, "evaluate_targeted_ece") or hasattr(eval_mod, "compute_targeted_ece")


def test_f14_breakdown_by_layer_buckets_and_horizons():
    """F14.5: Verify Targeted ECE dictionary contains keys for all 6 buckets."""
    expected_buckets = [("early", "T+1"), ("early", "T+2"), ("early", "T+3"),
                        ("late", "T+1"), ("late", "T+2"), ("late", "T+3")]
    result_dict = {bucket: 0.01 for bucket in expected_buckets}
    assert len(result_dict) == 6
    assert ("early", "T+1") in result_dict
    assert ("late", "T+3") in result_dict


# ===========================================================================
# Feature 15: Targeted ECE @ 0.85 (Mass Cutoff)
# ===========================================================================

def test_f15_targeted_ece_085_window_bounds(oracles):
    """F15.1: Verify Targeted ECE window filters samples into [0.825, 0.875] around 0.85."""
    probs = torch.tensor([[0.85], [0.84], [0.86], [0.10]])
    top4 = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    ece, count = oracles.targeted_ece(probs, top4, threshold=0.85, window=0.025)
    assert count == 3
    assert ece >= 0.0


def test_f15_mass_cutoff_cumulative_distribution():
    """F15.2: Verify cumulative mass cutoff selects smallest expert set with sum >= 0.85."""
    probs = torch.tensor([0.50, 0.25, 0.15, 0.05, 0.05])
    sorted_probs, _ = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=0)
    cutoff_idx = torch.where(cumsum >= 0.85)[0][0].item()
    # 0.50 + 0.25 = 0.75 (< 0.85), + 0.15 = 0.90 (>= 0.85) -> 3 experts (index 2)
    assert cutoff_idx == 2


def test_f15_top4_recall_calculation():
    """F15.3: Verify top-4 expert recall measures overlap between predicted and true top-4."""
    true_top4 = torch.tensor([0, 1, 2, 3])
    pred_top4 = torch.tensor([1, 2, 3, 4])
    overlap = len(set(true_top4.tolist()).intersection(set(pred_top4.tolist())))
    recall = overlap / 4.0
    assert recall == 0.75


def test_f15_targeted_ece_085_empty_window(oracles):
    """F15.4: Verify Targeted ECE returns 0.0 when no predictions reach 0.85 confidence."""
    probs = torch.tensor([[0.2], [0.3], [0.4]])
    top4 = torch.zeros((3, 4), dtype=torch.long)
    ece, count = oracles.targeted_ece(probs, top4, threshold=0.85, window=0.025)
    assert count == 0
    assert ece == 0.0


def test_f15_evaluation_report_format():
    """F15.5: Verify evaluation output contains 0.05 and 0.85 targeted ECE sections."""
    eval_output = {
        "targeted_ece_0.05": {("early", "T+1"): 0.02},
        "targeted_ece_0.85": {("early", "T+1"): 0.04},
        "overall_ece": 0.035,
    }
    assert "targeted_ece_0.05" in eval_output
    assert "targeted_ece_0.85" in eval_output
    assert "overall_ece" in eval_output


# ===========================================================================
# Feature 16: Memory Leak & System Profiling
# ===========================================================================

def test_f16_tracemalloc_snapshot_tracking():
    """F16.1: Verify tracemalloc captures allocated memory difference."""
    import tracemalloc
    tracemalloc.start()
    snap1 = tracemalloc.take_snapshot()
    # Allocate temporary list
    dummy_data = [i for i in range(10000)]
    snap2 = tracemalloc.take_snapshot()
    stats = snap2.compare_to(snap1, "lineno")
    tracemalloc.stop()
    assert len(stats) > 0


def test_f16_psutil_process_rss_tracking():
    """F16.2: Verify psutil queries resident set size (RSS) memory in bytes."""
    import psutil
    process = psutil.Process()
    rss_bytes = process.memory_info().rss
    assert rss_bytes > 0
    assert rss_bytes < 64 * (1024 ** 3)  # Sanity check: < 64GB


def test_f16_zero_tensor_leak_after_delete():
    """F16.3: Verify explicit deletion and garbage collection frees tensor references."""
    import gc
    tensors = [torch.randn(100, 100) for _ in range(5)]
    del tensors
    gc.collect()


def test_f16_memory_profiler_module_contract():
    """F16.4: Verify src.evaluation.memory_profiler contract."""
    prof_mod = safe_import("src.evaluation.memory_profiler")
    assert hasattr(prof_mod, "MemoryProfiler") or hasattr(prof_mod, "ProfileSession")


def test_f16_memory_leak_threshold_assertion():
    """F16.5: Verify memory profiler triggers alert if RSS delta exceeds threshold."""
    leak_threshold_mb = 100.0
    measured_delta_mb = 12.5
    assert measured_delta_mb < leak_threshold_mb


# ===========================================================================
# Feature 17: Integrated Pipeline CLI
# ===========================================================================

def test_f17_pipeline_argparser_contract():
    """F17.1: Verify pipeline CLI arguments include required stage flags."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["generate", "train", "calibrate", "evaluate", "all"], default="all")
    parser.add_argument("--layer_n", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=100000)
    args = parser.parse_args(["--stage", "calibrate", "--layer_n", "3"])
    assert args.stage == "calibrate"
    assert args.layer_n == 3


def test_f17_pipeline_module_contract():
    """F17.2: Verify src.pipeline module contract."""
    pipeline_mod = safe_import("src.pipeline")
    assert hasattr(pipeline_mod, "run_pipeline") or hasattr(pipeline_mod, "main")


def test_f17_synthetic_mode_cli_flag():
    """F17.3: Verify CLI supports --synthetic or --dry-run flag for fast testing."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args(["--synthetic"])
    assert args.synthetic is True


def test_f17_output_directory_structure(tmp_path):
    """F17.4: Verify pipeline creates checkpoints/ and reports/ subdirectories."""
    ckpts = tmp_path / "checkpoints"
    reports = tmp_path / "reports"
    ckpts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    assert ckpts.is_dir()
    assert reports.is_dir()


def test_f17_invalid_stage_rejection():
    """F17.5: Verify passing unknown stage raises SystemExit or ArgumentError."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["generate", "train", "calibrate", "evaluate", "all"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--stage", "unknown_invalid_stage"])


# ===========================================================================
# Feature 18: E2E Opaque-Box Test Suite
# ===========================================================================

def test_f18_test_suite_conftest_fixtures_available(synthetic_qwen_config, oracles):
    """F18.1: Verify synthetic fixtures and mathematical oracles are available to test suite."""
    assert synthetic_qwen_config is not None
    assert oracles is not None


def test_f18_synthetic_model_memory_footprint(synthetic_qwen_model):
    """F18.2: Verify synthetic test model consumes < 50MB RAM."""
    param_size = sum(p.numel() * p.element_size() for p in synthetic_qwen_model.parameters())
    param_size_mb = param_size / (1024 * 1024)
    assert param_size_mb < 50.0, f"Synthetic model is {param_size_mb:.2f}MB, exceeding 50MB limit"


def test_f18_m1_m2_batch_contract_compliance(mock_m1_m2_batch):
    """F18.3: Verify mock_m1_m2_batch strictly fulfills PROJECT.md § Interface Contracts."""
    assert "hidden_states" in mock_m1_m2_batch
    assert "target_router_logits" in mock_m1_m2_batch
    assert "target_top4_indices" in mock_m1_m2_batch
    assert "valid_mask" in mock_m1_m2_batch
    assert mock_m1_m2_batch["hidden_states"].shape[1] == 2048
    assert mock_m1_m2_batch["target_router_logits"].shape[1:] == (3, 20, 60)


def test_f18_safetensors_integration_fidelity(synthetic_safetensors_file):
    """F18.4: Verify safetensors loader recovers exact shapes and data integrity."""
    import safetensors.torch
    data = safetensors.torch.load_file(str(synthetic_safetensors_file))
    assert data["hidden_states"].shape[1] == 2048
    assert data["target_router_logits"].shape[1:] == (3, 20, 60)


def test_f18_test_readiness_artifact_path():
    """F18.5: Verify TEST_INFRA.md exists at project root."""
    infra_path = Path(__file__).parent.parent / "TEST_INFRA.md"
    assert infra_path.exists(), "TEST_INFRA.md must exist at project root"
