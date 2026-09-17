"""Adversarial Stress Test Suite: Milestone 1 Boundary Conditions & Data Integrity.

Written by empirical challenger (teamwork_preview_challenger_m1_2).
Covers:
1. Sequence-atomic partitioning: disjointness, completeness, arbitrary split ratios, token isolation.
2. Safetensors serialization: non-contiguous layouts, strange strides, large batches, DataLoader integration.
3. Gradient isolation: exact 0.0 gradient norm on masked boundary tokens across deep layers.
"""

import math
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import (
    AlignedSequence,
    DatasetSplitConfig,
    MoECalibrationDataset,
    align_sequence_targets,
    build_and_split_calibration_datasets,
    load_dataset_safetensors,
    partition_sequence_indices,
    save_dataset_safetensors,
)
from src.data.model_loader import get_synthetic_model
from src.data.stream_extractor import ExtractionBatch


# ===========================================================================
# 1. Sequence-Atomic Partitioning & Strict Isolation Stress Tests
# ===========================================================================

@pytest.mark.parametrize("num_sequences", [2, 3, 5, 10, 50, 98, 100, 250, 1000])
@pytest.mark.parametrize("train_ratio", [0.1, 0.25, 0.333, 0.5, 0.667, 0.75, 0.8, 0.9, 0.95, 0.99])
def test_partition_disjointness_arbitrary_train_ratios(num_sequences: int, train_ratio: float):
    """Stress test sequence-atomic partitioning across arbitrary sequence counts and split ratios.
    
    Invariants asserted:
    - Train and calib indices are strictly disjoint (intersection is empty).
    - Union of train and calib indices covers all sequences exactly.
    - Each split contains at least 1 sequence (or raises ValueError if mathematically impossible).
    """
    try:
        train_idx, calib_idx = partition_sequence_indices(num_sequences, train_ratio=train_ratio)
    except ValueError as e:
        # If mathematically num_sequences * ratio leaves 0 in one split, ValueError is expected
        n_train = max(1, int(round(num_sequences * train_ratio)))
        n_calib = num_sequences - n_train
        assert n_train < 1 or n_calib < 1
        return

    # Invariant 1: Non-empty splits
    assert len(train_idx) >= 1, f"Train split empty for N={num_sequences}, ratio={train_ratio}"
    assert len(calib_idx) >= 1, f"Calib split empty for N={num_sequences}, ratio={train_ratio}"

    # Invariant 2: Completeness
    assert len(train_idx) + len(calib_idx) == num_sequences

    # Invariant 3: Strict disjointness
    set_train = set(train_idx)
    set_calib = set(calib_idx)
    assert set_train.isdisjoint(set_calib), f"Disjointness violated! Overlap: {set_train & set_calib}"
    assert set_train | set_calib == set(range(num_sequences))


@pytest.mark.parametrize("calib_ratio", [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 0.9, 0.99])
def test_partition_disjointness_calib_ratios(calib_ratio: float):
    """Verify partitioning when explicit calib_ratio is provided (98 sequences standard)."""
    num_sequences = 98
    train_idx, calib_idx = partition_sequence_indices(num_sequences, calib_ratio=calib_ratio)

    assert set(train_idx).isdisjoint(set(calib_idx))
    assert len(train_idx) + len(calib_idx) == num_sequences
    assert len(train_idx) >= 1 and len(calib_idx) >= 1


def test_partition_invalid_inputs_defensive_exceptions():
    """Verify strict rejection of invalid partitioning configurations."""
    # Fewer than 2 sequences
    with pytest.raises(ValueError, match="At least 2 sequences required"):
        partition_sequence_indices(1, train_ratio=0.8)

    with pytest.raises(ValueError, match="At least 2 sequences required"):
        partition_sequence_indices(0, train_ratio=0.8)

    # Invalid ratios <= 0.0 or >= 1.0
    for bad_ratio in [-0.5, 0.0, 1.0, 1.5]:
        with pytest.raises(ValueError, match="train_ratio must be between 0 and 1"):
            partition_sequence_indices(10, train_ratio=bad_ratio)

        with pytest.raises(ValueError, match="calib_ratio must be between 0 and 1"):
            partition_sequence_indices(10, calib_ratio=bad_ratio)

    # N=2 with extreme ratio that rounds to 2 train / 0 calib
    with pytest.raises(ValueError, match="Invalid split configuration"):
        partition_sequence_indices(2, train_ratio=0.99)

    with pytest.raises(ValueError, match="Invalid split configuration"):
        partition_sequence_indices(2, calib_ratio=0.99)


def test_partition_shuffle_determinism_and_entropy():
    """Verify shuffle determinism with same seed and different permutations with different seeds."""
    N = 98
    # Same seed -> identical partitions
    t1, c1 = partition_sequence_indices(N, train_ratio=0.8, shuffle=True, seed=12345)
    t2, c2 = partition_sequence_indices(N, train_ratio=0.8, shuffle=True, seed=12345)
    assert t1 == t2
    assert c1 == c2

    # Different seed -> different partitions
    t3, c3 = partition_sequence_indices(N, train_ratio=0.8, shuffle=True, seed=99999)
    assert t1 != t3
    assert c1 != c3

    # Disjointness holds regardless of shuffle seed
    assert set(t1).isdisjoint(set(c1))
    assert set(t3).isdisjoint(set(c3))


@pytest.mark.parametrize("split_ratio", [0.1, 0.2, 0.5, 0.8, 0.9])
def test_token_level_strict_isolation_in_datasets(split_ratio: float):
    """Adversarial Oracle: inject globally unique token identifiers per sequence.
    
    Verify that no token or representation from any training sequence ever appears
    in the calibration dataset (and vice versa) at the tensor level.
    """
    num_sequences = 10
    seq_len = 16
    d_model = 32
    num_layers = 4
    num_experts = 8

    # Create synthetic sequence batches where each sequence has unique scalar offsets
    sequences: List[ExtractionBatch] = []
    for seq_i in range(num_sequences):
        # Unique signature within FP16 exact integer representation range (< 2048)
        # Token at pos j has integer value seq_i * 50 + j (max = 9*50+16 = 466)
        token_tags = torch.arange(seq_len, dtype=torch.float32) + (seq_i * 50.0)
        h = token_tags.unsqueeze(-1).expand(seq_len, d_model).clone().to(torch.float16)
        r = torch.randn(seq_len, num_layers, num_experts, dtype=torch.float16)
        top4_idx = torch.zeros(seq_len, num_layers, 4, dtype=torch.int64)
        top4_pr = torch.zeros(seq_len, num_layers, 4, dtype=torch.float16)
        batch = ExtractionBatch(
            sequence_idx=seq_i,
            input_ids=torch.zeros(1, seq_len, dtype=torch.int64),
            hidden_states=h.unsqueeze(0),
            router_logits=r.unsqueeze(0),
            top4_indices=top4_idx.unsqueeze(0),
            top4_probs=top4_pr.unsqueeze(0),
            seq_len=seq_len,
        )
        sequences.append(batch)

    cfg = DatasetSplitConfig(
        train_ratio=split_ratio,
        deep_layer_start=1,
        deep_layer_end=4,
        drop_boundary_tokens=False,
    )
    train_ds, calib_ds = build_and_split_calibration_datasets(sequences, config=cfg)

    # Extract all sequence tag IDs present in train hidden states
    train_tags = [(int(val.item()) // 50) for val in train_ds.hidden_states[:, 0]]
    calib_tags = [(int(val.item()) // 50) for val in calib_ds.hidden_states[:, 0]]

    train_seq_set = set(train_tags)
    calib_seq_set = set(calib_tags)

    # Assert absolute token-level disjointness
    assert train_seq_set.isdisjoint(calib_seq_set), (
        f"Token leakage detected! Train sequences {train_seq_set} overlap with calib {calib_seq_set}"
    )

    # Verify metadata accuracy
    meta_train_indices = set(map(int, train_ds.metadata["sequence_indices"].split(",")))
    meta_calib_indices = set(map(int, calib_ds.metadata["sequence_indices"].split(",")))
    assert train_seq_set == meta_train_indices
    assert calib_seq_set == meta_calib_indices
    assert meta_train_indices.isdisjoint(meta_calib_indices)


# ===========================================================================
# 2. Safetensors Serialization, Strange Strides & DataLoader Stress Tests
# ===========================================================================

def test_safetensors_serialization_strange_strides():
    """Adversarial stress: pass non-contiguous tensors with diverse strange strides.
    
    Validates that save_dataset_safetensors automatically forces contiguity on:
    - Transposed 2D tensors (e.g. h.t())
    - Multi-dimensional sliced tensors with step > 1 (e.g. logits[::2, :, ::2, :])
    - Permuted tensors (e.g. top4.permute(0, 3, 2, 1))
    - Expanded tensors with zero stride (e.g. mask.expand(N, 3))
    - Custom as_strided views
    """
    N = 64
    d_model = 128
    H = 3
    num_layers = 6
    num_experts = 16
    top_k = 4

    # 1. Transposed 2D tensor: shape (d_model, N) transposed to (N, d_model)
    raw_h = torch.randn(d_model, N, dtype=torch.float32)
    h_non_contig = raw_h.t()
    assert not h_non_contig.is_contiguous()

    # 2. Sliced with step 2 along dim 0 and dim 2
    raw_logits = torch.randn(N * 2, H, num_layers * 2, num_experts, dtype=torch.float32)
    logits_non_contig = raw_logits[::2, :, ::2, :]
    assert not logits_non_contig.is_contiguous()

    # 3. Permuted 4D tensor: (N, top_k, num_layers, H) permuted to (N, H, num_layers, top_k)
    raw_top4 = torch.randint(0, num_experts, (N, top_k, num_layers, H), dtype=torch.int64)
    top4_non_contig = raw_top4.permute(0, 3, 2, 1)
    assert not top4_non_contig.is_contiguous()

    # 4. Expanded tensor with stride 0 along dim 0
    raw_mask = torch.tensor([[True, True, False]], dtype=torch.bool)
    mask_expanded = raw_mask.expand(N, H)
    assert not mask_expanded.is_contiguous()

    # 5. Strided tensor via as_strided
    base_storage = torch.randn(N * d_model * 2)
    h_as_strided = torch.as_strided(base_storage, size=(N, d_model), stride=(d_model * 2, 1))
    assert not h_as_strided.is_contiguous()

    tensors = {
        "hidden_states": h_as_strided,
        "target_router_logits": logits_non_contig,
        "target_top4_indices": top4_non_contig,
        "valid_mask": mask_expanded,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = Path(tmpdir) / "strided_stress.safetensors"
        save_dataset_safetensors(tensors, fpath, metadata={"test": "strange_strides"})

        # Load back and verify contiguous layout + exact numerical equality
        loaded = load_dataset_safetensors(fpath)

        for k in tensors.keys():
            assert loaded[k].is_contiguous(), f"Loaded tensor '{k}' is not contiguous!"
            assert torch.equal(loaded[k], tensors[k]), f"Tensor '{k}' values altered during serialization!"


def test_safetensors_contract_schema_rejection():
    """Verify strict validation and rejection of malformed or schema-violating tensors."""
    valid_data = {
        "hidden_states": torch.randn(10, 64),
        "target_router_logits": torch.randn(10, 3, 4, 16),
        "target_top4_indices": torch.zeros(10, 3, 4, 4, dtype=torch.int64),
        "valid_mask": torch.ones(10, 3, dtype=torch.bool),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = Path(tmpdir) / "invalid.safetensors"

        # Missing required key
        for key in valid_data.keys():
            corrupt = {k: v for k, v in valid_data.items() if k != key}
            with pytest.raises(KeyError, match="Missing required contract tensors"):
                save_dataset_safetensors(corrupt, fpath)

        # Mismatched sample count
        mismatched = dict(valid_data)
        mismatched["hidden_states"] = torch.randn(11, 64)  # 11 != 10
        with pytest.raises(ValueError, match="Sample count mismatch"):
            save_dataset_safetensors(mismatched, fpath)

        # Invalid dimensionality
        dim_corrupt = dict(valid_data)
        dim_corrupt["hidden_states"] = torch.randn(10, 64, 1)  # 3D instead of 2D
        with pytest.raises(ValueError, match="hidden_states must be 2D"):
            save_dataset_safetensors(dim_corrupt, fpath)

        dim_corrupt_logits = dict(valid_data)
        dim_corrupt_logits["target_router_logits"] = torch.randn(10, 3, 4)  # 3D instead of 4D
        with pytest.raises(ValueError, match="target_router_logits must be 4D"):
            save_dataset_safetensors(dim_corrupt_logits, fpath)


@pytest.mark.parametrize("batch_size", [1, 2, 7, 32, 64, 128, 512, 1024, 4096])
@pytest.mark.parametrize("in_memory", [True, False])
def test_moe_calibration_dataset_dataloader_batch_scaling(batch_size: int, in_memory: bool):
    """Stress test PyTorch DataLoader iteration across extreme batch sizes and modes.
    
    Tests:
    - B=1 (minimal) to B=4096 (exceeding dataset size)
    - in_memory=True (RAM backed) vs in_memory=False (safe_open memory-mapped)
    - Full iteration integrity, batch shapes, and collate correctness
    """
    N = 256
    d_model = 64
    H = 3
    num_layers = 4
    num_experts = 16
    top_k = 4

    data = {
        "hidden_states": torch.randn(N, d_model),
        "target_router_logits": torch.randn(N, H, num_layers, num_experts),
        "target_top4_indices": torch.randint(0, num_experts, (N, H, num_layers, top_k), dtype=torch.int64),
        "valid_mask": torch.ones(N, H, dtype=torch.bool),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = Path(tmpdir) / "loader_stress.safetensors"
        save_dataset_safetensors(data, fpath)

        dataset = MoECalibrationDataset(filepath=fpath, in_memory=in_memory)
        assert len(dataset) == N
        assert dataset.d_model == d_model
        assert dataset.num_horizons == H
        assert dataset.num_deep_layers == num_layers
        assert dataset.num_experts == num_experts

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_samples_iterated = 0

        for batch in loader:
            bs = batch.hidden_states.shape[0]
            total_samples_iterated += bs
            assert batch.hidden_states.shape == (bs, d_model)
            assert batch.target_router_logits.shape == (bs, H, num_layers, num_experts)
            assert batch.target_top4_indices.shape == (bs, H, num_layers, top_k)
            assert batch.valid_mask.shape == (bs, H)

        assert total_samples_iterated == N


def test_moe_calibration_dataset_negative_indexing_and_slicing():
    """Verify negative indexing and slice operations on MoECalibrationDataset."""
    N = 50
    data = {
        "hidden_states": torch.randn(N, 32),
        "target_router_logits": torch.randn(N, 3, 2, 8),
        "target_top4_indices": torch.zeros(N, 3, 2, 4, dtype=torch.int64),
        "valid_mask": torch.ones(N, 3, dtype=torch.bool),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = Path(tmpdir) / "slice_stress.safetensors"
        save_dataset_safetensors(data, fpath)

        # In-memory mode
        ds_mem = MoECalibrationDataset(filepath=fpath, in_memory=True)
        assert torch.equal(ds_mem[-1].hidden_states, data["hidden_states"][-1])
        assert torch.equal(ds_mem[-5].hidden_states, data["hidden_states"][-5])

        sliced_ds = ds_mem[10:30]
        assert isinstance(sliced_ds, MoECalibrationDataset)
        assert len(sliced_ds) == 20
        assert torch.equal(sliced_ds.hidden_states, data["hidden_states"][10:30])

        # Memory-mapped mode
        ds_mmap = MoECalibrationDataset(filepath=fpath, in_memory=False)
        assert torch.equal(ds_mmap[-1].hidden_states, data["hidden_states"][-1])
        assert torch.equal(ds_mmap[-5].hidden_states, data["hidden_states"][-5])

        # Slice indexing on mmap must raise RuntimeError
        with pytest.raises(RuntimeError, match="Slice indexing requires in-memory dataset"):
            _ = ds_mmap[10:30]


def test_moe_calibration_dataset_filter_and_subselect():
    """Verify filter_valid, select_layers, and select_horizon functional methods."""
    N = 20
    valid_mask = torch.ones(N, 3, dtype=torch.bool)
    valid_mask[5, 2] = False  # sample 5 invalid at horizon 2
    valid_mask[10, :] = False  # sample 10 completely invalid

    data = {
        "hidden_states": torch.randn(N, 32),
        "target_router_logits": torch.randn(N, 3, 6, 8),
        "target_top4_indices": torch.zeros(N, 3, 6, 4, dtype=torch.int64),
        "valid_mask": valid_mask,
    }

    ds = MoECalibrationDataset(
        hidden_states=data["hidden_states"],
        target_router_logits=data["target_router_logits"],
        target_top4_indices=data["target_top4_indices"],
        valid_mask=data["valid_mask"],
    )

    # 1. filter_valid: drops samples 5 and 10 -> exactly 18 samples remain
    filtered = ds.filter_valid()
    assert len(filtered) == 18
    assert filtered.valid_mask.all()

    # 2. select_layers: select layers [1, 3, 5]
    layer_sliced = ds.select_layers([1, 3, 5])
    assert layer_sliced.num_deep_layers == 3
    assert layer_sliced.target_router_logits.shape[2] == 3
    assert torch.equal(layer_sliced.target_router_logits, data["target_router_logits"][:, :, [1, 3, 5], :])

    # 3. select_horizon: select horizon index 1 (T+2)
    horizon_sliced = ds.select_horizon(1)
    assert horizon_sliced.num_horizons == 1
    assert horizon_sliced.target_router_logits.shape[1] == 1
    assert torch.equal(horizon_sliced.valid_mask, valid_mask[:, 1:2])


# ===========================================================================
# 3. Gradient Isolation on Masked Boundary Tokens Stress Tests
# ===========================================================================

def test_gradient_isolation_tail_boundary_token_exact_zero_norm():
    """Adversarial Oracle: Assert that token L-1 hidden state receives EXACTLY 0.0 gradient norm.
    
    When sequences are aligned without dropping boundary tokens, token L-1 has
    valid_mask = [False, False, False]. Training loss computed with valid_mask
    must produce zero gradient on token L-1 hidden state.
    """
    L = 16
    d_model = 64
    num_layers = 4
    num_experts = 16
    horizons = (1, 2, 3)

    hidden_states = torch.randn(L, d_model, requires_grad=True)
    router_logits = torch.randn(L, num_layers, num_experts)

    aligned = align_sequence_targets(
        hidden_states=hidden_states,
        router_logits=router_logits,
        deep_layer_start=1,
        deep_layer_end=4,
        horizons=horizons,
        drop_boundary_tokens=False,
    )

    # Speculative projection head: d_model -> (H * layers * experts)
    head = nn.Linear(d_model, len(horizons) * num_layers * num_experts)
    pred_logits = head(hidden_states).view(L, len(horizons), num_layers, num_experts)

    # Masked Cross-Entropy / MSE loss
    target = aligned.target_router_logits
    mask = aligned.valid_mask  # (L, 3)
    expanded_mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(pred_logits)

    loss = ((pred_logits - target) ** 2 * expanded_mask.float()).sum() / expanded_mask.float().sum().clamp(min=1)
    loss.backward()

    # 1. Gradient for token L-1 (index 15) must have EXACTLY 0.0 L2 norm
    grad_norm_tail = hidden_states.grad[-1].norm().item()
    assert grad_norm_tail == 0.0, f"Leakage detected! Tail token gradient norm: {grad_norm_tail}"

    # 2. Gradient for valid tokens (e.g. token 0) must be strictly non-zero
    grad_norm_head = hidden_states.grad[0].norm().item()
    assert grad_norm_head > 0.0, "Valid token should receive non-zero gradient!"


def test_gradient_isolation_deep_layers_across_backbone():
    """Adversarial Oracle: verify 0.0 gradient norm across a 6-layer deep network from masked tokens.
    
    Feeds a batch containing ONLY masked boundary tokens (where valid_mask is all False).
    Asserts:
    - Loss evaluates cleanly to 0.0 without NaN
    - Autograd backwards computes 0.0 gradient norm across ALL deep layer weights and biases
    """
    d_model = 32
    deep_backbone = nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Linear(d_model, d_model),
    )

    B = 8  # 8 masked boundary samples
    x = torch.randn(B, d_model)
    h_N = deep_backbone(x)

    # All-false mask simulating boundary tokens
    valid_mask = torch.zeros(B, 3, dtype=torch.bool)
    target = torch.randn(B, 3, 4, 16)

    head = nn.Linear(d_model, 3 * 4 * 16)
    pred = head(h_N).view(B, 3, 4, 16)

    loss_elem = (pred - target) ** 2
    expanded_mask = valid_mask.unsqueeze(-1).unsqueeze(-1).expand_as(loss_elem)
    masked_loss = (loss_elem * expanded_mask.float()).sum() / expanded_mask.float().sum().clamp(min=1)

    assert masked_loss.item() == 0.0
    masked_loss.backward()

    # Verify zero gradient across all backbone layers
    total_backbone_grad_norm = 0.0
    for name, param in deep_backbone.named_parameters():
        if param.grad is not None:
            norm = param.grad.norm().item()
            total_backbone_grad_norm += norm
            assert norm == 0.0, f"Deep backbone layer '{name}' received non-zero grad {norm} from masked tokens!"

    assert total_backbone_grad_norm == 0.0


def test_gradient_isolation_horizon_granularity():
    """Verify granular horizon-level gradient isolation for tokens L-3 and L-2.
    
    - Token L-3: T+1, T+2 are valid; T+3 is invalid.
      Gradients with respect to T+3 speculative logits MUST be identically 0.0.
    - Token L-2: T+1 is valid; T+2, T+3 are invalid.
      Gradients with respect to T+2 and T+3 speculative logits MUST be identically 0.0.
    """
    L = 8
    d_model = 32
    num_layers = 2
    num_experts = 8

    hidden_states = torch.randn(L, d_model)
    router_logits = torch.randn(L, num_layers, num_experts)

    aligned = align_sequence_targets(
        hidden_states=hidden_states,
        router_logits=router_logits,
        deep_layer_start=1,
        deep_layer_end=2,
        horizons=(1, 2, 3),
        drop_boundary_tokens=False,
    )

    # Speculative logits directly as a leaf tensor to inspect per-horizon gradients
    pred_logits = torch.randn(L, 3, num_layers, num_experts, requires_grad=True)
    target = aligned.target_router_logits
    mask = aligned.valid_mask  # shape (8, 3)

    expanded_mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(pred_logits)
    loss = ((pred_logits - target) ** 2 * expanded_mask.float()).sum() / expanded_mask.float().sum().clamp(min=1)
    loss.backward()

    # Token L-3 (idx 5): T+3 (idx 2) is invalid
    assert pred_logits.grad[5, 2].norm().item() == 0.0
    assert pred_logits.grad[5, 0].norm().item() > 0.0
    assert pred_logits.grad[5, 1].norm().item() > 0.0

    # Token L-2 (idx 6): T+2 (idx 1) and T+3 (idx 2) are invalid
    assert pred_logits.grad[6, 1].norm().item() == 0.0
    assert pred_logits.grad[6, 2].norm().item() == 0.0
    assert pred_logits.grad[6, 0].norm().item() > 0.0

    # Token L-1 (idx 7): All invalid
    assert pred_logits.grad[7].norm().item() == 0.0
