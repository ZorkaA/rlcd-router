"""Empirical Adversarial Stress Testing for Milestone 1.

Executed by: teamwork_preview_challenger_m1_1 (EMPIRICAL CHALLENGER)
Focus Areas:
1. Zero-OOM streaming extractor over 100 sequences (CPU and MPS) tracking psutil RSS & torch.Tensor count.
2. Trailing token masking on extreme sequence lengths (L=1, 2, 3, 4, 0).
3. Device and dtype resolution safety (CPU vs MPS, bfloat16 rejection, invalid dtypes, cross-device transfers).
"""

import gc
import os
from pathlib import Path
import tempfile
import psutil
import pytest
import torch

from src.config import (
    resolve_device,
    resolve_dtype,
    validate_device_dtype,
    LOOKAHEAD_HORIZONS,
)
from src.data.model_loader import (
    get_synthetic_model,
    get_synthetic_tokenizer,
)
from src.data.stream_extractor import (
    StreamExtractor,
    StreamExtractorConfig,
    ExtractionBatch,
)
from src.data.dataset import (
    align_sequence_targets,
    build_and_split_calibration_datasets,
    partition_sequence_indices,
    MoECalibrationDataset,
    DatasetSplitConfig,
)


# =====================================================================
# 1. Zero-OOM Streaming Extractor Stress Tests (CPU & MPS)
# =====================================================================

def test_stream_extractor_100_sequences_cpu_zero_leak():
    """Stress test: Stream 100 sequences through StreamExtractor on CPU.
    Assert:
    - Active torch.Tensor delta is exactly 0 after completing the loop and GC.
    - RSS growth between sequence 20 and sequence 100 is strictly bounded (<20MB).
    """
    gc.collect()
    process = psutil.Process(os.getpid())

    model = get_synthetic_model(device="cpu", dtype="float32")
    config = StreamExtractorConfig(seq_len=128, cache_flush_interval=10, device="cpu")
    extractor = StreamExtractor(model, config=config)

    initial_tensors = sum(1 for o in gc.get_objects() if isinstance(o, torch.Tensor))
    initial_rss_mb = process.memory_info().rss / (1024 * 1024)

    rss_checkpoints = {}
    tensor_checkpoints = {}

    gen = extractor.stream_synthetic(num_sequences=100, seq_len=128, vocab_size=500, seed=123)
    for idx, batch in enumerate(gen):
        assert isinstance(batch, ExtractionBatch)
        assert batch.hidden_states.device.type == "cpu"
        assert batch.router_logits.device.type == "cpu"
        step = idx + 1
        if step % 20 == 0:
            gc.collect()
            rss_checkpoints[step] = process.memory_info().rss / (1024 * 1024)
            tensor_checkpoints[step] = sum(1 for o in gc.get_objects() if isinstance(o, torch.Tensor))
        del batch

    gc.collect()
    final_tensors = sum(1 for o in gc.get_objects() if isinstance(o, torch.Tensor))
    final_rss_mb = process.memory_info().rss / (1024 * 1024)

    # Invariant 1: Zero tensor accumulation
    tensor_delta = final_tensors - initial_tensors
    assert tensor_delta == 0, f"Leaked {tensor_delta} torch.Tensor objects on CPU!"

    # Invariant 2: Asymptotic bounded RSS growth (step 20 to 100 delta < 20MB)
    post_warmup_rss_delta = rss_checkpoints[100] - rss_checkpoints[20]
    assert post_warmup_rss_delta < 20.0, (
        f"Unbounded CPU memory growth! Seq 20->100 RSS grew by {post_warmup_rss_delta:.2f}MB "
        f"(Seq 20: {rss_checkpoints[20]:.2f}MB, Seq 100: {rss_checkpoints[100]:.2f}MB)"
    )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple Silicon MPS not available")
def test_stream_extractor_100_sequences_mps_zero_leak():
    """Stress test: Stream 100 sequences through StreamExtractor on MPS backend.
    Assert:
    - MPS current allocated memory does not grow across sequences (delta == 0.0 MB).
    - Active torch.Tensor delta is exactly 0.
    - RSS growth between sequence 40 and sequence 100 is strictly bounded (<50MB).
    """
    gc.collect()
    torch.mps.empty_cache()
    process = psutil.Process(os.getpid())

    model = get_synthetic_model(device="mps", dtype="float16")
    config = StreamExtractorConfig(seq_len=128, cache_flush_interval=10, device="mps")
    extractor = StreamExtractor(model, config=config)

    initial_mps_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
    initial_tensors = sum(1 for o in gc.get_objects() if isinstance(o, torch.Tensor))

    rss_checkpoints = {}
    mps_checkpoints = {}

    gen = extractor.stream_synthetic(num_sequences=100, seq_len=128, vocab_size=500, seed=456)
    for idx, batch in enumerate(gen):
        assert batch.hidden_states.device.type == "cpu"
        assert batch.router_logits.device.type == "cpu"
        step = idx + 1
        if step in (20, 40, 60, 80, 100):
            rss_checkpoints[step] = process.memory_info().rss / (1024 * 1024)
            mps_checkpoints[step] = torch.mps.current_allocated_memory() / (1024 * 1024)
        del batch

    gc.collect()
    torch.mps.empty_cache()
    final_mps_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
    final_tensors = sum(1 for o in gc.get_objects() if isinstance(o, torch.Tensor))

    # Invariant 1: Zero MPS memory accumulation
    mps_delta = final_mps_mb - initial_mps_mb
    assert abs(mps_delta) < 0.1, f"MPS memory leaked: delta={mps_delta:.2f}MB"

    # Invariant 2: Zero active tensor accumulation
    tensor_delta = final_tensors - initial_tensors
    assert tensor_delta == 0, f"Leaked {tensor_delta} torch.Tensor objects on MPS!"

    # Invariant 3: Post-warmup host RSS asymptotic stability (Seq 40 to 100)
    post_warmup_rss_delta = rss_checkpoints[100] - rss_checkpoints[40]
    assert post_warmup_rss_delta < 50.0, (
        f"Host RSS growth not bounded on MPS: {post_warmup_rss_delta:.2f}MB between seq 40 and 100."
    )


# =====================================================================
# 2. Extreme Sequence Length & Trailing Token Masking Stress Tests
# =====================================================================

@pytest.mark.parametrize("L", [1, 2, 3, 4])
@pytest.mark.parametrize("drop_boundary", [False, True])
def test_trailing_token_masking_extreme_lengths(L: int, drop_boundary: bool):
    """Stress test: align_sequence_targets with extreme short sequence lengths L in {1, 2, 3, 4}.
    Assert:
    - No IndexError or dimension mismatch is raised.
    - valid_mask strictly reflects lookahead horizon feasibility.
    - Zero out-of-bounds context contamination.
    """
    d_model = 32
    num_layers = 6
    num_experts = 16
    horizons = (1, 2, 3)
    top_k = 4

    h = torch.randn(L, d_model)
    r = torch.randn(L, num_layers, num_experts)

    aligned = align_sequence_targets(
        hidden_states=h,
        router_logits=r,
        deep_layer_start=3,
        deep_layer_end=6,
        horizons=horizons,
        top_k=top_k,
        drop_boundary_tokens=drop_boundary,
    )

    if drop_boundary:
        expected_len = max(0, L - max(horizons))
        assert aligned.hidden_states.shape[0] == expected_len
        assert aligned.target_router_logits.shape[0] == expected_len
        assert aligned.valid_mask.shape[0] == expected_len
        if expected_len > 0:
            assert torch.all(aligned.valid_mask)
    else:
        assert aligned.hidden_states.shape[0] == L
        assert aligned.target_router_logits.shape[0] == L
        assert aligned.valid_mask.shape == (L, len(horizons))

        # Check precise boolean truth table
        if L == 1:
            # L=1: cannot lookahead 1, 2, or 3
            assert torch.equal(aligned.valid_mask, torch.tensor([[False, False, False]]))
            # Unmasked logits must be zeroed out
            assert torch.equal(aligned.target_router_logits, torch.zeros_like(aligned.target_router_logits))

        elif L == 2:
            expected = torch.tensor([
                [True, False, False],   # Token 0: T+1 valid
                [False, False, False],  # Token 1: none valid
            ])
            assert torch.equal(aligned.valid_mask, expected)

        elif L == 3:
            expected = torch.tensor([
                [True, True, False],    # Token 0: T+1, T+2 valid
                [True, False, False],   # Token 1: T+1 valid
                [False, False, False],  # Token 2: none valid
            ])
            assert torch.equal(aligned.valid_mask, expected)

        elif L == 4:
            expected = torch.tensor([
                [True, True, True],     # Token 0: T+1, T+2, T+3 valid
                [True, True, False],    # Token 1: T+1, T+2 valid
                [True, False, False],   # Token 2: T+1 valid
                [False, False, False],  # Token 3: none valid
            ])
            assert torch.equal(aligned.valid_mask, expected)


def test_trailing_token_masking_target_value_fidelity():
    """Verify target values in aligned logits exactly match future timestep logits."""
    L = 5
    d_model = 8
    num_layers = 6
    num_experts = 4
    h = torch.arange(L * d_model, dtype=torch.float32).view(L, d_model)
    r = torch.arange(L * num_layers * num_experts, dtype=torch.float32).view(L, num_layers, num_experts)

    aligned = align_sequence_targets(
        hidden_states=h,
        router_logits=r,
        deep_layer_start=3,
        deep_layer_end=6,
        horizons=(1, 2, 3),
        top_k=2,
        drop_boundary_tokens=False,
    )

    # Deep layers are indices 2..5
    deep_r = r[:, 2:6, :]

    # Token 0 targets: T+1 -> deep_r[1], T+2 -> deep_r[2], T+3 -> deep_r[3]
    assert torch.equal(aligned.target_router_logits[0, 0], deep_r[1])
    assert torch.equal(aligned.target_router_logits[0, 1], deep_r[2])
    assert torch.equal(aligned.target_router_logits[0, 2], deep_r[3])

    # Token 1 targets: T+1 -> deep_r[2], T+2 -> deep_r[3], T+3 -> deep_r[4]
    assert torch.equal(aligned.target_router_logits[1, 0], deep_r[2])
    assert torch.equal(aligned.target_router_logits[1, 1], deep_r[3])
    assert torch.equal(aligned.target_router_logits[1, 2], deep_r[4])

    # Token 2 targets: T+1 -> deep_r[3], T+2 -> deep_r[4], T+3 is invalid (0)
    assert torch.equal(aligned.target_router_logits[2, 0], deep_r[3])
    assert torch.equal(aligned.target_router_logits[2, 1], deep_r[4])
    assert torch.equal(aligned.target_router_logits[2, 2], torch.zeros_like(deep_r[0]))


def test_build_and_split_calibration_datasets_extreme_lengths(tmp_path):
    """Verify build_and_split_calibration_datasets handles extreme short sequences."""
    sequences = [
        (torch.randn(1, 32), torch.randn(1, 6, 8)),
        (torch.randn(2, 32), torch.randn(2, 6, 8)),
        (torch.randn(3, 32), torch.randn(3, 6, 8)),
        (torch.randn(4, 32), torch.randn(4, 6, 8)),
    ]

    cfg = DatasetSplitConfig(
        train_ratio=0.5,
        deep_layer_start=3,
        deep_layer_end=6,
        horizons=(1, 2, 3),
        top_k=2,
    )

    train_ds, calib_ds = build_and_split_calibration_datasets(
        sequences=sequences,
        output_dir=tmp_path,
        config=cfg,
    )

    assert len(train_ds) == 3   # 1 + 2 = 3 tokens
    assert len(calib_ds) == 7   # 3 + 4 = 7 tokens

    # Check safetensors files created and loadable
    loaded_train = MoECalibrationDataset.from_safetensors(tmp_path / "train_data.safetensors")
    assert len(loaded_train) == 3
    assert loaded_train.valid_mask.shape == (3, 3)


# =====================================================================
# 3. Device & Dtype Resolution Safety Stress Tests
# =====================================================================

def test_resolve_device_all_branches():
    """Verify resolve_device behavior across valid and invalid device specifications."""
    # None or 'auto'
    resolved_auto = resolve_device(None)
    assert resolved_auto.type in ("mps", "cuda", "cpu")

    # Explicit cpu
    dev_cpu = resolve_device("cpu")
    assert dev_cpu.type == "cpu"

    # Explicit torch.device
    dev_direct = resolve_device(torch.device("cpu"))
    assert dev_direct.type == "cpu"

    # Cuda unavailable rejection
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA backend requested but torch.cuda is not available"):
            resolve_device("cuda")

    # Invalid device string
    with pytest.raises(RuntimeError):
        resolve_device("invalid_device_name_xyz")


def test_resolve_dtype_cpu_comprehensive():
    """Verify resolve_dtype on CPU backend for all valid and invalid inputs."""
    # Defaults
    assert resolve_dtype("cpu", None) == torch.float32
    assert resolve_dtype("cpu", "auto") == torch.float32

    # String aliases
    assert resolve_dtype("cpu", "float16") == torch.float16
    assert resolve_dtype("cpu", "fp16") == torch.float16
    assert resolve_dtype("cpu", "float32") == torch.float32
    assert resolve_dtype("cpu", "fp32") == torch.float32
    assert resolve_dtype("cpu", "bfloat16") == torch.bfloat16
    assert resolve_dtype("cpu", "bf16") == torch.bfloat16

    # Unsupported string
    with pytest.raises(ValueError, match="Unsupported dtype string"):
        resolve_dtype("cpu", "complex64")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple Silicon MPS not available")
def test_resolve_dtype_mps_rejections_and_acceptances():
    """Verify resolve_dtype on MPS backend strictly rejects bfloat16 and unsupported dtypes."""
    # Defaults
    assert resolve_dtype("mps", None) == torch.float16
    assert resolve_dtype("mps", "auto") == torch.float16

    # Accepted dtypes
    assert resolve_dtype("mps", "float16") == torch.float16
    assert resolve_dtype("mps", "float32") == torch.float32
    assert resolve_dtype("mps", torch.float16) == torch.float16
    assert resolve_dtype("mps", torch.float32) == torch.float32

    # STRICT REJECTION: bfloat16 in all forms
    for bf16_input in ("bfloat16", "bf16", torch.bfloat16):
        with pytest.raises(ValueError, match="BFloat16 is not supported on Apple Silicon MPS backend"):
            resolve_dtype("mps", bf16_input)

    # STRICT REJECTION: unsupported non-float types
    for bad_dtype in ("int32", "int64", "float64"):
        with pytest.raises(ValueError):
            resolve_dtype("mps", bad_dtype)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple Silicon MPS not available")
def test_cross_device_execution_safety():
    """Verify model execution on MPS handles CPU inputs and offloads outputs to CPU in FP16."""
    model = get_synthetic_model(device="mps", dtype="float16")
    cfg = StreamExtractorConfig(seq_len=64, device="mps", dtype=torch.float16)
    extractor = StreamExtractor(model, config=cfg)

    cpu_input = torch.randint(0, 500, (1, 64), dtype=torch.int64, device="cpu")
    batch = extractor.extract_single_sequence(cpu_input)

    # Batch outputs must all be on CPU
    assert batch.input_ids.device.type == "cpu"
    assert batch.hidden_states.device.type == "cpu"
    assert batch.router_logits.device.type == "cpu"
    assert batch.top4_indices.device.type == "cpu"
    assert batch.top4_probs.device.type == "cpu"

    # Precision offloaded must match config
    assert batch.hidden_states.dtype == torch.float16
    assert batch.router_logits.dtype == torch.float16
    assert batch.top4_probs.dtype == torch.float16
    assert batch.top4_indices.dtype == torch.int64
