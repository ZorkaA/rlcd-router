"""Pytest configuration and shared fixtures for Asynchronous MoE Router Phase 1 Calibration.
Provides fast synthetic Qwen2Moe fixtures, data fixtures, and mathematical reference oracles.
"""

import os
import sys
import importlib
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure workspace root is on sys.path
WORKSPACE_ROOT = Path(__file__).parent.parent.resolve()
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


# ---------------------------------------------------------------------------
# Module import helper for progressive milestone testing
# ---------------------------------------------------------------------------

def safe_import(module_path: str, object_name: str = None) -> Any:
    """Safely import a module or object from src/.
    If the module is not yet implemented, gracefully skip the test.
    """
    try:
        mod = importlib.import_module(module_path)
        if object_name is not None:
            if not hasattr(mod, object_name):
                pytest.skip(f"Object '{object_name}' not yet implemented in {module_path}")
            return getattr(mod, object_name)
        return mod
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Module '{module_path}' not yet implemented in current milestone: {e}")


# ---------------------------------------------------------------------------
# Device and Configuration Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def default_device() -> str:
    """Return 'mps' if Apple Silicon MPS is available, else 'cpu'."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@pytest.fixture(scope="session")
def synthetic_qwen_config():
    """Fast, lightweight Qwen2MoeConfig for offline testing (<50MB, no 28GB download).
    6 layers, 16 experts, top-4 selection, hidden size 64.
    """
    from transformers import Qwen2MoeConfig
    return Qwen2MoeConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=16,
        num_experts_per_tok=4,
        shared_expert_intermediate_size=128,
        output_router_logits=True,
        torch_dtype="float32",
    )


@pytest.fixture(scope="session")
def synthetic_qwen_model(synthetic_qwen_config):
    """Instantiate a real Qwen2MoeForCausalLM model with synthetic config on CPU."""
    from transformers import Qwen2MoeForCausalLM
    model = Qwen2MoeForCausalLM(synthetic_qwen_config)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Interface Contract Data Fixtures (M1 <-> M2 <-> M3 <-> M4)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_m1_m2_batch() -> Dict[str, torch.Tensor]:
    """Mock batch strictly conforming to M1 <-> M2 Interface Contract in PROJECT.md.
    - hidden_states: (num_samples, 2048) float32
    - target_router_logits: (num_samples, 3, 20, 60) float32
    - target_top4_indices: (num_samples, 3, 20, 4) int64
    - valid_mask: (num_samples, 3) bool
    """
    num_samples = 32
    torch.manual_seed(42)

    hidden_states = torch.randn(num_samples, 2048, dtype=torch.float32)
    target_router_logits = torch.randn(num_samples, 3, 20, 60, dtype=torch.float32)
    target_top4_indices = torch.topk(target_router_logits, k=4, dim=-1).indices
    valid_mask = torch.ones(num_samples, 3, dtype=torch.bool)

    # Simulate trailing sequence boundaries where future horizons extend past sequence end
    valid_mask[-1, :] = torch.tensor([True, False, False])  # T+1 valid, T+2/T+3 invalid
    valid_mask[-2, :] = torch.tensor([True, True, False])   # T+1, T+2 valid, T+3 invalid

    return {
        "hidden_states": hidden_states,
        "target_router_logits": target_router_logits,
        "target_top4_indices": target_top4_indices,
        "valid_mask": valid_mask,
    }


@pytest.fixture
def synthetic_safetensors_file(tmp_path, mock_m1_m2_batch) -> Path:
    """Save mock M1 dataset to temporary .safetensors file and return path."""
    import safetensors.torch
    file_path = tmp_path / "train_data.safetensors"
    # Convert bool tensor to int8 for safetensors compatibility if needed, or direct bool
    tensors_to_save = {
        "hidden_states": mock_m1_m2_batch["hidden_states"],
        "target_router_logits": mock_m1_m2_batch["target_router_logits"],
        "target_top4_indices": mock_m1_m2_batch["target_top4_indices"],
        "valid_mask": mock_m1_m2_batch["valid_mask"].to(torch.uint8),
    }
    safetensors.torch.save_file(tensors_to_save, str(file_path))
    return file_path


@pytest.fixture
def mock_speculative_logits() -> torch.Tensor:
    """Uncalibrated speculative logits of shape (N, 3, 20, 60)."""
    torch.manual_seed(123)
    return torch.randn(32, 3, 20, 60, dtype=torch.float32)


@pytest.fixture
def mock_temperature_grid() -> torch.Tensor:
    """2x3 temperature scalar grid (early/late x T+1..T+3)."""
    return torch.tensor([
        [1.15, 1.25, 1.40],  # Early layers 5-10: T+1, T+2, T+3
        [1.05, 1.10, 1.20],  # Late layers 11-24: T+1, T+2, T+3
    ], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Authoritative Mathematical Reference Oracles
# ---------------------------------------------------------------------------

class MathematicalOracles:
    """Authoritative mathematical reference implementations for verification."""

    @staticmethod
    def rkhs_mmce(
        confidences: torch.Tensor,
        correctness: torch.Tensor,
        sigma: float = 0.2,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Kumar et al. (ICML 2018) RKHS MMCE formulation:
        MMCE = sqrt(max(0, e^T K e / m^2) + eps)
        where e_i = r_i - c_i, K_{i,j} = exp(-(c_i - c_j)^2 / (2*sigma^2)).
        """
        assert confidences.ndim == 1 and correctness.ndim == 1
        m = confidences.size(0)
        if m == 0:
            return torch.tensor(0.0)

        e = correctness.float() - confidences.float()  # (m,)
        diffs = confidences.unsqueeze(0) - confidences.unsqueeze(1)  # (m, m)
        k_mat = torch.exp(-(diffs ** 2) / (2.0 * (sigma ** 2)))     # (m, m)

        e_col = e.unsqueeze(1)  # (m, 1)
        quad = torch.matmul(e.unsqueeze(0), torch.matmul(k_mat, e_col)).squeeze()
        quad_normalized = quad / (float(m) ** 2)
        return torch.sqrt(torch.clamp(quad_normalized, min=0.0) + eps)

    @staticmethod
    def soft_cross_entropy(
        pred_logits: torch.Tensor,
        target_distribution: torch.Tensor,
    ) -> torch.Tensor:
        """Soft Cross-Entropy loss: -sum_e (target_e * log_softmax(pred_e))."""
        log_probs = F.log_softmax(pred_logits, dim=-1)
        return -torch.sum(target_distribution * log_probs, dim=-1).mean()

    @staticmethod
    def nll_l2_regularized_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        temperature: torch.Tensor,
        alpha_l2: float = 0.01,
    ) -> torch.Tensor:
        """Calibrated NLL + L2 penalty toward T=1.0.
        loss = NLL(logits / T, targets) + 0.5 * alpha_l2 * (T - 1.0)^2.
        """
        scaled_logits = logits / temperature
        nll = F.cross_entropy(scaled_logits, targets)
        reg = 0.5 * alpha_l2 * torch.sum((temperature - 1.0) ** 2)
        return nll + reg

    @staticmethod
    def apply_temperature_grid(
        logits: torch.Tensor,
        temperatures: torch.Tensor,
    ) -> torch.Tensor:
        """Apply 2x3 temperature grid to (N, 3, 20, 60) logits.
        Early layers: 0..5 (Layers 5-10, 6 layers).
        Late layers: 6..19 (Layers 11-24, 14 layers).
        """
        out = logits.clone()
        for h in range(3):
            t_early = temperatures[0, h]
            t_late = temperatures[1, h]
            out[:, h, :6, :] = out[:, h, :6, :] / t_early
            out[:, h, 6:, :] = out[:, h, 6:, :] / t_late
        return out

    @staticmethod
    def targeted_ece(
        calibrated_probs: torch.Tensor,
        ground_truth_top4: torch.Tensor,
        threshold: float = 0.05,
        window: float = 0.025,
    ) -> Tuple[float, int]:
        """Compute localized Targeted ECE in [threshold - window, threshold + window].
        calibrated_probs: (N, num_experts)
        ground_truth_top4: (N, 4) - indices of true top 4 experts
        """
        N, num_experts = calibrated_probs.shape
        # Create binary ground truth matrix (N, num_experts)
        gt_binary = torch.zeros_like(calibrated_probs)
        for i in range(N):
            gt_binary[i, ground_truth_top4[i]] = 1.0

        p_flat = calibrated_probs.flatten()
        y_flat = gt_binary.flatten()

        mask = (p_flat >= (threshold - window)) & (p_flat <= (threshold + window))
        count = int(mask.sum().item())
        if count == 0:
            return 0.0, 0

        bin_conf = p_flat[mask].mean().item()
        bin_acc = y_flat[mask].mean().item()
        return abs(bin_acc - bin_conf), count


@pytest.fixture(scope="session")
def oracles() -> MathematicalOracles:
    """Return the mathematical reference oracle instance."""
    return MathematicalOracles()
