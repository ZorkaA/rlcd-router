"""Differentiable RKHS MMCE Penalty and Soft Cross-Entropy Loss.

Implements the two-component calibration training objective from the project
specification:

1. **Soft Cross-Entropy (CE)**: Computes the soft cross-entropy between
   predicted speculative logits and native router probability distributions
   across all 20 deep layers and 3 horizons.

2. **RKHS Maximum Mean Calibration Error (MMCE)**: A differentiable
   calibration penalty based on Kumar et al. (ICML 2018), using a Gaussian
   RBF kernel (default σ=0.2) and inverse class weighting.

Combined loss: ``total = CE + λ·MMCE`` with tunable λ (default 1.0).

Reference:
    Kumar, Sarawagi, Jain. *Trainable Calibration Measures For Neural
    Networks From Kernel Mean Embeddings*, ICML 2018.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    DEFAULT_MMCE_LAMBDA,
    DEFAULT_MMCE_SIGMA,
    NUM_DEEP_LAYERS,
    NUM_EXPERTS,
    NUM_HORIZONS,
)

logger = logging.getLogger(__name__)


# =====================================================================
# 1. Soft Cross-Entropy Loss
# =====================================================================

def soft_cross_entropy_loss(
    pred_logits: torch.Tensor,
    target_logits: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute soft cross-entropy loss against native router distributions.

    The target router logits are converted to probability distributions via
    softmax, and the loss is the KL-divergence-equivalent soft CE:

        CE = -Σ_e target_prob_e · log_softmax(pred_logit_e)

    averaged over all valid (sample, horizon, layer) combinations.

    Args:
        pred_logits: Predicted speculative logits, shape
            ``(batch, num_horizons, num_deep_layers, num_experts)``.
        target_logits: Ground-truth native router logits, same shape.
        valid_mask: Optional boolean mask of shape ``(batch, num_horizons)``
            indicating which (sample, horizon) pairs are valid.  If ``None``,
            all pairs are treated as valid.

    Returns:
        Scalar tensor with the mean soft cross-entropy loss.

    Raises:
        ValueError: If input shapes do not match or have wrong dimensionality.
    """
    if pred_logits.shape != target_logits.shape:
        raise ValueError(
            f"Shape mismatch: pred_logits {pred_logits.shape} vs "
            f"target_logits {target_logits.shape}"
        )
    if pred_logits.dim() != 4:
        raise ValueError(
            f"Expected 4-D tensors (batch, horizons, layers, experts), "
            f"got {pred_logits.dim()}-D"
        )

    batch, num_horizons, num_layers, num_experts = pred_logits.shape

    # Target distributions via softmax over the expert dimension
    target_probs = F.softmax(target_logits, dim=-1)  # (B, H, L, E)

    # Predicted log-probabilities
    log_probs = F.log_softmax(pred_logits, dim=-1)   # (B, H, L, E)

    # Per-element soft CE: -sum over experts -> (B, H, L)
    per_position_ce = -torch.sum(target_probs * log_probs, dim=-1)

    if valid_mask is not None:
        if valid_mask.shape != (batch, num_horizons):
            raise ValueError(
                f"valid_mask shape {valid_mask.shape} does not match "
                f"expected ({batch}, {num_horizons})"
            )
        # Expand mask to cover layer dimension: (B, H, 1) -> broadcast over L
        mask_expanded = valid_mask.unsqueeze(-1).float()  # (B, H, 1)
        masked_ce = per_position_ce * mask_expanded       # (B, H, L)

        # Mean over valid positions only
        num_valid = mask_expanded.sum() * num_layers
        if num_valid > 0:
            ce_loss = masked_ce.sum() / num_valid
        else:
            ce_loss = torch.tensor(0.0, device=pred_logits.device, dtype=pred_logits.dtype)
    else:
        ce_loss = per_position_ce.mean()

    return ce_loss


# =====================================================================
# 2. RKHS MMCE Penalty
# =====================================================================

def rkhs_mmce_penalty(
    pred_logits: torch.Tensor,
    target_logits: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    sigma: float = DEFAULT_MMCE_SIGMA,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute the differentiable RKHS Maximum Mean Calibration Error.

    Implements the MMCE formulation from Kumar et al. (ICML 2018):

        MMCE = sqrt(max(0, e^T K e / m^2) + ε)

    where:
        - e_i = correctness_i - confidence_i  (calibration residual)
        - K_{i,j} = exp(-(c_i - c_j)^2 / (2σ^2))  (Gaussian RBF kernel)
        - m = number of samples
        - Inverse class weighting is applied to balance the penalty across
          expert classes with different prevalences.

    The "confidence" is the predicted probability for each expert, and
    "correctness" is 1 if the expert was in the native top-k routing decision,
    0 otherwise.  We compute MMCE *per (horizon, layer)* and average.

    Args:
        pred_logits: Predicted speculative logits, shape
            ``(batch, num_horizons, num_deep_layers, num_experts)``.
        target_logits: Ground-truth native router logits, same shape.
        valid_mask: Optional ``(batch, num_horizons)`` boolean mask.
        sigma: Gaussian RBF bandwidth (default 0.2).
        eps: Small constant for numerical stability inside sqrt.

    Returns:
        Scalar MMCE penalty tensor.
    """
    if pred_logits.shape != target_logits.shape:
        raise ValueError(
            f"Shape mismatch: pred_logits {pred_logits.shape} vs "
            f"target_logits {target_logits.shape}"
        )
    if pred_logits.dim() != 4:
        raise ValueError(
            f"Expected 4-D tensors (batch, horizons, layers, experts), "
            f"got {pred_logits.dim()}-D"
        )

    batch, num_horizons, num_layers, num_experts = pred_logits.shape

    # Predicted probabilities (confidence)
    pred_probs = F.softmax(pred_logits, dim=-1)        # (B, H, L, E)

    # Ground-truth binary correctness: 1 for the top-k selected experts
    # Use the native router logits to determine which experts are selected
    target_probs = F.softmax(target_logits, dim=-1)    # (B, H, L, E)

    # "Correctness" per expert = target probability (soft matching)
    # This allows gradient flow through the MMCE penalty
    correctness = target_probs

    # Build validity mask per (batch, horizon) -> expand to cover layers
    if valid_mask is not None:
        validity = valid_mask.float()  # (B, H)
    else:
        validity = torch.ones(batch, num_horizons, device=pred_logits.device)

    mmce_accum = torch.tensor(0.0, device=pred_logits.device, dtype=pred_logits.dtype)
    num_valid_positions = 0

    for h in range(num_horizons):
        for l in range(num_layers):
            # Mask to valid batch elements for this horizon
            v = validity[:, h]  # (B,)
            valid_indices = v > 0.5
            if valid_indices.sum() == 0:
                continue

            # Gather confidences and correctness for valid samples
            conf = pred_probs[valid_indices, h, l, :]  # (m, E)
            corr = correctness[valid_indices, h, l, :]  # (m, E)

            # Flatten across samples and experts for kernel computation
            c_flat = conf.reshape(-1)    # (m*E,)
            r_flat = corr.reshape(-1)    # (m*E,)

            m = c_flat.size(0)
            if m == 0:
                continue

            # Inverse class weighting: weight = 1 / (class_prevalence + eps)
            # Classes with low prevalence get upweighted
            mean_corr = r_flat.mean()
            # Weight positive and negative classes inversely
            w_pos = 1.0 / (mean_corr + eps)
            w_neg = 1.0 / (1.0 - mean_corr + eps)
            weights = torch.where(r_flat > 0.5, w_pos, w_neg)
            # Normalise weights to sum to m
            weights = weights * (m / (weights.sum() + eps))

            # Calibration residuals
            e = r_flat - c_flat  # (m,)
            e_weighted = e * weights

            # Gaussian RBF kernel matrix
            diffs = c_flat.unsqueeze(0) - c_flat.unsqueeze(1)  # (m, m)
            K = torch.exp(-(diffs ** 2) / (2.0 * sigma ** 2))   # (m, m)

            # Quadratic form: e^T K e / m^2
            quad = torch.matmul(
                e_weighted.unsqueeze(0),
                torch.matmul(K, e_weighted.unsqueeze(1)),
            ).squeeze()
            quad_normalised = quad / (float(m) ** 2)

            mmce_val = torch.sqrt(torch.clamp(quad_normalised, min=0.0) + eps)
            mmce_accum = mmce_accum + mmce_val
            num_valid_positions += 1

    if num_valid_positions > 0:
        mmce_accum = mmce_accum / float(num_valid_positions)

    return mmce_accum


# =====================================================================
# 3. Combined Loss
# =====================================================================

def combined_loss(
    pred_logits: torch.Tensor,
    target_logits: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    mmce_lambda: float = DEFAULT_MMCE_LAMBDA,
    mmce_sigma: float = DEFAULT_MMCE_SIGMA,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the combined calibration training loss: CE + λ·MMCE.

    Args:
        pred_logits: Predicted speculative logits ``(B, H, L, E)``.
        target_logits: Ground-truth native router logits ``(B, H, L, E)``.
        valid_mask: Optional ``(B, H)`` boolean mask.
        mmce_lambda: Weighting coefficient for the MMCE penalty (default 1.0).
        mmce_sigma: RBF kernel bandwidth for MMCE (default 0.2).

    Returns:
        Tuple of ``(total_loss, ce_loss, mmce_loss)`` scalar tensors.
    """
    ce = soft_cross_entropy_loss(pred_logits, target_logits, valid_mask)
    mmce = rkhs_mmce_penalty(pred_logits, target_logits, valid_mask, sigma=mmce_sigma)
    total = ce + mmce_lambda * mmce

    return total, ce, mmce


class CombinedCalibrationLoss(nn.Module):
    """Module wrapper for the combined CE + λ·MMCE training loss.

    Convenient for use in standard PyTorch training loops where losses
    are expected to be ``nn.Module`` instances.

    Args:
        mmce_lambda: Weighting coefficient for the MMCE penalty.
        mmce_sigma: Gaussian RBF kernel bandwidth.
    """

    def __init__(
        self,
        mmce_lambda: float = DEFAULT_MMCE_LAMBDA,
        mmce_sigma: float = DEFAULT_MMCE_SIGMA,
    ) -> None:
        super().__init__()
        self.mmce_lambda = mmce_lambda
        self.mmce_sigma = mmce_sigma

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_logits: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute combined loss.

        Returns:
            Tuple of ``(total_loss, ce_loss, mmce_loss)`` scalar tensors.
        """
        return combined_loss(
            pred_logits,
            target_logits,
            valid_mask,
            mmce_lambda=self.mmce_lambda,
            mmce_sigma=self.mmce_sigma,
        )
