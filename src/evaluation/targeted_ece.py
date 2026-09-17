"""Targeted ECE (Expected Calibration Error) Evaluation for MoE Router.

Implements localized calibration metrics at two decision thresholds:
- 0.05 (abort threshold): measures calibration near the speculative abort boundary
- 0.85 (mass cutoff): measures calibration of cumulative top-4 expert probability mass

Results are stratified by layer bucket {early, late} × horizon {T+1, T+2, T+3}.

Reference: Naeini et al., "Obtaining Well Calibrated Probabilities Using
Bayesian Binning into Quantiles" (AAAI 2015) adapted with windowed localization.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from src.config import (
    ABORT_THRESHOLD,
    MASS_CUTOFF_THRESHOLD,
    TARGETED_ECE_BANDWIDTH,
    TARGETED_ECE_NUM_BINS,
    TARGETED_ECE_THRESHOLDS,
    NUM_DEEP_LAYERS,
    NUM_EARLY_LAYERS,
    NUM_LATE_LAYERS,
    NUM_HORIZONS,
    NUM_EXPERTS,
    GRID_BUCKETS,
    GRID_HORIZONS,
)

logger = logging.getLogger(__name__)


def _create_binary_ground_truth(
    ground_truth_top4: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Convert top-4 expert indices to binary indicator matrix.

    Args:
        ground_truth_top4: LongTensor of shape (..., 4) with expert indices.
        num_experts: Total number of experts (typically 60).

    Returns:
        Float tensor of the same leading shape + (num_experts,) with 1.0 at
        ground-truth expert positions and 0.0 elsewhere.
    """
    leading_shape = ground_truth_top4.shape[:-1]
    gt_binary = torch.zeros(*leading_shape, num_experts,
                            dtype=torch.float32,
                            device=ground_truth_top4.device)
    gt_binary.scatter_(-1, ground_truth_top4, 1.0)
    return gt_binary


def compute_targeted_ece_single(
    calibrated_probs: torch.Tensor,
    gt_binary: torch.Tensor,
    threshold: float,
    bandwidth: float = TARGETED_ECE_BANDWIDTH,
) -> Tuple[float, int]:
    """Compute windowed Targeted ECE around a single threshold.

    Selects all expert-level probability predictions within
    [threshold - bandwidth, threshold + bandwidth], then computes the
    absolute difference between mean confidence and mean accuracy in
    that window.

    Args:
        calibrated_probs: Probability tensor of shape (N, num_experts),
            each row sums to ~1.0.
        gt_binary: Binary ground-truth matrix of same shape, with 1.0
            for top-4 experts.
        threshold: Center of the evaluation window.
        bandwidth: Half-width of the evaluation window.

    Returns:
        Tuple of (targeted_ece_value, num_samples_in_window).
        Returns (0.0, 0) when no predictions fall in the window.
    """
    p_flat = calibrated_probs.flatten()
    y_flat = gt_binary.flatten()

    lo = max(threshold - bandwidth, 0.0)
    hi = min(threshold + bandwidth, 1.0)
    mask = (p_flat >= lo) & (p_flat <= hi)
    count = int(mask.sum().item())

    if count == 0:
        logger.debug(
            "No predictions in window [%.4f, %.4f] for threshold %.4f",
            lo, hi, threshold,
        )
        return 0.0, 0

    bin_conf = p_flat[mask].mean().item()
    bin_acc = y_flat[mask].mean().item()
    ece_val = abs(bin_acc - bin_conf)

    logger.debug(
        "Targeted ECE @ %.4f: ece=%.6f, conf=%.6f, acc=%.6f, n=%d",
        threshold, ece_val, bin_conf, bin_acc, count,
    )
    return ece_val, count


def compute_cumulative_mass_stats(
    calibrated_probs: torch.Tensor,
    ground_truth_top4: torch.Tensor,
) -> Dict[str, float]:
    """Compute cumulative probability mass statistics for top-4 experts.

    For each sample, sums the calibrated probability of the ground-truth
    top-4 experts to get the cumulative mass. Reports mean, std, min, max,
    and the fraction of samples where the mass exceeds the 0.85 cutoff.

    Args:
        calibrated_probs: (N, num_experts) calibrated probabilities.
        ground_truth_top4: (N, 4) indices of ground-truth top experts.

    Returns:
        Dictionary with cumulative mass statistics.
    """
    # Gather probabilities for the ground-truth top-4 experts
    gt_probs = torch.gather(calibrated_probs, dim=-1, index=ground_truth_top4)
    cum_mass = gt_probs.sum(dim=-1)  # (N,)

    stats = {
        "mean": float(cum_mass.mean().item()),
        "std": float(cum_mass.std().item()) if cum_mass.numel() > 1 else 0.0,
        "min": float(cum_mass.min().item()),
        "max": float(cum_mass.max().item()),
        "fraction_above_0.85": float((cum_mass >= MASS_CUTOFF_THRESHOLD).float().mean().item()),
        "num_samples": int(cum_mass.numel()),
    }

    logger.info(
        "Cumulative mass stats: mean=%.4f, std=%.4f, frac_above_0.85=%.4f",
        stats["mean"], stats["std"], stats["fraction_above_0.85"],
    )
    return stats


def _compute_overall_ece(
    calibrated_probs: torch.Tensor,
    gt_binary: torch.Tensor,
    num_bins: int = TARGETED_ECE_NUM_BINS,
) -> float:
    """Compute standard (global) ECE over the full probability range.

    Uses equal-width binning in [0, 1] with `num_bins` bins.

    Args:
        calibrated_probs: (N, num_experts) calibrated probability tensor.
        gt_binary: (N, num_experts) binary ground-truth tensor.
        num_bins: Number of equal-width bins.

    Returns:
        Scalar ECE value.
    """
    p_flat = calibrated_probs.flatten()
    y_flat = gt_binary.flatten()
    total = p_flat.numel()

    if total == 0:
        return 0.0

    bin_boundaries = torch.linspace(0.0, 1.0, num_bins + 1, device=p_flat.device)
    ece = 0.0

    for i in range(num_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        if i < num_bins - 1:
            mask = (p_flat >= lo) & (p_flat < hi)
        else:
            # Include right boundary in last bin
            mask = (p_flat >= lo) & (p_flat <= hi)

        count = int(mask.sum().item())
        if count == 0:
            continue

        bin_conf = p_flat[mask].mean().item()
        bin_acc = y_flat[mask].mean().item()
        ece += (count / total) * abs(bin_acc - bin_conf)

    return ece


def _get_layer_bucket_slice(bucket: str) -> slice:
    """Return the deep-layer dimension slice for a given bucket name.

    Args:
        bucket: One of 'early' or 'late'.

    Returns:
        Slice indexing into the 20-deep-layer dimension:
          - 'early': layers 5-10 -> indices 0..5 (6 layers)
          - 'late':  layers 11-24 -> indices 6..19 (14 layers)
    """
    if bucket == "early":
        return slice(0, NUM_EARLY_LAYERS)   # 0..5
    elif bucket == "late":
        return slice(NUM_EARLY_LAYERS, NUM_DEEP_LAYERS)  # 6..19
    else:
        raise ValueError(f"Unknown layer bucket '{bucket}'. Expected 'early' or 'late'.")


def _get_horizon_index(horizon: str) -> int:
    """Convert horizon label to integer index.

    Args:
        horizon: One of 'T+1', 'T+2', 'T+3'.

    Returns:
        Integer index (0, 1, or 2).
    """
    mapping = {"T+1": 0, "T+2": 1, "T+3": 2}
    if horizon not in mapping:
        raise ValueError(f"Unknown horizon '{horizon}'. Expected one of {list(mapping.keys())}.")
    return mapping[horizon]


def evaluate_targeted_ece(
    calibrated_probs: torch.Tensor,
    ground_truth_top4: torch.Tensor,
    thresholds: Optional[List[float]] = None,
    bandwidth: float = TARGETED_ECE_BANDWIDTH,
) -> dict:
    """Evaluate Targeted Expected Calibration Error across layer buckets and horizons.

    Computes localized calibration metrics at each threshold, broken down by
    {early, late} × {T+1, T+2, T+3}. Also computes cumulative mass statistics
    at the 0.85 threshold and an overall ECE.

    Args:
        calibrated_probs: Calibrated probability tensor of shape (N, 3, 20, 60).
            - Dim 1: Horizon (T+1, T+2, T+3).
            - Dim 2: Deep layer index (0=Layer 5, ..., 19=Layer 24).
            - Dim 3: Expert probabilities (should sum to ~1.0 per expert group).
        ground_truth_top4: LongTensor of shape (N, 3, 20, 4) with ground-truth
            top-4 expert indices.
        thresholds: List of threshold values. Defaults to [0.05, 0.85].
        bandwidth: Half-width of the evaluation window around each threshold.

    Returns:
        Dictionary with the following structure::

            {
                "targeted_ece_0.05": {
                    ("early", "T+1"): float,
                    ("early", "T+2"): float,
                    ("early", "T+3"): float,
                    ("late", "T+1"): float,
                    ("late", "T+2"): float,
                    ("late", "T+3"): float,
                },
                "targeted_ece_0.85": { ... },
                "cumulative_mass_0.85_stats": {
                    "mean": float, "std": float, "min": float, "max": float,
                    "fraction_above_0.85": float, "num_samples": int,
                },
                "overall_ece": float,
            }
    """
    if thresholds is None:
        thresholds = list(TARGETED_ECE_THRESHOLDS)

    # Validate input shapes
    if calibrated_probs.ndim != 4:
        raise ValueError(
            f"calibrated_probs must be 4D (N, 3, 20, num_experts), "
            f"got shape {calibrated_probs.shape}"
        )
    if ground_truth_top4.ndim != 4:
        raise ValueError(
            f"ground_truth_top4 must be 4D (N, 3, 20, 4), "
            f"got shape {ground_truth_top4.shape}"
        )

    N, num_horizons, num_layers, num_experts = calibrated_probs.shape
    _, _, _, k = ground_truth_top4.shape

    logger.info(
        "Evaluating Targeted ECE: N=%d, horizons=%d, layers=%d, experts=%d, k=%d",
        N, num_horizons, num_layers, num_experts, k,
    )
    logger.info("Thresholds: %s, bandwidth: %.4f", thresholds, bandwidth)

    result: dict = {}

    # Compute targeted ECE for each threshold × bucket × horizon
    for threshold in thresholds:
        key = f"targeted_ece_{threshold}"
        result[key] = {}

        for bucket in GRID_BUCKETS:
            layer_slice = _get_layer_bucket_slice(bucket)

            for horizon in GRID_HORIZONS:
                h_idx = _get_horizon_index(horizon)

                # Extract slice: (N, num_bucket_layers, num_experts)
                probs_slice = calibrated_probs[:, h_idx, layer_slice, :]
                gt_slice = ground_truth_top4[:, h_idx, layer_slice, :]

                # Reshape to 2D for ECE computation
                n_bucket_layers = probs_slice.shape[1]
                probs_2d = probs_slice.reshape(-1, num_experts)
                gt_2d = gt_slice.reshape(-1, k)

                # Create binary ground truth
                gt_binary = _create_binary_ground_truth(gt_2d, num_experts)

                ece_val, count = compute_targeted_ece_single(
                    probs_2d, gt_binary, threshold, bandwidth
                )

                result[key][(bucket, horizon)] = ece_val
                logger.info(
                    "  T-ECE @ %.2f [%s, %s]: %.6f (n=%d)",
                    threshold, bucket, horizon, ece_val, count,
                )

    # Compute cumulative mass stats at 0.85 threshold
    # Reshape to 2D for mass computation
    probs_2d_all = calibrated_probs.reshape(-1, num_experts)
    gt_2d_all = ground_truth_top4.reshape(-1, k)
    result["cumulative_mass_0.85_stats"] = compute_cumulative_mass_stats(
        probs_2d_all, gt_2d_all
    )

    # Compute overall ECE across all predictions
    gt_binary_all = _create_binary_ground_truth(gt_2d_all, num_experts)
    result["overall_ece"] = _compute_overall_ece(probs_2d_all, gt_binary_all)
    logger.info("Overall ECE: %.6f", result["overall_ece"])

    return result
