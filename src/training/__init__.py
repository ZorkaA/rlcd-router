"""Training module for MoE Speculative Router Calibration.

Contains the MMCE loss functions (soft CE + RKHS MMCE penalty) and the
end-to-end training loop for the Medusa Speculative Head.
"""

from src.training.mmce_loss import (
    soft_cross_entropy_loss,
    rkhs_mmce_penalty,
    combined_loss,
    CombinedCalibrationLoss,
)
from src.training.trainer import Trainer

__all__ = [
    "soft_cross_entropy_loss",
    "rkhs_mmce_penalty",
    "combined_loss",
    "CombinedCalibrationLoss",
    "Trainer",
]
