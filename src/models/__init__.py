"""Models module for MoE Speculative Router Calibration.

Contains the Medusa Speculative Head — 3 independent linear projection heads
for multi-horizon (T+1, T+2, T+3) router distribution prediction from
Layer 3 hidden states.
"""

from src.models.medusa_head import MedusaSpeculativeHead

__all__ = [
    "MedusaSpeculativeHead",
]
