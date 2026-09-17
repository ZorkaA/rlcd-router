"""Medusa Speculative Head for MoE Router Calibration.

Implements a ``MedusaSpeculativeHead(nn.Module)`` with 3 independent linear
projection heads, one per lookahead horizon (T+1, T+2, T+3).  Each head maps
Layer 3 hidden states (``input_dim=2048``) to 20 deep layers × 60 experts =
1200 logits, producing unscaled speculative router logits.

Interface Contract (M2 ↔ M3):
    - Forward signature: ``forward(hidden_states: Tensor) -> Tensor``
    - Input:  ``(batch, input_dim)`` or ``(batch, seq_len, input_dim)``
    - Output: ``(batch, [seq_len,] 3, 20, 60)`` unscaled speculative logits
    - Checkpoint: ``checkpoints/speculative_head.pt`` (state_dict + config)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.config import (
    HIDDEN_SIZE,
    NUM_DEEP_LAYERS,
    NUM_EXPERTS,
    NUM_HORIZONS,
    SPEC_HEAD_INPUT_DIM,
    SPEC_HEAD_OUTPUT_DIM_PER_HORIZON,
)

logger = logging.getLogger(__name__)


class MedusaSpeculativeHead(nn.Module):
    """Three independent linear projection heads for multi-horizon speculative routing.

    Each horizon head is a single ``nn.Linear(input_dim, output_dim_per_horizon)``
    mapping from Layer 3 hidden representations to flattened deep-layer router
    logits.  The outputs are reshaped to ``(batch, [seq_len,] num_horizons,
    num_deep_layers, num_experts)`` for downstream loss computation and
    temperature scaling.

    Args:
        input_dim: Dimensionality of the tap-layer hidden states (default 2048).
        num_deep_layers: Number of deep routing layers to predict (default 20).
        num_experts: Number of routed experts per layer (default 60).
        num_horizons: Number of lookahead horizons (default 3 for T+1..T+3).

    Example::

        >>> head = MedusaSpeculativeHead()
        >>> x = torch.randn(8, 2048)        # batch of 8
        >>> out = head(x)                     # (8, 3, 20, 60)
        >>> x3d = torch.randn(8, 10, 2048)  # batch with seq_len=10
        >>> out3d = head(x3d)                # (8, 10, 3, 20, 60)
    """

    def __init__(
        self,
        input_dim: int = SPEC_HEAD_INPUT_DIM,
        num_deep_layers: int = NUM_DEEP_LAYERS,
        num_experts: int = NUM_EXPERTS,
        num_horizons: int = NUM_HORIZONS,
    ) -> None:
        super().__init__()

        # ----- Input validation -----
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if num_deep_layers <= 0:
            raise ValueError(f"num_deep_layers must be positive, got {num_deep_layers}")
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        if num_horizons <= 0:
            raise ValueError(f"num_horizons must be positive, got {num_horizons}")

        self.input_dim = input_dim
        self.num_deep_layers = num_deep_layers
        self.num_experts = num_experts
        self.num_horizons = num_horizons
        self.output_dim_per_horizon = num_deep_layers * num_experts

        logger.info(
            "Initialising MedusaSpeculativeHead: input_dim=%d, "
            "num_horizons=%d, num_deep_layers=%d, num_experts=%d, "
            "output_dim_per_horizon=%d",
            input_dim,
            num_horizons,
            num_deep_layers,
            num_experts,
            self.output_dim_per_horizon,
        )

        # Create independent linear heads — one per horizon
        self.heads = nn.ModuleList([
            nn.Linear(input_dim, self.output_dim_per_horizon)
            for _ in range(num_horizons)
        ])

        logger.debug(
            "Created %d independent linear heads, total parameters: %d",
            num_horizons,
            sum(p.numel() for p in self.parameters()),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to multi-horizon, multi-layer router logits.

        Args:
            hidden_states: Input tensor of shape ``(batch, input_dim)`` or
                ``(batch, seq_len, input_dim)``.

        Returns:
            Unscaled speculative logits of shape
            ``(batch, 3, 20, 60)`` for 2-D input or
            ``(batch, seq_len, 3, 20, 60)`` for 3-D input.

        Raises:
            ValueError: If the last dimension does not match ``input_dim`` or
                the tensor has fewer than 2 dimensions.
        """
        if hidden_states.dim() < 2:
            raise ValueError(
                f"hidden_states must be at least 2-D, got shape {hidden_states.shape}"
            )
        if hidden_states.shape[-1] != self.input_dim:
            raise ValueError(
                f"Last dimension of hidden_states must be {self.input_dim}, "
                f"got {hidden_states.shape[-1]}"
            )

        # Determine whether we received a 2-D (batch, dim) or 3-D (batch, seq, dim) input
        is_2d = hidden_states.dim() == 2
        if is_2d:
            # Add a dummy seq_len=1 dimension for uniform processing
            hidden_states = hidden_states.unsqueeze(1)  # (batch, 1, input_dim)

        batch_size, seq_len, _ = hidden_states.shape

        # Compute each head independently and stack along the horizon axis
        horizon_outputs = []
        for head in self.heads:
            # head: (batch, seq_len, input_dim) -> (batch, seq_len, output_dim_per_horizon)
            h_out = head(hidden_states)
            # Reshape to (batch, seq_len, num_deep_layers, num_experts)
            h_out = h_out.view(batch_size, seq_len, self.num_deep_layers, self.num_experts)
            horizon_outputs.append(h_out)

        # Stack along horizon dimension: (batch, seq_len, num_horizons, num_deep_layers, num_experts)
        out = torch.stack(horizon_outputs, dim=2)

        if is_2d:
            # Remove the seq_len dimension: (batch, num_horizons, num_deep_layers, num_experts)
            out = out.squeeze(1)

        return out

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return a serialisable dictionary of architecture hyper-parameters."""
        return {
            "input_dim": self.input_dim,
            "num_deep_layers": self.num_deep_layers,
            "num_experts": self.num_experts,
            "num_horizons": self.num_horizons,
        }

    def save_checkpoint(self, filepath: str) -> None:
        """Save state_dict and config to a PyTorch checkpoint.

        Args:
            filepath: Path to write the ``.pt`` checkpoint file.
        """
        from pathlib import Path

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "state_dict": self.state_dict(),
            "config": self.get_config(),
        }
        torch.save(checkpoint, str(path))
        logger.info("Saved MedusaSpeculativeHead checkpoint to %s", filepath)

    @classmethod
    def from_checkpoint(
        cls,
        filepath: str,
        device: Optional[torch.device] = None,
    ) -> "MedusaSpeculativeHead":
        """Restore a MedusaSpeculativeHead from a saved checkpoint.

        Args:
            filepath: Path to the ``.pt`` checkpoint file.
            device: Optional device to map the loaded weights onto.

        Returns:
            An initialised ``MedusaSpeculativeHead`` with loaded weights.
        """
        map_location = device if device is not None else "cpu"
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=True)

        config = checkpoint["config"]
        head = cls(
            input_dim=config["input_dim"],
            num_deep_layers=config["num_deep_layers"],
            num_experts=config["num_experts"],
            num_horizons=config["num_horizons"],
        )
        head.load_state_dict(checkpoint["state_dict"])
        if device is not None:
            head = head.to(device)
        logger.info("Loaded MedusaSpeculativeHead from %s", filepath)
        return head
