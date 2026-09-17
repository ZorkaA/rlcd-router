"""2×3 Temperature Scaling Grid Module.

This module defines ``TemperatureGrid(nn.Module)`` that holds a learnable
``(2, 3)`` parameter grid of temperature scalars for post-hoc temperature
scaling of speculative router logits.

Grid Layout
-----------
Dim 0 (layer bucket):
    - 0 = "early" → Deep layers 5–10 (deep_layer indices 0–5, 6 layers)
    - 1 = "late"  → Deep layers 11–24 (deep_layer indices 6–19, 14 layers)

Dim 1 (horizon):
    - 0 = T+1
    - 1 = T+2
    - 2 = T+3

Interface Contract (M3 ↔ M4)
-----------------------------
- ``scale_logits(logits: Tensor) -> Tensor``
  Input:  ``(N, 3, 20, 60)`` uncalibrated speculative logits
  Output: ``(N, 3, 20, 60)`` temperature-scaled logits

- Serialization to both ``.pt`` (state_dict) and ``.json`` (human-readable).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.config import (
    GRID_SHAPE,
    NUM_EARLY_LAYERS,
    NUM_LATE_LAYERS,
    NUM_DEEP_LAYERS,
    NUM_HORIZONS,
    NUM_EXPERTS,
    TEMP_GRID_CHECKPOINT,
    TEMP_GRID_JSON,
)

logger = logging.getLogger(__name__)


class TemperatureGrid(nn.Module):
    """Learnable 2×3 temperature scaling grid for speculative router logits.

    Parameters
    ----------
    init_temperatures : torch.Tensor, optional
        Initial temperature values of shape ``(2, 3)``.  Defaults to all-ones
        (identity scaling).
    min_temp : float
        Lower clamp for temperatures during forward.  Prevents division by
        near-zero values.  Default: ``0.01``.

    Attributes
    ----------
    temperatures : nn.Parameter
        Learnable ``(2, 3)`` parameter grid.
    num_early_layers : int
        Number of early deep layers (6).
    num_late_layers : int
        Number of late deep layers (14).
    """

    # Class-level constants for layer bucket boundaries (deep_layer index space)
    NUM_EARLY_LAYERS: int = NUM_EARLY_LAYERS  # 6  (Layers 5–10)
    NUM_LATE_LAYERS: int = NUM_LATE_LAYERS    # 14 (Layers 11–24)

    def __init__(
        self,
        init_temperatures: Optional[torch.Tensor] = None,
        min_temp: float = 0.01,
    ) -> None:
        super().__init__()

        if init_temperatures is not None:
            if init_temperatures.shape != (GRID_SHAPE[0], GRID_SHAPE[1]):
                raise ValueError(
                    f"init_temperatures must have shape {GRID_SHAPE}, "
                    f"got {init_temperatures.shape}"
                )
            self.temperatures = nn.Parameter(init_temperatures.clone().float())
        else:
            self.temperatures = nn.Parameter(torch.ones(GRID_SHAPE[0], GRID_SHAPE[1]))

        self.min_temp = min_temp
        self.num_early_layers = NUM_EARLY_LAYERS
        self.num_late_layers = NUM_LATE_LAYERS

        logger.info(
            "TemperatureGrid initialized: shape=%s, min_temp=%.4f, "
            "init_values=\n%s",
            tuple(self.temperatures.shape),
            self.min_temp,
            self.temperatures.data,
        )

    def _clamped_temperatures(self) -> torch.Tensor:
        """Return temperatures clamped to ``[min_temp, ∞)`` for numerical safety.

        Returns
        -------
        torch.Tensor
            Clamped temperatures of shape ``(2, 3)``.
        """
        return self.temperatures.clamp(min=self.min_temp)

    def scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to speculative router logits.

        Each ``(horizon, deep_layer)`` slice of the input tensor is divided by
        the scalar temperature from the corresponding grid cell.

        Parameters
        ----------
        logits : torch.Tensor
            Uncalibrated speculative logits of shape ``(N, 3, 20, 60)``.

        Returns
        -------
        torch.Tensor
            Temperature-scaled logits, same shape ``(N, 3, 20, 60)``.

        Raises
        ------
        ValueError
            If logits do not have 4 dimensions or incompatible horizon/layer
            counts.
        """
        if logits.ndim != 4:
            raise ValueError(
                f"Expected 4-D logits (N, horizons, layers, experts), "
                f"got {logits.ndim}-D tensor with shape {logits.shape}"
            )

        n, n_horizons, n_layers, n_experts = logits.shape

        if n_horizons != NUM_HORIZONS:
            raise ValueError(
                f"Horizon dim must be {NUM_HORIZONS}, got {n_horizons}"
            )
        if n_layers != NUM_DEEP_LAYERS:
            raise ValueError(
                f"Layer dim must be {NUM_DEEP_LAYERS}, got {n_layers}"
            )

        temps = self._clamped_temperatures()  # (2, 3)

        # Build a (3, 20) temperature map that broadcasts over (N, 3, 20, 60)
        # For each horizon h:
        #   layers 0..5  (early) get temps[0, h]
        #   layers 6..19 (late)  get temps[1, h]
        temp_map = torch.empty(
            n_horizons, n_layers, device=logits.device, dtype=logits.dtype
        )
        for h in range(n_horizons):
            temp_map[h, :self.num_early_layers] = temps[0, h]
            temp_map[h, self.num_early_layers:] = temps[1, h]

        # Reshape for broadcasting: (1, 3, 20, 1) to divide (N, 3, 20, 60)
        temp_map = temp_map.unsqueeze(0).unsqueeze(-1)  # (1, 3, 20, 1)

        scaled = logits / temp_map

        logger.debug(
            "scale_logits: input shape=%s, temps_early=%s, temps_late=%s",
            logits.shape,
            temps[0].tolist(),
            temps[1].tolist(),
        )

        return scaled

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Forward pass — alias for ``scale_logits``.

        Parameters
        ----------
        logits : torch.Tensor
            Shape ``(N, 3, 20, 60)``.

        Returns
        -------
        torch.Tensor
            Scaled logits, same shape.
        """
        return self.scale_logits(logits)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save_pt(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Save the temperature grid state dict to a ``.pt`` file.

        Parameters
        ----------
        path : str or Path, optional
            Destination path.  Defaults to ``TEMP_GRID_CHECKPOINT``.

        Returns
        -------
        Path
            The file path that was written.
        """
        path = Path(path) if path is not None else TEMP_GRID_CHECKPOINT
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info("Saved temperature grid (.pt) → %s", path)
        return path

    def save_json(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Save the temperature grid as a human-readable JSON file.

        The JSON contains the full 2×3 grid with row/column labels for
        inspection by downstream systems and humans.

        Parameters
        ----------
        path : str or Path, optional
            Destination path.  Defaults to ``TEMP_GRID_JSON``.

        Returns
        -------
        Path
            The file path that was written.
        """
        path = Path(path) if path is not None else TEMP_GRID_JSON
        path.parent.mkdir(parents=True, exist_ok=True)

        temps = self.temperatures.detach().cpu()
        payload: Dict[str, Any] = {
            "grid_shape": list(temps.shape),
            "buckets": ["early", "late"],
            "horizons": ["T+1", "T+2", "T+3"],
            "temperatures": {
                "early": {
                    "T+1": float(temps[0, 0]),
                    "T+2": float(temps[0, 1]),
                    "T+3": float(temps[0, 2]),
                },
                "late": {
                    "T+1": float(temps[1, 0]),
                    "T+2": float(temps[1, 1]),
                    "T+3": float(temps[1, 2]),
                },
            },
            "flat_values": temps.tolist(),
        }

        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Saved temperature grid (.json) → %s", path)
        return path

    def save(
        self,
        pt_path: Optional[Union[str, Path]] = None,
        json_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[Path, Path]:
        """Save the temperature grid in both ``.pt`` and ``.json`` formats.

        Parameters
        ----------
        pt_path : str or Path, optional
            Path for ``.pt`` checkpoint.
        json_path : str or Path, optional
            Path for ``.json`` file.

        Returns
        -------
        tuple of (Path, Path)
            ``(pt_path, json_path)`` that were written.
        """
        p = self.save_pt(pt_path)
        j = self.save_json(json_path)
        return p, j

    @classmethod
    def load_pt(
        cls,
        path: Optional[Union[str, Path]] = None,
        min_temp: float = 0.01,
    ) -> "TemperatureGrid":
        """Load a ``TemperatureGrid`` from a ``.pt`` checkpoint.

        Parameters
        ----------
        path : str or Path, optional
            Source path.  Defaults to ``TEMP_GRID_CHECKPOINT``.
        min_temp : float
            Minimum temperature clamp.

        Returns
        -------
        TemperatureGrid
            Loaded module.
        """
        path = Path(path) if path is not None else TEMP_GRID_CHECKPOINT
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        grid = cls(min_temp=min_temp)
        grid.load_state_dict(state_dict)
        logger.info("Loaded temperature grid (.pt) ← %s", path)
        return grid

    @classmethod
    def load_json(
        cls,
        path: Optional[Union[str, Path]] = None,
        min_temp: float = 0.01,
    ) -> "TemperatureGrid":
        """Load a ``TemperatureGrid`` from a ``.json`` file.

        Parameters
        ----------
        path : str or Path, optional
            Source path.  Defaults to ``TEMP_GRID_JSON``.
        min_temp : float
            Minimum temperature clamp.

        Returns
        -------
        TemperatureGrid
            Loaded module.
        """
        path = Path(path) if path is not None else TEMP_GRID_JSON
        with open(path, "r") as f:
            payload = json.load(f)

        temps_list = payload["flat_values"]
        temps_tensor = torch.tensor(temps_list, dtype=torch.float32)
        grid = cls(init_temperatures=temps_tensor, min_temp=min_temp)
        logger.info("Loaded temperature grid (.json) ← %s", path)
        return grid

    def __repr__(self) -> str:
        temps = self.temperatures.data
        return (
            f"TemperatureGrid(\n"
            f"  early(5-10):  T+1={temps[0,0]:.4f}, T+2={temps[0,1]:.4f}, T+3={temps[0,2]:.4f}\n"
            f"  late(11-24):  T+1={temps[1,0]:.4f}, T+2={temps[1,1]:.4f}, T+3={temps[1,2]:.4f}\n"
            f"  min_temp={self.min_temp}\n"
            f")"
        )
