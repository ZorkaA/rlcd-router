"""LBFGS NLL Optimizer for Temperature Grid Calibration.

Fits the ``TemperatureGrid`` parameters on held-out calibration data by
strictly minimizing **Negative Log-Likelihood (NLL)** with L2 regularization
toward T=1.0.

Optimization Details
--------------------
- Optimizer: ``torch.optim.LBFGS`` with **strong Wolfe** line search.
- Loss: ``NLL(softmax(logits / T), target_distribution) + α/2 · ‖T − 1‖²``
- The NLL is computed as soft cross-entropy against the native router
  distribution (not hard labels), because the target is a full distribution
  over 60 experts.
- CRITICAL: ECE is **NOT** used for optimization — it is non-differentiable.

Data Contract
-------------
Calibration data is loaded from ``calib_data.safetensors`` (M1 output)
containing:
  - ``hidden_states``: ``(N, 2048)``
  - ``target_router_logits``: ``(N, 3, 20, 60)``
  - ``valid_mask``: ``(N, 3)``

If a trained speculative head checkpoint exists, it is loaded and used to
generate logits from hidden states. Otherwise the optimizer accepts logits
directly.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.calibration.grid import TemperatureGrid
from src.config import (
    CALIB_DATA_FILE,
    CalibrationConfig,
    DEFAULT_L2_REG_ALPHA,
    DEFAULT_LBFGS_HISTORY_SIZE,
    DEFAULT_LBFGS_LR,
    DEFAULT_LBFGS_MAX_ITER,
    DEFAULT_LBFGS_TOLERANCE_CHANGE,
    DEFAULT_LBFGS_TOLERANCE_GRAD,
    GRID_SHAPE,
    NUM_DEEP_LAYERS,
    NUM_EXPERTS,
    NUM_HORIZONS,
    SPEC_HEAD_CHECKPOINT,
)

logger = logging.getLogger(__name__)


def _compute_nll_loss(
    scaled_logits: torch.Tensor,
    target_distributions: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute soft NLL loss between scaled logits and target distributions.

    This is the soft cross-entropy:
        ``-Σ_e target_e · log_softmax(scaled_logit_e)``
    averaged over all valid samples, horizons, and layers.

    Parameters
    ----------
    scaled_logits : torch.Tensor
        Temperature-scaled logits, shape ``(N, 3, 20, 60)``.
    target_distributions : torch.Tensor
        Soft target distributions (from softmax of native router logits),
        shape ``(N, 3, 20, 60)``.
    valid_mask : torch.Tensor, optional
        Boolean mask of shape ``(N, 3)`` indicating which horizon samples
        are valid.  If None, all samples are considered valid.

    Returns
    -------
    torch.Tensor
        Scalar soft NLL loss.
    """
    # log_softmax over the expert dimension (dim=-1)
    log_probs = F.log_softmax(scaled_logits, dim=-1)  # (N, 3, 20, 60)

    # Per-sample, per-horizon, per-layer NLL: -Σ_e target_e · log(p_e)
    nll_per_sample = -torch.sum(target_distributions * log_probs, dim=-1)  # (N, 3, 20)

    if valid_mask is not None:
        # Expand valid_mask (N, 3) → (N, 3, 1) for broadcasting over layers
        mask = valid_mask.unsqueeze(-1).float()  # (N, 3, 1)
        nll_per_sample = nll_per_sample * mask
        total_valid = mask.sum() * nll_per_sample.shape[-1]  # count valid entries
        if total_valid > 0:
            return nll_per_sample.sum() / total_valid
        else:
            logger.warning("No valid samples in batch for NLL computation")
            return torch.tensor(0.0, device=scaled_logits.device, requires_grad=True)
    else:
        return nll_per_sample.mean()


def _compute_l2_regularization(
    temperatures: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Compute L2 regularization penalty toward T=1.0.

    Parameters
    ----------
    temperatures : torch.Tensor
        Temperature parameters, shape ``(2, 3)``.
    alpha : float
        Regularization strength.

    Returns
    -------
    torch.Tensor
        Scalar: ``0.5 * alpha * Σ(T - 1.0)²``
    """
    return 0.5 * alpha * torch.sum((temperatures - 1.0) ** 2)


class LBFGSOptimizer:
    """Fits a ``TemperatureGrid`` via LBFGS NLL minimization.

    Parameters
    ----------
    temperature_grid : TemperatureGrid
        The temperature grid module to optimize.
    config : CalibrationConfig, optional
        Calibration configuration.  If None, defaults are used.
    lr : float, optional
        LBFGS learning rate.  Overrides config if provided.
    max_iter : int, optional
        Max LBFGS iterations per ``step()`` call.  Overrides config.
    history_size : int, optional
        LBFGS history size.  Overrides config.
    l2_reg_alpha : float, optional
        L2 regularization strength toward T=1.0.  Overrides config.
    device : torch.device or str, optional
        Device to run optimization on.  Defaults to CPU.
    """

    def __init__(
        self,
        temperature_grid: TemperatureGrid,
        config: Optional[CalibrationConfig] = None,
        lr: Optional[float] = None,
        max_iter: Optional[int] = None,
        history_size: Optional[int] = None,
        l2_reg_alpha: Optional[float] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.grid = temperature_grid
        self.config = config or CalibrationConfig()

        self.lr = lr if lr is not None else self.config.lr
        self.max_iter = max_iter if max_iter is not None else self.config.max_iter
        self.history_size = (
            history_size if history_size is not None else self.config.history_size
        )
        self.l2_reg_alpha = (
            l2_reg_alpha if l2_reg_alpha is not None else self.config.l2_reg_alpha
        )
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        # Move grid to device
        self.grid.to(self.device)

        # Build LBFGS optimizer with strong Wolfe line search
        self.optimizer = torch.optim.LBFGS(
            self.grid.parameters(),
            lr=self.lr,
            max_iter=self.max_iter,
            history_size=self.history_size,
            tolerance_grad=DEFAULT_LBFGS_TOLERANCE_GRAD,
            tolerance_change=DEFAULT_LBFGS_TOLERANCE_CHANGE,
            line_search_fn="strong_wolfe",
        )

        self._loss_history: List[float] = []
        self._nll_history: List[float] = []
        self._reg_history: List[float] = []

        logger.info(
            "LBFGSOptimizer initialized: lr=%.4f, max_iter=%d, "
            "history_size=%d, l2_reg_alpha=%.6f, device=%s",
            self.lr,
            self.max_iter,
            self.history_size,
            self.l2_reg_alpha,
            self.device,
        )

    @property
    def loss_history(self) -> List[float]:
        """Return the history of total loss values during optimization."""
        return self._loss_history

    @property
    def nll_history(self) -> List[float]:
        """Return the history of NLL loss values during optimization."""
        return self._nll_history

    @property
    def reg_history(self) -> List[float]:
        """Return the history of regularization penalty values during optimization."""
        return self._reg_history

    def _prepare_target_distributions(
        self,
        target_router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Convert raw target router logits to soft probability distributions.

        Parameters
        ----------
        target_router_logits : torch.Tensor
            Raw router logits, shape ``(N, 3, 20, 60)``.

        Returns
        -------
        torch.Tensor
            Soft target distributions via softmax over expert dim,
            shape ``(N, 3, 20, 60)``.
        """
        return F.softmax(target_router_logits, dim=-1)

    def fit(
        self,
        speculative_logits: torch.Tensor,
        target_router_logits: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        num_steps: int = 1,
    ) -> Dict[str, Any]:
        """Run LBFGS optimization to fit the temperature grid.

        Parameters
        ----------
        speculative_logits : torch.Tensor
            Uncalibrated speculative logits of shape ``(N, 3, 20, 60)``.
            These come from the speculative head (M2) run on calibration data.
        target_router_logits : torch.Tensor
            Native router logits (ground truth) of shape ``(N, 3, 20, 60)``
            from the calibration split (M1).
        valid_mask : torch.Tensor, optional
            Boolean mask of shape ``(N, 3)`` indicating valid horizons.
        num_steps : int
            Number of LBFGS ``step()`` calls (outer iterations).
            Each step runs up to ``max_iter`` internal iterations.

        Returns
        -------
        dict
            Optimization summary with keys: ``final_loss``, ``final_nll``,
            ``final_reg``, ``temperatures``, ``num_steps``, ``elapsed_seconds``.
        """
        logger.info(
            "Starting LBFGS optimization: %d outer step(s), "
            "speculative_logits shape=%s, targets shape=%s",
            num_steps,
            speculative_logits.shape,
            target_router_logits.shape,
        )

        # Move data to device
        spec_logits = speculative_logits.to(self.device).detach()
        target_logits = target_router_logits.to(self.device).detach()

        # Prepare soft target distributions
        target_dist = self._prepare_target_distributions(target_logits)

        if valid_mask is not None:
            vmask = valid_mask.to(self.device)
        else:
            vmask = None

        start_time = time.time()

        for step_i in range(num_steps):
            step_loss_val = [0.0]
            step_nll_val = [0.0]
            step_reg_val = [0.0]
            eval_count = [0]

            def closure():
                self.optimizer.zero_grad()

                # Scale logits with current temperatures
                scaled = self.grid.scale_logits(spec_logits)

                # Compute NLL loss
                nll = _compute_nll_loss(scaled, target_dist, vmask)

                # Compute L2 regularization
                reg = _compute_l2_regularization(
                    self.grid.temperatures, self.l2_reg_alpha
                )

                # Total loss
                loss = nll + reg
                loss.backward()

                step_loss_val[0] = loss.item()
                step_nll_val[0] = nll.item()
                step_reg_val[0] = reg.item()
                eval_count[0] += 1

                return loss

            self.optimizer.step(closure)

            self._loss_history.append(step_loss_val[0])
            self._nll_history.append(step_nll_val[0])
            self._reg_history.append(step_reg_val[0])

            logger.info(
                "Step %d/%d: total_loss=%.6f, nll=%.6f, l2_reg=%.6f, "
                "closure_evals=%d, temps=\n%s",
                step_i + 1,
                num_steps,
                step_loss_val[0],
                step_nll_val[0],
                step_reg_val[0],
                eval_count[0],
                self.grid.temperatures.data,
            )

        elapsed = time.time() - start_time

        final_temps = self.grid.temperatures.detach().cpu().clone()
        result = {
            "final_loss": self._loss_history[-1] if self._loss_history else 0.0,
            "final_nll": self._nll_history[-1] if self._nll_history else 0.0,
            "final_reg": self._reg_history[-1] if self._reg_history else 0.0,
            "temperatures": final_temps,
            "num_steps": num_steps,
            "elapsed_seconds": elapsed,
        }

        logger.info(
            "LBFGS optimization complete in %.2fs. Final temperatures:\n%s",
            elapsed,
            final_temps,
        )

        return result

    def fit_from_safetensors(
        self,
        calib_data_path: Optional[Union[str, Path]] = None,
        speculative_head: Optional[nn.Module] = None,
        speculative_logits: Optional[torch.Tensor] = None,
        num_steps: int = 1,
    ) -> Dict[str, Any]:
        """Fit the temperature grid using calibration data from safetensors.

        This method loads the calibration split produced by M1, optionally
        runs the speculative head (M2) to generate logits, and then calls
        ``fit()`` to optimize the temperature grid.

        Parameters
        ----------
        calib_data_path : str or Path, optional
            Path to calibration safetensors file.  Defaults to
            ``CALIB_DATA_FILE``.
        speculative_head : nn.Module, optional
            Trained speculative head.  If provided, it is used to generate
            logits from hidden states.  Must accept ``(N, 2048)`` input and
            produce ``(N, 3, 20, 60)`` output.
        speculative_logits : torch.Tensor, optional
            Pre-computed speculative logits.  If provided, ``speculative_head``
            is ignored and these logits are used directly.
        num_steps : int
            Number of LBFGS outer steps.

        Returns
        -------
        dict
            Optimization summary (same as ``fit()``).

        Raises
        ------
        FileNotFoundError
            If calibration data file does not exist.
        ValueError
            If neither ``speculative_head`` nor ``speculative_logits`` is
            provided.
        """
        import safetensors.torch

        path = Path(calib_data_path) if calib_data_path is not None else CALIB_DATA_FILE

        if not path.exists():
            raise FileNotFoundError(
                f"Calibration data file not found: {path}. "
                f"Run Milestone 1 data generation first."
            )

        logger.info("Loading calibration data from %s", path)
        data = safetensors.torch.load_file(str(path))

        hidden_states = data["hidden_states"]
        target_router_logits = data["target_router_logits"]

        # Handle valid_mask: safetensors may store as uint8
        valid_mask = data.get("valid_mask", None)
        if valid_mask is not None and valid_mask.dtype == torch.uint8:
            valid_mask = valid_mask.bool()

        logger.info(
            "Calibration data loaded: hidden_states=%s, "
            "target_router_logits=%s, valid_mask=%s",
            hidden_states.shape,
            target_router_logits.shape,
            valid_mask.shape if valid_mask is not None else None,
        )

        # Generate or use speculative logits
        if speculative_logits is not None:
            spec_logits = speculative_logits
            logger.info(
                "Using pre-computed speculative logits: shape=%s",
                spec_logits.shape,
            )
        elif speculative_head is not None:
            logger.info("Running speculative head in eval mode to generate logits")
            speculative_head.eval()
            speculative_head.to(self.device)
            with torch.inference_mode():
                hs = hidden_states.to(self.device)
                spec_logits = speculative_head(hs)
                # Ensure shape is (N, 3, 20, 60) — handle potential (N, seq, 3, 20, 60)
                if spec_logits.ndim == 5 and spec_logits.shape[1] == 1:
                    spec_logits = spec_logits.squeeze(1)
            spec_logits = spec_logits.clone().detach().cpu()
            logger.info("Generated speculative logits: shape=%s", spec_logits.shape)
        else:
            raise ValueError(
                "Either 'speculative_logits' or 'speculative_head' must be "
                "provided to generate logits for calibration."
            )

        return self.fit(
            speculative_logits=spec_logits,
            target_router_logits=target_router_logits,
            valid_mask=valid_mask,
            num_steps=num_steps,
        )

    def save_results(
        self,
        pt_path: Optional[Union[str, Path]] = None,
        json_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[Path, Path]:
        """Save the optimized temperature grid in both formats.

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
        return self.grid.save(pt_path=pt_path, json_path=json_path)
