"""Milestone 3: Grid-Based Temperature Scaling with LBFGS NLL Optimization.

This package provides:
- ``TemperatureGrid``: A 2×3 temperature scaling grid (nn.Module) indexed by
  {early, late} layer buckets × {T+1, T+2, T+3} lookahead horizons.
- ``LBFGSOptimizer``: Fits the temperature grid on held-out calibration data
  by minimizing NLL with L2 regularization toward T=1.0 using
  ``torch.optim.LBFGS`` with strong Wolfe line search.
"""

from src.calibration.grid import TemperatureGrid
from src.calibration.lbfgs_optimizer import LBFGSOptimizer

__all__ = ["TemperatureGrid", "LBFGSOptimizer"]
