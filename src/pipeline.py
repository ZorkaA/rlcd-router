"""End-to-End Calibration Pipeline CLI for MoE Router.

Orchestrates all phases of the calibration pipeline:
1. Data generation/loading (M1)
2. Speculative head training (M2)
3. Temperature scaling calibration (M3)
4. Targeted ECE evaluation (M4)

Supports --synthetic mode for testing with small models.
Outputs evaluation report to JSON.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from src.config import (
    PipelineConfig,
    ModelConfig,
    ExtractionConfig,
    TrainingConfig,
    CalibrationConfig,
    EvaluationConfig,
    EVALUATION_REPORT,
    RESULTS_DIR,
    DATA_DIR,
    CHECKPOINTS_DIR,
    NUM_DEEP_LAYERS,
    NUM_HORIZONS,
    NUM_EXPERTS,
    NUM_EXPERTS_PER_TOK,
    SYNTHETIC_HIDDEN_SIZE,
    SYNTHETIC_NUM_EXPERTS,
    SYNTHETIC_NUM_EXPERTS_PER_TOK,
    GRID_BUCKETS,
    GRID_HORIZONS,
    resolve_device,
)
from src.evaluation.targeted_ece import evaluate_targeted_ece
from src.evaluation.memory_profiler import MemoryProfiler, MemoryReport

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    """Configure root logger with timestamp, level, and module info.

    Args:
        verbose: If True, set level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _generate_synthetic_data(
    config: PipelineConfig,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Generate synthetic train and calibration data for testing.

    Creates random hidden states, router logits, top-4 indices, and valid masks
    using dimensions from the pipeline config.

    Args:
        config: Pipeline configuration.
        device: Target device for tensors.

    Returns:
        Tuple of (train_data, calib_data) dictionaries, each containing:
        - hidden_states: (N, hidden_size)
        - target_router_logits: (N, 3, num_deep_layers, num_experts)
        - target_top4_indices: (N, 3, num_deep_layers, 4)
        - valid_mask: (N, 3)
    """
    hidden_size = config.model.hidden_size
    num_experts = config.model.num_experts
    num_deep_layers = len(config.model.deep_layers)
    num_horizons = len(config.model.lookahead_horizons)
    k = config.model.num_experts_per_tok

    logger.info(
        "Generating synthetic data: hidden=%d, experts=%d, layers=%d, horizons=%d",
        hidden_size, num_experts, num_deep_layers, num_horizons,
    )

    def make_split(n_samples: int) -> Dict[str, torch.Tensor]:
        torch.manual_seed(42 + n_samples)
        hidden_states = torch.randn(n_samples, hidden_size, dtype=torch.float32, device=device)
        target_router_logits = torch.randn(
            n_samples, num_horizons, num_deep_layers, num_experts,
            dtype=torch.float32, device=device,
        )
        target_top4_indices = torch.topk(target_router_logits, k=k, dim=-1).indices
        valid_mask = torch.ones(n_samples, num_horizons, dtype=torch.bool, device=device)
        # Simulate boundary trimming on last 2 samples
        if n_samples >= 2:
            valid_mask[-1, :] = torch.tensor([True, False, False])
            valid_mask[-2, :] = torch.tensor([True, True, False])

        return {
            "hidden_states": hidden_states,
            "target_router_logits": target_router_logits,
            "target_top4_indices": target_top4_indices,
            "valid_mask": valid_mask,
        }

    train_data = make_split(128)
    calib_data = make_split(32)

    logger.info("Synthetic data generated: train=%d, calib=%d samples",
                train_data["hidden_states"].shape[0],
                calib_data["hidden_states"].shape[0])
    return train_data, calib_data


def _load_data(
    config: PipelineConfig,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Load train and calibration data from safetensors files.

    Args:
        config: Pipeline configuration with file paths.

    Returns:
        Tuple of (train_data, calib_data) dictionaries.

    Raises:
        FileNotFoundError: If data files do not exist.
    """
    from safetensors.torch import load_file

    train_path = config.extraction.output_dir / "train_data.safetensors"
    calib_path = config.extraction.output_dir / "calib_data.safetensors"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            "Run data generation first or use --synthetic mode."
        )
    if not calib_path.exists():
        raise FileNotFoundError(
            f"Calibration data not found at {calib_path}. "
            "Run data generation first or use --synthetic mode."
        )

    logger.info("Loading training data from %s", train_path)
    train_data = load_file(str(train_path))

    logger.info("Loading calibration data from %s", calib_path)
    calib_data = load_file(str(calib_path))

    # Convert valid_mask back to bool if stored as uint8
    for data in [train_data, calib_data]:
        if "valid_mask" in data and data["valid_mask"].dtype == torch.uint8:
            data["valid_mask"] = data["valid_mask"].bool()

    return train_data, calib_data


def _run_synthetic_training(
    train_data: Dict[str, torch.Tensor],
    config: PipelineConfig,
    device: torch.device,
) -> torch.nn.Module:
    """Simulate speculative head training in synthetic mode.

    Creates a simple linear model and runs a few training steps.
    In production mode, this would use the full MedusaSpeculativeHead
    from src.models.medusa_head.

    Args:
        train_data: Training data dictionary.
        config: Pipeline configuration.
        device: Target device.

    Returns:
        Trained model (or simple linear stand-in for synthetic mode).
    """
    hidden_size = config.model.hidden_size
    num_experts = config.model.num_experts
    num_deep_layers = len(config.model.deep_layers)
    num_horizons = len(config.model.lookahead_horizons)
    output_dim = num_horizons * num_deep_layers * num_experts

    logger.info(
        "Running synthetic training: %d -> %d, %d epochs",
        hidden_size, output_dim, min(config.training.epochs, 2),
    )

    model = torch.nn.Linear(hidden_size, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

    hs = train_data["hidden_states"].to(device)
    targets = train_data["target_router_logits"].to(device)
    target_flat = targets.reshape(targets.shape[0], -1)

    num_epochs = min(config.training.epochs, 2)  # Cap for speed in synthetic mode
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        pred = model(hs)
        loss = F.mse_loss(pred, target_flat)
        loss.backward()
        optimizer.step()
        logger.info("  Epoch %d/%d — loss: %.6f", epoch + 1, num_epochs, loss.item())

    model.eval()
    return model


def _run_synthetic_calibration(
    calib_data: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    config: PipelineConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simulate temperature scaling calibration in synthetic mode.

    Generates calibrated probabilities by applying softmax to model
    predictions divided by synthetic temperature values.

    Args:
        calib_data: Calibration data dictionary.
        model: Trained (or synthetic) model.
        config: Pipeline configuration.
        device: Target device.

    Returns:
        Tuple of (calibrated_probs, ground_truth_top4) for evaluation.
        calibrated_probs: (N, 3, num_deep_layers, num_experts)
        ground_truth_top4: (N, 3, num_deep_layers, 4)
    """
    num_experts = config.model.num_experts
    num_deep_layers = len(config.model.deep_layers)
    num_horizons = len(config.model.lookahead_horizons)

    logger.info("Running synthetic calibration")

    with torch.inference_mode():
        hs = calib_data["hidden_states"].to(device)
        pred_flat = model(hs)  # (N, num_horizons * num_deep_layers * num_experts)

    # Reshape to (N, 3, num_deep_layers, num_experts)
    N = pred_flat.shape[0]
    pred_logits = pred_flat.reshape(N, num_horizons, num_deep_layers, num_experts)

    # Apply simple temperature scaling (T=1.0 for synthetic)
    temperature = torch.ones(2, 3, dtype=torch.float32, device=device)

    # Scale logits
    calibrated_logits = pred_logits.clone()
    for h in range(num_horizons):
        # Early layers: indices 0..5
        early_end = min(6, num_deep_layers)
        calibrated_logits[:, h, :early_end, :] /= temperature[0, h]
        # Late layers: indices 6..
        if num_deep_layers > early_end:
            calibrated_logits[:, h, early_end:, :] /= temperature[1, h]

    # Convert to probabilities via softmax
    calibrated_probs = F.softmax(calibrated_logits, dim=-1)

    ground_truth_top4 = calib_data["target_top4_indices"].to(device)

    logger.info(
        "Calibrated probs shape: %s, ground truth shape: %s",
        calibrated_probs.shape, ground_truth_top4.shape,
    )
    return calibrated_probs, ground_truth_top4


def _serialize_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Convert report dict to JSON-serializable format.

    Converts tuple keys like ("early", "T+1") to string keys like
    "early|T+1" for JSON compatibility.

    Args:
        report: Raw evaluation report dictionary.

    Returns:
        JSON-serializable dictionary.
    """
    serialized: Dict[str, Any] = {}
    for key, value in report.items():
        if isinstance(value, dict):
            inner: Dict[str, Any] = {}
            for k, v in value.items():
                if isinstance(k, tuple):
                    str_key = "|".join(str(x) for x in k)
                    inner[str_key] = v
                else:
                    inner[str(k)] = v
            serialized[key] = inner
        else:
            serialized[key] = value
    return serialized


def run_pipeline(
    synthetic: bool = False,
    verbose: bool = False,
    output_path: Optional[Path] = None,
    device: Optional[str] = None,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    """Run the complete MoE Router Calibration Pipeline end-to-end.

    Orchestrates four phases:
    1. **Data Phase**: Load or generate (synthetic) training/calibration data.
    2. **Training Phase**: Train the speculative head model.
    3. **Calibration Phase**: Apply temperature scaling to produce calibrated probs.
    4. **Evaluation Phase**: Compute targeted ECE metrics and memory profile.

    Args:
        synthetic: If True, use synthetic small-scale data and models
            for testing. No real model download required.
        verbose: Enable DEBUG-level logging.
        output_path: Path to write JSON evaluation report.
            Defaults to results/evaluation_report.json.
        device: Target device string ('cpu', 'mps', 'cuda', 'auto').
        config: Optional pre-built PipelineConfig. If None, uses defaults
            (adjusted for synthetic mode if applicable).

    Returns:
        Complete evaluation report dictionary with all metrics.
    """
    _setup_logging(verbose)
    logger.info("=" * 70)
    logger.info("MoE Router Calibration Pipeline — Starting")
    logger.info("=" * 70)
    start_time = time.time()

    # Resolve device
    resolved_device = resolve_device(device)
    logger.info("Device: %s", resolved_device)

    # Build config
    if config is None:
        config = PipelineConfig()
        if synthetic:
            config.model.hidden_size = SYNTHETIC_HIDDEN_SIZE
            config.model.num_experts = SYNTHETIC_NUM_EXPERTS
            config.model.num_experts_per_tok = SYNTHETIC_NUM_EXPERTS_PER_TOK
            config.model.deep_layers = tuple(range(3, 7))  # 4 layers
            config.model.deep_layer_indices = tuple(range(2, 6))
            config.training.input_dim = SYNTHETIC_HIDDEN_SIZE
            config.training.output_dim_per_horizon = (
                len(config.model.deep_layers) * config.model.num_experts
            )
            config.training.epochs = 2

    if output_path is None:
        output_path = config.evaluation.report_path

    # Ensure output directories exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "pipeline_config": {
            "synthetic": synthetic,
            "device": str(resolved_device),
        },
        "phases": {},
        "timings": {},
    }

    # ---- Phase 1: Data Loading/Generation ----
    phase_start = time.time()
    logger.info("-" * 50)
    logger.info("Phase 1: Data Loading/Generation")
    logger.info("-" * 50)

    if synthetic:
        train_data, calib_data = _generate_synthetic_data(config, resolved_device)
    else:
        train_data, calib_data = _load_data(config)

    report["timings"]["data_phase_seconds"] = time.time() - phase_start
    report["phases"]["data"] = {
        "train_samples": int(train_data["hidden_states"].shape[0]),
        "calib_samples": int(calib_data["hidden_states"].shape[0]),
        "mode": "synthetic" if synthetic else "loaded",
    }
    logger.info("Phase 1 complete in %.2fs", report["timings"]["data_phase_seconds"])

    # ---- Phase 2: Speculative Head Training ----
    phase_start = time.time()
    logger.info("-" * 50)
    logger.info("Phase 2: Speculative Head Training")
    logger.info("-" * 50)

    if synthetic:
        model = _run_synthetic_training(train_data, config, resolved_device)
    else:
        # In production, import and use MedusaSpeculativeHead + Trainer
        logger.info("Loading pre-trained speculative head from %s", config.training.checkpoint_path)
        # Placeholder for M2 integration
        raise NotImplementedError(
            "Production speculative head training requires M2 modules. "
            "Use --synthetic for testing."
        )

    report["timings"]["training_phase_seconds"] = time.time() - phase_start
    report["phases"]["training"] = {
        "mode": "synthetic" if synthetic else "full",
        "epochs": min(config.training.epochs, 2) if synthetic else config.training.epochs,
    }
    logger.info("Phase 2 complete in %.2fs", report["timings"]["training_phase_seconds"])

    # ---- Phase 3: Temperature Scaling Calibration ----
    phase_start = time.time()
    logger.info("-" * 50)
    logger.info("Phase 3: Temperature Scaling Calibration")
    logger.info("-" * 50)

    if synthetic:
        calibrated_probs, ground_truth_top4 = _run_synthetic_calibration(
            calib_data, model, config, resolved_device
        )
    else:
        # In production, use TemperatureGrid from M3
        raise NotImplementedError(
            "Production temperature scaling requires M3 modules. "
            "Use --synthetic for testing."
        )

    report["timings"]["calibration_phase_seconds"] = time.time() - phase_start
    report["phases"]["calibration"] = {
        "mode": "synthetic" if synthetic else "full",
        "calibrated_probs_shape": list(calibrated_probs.shape),
    }
    logger.info("Phase 3 complete in %.2fs", report["timings"]["calibration_phase_seconds"])

    # ---- Phase 4: Targeted ECE Evaluation ----
    phase_start = time.time()
    logger.info("-" * 50)
    logger.info("Phase 4: Targeted ECE Evaluation")
    logger.info("-" * 50)

    # Run evaluation with memory profiling
    profiler = MemoryProfiler(
        label="evaluation_phase",
        max_tensor_delta=100,  # Allow some tensor growth for evaluation
    )

    with profiler:
        eval_results = evaluate_targeted_ece(
            calibrated_probs=calibrated_probs.cpu(),
            ground_truth_top4=ground_truth_top4.cpu(),
            thresholds=list(config.evaluation.thresholds),
            bandwidth=config.evaluation.bandwidth,
        )

    memory_report = profiler.report

    report["timings"]["evaluation_phase_seconds"] = time.time() - phase_start
    report["phases"]["evaluation"] = _serialize_report(eval_results)
    report["memory_profile"] = memory_report.to_dict()
    report["timings"]["total_seconds"] = time.time() - start_time

    # ---- Write Report ----
    serialized_report = _serialize_report(report)
    with open(output_path, "w") as f:
        json.dump(serialized_report, f, indent=2, default=str)

    logger.info("=" * 70)
    logger.info("Pipeline complete in %.2fs", report["timings"]["total_seconds"])
    logger.info("Report written to %s", output_path)
    logger.info("Overall ECE: %.6f", eval_results.get("overall_ece", float("nan")))
    logger.info("Memory profile passed: %s", memory_report.passed_assertions)
    logger.info("=" * 70)

    return report


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the pipeline.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="MoE Router Calibration Pipeline — End-to-End Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--synthetic",
        action="store_true",
        default=False,
        help="Use synthetic small-scale data and models for testing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "mps", "cuda", "auto"],
        help="Target compute device.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write JSON evaluation report.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )

    return parser


def main(argv: Optional[list] = None) -> Dict[str, Any]:
    """Main entry point for the CLI pipeline.

    Args:
        argv: Command-line arguments. If None, uses sys.argv.

    Returns:
        Evaluation report dictionary.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    output_path = Path(args.output) if args.output else None

    return run_pipeline(
        synthetic=args.synthetic,
        verbose=args.verbose,
        output_path=output_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()
