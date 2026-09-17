"""Global Configuration and Constants for Asynchronous MoE Router Calibration Pipeline.

This module defines all architecture specifications, hyperparameter presets,
device resolution helpers, directory paths, and structured configuration dataclasses
used across Milestones 1 to 4.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch

# =====================================================================
# 1. Filesystem & Directory Paths
# =====================================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
CHECKPOINTS_DIR: Path = PROJECT_ROOT / "checkpoints"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

# Standard Data and Artifact File Paths
TRAIN_DATA_FILE: Path = DATA_DIR / "train_data.safetensors"
CALIB_DATA_FILE: Path = DATA_DIR / "calib_data.safetensors"
SPEC_HEAD_CHECKPOINT: Path = CHECKPOINTS_DIR / "speculative_head.pt"
TEMP_GRID_CHECKPOINT: Path = CHECKPOINTS_DIR / "temperature_grid.pt"
TEMP_GRID_JSON: Path = CHECKPOINTS_DIR / "temperature_grid.json"
EVALUATION_REPORT: Path = RESULTS_DIR / "evaluation_report.json"

# =====================================================================
# 2. Base MoE Model Architecture (Qwen/Qwen1.5-MoE-A2.7B)
# =====================================================================
MODEL_NAME: str = "Qwen/Qwen1.5-MoE-A2.7B"
HIDDEN_SIZE: int = 2048
NUM_HIDDEN_LAYERS: int = 24
NUM_EXPERTS: int = 60
NUM_EXPERTS_PER_TOK: int = 4
INTERMEDIATE_SIZE: int = 5632
MOE_INTERMEDIATE_SIZE: int = 1408
SHARED_EXPERT_INTERMEDIATE_SIZE: int = 5632
VOCAB_SIZE: int = 151936

# =====================================================================
# 3. Layer Tapping, Horizons & Deep Layer Routing
# =====================================================================
# Tap layer: Layer 3 output in 1-based indexing (outputs.hidden_states[3])
TAP_LAYER_INDEX: int = 3

# Deep layers: Layers 5 to 24 inclusive (1-indexed: 5..24, 20 layers total)
DEEP_LAYERS: Tuple[int, ...] = tuple(range(5, 25))
# 0-based indices for router_logits list extraction (outputs.router_logits[4..23])
DEEP_LAYER_INDICES: Tuple[int, ...] = tuple(range(4, 24))
NUM_DEEP_LAYERS: int = len(DEEP_LAYERS)  # 20

# Early deep bucket: Layers 5 to 10 (1-indexed) -> router_logits indices 4..9
EARLY_LAYERS: Tuple[int, ...] = tuple(range(5, 11))
EARLY_LAYER_INDICES: Tuple[int, ...] = tuple(range(4, 10))
NUM_EARLY_LAYERS: int = len(EARLY_LAYERS)  # 6

# Late deep bucket: Layers 11 to 24 (1-indexed) -> router_logits indices 10..23
LATE_LAYERS: Tuple[int, ...] = tuple(range(11, 25))
LATE_LAYER_INDICES: Tuple[int, ...] = tuple(range(10, 24))
NUM_LATE_LAYERS: int = len(LATE_LAYERS)  # 14

# Lookahead horizons: T+1, T+2, T+3
LOOKAHEAD_HORIZONS: Tuple[int, ...] = (1, 2, 3)
NUM_HORIZONS: int = len(LOOKAHEAD_HORIZONS)  # 3
MAX_LOOKAHEAD: int = max(LOOKAHEAD_HORIZONS)  # 3

# =====================================================================
# 4. Corpus Streaming & Train/Calibration Partitioning (Milestone 1)
# =====================================================================
SEQUENCE_LENGTH: int = 1024
BATCH_SIZE: int = 1
TARGET_TOKENS: int = 100_000

# 98 sequences * 1024 tokens = 100,352 tokens
TOTAL_SEQUENCES: int = 98
TRAIN_SEQUENCES: int = 80  # 81,920 tokens (~81.6%)
CALIB_SEQUENCES: int = 18  # 18,432 tokens (~18.4%)
TRAIN_SPLIT_RATIO: float = 0.80
CALIB_SPLIT_RATIO: float = 0.20

# Boundary trimming: trailing tokens (L-3..L-1) in a sequence cannot look ahead
# across sequence boundaries and must be masked out.
BOUNDARY_TRIM_TOKENS: int = MAX_LOOKAHEAD

# =====================================================================
# 5. Speculative Head Architecture & MMCE Training (Milestone 2)
# =====================================================================
SPEC_HEAD_INPUT_DIM: int = HIDDEN_SIZE  # 2048
# Output dim per horizon = 20 deep layers * 60 experts = 1200 logits
SPEC_HEAD_OUTPUT_DIM_PER_HORIZON: int = NUM_DEEP_LAYERS * NUM_EXPERTS  # 1200
SPEC_HEAD_TOTAL_OUTPUT_DIM: int = NUM_HORIZONS * SPEC_HEAD_OUTPUT_DIM_PER_HORIZON  # 3600

DEFAULT_LEARNING_RATE: float = 1e-3
DEFAULT_WEIGHT_DECAY: float = 1e-4
DEFAULT_BATCH_SIZE_TRAIN: int = 32
DEFAULT_TRAIN_EPOCHS: int = 5
DEFAULT_MMCE_LAMBDA: float = 1.0
DEFAULT_MMCE_SIGMA: float = 0.2

# =====================================================================
# 6. Grid-Based Temperature Scaling (Milestone 3)
# =====================================================================
GRID_BUCKETS: Tuple[str, ...] = ("early", "late")
GRID_HORIZONS: Tuple[str, ...] = ("T+1", "T+2", "T+3")
GRID_SHAPE: Tuple[int, int] = (len(GRID_BUCKETS), len(GRID_HORIZONS))  # (2, 3)

DEFAULT_L2_REG_ALPHA: float = 0.1
DEFAULT_LBFGS_LR: float = 1.0
DEFAULT_LBFGS_MAX_ITER: int = 100
DEFAULT_LBFGS_HISTORY_SIZE: int = 10
DEFAULT_LBFGS_TOLERANCE_GRAD: float = 1e-5
DEFAULT_LBFGS_TOLERANCE_CHANGE: float = 1e-9

# =====================================================================
# 7. Targeted Gating & Evaluation (Milestone 4)
# =====================================================================
ABORT_THRESHOLD: float = 0.05
MASS_CUTOFF_THRESHOLD: float = 0.85
TARGETED_ECE_THRESHOLDS: Tuple[float, ...] = (ABORT_THRESHOLD, MASS_CUTOFF_THRESHOLD)
TARGETED_ECE_BANDWIDTH: float = 0.05
TARGETED_ECE_NUM_BINS: int = 10

# =====================================================================
# 8. Synthetic Model Fixture Configuration (Testing & CI)
# =====================================================================
SYNTHETIC_VOCAB_SIZE: int = 1000
SYNTHETIC_HIDDEN_SIZE: int = 64
SYNTHETIC_INTERMEDIATE_SIZE: int = 128
SYNTHETIC_NUM_HIDDEN_LAYERS: int = 6
SYNTHETIC_NUM_ATTENTION_HEADS: int = 4
SYNTHETIC_NUM_KEY_VALUE_HEADS: int = 4
SYNTHETIC_NUM_EXPERTS: int = 16
SYNTHETIC_NUM_EXPERTS_PER_TOK: int = 4
SYNTHETIC_MOE_INTERMEDIATE_SIZE: int = 64
SYNTHETIC_SHARED_EXPERT_INTERMEDIATE_SIZE: int = 128
SYNTHETIC_TAP_LAYER_INDEX: int = 2
SYNTHETIC_DEEP_LAYERS: Tuple[int, ...] = (3, 4, 5, 6)
SYNTHETIC_DEEP_LAYER_INDICES: Tuple[int, ...] = (2, 3, 4, 5)

# =====================================================================
# 9. Device & Precision Helpers
# =====================================================================
def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Resolve target torch.device with automatic detection.

    Priority order when device is None or 'auto':
    1. Apple Silicon MPS ('mps') if available and built.
    2. NVIDIA CUDA ('cuda') if available.
    3. CPU ('cpu').

    Raises:
        RuntimeError: If explicit device ('mps' or 'cuda') is requested but not available.
    """
    if device is None or device == "auto":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    dev = torch.device(device) if isinstance(device, str) else device
    if dev.type == "mps" and not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
        raise RuntimeError("MPS backend requested but torch.backends.mps is not available/built.")
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA backend requested but torch.cuda is not available.")
    return dev


def resolve_dtype(
    device: Union[str, torch.device],
    dtype: Optional[Union[str, torch.dtype]] = None,
) -> torch.dtype:
    """Resolve and validate torch.dtype for target device.

    CRITICAL CONSTRAINT:
    PyTorch 2.2.2 on macOS MPS does NOT support `torch.bfloat16`
    (raises TypeError: BFloat16 is not supported on MPS).
    Therefore on MPS:
    - Default is `torch.float16`.
    - Explicit `torch.bfloat16` is strictly rejected with a clear ValueError.
    - `torch.float16` and `torch.float32` are allowed.

    On CPU:
    - Default is `torch.float32`.
    - `torch.bfloat16`, `torch.float16`, and `torch.float32` are allowed.

    Raises:
        ValueError: If bfloat16 is requested on MPS or unsupported dtype is specified.
    """
    dev = resolve_device(device) if isinstance(device, str) else device

    dtype_map: Dict[str, torch.dtype] = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }

    resolved: Optional[torch.dtype] = dtype
    if isinstance(dtype, str):
        d_lower = dtype.lower()
        if d_lower in dtype_map:
            resolved = dtype_map[d_lower]
        elif d_lower == "auto":
            resolved = None
        else:
            raise ValueError(f"Unsupported dtype string '{dtype}'. Valid: {list(dtype_map.keys())}")

    if dev.type == "mps":
        if resolved == torch.bfloat16:
            raise ValueError(
                "BFloat16 is not supported on Apple Silicon MPS backend (PyTorch 2.2.2). "
                "Please use torch.float16 or torch.float32 instead."
            )
        if resolved is None:
            return torch.float16
        if resolved not in (torch.float16, torch.float32):
            raise ValueError(f"Unsupported dtype {resolved} for MPS. Use torch.float16 or torch.float32.")
        return resolved

    elif dev.type == "cpu":
        if resolved is None:
            return torch.float32
        return resolved

    elif dev.type == "cuda":
        if resolved is None:
            return torch.float16
        return resolved

    else:
        return resolved if resolved is not None else torch.float32


def validate_device_dtype(device: torch.device, dtype: torch.dtype) -> None:
    """Validate that device and dtype combination is fully supported."""
    resolve_dtype(device, dtype)


# =====================================================================
# 10. Structured Dataclass Configurations
# =====================================================================
@dataclass
class ModelConfig:
    """Base model architecture and tapping configuration."""
    model_name: str = MODEL_NAME
    hidden_size: int = HIDDEN_SIZE
    num_hidden_layers: int = NUM_HIDDEN_LAYERS
    num_experts: int = NUM_EXPERTS
    num_experts_per_tok: int = NUM_EXPERTS_PER_TOK
    tap_layer_index: int = TAP_LAYER_INDEX
    deep_layers: Tuple[int, ...] = DEEP_LAYERS
    deep_layer_indices: Tuple[int, ...] = DEEP_LAYER_INDICES
    early_layers: Tuple[int, ...] = EARLY_LAYERS
    late_layers: Tuple[int, ...] = LATE_LAYERS
    lookahead_horizons: Tuple[int, ...] = LOOKAHEAD_HORIZONS
    device: Optional[str] = None
    dtype: Optional[str] = None


@dataclass
class ExtractionConfig:
    """Corpus streaming and activation harvesting configuration."""
    sequence_length: int = SEQUENCE_LENGTH
    batch_size: int = BATCH_SIZE
    target_tokens: int = TARGET_TOKENS
    total_sequences: int = TOTAL_SEQUENCES
    train_sequences: int = TRAIN_SEQUENCES
    calib_sequences: int = CALIB_SEQUENCES
    train_split_ratio: float = TRAIN_SPLIT_RATIO
    calib_split_ratio: float = CALIB_SPLIT_RATIO
    output_dir: Path = DATA_DIR


@dataclass
class TrainingConfig:
    """Medusa speculative head training configuration."""
    input_dim: int = SPEC_HEAD_INPUT_DIM
    output_dim_per_horizon: int = SPEC_HEAD_OUTPUT_DIM_PER_HORIZON
    num_horizons: int = NUM_HORIZONS
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    batch_size: int = DEFAULT_BATCH_SIZE_TRAIN
    epochs: int = DEFAULT_TRAIN_EPOCHS
    mmce_lambda: float = DEFAULT_MMCE_LAMBDA
    mmce_sigma: float = DEFAULT_MMCE_SIGMA
    checkpoint_path: Path = SPEC_HEAD_CHECKPOINT


@dataclass
class CalibrationConfig:
    """LBFGS temperature grid scaling configuration."""
    grid_shape: Tuple[int, int] = GRID_SHAPE
    lr: float = DEFAULT_LBFGS_LR
    max_iter: int = DEFAULT_LBFGS_MAX_ITER
    history_size: int = DEFAULT_LBFGS_HISTORY_SIZE
    l2_reg_alpha: float = DEFAULT_L2_REG_ALPHA
    checkpoint_path: Path = TEMP_GRID_CHECKPOINT
    json_path: Path = TEMP_GRID_JSON


@dataclass
class EvaluationConfig:
    """Targeted Gating evaluation configuration."""
    thresholds: Tuple[float, ...] = TARGETED_ECE_THRESHOLDS
    abort_threshold: float = ABORT_THRESHOLD
    mass_cutoff_threshold: float = MASS_CUTOFF_THRESHOLD
    bandwidth: float = TARGETED_ECE_BANDWIDTH
    report_path: Path = EVALUATION_REPORT


@dataclass
class PipelineConfig:
    """Complete unified configuration for the calibration pipeline."""
    model: ModelConfig = field(default_factory=ModelConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
