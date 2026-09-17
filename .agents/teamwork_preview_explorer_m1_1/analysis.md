# Technical Specification & Architectural Blueprint: Model Loading and Global Configuration

**Author**: `teamwork_preview_explorer_m1_1`  
**Target Modules**: `src/config.py` and `src/data/model_loader.py`  
**Milestone**: M1 (Data Partitioning & Generation)  
**Date**: 2026-09-17  

---

## 1. Executive Summary

This investigation establishes the exact technical specifications, device/dtype resolution logic, configuration constants, and code blueprints for `src/config.py` and `src/data/model_loader.py`. 

### Key Findings & Determinations:
1. **MPS BFloat16 Incompatibility**: Under PyTorch 2.2.2 on macOS ARM64, attempting to allocate or cast tensors to `torch.bfloat16` on Apple Silicon MPS raises verbatim `TypeError: BFloat16 is not supported on MPS`. Consequently, all operations on `device="mps"` **MUST** strictly enforce `torch.float16` (or `torch.float32`), while CPU supports `torch.float32` and `torch.bfloat16`.
2. **MoE Output Structure Parity**: In HuggingFace `Qwen2MoeForCausalLM`, `outputs.hidden_states` is a tuple of length 25 (index 0 is token embeddings, indices 1–24 are decoder layer outputs). Layer 3 output is accessed at `outputs.hidden_states[3]`. Conversely, `outputs.router_logits` is a tuple of 24 tensors (one per decoder layer, with no embedding offset), where each tensor has shape `(B * L, num_experts)`. Deep layers (Layers 5–24) correspond exactly to slice indices `4..23`.
3. **Zero-Download Synthetic Fixture**: The full `Qwen1.5-MoE-A2.7B` model weighs 26.67 GB and is not fully cached locally. To enable sub-second CI and opaque-box testing without external network access, we designed `get_synthetic_model()`: a scaled-down `Qwen2MoeConfig` ($d=64$, 6 layers, 16 experts, top-4) having ~1.56M parameters (<6 MB RAM) that initializes in 0.05 seconds and executes full 1024-token forward passes in 0.6 seconds while maintaining 100% architectural parity.

---

## 2. Empirical Findings & Host Runtime Analysis

### 2.1 Host Environment Measurements
- **Python**: 3.10.14 (`/opt/anaconda3/bin/python3`)
- **PyTorch**: 2.2.2
- **Transformers**: 4.44.0
- **Hardware**: Apple Silicon ARM64 (14 cores), 36.00 GB unified RAM, 187 GB free disk.
- **Backends**: `torch.cuda.is_available() == False`, `torch.backends.mps.is_available() == True`, `torch.backends.mps.is_built() == True`.

### 2.2 Device & Precision Failure Modes
We executed direct empirical probes on device and dtype behaviors:

| Device | Dtype | Empirical Result | Traceback / Behavior |
| :--- | :--- | :--- | :--- |
| `mps` | `torch.float32` | **PASS** | Synchronized tensor arithmetic succeeds. |
| `mps` | `torch.float16` | **PASS** | Fast GEMM execution ($1.27 \text{ ms}$ per $1024 \times 2048 \times 2048$ matmul). |
| `mps` | `torch.bfloat16` | **FAIL** | `TypeError: BFloat16 is not supported on MPS` |
| `mps` | `tensor.to('mps')` from CPU bf16 | **FAIL** | `TypeError: BFloat16 is not supported on MPS` |
| `cpu` | `torch.float32` | **PASS** | Full compatibility. |
| `cpu` | `torch.bfloat16` | **PASS** | CPU native vector execution succeeds. |
| `cpu` | `torch.float16` | **PASS** | Standard CPU execution succeeds. |

**Critical Architectural Decision**:
In `src/config.py` and `src/data/model_loader.py`:
- Any request for `device="mps"` defaults to `torch.float16`.
- Any explicit request for `bfloat16` when targeting `mps` triggers an immediate, helpful `ValueError`:  
  `"BFloat16 is not supported on Apple Silicon MPS backend (PyTorch 2.2.2). Please use torch.float16 or torch.float32 instead."`
- The base model's default `config.json` specifies `"torch_dtype": "bfloat16"`. If loaded naively with `torch_dtype="auto"`, PyTorch attempts to load BF16 onto MPS and crashes. The loader **MUST** override this with `torch_dtype=torch.float16`.

### 2.3 HuggingFace Hub Local Cache Audit
Inspection of `/Users/jack/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B`:
- `config.json`: Present (24 layers, 2048 hidden size, 60 experts, top-4).
- `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`: Present (151,646 tokens).
- Weight shards (`model-00001-of-00008.safetensors`, etc.): **Not present** locally.
- **Implication**: Unit tests and local development cannot depend on downloading 26.67 GB on every run. The synthetic model fixture `get_synthetic_model()` is mandatory for automated test execution.

---

## 3. Architecture & Output Mechanics of `Qwen2MoeForCausalLM`

### 3.1 Model Dimensions
```
Model Identifier: Qwen/Qwen1.5-MoE-A2.7B
├── Hidden size (d_model): 2048
├── Total decoder layers: 24
├── Attention heads: 16 (Query), 16 (Key/Value)
├── Intermediate size: 5632 (Dense FFN fallback)
├── Routed experts per layer: 60
├── Active experts per token (top_k): 4
├── MoE intermediate size: 1408 (per routed expert)
├── Shared expert intermediate size: 5632
├── Decoder sparse step: 1 (MoE present at EVERY layer)
└── Vocab size: 151936
```

### 3.2 Output Tensor Shapes & Slicing Specifications

When called with `output_hidden_states=True, output_router_logits=True, use_cache=False`:

#### 1. Hidden States (`outputs.hidden_states`):
- Type: `Tuple[torch.Tensor, ...]`
- Length: `25` (`num_hidden_layers + 1`)
- `hidden_states[0]`: Embedding layer representation, shape `(B, L, 2048)`.
- `hidden_states[1]`: Layer 1 representation, shape `(B, L, 2048)`.
- `hidden_states[3]`: **Layer 3 representation (Tap Layer)**, shape `(B, L, 2048)`.
- `hidden_states[24]`: Layer 24 representation, shape `(B, L, 2048)`.

#### 2. Router Logits (`outputs.router_logits`):
- Type: `Tuple[torch.Tensor, ...]`
- Length: `24` (`num_hidden_layers`)
- Indexing: `router_logits[0]` corresponds to Layer 1, ..., `router_logits[23]` corresponds to Layer 24.
- Tensor Shape: `(B * L, 60)` (flattened batch and sequence dimension).
- For $B=1, L=1024$: shape is `(1024, 60)`.
- For batch extraction: reshape via `.view(B, L, 60)`.

#### 3. Deep Layers Slicing:
Deep layers are defined as Layers 5–24 (1-indexed, 20 layers total).
- 0-based router logit indices: `range(4, 24)` (i.e. indices 4 through 23).
- Early deep bucket: Layers 5–10 $\to$ indices `4..9` (6 layers, slice `[0:6]` of deep layers).
- Late deep bucket: Layers 11–24 $\to$ indices `10..23` (14 layers, slice `[6:20]` of deep layers).

#### 4. Cache Disabling:
- `use_cache=False` must be configured both on `model.config.use_cache = False` and in forward calls to eliminate KV-cache memory allocation during token streaming.

---

## 4. Technical Specification for `src/config.py`

`src/config.py` acts as the single source of truth for all constants, paths, hyperparameter presets, device helpers, and structured configuration dataclasses.

### 4.1 Constants Catalog

| Category | Constant Name | Type | Value | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Model** | `MODEL_NAME` | `str` | `"Qwen/Qwen1.5-MoE-A2.7B"` | HuggingFace model identifier |
| | `HIDDEN_SIZE` | `int` | `2048` | Model hidden dimension $d_{\text{model}}$ |
| | `NUM_HIDDEN_LAYERS` | `int` | `24` | Total decoder layers |
| | `NUM_EXPERTS` | `int` | `60` | Routed experts per MoE layer |
| | `NUM_EXPERTS_PER_TOK` | `int` | `4` | Top-$k$ routed experts per token |
| | `VOCAB_SIZE` | `int` | `151936` | Tokenizer vocabulary size |
| **Layer Tapping** | `TAP_LAYER_INDEX` | `int` | `3` | Index in `outputs.hidden_states` (Layer 3) |
| | `DEEP_LAYERS` | `Tuple[int, ...]` | `(5..24)` | 1-indexed deep layer IDs (20 layers) |
| | `DEEP_LAYER_INDICES` | `Tuple[int, ...]` | `(4..23)` | 0-indexed router logits indices |
| | `EARLY_LAYERS` | `Tuple[int, ...]` | `(5..10)` | 1-indexed early deep layers (6 layers) |
| | `EARLY_LAYER_INDICES` | `Tuple[int, ...]` | `(4..9)` | 0-indexed early deep router logits |
| | `LATE_LAYERS` | `Tuple[int, ...]` | `(11..24)` | 1-indexed late deep layers (14 layers) |
| | `LATE_LAYER_INDICES` | `Tuple[int, ...]` | `(10..23)` | 0-indexed late deep router logits |
| | `NUM_DEEP_LAYERS` | `int` | `20` | Count of deep layers |
| | `NUM_EARLY_LAYERS` | `int` | `6` | Count of early deep layers |
| | `NUM_LATE_LAYERS` | `int` | `14` | Count of late deep layers |
| **Horizons** | `LOOKAHEAD_HORIZONS` | `Tuple[int, ...]` | `(1, 2, 3)` | Lookahead steps ($T+1, T+2, T+3$) |
| | `NUM_HORIZONS` | `int` | `3` | Number of lookahead steps |
| | `MAX_LOOKAHEAD` | `int` | `3` | Maximum lookahead step |
| **Streaming** | `SEQUENCE_LENGTH` | `int` | `1024` | Sequence chunk length $L$ |
| | `BATCH_SIZE` | `int` | `1` | Batch size $B$ for zero-OOM generation |
| | `TARGET_TOKENS` | `int` | `100_000` | Target token corpus size |
| | `TOTAL_SEQUENCES` | `int` | `98` | 98 seqs $\times$ 1024 = 100,352 tokens |
| | `TRAIN_SEQUENCES` | `int` | `80` | Train partition (81,920 tokens, 81.6%) |
| | `CALIB_SEQUENCES` | `int` | `18` | Calib partition (18,432 tokens, 18.4%) |
| | `TRAIN_SPLIT_RATIO` | `float` | `0.80` | Nominal train split target |
| | `CALIB_SPLIT_RATIO` | `float` | `0.20` | Nominal calib split target |
| | `BOUNDARY_TRIM_TOKENS`| `int` | `3` | Trailing tokens masked ($L-3..L-1$) |
| **Spec Head** | `SPEC_HEAD_INPUT_DIM` | `int` | `2048` | Input dimension ($d_{\text{model}}$) |
| | `SPEC_HEAD_OUTPUT_DIM_PER_HORIZON` | `int` | `1200` | $20 \text{ layers} \times 60 \text{ experts}$ |
| | `SPEC_HEAD_TOTAL_OUTPUT_DIM` | `int` | `3600` | $3 \times 1200$ total speculative outputs |
| | `DEFAULT_LEARNING_RATE` | `float` | `1e-3` | Training learning rate |
| | `DEFAULT_WEIGHT_DECAY` | `float` | `1e-4` | AdamW weight decay |
| | `DEFAULT_MMCE_LAMBDA` | `float` | `1.0` | Tunable MMCE penalty weight |
| | `DEFAULT_MMCE_SIGMA` | `float` | `0.2` | RBF kernel bandwidth |
| **Temp Grid** | `GRID_BUCKETS` | `Tuple[str, str]` | `("early", "late")` | Bucket names |
| | `GRID_HORIZONS` | `Tuple[str, ...]` | `("T+1", "T+2", "T+3")` | Horizon names |
| | `GRID_SHAPE` | `Tuple[int, int]` | `(2, 3)` | Grid dimensions |
| | `DEFAULT_L2_REG_ALPHA`| `float` | `0.1` | Regularization toward $T=1.0$ |
| | `DEFAULT_LBFGS_LR` | `float` | `1.0` | LBFGS optimizer step size |
| | `DEFAULT_LBFGS_MAX_ITER`| `int` | `100` | Max iterations per optimization step |
| **Evaluation** | `ABORT_THRESHOLD` | `float` | `0.05` | Speculative execution abort boundary |
| | `MASS_CUTOFF_THRESHOLD`| `float` | `0.85` | Expert cumulative probability cutoff |
| | `TARGETED_ECE_THRESHOLDS`| `Tuple[float, float]` | `(0.05, 0.85)` | Decision boundaries for T-ECE |
| **Synthetic** | `SYNTHETIC_VOCAB_SIZE`| `int` | `1000` | Test vocabulary size |
| | `SYNTHETIC_HIDDEN_SIZE`| `int` | `64` | Test hidden dimension |
| | `SYNTHETIC_NUM_HIDDEN_LAYERS` | `int` | `6` | Test decoder layers |
| | `SYNTHETIC_NUM_EXPERTS`| `int` | `16` | Test routed experts |
| | `SYNTHETIC_NUM_EXPERTS_PER_TOK` | `int` | `4` | Test top-$k$ |
| | `SYNTHETIC_TAP_LAYER_INDEX` | `int` | `2` | Test tap layer |
| | `SYNTHETIC_DEEP_LAYERS`| `Tuple[int, ...]` | `(3, 4, 5, 6)` | Test deep layers (1-indexed) |
| | `SYNTHETIC_DEEP_LAYER_INDICES`| `Tuple[int, ...]` | `(2, 3, 4, 5)` | Test deep layer router indices |
| **Paths** | `TRAIN_DATA_FILE` | `Path` | `data/train_data.safetensors` | Training dataset artifact |
| | `CALIB_DATA_FILE` | `Path` | `data/calib_data.safetensors` | Held-out calibration artifact |
| | `SPEC_HEAD_CHECKPOINT`| `Path` | `checkpoints/speculative_head.pt`| Speculative head model |
| | `TEMP_GRID_CHECKPOINT`| `Path` | `checkpoints/temperature_grid.pt`| Fitted temperature grid |
| | `EVALUATION_REPORT` | `Path` | `results/evaluation_report.json` | Final evaluation metrics report |

### 4.2 Device Resolution Algorithm
```python
def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    if device is None or device == "auto":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    # Explicit validation...
```

### 4.3 Dtype Resolution Algorithm
```python
def resolve_dtype(
    device: Union[str, torch.device],
    dtype: Optional[Union[str, torch.dtype]] = None,
) -> torch.dtype:
    dev = resolve_device(device) if isinstance(device, str) else device
    # Normalizes strings ('float16', 'fp16', etc.)
    # If device.type == 'mps':
    #   Rejects torch.bfloat16 with descriptive ValueError
    #   Defaults to torch.float16
    # If device.type == 'cpu':
    #   Defaults to torch.float32
```

---

## 5. Technical Specification for `src/data/model_loader.py`

### 5.1 Public Function Signatures

#### `load_model`
```python
def load_model(
    model_name_or_path: str = MODEL_NAME,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    output_hidden_states: bool = True,
    output_router_logits: bool = True,
    use_cache: bool = False,
    trust_remote_code: bool = True,
    local_files_only: bool = False,
) -> PreTrainedModel:
```
- Resolves device and dtype via `resolve_device` and `resolve_dtype`.
- Invokes `AutoModelForCausalLM.from_pretrained(...)`.
- Explicitly enforces:
  ```python
  model.config.output_hidden_states = output_hidden_states
  model.config.output_router_logits = output_router_logits
  model.config.use_cache = use_cache
  ```
- Places model on target device and sets `model.eval()`.

#### `load_tokenizer`
```python
def load_tokenizer(
    model_name_or_path: str = MODEL_NAME,
    trust_remote_code: bool = True,
    local_files_only: bool = False,
) -> PreTrainedTokenizerBase:
```
- Loads tokenizer from cache or Hub.
- Automatically assigns `tokenizer.pad_token = tokenizer.eos_token` if unassigned.

#### `load_model_and_tokenizer`
```python
def load_model_and_tokenizer(
    model_name_or_path: str = MODEL_NAME,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    output_hidden_states: bool = True,
    output_router_logits: bool = True,
    use_cache: bool = False,
    trust_remote_code: bool = True,
    local_files_only: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
```

### 5.2 Synthetic Model Factory & Tokenizer Mock

#### `get_synthetic_config`
Generates a minimal `Qwen2MoeConfig` ($d=64$, 6 layers, 16 experts, top-4).

#### `get_synthetic_model`
```python
def get_synthetic_model(
    vocab_size: int = SYNTHETIC_VOCAB_SIZE,
    hidden_size: int = SYNTHETIC_HIDDEN_SIZE,
    intermediate_size: int = SYNTHETIC_INTERMEDIATE_SIZE,
    num_hidden_layers: int = SYNTHETIC_NUM_HIDDEN_LAYERS,
    num_attention_heads: int = SYNTHETIC_NUM_ATTENTION_HEADS,
    num_key_value_heads: int = SYNTHETIC_NUM_KEY_VALUE_HEADS,
    num_experts: int = SYNTHETIC_NUM_EXPERTS,
    num_experts_per_tok: int = SYNTHETIC_NUM_EXPERTS_PER_TOK,
    moe_intermediate_size: int = SYNTHETIC_MOE_INTERMEDIATE_SIZE,
    shared_expert_intermediate_size: int = SYNTHETIC_SHARED_EXPERT_INTERMEDIATE_SIZE,
    device: Union[str, torch.device] = "cpu",
    dtype: Optional[Union[str, torch.dtype]] = None,
    output_hidden_states: bool = True,
    output_router_logits: bool = True,
    use_cache: bool = False,
) -> Qwen2MoeForCausalLM:
```
- Instantiates `Qwen2MoeForCausalLM(config)` without network or disk IO.
- Defaults to CPU for fast test execution, but fully supports MPS with `torch.float16`.
- Guards against BFloat16 on MPS.

#### `SyntheticTokenizer`
A lightweight class implementing:
- `encode(text: str, return_tensors: Optional[str] = None)`
- `decode(token_ids: Union[List[int], torch.Tensor])`
- `__call__(text: Union[str, List[str]], return_tensors: Optional[str] = None)`
- Attributes: `vocab_size`, `pad_token_id`, `eos_token_id`, `bos_token_id`.

### 5.3 Synthetic Model Benchmarks

| Metric | CPU (Float32) | MPS (Float16) |
| :--- | :--- | :--- |
| Model Instantiation Time | 0.052 s | 0.058 s |
| Sequence $L=32$ Forward Pass | 0.012 s | 0.008 s |
| Sequence $L=1024$ Forward Pass | 0.610 s | 0.440 s |
| Total Model Parameters | 1,561,920 | 1,561,920 |
| Memory Footprint in RAM | ~6.2 MB | ~3.1 MB |

---

## 6. Integration Contract Alignment

### 6.1 Alignment with Explorer 2 (`stream_extractor.py`)
- **Tap Layer**: Explorer 2 accesses `outputs.hidden_states[TAP_LAYER_INDEX]` (index 3). Shape: `(B, L, 2048)`.
- **Router Logits Extraction**: Explorer 2 iterates over `DEEP_LAYER_INDICES` (indices 4..23) to harvest deep router logits.
- **Logit Shape Handling**: Explorer 2 accounts for Transformers flattening router logits to `(B * L, num_experts)`. For $B=1$, this is `(L, 60)`; for general batches, Explorer 2 will reshape to `(B, L, 60)`.
- **Memory Hygiene**: Disabling KV cache (`use_cache=False`) is baked into `load_model` defaults.

### 6.2 Alignment with Explorer 3 (`dataset.py`)
- **Sequences & Partitioning**: Explorer 3 splits the 98 sequences into 80 train (81.6%) and 18 calib (18.4%).
- **Boundary Trimming**: Explorer 3 masks the last `BOUNDARY_TRIM_TOKENS` (3 tokens, $L-3..L-1$) per sequence.
- **Artifact Paths**: Explorer 3 persists to `TRAIN_DATA_FILE` and `CALIB_DATA_FILE`.

### 6.3 Alignment with E2E Testing Track (`tests/conftest.py`)
- `tests/conftest.py` can directly import `get_synthetic_model` and `get_synthetic_tokenizer` from `src.data.model_loader`.
- Fixtures can instantiate CPU and MPS test models in <0.1s without downloading 26 GB weights.

---

## 7. Artifact Deliverables in Working Directory

1. `proposed_config.py`: Complete, syntax-verified implementation of `src/config.py`.
2. `proposed_model_loader.py`: Complete, syntax-verified implementation of `src/data/model_loader.py`.
3. `analysis.md`: This comprehensive specification document.
4. `handoff.md`: Formal 5-component handoff report.
