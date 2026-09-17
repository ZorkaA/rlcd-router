# Comprehensive Specification Mining Analysis: Phase 1 PyTorch ML Calibration for Asynchronous MoE Router

**Agent ID**: `teamwork_preview_spec_miner_survey_1`  
**Date**: 2026-09-17  
**Authoritative Sources**:
- `/Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md`
- Hugging Face `Qwen/Qwen1.5-MoE-A2.7B` Architecture & Transformers Source (`transformers.models.qwen2_moe.modeling_qwen2_moe`)
- Foundational Literature:
  - Medusa: Simple LLM Inference Acceleration with Multiple Heads (Cai et al., 2024)
  - Trainable Calibration Measures for Neural Networks from Kernel Mean Embeddings (Kumar, Sarawagi, Jain, ICML 2018)
  - On Calibration of Modern Neural Networks (Guo et al., ICML 2017)

---

## 1. Executive Summary

This document establishes the exhaustive formal specification for Phase 1 PyTorch ML Calibration scripts for an Asynchronous MoE Router. The pipeline comprises four foundational pillars:
1. **R1: Data Partitioning & Generation**: Streaming a 100k-token corpus through `Qwen/Qwen1.5-MoE-A2.7B`, logging Layer $N$ hidden states and native router decisions across all 24 layers, and enforcing a strictly isolated 15–20% held-out calibration split without OOM.
2. **R2: Linear Speculative Head & Training**: Designing and training a Medusa-style linear speculative routing head on Layer $N$ hidden states to forecast deep-layer routing distributions (Layers 5–24) across lookahead horizons $T+1, T+2, T+3$ using Cross-Entropy (CE) plus an optional tunable RKHS Maximum Mean Calibration Error (MMCE) penalty ($\mathcal{L} = \text{CE} + \lambda \cdot \text{MMCE}$).
3. **R3: Grid-Based Temperature Scaling**: Fitting a $2 \times 3$ grid of post-hoc temperature scalars $T(b, h)$ indexed by $\{\text{early layers 5-10}, \text{late layers 11-24}\} \times \{T+1, T+2, T+3\}$ on the held-out calibration split via LBFGS minimizing Negative Log-Likelihood (NLL) with L2 regularization towards $T=1.0$ for sparse buckets, strictly excluding ECE from optimization.
4. **R4: Automated Testing & Verification**: Implementing end-to-end test suites, continuous memory leak monitoring, and evaluating Targeted Gating criteria reporting Targeted ECE at $0.05$ (abort threshold) and $0.85$ (mass cutoff threshold) broken down by layer bucket and horizon.

---

## 2. Authoritative Specification Extraction

### 2.1 System Architecture & Parameters (`Qwen/Qwen1.5-MoE-A2.7B`)
Probed directly from the authoritative model configuration (`Qwen2MoeConfig`):
- **Model Type**: `qwen2_moe` (`Qwen2MoeForCausalLM`)
- **Total Layers ($L_{total}$)**: 24 decoder layers (`num_hidden_layers = 24`, indexed $0 \dots 23$ in PyTorch or $1 \dots 24$ in 1-based indexing).
- **Hidden Dimension ($d_{model}$)**: 2048 (`hidden_size = 2048`).
- **Total Routed Experts ($E$)**: 60 experts (`num_experts = 60`).
- **Routed Experts Selected per Token ($k_{top}$)**: 4 experts (`num_experts_per_tok = 4`).
- **Shared Experts**: 1 shared MLP with intermediate size 5632 (`shared_expert_intermediate_size = 5632`).
- **Gating Mechanism**: Linear layer `gate = nn.Linear(2048, 60, bias=False)`.
  $$\text{router\_logits} = h \cdot W_{gate}^T \in \mathbb{R}^{B \times L \times 60}$$
  $$\text{routing\_weights} = \text{softmax}(\text{router\_logits}, \text{dim}=-1)$$
  $$\text{selected\_experts} = \text{topk}(\text{routing\_weights}, k=4)$$
- **Deep Layers Target**: Layers 5–24 (1-based, corresponding to PyTorch module indices `model.layers[4:24]`, totaling 20 layers).
- **Layer $N$ Tap**: An intermediate pre-deep layer $N \in \{2, 3, 4\}$ (1-based, or indices `1, 2, 3`). Default Layer $N = 3$ or $4$, capturing early contextual representations before deep routing execution.

---

## 3. Mathematical Requirements & Formulations

### 3.1 R2: Loss Formulations (Cross-Entropy & MMCE)

#### 3.1.1 Cross-Entropy Loss ($\mathcal{L}_{CE}$)
For token at position $T$, lookahead horizon $k \in \{1, 2, 3\}$, and deep layer $l \in \{5, \dots, 24\}$:
The speculative head outputs unnormalized logits $\hat{z}_{T, k, l} \in \mathbb{R}^{60}$.
The predicted probability distribution is:
$$\hat{p}_{T, k, l} = \text{softmax}(\hat{z}_{T, k, l}) \in \Delta^{59}$$

With native router ground truth $y_{T+k, l} \in \{0, \dots, 59\}$ (top-1 expert) or native routing distribution $p_{T+k, l}^* \in \Delta^{59}$:
For hard multi-class classification (top-1 expert):
$$\mathcal{L}_{CE}(T, k, l) = -\log \hat{p}_{T, k, l, y_{T+k, l}}$$
For soft target distribution (KL / Soft CE):
$$\mathcal{L}_{CE\_soft}(T, k, l) = -\sum_{e=1}^{60} p_{T+k, l, e}^* \log \hat{p}_{T, k, l, e}$$

The total Cross-Entropy loss over batch of $M$ valid tokens, $K=3$ horizons, and $L_{deep}=20$ layers is:
$$\mathcal{L}_{CE} = \frac{1}{M \cdot K \cdot L_{deep}} \sum_{i=1}^M \sum_{k=1}^K \sum_{l=5}^{24} \mathcal{L}_{CE}(i, k, l)$$

#### 3.1.2 Maximum Mean Calibration Error (MMCE) Penalty
Based on the RKHS formulation of Kumar et al. (ICML 2018):
For a prediction set of size $m$, let $c_i = \max_e \hat{p}_{i, e}$ denote the confidence of sample $i$, and let $r_i = \mathbf{1}\{\hat{y}_i = y_i\}$ be the binary correctness indicator ($1$ if top prediction matches ground truth, $0$ otherwise).

The calibration residual is defined as:
$$e_i = r_i - c_i$$

A universal continuous kernel over confidence values is employed, specifically a Gaussian RBF kernel:
$$k(c_i, c_j) = \exp\left( -\frac{(c_i - c_j)^2}{2\sigma^2} \right)$$
where $\sigma$ is the kernel bandwidth (standard default $\sigma = 0.2$ or median heuristic).

The empirical squared MMCE ($\text{MMCE}^2$) is:
$$\text{MMCE}^2 = \frac{1}{m^2} \sum_{i=1}^m \sum_{j=1}^m e_i e_j k(c_i, c_j) = \frac{1}{m^2} \mathbf{e}^T \mathbf{K} \mathbf{e}$$
where $\mathbf{e} = [e_1, \dots, e_m]^T$ and $\mathbf{K}_{i, j} = k(c_i, c_j)$.

To guarantee numerical stability and smooth backpropagation, the loss penalty is:
$$\text{MMCE} = \sqrt{\max\left(\mathbf{e}^T \mathbf{K} \mathbf{e} / m^2, 0\right) + \epsilon}$$
where $\epsilon = 10^{-8}$.

The composite objective is:
$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \cdot \text{MMCE}$$
where $\lambda \ge 0$ is a tunable hyperparameter. When $\lambda = 0$, standard CE is recovered.

---

### 3.2 R3: Grid-Based Temperature Scaling

#### 3.2.1 Grid Partitioning Structure
The grid defines 6 discrete temperature parameters $T_{b, h} > 0$:
$$\mathcal{G} = \{ b \in \{\text{early}, \text{late}\} \} \times \{ h \in \{T+1, T+2, T+3\} \}$$
- `early layers`: Layers 5–10 ($l \in [5, 10]$, 6 layers).
- `late layers`: Layers 11–24 ($l \in [11, 24]$, 14 layers).
- `horizons`: $h \in \{1, 2, 3\}$.

#### 3.2.2 Optimization Objective per Grid Cell $(b, h)$
For all tokens and layers belonging to bucket $b$ and horizon $h$, with uncalibrated logits $z_i \in \mathbb{R}^{60}$ and true native expert $y_i \in \{0, \dots, 59\}$:
Calibrated logits:
$$\tilde{z}_i(T) = \frac{z_i}{T_{b, h}}$$
Negative Log-Likelihood (NLL):
$$\text{NLL}(T_{b, h}) = -\frac{1}{N_{b, h}} \sum_{i=1}^{N_{b, h}} \left[ \frac{z_{i, y_i}}{T_{b, h}} - \log \sum_{j=1}^{60} \exp\left(\frac{z_{i, j}}{T_{b, h}}\right) \right]$$

#### 3.2.3 L2 Regularization towards $T = 1.0$
To ensure stability and prevent extreme temperature drifts in sparse buckets:
$$\mathcal{L}_{calib}(T_{b, h}) = \text{NLL}(T_{b, h}) + \frac{\gamma}{2} (T_{b, h} - 1.0)^2$$
where $\gamma > 0$ is the regularization coefficient (e.g., $\gamma = 0.01$ or adaptive $\gamma / N_{b, h}$).

#### 3.2.4 LBFGS Optimization Contract
- Strictly executed on the **held-out calibration split** ($15\text{--}20\%$).
- **Strict Invariant**: ECE is NON-DIFFERENTIABLE and is NEVER included in the objective function.
- Initialization: $T_{b, h}^{(0)} = 1.0$ for all 6 cells.
- Solver: `torch.optim.LBFGS(params, lr=0.1, max_iter=50, line_search_fn='strong_wolfe')`.
- Parameter constraint: $T_{b, h} > 0$ enforced via reparameterization $T = \exp(\theta)$ or projected gradient box bounds $[0.01, 10.0]$.

---

### 3.3 R4: Targeted Gating & Evaluation Metrics

#### 3.3.1 Decision Boundaries
In an asynchronous MoE router, routing distributions determine prefetching and hardware offloading:
1. **0.05 Abort Boundary**: Experts with routing probability $< 0.05$ are pruned/aborted. Speculative dispatch is cancelled to save memory and interconnect bandwidth.
2. **0.85 Mass Cutoff Boundary**: Cumulative probability cutoff or high-confidence selection threshold. If an expert or top subset exceeds $0.85$, further expert search and compute are terminated.

#### 3.3.2 Targeted Expected Calibration Error (Targeted ECE)
Standard ECE bins probabilities across $[0, 1]$ into $M$ bins:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$

**Targeted ECE at boundary $p^* \in \{0.05, 0.85\}$**:
Evaluates calibration error in the localized decision window around $p^*$, parameterized by half-width $\delta$ (e.g. $\delta = 0.025$ or local bin $[p^* - \delta, p^* + \delta]$):
$$\text{Targeted-ECE}(p^*) = |\text{acc}(B_{p^*}) - \text{conf}(B_{p^*})|$$
where:
$$B_{p^*} = \{ i : |\hat{p}_i - p^*| \le \delta \}$$
$$\text{conf}(B_{p^*}) = \frac{1}{|B_{p^*}|} \sum_{i \in B_{p^*}} \hat{p}_i$$
$$\text{acc}(B_{p^*}) = \frac{1}{|B_{p^*}|} \sum_{i \in B_{p^*}} \mathbf{1}\{\text{true expert matches event}\}$$

If $|B_{p^*}| = 0$, the metric reports 0.0 with sample count $0$.
Targeted ECE must be evaluated and printed for:
- 2 layer buckets (`early 5-10`, `late 11-24`)
- 3 lookahead horizons ($T+1, T+2, T+3$)
- 2 boundaries ($p^* = 0.05$, $p^* = 0.85$)
- Pre-calibration vs. Post-calibration comparison ($2 \times 3 \times 2 \times 2 = 24$ data points).

---

## 4. Features Discovered

| # | Category | Feature | Description | Inputs | Outputs | Error Behavior | Discovered Via |
|---|----------|---------|-------------|--------|---------|----------------|----------------|
| 1 | R1: Model Loading | `load_qwen_moe_base` | Loads `Qwen/Qwen1.5-MoE-A2.7B` in bfloat16/float16 with `output_hidden_states=True` and `output_router_logits=True`. | `model_name: str`, `device: str`, `dtype: torch.dtype` | `model: Qwen2MoeForCausalLM`, `tokenizer: AutoTokenizer` | Raises `OSError` if weights missing; raises `OutOfMemoryError` if insufficient VRAM/RAM. | Hugging Face model config & `ORIGINAL_REQUEST.md` R1 |
| 2 | R1: Streaming Pipeline | `stream_corpus_tokens` | Streams 100k tokens in fixed chunk sizes (e.g. $L=512/1024$) without materializing the full dataset in memory. | `dataset_name: str`, `tokenizer`, `total_tokens: int=100000`, `seq_len: int=1024` | Generator yielding `torch.Tensor` of shape `(batch, seq_len)` | Raises `ValueError` if token count unreachable; handles short docs with truncation/padding. | `ORIGINAL_REQUEST.md` R1 |
| 3 | R1: State Extraction | `extract_layer_n_and_router_logits` | Executes forward pass under `torch.no_grad()`, extracts Layer $N$ hidden states and native router logits for all 24 layers. | `model`, `input_ids: Tensor (B, L)`, `layer_n: int=3` | `hidden_states_n: Tensor (B, L, 2048)`, `router_logits: Tuple[Tensor (B*L, 60)]` | Raises `IndexError` if `layer_n` out of range $[0, 24]$; OOM if not under `no_grad()`. | `ORIGINAL_REQUEST.md` R1 & `Qwen2MoeModel` source |
| 4 | R1: Split Isolation | `partition_train_calib_split` | Carves off 15–20% held-out calibration split strictly at sequence/document boundaries with deterministic random seed. | `dataset_records`, `calib_ratio: float=0.20`, `seed: int=42` | `train_data: Dict/Path`, `calib_data: Dict/Path` | Raises `AssertionError` if overlap detected between train and calib token IDs. | `ORIGINAL_REQUEST.md` R1 |
| 5 | R1: Memory Offloading | `chunked_disk_serializer` | Flushes batches of extracted tensors to disk in memory-mapped or chunked `.pt` files, running garbage collection. | `tensors: Dict[str, Tensor]`, `output_dir: str`, `chunk_idx: int` | `saved_paths: List[str]` | Raises `IOError` if disk full; warns if RAM allocation increases after GC. | `ORIGINAL_REQUEST.md` R1 & R4 |
| 6 | R2: Head Architecture | `MedusaLinearSpeculativeHead` | Linear speculative module attached to Layer $N$ hidden state ($d=2048$), outputting routing logits for layers 5–24 and horizons $T+1..T+3$. | `hidden_state: Tensor (B, L, 2048)` | `pred_logits: Tensor (B, L, 3, 20, 60)` | Raises `ValueError` if input tensor last dimension $\ne 2048$. | `ORIGINAL_REQUEST.md` R2 & Medusa spec |
| 7 | R2: Cross-Entropy Loss | `speculative_cross_entropy_loss` | Computes multi-class or soft cross-entropy between predicted routing distributions and native router decisions across deep layers. | `pred_logits: (B, L, 3, 20, 60)`, `targets: (B, L, 3, 20)`, `mask: (B, L)` | `loss: Tensor scalar` | Handles sequence boundary padding/masking at $T+1, T+2, T+3$. | `ORIGINAL_REQUEST.md` R2 |
| 8 | R2: RKHS MMCE Penalty | `compute_rkhs_mmce` | Computes differentiable empirical Maximum Mean Calibration Error using Gaussian RBF kernel over predicted confidences. | `confidences: Tensor (M,)`, `correctness: Tensor (M,)`, `sigma: float=0.2` | `mmce_loss: Tensor scalar` | Adds $\epsilon=10^{-8}$ inside $\sqrt{\cdot}$ to prevent NaN gradients when error approaches 0. | `ORIGINAL_REQUEST.md` R2 & Kumar et al. (2018) |
| 9 | R2: Speculative Trainer | `train_speculative_head` | Trains speculative head parameters with AdamW using composite loss $\mathcal{L} = \text{CE} + \lambda \cdot \text{MMCE}$, logging loss and accuracy curves. | `head: nn.Module`, `train_loader`, `epochs: int`, `lr: float`, `lambda_mmce: float` | `trained_head`, `training_history: Dict[str, List[float]]` | Detects exploding gradients; validates $\lambda \ge 0$. | `ORIGINAL_REQUEST.md` R2 |
| 10 | R3: Grid Indexing | `RouterCalibrationGrid` | Manages the $2 \times 3$ matrix of temperature scalars indexed by `{early (5-10), late (11-24)} x {T+1, T+2, T+3}`. | None | Grid container initialized to $1.0$ with bucket mapping logic | Raises `KeyError` if query layer $< 5$ or $> 24$, or horizon $\notin \{1, 2, 3\}$. | `ORIGINAL_REQUEST.md` R3 |
| 11 | R3: LBFGS Calibration | `fit_grid_temperature_scaling` | Optimizes temperature scalars per bucket on held-out split using LBFGS to strictly minimize NLL with L2 regularizer towards $T=1.0$. | `calib_logits: Dict`, `calib_targets: Dict`, `l2_reg: float=0.01` | `calibrated_grid: RouterCalibrationGrid` | Fails fast if ECE metric is passed to optimizer closure. Enforces $T > 0$. | `ORIGINAL_REQUEST.md` R3 |
| 12 | R3: Calibrated Inference | `apply_calibrated_temperature` | Applies fitted temperatures $T(b, h)$ to speculative logits to produce calibrated routing probabilities. | `raw_logits: Tensor`, `layer_idx: int`, `horizon: int` | `calibrated_probs: Tensor` | Raises `ValueError` if temperature $\le 0$. | `ORIGINAL_REQUEST.md` R3 |
| 13 | R4: Targeted ECE Metric | `compute_targeted_ece` | Computes local expected calibration error specifically at decision thresholds $0.05$ (abort) and $0.85$ (mass cutoff). | `probs: Tensor`, `targets: Tensor`, `threshold: float`, `window: float=0.025` | `Dict[str, float]` with `targeted_ece`, `sample_count`, `bin_acc`, `bin_conf` | Returns `0.0` with `sample_count=0` if no predictions fall in boundary window. | `ORIGINAL_REQUEST.md` R4 |
| 14 | R4: Evaluation Reporter | `evaluate_targeted_gating_report` | Generates comprehensive calibration report comparing pre- and post-calibration Targeted ECE at $0.05$ and $0.85$ across all 6 grid buckets. | `raw_preds`, `calibrated_preds`, `ground_truth` | Formatted Markdown / tabular report string and structured dictionary | Raises `ValueError` if missing buckets or empty evaluation data. | `ORIGINAL_REQUEST.md` R4 |
| 15 | R4: Memory Monitor | `MemoryLeakTracker` | Continuously samples RAM (RSS) and device VRAM/MPS memory during streaming and training to detect monotonic leaks. | `step_name: str` | `memory_stats: Dict[str, float]` | Logs warning or raises test failure if memory growth exceeds leak threshold across iterations. | `ORIGINAL_REQUEST.md` R4 |
| 16 | R4: Mock Test Generator | `SyntheticMoEDataGenerator` | Generates synthetic hidden states and router logits matching exact Qwen1.5-MoE tensor schemas for fast, offline unit & integration testing. | `num_samples: int`, `seq_len: int`, `num_experts: int=60`, `d_model: int=2048` | Synthetic dataset dictionary | Validates all downstream components without requiring 28GB model download. | Acceptance Criteria: CI/Test Suite |

---

## 5. Edge Cases & Boundary Conditions

| # | Feature | Input / Condition | Observed / Required Behavior |
|---|---------|-------------------|------------------------------|
| 1 | Lookahead Slicing | Token index $T \ge L - 3$ at sequence end | Targets for $T+1, T+2, T+3$ extend beyond sequence length $L$. Head must truncate or apply causal sequence mask so tail tokens do not access out-of-bounds targets or wrap around. |
| 2 | Sparse / Empty Grid Buckets | Held-out calibration split has very few samples in late layer + $T+3$ bucket | NLL gradient becomes unstable or unconstrained. L2 regularization $\frac{\gamma}{2}(T - 1.0)^2$ pulls temperature toward $1.0$, preventing divergence. |
| 3 | MMCE Single Sample / Constant Confidence | Batch size $m=1$ or all confidences identical ($c_i - c_j = 0$) | Kernel matrix $\mathbf{K}$ has all entries $1.0$. The numerator $\mathbf{e}^T \mathbf{K} \mathbf{e}$ can become zero. Without $\epsilon=10^{-8}$, $\frac{d}{dx} \sqrt{x}$ at $x=0$ yields `NaN` gradients. Must use $\sqrt{x + \epsilon}$. |
| 4 | Temperature Near Zero or Negative | Optimizer proposes $T \le 0$ during line search | Division by zero or flipped sign in logits. Enforce strictly positive temperature via parameterization $T = \exp(\theta)$ or box bounds $[0.01, 10.0]$ in LBFGS. |
| 5 | Float16 / Bfloat16 Underflow in Scaled Softmax | Extreme logit values divided by small temperature ($T < 0.1$) | Exponentiation overflows or underflows in 16-bit. Cast logits to `float32` before temperature scaling and softmax computation. |
| 6 | Zero Predictions in Targeted Window | No tokens predict confidence in $[0.05 - \delta, 0.05 + \delta]$ or $[0.85 - \delta, 0.85 + \delta]$ | Targeted ECE bin is empty ($|B_{p^*}| = 0$). Must return `targeted_ece = 0.0` with `sample_count = 0` rather than dividing by zero. |
| 7 | Offline / CI Execution | Running `pytest` in an environment without internet access or GPU | Cannot download 14B Qwen weights. Must support synthetic/mock mode using `SyntheticMoEDataGenerator` or lightweight test configuration so all tests pass end-to-end. |
| 8 | Memory Leak via Un-detached Tensors | Hidden states or router logits retained in memory with computation graph attached | Retaining autograd graph across 100k tokens exhausts 36GB RAM in <5 batches. All extracted states must be explicitly `.detach().cpu()` before buffering or saving. |
| 9 | Optimization Objective Verification | Developer mistakenly incorporates ECE into LBFGS loss function | ECE is non-differentiable step-function. An assertion in `fit_grid_temperature_scaling` must explicitly verify that the loss function is strictly NLL (+ L2 regularizer) and disallow any ECE objective. |
| 10 | Strict Train/Calibration Leakage | Accidental token overlap due to sliding window tokenization | Token IDs from training split present in calibration split. Must perform disjoint hash or document-level split validation to guarantee 0% data leakage. |

---

## 6. Interface Contracts & Data Schemas

### 6.1 Extracted Data Schema (`data_partitioning`)
```python
{
    "token_ids": torch.Tensor,        # (B, L) int64
    "layer_n_hidden_states": torch.Tensor, # (B, L, 2048) float32/bfloat16
    "router_logits": torch.Tensor,    # (B, L, 24, 60) float32
    "top1_expert_indices": torch.Tensor, # (B, L, 24) int64
    "top4_expert_indices": torch.Tensor, # (B, L, 24, 4) int64
    "top4_routing_weights": torch.Tensor # (B, L, 24, 4) float32
}
```

### 6.2 Medusa Linear Speculative Head Interface (`speculative_head`)
```python
class MedusaLinearSpeculativeHead(nn.Module):
    def __init__(
        self,
        d_model: int = 2048,
        num_deep_layers: int = 20, # Layers 5-24
        horizons: List[int] = [1, 2, 3],
        num_experts: int = 60
    ): ...

    def forward(self, h_n: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_n: Hidden state at Layer N of shape (B, L, d_model)
        Returns:
            pred_logits: Tensor of shape (B, L, len(horizons), num_deep_layers, num_experts)
                         i.e. (B, L, 3, 20, 60)
        """
```

### 6.3 Temperature Grid Interface (`temperature_scaling`)
```python
class RouterCalibrationGrid(nn.Module):
    def __init__(self):
        # 6 positive scalar parameters
        # buckets: 'early' (layers 5-10), 'late' (layers 11-24)
        # horizons: 'T+1', 'T+2', 'T+3'
        ...

    def get_temperature(self, layer_idx: int, horizon: int) -> torch.Tensor: ...
    def forward(self, logits: torch.Tensor, layer_idx: int, horizon: int) -> torch.Tensor: ...
```

### 6.4 Targeted Gating Evaluation Output Contract (`verification`)
```python
{
    "early_layers": {
        "T+1": {"target_0.05": {"ece_uncal": float, "ece_cal": float, "count": int},
                "target_0.85": {"ece_uncal": float, "ece_cal": float, "count": int}},
        "T+2": { ... },
        "T+3": { ... }
    },
    "late_layers": {
        "T+1": { ... },
        "T+2": { ... },
        "T+3": { ... }
    }
}
```

---

## 7. Acceptance Criteria Verification Plan

| Criterion | Verification Method | Pass Threshold |
|---|---|---|
| **AC1: 100k-token Data Generation without OOM** | Run generation on streaming corpus; monitor RSS via `psutil`. | Completes 100k tokens without crashing; RSS remains bounded (<80% RAM). |
| **AC2: Speculative Head Training & Loss Curves** | Train head on training split; log loss curves for $\mathcal{L}_{total}, \mathcal{L}_{CE}, \text{MMCE}$. | Loss decreases monotonically; validation loss converges; log files emitted. |
| **AC3: LBFGS Temperature Scaling Output** | Run LBFGS calibration on held-out split; output $2 \times 3$ grid. | Returns 6 scalars $T > 0$; calibration NLL lower than uncalibrated NLL. |
| **AC4: Automated Test Suite End-to-End** | Run `pytest -v tests/`. | 100% tests pass with 0 errors across unit, integration, and memory tests. |
| **AC5: Targeted ECE at 0.05 and 0.85** | Run final evaluation script on held-out split. | Table of Targeted ECE at $0.05$ and $0.85$ printed in evaluation report. |
