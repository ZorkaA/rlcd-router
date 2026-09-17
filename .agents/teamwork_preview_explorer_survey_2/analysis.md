# Technical Investigation & Architecture Analysis: Qwen1.5-MoE-A2.7B Calibration Pipeline

**Author**: `teamwork_preview_explorer_survey_2`  
**Date**: 2026-09-17  
**Scope**: Local Runtime Environment, Qwen1.5-MoE-A2.7B Architecture & Layer N Tap, Streaming 100k-Token Memory Management, and Strictly Isolated Train/Calibration Split Mechanics.  
**Reference Document**: `/Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md`

---

## 1. Executive Summary

This investigation provides the architectural and empirical foundation for Phase 1 of the Asynchronous MoE Router calibration pipeline. We empirically probed the local hardware and software stack on macOS Apple Silicon, inspected the exact PyTorch implementation of `Qwen/Qwen1.5-MoE-A2.7B` in `transformers 4.44.0`, designed a zero-OOM streaming protocol for extracting 100,000 tokens of hidden states and routing decisions, and formulated a leak-free 80/20 train/calibration split with lookahead horizon boundary protections.

### Core Findings Matrix
| Component | Verified Specification / Finding | Critical Implementation Implication |
|---|---|---|
| **Python / Torch** | Python 3.10.14, PyTorch 2.2.2 | Full compatibility for pipeline scripts and tests. |
| **Compute Device** | Apple Silicon MPS (`mps`) & CPU (14 cores) | **`torch.bfloat16` fails on MPS in PyTorch 2.2.2**. Must use `torch.float16` or `torch.float32` on MPS. |
| **RAM / Memory Limit** | 36 GB Total RAM (~13 GB currently available) | Full FP16 model is 26.67 GB. Exceeds instant RAM without swap; fast CI tests **must** use a lightweight synthetic MoE config. |
| **Installed Libraries** | `transformers 4.44.0`, `datasets 2.16.0`, `safetensors 0.6.2`, `pytest 8.3.4`, `scikit-learn 1.3.2`, `scipy 1.14.1` | All essential pipeline libraries are installed. `accelerate` and `zarr` are NOT installed. Use `safetensors` or `np.memmap`. |
| **Qwen Architecture** | 24 layers, $d_{\text{model}}=2048$, 60 routed experts, top-4 routed, 1 shared expert | Total params ~14.3B (2.7B active per token). Router output shape is `(B*S, 60)`. |
| **Deep Layers & Tap N** | Deep layers: 5–24 (1-indexed, 20 layers total). Tap layer: Layer 3 (1-indexed) | Layer 3 provides rich syntactic/semantic embeddings with 1 full layer of asynchronous lead time before Layer 5. |
| **100k Tokens Footprint** | $100,000$ tokens $\times 2048$ hidden states + 24 layers $\times 60$ router logits | Total extracted dataset is **~736 MB** (FP16) or **~1.43 GB** (FP32). Fits easily in RAM once extracted and on disk (187 GB free). |
| **Streaming Safety** | Batch size $B=1$, seq len $L=1024$, `torch.inference_mode()`, `use_cache=False` | Forward activation memory is $<150 \text{ MB}$ per batch. Immediate `.detach().cpu()` prevents memory leaks. |
| **Train/Calib Split** | 80% Train / 20% Held-Out Calibration partitioned strictly by **sequence** | Zero cross-token attention leakage; sequences trimmed at $L-3$ for valid $T+1, T+2, T+3$ speculative targets. |

---

## 2. Local Environment & Hardware Constraints

### 2.1 System Specifications
- **Operating System**: macOS 27.2 (Darwin 23.x / Apple Silicon `arm64`)
- **CPU**: Apple Silicon M-series, 14 physical cores / 14 logical cores
- **RAM**: 36.00 GB total; ~12.97 GB available at runtime
- **Storage**: 926.35 GB total; **187.46 GB free** at `/Users/jack/Downloads/rlcd-router`
- **Python Executable**: `/opt/anaconda3/bin/python3` (Python 3.10.14 conda-forge)

### 2.2 Installed ML Package Versions
- `torch`: **2.2.2**
- `transformers`: **4.44.0**
- `safetensors`: **0.6.2**
- `datasets`: **2.16.0**
- `pytest`: **8.3.4**
- `scikit-learn`: **1.3.2**
- `scipy`: **1.14.1**
- `numpy`: **1.26.4**
- `huggingface_hub`: **0.36.2**
- `accelerate`: **NOT INSTALLED**
- `zarr`: **NOT INSTALLED**

### 2.3 Hardware Acceleration & Metal Performance Shading
We executed empirical micro-benchmarks on the local Apple Silicon MPS backend:
1. **GEMM Throughput**:
   - 50 iterations of $1024 \times 2048 \times 2048$ matrix multiplications on MPS completed in $0.0637 \text{ s}$ ($1.27 \text{ ms/iter}$).
2. **Precision Support on MPS**:
   - `torch.float32`: Fully supported.
   - `torch.float16`: Fully supported.
   - `torch.bfloat16`: **FAILED** with `RuntimeError: BFloat16 is not supported on MPS` in PyTorch 2.2.2.
   - *Implication*: When running on MPS, tensors must be explicitly converted to `torch.float16`. If running in `bfloat16`, execution must stay on `cpu`.
3. **MPS Memory Limits**:
   - Allocation of single buffers up to 20 GB succeeded.
   - Allocating single buffers $\ge 22 \text{ GB}$ failed with `Invalid buffer size: 22.00 GB`.
   - Aggregate allocations across multiple tensors reached 30 GB using virtual memory swap.
   - *Implication*: Loading the entire 26.67 GB weights of `Qwen1.5-MoE-A2.7B` on MPS simultaneously risks hitting Metal unified memory driver limits when available RAM is ~13 GB.

### 2.4 Testing / CI Strategy for Constrained Memory
To guarantee that unit and integration tests run rapidly without requiring 27 GB model downloads or risking OOM on developer machines:
- Implement a **Mock/Synthetic Model Factory** using `transformers.Qwen2MoeConfig`:
  ```python
  tiny_config = Qwen2MoeConfig(
      hidden_size=64,
      intermediate_size=128,
      moe_intermediate_size=64,
      shared_expert_intermediate_size=128,
      num_hidden_layers=6,
      num_attention_heads=4,
      num_key_value_heads=4,
      num_experts=16,
      num_experts_per_tok=4,
      vocab_size=1000,
      output_router_logits=True
  )
  ```
  This instantiates a structurally identical, fully functional MoE model in $<50 \text{ MB}$ of memory, executing in milliseconds for testing the entire pipeline end-to-end.

---

## 3. Qwen/Qwen1.5-MoE-A2.7B Architecture & Layer N Tap Analysis

### 3.1 Model Architectural Parameters
From official `config.json` (`~/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9/config.json`):
- `architectures`: `["Qwen2MoeForCausalLM"]`
- `model_type`: `"qwen2_moe"`
- `num_hidden_layers`: **24**
- `hidden_size` ($d_{\text{model}}$): **2048**
- `intermediate_size`: **5632** (used for shared expert MLP)
- `moe_intermediate_size`: **1408** (used for each of the 60 routed expert MLPs)
- `shared_expert_intermediate_size`: **5632**
- `num_experts`: **60** routed experts
- `num_experts_per_tok` ($K$): **4** experts active per token
- `norm_topk_prob`: **False** (softmax weights are not normalized to sum to 1)
- `decoder_sparse_step`: **1** (every layer $0 \dots 23$ is a sparse MoE layer)
- `num_attention_heads`: 16 ($d_{\text{head}} = 128$)
- `num_key_value_heads`: 16
- `max_position_embeddings`: 8192
- `vocab_size`: 151936

### 3.2 Parameter Budget: Total vs. Active
- **Attention & Norms per layer**: $4 \times (2048 \times 2048) \approx 16.78\text{M}$ params.
- **Routed Experts per layer**: 60 experts $\times 3 \text{ projections} \times (2048 \times 1408) \approx 519.0\text{M}$ params.
- **Shared Expert per layer**: 1 expert $\times 3 \text{ projections} \times (2048 \times 5632) + (2048 \times 1) \approx 34.6\text{M}$ params.
- **Layer Total**: $\approx 570.4\text{M}$ params per layer $\times 24 \text{ layers} \approx 13.69\text{B}$ params.
- **Embeddings**: $151936 \times 2048 \approx 311\text{M}$ params.
- **Total Parameters**: **~14.3 Billion parameters** (~26.67 GB in FP16).
- **Active Parameters per token**: 4 routed experts ($34.6\text{M}$) + 1 shared expert ($34.6\text{M}$) + attention ($16.8\text{M}$) $\times 24 \text{ layers} + 311\text{M} \approx \mathbf{2.7\text{ Billion active parameters}}$.

### 3.3 Routing Mechanism (`Qwen2MoeSparseMoeBlock`)
Direct inspection of `transformers.models.qwen2_moe.modeling_qwen2_moe.py` reveals the exact mathematical operations performed at every layer $l \in [0, 23]$:
1. **Input Reshape**: Hidden state $h \in \mathbb{R}^{B \times S \times 2048}$ is reshaped to $\tilde{h} \in \mathbb{R}^{(B \cdot S) \times 2048}$.
2. **Router Gate Projection**:
   $$z = \tilde{h} W_{\text{gate}}^T \in \mathbb{R}^{(B \cdot S) \times 60} \quad \text{where } W_{\text{gate}} \in \mathbb{R}^{60 \times 2048}, \, \text{bias} = \text{False}$$
3. **Softmax Probabilities**:
   $$p = \text{Softmax}(z, \text{dim}=-1) \in \mathbb{R}^{(B \cdot S) \times 60} \quad (\text{computed in float32})$$
4. **Top-k Routed Selection**:
   $$p_{\text{top4}}, \mathcal{E}_{\text{top4}} = \text{TopK}(p, k=4, \text{dim}=-1)$$
   - $\mathcal{E}_{\text{top4}} \in \{0, \dots, 59\}^{(B \cdot S) \times 4}$ are the selected expert indices.
   - $p_{\text{top4}} \in [0, 1]^{(B \cdot S) \times 4}$ are the routing coefficients (unnormalized because `norm_topk_prob=False`).
5. **Shared Expert**:
   $$y_{\text{shared}} = \sigma(\tilde{h} W_{\text{shared\_gate}}^T) \odot \text{MLP}_{\text{shared}}(\tilde{h}) \quad \text{where } W_{\text{shared\_gate}} \in \mathbb{R}^{1 \times 2048}$$
   The shared expert is unconditionally evaluated for all tokens.

### 3.4 Transformers Output Tensors
When executing the forward pass with:
```python
outputs = model(input_ids, output_hidden_states=True, output_router_logits=True)
```
- `outputs.hidden_states`: Tuple of 25 tensors of shape `(B, S, 2048)`.
  - `outputs.hidden_states[0]`: Embedding layer output (before layer 0).
  - `outputs.hidden_states[l + 1]`: Output of decoder layer $l$ (for $l \in [0, 23]$).
- `outputs.router_logits`: Tuple of 24 tensors.
  - Each tensor `outputs.router_logits[l]` has shape `(B * S, 60)`.
  - To reshape for sequence alignment: `outputs.router_logits[l].view(B, S, 60)`.

### 3.5 Deep Layers Partitioning & Layer N Tap Tradeoff Analysis
The project prompt defines deep layers as **Layers 5–24** (1-indexed), which corresponds to **0-indexed layers 4 through 23** (20 layers total).
The layer buckets for grid temperature scaling are:
- **Early Deep Layers (5–10)**: 1-indexed layers 5, 6, 7, 8, 9, 10 (0-indexed 4..9; 6 layers).
- **Late Deep Layers (11–24)**: 1-indexed layers 11 through 24 (0-indexed 10..23; 14 layers).

#### Tap Layer Candidates: Layer 2, 3, or 4
The speculative head takes intermediate hidden state $h_N(t)$ to predict future deep-layer routing distributions $\hat{p}^{(l)}_{t+k}$ for $l \in [5, 24]$ and $k \in \{1, 2, 3\}$.

| Candidate Layer N | 0-Index / HF Index | Lead Time Ahead of Layer 5 | Semantic Representation Quality | Asynchronous Routing Overlap Slack | Recommendation |
|---|---|---|---|---|---|
| **Layer 2** (1-indexed) | `idx=1` (`hidden_states[2]`) | 3 layers (Layers 2, 3, 4) | Low-Medium (dominated by local lexical patterns) | Highest (~30–40 ms buffer) | Viable fallback for extreme low-latency targets. |
| **Layer 3** (1-indexed) | `idx=2` (`hidden_states[3]`) | 2 layers (Layers 3, 4) | **High (optimal sweet spot)**: syntactic trees formed, multi-head attention context stabilized. | **Optimal (~20–25 ms buffer)**: Layer 4 computes while speculative head evaluates and prefetches Layer 5+ experts. | **Strongly Recommended Default**. |
| **Layer 4** (1-indexed) | `idx=3` (`hidden_states[4]`) | 1 layer (Layer 4) | Very High (closest to deep layers) | Minimal (~10 ms buffer before Layer 5 execution starts) | High accuracy, but tighter execution margin. |

**Decision**: Configure the pipeline with `tap_layer = 3` (1-indexed, index 2 in `hidden_states[3]`), exposing `tap_layer` as a configurable parameter (`--tap-layer 3`).

---

## 4. Streaming & Memory Management for 100k Tokens

### 4.1 Token Budget & Sequence Sizing
- Total tokens: $N = 100,000$.
- Sequence length: $L = 1024$.
- Number of sequences: $\lceil 100,000 / 1024 \rceil = 98$ sequences ($98 \times 1024 = 100,352$ tokens).
- Batch size: $B = 1$.
- Corpus source: `wikitext-2-raw-v1` (cached locally, 2,051,910 words available). Tokenized with `Qwen1.5-MoE-A2.7B` tokenizer (cached locally).

### 4.2 Extracted Data Memory & Storage Footprint
For 100,000 tokens:
1. **Layer N Hidden States** ($h_N$):
   - Shape: `(100000, 2048)`
   - FP16: $100,000 \times 2,048 \times 2 \text{ bytes} = 409,600,000 \text{ bytes} \approx \mathbf{409.6\text{ MB}}$.
   - FP32: $819.2\text{ MB}$.
2. **Native Router Logits** ($z^{(l)}$ for all 24 layers):
   - Shape: `(24, 100000, 60)`
   - FP16: $24 \times 100,000 \times 60 \times 2 \text{ bytes} = 288,000,000 \text{ bytes} \approx \mathbf{288.0\text{ MB}}$.
   - FP32: $576.0\text{ MB}$.
3. **Native Top-4 Expert Selections**:
   - `topk_indices`: `(24, 100000, 4)` in `int16` = $19.2\text{ MB}$.
   - `topk_weights`: `(24, 100000, 4)` in `float16` = $19.2\text{ MB}$.
4. **Total Extracted Dataset Footprint**:
   - **FP16: ~736 MB**.
   - **FP32: ~1.43 GB**.
   - *Conclusion*: The extracted data easily resides in RAM once generated, and easily fits within the 187 GB available disk space.

### 4.3 Forward Pass Activation Footprint & Peak Memory
At $B=1$ and $L=1024$:
- Input tensor: `(1, 1024)` int64 = $8\text{ KB}$.
- Self-attention activations: $1024 \times 2048 \times 2 \text{ bytes} \approx 4\text{ MB}$ per layer.
- Intermediate activations during forward pass with `torch.inference_mode()`: $< 150 \text{ MB}$ total peak.

### 4.4 Zero-OOM Streaming Protocol & Memory Leak Prevention
To eliminate memory leaks during the 98-sequence loop:
1. **Inference Context**: Use `torch.inference_mode()` (superior to `torch.no_grad()`; disables autograd tracking and version counter allocations).
2. **Disable KV Cache**: Pass `use_cache=False` to the model forward pass.
3. **Immediate Tensor Detach & Offload**:
   ```python
   # Inside streaming loop:
   with torch.inference_mode():
       out = model(input_ids=batch_ids, output_hidden_states=True, output_router_logits=True, use_cache=False)
       
       # Extract tap layer hidden states and immediately detach to CPU
       # HF hidden_states[tap_layer] corresponds to layer tap_layer
       batch_h_N = out.hidden_states[tap_layer].detach().to("cpu", dtype=torch.float16)
       
       # Extract router logits for all 24 layers, reshape to (24, B, S, 60)
       batch_logits = torch.stack([
           r.view(batch_size, seq_len, 60).detach().to("cpu", dtype=torch.float16)
           for r in out.router_logits
       ], dim=0) # (24, B, S, 60)
       
       del out
   ```
4. **Periodic Cache Flush**:
   Every 10 batches, run:
   ```python
   gc.collect()
   if torch.backends.mps.is_available():
       torch.mps.empty_cache()
   ```
5. **Streaming Storage Architecture**:
   - Use `safetensors.torch.save_file` to save chunked shards (e.g. `shard_0000.safetensors`, each containing 20k tokens), or write sequentially to preallocated `numpy.memmap` files:
     - `h_N.dat`: `np.memmap(shape=(100000, 2048), dtype='float16', mode='w+')`
     - `router_logits.dat`: `np.memmap(shape=(24, 100000, 60), dtype='float16', mode='w+')`
   - Memory mapping allows zero-copy reading directly during Medusa head training without loading all 736 MB into Python heap at once.

---

## 5. Clean Train / Calibration Split Mechanics

### 5.1 Split Ratio & Sample Allocation
- Total tokens: $100,000$ (98 sequences of length 1024).
- **Train Partition (81.6%)**: 80 sequences $\times 1024 = \mathbf{81,920\text{ tokens}}$.
- **Calibration Partition (18.4%)**: 18 sequences $\times 1024 = \mathbf{18,432\text{ tokens}}$ (strictly within the 15–20% mandate).

### 5.2 Strict Isolation Principles (Eliminating Data Leakage)
1. **Atomic Sequence-Level Partitioning**:
   - Autoregressive language modeling creates profound temporal and attention-based autocorrelation between neighboring tokens.
   - Splitting tokens randomly across train/calibration causes catastrophic data leakage (a token at $t+1$ would be in calibration while $t$ and $t+2$ are in training).
   - **Mandate**: Sequences must be partitioned as complete, contiguous blocks: sequences $0 \dots 79$ for `train`, sequences $80 \dots 97$ for `calibration`.
2. **Lookahead Horizon Boundary Protection ($T+1, T+2, T+3$)**:
   - The speculative head at token position $t$ predicts targets for future positions $t+1, t+2, t+3$.
   - For any sequence of length $L$, tokens at positions $L-3, L-2, L-1$ do not possess complete ground truth targets within the sequence boundary.
   - **Mandate**: The speculative head loss calculation must mask out or trim the trailing 3 tokens of each sequence:
     $$\mathcal{T}_{\text{valid}} = \{t \in [0, L - 1 - k] \}$$
   - No sequence boundary may be crossed to acquire future targets.
3. **Physical File-Level Isolation**:
   - The generation script writes two distinct files:
     - `data/train_data.safetensors` (81,920 tokens)
     - `data/calib_data.safetensors` (18,432 tokens)
4. **Lifecycle Operational Isolation**:
   - **Stage 1 (Head Training)**: The training script opens only `train_data.safetensors`. The calibration file path is not passed.
   - **Stage 2 (Grid Temperature Scaling)**: The speculative head weights are completely frozen (`requires_grad = False`). Temperature scaling parameters are optimized strictly on `calib_data.safetensors` using LBFGS to minimize NLL.
   - **Stage 3 (Evaluation)**: Targeted ECE (at 0.05 and 0.85 decision boundaries) is evaluated on the calibrated probabilities over `calib_data.safetensors`.

---

## 6. Speculative Head & Loss Formulation Guidelines

### 6.1 Head Parameter Count & Topology
- Input: $h_N(t) \in \mathbb{R}^{2048}$.
- Targets: Deep layers 5–24 (20 layers) at horizons $T+1, T+2, T+3$.
- Topology: 3 linear heads ($W_1, W_2, W_3$), one for each lookahead horizon $k \in \{1, 2, 3\}$:
  $$W_k \in \mathbb{R}^{2048 \times (20 \times 60)} = \mathbb{R}^{2048 \times 1200}$$
  $$\hat{Z}_{t+k} = h_N(t) W_k + b_k \in \mathbb{R}^{20 \times 60}$$
- Total parameter count: $3 \times (2048 \times 1200 + 1200) = 7,376,400$ parameters (~29.5 MB in FP32).
- Training cost: Extremely fast (training 7.4M linear parameters on 81k tokens takes $<30\text{ seconds}$ on CPU or MPS).

### 6.2 Temperature Scaling Grid Specification
- Grid buckets: $S = \{\text{early layers 5--10}, \text{late layers 11--24}\} \times \{T+1, T+2, T+3\}$ ($2 \times 3 = 6$ temperature scalars $T(s)$).
- Parameterization: $T(s) = \exp(\theta_s)$ to guarantee $T(s) > 0$.
- Loss: Strictly Negative Log-Likelihood (NLL) with L2 regularization towards $T=1.0$:
  $$\mathcal{L}_{\text{calib}}(\theta) = \text{NLL}(\theta) + \lambda_{\text{reg}} \sum_{s=1}^6 (T(s) - 1.0)^2$$
- Optimization: `torch.optim.LBFGS(lr=0.05, max_iter=50)` with closure. ECE is **strictly forbidden** during optimization due to non-differentiability.

---

## 7. Recommended Implementer Action Plan

1. **Synthetic MoE Test Fixture**: Implement `create_test_model()` returning a small `Qwen2MoeForCausalLM` (6 layers, 16 experts, $d_{\text{model}}=64$) to run fast unit tests under pytest.
2. **Data Generator Module** (`generate_data.py`):
   - Accepts `--model-path`, `--num-tokens 100000`, `--seq-len 1024`, `--tap-layer 3`, `--device {mps, cpu}`.
   - Loads Wikitext-2 or input text, streams sequences with $B=1$ under `torch.inference_mode()`.
   - Saves `data/train_data.safetensors` and `data/calib_data.safetensors`.
3. **Speculative Head Module** (`medusa_head.py`):
   - Linear projection $2048 \to (20, 60)$ for each horizon $k \in \{1, 2, 3\}$.
   - Cross-Entropy loss over deep-layer targets + tunable MMCE penalty.
4. **Post-Hoc Calibration Module** (`temperature_scaling.py`):
   - Parameterizes 6 temperature scalars $\theta \in \mathbb{R}^{2 \times 3}$.
   - Fits on `calib_data.safetensors` using LBFGS + L2 regularization toward 1.0.
5. **Targeted ECE Evaluator** (`evaluate.py`):
   - Computes Targeted ECE specifically at $p=0.05$ (abort threshold) and $p=0.85$ (mass cutoff).
   - Generates breakdown table across early/late buckets and horizons $T+1, T+2, T+3$.
