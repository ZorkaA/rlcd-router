# Project: Phase 1 PyTorch ML Calibration Scripts for Asynchronous MoE Router

## Architecture
The system implements a high-throughput, memory-efficient calibration and speculative routing pipeline for `Qwen/Qwen1.5-MoE-A2.7B` on Apple Silicon (MPS) and CPU. It predicts deep-layer routing distributions (Layers 5–24) asynchronously from an intermediate layer (default Layer 3) across multiple lookahead horizons ($T+1, T+2, T+3$), performs post-hoc temperature scaling via regularized NLL optimization, and evaluates Targeted Gating calibration metrics.

```
                  +----------------------------------------------+
                  |               100k-Token Corpus              |
                  +----------------------------------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |  Data Generation & Sequence-Level Split     |
                  |  - Stream Qwen1.5-MoE-A2.7B (Zero-OOM)       |
                  |  - Extract h_N (Layer 3) & Native Router Logits |
                  |  - Split: 80% Train / 20% Held-Out Calib     |
                  +----------------------------------------------+
                                /                  \
                               /                    \
                              v                      v
                +-------------------------+  +-------------------------------+
                | Train Split (80%)       |  | Held-Out Calib Split (20%)    |
                +-------------------------+  +-------------------------------+
                              |                              |
                              v                              |
                +-------------------------+                  |
                | Medusa Speculative Head |                  |
                | - 3 Linear Heads        |                  |
                | - CE Loss + λ·MMCE      |                  |
                +-------------------------+                  |
                              |                              |
                              +--------------+---------------+
                                             |
                                             v
                             +-------------------------------+
                             | Grid-Based Temperature Scaling|
                             | - {early, late} x {T+1..T+3}  |
                             | - LBFGS NLL Min + L2 Reg      |
                             | - Strict ECE-Free Optimization|
                             +-------------------------------+
                                             |
                                             v
                             +-------------------------------+
                             | Targeted Gating Evaluation    |
                             | - T-ECE @ 0.05 (Abort)        |
                             | - T-ECE @ 0.85 (Mass Cutoff)  |
                             | - Memory Leak Verification    |
                             +-------------------------------+
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Model Loading & MPS/CPU Support | Load `Qwen/Qwen1.5-MoE-A2.7B` with FP16 on MPS or CPU, with synthetic config fixture for rapid testing | M1 | Survey R1 |
| 2 | Zero-OOM Streaming Generator | Stream 100k tokens in $B=1, L=1024$ chunks with `torch.inference_mode()`, immediate CPU detach, and memory flushing | M1 | Survey R1 |
| 3 | Hidden State & Router Logits Extraction | Capture Layer N (Layer 3, $d=2048$) hidden states and native router logits ($24 \times 60$) across all layers | M1 | Survey R1 |
| 4 | Strictly Isolated Train/Calib Split | Sequence-atomic 80/20 partitioning with trailing token masking ($L-3..L-1$) for horizon isolation | M1 | Survey R1 |
| 5 | Dataset Persistence & Sharding | Persist train and held-out calibration datasets efficiently using `safetensors` format | M1 | Survey R1 |
| 6 | Medusa Linear Speculative Head | Attach 3 linear projection heads ($2048 \to 1200$) predicting Layers 5–24 router logits for $T+1, T+2, T+3$ | M2 | Survey R2 |
| 7 | Multi-Horizon Cross-Entropy Loss | Soft cross-entropy against native router distributions across all 20 deep layers and 3 horizons | M2 | Survey R2 |
| 8 | Tunable RKHS MMCE Penalty | Maximum Mean Calibration Error penalty with Gaussian RBF kernel ($\sigma=0.2$) and inverse class weighting | M2 | Survey R2 |
| 9 | Speculative Head Training Loop | End-to-end training loop with loss logging (CE, MMCE, total) and checkpoint saving | M2 | Survey R2 |
| 10 | 2x3 Temperature Scaling Grid | Temperature grid container indexed by {early 5-10, late 11-24} x {T+1, T+2, T+3} (6 scalar parameters) | M3 | Survey R3 |
| 11 | LBFGS NLL Minimization | Optimize temperature scalars strictly minimizing NLL via `torch.optim.LBFGS` with strong Wolfe line search | M3 | Survey R3 |
| 12 | L2 Regularization toward T=1.0 | Regularize temperature scalars with $\alpha \cdot (T - 1.0)^2$ to prevent extreme temperatures in sparse buckets | M3 | Survey R3 |
| 13 | Calibrated Probability Inference | Apply fitted temperature grid to uncalibrated speculative logits to output calibrated router probabilities | M3 | Survey R3 |
| 14 | Targeted ECE @ 0.05 (Abort) | Window-based and kernel-based Targeted ECE around 0.05 decision boundary for speculative abort | M4 | Survey R4 |
| 15 | Targeted ECE @ 0.85 (Mass Cutoff) | Cumulative mass cutoff calibration error and native top-4 expert recall evaluation | M4 | Survey R4 |
| 16 | Memory Leak & System Profiling | Tracemalloc and psutil RSS tracking asserting 0 tensor delta and bounded memory usage | M4 | Survey R4 |
| 17 | Integrated Pipeline CLI | End-to-end execution pipeline running data generation, training, calibration, and evaluation | M4 | Survey R4 |
| 18 | E2E Opaque-Box Test Suite | Comprehensive 4-tier test suite verifying full pipeline integration, edge cases, and acceptance criteria | M_Final | Survey AC |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Data Partitioning & Generation | Model loader, zero-OOM 100k streaming, Layer N extraction, isolated 80/20 train/calib dataset creation (Features 1–5) | none | PLANNED |
| M2 | Speculative Head & Training with MMCE | Medusa linear heads (T+1..T+3), soft CE loss, RKHS MMCE penalty, training pipeline & checkpoints (Features 6–9) | M1 | PLANNED |
| M3 | Grid-Based Temperature Scaling | 2x3 temperature grid, LBFGS NLL minimization with L2 regularizer, calibrated inference (Features 10–13) | M1, M2 | PLANNED |
| M4 | Targeted Gating & Pipeline Integration | Targeted ECE at 0.05 & 0.85, memory leak profiler, end-to-end calibration CLI runner (Features 14–17) | M1, M2, M3 | PLANNED |
| M_Final | E2E Acceptance & Adversarial Hardening | Pass 100% of E2E Test Suite (Tiers 1–4) published by E2E Testing Track, followed by Tier 5 Adversarial Hardening (Feature 18) | M1, M2, M3, M4, TEST_READY | PLANNED |

## Interface Contracts

### M1 ↔ M2: Calibration & Training Data Contract
- Data format: `train_data.safetensors` and `calib_data.safetensors`
- Tensors:
  - `hidden_states`: FloatTensor of shape `(num_samples, 2048)` (FP16/FP32), representing Layer 3 outputs.
  - `target_router_logits`: FloatTensor of shape `(num_samples, 3, 20, 60)`, where:
    - Dim 1: Horizon offset ($0 \to T+1, 1 \to T+2, 2 \to T+3$).
    - Dim 2: Deep layer index ($0 \to \text{Layer } 5, \dots, 19 \to \text{Layer } 24$).
    - Dim 3: Logits for 60 routed experts.
  - `target_top4_indices`: LongTensor of shape `(num_samples, 3, 20, 4)`.
  - `valid_mask`: BoolTensor of shape `(num_samples, 3)` indicating boundary validity for future horizons.

### M2 ↔ M3: Speculative Head Model Contract
- Module: `MedusaSpeculativeHead(nn.Module)`
- Forward signature: `forward(hidden_states: torch.Tensor) -> torch.Tensor`
  - Input: `(batch_size, 2048)` or `(batch_size, seq_len, 2048)`
  - Output: `(batch_size, seq_len, 3, 20, 60)` unscaled speculative logits
- Checkpoint: `checkpoints/speculative_head.pt` (state_dict + config)

### M3 ↔ M4: Temperature Grid Scaling Contract
- Module: `TemperatureGrid(nn.Module)`
- Parameter grid: `temperatures = nn.Parameter(torch.ones(2, 3))`
  - Dim 0: Layer bucket (0: Early Layers 5–10, 1: Late Layers 11–24)
  - Dim 1: Lookahead horizon (0: $T+1$, 1: $T+2$, 2: $T+3$)
- Scaling function: `scale_logits(logits: torch.Tensor) -> torch.Tensor`
  - Input: `(N, 3, 20, 60)` speculative logits
  - Output: `(N, 3, 20, 60)` temperature-scaled logits
- Serialization: `checkpoints/temperature_grid.json` or `.pt`

### M4: Targeted Gating Metrics Contract
- Function: `evaluate_targeted_ece(calibrated_probs: torch.Tensor, ground_truth_top4: torch.Tensor, thresholds: list = [0.05, 0.85]) -> dict`
- Output Dictionary:
  ```python
  {
      "targeted_ece_0.05": {
          ("early", "T+1"): float,
          ("early", "T+2"): float,
          ...
          ("late", "T+3"): float,
      },
      "targeted_ece_0.85": { ... },
      "cumulative_mass_0.85_stats": { ... },
      "overall_ece": float,
  }
  ```

## Code Layout
```
/Users/jack/Downloads/rlcd-router/
├── src/
│   ├── __init__.py
│   ├── config.py                  # Global hyperparameters, paths, layer definitions
│   ├── data/                      # Milestone 1 ownership
│   │   ├── __init__.py
│   │   ├── model_loader.py        # Qwen1.5-MoE loader & synthetic test fixture
│   │   ├── stream_extractor.py    # Zero-OOM streaming & activation harvesting
│   │   └── dataset.py             # Sequence-level splitting & safetensors I/O
│   ├── models/                    # Milestone 2 ownership
│   │   ├── __init__.py
│   │   └── medusa_head.py         # 3-horizon linear speculative projection head
│   ├── training/                  # Milestone 2 ownership
│   │   ├── __init__.py
│   │   ├── mmce_loss.py           # Differentiable RKHS MMCE penalty & soft CE
│   │   └── trainer.py             # Head training loop & loss curve logger
│   ├── calibration/               # Milestone 3 ownership
│   │   ├── __init__.py
│   │   ├── grid.py                # 2x3 temperature grid module
│   │   └── lbfgs_optimizer.py     # LBFGS NLL optimizer with L2 regularization
│   ├── evaluation/                # Milestone 4 ownership
│   │   ├── __init__.py
│   │   ├── targeted_ece.py        # 0.05 abort & 0.85 mass cutoff metrics
│   │   └── memory_profiler.py     # Tracemalloc & psutil leak detection
│   └── pipeline.py                # Milestone 4 ownership: End-to-end runner CLI
├── tests/                         # E2E Testing Track ownership (opaque-box)
│   ├── __init__.py
│   ├── conftest.py                # Shared fixtures (synthetic model, mock data)
│   ├── test_tier1_features.py     # Feature coverage (>=5 per feature)
│   ├── test_tier2_boundaries.py   # Boundary & corner cases
│   ├── test_tier3_pairwise.py     # Cross-feature combinations
│   └── test_tier4_workloads.py    # Real-world end-to-end application scenarios
├── scripts/
│   └── run_calibration.py         # Entry-point runner script
├── ORIGINAL_REQUEST.md            # Authoritative user requirements
├── PROJECT.md                     # Project architecture, inventory, milestones, contracts
├── TEST_INFRA.md                  # E2E Test Track index & test architecture
└── TEST_READY.md                  # E2E Test Track signal when ready
```
