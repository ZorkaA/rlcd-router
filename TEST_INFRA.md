# Test Infrastructure Specification: Asynchronous MoE Router Phase 1 Calibration

## 1. Test Philosophy & Principles

The testing infrastructure for the Asynchronous MoE Router Phase 1 Calibration project is engineered according to rigorous **opaque-box (black-box)**, specification-driven principles.

1. **Opaque-Box Requirement Verification**: Tests treat system modules through their formal public interfaces and contracts defined in `PROJECT.md`. Tests never patch private internals, rely on private implementation details, or bypass interface contracts.
2. **Deterministic & Self-Contained Execution**: Every test manages its own state and cleanups. Tests execute independently of execution order and are fully reproducible across local machines, Apple Silicon (MPS), CPU, and CI/CD pipelines.
3. **Progressive Testability**: Testing operates seamlessly across milestone development. In early milestones before `src/` modules are complete, tests run against fast synthetic fixtures or skip gracefully with explicit skip diagnostics. Once implementation modules land, tests automatically bind to them and enforce contract correctness.
4. **Fast Offline Execution**: Testing does not require downloading 28 GB of production weights over the network. A lightweight synthetic `Qwen2MoeConfig` fixture ($d=64, L=6, E=16, k=4$, footprint $<50\text{ MB}$) executes forward passes, activation extractions, and training loops in milliseconds.
5. **No Facade Tests**: Every test asserts real mathematical properties, tensor shapes, gradient flows, calibration invariants, and error boundaries.

---

## 2. Test Methodology

The test suite is structured across four orthogonal testing tiers derived from software quality engineering disciplines:

### Tier 1: Category-Partition Feature Testing (`test_tier1_features.py`)
- Each of the **18 inventoried features** in `PROJECT.md` is tested across its primary functional categories (happy path, parameter variations, contract types, output shapes, and expected error handling).
- **Threshold**: Minimum of 5 comprehensive test cases per feature (total $\ge 90$ feature test cases).

### Tier 2: Boundary Value Analysis & Corner Conditions (`test_tier2_boundaries.py`)
- Analyzes extremes, discontinuity points, and edge boundaries:
  - Horizon truncation at sequence tails ($T \ge L-3$) with `valid_mask` propagation.
  - Zero/sparse sample buckets in calibration grids where $N_{b, h} \to 0$.
  - Extreme and boundary temperature scaling values ($T \to 0$, $T \to \infty$, negative values).
  - Single-sample batches ($m=1$) and identical confidences in RKHS MMCE kernel calculation ($\mathbf{e}^T \mathbf{K} \mathbf{e} \to 0$ with $\epsilon=10^{-8}$ numerical stability).
  - Empty bins in Targeted ECE windows ($|B_{p^*}| = 0$).
  - FP16/BFloat16 underflow/overflow prevention.
  - Empty datasets, single token streams, and invalid layer indices.
- **Threshold**: Minimum of 5 tests per feature boundary area.

### Tier 3: Pairwise Combinatorial Interaction Testing (`test_tier3_pairwise.py`)
- Systematically covers interactions between orthogonal dimensions:
  - Layer buckets ({early: Layers 5–10, late: Layers 11–24}) $\times$ Horizons ($T+1, T+2, T+3$).
  - Loss components: Cross-Entropy alone ($\lambda=0$) vs. CE + MMCE ($\lambda \in \{0.1, 0.5, 1.0, 5.0\}$) across varying kernel bandwidths ($\sigma \in \{0.1, 0.2, 0.5\}$).
  - Loss targets: Hard top-1 expert targets vs. Soft routing distributions.
  - Optimization regularizers: NLL with no L2 ($\alpha=0$) vs. NLL with L2 ($\alpha \in \{0.001, 0.01, 0.1\}$) across dense vs. sparse data splits.
  - Execution targets: CPU vs. MPS device placement.
  - Data precisions: Float32 vs. Float16/BFloat16 inputs and gradient flows.

### Tier 4: Real-World Workloads & Invariant Stress Testing (`test_tier4_workloads.py`)
- Emulates production pipelines end-to-end:
  - Full pipeline workflow: Corpus streaming $\to$ Hidden state extraction $\to$ Dataset sharding $\to$ Speculative head training $\to$ Post-hoc LBFGS temperature calibration $\to$ Targeted ECE evaluation report.
  - Memory leak verification: Tracemalloc and psutil RSS tracking across iterative streaming steps, asserting zero uncollected tensor leaks.
  - Targeted ECE threshold verification at $0.05$ (abort) and $0.85$ (mass cutoff) decision points, verifying pre- vs. post-calibration error reduction.
  - Serialization roundtrip: Checkpoint save and load fidelity for speculative head weights and temperature grid configurations.

---

## 3. Feature Inventory & Test Mapping

| # | Feature Name | Component Module | Primary Test File | Tier 1 Tests | Tier 2/3/4 Tests |
|---|--------------|------------------|-------------------|--------------|------------------|
| 1 | Model Loading & MPS/CPU Support | `src.data.model_loader` | `test_tier1_features.py` | `test_f01_*` (5 tests) | Device fallback, invalid dtype |
| 2 | Zero-OOM Streaming Generator | `src.data.stream_extractor` | `test_tier1_features.py` | `test_f02_*` (5 tests) | Chunk boundary, token truncation |
| 3 | Hidden State & Router Logits Extraction | `src.data.stream_extractor` | `test_tier1_features.py` | `test_f03_*` (5 tests) | Layer index out of bounds, tensor shapes |
| 4 | Strictly Isolated Train/Calib Split | `src.data.dataset` | `test_tier1_features.py` | `test_f04_*` (5 tests) | Split overlap 0%, ratio verification |
| 5 | Dataset Persistence & Sharding | `src.data.dataset` | `test_tier1_features.py` | `test_f05_*` (5 tests) | Safetensors schema, metadata integrity |
| 6 | Medusa Linear Speculative Head | `src.models.medusa_head` | `test_tier1_features.py` | `test_f06_*` (5 tests) | Projection dimensions, batched forward |
| 7 | Multi-Horizon Cross-Entropy Loss | `src.training.mmce_loss` | `test_tier1_features.py` | `test_f07_*` (5 tests) | Soft CE vs hard CE, masking |
| 8 | Tunable RKHS MMCE Penalty | `src.training.mmce_loss` | `test_tier1_features.py` | `test_f08_*` (5 tests) | Gaussian RBF kernel, gradient stability |
| 9 | Speculative Head Training Loop | `src.training.trainer` | `test_tier1_features.py` | `test_f09_*` (5 tests) | AdamW convergence, loss logging |
| 10 | 2x3 Temperature Scaling Grid | `src.calibration.grid` | `test_tier1_features.py` | `test_f10_*` (5 tests) | Bucket mapping, scalar bounds |
| 11 | LBFGS NLL Minimization | `src.calibration.lbfgs_optimizer` | `test_tier1_features.py` | `test_f11_*` (5 tests) | Wolfe line search, strict ECE exclusion |
| 12 | L2 Regularization toward T=1.0 | `src.calibration.lbfgs_optimizer` | `test_tier1_features.py` | `test_f12_*` (5 tests) | Shrinkage in sparse buckets, $\alpha$ scaling |
| 13 | Calibrated Probability Inference | `src.calibration.grid` | `test_tier1_features.py` | `test_f13_*` (5 tests) | Softmax temperature division, simplex sum=1 |
| 14 | Targeted ECE @ 0.05 (Abort) | `src.evaluation.targeted_ece` | `test_tier1_features.py` | `test_f14_*` (5 tests) | Window $\delta=0.025$, empty bin handling |
| 15 | Targeted ECE @ 0.85 (Mass Cutoff) | `src.evaluation.targeted_ece` | `test_tier1_features.py` | `test_f15_*` (5 tests) | Cumulative mass cutoff, top-4 recall |
| 16 | Memory Leak & System Profiling | `src.evaluation.memory_profiler` | `test_tier1_features.py` | `test_f16_*` (5 tests) | Tracemalloc & psutil RSS bounds |
| 17 | Integrated Pipeline CLI | `src.pipeline` | `test_tier1_features.py` | `test_f17_*` (5 tests) | CLI flag parsing, stage execution |
| 18 | E2E Opaque-Box Test Suite | `tests/` | `test_tier1_features.py` | `test_f18_*` (5 tests) | Self-verification, fixture availability |

---

## 4. Fast Synthetic Fixture Architecture

To achieve rapid, deterministic local execution without requiring the 28GB `Qwen/Qwen1.5-MoE-A2.7B` weight download, `tests/conftest.py` provides:

1. **`synthetic_qwen_config`**:
   - `hidden_size = 64`
   - `intermediate_size = 128`
   - `num_hidden_layers = 6`
   - `num_experts = 16`
   - `num_experts_per_tok = 4`
   - `vocab_size = 256`
   - `output_router_logits = True`
2. **`synthetic_qwen_model`**: Instantiated `Qwen2MoeForCausalLM(synthetic_qwen_config)` initialized on CPU or MPS.
3. **`synthetic_batch_contract`**: Conforms to M1 $\leftrightarrow$ M2 Interface Contract:
   - `hidden_states`: `(N, 2048)` float tensor
   - `target_router_logits`: `(N, 3, 20, 60)` float tensor
   - `target_top4_indices`: `(N, 3, 20, 4)` long tensor
   - `valid_mask`: `(N, 3)` boolean tensor
4. **`reference_oracles`**: Pure mathematical reference implementations of MMCE loss, NLL with L2 regularizer, temperature scaling, and Targeted ECE for exact numerical cross-validation.

---

## 5. Quality Thresholds & Acceptance Criteria

| Metric | Target / Threshold | Enforcement Mechanism |
|---|---|---|
| Feature Coverage | 100% of 18 features covered ($\ge 5$ tests per feature) | `test_tier1_features.py` |
| Boundary Verification | Trailing masks, sparse buckets, singular batches, extreme temperatures | `test_tier2_boundaries.py` |
| Combinatorial Verification | 2 layer buckets $\times$ 3 horizons $\times$ 4 MMCE lambdas | `test_tier3_pairwise.py` |
| Workload Verification | Full end-to-end calibration run with zero crashes | `test_tier4_workloads.py` |
| Memory Stability | RSS increase $\le 50\text{ MB}$ across 10 streaming cycles; zero uncollected tensor leaks | `test_tier4_workloads.py` |
| Strict Invariant | ECE metric MUST NEVER be included in LBFGS optimization objective | `test_tier1_features.py` & `test_tier2_boundaries.py` |
| Pass Rate | 100% pass on available modules; graceful skipping for unbuilt modules | `pytest -v tests/` |

---

## 6. How to Run the Tests

```bash
# Run the entire test suite with verbose output
pytest -v tests/

# Run specific tiers
pytest -v tests/test_tier1_features.py
pytest -v tests/test_tier2_boundaries.py
pytest -v tests/test_tier3_pairwise.py
pytest -v tests/test_tier4_workloads.py

# Run with test coverage report
pytest -v --cov=src tests/
```
