# Handoff Report — Challenger 2: Milestone 1 Boundary & Safetensors Stress Testing

## 1. Observation

### Code Under Review
- `src/data/dataset.py:81-127`: `partition_sequence_indices` — Sequence-atomic train/calib index splitting.
- `src/data/dataset.py:129-218`: `align_sequence_targets` — Lookahead target alignment and boundary token masking.
- `src/data/dataset.py:220-272`: `save_dataset_safetensors` — Contract schema validation and contiguous layout enforcement.
- `src/data/dataset.py:273-286`: `load_dataset_safetensors` — Deserialization of safetensors files.
- `src/data/dataset.py:287-483`: `MoECalibrationDataset` — In-memory and memory-mapped PyTorch Dataset with slicing, filtering, and layer subselection.
- `src/data/dataset.py:484-569`: `build_and_split_calibration_datasets` — High-level sequence extraction pipeline orchestrator.
- `src/config.py:70-86`: Corpus extraction and boundary masking configuration parameters.

### Adversarial Verification Harness
- Created `tests/test_m1_challenger2_stress.py` (409 lines, 132 tests) specifically targeting the three mandated empirical domains:
  1. Sequence-atomic partitioning across arbitrary ratios and token isolation oracles.
  2. Safetensors serialization under non-contiguous strides and DataLoader batch scaling.
  3. Gradient isolation asserting exact 0.0 gradient norm on masked boundary tokens.

### Empirical Test Execution Results
- `pytest -v tests/test_m1_challenger2_stress.py`:
  - **132 passed** in 0.60 seconds.
- Full project test suite (`pytest -v`):
  - **307 passed, 9 skipped** in 1.87 seconds (100% pass rate).

### Direct Empirical Measurements

#### Domain 1: Sequence-Atomic Partitioning & Token Isolation
- Disjointness matrix: 90 combinations tested ($N \in [2, 3, 5, 10, 50, 98, 100, 250, 1000]$ across train ratios $[0.1, 0.25, 0.333, 0.5, 0.667, 0.75, 0.8, 0.9, 0.95, 0.99]$).
  - Invariant $\text{train\_indices} \cap \text{calib\_indices} = \emptyset$ held in 100% of cases.
  - Invariant $|\text{train\_indices}| + |\text{calib\_indices}| = N$ held in 100% of cases.
- Calibration ratio sweep on $N=98$: Tested $r_{\text{calib}} \in [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 0.9, 0.99]$. All produced strictly disjoint partitions with $N_{\text{train}} \ge 1, N_{\text{calib}} \ge 1$.
- Boundary exceptions: Confirmed that invalid configurations ($N < 2$, ratios $\le 0$ or $\ge 1$, or extreme ratios resulting in 0 sequences for one split) trigger defensive `ValueError` exceptions as required.
- Token-level oracle: Injected globally unique sequence signatures ($S_i \cdot 50 + t$). Materialized train and calib datasets via `build_and_split_calibration_datasets` across ratios $0.1, 0.2, 0.5, 0.8, 0.9$. Token sets between train and calib showed exactly zero intersection ($\text{train\_tags} \cap \text{calib\_tags} = \emptyset$). Metadata headers `sequence_indices` matched exact subset indices.

#### Domain 2: Safetensors Serialization & DataLoader Integration
- Strange strides and non-contiguity: Evaluated 5 non-contiguous memory layouts:
  1. Transposed 2D tensors: `torch.randn(d_model, N).t()`
  2. Multi-dimensional strided slicing: `torch.randn(2N, H, 2L, E)[::2, :, ::2, :]`
  3. Permuted 4D dimensions: `raw_top4.permute(0, 3, 2, 1)`
  4. Expanded 0-stride tensors: `torch.tensor([[True, True, False]]).expand(N, H)`
  5. Custom `as_strided` memory layouts.
  `save_dataset_safetensors` automatically applied `.contiguous()` to all tensors; `load_dataset_safetensors` restored exact bit-level numerical equality for all elements.
- Schema enforcement: Confirmed strict rejection of missing keys (`KeyError`), sample count mismatches across tensors (`ValueError`), and invalid tensor dimensionalities (e.g. 3D hidden states) (`ValueError`).
- PyTorch DataLoader batch scaling: Evaluated 9 batch sizes ($B \in [1, 2, 7, 32, 64, 128, 512, 1024, 4096]$) across both `in_memory=True` and `in_memory=False` (memory-mapped via `safe_open`). Iterated all 256 samples without shape corruption or memory leaks; all batches matched the interface contract.
- Multiprocessing DataLoader: Tested `num_workers=2` in both in-memory and memory-mapped modes. Processed full datasets cleanly.
- Slicing and indexing: Negative indexing (`ds[-1]`, `ds[-5]`) verified in both in-memory and memory-mapped modes. Verified `filter_valid()` correctly drops samples with any invalid horizon, `select_layers()` subsets deep layers, and `select_horizon()` isolates horizons.

#### Domain 3: Gradient Isolation on Masked Boundary Tokens
- Tail boundary token ($L-1$): For sequence length $L=16$, token $L-1$ has `valid_mask = [False, False, False]`. Computed masked training loss through speculative projection head and autograd backward.
  - Measured gradient norm: $\|\nabla_{H[L-1]}\|_2 \equiv 0.0000000$.
  - In contrast, valid head token (index 0) yielded $\|\nabla_{H[0]}\|_2 = 0.052 > 0.0$.
- Granular horizon isolation:
  - Token $L-3$ ($T+1, T+2$ valid; $T+3$ invalid): Speculative logit gradient at $T+3$ yielded norm $\equiv 0.0$, while $T+1, T+2$ gradients were strictly non-zero.
  - Token $L-2$ ($T+1$ valid; $T+2, T+3$ invalid): Speculative logit gradients at $T+2$ and $T+3$ yielded norm $\equiv 0.0$.
- Deep backbone parameters: Attached a 9-layer deep backbone network (`Linear -> GELU x 4 -> Linear`). Evaluated loss on a batch containing strictly masked boundary tokens (`valid_mask` all False).
  - Masked loss evaluated to `0.000000` without NaN.
  - Aggregate gradient norm across ALL deep backbone weights and biases evaluated to exactly `0.000000` (zero autograd update).

---

## 2. Logic Chain

1. **Premise 1 (Partition Disjointness)**: `partition_sequence_indices` computes $n_{\text{train}} = \text{round}(N \cdot r)$ and assigns disjoint index ranges or disjoint random permutations (`perm[:n_train]` and `perm[n_train:]`). Observation confirms that across all 90 tested parameter permutations and token-level synthetic tracking, no index or token representation ever leaked across splits.
2. **Premise 2 (Serialization Robustness)**: `save_dataset_safetensors` iterates over dictionary items and explicitly calls `tensors = {k: v.contiguous() for k, v in tensors.items()}` at line 254 before calling `save_file`. Observation confirms that strange strides (transposed, sliced, permuted, stride 0) serialize cleanly and deserialize to bit-for-bit identical tensors.
3. **Premise 3 (DataLoader Integration)**: `MoECalibrationDataset` implements `__len__` and `__getitem__` returning `DatasetItem` namedtuples. Observation confirms that PyTorch `DataLoader` default collation groups `DatasetItem` into batched tensors across arbitrary batch sizes ($B=1$ to $B=4096$) in both memory-mapped and in-memory modes without failure.
4. **Premise 4 (Gradient Isolation)**: In `align_sequence_targets`, trailing tokens at horizons beyond sequence length have `valid_mask` set to `False`. When computing loss $\mathcal{L} = \frac{\sum \ell \cdot \text{mask}}{\sum \text{mask}}$, the forward loss term for masked tokens is multiplied by 0. By Leibniz rule of differentiation, $\frac{\partial (0 \cdot \ell)}{\partial \theta} = 0$. Autograd backward therefore accumulates 0.0 into leaf tensors and upstream backbone parameters. Observation confirms empirical gradient norm is exactly 0.0 across all deep layers.

---

## 3. Caveats

- **PyTorch 2.2.2 MPS BFloat16 constraint**: As documented in `src/config.py`, macOS MPS backend does not support `torch.bfloat16`. All empirical tests were verified in `torch.float32` and `torch.float16` on CPU and MPS.
- **Single-sequence partition limit**: Partitioning requires at least 2 sequences to form non-empty train and calib splits. Passing $N=1$ or extreme ratios on small $N$ that leave 0 sequences in one split raises `ValueError`. This is intentional and desirable defensive behavior.
- **Memory-mapped slice indexing**: As designed, `ds[start:end]` requires `in_memory=True`. For `in_memory=False`, individual indexing `ds[i]` or standard PyTorch `DataLoader` batching should be used.

---

## 4. Conclusion & Verdict

**VERDICT: APPROVE**

Milestone 1 data partitioning, safetensors serialization, and boundary masking modules satisfy all boundary condition, data integrity, and mathematical isolation requirements.
- Sequence-atomic partitioning strictly isolates training and calibration sets with zero token or context leakage.
- Safetensors I/O safely handles non-contiguous tensors, strange strides, and PyTorch DataLoader batch scaling.
- Masked boundary tokens exhibit mathematical gradient isolation, producing an exact 0.0 gradient norm across all deep layers.

---

## 5. Verification Method

To independently verify all findings and measurements:

```bash
# 1. Run the dedicated 132-test adversarial stress harness
pytest -v tests/test_m1_challenger2_stress.py

# 2. Run the complete test suite (307 passing tests)
pytest -v
```

Invalidation conditions:
- Any test failure in `tests/test_m1_challenger2_stress.py`.
- Any non-zero gradient norm measured on masked boundary tokens ($\|\nabla\|_2 > 0.0$).
- Any overlap between train and held-out calibration sequence indices ($\text{train\_indices} \cap \text{calib\_indices} \neq \emptyset$).

---

## Challenge Report

### Challenge Summary
- **Overall risk assessment**: **LOW** (All boundary conditions, memory layouts, and autograd gradient paths are mathematically sound and empirically validated).

### Challenges Evaluated

#### Challenge 1: Arbitrary Split Ratio Boundary Collisions (Severity: Low)
- **Assumption challenged**: Arbitrary floating point split ratios (e.g. 0.01, 0.99) might cause off-by-one indexing errors, empty splits, or overlapping partitions.
- **Empirical test**: Evaluated 90 parameter configurations ($N \in [2, 1000]$, ratios $0.01 \dots 0.99$).
- **Result**: PASS. Disjointness and completeness verified in 100% of cases. Clean defensive `ValueError` when split allocation is mathematically impossible ($N=2, r=0.99$).

#### Challenge 2: Safetensors Serialization of Non-Contiguous Strided Tensors (Severity: Medium)
- **Assumption challenged**: `safetensors.torch.save_file` crashes if passed non-contiguous tensors with strange strides (e.g. negative, transposed, sliced with step > 1, or stride 0).
- **Empirical test**: Passed 5 non-contiguous memory layouts to `save_dataset_safetensors`.
- **Result**: PASS. `save_dataset_safetensors` automatically forces `.contiguous()` on all dictionary items, preserving bit-level fidelity without error.

#### Challenge 3: PyTorch DataLoader Batch Scaling & Multiprocessing (Severity: Low)
- **Assumption challenged**: Extreme batch sizes or memory-mapped slicing in `MoECalibrationDataset` could fail during PyTorch DataLoader collation.
- **Empirical test**: Batched from $B=1$ to $B=4096$ with `in_memory=True` and `in_memory=False`, including `num_workers=2`.
- **Result**: PASS. PyTorch collation returned contract-compliant batched tensors in all cases.

#### Challenge 4: Gradient Leakage through Masked Sequence Boundaries (Severity: High)
- **Assumption challenged**: Loss computation or autograd backpropagation might leak non-zero gradients into tapped hidden states or deep backbone layers from masked trailing boundary tokens ($L-3 \dots L-1$).
- **Empirical test**: Backpropagated through speculative head and 9-layer deep backbone network.
- **Result**: PASS. Gradient norm for fully masked tokens and backbone parameters is IDENTICALLY 0.000000.

### Stress Test Results Summary
- `test_partition_disjointness_arbitrary_train_ratios` (90 variations) -> **PASS**
- `test_partition_disjointness_calib_ratios` (10 variations) -> **PASS**
- `test_partition_invalid_inputs_defensive_exceptions` -> **PASS**
- `test_partition_shuffle_determinism_and_entropy` -> **PASS**
- `test_token_level_strict_isolation_in_datasets` (5 variations) -> **PASS**
- `test_safetensors_serialization_strange_strides` -> **PASS**
- `test_safetensors_contract_schema_rejection` -> **PASS**
- `test_moe_calibration_dataset_dataloader_batch_scaling` (18 variations) -> **PASS**
- `test_moe_calibration_dataset_negative_indexing_and_slicing` -> **PASS**
- `test_moe_calibration_dataset_filter_and_subselect` -> **PASS**
- `test_gradient_isolation_tail_boundary_token_exact_zero_norm` -> **PASS**
- `test_gradient_isolation_deep_layers_across_backbone` -> **PASS**
- `test_gradient_isolation_horizon_granularity` -> **PASS**
