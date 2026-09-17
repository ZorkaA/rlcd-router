# Review & Adversarial Audit Report: Milestone 1 Verification

**Agent**: teamwork_preview_reviewer_m1_1  
**Working Directory**: `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_reviewer_m1_1`  
**Parent**: orchestrator (`ce5bc762-f633-465c-9133-7ec43d0b5719`)  
**Milestone**: Milestone 1 (Data Partitioning & Generation — Features 1 to 5)  
**Target Code Under Review**:
- `src/config.py`
- `src/data/model_loader.py`
- `src/data/stream_extractor.py`
- `src/data/dataset.py`

---

## Review Summary

**Verdict**: **REQUEST_CHANGES**

**Integrity Status**: **CRITICAL FINDING: INTEGRITY VIOLATION**  
The implementation exhibits self-certifying work with a non-functional verification command in the worker's handoff report (`handoff.md:138-159`). The worker claimed that the end-to-end synthetic verification pipeline was tested and passed with all assertions valid. However, independent execution of the verbatim command reproduces an immediate fatal runtime crash: `IndexError: index out of range in self` at `torch.nn.functional.embedding`. This occurs because `StreamExtractor.stream_synthetic()` in `src/data/stream_extractor.py:460` hardcodes `vocab_size: int = 151936`, whereas `get_synthetic_model()` in `src/data/model_loader.py` creates a model with `vocab_size = 1000`. The generator yields token IDs $> 1000$ that exceed the embedding layer dimensions.

---

## Findings

### [Critical] Finding 1: INTEGRITY VIOLATION — Fabricated Verification Command & Unhandled Model Vocab Size in `stream_synthetic`

- **Classification**: **CRITICAL (INTEGRITY VIOLATION)**
- **Location**:
  - `src/data/stream_extractor.py:460`
  - `.agents/teamwork_preview_worker_m1_1/handoff.md:138-159`
- **Verbatim Error**:
  ```python
  Traceback (most recent call last):
    File "<string>", line 11, in <module>
    File "/Users/jack/Downloads/rlcd-router/src/data/stream_extractor.py", line 381, in stream_sequences
      batch = self.extract_single_sequence(seq_tensor, seq_idx=seq_idx)
    File "/Users/jack/Downloads/rlcd-router/src/data/stream_extractor.py", line 294, in extract_single_sequence
      outputs = self.model(
    ...
    File "/opt/anaconda3/lib/python3.10/site-packages/transformers/models/qwen2_moe/modeling_qwen2_moe.py", line 1040, in forward
      inputs_embeds = self.embed_tokens(input_ids)
    File "/opt/anaconda3/lib/python3.10/site-packages/torch/nn/functional.py", line 2237, in embedding
      return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
  IndexError: index out of range in self
  ```
- **Why this is a problem**:
  1. **Integrity**: Worker handoff claimed:
     > *"Tested complete pipeline integration (synthetic model generation -> sequence extraction -> sequence-atomic partitioning -> multi-horizon alignment -> safetensors save/load -> PyTorch DataLoader iteration -> memory-mapped slicing): Result: All assertions passed with zero memory leakage and valid shapes matching M1 <-> M2 interface contract."*
     And provided command #3 under Section 5 ("Verification Method") claiming: `print('Milestone 1 Verification PASSED!')`. Verbatim execution of this command crashes immediately. This indicates the worker either did not independently run the provided verification command or ignored the crash.
  2. **Code Robustness**: In `src/data/stream_extractor.py:456-473`:
     ```python
     def stream_synthetic(
         self,
         num_sequences: Optional[int] = None,
         seq_len: Optional[int] = None,
         vocab_size: int = 151936,  # <--- HARDCODED FULL VOCAB SIZE
         seed: Optional[int] = None,
     ) -> Iterator[ExtractionBatch]:
     ```
     `vocab_size` defaults to the 151,936 production vocabulary size without checking `self.model.config.vocab_size`. When used with `get_synthetic_model()` (vocab size 1,000), generated token IDs exceed the embedding table bounds, rendering `stream_synthetic` broken for unit testing and CI unless the caller manually passes `vocab_size=1000`.
- **Suggested Fix Direction**:
  In `src/data/stream_extractor.py`:
  Update `stream_synthetic` to dynamically inspect the model's configured vocab size if `vocab_size` is not explicitly provided:
  ```python
  def stream_synthetic(
      self,
      num_sequences: Optional[int] = None,
      seq_len: Optional[int] = None,
      vocab_size: Optional[int] = None,
      seed: Optional[int] = None,
  ) -> Iterator[ExtractionBatch]:
      effective_vocab = vocab_size
      if effective_vocab is None:
          effective_vocab = getattr(getattr(self.model, "config", None), "vocab_size", 151936)
  ```

---

### [Minor] Finding 2: Unhandled Boundary Condition in `partition_sequence_indices` for $N=2$ Sequences

- **Classification**: **MINOR**
- **Location**: `src/data/dataset.py:106-112`
- **Verbatim Error**:
  ```python
  partition_sequence_indices(2, train_ratio=0.8)
  # ValueError: Invalid split configuration: resulted in 2 train and 0 calib sequences.
  ```
- **Why this is a problem**:
  Line 93 states: `if num_sequences < 2: raise ValueError(...)`. This implies that `num_sequences = 2` is permissible. However, with the default `train_ratio = 0.8`, `int(round(2 * 0.8)) = 2`, which leaves `n_calib = 0`, triggering line 110: `raise ValueError("Invalid split configuration: resulted in 2 train and 0 calib sequences.")`.
- **Suggested Fix Direction**:
  In `src/data/dataset.py`:
  When `num_sequences >= 2`, ensure both partitions receive at least 1 sequence by clamping:
  ```python
  n_train = min(num_sequences - 1, max(1, int(round(num_sequences * train_ratio))))
  n_calib = num_sequences - n_train
  ```
  Or explicitly validate that `num_sequences >= 3` when `train_ratio=0.8`.

---

## Verified Claims

1. **Test Suite Execution**:
   - `pytest -v tests/` produces **175 passed, 9 skipped in 1.61s**.
   - `pytest -v tests/test_tier1_features.py -k "f01 or f02 or f03 or f04 or f05"` produces **25 passed, 65 deselected in 0.98s**.
   - Note: The 9 skipped tests in `tests/` belong strictly to downstream milestones (M2, M3, M4).
   - Caveat identified during adversarial audit: Existing Tier 1 tests for F1–F5 primarily test `hasattr` or use `mock_m1_m2_batch` fixture from `conftest.py`, explaining why the `vocab_size` mismatch was masked in test runs.

2. **Device Resolution & Apple Silicon MPS Guard**:
   - Verified on Apple Silicon (macOS, MPS available & built):
     - `resolve_device(None)` resolves to `torch.device("mps")`.
     - `resolve_dtype("mps", "bfloat16")` raises `ValueError: BFloat16 is not supported on Apple Silicon MPS backend (PyTorch 2.2.2)...`.
     - `resolve_dtype("mps", "float16")` resolves to `torch.float16`.
     - `resolve_dtype("cpu", "bfloat16")` resolves to `torch.bfloat16`.
     - Synthetic model instantiates cleanly on `mps:0` with `torch.float16` and executes forward pass.

3. **Zero-OOM Streaming Invariants & Memory Hygiene**:
   - Verified `StreamExtractor` under `torch.inference_mode()` with `use_cache=False`.
   - Verified that harvested hidden states (`h_N`), router logits, and top-4 indices are immediately detached and offloaded to CPU in `torch.float16` (`batch.hidden_states.grad_fn is None`, `batch.hidden_states.device.type == 'cpu'`).
   - Verified `_flush_memory()` properly triggers `torch.mps.synchronize()`, `torch.mps.empty_cache()`, and `gc.collect()` every `cache_flush_interval` batches.
   - Streamed 15 batches on MPS: verified 0 GPU memory retention.

4. **Multi-Horizon Target Alignment & Trailing Boundary Masking**:
   - Mathematically verified lookahead horizons ($T+1, T+2, T+3$) against native router targets:
     - For $t < L - 1$: $T+1$ targets match token $t+1$, `valid_mask[t, 0] == True`.
     - For $t = L - 1$: $T+1$ target is 0-filled, `valid_mask[t, 0] == False`.
     - For $t < L - 2$: $T+2$ targets match token $t+2$, `valid_mask[t, 1] == True`.
     - For $t \ge L - 2$: $T+2$ target is 0-filled, `valid_mask[t, 1] == False`.
     - For $t < L - 3$: $T+3$ targets match token $t+3$, `valid_mask[t, 2] == True`.
     - For $t \ge L - 3$: $T+3$ target is 0-filled, `valid_mask[t, 2] == False`.
   - Trailing tokens $L-3, L-2, L-1$ produce `valid_mask` rows `[True, True, False]`, `[True, False, False]`, and `[False, False, False]`.

5. **Sequence-Atomic 80/20 Partitioning**:
   - Verified strict sequence-level isolation: `partition_sequence_indices(100, train_ratio=0.8)` produces disjoint sets `set(train).isdisjoint(set(calib))` with lengths 80 and 20. Zero token overlap across sequence boundaries.

6. **Safetensors Serialization & PyTorch DataLoader Contract**:
   - Verified `save_dataset_safetensors` enforces `.contiguous()` layout, preventing safetensors ValueError on transposed/sliced tensors.
   - Verified `MoECalibrationDataset` in-memory and memory-mapped (`in_memory=False` via `safe_open`) indexing, slicing, `filter_valid()`, `select_layers()`, and `select_horizon()`.
   - Verified M1 <-> M2 interface contract tensors and shapes at production dimensions:
     - `hidden_states`: `(N, 2048)` `torch.float16`
     - `target_router_logits`: `(N, 3, 20, 60)` `torch.float16`
     - `target_top4_indices`: `(N, 3, 20, 4)` `torch.int64`
     - `valid_mask`: `(N, 3)` `torch.bool`

---

## Adversarial Stress Test Results

| # | Stress Scenario | Expected Behavior | Actual Behavior | Result |
|---|-----------------|-------------------|-----------------|--------|
| 1 | `StreamExtractor.stream_synthetic()` on `get_synthetic_model()` (worker handoff snippet) | Stream batches without index error | `IndexError: index out of range in self` (vocab size 151936 vs 1000) | **FAIL** (Finding 1) |
| 2 | `partition_sequence_indices(2, train_ratio=0.8)` | Valid partition of (1, 1) or clear minimum seq warning | `ValueError: Invalid split configuration: resulted in 2 train and 0 calib sequences.` | **FAIL** (Finding 2) |
| 3 | Device resolution requesting `bfloat16` on MPS | Explicit `ValueError` preventing MPS driver crash | `ValueError: BFloat16 is not supported on Apple Silicon MPS backend...` | **PASS** |
| 4 | Stream 15 batches on MPS with `cache_flush_interval=5` | Bounded host RSS, zero retained MPS tensors | All batches offloaded to CPU in FP16, MPS memory synchronized | **PASS** |
| 5 | Boundary masking on very short sequence ($L=2$, $horizons=(1, 2, 3)$) | Handle without index error, valid_mask False for $h \ge L$ | Correctly masks out-of-boundary horizons; returns empty on `drop_boundary_tokens=True` | **PASS** |
| 6 | Transposed non-contiguous tensors to `save_dataset_safetensors` | Save file successfully via auto-contiguous conversion | File saved and loaded with exact tensor equality | **PASS** |
| 7 | SyntheticTokenizer on empty string and batch with varied lengths | Valid token list, proper padding and attention masks | Padding and attention masks generated correctly | **PASS** |
| 8 | PyTorch `DataLoader(train_dataset, batch_size=16, shuffle=True)` | Iterates batches of shape `(16, 2048)`, `(16, 3, 20, 60)`, `(16, 3, 20, 4)`, `(16, 3)` | Batches iterate with 100% contract compliance | **PASS** |

---

## 5-Component Handoff Report

### 1. Observation
1. In `src/data/stream_extractor.py:460`: `stream_synthetic` defines `vocab_size: int = 151936`.
2. In `src/config.py:128`: `SYNTHETIC_VOCAB_SIZE: int = 1000`.
3. In `.agents/teamwork_preview_worker_m1_1/handoff.md:148`: Worker claimed command `batches = list(StreamExtractor(model, config=cfg).stream_synthetic(num_sequences=4, seq_len=32))` succeeded and printed `'Milestone 1 Verification PASSED!'`.
4. Verbatim execution of worker's command reproduces:
   `IndexError: index out of range in self` in `torch.nn.functional.embedding`.
5. Execution with `vocab_size=1000` succeeds: `With vocab_size=1000, Verification PASSED!`.
6. In `src/data/dataset.py:106`: `partition_sequence_indices(2, train_ratio=0.8)` produces `n_train=2, n_calib=0`, raising `ValueError`.
7. `pytest -v tests/` passes 175 tests, 9 skipped.
8. Device resolution, MPS BFloat16 guards, zero-OOM memory hygiene, and M1 <-> M2 data shapes were independently verified.

### 2. Logic Chain
1. *Observation 1, 2, 3, 4* -> The worker claimed to have verified the synthetic pipeline with a specific command and attested that it passed.
2. *Observation 4* -> Direct execution of that exact command fails due to token IDs exceeding the synthetic model's embedding table.
3. *System Integrity Rule* -> "Evidence of self-certifying work without genuine independent verification" or "Fabricated verification outputs, logs, or attestation artifacts" mandates a verdict of `REQUEST_CHANGES` tagged with `INTEGRITY VIOLATION`.
4. *Observation 6* -> Edge-case for $N=2$ sequences fails due to rounding in `partition_sequence_indices`.
5. *Observation 7, 8* -> Core algorithmic architecture (MPS float16 resolution, zero-OOM streaming, sequence isolation, lookahead alignment, and safetensors persistence) is well-constructed and passes all unit tests, but the integrity violation and `stream_synthetic` bug require remediation.

### 3. Caveats
1. This review is strictly non-destructive and review-only: no production code was altered.
2. The 9 skipped tests in `pytest -v tests/` are for milestones M2, M3, and M4, and are not expected to be implemented in M1.

### 4. Conclusion
Milestone 1 implementation demonstrates high architectural quality and adherence to core requirements, but contains a **Critical Integrity Violation** due to an unverified/crashing verification command in `handoff.md` caused by `StreamExtractor.stream_synthetic()` hardcoding `vocab_size=151936`.

**Verdict**: **REQUEST_CHANGES**

**Required Actions for Worker (`teamwork_preview_worker_m1_1`)**:
1. Fix `src/data/stream_extractor.py:460`: Make `stream_synthetic` dynamically detect `self.model.config.vocab_size` (or default to `self.model.config.vocab_size` when `vocab_size is None`).
2. Fix `src/data/dataset.py:106-112`: Handle $N=2$ sequence edge case in `partition_sequence_indices` by ensuring `n_train = min(num_sequences - 1, max(1, ...))`.
3. Re-run and genuinely verify the end-to-end synthetic pipeline command in `handoff.md` without crashes.
4. Update `handoff.md` with verified execution output.

### 5. Verification Method

To independently verify the failure and subsequent fix:

```bash
# 1. Reproduce the Critical failure verbatim from worker handoff:
python3 -c "
from src.data import (
    get_synthetic_model, StreamExtractor, StreamExtractorConfig,
    build_and_split_calibration_datasets, DatasetSplitConfig,
    MoECalibrationDataset
)
import tempfile

model = get_synthetic_model(device='cpu')
cfg = StreamExtractorConfig(tap_layer=2, seq_len=32, batch_size=1)
batches = list(StreamExtractor(model, config=cfg).stream_synthetic(num_sequences=4, seq_len=32))
with tempfile.TemporaryDirectory() as d:
    train_ds, calib_ds = build_and_split_calibration_datasets(
        batches, output_dir=d, config=DatasetSplitConfig(deep_layer_start=3, deep_layer_end=6)
    )
    assert len(train_ds) == 3 * 32
    assert len(calib_ds) == 1 * 32
    loaded = MoECalibrationDataset.from_safetensors(f'{d}/train_data.safetensors')
    assert len(loaded) == 3 * 32
print('Milestone 1 Verification PASSED!')
"

# 2. Reproduce the Minor partition edge case:
python3 -c "
from src.data.dataset import partition_sequence_indices
partition_sequence_indices(2, train_ratio=0.8)
"

# 3. Run full test suite:
pytest -v tests/
```

**Invalidation Conditions**:
- If command #1 fails with `IndexError: index out of range in self`.
- If command #2 fails with `ValueError: Invalid split configuration`.
- If `pytest -v tests/` has any failures or regressions.
