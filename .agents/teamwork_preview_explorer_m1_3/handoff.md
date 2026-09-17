# Handoff Report: Dataset Splitting, Horizon Alignment & Persistence Specification

**Agent**: `teamwork_preview_explorer_m1_3`  
**Working Directory**: `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_3`  
**Target Code Path**: `/Users/jack/Downloads/rlcd-router/src/data/dataset.py`  
**Parent**: `orchestrator` (`ce5bc762-f633-465c-9133-7ec43d0b5719`)  
**Handoff Type**: Hard (Investigation & Specification Complete)  
**Date**: 2026-09-17  

---

## 1. Observation

### 1.1 Direct Specification & Architecture Observations
- **`ORIGINAL_REQUEST.md` (Lines 18–20, R1)**:
  > "Stream a 100k-token corpus through it, logging the Layer N hidden states and the native gating decisions across all layers. Carve off a 15-20% held-out calibration split strictly isolated from training data."
- **`PROJECT.md` (Lines 57–58, Features 4 & 5)**:
  > "Feature 4: Strictly Isolated Train/Calib Split: Sequence-atomic 80/20 partitioning with trailing token masking (L-3..L-1) for horizon isolation"  
  > "Feature 5: Dataset Persistence & Sharding: Persist train and held-out calibration datasets efficiently using safetensors format"
- **`PROJECT.md` (Lines 84–94, M1 ↔ M2 Contract)**:
  > Data format: `train_data.safetensors` and `calib_data.safetensors`  
  > Tensors:  
  > - `hidden_states`: FloatTensor `(num_samples, 2048)`  
  > - `target_router_logits`: FloatTensor `(num_samples, 3, 20, 60)` (horizons: T+1, T+2, T+3; layers: 5–24; experts: 60)  
  > - `target_top4_indices`: LongTensor `(num_samples, 3, 20, 4)`  
  > - `valid_mask`: BoolTensor `(num_samples, 3)` indicating boundary validity for future horizons.

### 1.2 Empirical Runtime Tool Commands & Verbatim Outputs
1. **Environment Versions**:
   - Command: `python3 -c "import torch, safetensors, numpy; print(torch.__version__, safetensors.__version__, numpy.__version__)"`
   - Output: `2.2.2 0.6.2 1.26.4`
2. **Safetensors Non-Contiguous Error Discovery**:
   - Command: Saving transposed non-contiguous tensor `x = torch.randn(10, 10).t()` via `safetensors.torch.save_file`.
   - Verbatim error:
     > `ValueError: You are trying to save a non contiguous tensor: 'x' which is not allowed. It either means you are trying to save tensors which are reference of each other in which case it's recommended to save only the full tensors, and reslice at load time, or simply call .contiguous() on your tensor to pack it before saving.`
   - Resolution: Calling `.contiguous()` on all contract tensors is strictly enforced prior to saving.
3. **End-to-End Simulation on 98 Sequences (100,352 Tokens)**:
   - Command: Simulated 98 sequences ($L=1024, d=2048, 20\text{ deep layers}, 60\text{ experts}$) split into 80 train sequences and 18 calib sequences.
   - Result:
     - Train tensors: `hidden_states` `torch.Size([81920, 2048])`, `target_router_logits` `torch.Size([81920, 3, 20, 60])`, `target_top4_indices` `torch.Size([81920, 3, 20, 4])`, `valid_mask` `torch.Size([81920, 3])`
     - Calib tensors: `hidden_states` `torch.Size([18432, 2048])`, `target_router_logits` `torch.Size([18432, 3, 20, 60])`, `target_top4_indices` `torch.Size([18432, 3, 20, 4])`, `valid_mask` `torch.Size([18432, 3])`
     - Save time: `0.618 s` to disk.
     - On-disk sizes: `train_data.safetensors` is `1032.73 MB`, `calib_data.safetensors` is `232.37 MB` (total: `1.26 GB`).
4. **PyTorch DataLoader Iteration Throughput**:
   - Command: Iterated 10,000 samples with `DataLoader(MoECalibrationDataset, batch_size=128, shuffle=True)`.
   - Output: `DataLoader iterated 10000 samples in 62.37 ms (160346 samples/sec)`.
5. **Autograd Mask Invariant Verification**:
   - Command: Executed soft cross-entropy with `valid_mask[-1, 2] = False` and called `.backward()`.
   - Output: `Grad norm for valid positions: 0.0024218`, `Grad norm for invalid positions: 0.0`. Exactly zero gradient leakage across sequence boundary tokens.

---

## 2. Logic Chain

1. **Why Partitioning Must Be Sequence-Atomic (Observation 1.1)**:
   - Self-attention operates across the sequence window of length $L=1024$.
   - If tokens within the same sequence were partitioned across train and calib, attention states would leak information from train to calib, invalidating post-hoc mathematical calibration.
   - Partitioning entire sequences (e.g., 80 sequences train, 18 sequences calib = 18.4% held-out) guarantees that the calibration split has zero sequence overlap and zero context contamination.

2. **Why Trailing Boundary Masking Is Mandatory (Observations 1.1 & 1.2)**:
   - To predict lookahead routing distributions at $T+1, T+2, T+3$, position $t$ accesses tokens at $t+1, t+2, t+3$.
   - For trailing positions $t = L-3, L-2, L-1$, the lookahead targets cross beyond the sequence boundary into future independent sequences or padding.
   - Setting `valid_mask[L-3] = [True, True, False]`, `valid_mask[L-2] = [True, False, False]`, and `valid_mask[L-1] = [False, False, False]` with zero-filled targets ensures that downstream loss functions multiply by $0.0$, producing exactly $0.0$ gradient norm (verified in Observation 1.2.5).

3. **Why Contiguity Enforcement Is Required (Observation 1.2.2)**:
   - `safetensors.torch.save_file` throws `ValueError` on non-contiguous tensors.
   - Slicing layers `router_logits[:, 4:24, :]` or concatenating sequence batches can produce non-contiguous strides in PyTorch.
   - Calling `.contiguous()` inside `save_dataset_safetensors()` guarantees zero serialization crashes.

4. **Why Dual-Mode Dataset Loading Is Optimal (Observations 1.2.3 & 1.2.4)**:
   - The total calibration dataset is 1.26 GB (1.03 GB train, 232 MB calib).
   - On machines with $\ge 8\text{ GB}$ RAM, loading into memory provides 160,000+ samples/sec throughput.
   - For memory-constrained runs or rapid inspection, `safe_open` slice access provides zero-RAM initialization.

---

## 3. Caveats

1. **Synthetic Test Fixtures vs 24-Layer Model**:
   - The genuine `Qwen/Qwen1.5-MoE-A2.7B` has 24 layers and 60 experts, where deep layers are Layers 5–24 (20 layers).
   - The fast synthetic fixture (`Qwen2MoeConfig`) has 6 layers and 16 experts.
   - To prevent crashes when running unit tests on synthetic models, `deep_layer_start` and `deep_layer_end` are configurable parameters with defaults 5 and 24. For a 6-layer model, passing `deep_layer_start=4, deep_layer_end=6` allows full verification.
2. **Short Sequences ($L \le 3$)**:
   - If a sequence has length $L \le 3$, horizon $T+3$ cannot be formed. The alignment algorithm handles this gracefully by returning `valid_mask` with all `False` for missing lookaheads rather than throwing an `IndexError`.
3. **Boundary Truncation Mode**:
   - If `drop_boundary_tokens=True` is chosen, exactly 3 tokens are dropped per sequence. For 98 sequences, this leaves $98 \times 1021 = 100,058$ samples, which still exceeds the 100k token requirement.

---

## 4. Conclusion

1. The exact technical specification and production code blueprint for `src/data/dataset.py` is established in `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_3/analysis.md`.
2. The design satisfies 100% of the requirements from `ORIGINAL_REQUEST.md` (R1), `PROJECT.md` (Features 4 & 5, M1 ↔ M2 Contract), and `DISPATCH.md`.
3. Key interfaces ready for the worker implementation:
   - `partition_sequence_indices(num_sequences, train_ratio=0.8, calib_ratio=None, shuffle=False, seed=42)`
   - `align_sequence_targets(hidden_states, router_logits, deep_layer_start=5, deep_layer_end=24, horizons=(1, 2, 3), top_k=4, drop_boundary_tokens=False)`
   - `save_dataset_safetensors(tensors, filepath, metadata=None)`
   - `load_dataset_safetensors(filepath, device="cpu")`
   - `MoECalibrationDataset(Dataset)` with `filter_valid()`, `select_layers()`, `select_horizon()`, and `from_safetensors()`
   - `build_and_split_calibration_datasets(sequences, output_dir, config=None)`

---

## 5. Verification Method

To independently verify the technical findings and contract compliance:

```bash
# 1. Run the end-to-end dataset simulation test
python3 -c "
import torch
from safetensors.torch import save_file, load_file
import tempfile, os

# Contract verification
N = 1000
h = torch.randn(N, 2048, dtype=torch.float16)
l = torch.randn(N, 3, 20, 60, dtype=torch.float16)
top4 = torch.randint(0, 60, (N, 3, 20, 4), dtype=torch.int64)
mask = torch.ones(N, 3, dtype=torch.bool)
mask[-3:, 2] = False

with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, 'train_data.safetensors')
    save_file({'hidden_states': h, 'target_router_logits': l, 'target_top4_indices': top4, 'valid_mask': mask}, p)
    loaded = load_file(p)
    assert loaded['hidden_states'].shape == (N, 2048)
    assert loaded['target_router_logits'].shape == (N, 3, 20, 60)
    assert loaded['target_top4_indices'].shape == (N, 3, 20, 4)
    assert loaded['valid_mask'].shape == (N, 3)
    print('Safetensors M1-M2 contract verification SUCCESS')
"
```

Expected output:
`Safetensors M1-M2 contract verification SUCCESS`
