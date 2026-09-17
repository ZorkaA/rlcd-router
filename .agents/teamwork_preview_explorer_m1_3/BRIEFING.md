# BRIEFING — 2026-09-17T07:31:42Z

## Mission
Investigate and design the exact technical specification and code blueprint for `src/data/dataset.py` (80/20 sequence-level partitioning, T+1..T+3 horizon alignment for deep layers 5-24, boundary token masking, safetensors I/O, and PyTorch MoECalibrationDataset).

## 🔒 My Identity
- Archetype: explorer
- Roles: investigation, specification_design, synthesis
- Working directory: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_3
- Original parent: ce5bc762-f633-465c-9133-7ec43d0b5719
- Milestone: M1 (Data Partitioning & Generation)

## 🔒 Key Constraints
- Read-only investigation — do NOT implement source code directly.
- All investigation outputs go into `.agents/teamwork_preview_explorer_m1_3/`.
- Strict isolation of train vs calibration sequences (zero token/sequence contamination).
- Boundary token masking: for sequences of length $L$, lookahead $T+1..T+3$ cannot cross sequence boundaries ($L-3..L-1$ masked).
- Storage format: `safetensors.torch` (`train_data.safetensors`, `calib_data.safetensors`).
- PyTorch Dataset returning `(hidden_states, target_router_logits, target_top4_indices, valid_mask)`.

## Current Parent
- Conversation ID: ce5bc762-f633-465c-9133-7ec43d0b5719
- Updated: 2026-09-17T07:31:42Z

## Investigation State
- **Explored paths**: `ORIGINAL_REQUEST.md`, `PROJECT.md`, `DISPATCH.md`, Survey 2 & 3 handoffs, PyTorch & Safetensors empirical runtime tests.
- **Key findings**:
  - Sequence-atomic splitting partitions whole sequences (80 train / 18 calib = 18.4% held-out) with zero context contamination.
  - Slicing deep layers `router_logits[:, 4:24, :]` extracts 20 deep layers (Layers 5–24).
  - Trailing tokens $t \in \{L-3, L-2, L-1\}$ are masked with `valid_mask` and zero-padded targets to prevent boundary lookahead bleed.
  - Soft cross-entropy loss with `valid_mask` produces mathematically verified $0.0$ autograd gradient norm at invalid positions.
  - `safetensors.torch.save_file` strictly requires `.contiguous()` tensors.
  - Total dataset size for 100k tokens is 1.26 GB (1.03 GB train, 232 MB calib), saving in 0.618 s.
  - PyTorch DataLoader achieves 160,346 samples/sec throughput on in-memory dataset.
- **Unexplored areas**: None. Full specification and blueprint completed.

## Key Decisions Made
- Implemented sequence-atomic partitioning via `partition_sequence_indices()` with disjointness assertions.
- Designed vectorized multi-horizon target alignment with trailing token boundary masking in `align_sequence_targets()`.
- Added strict `.contiguous()` enforcement and schema validation in `save_dataset_safetensors()`.
- Architected `MoECalibrationDataset(Dataset)` with dual loading modes (in-memory and mmap via `safe_open`), NamedTuple `DatasetItem` supporting tuple unpacking and attribute access, and filtering helpers (`filter_valid()`, `select_layers()`, `select_horizon()`).

## Artifact Index
- `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_3/analysis.md` — In-depth technical analysis and blueprints
- `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_3/handoff.md` — 5-component handoff report
- `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_3/progress.md` — Heartbeat and progress tracker
