# Milestone 1 Explorer 3: Dataset Splitting, Horizon Alignment & Persistence

You are teamwork_preview_explorer_m1_3.
Working directory: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_3
Parent: orchestrator (ce5bc762-f633-465c-9133-7ec43d0b5719)

MANDATORY INPUT:
- Read /Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md
- Read /Users/jack/Downloads/rlcd-router/PROJECT.md

YOUR ROLE & OBJECTIVE:
Investigate and design the exact technical specification and code blueprint for `src/data/dataset.py`:
1. Strictly Isolated 15–20% Held-Out Split:
   - Sequence-atomic splitting (e.g. 80 sequences train, 18 sequences calib = 18.4% held-out).
   - Zero context contamination between train and held-out calibration sets.
2. Horizon Alignment & Target Construction for Deep Layers (Layers 5–24, 20 layers) at $T+1, T+2, T+3$:
   - For token position $t$ in a sequence of length $L$, the lookahead targets are at $t+1, t+2, t+3$.
   - Boundary handling: Trailing tokens ($t = L-3, L-2, L-1$) cannot look ahead across sequence boundaries. Provide a boolean mask `valid_mask` or truncate valid training pairs so invalid cross-boundary targets are excluded.
3. Storage & I/O:
   - Save datasets using `safetensors.torch.save_file` and load via `safetensors.torch.load_file` (or memory mapping).
   - PyTorch Dataset class `MoECalibrationDataset(Dataset)` returning `(hidden_state, target_router_logits, target_top4, valid_mask)`.
Write `analysis.md` and `handoff.md` in your working directory.

## 2026-09-17T07:31:42Z
You are teamwork_preview_explorer_m1_3.
Your working directory is: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_3
Your parent is orchestrator (conversation ID: ce5bc762-f633-465c-9133-7ec43d0b5719).

MANDATORY INPUT:
- Read /Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md
- Read /Users/jack/Downloads/rlcd-router/PROJECT.md
- Read /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_3/DISPATCH.md

YOUR ROLE & OBJECTIVE:
Investigate and design the exact technical specification and code blueprint for `src/data/dataset.py`:
- Strictly isolated 80/20 train/calib sequence-level partitioning.
- Multi-horizon alignment (T+1, T+2, T+3) for deep layers (5-24) with boundary token masking (L-3..L-1).
- Safetensors persistence and PyTorch Dataset class.
Write analysis.md and handoff.md in your working directory, and notify parent.
