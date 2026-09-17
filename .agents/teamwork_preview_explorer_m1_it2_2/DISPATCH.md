# Milestone 1 Iteration 2 Explorer 2: Dataset Partitioning Clamping Remediation

You are teamwork_preview_explorer_m1_it2_2.
Working directory: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_it2_2
Parent: orchestrator (ce5bc762-f633-465c-9133-7ec43d0b5719)

MANDATORY INPUTS:
- /Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md
- /Users/jack/Downloads/rlcd-router/PROJECT.md
- /Users/jack/Downloads/rlcd-router/.agents/orchestrator/GATE_STATUS.md
- Reviewer 1 report: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_reviewer_m1_1/handoff.md
- Code under review: `src/data/dataset.py`

YOUR ROLE & OBJECTIVE:
Investigate the root cause and provide the exact fix specification for Reviewer 1's Finding 2:
In `src/data/dataset.py:106-112`, `partition_sequence_indices(2, train_ratio=0.8)` produces `n_train = 2, n_calib = 0` due to `round(2 * 0.8) = 2`, triggering `ValueError: Invalid split configuration: resulted in 2 train and 0 calib sequences.`
Design the exact fix ensuring that for any `num_sequences >= 2`, both train and calibration partitions receive at least 1 sequence by clamping:
`n_train = min(num_sequences - 1, max(1, int(round(num_sequences * train_ratio))))`
Write analysis.md and handoff.md in your working directory and notify parent.
