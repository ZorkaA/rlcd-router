# Original User Request

## Initial Request — 2026-09-17T07:22:59Z

# Teamwork Project Prompt — Draft

> Status: Launched
> Goal: Craft prompt → get user approval → delegate to teamwork_preview
> Requested team: Full Team

Build the Phase 1 PyTorch ML Calibration scripts for an Asynchronous MoE Router. This pipeline generates calibration data, trains a linear speculative Medusa head, and executes post-hoc mathematical calibration (NLL Temperature Scaling Grid & Targeted ECE evaluation).

Working directory: /Users/jack/Downloads/rlcd-router
Integrity mode: development

## Requirements

### R1. Data Partitioning & Generation
Implement a PyTorch script to load `Qwen/Qwen1.5-MoE-A2.7B`. Stream a 100k-token corpus through it, logging the Layer N hidden states and the native gating decisions across all layers. Carve off a 15-20% held-out calibration split strictly isolated from training data. 

### R2. Linear Speculative Head & Training
Implement a Medusa-style linear speculative head attached to Layer N to predict deep-layer routing distributions (Layers 5-24) for $T+1$ through $T+3$. Train using Cross-Entropy loss. If confidence ranking exhibits flaws, implement a tunable Maximum Mean Calibration Error (MMCE) penalty (CE + λ·MMCE).

### R3. Grid-Based Temperature Scaling
Implement a post-hoc calibration pass. Fit a grid of temperature scalars $T(s)$ indexed by `{early layers 5-10, late layers 11-24} × {T+1, T+2, T+3}`. Fit these scalars on the held-out split by strictly minimizing Negative Log-Likelihood (NLL) via LBFGS. Apply L2 regularization toward $T=1.0$ for sparse buckets. **Do not use ECE for optimization**, as it is non-differentiable.

### R4. Automated Testing & Verification
Implement extensive logging and a comprehensive automated test suite. Monitor memory usage to catch leaks. Evaluate the "Targeted Gating" criteria on the held-out split: report Targeted ECE specifically at the 0.05 (abort) and 0.85 (mass cutoff) decision boundaries, broken down by layer bucket and lookahead horizon.

## Acceptance Criteria

### Execution & Correctness
- [ ] The pipeline successfully generates training data from Qwen1.5-MoE-A2.7B without OOM crashes.
- [ ] The speculative head trains successfully, logging loss curves.
- [ ] The LBFGS temperature scaling script successfully outputs the grid of scalars minimizing NLL.
- [ ] The test suite runs end-to-end without errors.
- [ ] Targeted ECE at 0.05 and 0.85 is explicitly computed and printed in the final evaluation report.
