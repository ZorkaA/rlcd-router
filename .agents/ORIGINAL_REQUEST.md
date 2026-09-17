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

## Phase 2 Request — 2026-09-17T12:27:21Z

# Teamwork Project Prompt — Draft

> Status: Launched
> Goal: Craft prompt → get user approval → delegate to teamwork_preview
> Requested team: [none — teamwork routes from the description]

Build Phase 2 (Swift/Metal Execution Pipeline) for the Asynchronous MoE Router.

Working directory: /Users/jack/Downloads/rlcd-router
Integrity mode: development

Please strictly follow the Phase 2 specifications defined in the Phase 2 section of `implementation_plan.md`.

## Requirements

### R1. Explicit Prefetching & Dual-Queue Fast I/O Setup
Implement Metal 3 Fast I/O dual-queues (`speculativeQueue` with PriorityLow and maxCommandBufferCount 16 for external NVMe, `fallbackQueue` with PriorityHigh). Use `MTLIOFileHandle` for explicit block reads (`loadBytes`/`loadBuffer`). Use `MTLSharedEvent` for zero-CPU synchronization.

### R2. Ring Buffer Pool & Fallback Pool
Allocate a fixed array of `MTLBuffer`s for the speculative Ring Buffer. Implement a strictly isolated 500MB Fallback Buffer Pool. On a cache-miss deadlock, allocate from the Fallback Pool, dispatch to `fallbackQueue`, mark the speculative slot as abandoned/dirty, and drop the signal when its IO callback fires.

### R3. Dispatch-Time LRU via Execution Log
The CPU must update LRU metadata *only* by draining the GPU Execution Log, never via pre-routing prediction. No GPU atomic timestamp updates.

### R4. ICB Conditional Execution & Cascading No-Ops
Implement a global 1-byte `abort_flag` buffer.
- **Expert Kernels:** Use ICB native conditional execution (Gating Kernel reads `abort_flag`, if true, writes zero threads to the ICB execution grid).
- **Standard Layer Kernels:** Use cascading `if (*abort_flag) return;` no-ops. Do not use `.untracked` buffers to preserve Metal's default hazard tracking.

### R5. MLX-Swift Execution Log & Recalibration
Implement a background Swift task that reads the Execution Log for Brier-score recalibration. Explicitly limit the MLX metal cache to 200MB (`mlx.core.metal.set_cache_limit`) to prevent OS-level memory compression of the Ring Buffer.

## Acceptance Criteria

### Execution & Safety
- [ ] Pipeline executes successfully without kernel panics or silent memory corruption.
- [ ] Memory leaks are non-existent (strict buffer lifecycle management).
- [ ] All parameters remain within acceptable bounds.
- [ ] Pipeline seamlessly handles speculative I/O fetches and falls back to synchronous fetches cleanly on cache misses.
- [ ] The `abort_flag` properly no-ops execution without corrupting the residual stream `x`.

## Follow-up Constraint — 2026-09-17T12:27:51Z

A quick constraint from the user while you build: Be extremely mindful of the overall memory footprint. Do not use 100% of the available RAM. Ensure the OS and the background agent processes have enough memory to run without OOMing or swapping heavily. Ensure your Ring Buffer and MLX limits stay safely within a conservative ceiling.
