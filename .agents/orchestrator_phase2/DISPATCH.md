## 2026-09-17T12:28:36Z

<USER_REQUEST>
You are the Project Orchestrator for Phase 2: Swift/Metal Execution Pipeline of the Asynchronous MoE Router project.

Your assigned working directory is:
/Users/jack/Downloads/rlcd-router/.agents/orchestrator_phase2

The authoritative user request is recorded in:
/Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md

Project root: /Users/jack/Downloads/rlcd-router
Integrity mode: development

## Phase 2 Objectives & Requirements:
Build Phase 2 (Swift/Metal Execution Pipeline) for the Asynchronous MoE Router.
Please strictly follow the Phase 2 specifications defined in the Phase 2 section of ORIGINAL_REQUEST.md.

### R1. Explicit Prefetching & Dual-Queue Fast I/O Setup
Implement Metal 3 Fast I/O dual-queues (speculativeQueue with PriorityLow and maxCommandBufferCount 16 for external NVMe, fallbackQueue with PriorityHigh). Use MTLIOFileHandle for explicit block reads (loadBytes/loadBuffer). Use MTLSharedEvent for zero-CPU synchronization.

### R2. Ring Buffer Pool & Fallback Pool
Allocate a fixed array of MTLBuffers for the speculative Ring Buffer. Implement a strictly isolated 500MB Fallback Buffer Pool. On a cache-miss deadlock, allocate from the Fallback Pool, dispatch to fallbackQueue, mark the speculative slot as abandoned/dirty, and drop the signal when its IO callback fires.

### R3. Dispatch-Time LRU via Execution Log
The CPU must update LRU metadata *only* by draining the GPU Execution Log, never via pre-routing prediction. No GPU atomic timestamp updates.

### R4. ICB Conditional Execution & Cascading No-Ops
Implement a global 1-byte abort_flag buffer.
- Expert Kernels: Use ICB native conditional execution (Gating Kernel reads abort_flag, if true, writes zero threads to the ICB execution grid).
- Standard Layer Kernels: Use cascading `if (*abort_flag) return;` no-ops. Do not use `.untracked` buffers to preserve Metal's default hazard tracking.

### R5. MLX-Swift Execution Log & Recalibration
Implement a background Swift task that reads the Execution Log for Brier-score recalibration. Explicitly limit the MLX metal cache to 200MB (mlx.core.metal.set_cache_limit) to prevent OS-level memory compression of the Ring Buffer.

## Acceptance Criteria:
- Pipeline executes successfully without kernel panics or silent memory corruption.
- Memory leaks are non-existent (strict buffer lifecycle management).
- All parameters remain within acceptable bounds.
- Pipeline seamlessly handles speculative I/O fetches and falls back to synchronous fetches cleanly on cache misses.
- The abort_flag properly no-ops execution without corrupting the residual stream x.

## Critical Constraints:
- Memory footprint: Be extremely mindful of the overall memory footprint. Do not use 100% of the available RAM. Ensure the OS and background agent processes have enough memory to run without OOMing or swapping heavily. Ensure your Ring Buffer and MLX limits stay safely within a conservative ceiling.
- Pure orchestrator: Do not write code directly; decompose tasks and dispatch to specialists (explorers, workers, reviewers, challengers, test writers).
- Regularly update `progress.md` and `BRIEFING.md` in your working directory (.agents/orchestrator_phase2) so the Sentinel can track status and liveness.
- Report milestone completions, and when complete, signal completion for independent victory audit.
</USER_REQUEST>
