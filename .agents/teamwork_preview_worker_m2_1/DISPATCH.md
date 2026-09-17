# Dispatch Assignment: Milestone 2 Worker

**Assigned Agent**: `teamwork_preview_worker_m2_1`  
**Milestone**: Phase 2 Milestone 2 (Ring Buffer Pool & Fallback Pool — Requirement R2)  
**Assigned Working Directory**: `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_worker_m2_1`  
**Date**: 2026-09-17  

---

## Mandatory Integrity Warning
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

---

## 1. Objective
Implement the production-grade Swift/Metal buffer pool architecture for Milestone 2:
1. `Sources/AsyncMoERouter/BufferPools/SpeculativeRingBuffer.swift`
2. `Sources/AsyncMoERouter/BufferPools/FallbackBufferPool.swift`
3. `Sources/AsyncMoERouter/BufferPools/DeadlockResolver.swift`
4. `swift_tests/AsyncMoERouterTests/Unit/BufferPoolTests.swift`

---

## 2. Input Specifications & Authoritative Blueprints
You MUST read and implement the blueprints already prepared and verified by the exploration team:
- **SpeculativeRingBuffer blueprint**:
  `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m2_1/proposed_SpeculativeRingBuffer.swift`
  - Fixed array of 16 `MTLBuffer` slots in `.storageModeShared`.
  - Sizing: $17,301,504$ bytes (FP16 Qwen1.5-MoE-A2.7B) or 24,576 bytes (synthetic).
  - Dedicated `SyncEvent` per slot to eliminate cross-slot ticket interference.
  - OSAllocatedUnfairLock thread safety.
  - 5-state lifecycle: `.free`, `.prefetching`, `.ready`, `.inUse`, `.abandoned`.
  - Dispatch-time LRU timestamps updated only via GPU Execution Log.
- **FallbackBufferPool blueprint**:
  `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_spec_miner_m2_2/handoff.md` (Section 5)
  - Strict 500MB hard ceiling ($524,288,000$ bytes) with zero violation tolerance.
  - Strict isolation: rejects `.speculative` prefetch requests with `FallbackPoolError.speculativeRequestRejected`. Reserved exclusively for demand fetch cache misses.
  - Context-driven allocation: `DemandFetchContext(expert:tokenIndex:reason:)`.
  - Thread-safe tracking with double-free and foreign-slot rejection.
- **DeadlockResolver blueprint**:
  `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m2_3/handoff.md` (Section 4.1)
  - Emergency demand fetch via `FastIOEngine.fallbackQueue` (PriorityHigh).
  - Marking speculative slot as `.abandoned` (dirty slot) and issuing `tryCancel()`.
  - Signal dropping in `completedHandler` so speculative completion never unblocks compute or triggers corrupt double-execution.
- **BufferPoolTests unit test suite**:
  `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m2_3/handoff.md` (Section 4.2)
  - Tests covering ring buffer cycling, slot lifecycle, fallback pool 500MB ceiling invariant, speculative rejection isolation, cache-miss deadlock resolution, signal dropping, and concurrency safety.

---

## 3. Scope Boundaries & File Ownership
You exclusively own:
- `Sources/AsyncMoERouter/BufferPools/SpeculativeRingBuffer.swift`
- `Sources/AsyncMoERouter/BufferPools/FallbackBufferPool.swift`
- `Sources/AsyncMoERouter/BufferPools/DeadlockResolver.swift`
- `swift_tests/AsyncMoERouterTests/Unit/BufferPoolTests.swift`

Do not modify existing files in `Sources/AsyncMoERouter/FastIO/` or `Common/` unless strictly necessary for API consistency.

---

## 4. Verification & Git Commit
1. Run `swift build` in `/Users/jack/Downloads/rlcd-router`. Ensure zero build errors and zero warnings.
2. Run `swift test` in `/Users/jack/Downloads/rlcd-router`. Ensure 100% of tests pass across both `FastIOTests` and `BufferPoolTests`.
3. Proactively commit per user global rules:
   `git add . && git commit -m "feat(phase2-m2): Implement Speculative Ring Buffer and 500MB Fallback Pool"`
4. Record full test execution logs and verification results in `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_worker_m2_1/handoff.md`.
5. Send a completion message to the orchestrator.

## 2026-09-17T17:08:12Z
You are the Worker for Phase 2 Milestone 2: Ring Buffer Pool & Fallback Pool (Requirement R2).
Your assigned working directory is: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_worker_m2_1
Read your dispatch assignment at: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_worker_m2_1/DISPATCH.md
Read the authoritative user requirements at: /Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md
Read Phase 2 architecture at: /Users/jack/Downloads/rlcd-router/.agents/orchestrator_phase2/PROJECT.md

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

