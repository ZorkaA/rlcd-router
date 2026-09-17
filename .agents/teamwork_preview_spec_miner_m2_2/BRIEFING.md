# BRIEFING — 2026-09-17T17:06:00Z

## Mission
Discover, probe, and document the comprehensive specification and Swift blueprint for the strictly isolated 500MB Fallback Buffer Pool (Sources/AsyncMoERouter/BufferPools/FallbackBufferPool.swift).

## 🔒 My Identity
- Archetype: Specification Miner
- Roles: [Specification Miner, Teamwork Specialist]
- Working directory: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_spec_miner_m2_2
- Original parent: 913b8328-6b64-4881-a075-c0057bc23d84
- Milestone: Milestone 2 (Buffer Pools & Fallback Pool)

## 🔒 Key Constraints
- Strict 500MB Memory Ceiling: Exactly 524,288,000 bytes (500 * 1024 * 1024). Invariant: Total allocated fallback buffer capacity MUST NEVER exceed 500MB under any condition.
- Strict Isolation from Speculative Prefetching: Pool must reject speculative prefetch requests and remain reserved exclusively for demand-fetch cache misses.
- Buffer Lifecycle, Allocation, Tracking, and Clean Release: Zero fragmentation, zero memory leaks, thread-safe, return to pool when compute finishes.
- Read-only on production source code: Spec miners discover and document specifications with complete Swift implementation blueprints in handoff reports; do not implement directly in production tree.

## Current Parent
- Conversation ID: 913b8328-6b64-4881-a075-c0057bc23d84
- Updated: 2026-09-17T17:06:00Z

## Task Summary
- **What to build**: Comprehensive technical specification, edge case analysis, and production-grade Swift implementation blueprint for `FallbackBufferPool.swift` and associated types (`FallbackSlot`, `AllocationContext`, error handling, metrics).
- **Success criteria**:
  1. Complete feature enumeration and probe tables (common usage, boundary conditions, edge cases, invalid inputs).
  2. Proof and enforcement mechanisms of 500MB (524,288,000 bytes) hard memory ceiling.
  3. Strict isolation enforcement blocking speculative requests at interface and compile/runtime level.
  4. Complete lifecycle state tracking, reclamation, and leak-free pool recycling.
  5. Copy-pasteable, production-ready Swift code blueprint.
- **Interface contracts**: `/Users/jack/Downloads/rlcd-router/.agents/orchestrator_phase2/PROJECT.md`
- **Code layout**: `/Users/jack/Downloads/rlcd-router/Sources/AsyncMoERouter/BufferPools/FallbackBufferPool.swift`

## Key Decisions Made
- Architecture alignment: Slot size is 17,301,504 bytes for Qwen1.5-MoE-A2.7B (16.50 MiB, exactly 1056 x 16KB pages). 500MB capacity fits floor(524,288,000 / 17,301,504) = 30 slots (519,045,120 bytes, leaving 5,242,880 bytes headroom).
- Hard Invariant: `totalAllocatedBytes <= maxCapacityBytes` strictly enforced; slot 31 allocation attempts are refused with `capacityExhausted`.
- Strict Isolation: DemandFetchContext requirement and AllocationIntent dispatch actively reject speculative prefetch attempts (`speculativeRequestRejected`).
- Double-reclaim & foreign slot guards: Track active slots in `_inUseSlots: [Int: FallbackSlot]` to prevent free list corruption and concurrent buffer aliasing.
- Concurrency & Thread Safety: Protected by `OSAllocatedUnfairLock` without priority inversion or actor hop overhead.
- Backward Compatibility: Preserved legacy `allocate(device:)` and `reclaimLegacy(_:)` methods so existing call sites (`DeadlockResolver`, unit tests) continue working seamlessly.

## Artifact Index
- `.agents/teamwork_preview_spec_miner_m2_2/progress.md` — Liveness and task execution checklist
- `.agents/teamwork_preview_spec_miner_m2_2/handoff.md` — Final 5-component handoff report with complete blueprints

## Loaded Skills
- None required/specified in dispatch.
