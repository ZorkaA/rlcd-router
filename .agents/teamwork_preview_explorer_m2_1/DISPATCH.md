# DISPATCH: Milestone 2 Explorer 1 (Speculative Ring Buffer Architecture)

## Assigned Working Directory
/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m2_1

## Task Objective
Design the technical architecture and Swift implementation blueprint for the Speculative Ring Buffer Pool (`Sources/AsyncMoERouter/BufferPools/SpeculativeRingBuffer.swift`).
Key areas:
1. Fixed Array Allocation: Pre-allocate 16 `MTLBuffer` slots in `.storageModeShared` (sizing: 17.3MB per expert for FP16, or synthetic test size from `MoEArchitectureConfig`).
2. Slot Lifecycle State Machine: `free` -> `prefetching` -> `ready` -> `inUse` -> `free`, with `.abandoned` (dirty) state handling.
3. Concurrency & Synchronization: Thread-safe slot acquisition and release via `OSAllocatedUnfairLock` or Swift actor.
4. Dedicated `SyncEvent` per slot: Ensure zero cross-slot ticket interference.
5. Provide a complete, production-grade Swift code blueprint for `SpeculativeRingBuffer.swift`.

## Authoritative References
- Requirements: `/Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md` (Phase 2 section, R2)
- Architecture & Contracts: `/Users/jack/Downloads/rlcd-router/.agents/orchestrator_phase2/PROJECT.md`
- Survey 2 findings: `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_survey_2/handoff.md`
- Milestone 1 implementations in `Sources/AsyncMoERouter/FastIO/` and `Common/`.

## Deliverables
- Maintain `progress.md`.
- Write your comprehensive architecture report to `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m2_1/handoff.md`.
- Send message to orchestrator when complete.

## 2026-09-17T17:01:27Z
You are M2 Explorer 1 for Phase 2: Swift/Metal Execution Pipeline.
Your assigned working directory is: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m2_1
Read your dispatch assignment at: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m2_1/DISPATCH.md
Read the authoritative user requirements at: /Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md (Requirement R2)
Read Phase 2 architecture at: /Users/jack/Downloads/rlcd-router/.agents/orchestrator_phase2/PROJECT.md

Task:
Design the technical architecture and Swift implementation blueprint for the Speculative Ring Buffer Pool (Sources/AsyncMoERouter/BufferPools/SpeculativeRingBuffer.swift):
1. Fixed array of 16 MTLBuffer slots in .storageModeShared.
2. Slot lifecycle state machine (.free, .prefetching, .ready, .abandoned, .inUse) with OSAllocatedUnfairLock thread safety.
3. Dedicated SyncEvent per slot to prevent cross-slot ticket interference.
4. Slot acquisition, recycling, and dirty slot recovery.
Provide complete, copy-pasteable Swift code blueprints.

Deliverables:
- Maintain progress.md.
- Write report to /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m2_1/handoff.md following the Handoff Protocol.
- Send message to orchestrator when complete.
