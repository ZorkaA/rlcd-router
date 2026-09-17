# BRIEFING — 2026-09-17T21:05:58+04:00

## Mission
Design the technical architecture and complete Swift implementation blueprint for the Speculative Ring Buffer Pool (Sources/AsyncMoERouter/BufferPools/SpeculativeRingBuffer.swift).

## 🔒 My Identity
- Archetype: explorer
- Roles: explorer, synthesis
- Working directory: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m2_1
- Original parent: 913b8328-6b64-4881-a075-c0057bc23d84
- Milestone: Phase 2 Milestone 2 (M2 Explorer 1 - Speculative Ring Buffer Pool Architecture)

## 🔒 Key Constraints
- Read-only investigation — do NOT implement in source tree, provide blueprint in reports
- Design technical architecture and Swift implementation blueprint for SpeculativeRingBuffer.swift
- Fixed array of 16 MTLBuffer slots in .storageModeShared
- Slot lifecycle state machine (.free, .prefetching, .ready, .abandoned, .inUse) with OSAllocatedUnfairLock thread safety
- Dedicated SyncEvent per slot to prevent cross-slot ticket interference
- Slot acquisition, recycling, and dirty slot recovery
- Self-contained handoff.md with 5 components (Observation, Logic Chain, Caveats, Conclusion, Verification Method)
- .agents/ holds only agent metadata, no source/test/data files

## Current Parent
- Conversation ID: 913b8328-6b64-4881-a075-c0057bc23d84
- Updated: 2026-09-17T21:05:58+04:00

## Investigation State
- **Explored paths**:
  - `ORIGINAL_REQUEST.md` (Requirement R2)
  - `PROJECT.md` (Milestone 2 interface contracts & architecture)
  - `Sources/AsyncMoERouter/BufferPools/SpeculativeRingBuffer.swift`
  - `Sources/AsyncMoERouter/BufferPools/FallbackBufferPool.swift`
  - `Sources/AsyncMoERouter/BufferPools/DeadlockResolver.swift`
  - `Sources/AsyncMoERouter/Common/Types.swift`
  - `Sources/AsyncMoERouter/Common/Config.swift`
  - `Sources/AsyncMoERouter/FastIO/SyncEvent.swift`
  - `Sources/AsyncMoERouter/FastIO/FastIOEngine.swift`
  - `swift_tests/AsyncMoERouterTests/Unit/BufferPoolTests.swift`
  - `swift_tests/AsyncMoERouterTests/Unit/FastIOAdversarialTests.swift`
  - `swift_tests/AsyncMoERouterTests/Unit/FastIOChallenger2StressTests.swift`
  - `swift_tests/AsyncMoERouterTests/E2E/Tier1_FeatureTests.swift`
- **Key findings**:
  - Replaced `NSLock` with `OSAllocatedUnfairLock<PoolState>` to eliminate dynamic Objective-C messaging, heap allocations, and kernel transitions (<15ns uncontended latency).
  - Identified cross-slot ticket interference hazard: if a single `MTLSharedEvent` is shared across slots, out-of-order completion of a higher ticket premature unblocks waiting GPU shaders on dirty buffers. Solved by assigning an independent `SyncEvent` per slot.
  - Sizing confirmed: 16 slots × 17.3 MB = 276.8 MB FP16 (24 KB × 16 = 384 KB synthetic), exactly 1056 16KB pages per FP16 expert.
  - Lifecycle state machine fully formalized (.free -> .loading/.prefetching -> .ready -> .inUse -> .free) with dirty slot recovery via signal dropping in `completeIO`.
- **Unexplored areas**: None for M2 Explorer 1 scope.

## Key Decisions Made
- Encapsulated mutable pool state in `struct PoolState` inside `OSAllocatedUnfairLock<PoolState>`.
- Provided forward-compatible `.prefetching` alias and state helpers on `SlotState`.
- Designed dedicated `SyncEvent` per slot on `RingBufferSlot`.
- Produced copy-pasteable blueprint and patches in `.agents/teamwork_preview_explorer_m2_1/`.

## Artifact Index
- `.agents/teamwork_preview_explorer_m2_1/DISPATCH.md` — Assignment instructions
- `.agents/teamwork_preview_explorer_m2_1/BRIEFING.md` — Working memory and state index
- `.agents/teamwork_preview_explorer_m2_1/progress.md` — Liveness heartbeat and milestone tracking
- `.agents/teamwork_preview_explorer_m2_1/handoff.md` — Authoritative 5-component architectural report
- `.agents/teamwork_preview_explorer_m2_1/proposed_SpeculativeRingBuffer.swift` — Production-grade Swift implementation blueprint
- `.agents/teamwork_preview_explorer_m2_1/SpeculativeRingBuffer.patch` — Unified diff patch for SpeculativeRingBuffer.swift
- `.agents/teamwork_preview_explorer_m2_1/Types.patch` — Unified diff patch for Types.swift
