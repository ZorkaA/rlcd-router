# BRIEFING — 2026-09-17T17:08:12Z

## Mission
Implement and verify Phase 2 Milestone 2: Speculative Ring Buffer, 500MB Fallback Pool, and Deadlock Resolver for Requirement R2.

## 🔒 My Identity
- Archetype: Worker
- Roles: implementer, qa, specialist
- Working directory: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_worker_m2_1
- Original parent: 913b8328-6b64-4881-a075-c0057bc23d84
- Milestone: Phase 2 Milestone 2 (Ring Buffer Pool & Fallback Pool — Requirement R2)

## 🔒 Key Constraints
- Strict 500MB hard ceiling (524,288,000 bytes) for FallbackBufferPool with zero-tolerance invariant.
- FallbackBufferPool strictly isolated: rejects .speculative requests with FallbackPoolError.speculativeRequestRejected.
- Dedicated SyncEvent per slot in SpeculativeRingBuffer to eliminate cross-slot ticket interference.
- Fixed array of 16 MTLBuffer slots in .storageModeShared.
- 5-state lifecycle: .free, .prefetching, .ready, .inUse, .abandoned.
- Dispatch-time LRU timestamps updated only via GPU Execution Log.
- DeadlockResolver: emergency demand fetch on PriorityHigh fallbackQueue, speculative slot marked .abandoned, completedHandler signal dropped.
- Zero leaks, thread-safe, 100% tests passing on swift test.
- Proactively commit milestone via git add . && git commit -m "feat(phase2-m2): Implement Speculative Ring Buffer and 500MB Fallback Pool".

## Current Parent
- Conversation ID: 913b8328-6b64-4881-a075-c0057bc23d84
- Updated: 2026-09-17T17:08:12Z

## Task Summary
- **What to build**:
  - `Sources/AsyncMoERouter/BufferPools/SpeculativeRingBuffer.swift`
  - `Sources/AsyncMoERouter/BufferPools/FallbackBufferPool.swift`
  - `Sources/AsyncMoERouter/BufferPools/DeadlockResolver.swift`
  - `swift_tests/AsyncMoERouterTests/Unit/BufferPoolTests.swift`
- **Success criteria**: Zero build errors, zero warnings, 100% tests passing (FastIOTests & BufferPoolTests), zero leaks, git committed.
- **Interface contracts**: /Users/jack/Downloads/rlcd-router/.agents/orchestrator_phase2/PROJECT.md
- **Code layout**: Sources/AsyncMoERouter/BufferPools/, swift_tests/AsyncMoERouterTests/Unit/

## Key Decisions Made
- Adopting explorer and spec-miner blueprints directly as verified designs.

## Artifact Index
- Sources/AsyncMoERouter/BufferPools/SpeculativeRingBuffer.swift
- Sources/AsyncMoERouter/BufferPools/FallbackBufferPool.swift
- Sources/AsyncMoERouter/BufferPools/DeadlockResolver.swift
- swift_tests/AsyncMoERouterTests/Unit/BufferPoolTests.swift
- .agents/teamwork_preview_worker_m2_1/handoff.md

## Change Tracker
- **Files modified**: [TBD]
- **Build status**: [TBD]
- **Pending issues**: None

## Quality Status
- **Build/test result**: [TBD]
- **Lint status**: 0 violations
- **Tests added/modified**: [TBD]

## Loaded Skills
- None specified.
