# Progress — Challenger 1 (Milestone 3)

Last visited: 2026-09-18T02:31:00Z
Status: Adversarial verification complete — 2 empirical bugs reproduced and documented; verdict REQUEST_CHANGES.

## Completed Steps
- Read dispatch assignment, requirements, and worker handoff report.
- Reviewed implementation in `LRUWeightTracker.swift`, `ExecutionLog.swift`, and `SpeculativeRingBuffer.swift`.
- Designed and authored comprehensive adversarial test suite in `swift_tests/AsyncMoERouterTests/Unit/ExecutionLogLRUAdversarialTests.swift` covering:
  1. Speculative pre-routing lifecycle isolation (timestamps remain 0).
  2. Speculative slot abandonment and dirty recovery isolation.
  3. Eviction ordering priority (unexecuted slots ts=0 evicted before executed slots ts>0).
  4. Post-execution drain exclusivity and idempotence.
  5. Monotonic timestamp numerical non-regression.
  6. Out-of-order recency queue ordering.
  7. Duplicate token ingestion and cycle safety.
  8. Reversed timestamp ingestion sequence.
  9. MTLCommandBuffer completion drain handler lifecycle.
  10. Empirical challenge: Out-of-order recency queue corruption.
  11. Empirical challenge: Non-sequential sparse slot drain deadlock.
- Executed empirical test harness via `swift test --filter ExecutionLogLRUAdversarialTests`.
- Reproduced 2 critical defects empirically:
  1. Monotonic timestamp guard in `LRUWeightTracker.touch` executes `_unlink` and `_insertAfterHead` unconditionally, promoting older out-of-order entries to MRU head and corrupting the LRU queue.
  2. Sparse slot indexing in `writeExecutionLogEntry` (`slot = tokenIndex * 80 + layerIndex * 4 + horizonIndex`) creates gaps at slots 0..20, causing `drain()` sentinel check (`guard ... else { break }`) to abort at slot 0 and permanently fail to drain written entries.

## Current Step
- Writing handoff report with verdict REQUEST_CHANGES and notifying orchestrator.
