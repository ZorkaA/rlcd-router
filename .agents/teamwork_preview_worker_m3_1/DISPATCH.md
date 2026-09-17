# Dispatch Assignment: Milestone 3 Worker

**Assigned Agent**: `teamwork_preview_worker_m3_1`  
**Milestone**: Phase 2 Milestone 3 (GPU Execution Log & Dispatch-Time LRU Tracking — Requirement R3)  
**Assigned Working Directory**: `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_worker_m3_1`  
**Date**: 2026-09-18  

---

## Mandatory Integrity Warning
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

---

## 1. Objective
Implement the production-grade Swift/Metal GPU Execution Log and Dispatch-Time LRU Tracking subsystem for Milestone 3:
1. `Sources/AsyncMoERouter/ExecutionPipeline/ExecutionLog.swift` (and `GPUExecutionLog.swift` alias/implementation)
2. `Sources/AsyncMoERouter/ExecutionPipeline/LRUWeightTracker.swift`
3. `swift_tests/AsyncMoERouterTests/Unit/ExecutionLogTests.swift`

---

## 2. Input Specifications & Verified Blueprints
You MUST read and implement the blueprints already prepared and verified by the exploration team:
- **ExecutionLog blueprint**:
  `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m3_1/proposed_ExecutionLog.swift`
  - 32-byte aligned struct `ExecutionLogEntry` in `Sources/AsyncMoERouter/Common/Types.swift` (exact offsets 0, 4, 6, 8, 10, 12, 16, 24).
  - Pre-allocated 4096-entry `MTLBuffer` in `.storageModeShared` ($131,072\text{ bytes} = 128\text{ KB}$).
  - Zero-atomic GPU logging mechanism (`((tokenIndex * topK) + rank) & 4095u` or sequential dispatch offsets).
  - Fix MSL shader defect: MSL on Apple Silicon does not have `clock()`; pass uniform host/dispatch timestamp as kernel parameter.
  - Expose `public typealias GPUExecutionLog = ExecutionLog` for backward compatibility.
- **LRUWeightTracker blueprint**:
  `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_spec_miner_m3_2/handoff.md` (Section 5)
  - Post-execution invariant strictly enforced: LRU timestamps and recency order are updated ONLY by draining the GPU Execution Log, NEVER from speculative pre-routing predictions.
  - `OSAllocatedUnfairLock` thread safety (<15ns lock acquisition).
  - $O(1)$ recency operations (`touch`, `evictionCandidate`, `remove`, `recordAccess`, `lruExpert`).
  - Integration with `SpeculativeRingBuffer.updateLRUTimestamp(for:timestamp:)`.
  - Non-blocking completion handler hook: `registerCompletionDrain(on:executionLog:ringBuffer:onCompletion:)`.
- **ExecutionLogTests blueprint**:
  `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m3_3/proposed_ExecutionLogTests.swift`
  - 15 comprehensive unit tests covering 32-byte layout, circular wraparound, CPU drain, strict R3 invariant proof, concurrent write/drain stress, and runtime MSL compilation.

---

## 3. Scope Boundaries & File Ownership
You exclusively own:
- `Sources/AsyncMoERouter/ExecutionPipeline/ExecutionLog.swift`
- `Sources/AsyncMoERouter/ExecutionPipeline/LRUWeightTracker.swift`
- `swift_tests/AsyncMoERouterTests/Unit/ExecutionLogTests.swift`
- Updates to `Sources/AsyncMoERouter/Common/Types.swift` or `SpeculativeRingBuffer.swift` only as required by the blueprints.

---

## 4. Verification & Git Commit
1. Run `swift build` in `/Users/jack/Downloads/rlcd-router`. Ensure zero errors and zero warnings.
2. Run `swift test --filter ExecutionLogTests` in `/Users/jack/Downloads/rlcd-router`. Ensure all 15 tests pass.
3. Run `swift test` across the full test bundle to verify zero regressions across FastIOTests, BufferPoolTests, etc.
4. Proactively commit per user global rules:
   `git add . && git commit -m "feat(phase2-m3): Implement Zero-Atomic GPU Execution Log and Dispatch-Time LRU Tracker"`
5. Write your comprehensive 5-component handoff report (Observation, Logic Chain, Caveats, Conclusion, Verification Method) to:
   `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_worker_m3_1/handoff.md`
6. Update `progress.md` in your working directory and send a completion message to the orchestrator.
