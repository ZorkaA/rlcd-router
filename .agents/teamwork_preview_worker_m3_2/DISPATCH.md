# Dispatch: Milestone 3 Iteration 2 Worker (Execution Pipeline Remediation)

**Assigned Directory**: `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_worker_m3_2`  
**Parent Orchestrator**: `913b8328-6b64-4881-a075-c0057bc23d84`  
**Milestone**: Phase 2 Milestone 3 (GPU Execution Log & Dispatch-Time LRU — Requirement R3) Iteration 2  
**Date**: 2026-09-18  

## Mandatory Integrity Warning
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

## Authoritative Inputs
- User Requirements: `/Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md`
- Phase 2 Project Architecture: `/Users/jack/Downloads/rlcd-router/.agents/orchestrator_phase2/PROJECT.md`
- Forensic Auditor Evidence Report: `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_auditor_m3_1/handoff.md`
- Explorer 1 Blueprint & Patch (`LRUWeightTracker.swift`):
  `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m3_it2_1/handoff.md`
  `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m3_it2_1/proposed_LRUWeightTracker.swift`
  `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m3_it2_1/LRUWeightTracker_monotonic_guard.patch`
- Spec Miner 2 Blueprint (`ExecutionLog.swift`):
  `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_spec_miner_m3_it2_2/handoff.md`
- Explorer 3 Blueprint (`ExecutionLogTests.swift`):
  `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m3_it2_3/handoff.md`
- Challenger 1 Adversarial Tests:
  `/Users/jack/Downloads/rlcd-router/swift_tests/AsyncMoERouterTests/Unit/ExecutionLogLRUAdversarialTests.swift`
- Challenger 2 Stress Tests:
  `/Users/jack/Downloads/rlcd-router/swift_tests/AsyncMoERouterTests/Unit/ExecutionLogChallenger2StressTests.swift`

## File Ownership
You have exclusive write ownership of:
1. `Sources/AsyncMoERouter/ExecutionPipeline/LRUWeightTracker.swift`
2. `Sources/AsyncMoERouter/ExecutionPipeline/ExecutionLog.swift`
3. `swift_tests/AsyncMoERouterTests/Unit/ExecutionLogTests.swift`

## Remediation Tasks

### 1. Remediate `LRUWeightTracker.swift`
Apply the Explorer 1 blueprint to `Sources/AsyncMoERouter/ExecutionPipeline/LRUWeightTracker.swift`:
In `touch(expertKey:timestamp:slotIndex:)`:
```swift
_state.withLock { state in
    state.totalUpdates += 1

    if let node = state.map[expertKey] {
        if timestamp >= node.timestamp {
            node.timestamp = timestamp
            if let slot = slotIndex {
                node.slotIndex = slot
            }
            Self._unlink(node: node)
            Self._insertAfterHead(node: node, head: state.head)
        } else {
            if node.slotIndex == nil, let slot = slotIndex {
                node.slotIndex = slot
            }
        }
    } else {
        let node = Node(key: expertKey, timestamp: timestamp, slotIndex: slotIndex)
        state.map[expertKey] = node
        Self._insertAfterHead(node: node, head: state.head)
    }
}
```

### 2. Remediate `ExecutionLog.swift`
Apply the Spec Miner 2 blueprint to `Sources/AsyncMoERouter/ExecutionPipeline/ExecutionLog.swift`:
Update `drain()` from an early-terminating loop (`break`) into a circular sweep scanner (`continue`) over all `capacity` slots starting at `readHead`.
- Scan all `capacity` slots using `let slot = (state.readHead + scanned) & mask`.
- Check sentinel:
  ```swift
  let isOccupied = entry.tokenIndex != 0 || entry.timestamp != 0 || entry.confidenceScore != 0 || entry.layerIndex != 0 || entry.horizonIndex != 0 || entry.expertID != 0
  guard isOccupied else {
      scanned += 1
      continue
  }
  ```
- Collect occupied entry, zero the slot `ptr[slot] = ExecutionLogEntry()`.
- Track `lastOccupiedOffset = scanned`.
- After sweep, advance `readHead`: if any entries drained, `state.readHead = (state.readHead + lastOccupiedOffset + 1) & mask`.
- Ensure zero atomics (`atomic_fetch_add_explicit` is strictly forbidden).

### 3. Remediate `ExecutionLogTests.swift`
Apply the Explorer 3 blueprint to `swift_tests/AsyncMoERouterTests/Unit/ExecutionLogTests.swift`:
- Update `testExecutionLogMSLSource` to dynamically compile `GPUExecutionLog.mslKernelSource` at runtime via `device.makeLibrary(source:options:)` and verify compute pipeline state creation for `writeExecutionLogEntry`, `writeTokenGatingLog`, `writeExecutionLogBatch`.
- Add direct unit tests executing the production `writeExecutionLogEntry` kernel on GPU and asserting that `drain()` extracts the exact entry.

### 4. Build, Test & Commit
- Run `swift build` and fix any compiler warnings.
- Run `swift test` across all targets, verifying 100% pass across all suites including:
  - `ExecutionLogTests`
  - `ExecutionLogLRUAdversarialTests` (Test 10 and Test 11 MUST PASS)
  - `ExecutionLogChallenger2StressTests`
  - `BufferPoolTests`
  - `FastIOTests`
- Run `git add . && git commit -m "fix(phase2-m3): Resolve recency queue inversion, sparse log drain stall, and genuine MSL compilation"` per user global rules.

### 5. Report
Write your 5-component handoff report (Observation, Logic Chain, Caveats, Conclusion, Verification Method) with verbatim terminal outputs to:
`/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_worker_m3_2/handoff.md`

## 2026-09-18T02:41:21Z
You are the Worker for Phase 2 Milestone 3 Iteration 2 (Execution Pipeline Remediation).
Your assigned working directory is: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_worker_m3_2
Read your dispatch assignment at: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_worker_m3_2/DISPATCH.md
Read the authoritative user requirements at: /Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md
Read Phase 2 architecture at: /Users/jack/Downloads/rlcd-router/.agents/orchestrator_phase2/PROJECT.md

