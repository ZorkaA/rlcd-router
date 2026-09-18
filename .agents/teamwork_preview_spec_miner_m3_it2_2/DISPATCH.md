# Dispatch: Milestone 3 Iteration 2 Spec Miner 2 (ExecutionLog Slot Indexing & Drain Deadlock Remediation)

**Assigned Directory**: `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_spec_miner_m3_it2_2`  
**Parent Orchestrator**: `913b8328-6b64-4881-a075-c0057bc23d84`  
**Milestone**: Phase 2 Milestone 3 (GPU Execution Log & Dispatch-Time LRU — Requirement R3)  
**Date**: 2026-09-18  

## Authoritative Inputs
- User Requirements: `/Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md`
- Phase 2 Project Architecture: `/Users/jack/Downloads/rlcd-router/.agents/orchestrator_phase2/PROJECT.md`
- Full Forensic Auditor Evidence Report: `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_auditor_m3_1/handoff.md`
- Challenger 1 Adversarial Report: `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_challenger_m3_1/handoff.md`
- Current Implementation: `/Users/jack/Downloads/rlcd-router/Sources/AsyncMoERouter/ExecutionPipeline/ExecutionLog.swift`
- Adversarial Test File: `/Users/jack/Downloads/rlcd-router/swift_tests/AsyncMoERouterTests/Unit/ExecutionLogLRUAdversarialTests.swift`

## Problem Statement
In `ExecutionLog.swift`, `writeExecutionLogEntry` MSL shader computes:
`slot = (tokenIndex * 20u * 4u + layerIndex * 4u + horizonIndex) & (logCapacity - 1u);`
For `tokenIndex = 0, layerIndex = 5, horizonIndex = 1`, this writes to slot 21. Slots 0..20 are empty zeros.
When `drain()` executes on the CPU starting at `readHead = 0`:
```swift
while scanned < capacity {
    let slot = (state.readHead + scanned) & mask
    let entry = ptr[slot]
    guard entry.tokenIndex != 0 || entry.timestamp != 0 || entry.confidenceScore != 0 else {
        break
    }
    entries.append(entry)
    ...
```
The sentinel guard immediately triggers `break` at slot 0 and returns 0 entries, leaving `readHead` locked at 0. Valid entries at slot 21 are permanently stranded and dropped.

## Tasks
1. Investigate the slot indexing math and drain protocol across all 3 kernels in `ExecutionLog.swift`:
   - `writeExecutionLogEntry`
   - `writeTokenGatingLog`
   - `writeExecutionLogBatch`
2. Formulate the cleanest, most robust architectural reconciliation:
   - Ensure zero GPU atomics (`atomic_fetch_add_explicit` is strictly prohibited).
   - Ensure single-cycle bitwise masking `& (logCapacity - 1u)`.
   - Ensure that any entries written by ANY kernel can be drained by `drain()`.
   - Investigate whether `drain()` should scan the full buffer (e.g., scanning all `capacity` slots, or maintaining a circular write watermark / pointer, or using `continue` to skip empty sentinel slots while collecting all non-empty entries and zeroing them out).
3. Author an exact, production-grade Swift and Metal blueprint for `ExecutionLog.swift`.
4. Output your report to `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_spec_miner_m3_it2_2/handoff.md`.

## 2026-09-18T02:35:11Z
<USER_REQUEST>
You are Spec Miner 2 for Phase 2 Milestone 3 Iteration 2 (Execution Log Slot Indexing & Drain Remediation).
Your assigned working directory is: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_spec_miner_m3_it2_2
Read your dispatch assignment at: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_spec_miner_m3_it2_2/DISPATCH.md
Read the authoritative user requirements at: /Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md
Read Phase 2 architecture at: /Users/jack/Downloads/rlcd-router/.agents/orchestrator_phase2/PROJECT.md
Read the full Forensic Auditor Evidence Report at: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_auditor_m3_1/handoff.md
Read Challenger 1 Adversarial Report at: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_challenger_m3_1/handoff.md

Your focus is `Sources/AsyncMoERouter/ExecutionPipeline/ExecutionLog.swift`:
1. Analyze the exact failure reproduced in `ExecutionLogLRUAdversarialTests.swift:509` (Test 11):
   `writeExecutionLogEntry` in MSL computes `slot = (tokenIndex * 20u * 4u + layerIndex * 4u + horizonIndex) & mask`. For token 0, layer 5, horizon 1, slot = 21. Slots 0..20 are zeroed.
   In `drain()`, the scanning loop breaks on the very first zeroed slot (slot 0), returning 0 entries and leaving `readHead` locked at 0. Entries written at slot 21 are permanently lost and dropped.
2. Investigate the cleanest zero-atomic, high-performance resolution:
   - Zero atomics: `atomic_fetch_add_explicit` is strictly forbidden.
   - Investigate how `drain()` should consume entries: e.g. scanning across all `capacity` slots using `continue` instead of `break` to collect all non-empty entries and zero them out, or maintaining a circular streaming slot index formula.
3. Author a complete, production-ready Swift and Metal code blueprint for `ExecutionLog.swift`.
4. Write your 5-component handoff report (Observation, Logic Chain, Caveats, Conclusion, Verification Method) with full blueprint to:
   /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_spec_miner_m3_it2_2/handoff.md
5. Update progress.md and send a completion message to the orchestrator.
</USER_REQUEST>

