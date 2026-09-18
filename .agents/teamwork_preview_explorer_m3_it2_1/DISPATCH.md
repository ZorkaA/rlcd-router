# Dispatch: Milestone 3 Iteration 2 Explorer 1 (LRUWeightTracker Monotonic Queue Remediation)

**Assigned Directory**: `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m3_it2_1`  
**Parent Orchestrator**: `913b8328-6b64-4881-a075-c0057bc23d84`  
**Milestone**: Phase 2 Milestone 3 (GPU Execution Log & Dispatch-Time LRU — Requirement R3)  
**Date**: 2026-09-18  

## Authoritative Inputs
- User Requirements: `/Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md`
- Phase 2 Project Architecture: `/Users/jack/Downloads/rlcd-router/.agents/orchestrator_phase2/PROJECT.md`
- Full Forensic Auditor Evidence Report: `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_auditor_m3_1/handoff.md`
- Challenger 1 Adversarial Report: `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_challenger_m3_1/handoff.md`
- Current Implementation: `/Users/jack/Downloads/rlcd-router/Sources/AsyncMoERouter/ExecutionPipeline/LRUWeightTracker.swift`
- Adversarial Test File: `/Users/jack/Downloads/rlcd-router/swift_tests/AsyncMoERouterTests/Unit/ExecutionLogLRUAdversarialTests.swift`

## Problem Statement
In Milestone 3 Iteration 1, the Forensic Auditor issued an `INTEGRITY VIOLATION` veto and Challenger 1 issued `REQUEST_CHANGES` because `LRUWeightTracker.swift` corrupts the recency queue when out-of-order entries arrive.
In `touch(expertKey:timestamp:slotIndex:)` (lines 95-108):
```swift
if let node = state.map[expertKey] {
    // Monotonic guard: preserve later timestamp if an older entry arrives out-of-order
    if timestamp >= node.timestamp {
        node.timestamp = timestamp
    }
    if let slot = slotIndex {
        node.slotIndex = slot
    }
    Self._unlink(node: node)
    Self._insertAfterHead(node: node, head: state.head)
}
```
`_unlink` and `_insertAfterHead` execute unconditionally even when `timestamp < node.timestamp`. A stale log entry from the past promotes the node to MRU head and forces active experts to the LRU tail for eviction.

## Tasks
1. Read the authoritative inputs and analyze `LRUWeightTracker.swift`.
2. Author an exact, production-grade blueprint for `LRUWeightTracker.swift` fixing `touch(expertKey:timestamp:slotIndex:)`:
   - Only unlink and promote to `state.head` if `timestamp >= node.timestamp`.
   - If `timestamp < node.timestamp`, do NOT move the node in the doubly-linked list. If `node.slotIndex == nil, let slot = slotIndex`, bind the slot, but preserve recency.
   - Verify all existing methods: `recordAccess`, `drainAndRecord`, `evictionCandidate`, `evictLRUWithSlot`, `bindSlot`, `unbindSlot`.
3. Provide concrete code blueprint and verification steps in your handoff report.
4. Output your report to `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m3_it2_1/handoff.md`.

## 2026-09-18T02:35:11Z
You are Explorer 1 for Phase 2 Milestone 3 Iteration 2 (GPU Execution Log & Dispatch-Time LRU Remediation).
Your assigned working directory is: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m3_it2_1
Read your dispatch assignment at: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m3_it2_1/DISPATCH.md
Read the authoritative user requirements at: /Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md
Read Phase 2 architecture at: /Users/jack/Downloads/rlcd-router/.agents/orchestrator_phase2/PROJECT.md
Read the full Forensic Auditor Evidence Report at: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_auditor_m3_1/handoff.md
Read Challenger 1 Adversarial Report at: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_challenger_m3_1/handoff.md

Your focus is `Sources/AsyncMoERouter/ExecutionPipeline/LRUWeightTracker.swift`:
1. Analyze the exact failure reproduced in `ExecutionLogLRUAdversarialTests.swift:477` (Test 10):
   In `touch(expertKey:timestamp:slotIndex:)`, when `timestamp < node.timestamp` (an older log entry arrives out of order), `_unlink(node)` and `_insertAfterHead(node, head: state.head)` execute unconditionally, promoting the stale expert to MRU head and forcing recently used experts to the LRU eviction victim position.
2. Formulate the exact fix: only unlink and insert after head when `timestamp >= node.timestamp`. If `timestamp < node.timestamp`, preserve the node's position in the recency queue; update `node.slotIndex` only if `node.slotIndex == nil`.
3. Author a complete, production-ready Swift code blueprint for `LRUWeightTracker.swift`.
4. Write your 5-component handoff report (Observation, Logic Chain, Caveats, Conclusion, Verification Method) with full blueprint to:
   /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m3_it2_1/handoff.md
5. Update progress.md and send a completion message to the orchestrator.
