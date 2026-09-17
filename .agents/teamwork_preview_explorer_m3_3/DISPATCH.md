# Dispatch Assignment: Milestone 3 Explorer 3 (Integration & Unit Test Harness)

**Assigned Agent**: `teamwork_preview_explorer_m3_3`  
**Milestone**: Phase 2 Milestone 3 (GPU Execution Log & Dispatch-Time LRU Tracking — Requirement R3)  
**Assigned Working Directory**: `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m3_3`  
**Date**: 2026-09-18  

---

## 1. Objective
Design the comprehensive automated unit test harness and synthetic GPU verification kernels for Milestone 3:
- Target File: `swift_tests/AsyncMoERouterTests/Unit/ExecutionLogTests.swift`
- Synthetic Shaders: Runtime MSL kernel executing parallel mock gating and log writes to test zero-atomic GPU logging.

---

## 2. Technical Specifications
1. **Unit Test Coverage Requirements**:
   - `testExecutionLogEntryAlignmentAndLayout`: Assert exact 32-byte size and field offsets for `ExecutionLogEntry`.
   - `testZeroAtomicCircularLogBufferWrap`: Test logging 5,000 entries into a 4,096-entry circular buffer, verifying clean circular indexing and zero out-of-bounds writes.
   - `testCPULogDrainPostExecution`: Verify that CPU drains log entries correctly and updates `LRUWeightTracker`.
   - `testLRUUpdatedStrictlyPostExecution`: Verify that speculative pre-routing does NOT modify LRU timestamps, and that only log drain modifies them.
   - `testConcurrentLogWriteAndDrain`: Stress-test simultaneous GPU kernel log writes and CPU drain loops with zero data corruption.
2. **Synthetic GPU Shaders**:
   - Provide concrete MSL string for runtime compilation in `MetalContext.shared`.

---

## 3. Deliverables
- Maintain `progress.md` in your working directory.
- Deliver a comprehensive handoff report with complete code blueprints for `ExecutionLogTests.swift` and synthetic test helpers to:
  `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m3_3/handoff.md`
- Send a completion message to the orchestrator.

## 2026-09-17T21:41:24Z
You are Explorer 3 for Phase 2 Milestone 3: GPU Execution Log & Dispatch-Time LRU Tracking (Requirement R3).
Your assigned working directory is: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m3_3
Read your dispatch assignment at: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m3_3/DISPATCH.md
Read the authoritative user requirements at: /Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md
Read Phase 2 architecture at: /Users/jack/Downloads/rlcd-router/.agents/orchestrator_phase2/PROJECT.md

Tasks:
1. Design the comprehensive automated unit test harness for Milestone 3:
   - File: `swift_tests/AsyncMoERouterTests/Unit/ExecutionLogTests.swift`
   - Test cases:
     - 32-byte layout verification for ExecutionLogEntry.
     - 4,096-entry circular wraparound and indexing math.
     - Post-execution CPU log draining.
     - Strict verification that pre-routing predictions do NOT update LRU timestamps.
     - Concurrent GPU write and CPU drain stress testing.
   - Synthetic MSL gating/logging shaders for runtime execution via MetalContext.shared.
   - Provide complete, compilable Swift code for ExecutionLogTests.swift.
2. Deliver your handoff report to:
   /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m3_3/handoff.md
3. Update progress.md and send a message to the orchestrator when finished.
