# DISPATCH: M1 Explorer 3 (Zero-CPU Synchronization & Test Harness Design)

## Assigned Working Directory
/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_3

## Task Objective
Design zero-CPU GPU-IO synchronization via `MTLSharedEvent` and the M1 automated test harness.
Key areas:
1. `MTLSharedEvent` Mechanics:
   - Creation via `device.makeSharedEvent()`.
   - Signaling from `MTLIOCommandBuffer`: `ioCmd.signalEvent(sharedEvent, value: N)`.
   - Waiting on GPU Compute Queue: `computeCmd.encodeWaitForEvent(sharedEvent, value: N)`.
   - CPU-side non-blocking query (`sharedEvent.signaledValue`) vs zero-CPU GPU hardware wait.
   - Signal value lifecycle management (monotonic counter vs per-slot values).
2. Test Harness & Test Cases (`swift_tests/AsyncMoERouterTests/Unit/FastIOTests.swift`):
   - Unit tests verifying:
     - `speculativeQueue` initialized with priority `.low` and maxCommandBufferCount 16.
     - `fallbackQueue` initialized with priority `.high`.
     - `MTLIOFileHandle` successfully loads bytes/buffers into `MTLBuffer`.
     - `MTLSharedEvent` correctly coordinates between IO queue and compute queue without CPU sleeps or stalls.
     - Test fixtures (synthetic weight file generator creating binary test blocks).

## Authoritative References
- Requirements: `/Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md` (Phase 2 section)
- Architecture & Inventory: `/Users/jack/Downloads/rlcd-router/.agents/orchestrator_phase2/PROJECT.md`
- Survey 1 & 2 findings in `.agents/teamwork_preview_explorer_survey_1` and `survey_2`.

## Deliverables
- Keep `progress.md` updated.
- Write your comprehensive plan to `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_3/handoff.md`.
- Notify orchestrator via `send_message` when complete.

## 2026-09-17T12:39:22Z
User Request:
You are M1 Explorer 3 for Phase 2: Swift/Metal Execution Pipeline.
Your assigned working directory is: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_3
Read your dispatch assignment at: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_3/DISPATCH.md
Read the authoritative user requirements at: /Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md
Read Phase 2 architecture at: /Users/jack/Downloads/rlcd-router/.agents/orchestrator_phase2/PROJECT.md

Task:
Design zero-CPU GPU-IO synchronization via MTLSharedEvent and the automated unit test harness for Milestone 1:
1. MTLSharedEvent mechanics: creation, signaling from MTLIOCommandBuffer, waiting on MTLCommandBuffer (encodeWaitForEvent), non-blocking queries, and value lifecycle.
2. Design FastIOTests.swift in swift_tests/AsyncMoERouterTests/Unit/ with comprehensive unit tests for dual queues, block loading, and hardware synchronization.
3. Design synthetic test fixtures (binary weight file generator) for isolated testing.

Deliverables:
- Maintain progress.md.
- Write your report to /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_3/handoff.md following the Handoff Protocol.
- Send message to orchestrator when complete.

