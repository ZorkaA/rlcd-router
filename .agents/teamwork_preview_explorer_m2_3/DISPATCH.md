# DISPATCH: Milestone 2 Explorer 3 (Deadlock Resolution Protocol & Unit Test Harness)

## Assigned Working Directory
/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m2_3

## Task Objective
Design the Deadlock Resolution Protocol (`Sources/AsyncMoERouter/BufferPools/DeadlockResolver.swift`) and the Milestone 2 Unit Test Harness (`swift_tests/AsyncMoERouterTests/Unit/BufferPoolTests.swift`).
Key areas:
1. Cache-Miss Deadlock Resolution Protocol:
   - On cache-miss or prefetch lag: allocate demand buffer from the 500MB Fallback Pool.
   - Dispatch immediate fetch to `fallbackQueue` (PriorityHigh).
   - Mark the corresponding speculative slot as `.abandoned` (dirty).
   - Invoke `tryCancel()` on the speculative command buffer.
   - When the speculative I/O callback (`completedHandler`) fires, detect `.abandoned`, drop the signal (do not advance event or update router), and return the slot to `.free`.
2. Unit Test Suite (`BufferPoolTests.swift`):
   - Ring buffer slot cycling, wraparound, and state transitions.
   - Fallback pool strict 500MB isolation and capacity enforcement.
   - Cache-miss deadlock simulation: demand load on PriorityHigh, signal dropping on speculative cancellation, slot recovery.
   - 100% memory leak verification.

## Authoritative References
- Requirements: `/Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md` (Phase 2 section, R2)
- Architecture & Contracts: `/Users/jack/Downloads/rlcd-router/.agents/orchestrator_phase2/PROJECT.md`
- Survey 2 findings: `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_survey_2/handoff.md`

## Deliverables
- Maintain `progress.md`.
- Write your comprehensive plan to `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m2_3/handoff.md`.
- Send message to orchestrator when complete.

## 2026-09-17T17:01:27Z
You are M2 Explorer 3 for Phase 2: Swift/Metal Execution Pipeline.
Your assigned working directory is: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m2_3
Read your dispatch assignment at: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m2_3/DISPATCH.md
Read the authoritative user requirements at: /Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md (Requirement R2)
Read Phase 2 architecture at: /Users/jack/Downloads/rlcd-router/.agents/orchestrator_phase2/PROJECT.md

Task:
Design the Deadlock Resolution Protocol (Sources/AsyncMoERouter/BufferPools/DeadlockResolver.swift) and Milestone 2 Unit Test Suite (swift_tests/AsyncMoERouterTests/Unit/BufferPoolTests.swift):
1. Cache-miss deadlock resolution protocol: allocate from Fallback Pool, dispatch to fallbackQueue (PriorityHigh), mark speculative slot as .abandoned (dirty), invoke tryCancel(), and drop signal in completedHandler.
2. Design automated unit tests in BufferPoolTests.swift covering ring buffer cycling, fallback 500MB ceiling, deadlock resolution, signal dropping, and zero memory leaks.
Provide complete, copy-pasteable Swift code blueprints.

Deliverables:
- Maintain progress.md.
- Write report to /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m2_3/handoff.md following the Handoff Protocol.
- Send message to orchestrator when complete.
