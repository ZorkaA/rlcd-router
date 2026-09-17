# DISPATCH: Survey Explorer 2 (Metal 3 Fast I/O, Queues, Buffer Pools - R1 & R2)

## Assigned Working Directory
/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_survey_2

## Task Objective
Conduct an exhaustive technical survey and architectural specification for Requirements R1 and R2 of Phase 2:
1. Metal 3 Fast I/O Dual-Queue Setup (R1):
   - `speculativeQueue`: `MTLIOCommandQueue` with `PriorityLow` and `maxCommandBufferCount: 16` for external NVMe prefetching.
   - `fallbackQueue`: `MTLIOCommandQueue` with `PriorityHigh` for demand fetches on miss.
   - `MTLIOFileHandle` for explicit block reads (`loadBytes` / `loadBuffer`).
   - `MTLSharedEvent` for zero-CPU GPU-IO synchronization.
2. Ring Buffer Pool & Fallback Pool (R2):
   - Fixed array of `MTLBuffer`s for speculative Ring Buffer (size calculation, alignment, slot tracking).
   - Strictly isolated 500MB Fallback Buffer Pool.
   - Cache-miss deadlock resolution: allocate from Fallback Pool, dispatch to fallbackQueue, mark speculative slot as abandoned/dirty, drop signal when its IO callback fires.
3. Determine exact Metal API signatures, Swift structures, error handling, synchronization invariants, and memory footprint management.

## Authoritative Requirements Reference
Read `/Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md` (Phase 2 section) before starting work.

## Deliverables
Write your comprehensive investigation report to `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_survey_2/handoff.md` and keep `progress.md` updated.
When complete, notify orchestrator via `send_message`.

## 2026-09-17T12:29:50Z
<USER_REQUEST>
You are Survey Explorer 2 for Phase 2: Swift/Metal Execution Pipeline.
Your assigned working directory is: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_survey_2
Read your dispatch assignment at: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_survey_2/DISPATCH.md
Read the authoritative user requirements at: /Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md

Task:
Conduct an exhaustive technical survey and architectural specification for Requirements R1 and R2 of Phase 2:
1. Metal 3 Fast I/O Dual-Queue Setup (R1):
   - `speculativeQueue`: `MTLIOCommandQueue` with PriorityLow and maxCommandBufferCount 16 for external NVMe prefetching.
   - `fallbackQueue`: `MTLIOCommandQueue` with PriorityHigh for demand fetches on miss.
   - `MTLIOFileHandle` for explicit block reads (`loadBytes` / `loadBuffer`).
   - `MTLSharedEvent` for zero-CPU GPU-IO synchronization.
2. Ring Buffer Pool & Fallback Pool (R2):
   - Fixed array of `MTLBuffer`s for speculative Ring Buffer (size calculation, alignment, slot tracking).
   - Strictly isolated 500MB Fallback Buffer Pool.
   - Cache-miss deadlock resolution: allocate from Fallback Pool, dispatch to fallbackQueue, mark speculative slot as abandoned/dirty, drop signal when its IO callback fires.
3. Determine exact Metal API signatures, Swift structures, error handling, synchronization invariants, and memory footprint management.

Deliverables:
- Maintain progress.md in your working directory.
- Write your comprehensive findings to /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_survey_2/handoff.md following the Handoff Protocol.
- When finished, send a message to orchestrator with your summary.
</USER_REQUEST>

