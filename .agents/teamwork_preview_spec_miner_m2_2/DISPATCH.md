## 2026-09-17T17:01:27Z

# DISPATCH: Milestone 2 Spec Miner 2 (Isolated 500MB Fallback Buffer Pool)

## Assigned Working Directory
/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_spec_miner_m2_2

## Task Objective
Design the exact specification and Swift implementation blueprint for the strictly isolated 500MB Fallback Buffer Pool (`Sources/AsyncMoERouter/BufferPools/FallbackBufferPool.swift`).
Key areas:
1. Strict 500MB Memory Ceiling: Exactly $524,288,000$ bytes ($500 \times 1024 \times 1024$). Invariant: Total allocated fallback buffer capacity MUST NEVER exceed 500MB under any condition.
2. Strict Isolation from Speculation: Pool must reject any speculative prefetch requests. It is reserved exclusively for demand-fetch cache misses.
3. Buffer Lifecycle & Reuse: Allocation, tracking in-use buffers, and returning to pool when compute completes without memory leaks or fragmentation.
4. Provide a complete, production-grade Swift code blueprint for `FallbackBufferPool.swift`.

## Authoritative References
- Requirements: `/Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md` (Phase 2 section, R2)
- Architecture & Contracts: `/Users/jack/Downloads/rlcd-router/.agents/orchestrator_phase2/PROJECT.md`
- Survey 2 findings: `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_survey_2/handoff.md`

## Deliverables
- Maintain `progress.md`.
- Write your specification report to `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_spec_miner_m2_2/handoff.md`.
- Send message to orchestrator when complete.
