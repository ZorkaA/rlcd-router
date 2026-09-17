# BRIEFING — 2026-09-17T12:35:00Z

## Mission
Conduct an exhaustive technical survey and architectural specification for Requirements R1 and R2 of Phase 2 (Metal 3 Fast I/O Dual-Queue Setup, Ring Buffer Pool, Fallback Pool, and Cache-Miss Deadlock Resolution).

## 🔒 My Identity
- Archetype: explorer
- Roles: investigation, synthesis
- Working directory: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_survey_2
- Original parent: 913b8328-6b64-4881-a075-c0057bc23d84
- Milestone: Phase 2 Survey & Architecture Specification

## 🔒 Key Constraints
- Read-only investigation — do NOT modify production code or tests directly
- Adhere strictly to R1 and R2 requirements from ORIGINAL_REQUEST.md
- Ensure conservative memory footprint: Ring Buffer and Fallback Pool sizing must not trigger memory pressure or heavy swapping
- Communicate findings via handoff.md and send_message to caller (913b8328-6b64-4881-a075-c0057bc23d84)

## Current Parent
- Conversation ID: 913b8328-6b64-4881-a075-c0057bc23d84
- Updated: 2026-09-17T12:35:00Z

## Investigation State
- **Explored paths**:
  - Metal 3 Fast I/O Dual-Queue setup (`MTLIOCommandQueueDescriptor`, priorities, buffer count).
  - `MTLIOFileHandle` explicit block reads (`loadBytes` / `loadBuffer`) and alignment rules.
  - Zero-CPU GPU-IO synchronization via `MTLSharedEvent`.
  - Speculative Ring Buffer Pool sizing (16 slots = 276.8 MB) and lifecycle state transitions.
  - Strictly isolated 500MB Fallback Buffer Pool (holding up to 30 expert buffers).
  - Cache-miss deadlock resolution protocol, dirty slot abandonment, and signal dropping.
  - Empirical verification script covering all R1 and R2 requirements.
- **Key findings**:
  - `MTLIOPriorityLow` on `speculativeQueue` with `maxCommandBufferCount = 16` opportunistically prefetches weights without starving high-priority demands.
  - `MTLIOPriorityHigh` on `fallbackQueue` preempts background I/O in the Apple Silicon storage controller.
  - Each FP16 expert is exactly 17,301,504 bytes ($1056 \times 16,384$), perfectly page-aligned to 16KB on Apple Silicon.
  - Zero-CPU synchronization via `MTLSharedEvent` works flawlessly: compute commands wait on hardware events without waking the host CPU.
  - On a cache miss, marking the speculative slot as `.abandoned` and dropping the signal in `addCompletedHandler` cleanly avoids memory corruption and circular deadlocks.
  - Total dedicated Metal buffers for Ring Buffer (276.8MB) + Fallback Pool (500MB) = 776.8MB, leaving >8–10 GB of untouched RAM on the 36GB host.
- **Unexplored areas**: None for R1 and R2 scope. Requirements R1 and R2 are fully surveyed, specified, and verified.

## Key Decisions Made
- Specified a fixed 16-slot Ring Buffer (276.8MB FP16) pre-allocated in `.storageModeShared`.
- Formulated the 500MB Fallback Buffer Pool as strictly isolated, non-borrowable, holding up to 30 expert buffers.
- Defined the complete cache-miss deadlock resolution protocol with dirty slot marking, cooperative `tryCancel()`, and signal dropping on completion.
- Formulated production Swift structures using `OSAllocatedUnfairLock` to prevent actor hopping overhead.

## Artifact Index
- DISPATCH.md — Task assignment from Phase 2 orchestrator
- BRIEFING.md — Persistent state memory
- progress.md — Liveness heartbeat and milestone progress
- analysis.md — Comprehensive technical analysis report
- handoff.md — 5-component handoff report
