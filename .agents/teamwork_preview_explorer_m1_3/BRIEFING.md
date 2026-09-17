# BRIEFING — 2026-09-17T12:46:00Z

## Mission
Design zero-CPU GPU-IO synchronization via MTLSharedEvent and automated unit test harness (FastIOTests.swift and synthetic test fixtures) for Milestone 1: Fast I/O Engine & Dual-Queue Subsystem.

## 🔒 My Identity
- Archetype: explorer
- Roles: investigation, specification_design, synthesis
- Working directory: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_3
- Original parent: ce5bc762-f633-465c-9133-7ec43d0b5719
- Milestone: M1 (Data Partitioning & Generation)
- Phase 2 Parent: 913b8328-6b64-4881-a075-c0057bc23d84
- Phase 2 Milestone: Phase 2 Milestone 1 (Zero-CPU GPU-IO Synchronization & Test Harness Design)

## 🔒 Key Constraints
- Read-only investigation — do NOT implement source code directly.
- All investigation outputs go into `.agents/teamwork_preview_explorer_m1_3/`.
- Strict isolation of train vs calibration sequences (zero token/sequence contamination).
- Boundary token masking: for sequences of length $L$, lookahead $T+1..T+3$ cannot cross sequence boundaries ($L-3..L-1$ masked).
- Storage format: `safetensors.torch` (`train_data.safetensors`, `calib_data.safetensors`).
- PyTorch Dataset returning `(hidden_states, target_router_logits, target_top4_indices, valid_mask)`.
- Phase 2: Conservative memory ceiling (<1.5 GB dedicated Metal buffers, 500MB Fallback, 200MB MLX cache).
- Phase 2: Zero CPU stalls or spins during IO-GPU synchronization.
- Phase 2: APFS case collision avoidance (tests vs Tests -> swift_tests/AsyncMoERouterTests).

## Current Parent
- Conversation ID: 913b8328-6b64-4881-a075-c0057bc23d84
- Updated: 2026-09-17T12:46:00Z

## Investigation State
- **Explored paths**:
  - `ORIGINAL_REQUEST.md`, `PROJECT.md`, `DISPATCH.md`.
  - Survey 1 & 2 handoffs (`.agents/teamwork_preview_explorer_survey_1/handoff.md`, `survey_2/handoff.md`).
  - Empirical Metal runtime tests on Apple M3 Max:
    - `MTLSharedEvent` creation and signaledValue property queries (~151 ns).
    - `MTLIOCommandBuffer.signalEvent` to `MTLCommandBuffer.encodeWaitForEvent` zero-CPU hardware wait.
    - Out-of-order race condition discovery when using a single monotonic ticket on a shared event.
    - Signal cancellation (`tryCancel`) dropping signal to 0 and invoking `addCompletedHandler`.
    - `MTLSharedEventListener` notification callbacks.
    - Full 6-case prototype test suite executed with 100% pass rate.
- **Key findings**:
  - `MTLSharedEvent` wait condition is `signaledValue >= waitValue`. If transfers complete out-of-order, a single monotonic ticket would wake up commands prematurely.
  - Per-slot dedicated `MTLSharedEvent` (with generational ticketing per slot) guarantees race-free out-of-order transfers.
  - Non-blocking CPU query via `event.signaledValue` executes in ~151 ns with zero OS locks or kernel context switches.
  - Hardware wait on GPU compute command queue executes completely in GPU Command Processor (CP) with 0.0% CPU overhead.
  - SwiftPM test suite must reside in `swift_tests/AsyncMoERouterTests` to avoid APFS case collision with Python `tests/`.
- **Unexplored areas**: None. Technical specifications and test blueprints complete.

## Key Decisions Made
- Formulated Per-Slot Event Architecture to eliminate out-of-order ticket race hazard in Metal 3 Fast I/O.
- Designed `SyncEvent.swift` wrapper with `OSAllocatedUnfairLock`, non-blocking query, and hardware wait/signal encoders.
- Architected 1.57 MB fast synthetic test fixture in `TestHelpers.swift` with deterministic byte generation ($\text{Byte}(l,e,i) = (l\cdot 17 + e\cdot 31 + i) \pmod{256}$) and bitwise validator.
- Designed 12-case comprehensive unit test suite in `FastIOTests.swift` covering dual queues, DMA reads, page alignment, zero-CPU GPU compute synchronization, cancellation, and repeated load stability.

## Artifact Index
- `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_3/analysis.md` — Detailed technical specification and analysis
- `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_3/handoff.md` — 5-component handoff report
- `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_3/progress.md` — Heartbeat and progress tracker
- `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_3/DISPATCH.md` — Dispatch log


