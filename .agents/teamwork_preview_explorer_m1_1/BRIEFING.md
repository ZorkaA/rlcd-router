# BRIEFING — 2026-09-17T12:50:00Z

## Mission
Design the concrete SwiftPM package structure and foundational code architecture for Phase 2 Milestone 1: Fast I/O Engine & Dual-Queue Subsystem.

## 🔒 My Identity
- Archetype: teamwork_preview_explorer
- Roles: explorer, investigator, architect
- Working directory: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_1
- Original parent: ce5bc762-f633-465c-9133-7ec43d0b5719
- Milestone: M1 (Data Partitioning & Generation)
- Phase 2 Parent: 913b8328-6b64-4881-a075-c0057bc23d84
- Phase 2 Milestone: Phase 2 Milestone 1 (Fast I/O Engine & Dual-Queue Subsystem)

## 🔒 Key Constraints
- Read-only investigation — do NOT implement directly in src/
- Only write metadata, reports, and blueprints within .agents/teamwork_preview_explorer_m1_1/
- No bfloat16 on MPS (PyTorch 2.2.2 MPS crash workaround: use torch.float16)
- Model forward pass must configure output_hidden_states=True, output_router_logits=True, use_cache=False
- Zero-download synthetic model fixture required for testing and CI
- [Phase 2] Read-only investigation — do NOT modify root Sources/ directly; provide concrete copy-pasteable blueprints in handoff.md and proposed files in own working directory
- [Phase 2] SwiftPM test directory must be named `swift_tests/AsyncMoERouterTests` to avoid APFS case-insensitivity conflict with Python `tests/`
- [Phase 2] Metal shaders must be compiled at runtime via `device.makeLibrary(source:options:)` (runtime MSL compilation) to eliminate dependency on missing offline CLI `metal` toolchain
- [Phase 2] Target macOS 14.0+ / Swift 5.9+ / Metal 3
- [Phase 2] Strict memory ceiling: overall pipeline dedicated memory constrained to ~1.22 GB max

## Current Parent
- Conversation ID: 913b8328-6b64-4881-a075-c0057bc23d84
- Updated: 2026-09-17T12:50:00Z

## Investigation State
- **Explored paths**:
  - `ORIGINAL_REQUEST.md`: Phase 2 R1 & R2 Fast I/O, ring buffer, fallback pool, zero-CPU sync.
  - `.agents/orchestrator_phase2/PROJECT.md`: Milestones, feature inventory, M1 ↔ M2 contract.
  - Survey 1 & 2 findings: Apple M3 Max, Swift 6.4, Metal 3 Fast I/O headers, priority queues.
  - Empirical verification: verified runtime MSL compilation, dual Fast I/O queues, zero-CPU `MTLSharedEvent` hardware sync, 32-byte `ExecutionLogEntry`, and defensive bounds checks.
- **Key findings**:
  1. `MTLIOCommandQueueDescriptor` provides `priority = .low` (`speculativeQueue`) and `priority = .high` (`fallbackQueue`), with `maxCommandBufferCount = 16` and `type = .concurrent`.
  2. `MTLIOStatus` enum in Swift uses `.complete` (not `.completed`).
  3. `MTLIOCommandQueue` does not expose `.priority` as a runtime getter; priority must be tested via queue descriptor configuration.
  4. FP16 expert weights for `Qwen1.5-MoE-A2.7B` are exactly 17,301,504 bytes (1056 x 16 KB pages), naturally page-aligned for direct NVMe DMA.
  5. `ExecutionLogEntry` has exact stride 32 bytes and alignment 8.
  6. Apple Silicon Metal 3 zero-CPU GPU-IO synchronization verified: `ioCmd.signalEvent` -> `computeCmd.encodeWaitForEvent` passes data with zero CPU polling or context switching.
- **Unexplored areas**: None for Milestone 1 scope. Fully verified and tested.

## Key Decisions Made
- `Package.swift`: Designed SwiftPM manifest with custom paths `Sources/AsyncMoERouter` and `swift_tests/AsyncMoERouterTests`.
- `Config.swift`: Implemented `MoEArchitectureConfig` with `.qwen15MoEA27B` and `.synthetic`, and `MemoryBudgetConfig` with alignment utilities.
- `MetalContext.swift`: Implemented thread-safe runtime MSL compilation manager with `OSAllocatedUnfairLock` caching.
- `Types.swift`: Implemented `ExpertKey`, `SlotState`, `RingBufferSlot`, `ExecutionLogEntry`, and `FastIOError`.
- `FastIOEngine.swift`: Implemented `FastIOEngineProtocol` with `speculativeQueue` (PriorityLow, max 16) and `fallbackQueue` (PriorityHigh, max 16).
- `WeightFileHandle.swift` & `SyncEvent.swift`: Implemented clean zero-copy wrappers for `MTLIOFileHandle` and `MTLSharedEvent`.
- All 7 verification stages empirically validated.

## Artifact Index
- `DISPATCH.md` — Task log and status updates from orchestrator
- `BRIEFING.md` — Working memory and situational awareness
- `progress.md` — Liveness heartbeat and completed steps
- `proposed_Package.swift` — Blueprint for Package.swift
- `proposed_Config.swift` — Blueprint for Sources/AsyncMoERouter/Common/Config.swift
- `proposed_MetalContext.swift` — Blueprint for Sources/AsyncMoERouter/Common/MetalContext.swift
- `proposed_Types.swift` — Blueprint for Sources/AsyncMoERouter/Common/Types.swift
- `proposed_FastIOEngine.swift` — Blueprint for Sources/AsyncMoERouter/FastIO/FastIOEngine.swift
- `proposed_WeightFileHandle.swift` — Blueprint for Sources/AsyncMoERouter/FastIO/WeightFileHandle.swift
- `proposed_SyncEvent.swift` — Blueprint for Sources/AsyncMoERouter/FastIO/SyncEvent.swift
- `proposed_TestHelpers.swift` — Blueprint for swift_tests/AsyncMoERouterTests/Common/TestHelpers.swift
- `proposed_FastIOTests.swift` — Blueprint for swift_tests/AsyncMoERouterTests/Unit/FastIOTests.swift
- `handoff.md` — 5-component hard handoff report with copy-pasteable code blueprints
