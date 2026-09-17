# BRIEFING — 2026-09-17T12:54:00Z

## Mission
Implement Milestone 1 (Fast I/O Engine & Dual-Queue Subsystem) for Phase 2: Swift/Metal Execution Pipeline. Create Package.swift, Config.swift, MetalContext.swift, Types.swift, FastIOEngine.swift, WeightFileHandle.swift, SyncEvent.swift, TestHelpers.swift, and FastIOTests.swift, and verify 100% test pass.

## 🔒 My Identity
- Archetype: teamwork_preview_worker
- Roles: implementer, qa, specialist
- Working directory: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_worker_m1_1
- Original parent: orchestrator (ce5bc762-f633-465c-9133-7ec43d0b5719)
- Milestone: Milestone 1 (Data Partitioning & Generation)
- Phase 2 Parent: orchestrator (913b8328-6b64-4881-a075-c0057bc23d84)
- Phase 2 Milestone: Milestone 1 (Fast I/O Engine & Dual-Queue Subsystem)

## 🔒 Key Constraints
- EXCLUSIVE FILE WRITE OWNERSHIP:
  - /Users/jack/Downloads/rlcd-router/src/__init__.py
  - /Users/jack/Downloads/rlcd-router/src/config.py
  - /Users/jack/Downloads/rlcd-router/src/data/__init__.py
  - /Users/jack/Downloads/rlcd-router/src/data/model_loader.py
  - /Users/jack/Downloads/rlcd-router/src/data/stream_extractor.py
  - /Users/jack/Downloads/rlcd-router/src/data/dataset.py
- DO NOT CHEAT: Genuine implementation, no hardcoding, no mock facades.
- All tests for Features 1-5 must pass cleanly without regressions on existing tests.
- Support Apple Silicon MPS (FP16) and CPU (FP32/BF16); reject BF16 on MPS with clear ValueError.
- Zero-OOM guarantee: B=1, L=1024, torch.inference_mode(), use_cache=False, immediate CPU FP16 offload, periodic cache flushes.
- Sequence-atomic 80/20 train/calibration split with zero context contamination and trailing boundary masking (L-3..L-1).
- Phase 2 Milestone 1 Exclusive Write Ownership:
  - Package.swift
  - Sources/AsyncMoERouter/Common/Config.swift
  - Sources/AsyncMoERouter/Common/MetalContext.swift
  - Sources/AsyncMoERouter/Common/Types.swift
  - Sources/AsyncMoERouter/FastIO/FastIOEngine.swift
  - Sources/AsyncMoERouter/FastIO/WeightFileHandle.swift
  - Sources/AsyncMoERouter/FastIO/SyncEvent.swift
  - swift_tests/AsyncMoERouterTests/Common/TestHelpers.swift
  - swift_tests/AsyncMoERouterTests/Unit/FastIOTests.swift
- DO NOT modify Phase 1 files in src/ or tests/.
- Dual Fast I/O queues: speculativeQueue (PriorityLow, max 16), fallbackQueue (PriorityHigh, max 16).
- Zero-CPU synchronization via MTLSharedEvent.
- Conservative memory footprint (<1.0 GB total pipeline dedicated footprint).

## Current Parent
- Conversation ID: 913b8328-6b64-4881-a075-c0057bc23d84
- Updated: 2026-09-17T12:54:00Z

## Task Summary
- **What to build**: Production-grade Metal 3 Fast I/O dual-queue engine, runtime MSL compilation manager, MTLIOFileHandle block wrapper with 16KB alignment and defensive bounds checking, MTLSharedEvent zero-CPU hardware synchronization, synthetic test generator, and comprehensive automated unit tests.
- **Success criteria**: `swift build` compiles cleanly, `swift test` passes 100% of test cases.
- **Interface contracts**: .agents/orchestrator_phase2/PROJECT.md § M1 ↔ M2.
- **Code layout**: .agents/orchestrator_phase2/PROJECT.md § Code Layout.

## Change Tracker
- **Files modified**:
  - Package.swift (pending)
  - Sources/AsyncMoERouter/Common/Config.swift (pending)
  - Sources/AsyncMoERouter/Common/MetalContext.swift (pending)
  - Sources/AsyncMoERouter/Common/Types.swift (pending)
  - Sources/AsyncMoERouter/FastIO/FastIOEngine.swift (pending)
  - Sources/AsyncMoERouter/FastIO/WeightFileHandle.swift (pending)
  - Sources/AsyncMoERouter/FastIO/SyncEvent.swift (pending)
  - swift_tests/AsyncMoERouterTests/Common/TestHelpers.swift (pending)
  - swift_tests/AsyncMoERouterTests/Unit/FastIOTests.swift (pending)
- **Build status**: Pending implementation.
- **Pending issues**: None.

## Quality Status
- **Build/test result**: Pending.
- **Lint status**: Clean.
- **Tests added/modified**: FastIOTests.swift.

## Loaded Skills
- None required directly.

## Key Decisions Made
- Use runtime MSL compilation (`MTLDevice.makeLibrary(source:)`) in MetalContext to circumvent missing offline Metal CLI toolchain.
- Keep Swift test target in `swift_tests/AsyncMoERouterTests` to avoid APFS case-insensitivity conflict with `tests/`.
- Per-slot SyncEvent architecture to prevent out-of-order race conditions across concurrent expert transfers.
- Defensive client-side bounds checking in WeightFileHandle and FastIOEngine to prevent silent Metal DMA overflow.
- 16KB page alignment matching Apple Silicon hardware page size for peak NVMe DMA throughput.

## Artifact Index
- .agents/teamwork_preview_worker_m1_1/DISPATCH.md — Assignment instructions
- .agents/teamwork_preview_worker_m1_1/BRIEFING.md — Working memory & state
- .agents/teamwork_preview_worker_m1_1/progress.md — Execution & heartbeat log
- .agents/teamwork_preview_worker_m1_1/handoff.md — Final 5-component report

