# Progress — Survey Explorer 1

Last visited: 2026-09-17T12:34:40Z
Status: Completed

## Completed Steps
- [x] Initialized DISPATCH.md, BRIEFING.md, progress.md.
- [x] Surveyed macOS version (macOS 27.2 / Darwin 26.2, Apple M3 Max, arm64).
- [x] Surveyed Swift toolchain (Apple Swift 6.4, SwiftPM 6.4.0-dev, Swift Testing + XCTest).
- [x] Surveyed Metal offline compiler vs runtime compilation:
  - `xcrun -sdk macosx metal` fails because offline Metal Toolchain package is not downloaded in Xcode (`xcodebuild -downloadComponent MetalToolchain`).
  - Runtime Metal Shading Language compilation via `MTLDevice.makeLibrary(source:options:)` SUCCEEDS (Family Apple 7, 8, 9, Metal 3 supported).
  - Verified Swift compute kernel execution via runtime MSL compilation.
- [x] Surveyed Metal 3 Fast I/O and synchronization:
  - `MTLIOCommandQueue` (PriorityLow, maxCommandBufferCount 16, PriorityHigh) tested and working.
  - `MTLIOFileHandle` tested and working.
  - `MTLSharedEvent` zero-CPU synchronization verified between IO command buffer and compute queue.
- [x] Surveyed ICB (Indirect Command Buffer) creation and configuration.
- [x] Surveyed Python environment:
  - Conda base Python 3.10.14 (`torch` 2.2.2 MPS enabled, `safetensors` 0.6.2, `numpy` 1.26.4).
  - PyPI has `mlx` 0.32.2.
- [x] Surveyed MLX-Swift:
  - SPM dependency resolution verified with `mlx-swift` 0.31.6.
  - Metal cache limit API `MLX.GPU.set(cacheLimit: 200 * 1024 * 1024)` verified.
- [x] Surveyed Phase 1 artifacts and data structures:
  - `src/config.py`: Qwen1.5-MoE-A2.7B specs ($d=2048$, 24 layers, 60 experts, 4 active, tap layer 3, deep layers 5..24).
  - `MedusaSpeculativeHead`: 3 linear projection heads ($2048 \to 1200$) predicting layers 5-24 for $T+1..T+3$.
  - `TemperatureGrid`: 2x3 scalar grid serializable to JSON and PT.
  - `safetensors` contract tensors: `hidden_states`, `target_router_logits`, `target_top4_indices`, `valid_mask`.
- [x] Surveyed APFS filesystem case collision risk:
  - Verified `tests` (Python) and `Tests` (Swift default) collide on APFS case-insensitive filesystem.
  - Recommended isolated `swift_tests/AsyncMoERouterTests` path for Swift test target.
- [x] Profiled System Memory & defined conservative budget:
  - 36 GB Unified Memory.
  - Physical memory currently ~35GB used with ~8.9GB in memory compressor; 0MB swap used.
  - Conservative buffer ceiling established: 1.22 GB total (512 MB Speculative Ring Buffer + 500 MB Fallback Pool + 200 MB MLX Cache + 10 MB Logs/Flags).
- [x] Produced comprehensive 5-component handoff report in `handoff.md`.

## Current Step
- Sending final summary message to orchestrator.
