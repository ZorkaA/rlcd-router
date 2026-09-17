# Project: Phase 2 Swift/Metal Execution Pipeline for Asynchronous MoE Router

## Architecture
Phase 2 implements the high-performance, asynchronous Swift/Metal execution pipeline for the Asynchronous MoE Router (`Qwen/Qwen1.5-MoE-A2.7B`) on Apple Silicon Unified Memory Architecture (UMA). It coordinates speculative expert weight prefetching via Metal 3 Fast I/O, strict ring/fallback memory pools, post-execution LRU cache maintenance via GPU execution logging, conditional execution via Indirect Command Buffers (ICB), and background Brier-score recalibration with bounded MLX Metal cache.

```
       +-------------------------------------------------------------+
       |                  Token Hidden State Stream (h_N)            |
       +-------------------------------------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |         Intermediate Gating & Speculative Prefetch          |
       |  - Speculative Queue (PriorityLow, maxCmdBuf: 16)           |
       |  - Ring Buffer Pool (16 slots, Shared Storage)              |
       |  - Zero-CPU Synchronization via MTLSharedEvent              |
       +-------------------------------------------------------------+
                     /                                 \
      [Hit / In-Time] /                                   \ [Miss / Timeout]
                     v                                     v
       +-------------------------+            +---------------------------------+
       | Speculative Slot Ready  |            | Fallback Buffer Pool (500MB)    |
       | - MTLBuffer bound       |            | - Fallback Queue (PriorityHigh) |
       +-------------------------+            | - Mark Spec Slot Abandoned/Dirty|
                     \                                 /
                      \                               /
                       v                             v
       +-------------------------------------------------------------+
       |           ICB Conditional Execution & Gating Grid           |
       |  - Global 1-byte abort_flag buffer                          |
       |  - Expert Kernels: ICB Native Zero-Thread conditional grid  |
       |  - Standard Kernels: Cascading if (*abort_flag) return;      |
       |  - Hazard Tracking: MTLResourceHazardTrackingModeTracked    |
       |  - Residual stream x mathematically preserved               |
       +-------------------------------------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |                  GPU Execution Log (Ring)                   |
       |  - Deterministic mapped slots (32-byte aligned, 128KB)      |
       |  - Zero-atomic GPU logging from gating threadgroups         |
       +-------------------------------------------------------------+
                     /                                 \
                    v                                   v
       +-------------------------+            +---------------------------------+
       | CPU Dispatch-Time LRU   |            | Background Recalibration Actor  |
       | - Drain log post-exec   |            | - MLX Cache Clamped to 200MB    |
       | - O(1) Doubly-Linked LRU|            | - Analytical Brier Gradient/NR  |
       +-------------------------+            +---------------------------------+
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | SwiftPM Project & Runtime MSL Manager | SwiftPM package structure with runtime MSL compilation (`MTLDevice.makeLibrary(source:)`) and custom test path `swift_tests/AsyncMoERouterTests` | M1 | Survey R1 |
| 2 | Metal 3 Fast I/O Dual-Queue Engine | `speculativeQueue` (PriorityLow, maxCommandBufferCount 16, concurrent) and `fallbackQueue` (PriorityHigh, concurrent) | M1 | Survey R1 |
| 3 | MTLIOFileHandle DMA & Zero-CPU SharedEvent Sync | Explicit block reads (`load(buffer:offset:size:sourceHandle:sourceHandleOffset:)`) and zero-CPU `MTLSharedEvent` hardware synchronization | M1 | Survey R1 |
| 4 | Speculative Ring Buffer Pool | Fixed 16-slot pre-allocated `MTLBuffer` array in `.storageModeShared` with thread-safe slot lifecycle state machine | M2 | Survey R2 |
| 5 | Strictly Isolated 500MB Fallback Buffer Pool | Isolated 500MB pool ($524,288,000$ bytes) dedicated exclusively to demand-fetch cache misses, blocked from speculative allocations | M2 | Survey R2 |
| 6 | Cache-Miss Deadlock Resolution Protocol | Fallback allocation, PriorityHigh demand fetch, marking speculative slot as `.abandoned` (dirty), `tryCancel()`, signal dropping in completedHandler | M2 | Survey R2 |
| 7 | Zero-Atomic GPU Execution Log | 32-byte aligned circular ring buffer (4096 entries, 128KB) in `.storageModeShared`, zero-atomic deterministic slot logging from GPU gating kernels | M3 | Survey R3 |
| 8 | CPU Dispatch-Time LRU Weight Tracker | Lock-free log draining, O(1) doubly-linked list LRU metadata updates strictly post-execution (never via pre-routing prediction) | M3 | Survey R3 |
| 9 | Global 1-Byte Abort Flag & Hazard Tracking | Global 1-byte `abort_flag` buffer in `.storageModeShared` with default hazard tracking (`MTLResourceHazardTrackingModeTracked`), no `.untracked` buffers | M4 | Survey R4 |
| 10 | ICB Native Conditional Execution | Argument Buffer encapsulated ICB (`struct ICBContainer`), zero-thread grid dispatch on abort (`uint3(0,0,0)`), pipeline descriptor configuration | M4 | Survey R4 |
| 11 | Cascading No-Op Layer Kernels & Stream Invariance | Cascading `if (*abort_flag != 0) return;` at instruction 0, mathematical and memory proof of residual stream $x$ non-corruption on abort | M4 | Survey R4 |
| 12 | MLX Metal Cache 200MB Clamp | Clamping MLX Metal cache to 200MB (`MLX.GPU.set(cacheLimit: 200 * 1024 * 1024)`) to prevent macOS WKdm memory compression of Ring Buffer | M5 | Survey R5 |
| 13 | Background Recalibration Actor | Async background Swift actor consuming Execution Log entries, computing analytical Brier-score gradient and 1D Newton-Raphson temperature updates | M5 | Survey R5 |
| 14 | Integrated Asynchronous MoE Pipeline | High-level Swift coordinator binding Fast I/O, Ring/Fallback pools, ICB execution, LRU tracking, and Recalibration into a unified pipeline | M6 | Pipeline Integration |
| 15 | Comprehensive 4-Tier E2E Test Suite | Opaque-box test suite (Tiers 1-4) verifying R1-R5, zero memory leaks, zero data corruption, smooth fallback, and conservative memory ceiling | M6 | Acceptance Criteria |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Fast I/O Engine & Dual-Queue Subsystem | SwiftPM layout, runtime MSL compilation manager, dual Fast I/O queues, MTLIOFileHandle block loading, MTLSharedEvent zero-CPU sync (Features 1, 2, 3) | none | DONE |
| M2 | Ring Buffer & Isolated Fallback Buffer Pools | 16-slot speculative Ring Buffer, isolated 500MB Fallback Pool, cache-miss deadlock resolution, slot abandonment & signal drop (Features 4, 5, 6) | M1 | DONE |
| M3 | GPU Execution Log & Dispatch-Time LRU Tracking | 32-byte circular Execution Log buffer, zero-atomic GPU logging kernel, lock-free CPU drain, O(1) LRU weight tracker (Features 7, 8) | M1 | PLANNED |
| M4 | ICB Native Conditional Execution & Cascading Abort | Global 1-byte abort_flag, Argument Buffer ICB container, zero-thread conditional grid dispatch, cascading no-ops, hazard tracking preservation, residual stream x invariance (Features 9, 10, 11) | M1 | PLANNED |
| M5 | MLX Cache Limiting & Background Recalibration | 200MB MLX metal cache limit, async background RecalibrationActor, Brier score optimization with analytical gradient (Features 12, 13) | M1, M3 | PLANNED |
| M6 | Full Pipeline Integration & E2E Acceptance | Integrated pipeline runner, full E2E Test Suite (Tiers 1-4) passing 100%, Tier 5 Adversarial Coverage Hardening (Features 14, 15) | M1, M2, M3, M4, M5 | PLANNED |

## Interface Contracts

### M1 ↔ M2: Fast I/O & Buffer Allocation Contract
- Protocol: `FastIOEngineProtocol`
- Methods:
  - `loadSpeculative(handle: MTLIOFileHandle, offset: Int, size: Int, targetBuffer: MTLBuffer, targetOffset: Int) -> (MTLSharedEvent, UInt64)`
  - `loadFallback(handle: MTLIOFileHandle, offset: Int, size: Int, targetBuffer: MTLBuffer, targetOffset: Int) -> (MTLSharedEvent, UInt64)`
- Queue Properties:
  - `speculativeQueue`: priority `.low`, maxCommandBufferCount: 16
  - `fallbackQueue`: priority `.high`, maxCommandBufferCount: 16

### M2 ↔ M3/M4: Buffer Slot & State Machine Contract
- Data Structure: `RingBufferSlot`
  - `slotIndex: Int`
  - `buffer: MTLBuffer`
  - `state: SlotState` (`.free`, `.prefetching`, `.ready`, `.abandoned`, `.inUse`)
  - `sharedEvent: MTLSharedEvent`
  - `signalValue: UInt64`
- Method: `resolveCacheMiss(expertID: Int, fallbackPool: FallbackBufferPool) -> MTLBuffer`
  - Allocates from 500MB Fallback Pool
  - Issues PriorityHigh I/O read
  - Marks speculative slot as `.abandoned`

### M3 ↔ M5: Execution Log Contract
- Data Structure: `ExecutionLogEntry` (32 bytes, C-layout / Swift struct)
  - `tokenIndex: UInt32`
  - `layerIndex: UInt16`
  - `horizonIndex: UInt16`
  - `expertID: UInt16`
  - `padding: UInt16`
  - `confidenceScore: Float32`
  - `timestamp: UInt64`
  - `reserved: UInt64`
- Buffer: `MTLBuffer` (4096 entries = 128 KB, `.storageModeShared`)

### M4: Abort Flag & ICB Gating Contract
- Buffer: `abort_flag` (1 byte, `.storageModeShared`, tracked hazards)
- MSL Argument Buffer:
  ```metal
  struct ICBContainer {
      command_buffer icb [[id(0)]];
  };
  ```
- Conditional Dispatch:
  - If `*abort_flag == 0`: `icb.concurrent_dispatch_threads(gridSize, threadgroupSize)`
  - If `*abort_flag != 0`: `icb.concurrent_dispatch_threads(uint3(0,0,0), uint3(0,0,0))`

### M5: Brier Recalibration Contract
- Function: `recalibrateTemperature(currentT: Float, logEntries: [ExecutionLogEntry], groundTruth: [Int]) -> Float`
- Cache limit: `MLX.GPU.set(cacheLimit: 200 * 1024 * 1024)`

## Code Layout
```
/Users/jack/Downloads/rlcd-router/
├── Package.swift                                      # SwiftPM package configuration
├── Sources/
│   └── AsyncMoERouter/
│       ├── Common/
│       │   ├── Config.swift                           # Global constants, layer dimensions, memory budgets
│       │   ├── MetalContext.swift                     # MTLDevice, CommandQueue, Runtime MSL Compilation
│       │   └── Types.swift                            # Shared structs, SlotState, ExecutionLogEntry
│       ├── FastIO/                                    # Milestone 1 ownership
│       │   ├── FastIOEngine.swift                     # Dual-queue (speculative/fallback) controller
│       │   ├── WeightFileHandle.swift                 # MTLIOFileHandle block wrapper
│       │   └── SyncEvent.swift                        # MTLSharedEvent zero-CPU synchronization
│       ├── BufferPools/                               # Milestone 2 ownership
│       │   ├── SpeculativeRingBuffer.swift            # 16-slot MTLBuffer pool with lifecycle tracking
│       │   ├── FallbackBufferPool.swift               # Strictly isolated 500MB buffer pool
│       │   └── DeadlockResolver.swift                 # Cache-miss demand fetch & signal dropping
│       ├── ExecutionLog/                              # Milestone 3 ownership
│       │   ├── GPUExecutionLog.swift                  # 32-byte circular buffer & CPU drainer
│       │   ├── LRUWeightTracker.swift                 # O(1) doubly-linked post-execution LRU
│       │   └── Shaders/
│       │       └── ExecutionLogKernels.metal.swift    # Embedded MSL shader string for GPU logging
│       ├── ExecutionPipeline/                         # Milestone 4 ownership
│       │   ├── AbortController.swift                  # Global 1-byte abort_flag manager
│       │   ├── ICBController.swift                    # Indirect Command Buffer encoder & container
│       │   └── Shaders/
│       │       ├── ICBGatingKernels.metal.swift       # Conditional zero-thread ICB gating shader
│       │       └── LayerKernels.metal.swift           # Cascading if (*abort_flag) return; shaders
│       ├── Recalibration/                             # Milestone 5 ownership
│       │   ├── MLXCacheController.swift               # 200MB cache clamping
│       │   └── RecalibrationActor.swift               # Background async Brier score optimizer
│       └── Pipeline.swift                             # Milestone 6 ownership: Unified coordinator
├── swift_tests/                                       # Swift test target (isolated from python tests/)
│   └── AsyncMoERouterTests/
│       ├── Common/
│       │   └── TestHelpers.swift                      # Synthetic weights, mock files, test fixtures
│       ├── Unit/                                      # Unit tests per milestone
│       │   ├── FastIOTests.swift
│       │   ├── BufferPoolTests.swift
│       │   ├── ExecutionLogTests.swift
│       │   ├── ICBAbortTests.swift
│       │   └── RecalibrationTests.swift
│       └── E2E/                                       # Milestone 6 E2E Test Suite
│           ├── Tier1_FeatureTests.swift
│           ├── Tier2_BoundaryTests.swift
│           ├── Tier3_PairwiseTests.swift
│           └── Tier4_WorkloadTests.swift
├── ORIGINAL_REQUEST.md                                # Authoritative user requirements
└── .agents/orchestrator_phase2/
    ├── PROJECT.md                                     # Authoritative Phase 2 project architecture
    ├── BRIEFING.md                                    # Persistent working memory
    ├── progress.md                                    # Heartbeat and milestone checklist
    └── GATE_STATUS.md                                 # Gate verdict tracking
```
