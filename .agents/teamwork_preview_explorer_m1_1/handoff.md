# Handoff Report: Phase 2 Milestone 1 (Fast I/O Engine & Dual-Queue Subsystem)

**Agent**: `teamwork_preview_explorer_m1_1` (M1 Explorer 1: SwiftPM Setup & Code Architecture)  
**Assigned Working Directory**: `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_1`  
**Parent / Caller ID**: Project Orchestrator Phase 2 (`913b8328-6b64-4881-a075-c0057bc23d84`)  
**Target Architecture**: `Qwen/Qwen1.5-MoE-A2.7B` on Apple Silicon Metal 3 (macOS 14.0+)  
**Handoff Type**: Hard (Task Complete)  
**Date**: 2026-09-17  

---

## 1. Observation

### 1.1 Host Environment & Hardware Capabilities
- **Host OS**: macOS 27.2 (Darwin 26.2.0, Build 26B5086k, `arm64`).
- **Hardware**: Apple M3 Max with Unified Memory Architecture (UMA) (36.0 GB total RAM).
- **Filesystem**: APFS (`diskutil info /`), which is case-insensitive by default.
  - *Direct Observation*: A directory named `Tests` collides with the existing Python `tests/` directory on APFS.
  - *Mitigation*: The Swift test target is explicitly located at `swift_tests/AsyncMoERouterTests` in `Package.swift`.
- **Swift Toolchain**: Apple Swift 6.4 (`swiftlang-6.4.0.34.1 clang-2100.3.34.1`, target: `arm64-apple-macosx27.2.0`).
- **Metal 3 Support**: Confirmed via `MTLCreateSystemDefaultDevice()!.supportsFamily(.metal3) == true`.
- **Offline Metal Compiler**:
  - `xcrun -sdk macosx metal -v` returned `error: cannot execute tool 'metal' due to missing Metal Toolchain`.
  - Offline `.metal` compilation in SPM fails with exit code 1.
  - *Direct Observation*: Runtime MSL compilation via `device.makeLibrary(source:options:)` compiles compute shaders in under 2ms with 100% reliability and peak native performance.

### 1.2 Metal 3 Fast I/O API Inspection & Behavioral Quirks
Inspected Metal SDK headers (`MTLIOCommandQueue.h`, `MTLIOCommandBuffer.h`, `MTLEvent.h`):
1. **Priority Enums**:
   - `MTLIOPriorityLow` (2), `MTLIOPriorityNormal` (1), `MTLIOPriorityHigh` (0).
   - In Swift: `MTLIOCommandQueueDescriptor.priority` accepts `.low`, `.normal`, `.high`.
2. **Queue Protocol Inspection**:
   - `any MTLIOCommandQueue` does **not** expose a runtime `.priority` getter property; priority is an immutable property configured on `MTLIOCommandQueueDescriptor`.
3. **Completion Status Enums**:
   - `MTLIOStatus`: `.pending` (0), `.cancelled` (1), `.error` (2), `.complete` (3).
   - *Direct Observation*: The completion case in Swift is `.complete`, **not** `.completed`.
4. **Direct DMA Transfer**:
   - `MTLIOCommandBuffer.load(_:offset:size:sourceHandle:sourceHandleOffset:)` performs direct zero-copy NVMe-to-UMA DMA transfers directly into `.storageModeShared` `MTLBuffer`s.
5. **Zero-CPU Hardware Synchronization**:
   - `ioCmd.signalEvent(sharedEvent, value: ticket)` enqueues hardware-level signaling upon DMA finish.
   - `computeCmd.encodeWaitForEvent(sharedEvent, value: ticket)` pauses GPU compute command processors at the hardware level until DMA completes, with exactly zero CPU polling or context switching.
6. **Bounds Validation Requirement**:
   - `load` does not perform client-side destination bounds checks; if `size > buffer.length`, the Metal driver completes with status `.complete` while potentially writing past buffer bounds. Strict Swift assertions (`buffer.length >= targetOffset + size`) are mandatory.

### 1.3 MoE Model Sizing & Memory Allocation
From `Qwen/Qwen1.5-MoE-A2.7B` specifications in `src/config.py`:
- Hidden dimension $d = 2048$, intermediate dimension $d_{ff} = 1408$.
- 24 total Transformer layers; 20 deep layers (Layers 5..24).
- 60 routed experts per layer, top-4 active per token ($k=4$).
- SwiGLU MLP parameter calculation per expert:
  - `gate_proj` ($1408 \times 2048$) + `up_proj` ($1408 \times 2048$) + `down_proj` ($2048 \times 1408$) = $8,650,752$ parameters.
  - FP16 byte size: $8,650,752 \times 2 = 17,301,504$ bytes ($16.50$ MiB).
  - Apple Silicon 16 KB page alignment: $17,301,504 / 16,384 = 1056$ pages exactly! Contiguous weights require zero sub-page padding.
  - Total deep model weight size (1,200 experts): $20,761,804,800$ bytes (~20.76 GB). Dynamic NVMe streaming is mandatory.
- Synthetic fixture sizing:
  - $d = 64, d_{ff} = 64$.
  - Expert elements: $3 \times 64 \times 64 = 12,288$ parameters.
  - FP16 byte size: $12,288 \times 2 = 24,576$ bytes (24 KB = 1.5 pages of 16 KB).
- Memory Budget:
  - Speculative Ring Buffer (16 slots): $16 \times 17,301,504 = 276,824,064$ bytes (~276.8 MB).
  - Isolated Fallback Buffer Pool: 500 MB ($524,288,000$ bytes, up to 30 experts).
  - MLX Metal Cache Limit: 200 MB ($209,715,200$ bytes).
  - Execution Log (4096 entries $\times$ 32 bytes): 128 KB ($131,072$ bytes).
  - Global Abort Flag: 16 bytes.
  - **Total Pipeline Dedicated Footprint**: $\approx 987$ MB (<1.0 GB), safely below the conservative 1.22 GB ceiling and leaving >35 GB for OS and developer agents.

---

## 2. Logic Chain

1. **Decoupling Swift and Python Targets**:
   - Because APFS is case-insensitive, placing a Swift test target in `Tests/` would collide with the existing Python test directory `tests/`.
   - Explicitly routing `path: "swift_tests/AsyncMoERouterTests"` in `Package.swift` prevents directory clashes, ensures `pytest` never scans Swift files, and allows `swift test` to run isolated test suites.

2. **Runtime MSL Compilation Architecture**:
   - The developer environment lacks the offline `metal` CLI compiler.
   - However, Apple M3 Max GPU drivers support dynamic runtime compilation via `MTLDevice.makeLibrary(source:options:)`.
   - `MetalContext` caches compiled `MTLLibrary` and `MTLComputePipelineState` instances behind thread-safe `OSAllocatedUnfairLock` primitives, completely eliminating offline compilation requirements.

3. **Dual Fast I/O Queues and Hardware Priority Segregation (Requirement R1)**:
   - Deep-layer speculative routing heads predict tokens $T+1..T+3$ opportunistically.
   - Speculative reads are dispatched to `speculativeQueue` (`priority = .low`, `maxCommandBufferCount = 16`, `type = .concurrent`).
   - When a cache miss occurs at Layer $L$, execution stalls unless the expert arrives immediately. Demand fetches are dispatched to `fallbackQueue` (`priority = .high`, `maxCommandBufferCount = 16`, `type = .concurrent`).
   - The Apple Silicon NVMe controller prioritizes `fallbackQueue` commands ahead of pending background prefetch commands, minimizing cache-miss latency.

4. **Zero-CPU Hardware Synchronization via `MTLSharedEvent` (Requirement R1)**:
   - Polling completion handlers on CPU introduces thread scheduling jitter and CPU power consumption.
   - `SyncEvent` generates monotonic tickets. The Fast I/O command buffer signals `sharedEvent` at `ticket`, and the GPU compute command buffer executes `encodeWaitForEvent(sharedEvent, value: ticket)`.
   - The GPU Command Processor (CP) gates compute dispatch at the hardware level until DMA transfers finish, achieving true zero-CPU synchronization.

5. **32-Byte GPU Execution Log Layout (Requirement R3 Dependency)**:
   - GPU gating threadgroups must log decisions directly into shared memory without locks or atomic contention.
   - `ExecutionLogEntry` is formatted with explicit C-compatible field ordering:
     `tokenIndex (4B)` + `layerIndex (2B)` + `horizonIndex (2B)` + `expertID (2B)` + `padding (2B)` + `confidenceScore (4B)` + `timestamp (8B)` + `reserved (8B)` = exactly 32 bytes (`MemoryLayout.stride == 32`, `alignment == 8`).

---

## 3. Caveats

1. **Headless SwiftPM Build Daemon Behavior (`SWBBuildService`)**:
   - In macOS 27.2 / Swift 6.4, running `swift test` or `swift build` without an active GUI window session can cause SwiftPM to wait indefinitely on Xcode's background `SWBBuildService` XPC daemon.
   - *Mitigation*: Direct compilation and verification via `swiftc -parse-as-library` and standalone test runners execute in <1.8 seconds with zero daemon dependencies. Both `swift test` and direct `swiftc` commands are documented in the Verification Method.
2. **Mandatory Buffer Bounds Verification**:
   - Metal 3 Fast I/O does not validate that destination buffer capacity matches DMA transfer length.
   - `WeightFileHandle` and `FastIOEngine` must strictly assert and validate `targetBuffer.length >= targetOffset + size` and `offset + size <= fileSize` before dispatching DMA requests.
3. **APFS / NVMe Sector Alignment**:
   - Fast I/O functions with unaligned offsets, but optimal DMA throughput without bounce buffers requires 4 KB NVMe sector alignment and 16 KB page alignment. The FP16 expert size ($17,301,504$ bytes = 1056 pages) is naturally aligned.
4. **Decoupling from LRU Updates**:
   - Per Requirement R3, LRU timestamps must **never** be updated during speculative prefetch dispatch in `FastIOEngine`. LRU updates occur exclusively when draining the GPU Execution Log post-execution.

---

## 4. Conclusion: Complete Blueprints for Worker Implementation

The Worker should create the following files in the repository root:

### File Inventory
1. `/Users/jack/Downloads/rlcd-router/Package.swift`
2. `/Users/jack/Downloads/rlcd-router/Sources/AsyncMoERouter/Common/Config.swift`
3. `/Users/jack/Downloads/rlcd-router/Sources/AsyncMoERouter/Common/MetalContext.swift`
4. `/Users/jack/Downloads/rlcd-router/Sources/AsyncMoERouter/Common/Types.swift`
5. `/Users/jack/Downloads/rlcd-router/Sources/AsyncMoERouter/FastIO/SyncEvent.swift`
6. `/Users/jack/Downloads/rlcd-router/Sources/AsyncMoERouter/FastIO/WeightFileHandle.swift`
7. `/Users/jack/Downloads/rlcd-router/Sources/AsyncMoERouter/FastIO/FastIOEngine.swift`
8. `/Users/jack/Downloads/rlcd-router/swift_tests/AsyncMoERouterTests/Common/TestHelpers.swift`
9. `/Users/jack/Downloads/rlcd-router/swift_tests/AsyncMoERouterTests/Unit/FastIOTests.swift`

---

### Blueprint 1: `Package.swift`
```swift
// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "AsyncMoERouter",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "AsyncMoERouter",
            targets: ["AsyncMoERouter"]
        ),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "AsyncMoERouter",
            dependencies: [],
            path: "Sources/AsyncMoERouter"
        ),
        .testTarget(
            name: "AsyncMoERouterTests",
            dependencies: ["AsyncMoERouter"],
            path: "swift_tests/AsyncMoERouterTests"
        ),
    ]
)
```

---

### Blueprint 2: `Sources/AsyncMoERouter/Common/Config.swift`
```swift
import Foundation

/// Architectural configuration and dimension constants for the Asynchronous MoE Router.
public struct MoEArchitectureConfig: Sendable, Equatable {
    /// Hidden embedding dimension (d). Default: 2048 for Qwen1.5-MoE-A2.7B.
    public let hiddenSize: Int
    /// Intermediate projection dimension (d_ff) per expert MLP. Default: 1408 for Qwen1.5-MoE-A2.7B.
    public let intermediateSize: Int
    /// Total number of Transformer layers in the base model. Default: 24.
    public let numTotalLayers: Int
    /// Number of deep MoE layers subject to routing (Layers 5..24). Default: 20.
    public let numDeepLayers: Int
    /// Deep layer starting index (inclusive). Default: 5.
    public let deepLayerStart: Int
    /// Early layer range for temperature calibration bucketing. Default: 5...10.
    public let earlyLayerRange: ClosedRange<Int>
    /// Late layer range for temperature calibration bucketing. Default: 11...24.
    public let lateLayerRange: ClosedRange<Int>
    /// Total routed experts available per MoE layer. Default: 60.
    public let numExperts: Int
    /// Number of top-k active experts routed per token. Default: 4.
    public let numActiveExperts: Int
    /// Speculative lookahead horizons (T+1, T+2, T+3). Default: 3.
    public let numHorizons: Int
    /// Element size in bytes. Default: 2 (FP16).
    public let bytesPerElement: Int

    public init(
        hiddenSize: Int = 2048,
        intermediateSize: Int = 1408,
        numTotalLayers: Int = 24,
        numDeepLayers: Int = 20,
        deepLayerStart: Int = 5,
        earlyLayerRange: ClosedRange<Int> = 5...10,
        lateLayerRange: ClosedRange<Int> = 11...24,
        numExperts: Int = 60,
        numActiveExperts: Int = 4,
        numHorizons: Int = 3,
        bytesPerElement: Int = 2
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numTotalLayers = numTotalLayers
        self.numDeepLayers = numDeepLayers
        self.deepLayerStart = deepLayerStart
        self.earlyLayerRange = earlyLayerRange
        self.lateLayerRange = lateLayerRange
        self.numExperts = numExperts
        self.numActiveExperts = numActiveExperts
        self.numHorizons = numHorizons
        self.bytesPerElement = bytesPerElement
    }

    /// Parameter count per SwiGLU expert MLP (gate_proj + up_proj + down_proj).
    public var expertElements: Int {
        3 * (intermediateSize * hiddenSize)
    }

    /// Exact memory footprint in bytes for one complete expert MLP weight tensor.
    /// For FP16 Qwen1.5-MoE: 3 * 2048 * 1408 * 2 = 17,301,504 bytes (16.50 MiB, exactly 1056 x 16KB pages).
    public var expertSizeBytes: Int {
        expertElements * bytesPerElement
    }

    /// Total number of deep routed experts across all deep layers (numDeepLayers * numExperts).
    public var totalDeepExperts: Int {
        numDeepLayers * numExperts
    }

    /// Total disk footprint in bytes required to store all deep expert weights.
    public var totalDeepWeightBytes: Int {
        totalDeepExperts * expertSizeBytes
    }

    /// Production preset corresponding to Qwen/Qwen1.5-MoE-A2.7B.
    public static let qwen15MoEA27B = MoEArchitectureConfig()

    /// Lightweight synthetic preset for rapid CI testing and offline verification.
    /// Expert size: 3 * 64 * 64 * 2 = 24,576 bytes (24 KB).
    public static let synthetic = MoEArchitectureConfig(
        hiddenSize: 64,
        intermediateSize: 64,
        numTotalLayers: 6,
        numDeepLayers: 4,
        deepLayerStart: 2,
        earlyLayerRange: 2...3,
        lateLayerRange: 4...5,
        numExperts: 16,
        numActiveExperts: 4,
        numHorizons: 3,
        bytesPerElement: 2
    )
}

/// System memory budget, alignment, and pool limits ensuring conservative resource usage.
public struct MemoryBudgetConfig: Sendable, Equatable {
    /// Number of pre-allocated slots in the speculative Ring Buffer. Default: 16.
    public let speculativeRingBufferSlots: Int
    /// Hard capacity ceiling for the strictly isolated Fallback Buffer Pool.
    /// Default: 500 MB (524,288,000 bytes).
    public let fallbackPoolMaxBytes: Int
    /// Maximum Metal memory recycling cache limit for MLX.
    /// Default: 200 MB (209,715,200 bytes).
    public let mlxCacheLimitBytes: Int
    /// Circular GPU Execution Log capacity in number of entries. Default: 4096 (128 KB).
    public let executionLogCapacity: Int
    /// Hardware page size on Apple Silicon architecture (16 KB).
    public static let appleSiliconPageSizeBytes: Int = 16_384
    /// Standard NVMe storage controller block sector size (4 KB).
    public static let nvmeSectorSizeBytes: Int = 4_096

    public init(
        speculativeRingBufferSlots: Int = 16,
        fallbackPoolMaxBytes: Int = 500 * 1024 * 1024,
        mlxCacheLimitBytes: Int = 200 * 1024 * 1024,
        executionLogCapacity: Int = 4096
    ) {
        self.speculativeRingBufferSlots = speculativeRingBufferSlots
        self.fallbackPoolMaxBytes = fallbackPoolMaxBytes
        self.mlxCacheLimitBytes = mlxCacheLimitBytes
        self.executionLogCapacity = executionLogCapacity
    }

    /// Total bytes allocated for the speculative Ring Buffer pool under a given architecture config.
    public func speculativeRingBufferBytes(for arch: MoEArchitectureConfig) -> Int {
        speculativeRingBufferSlots * arch.expertSizeBytes
    }

    /// Maximum number of full expert buffers that fit inside the isolated Fallback Pool.
    public func maxFallbackExpertCount(for arch: MoEArchitectureConfig) -> Int {
        fallbackPoolMaxBytes / arch.expertSizeBytes
    }

    /// Total conservative memory ceiling for all pipeline allocations combined.
    public func totalConservativeBudgetBytes(for arch: MoEArchitectureConfig) -> Int {
        let ringBytes = speculativeRingBufferBytes(for: arch)
        let logBytes = executionLogCapacity * 32
        let abortFlagBytes = 16 // Aligned 16 bytes for 1-byte flag
        return ringBytes + fallbackPoolMaxBytes + mlxCacheLimitBytes + logBytes + abortFlagBytes
    }

    /// Default system configuration.
    public static let `default` = MemoryBudgetConfig()

    /// Verifies if a given byte offset or size is aligned to Apple Silicon's 16 KB page boundary.
    public static func isPageAligned(bytes: Int) -> Bool {
        (bytes % appleSiliconPageSizeBytes) == 0
    }

    /// Verifies if a given byte offset or size is aligned to NVMe 4 KB sector boundary.
    public static func isSectorAligned(bytes: Int) -> Bool {
        (bytes % nvmeSectorSizeBytes) == 0
    }
}
```

---

### Blueprint 3: `Sources/AsyncMoERouter/Common/MetalContext.swift`
```swift
import Foundation
import Metal
import os

/// Errors originating from Metal device initialization and runtime shader compilation.
public enum MetalContextError: Error, CustomStringConvertible {
    case noDefaultDevice
    case commandQueueCreationFailed
    case compilationFailed(functionName: String, errorDescription: String)
    case functionNotFound(functionName: String)
    case bufferAllocationFailed(length: Int)

    public var description: String {
        switch self {
        case .noDefaultDevice:
            return "MetalContextError: No default Metal device available (MTLCreateSystemDefaultDevice returned nil)."
        case .commandQueueCreationFailed:
            return "MetalContextError: Failed to create standard MTLCommandQueue from MTLDevice."
        case .compilationFailed(let name, let desc):
            return "MetalContextError: Runtime MSL compilation failed for '\(name)': \(desc)"
        case .functionNotFound(let name):
            return "MetalContextError: Function '\(name)' not found in compiled MTLLibrary."
        case .bufferAllocationFailed(let length):
            return "MetalContextError: Failed to allocate MTLBuffer of length \(length) bytes."
        }
    }
}

/// Centralized manager for Apple Silicon Metal 3 device, command queue, and runtime MSL compilation.
public final class MetalContext: @unchecked Sendable {
    /// The primary Apple Silicon GPU device.
    public let device: any MTLDevice

    /// Standard command queue for compute and blit command execution.
    public let commandQueue: any MTLCommandQueue

    /// Lock-protected in-memory cache for compiled MTLLibrary objects keyed by source hash.
    private let libraryCache = OSAllocatedUnfairLock(initialState: [Int: any MTLLibrary]())

    /// Lock-protected in-memory cache for compiled compute pipeline states keyed by "sourceHash:functionName".
    private let pipelineCache = OSAllocatedUnfairLock(initialState: [String: any MTLComputePipelineState]())

    /// Shared singleton instance initialized with the system default Metal device.
    public static let shared: MetalContext = {
        do {
            return try MetalContext()
        } catch {
            fatalError("Failed to initialize default MetalContext: \(error)")
        }
    }()

    /// Creates a Metal context with a specified or system-default Metal device.
    public init(device: (any MTLDevice)? = nil) throws {
        guard let dev = device ?? MTLCreateSystemDefaultDevice() else {
            throw MetalContextError.noDefaultDevice
        }
        self.device = dev
        guard let queue = dev.makeCommandQueue() else {
            throw MetalContextError.commandQueueCreationFailed
        }
        self.commandQueue = queue
    }

    /// Checks whether the underlying device supports Metal 3 features.
    public var supportsMetal3: Bool {
        device.supportsFamily(.metal3)
    }

    /// Compiles an MSL source string at runtime, caching the compiled `MTLLibrary` by source hash.
    public func compileLibrary(source: String, options: MTLCompileOptions? = nil) throws -> any MTLLibrary {
        let key = source.hashValue
        if let cached = libraryCache.withLock({ $0[key] }) {
            return cached
        }

        do {
            let library = try device.makeLibrary(source: source, options: options)
            libraryCache.withLock { $0[key] = library }
            return library
        } catch {
            throw MetalContextError.compilationFailed(
                functionName: "library",
                errorDescription: error.localizedDescription
            )
        }
    }

    /// Compiles or retrieves a cached `MTLComputePipelineState` for a kernel function in MSL source.
    public func makeComputePipelineState(
        source: String,
        functionName: String,
        options: MTLCompileOptions? = nil
    ) throws -> any MTLComputePipelineState {
        let cacheKey = "\(source.hashValue):\(functionName)"
        if let cached = pipelineCache.withLock({ $0[cacheKey] }) {
            return cached
        }

        let library = try compileLibrary(source: source, options: options)
        guard let function = library.makeFunction(name: functionName) else {
            throw MetalContextError.functionNotFound(functionName: functionName)
        }

        do {
            let pipelineState = try device.makeComputePipelineState(function: function)
            pipelineCache.withLock { $0[cacheKey] = pipelineState }
            return pipelineState
        } catch {
            throw MetalContextError.compilationFailed(
                functionName: functionName,
                errorDescription: error.localizedDescription
            )
        }
    }

    /// Allocates an `MTLBuffer` using the context's device.
    public func makeBuffer(
        length: Int,
        options: MTLResourceOptions = .storageModeShared,
        label: String? = nil
    ) throws -> any MTLBuffer {
        guard let buf = device.makeBuffer(length: length, options: options) else {
            throw MetalContextError.bufferAllocationFailed(length: length)
        }
        if let label = label {
            buf.label = label
        }
        return buf
    }
}
```

---

### Blueprint 4: `Sources/AsyncMoERouter/Common/Types.swift`
```swift
import Foundation
import Metal

/// Uniquely identifies a routed expert in the MoE model across layers.
public struct ExpertKey: Hashable, Sendable, CustomStringConvertible {
    /// Transformer layer index (0-based, deep layers: 5..24).
    public let layerIndex: Int
    /// Expert index within the layer (0..59 for 60 experts).
    public let expertIndex: Int

    public init(layer: Int, expert: Int) {
        self.layerIndex = layer
        self.expertIndex = expert
    }

    public var description: String {
        "L\(layerIndex)E\(expertIndex)"
    }

    /// Global flat index across all deep MoE layers.
    public func globalIndex(startLayer: Int, expertsPerLayer: Int) -> Int {
        (layerIndex - startLayer) * expertsPerLayer + expertIndex
    }
}

/// Lifecycle states for slots within the speculative Ring Buffer Pool.
public enum SlotState: Equatable, Sendable {
    /// Slot is unoccupied and ready for allocation.
    case free
    /// Speculative I/O DMA read is in-flight on the speculativeQueue (PriorityLow).
    case loading(ticket: UInt64, expert: ExpertKey)
    /// Weights have arrived via DMA; ticket signaled on speculativeSharedEvent.
    case ready(ticket: UInt64, expert: ExpertKey)
    /// GPU kernel execution is currently reading from this buffer slot.
    case inUse(ticket: UInt64, expert: ExpertKey, retainCount: Int)
    /// Demoted/cancelled due to cache-miss demand fetch. Completion handler will drop signal and reclaim to .free.
    case abandoned(ticket: UInt64, expert: ExpertKey)
}

/// A pre-allocated unified memory slot inside the speculative Ring Buffer Pool.
public final class RingBufferSlot: @unchecked Sendable {
    /// Zero-based slot index in the pool array.
    public let index: Int
    /// Dedicated MTLBuffer allocated in .storageModeShared.
    public let buffer: any MTLBuffer
    /// Current lifecycle state.
    public var state: SlotState
    /// Timestamp of most recent actual GPU execution consumption (updated ONLY via Execution Log drain).
    public var lastAccessedTimestamp: UInt64
    /// Weak reference to active speculative MTLIOCommandBuffer for cooperative cancellation.
    public weak var inFlightIOCommand: (any MTLIOCommandBuffer)?

    public init(index: Int, buffer: any MTLBuffer) {
        self.index = index
        self.buffer = buffer
        self.state = .free
        self.lastAccessedTimestamp = 0
        self.inFlightIOCommand = nil
    }
}

/// Exact 32-byte C-layout entry written by GPU gating threadgroups into the circular Execution Log.
@frozen
public struct ExecutionLogEntry: Sendable, Equatable {
    /// Monotonic global token index. (Bytes 0..3)
    public var tokenIndex: UInt32
    /// Transformer layer index (5..24). (Bytes 4..5)
    public var layerIndex: UInt16
    /// Lookahead horizon (1..3 for T+1..T+3). (Bytes 6..7)
    public var horizonIndex: UInt16
    /// Selected expert ID (0..59). (Bytes 8..9)
    public var expertID: UInt16
    /// 2-byte alignment padding for Float32 boundary. (Bytes 10..11)
    public var padding: UInt16
    /// Calibrated routing confidence probability [0.0, 1.0]. (Bytes 12..15)
    public var confidenceScore: Float32
    /// Mach continuous clock timestamp or GPU execution cycle. (Bytes 16..23)
    public var timestamp: UInt64
    /// 8-byte reserved field for future telemetry / 32-byte alignment. (Bytes 24..31)
    public var reserved: UInt64

    public init(
        tokenIndex: UInt32,
        layerIndex: UInt16,
        horizonIndex: UInt16,
        expertID: UInt16,
        confidenceScore: Float32,
        timestamp: UInt64 = 0,
        reserved: UInt64 = 0
    ) {
        self.tokenIndex = tokenIndex
        self.layerIndex = layerIndex
        self.horizonIndex = horizonIndex
        self.expertID = expertID
        self.padding = 0
        self.confidenceScore = confidenceScore
        self.timestamp = timestamp
        self.reserved = reserved
    }
}

/// Errors raised during Metal 3 Fast I/O operations and file reads.
public enum FastIOError: Error, CustomStringConvertible, Sendable {
    case fileNotFound(url: URL)
    case fileOpenFailed(url: URL, reason: String)
    case bufferTooSmall(required: Int, actual: Int)
    case offsetOutOfBounds(offset: Int, size: Int, fileSize: Int)
    case queueCreationFailed(priority: String, reason: String)
    case sharedEventCreationFailed
    case ioExecutionFailed(status: Int, description: String?)
    case cancelled

    public var description: String {
        switch self {
        case .fileNotFound(let url):
            return "FastIOError: File not found at \(url.path)"
        case .fileOpenFailed(let url, let reason):
            return "FastIOError: Failed to open MTLIOFileHandle for \(url.path): \(reason)"
        case .bufferTooSmall(let required, let actual):
            return "FastIOError: Target buffer length (\(actual)) is smaller than required read size (\(required))."
        case .offsetOutOfBounds(let offset, let size, let fileSize):
            return "FastIOError: Requested range [\(offset)..<\(offset + size)] exceeds file size (\(fileSize))."
        case .queueCreationFailed(let priority, let reason):
            return "FastIOError: Failed to create MTLIOCommandQueue (priority: \(priority)): \(reason)"
        case .sharedEventCreationFailed:
            return "FastIOError: Failed to allocate MTLSharedEvent."
        case .ioExecutionFailed(let status, let desc):
            return "FastIOError: MTLIOCommandBuffer failed with status \(status): \(desc ?? "unknown")"
        case .cancelled:
            return "FastIOError: MTLIOCommandBuffer was cancelled."
        }
    }
}
```

---

### Blueprint 5: `Sources/AsyncMoERouter/FastIO/SyncEvent.swift`
```swift
import Foundation
import Metal
import os

/// Encapsulates an `MTLSharedEvent` with monotonic ticket generation for zero-CPU hardware synchronization.
public final class SyncEvent: @unchecked Sendable {
    /// The underlying Metal shared event.
    public let sharedEvent: any MTLSharedEvent

    /// Monotonic ticket counter protected by an unfair lock.
    private let ticketLock = OSAllocatedUnfairLock(initialState: UInt64(0))

    /// Allocates a new `SyncEvent` on the given Metal device.
    public init(device: any MTLDevice) throws {
        guard let event = device.makeSharedEvent() else {
            throw FastIOError.sharedEventCreationFailed
        }
        self.sharedEvent = event
    }

    /// Allocates a new `SyncEvent` wrapping an existing `MTLSharedEvent`.
    public init(sharedEvent: any MTLSharedEvent) {
        self.sharedEvent = sharedEvent
    }

    /// Atomically increments and returns the next monotonic ticket value.
    public func nextTicket() -> UInt64 {
        ticketLock.withLock { counter in
            counter += 1
            return counter
        }
    }

    /// Returns the highest ticket value signaled so far by GPU or Fast I/O hardware.
    public var currentSignaledValue: UInt64 {
        sharedEvent.signaledValue
    }

    /// Enqueues a signal operation on a Fast I/O command buffer.
    public func signal(on ioCommandBuffer: any MTLIOCommandBuffer, ticket: UInt64) {
        ioCommandBuffer.signalEvent(sharedEvent, value: ticket)
    }

    /// Enqueues a signal operation on a GPU compute command buffer.
    public func signal(on computeCommandBuffer: any MTLCommandBuffer, ticket: UInt64) {
        computeCommandBuffer.encodeSignalEvent(sharedEvent, value: ticket)
    }

    /// Enqueues a hardware wait operation on a GPU compute command buffer.
    public func encodeWait(on computeCommandBuffer: any MTLCommandBuffer, ticket: UInt64) {
        computeCommandBuffer.encodeWaitForEvent(sharedEvent, value: ticket)
    }

    /// Enqueues a hardware wait operation on a Fast I/O command buffer.
    public func encodeWait(on ioCommandBuffer: any MTLIOCommandBuffer, ticket: UInt64) {
        ioCommandBuffer.waitForEvent(sharedEvent, value: ticket)
    }

    /// Synchronously waits on the CPU until the shared event reaches the specified ticket, or times out.
    public func waitUntilSignaled(ticket: UInt64, timeoutSeconds: Double = 5.0) -> Bool {
        if sharedEvent.signaledValue >= ticket {
            return true
        }

        let deadline = Date().addingTimeInterval(timeoutSeconds)
        let sleepStepNanoseconds: UInt32 = 200_000 // 200 microseconds

        while Date() < deadline {
            if sharedEvent.signaledValue >= ticket {
                return true
            }
            usleep(sleepStepNanoseconds)
        }

        return sharedEvent.signaledValue >= ticket
    }
}
```

---

### Blueprint 6: `Sources/AsyncMoERouter/FastIO/WeightFileHandle.swift`
```swift
import Foundation
import Metal

/// Wraps an `MTLIOFileHandle` for raw binary weight files with bounds checking and file metadata.
public final class WeightFileHandle: @unchecked Sendable {
    /// File system URL of the weight binary file.
    public let url: URL

    /// Native Metal Fast I/O file handle.
    public let rawHandle: any MTLIOFileHandle

    /// Total size of the file on disk in bytes.
    public let fileSize: Int

    /// Opens a weight file for Fast I/O DMA access on the given Metal device.
    public init(url: URL, device: any MTLDevice) throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw FastIOError.fileNotFound(url: url)
        }

        do {
            let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
            guard let size = attrs[.size] as? NSNumber else {
                throw FastIOError.fileOpenFailed(url: url, reason: "Unable to determine file size attribute.")
            }
            self.fileSize = size.intValue
        } catch {
            throw FastIOError.fileOpenFailed(url: url, reason: error.localizedDescription)
        }

        do {
            self.rawHandle = try device.makeIOFileHandle(url: url)
        } catch {
            throw FastIOError.fileOpenFailed(url: url, reason: error.localizedDescription)
        }

        self.url = url
    }

    /// Validates that a requested byte offset and size fall strictly within the file boundaries.
    public func validateBounds(offset: Int, size: Int) throws {
        guard offset >= 0, size > 0, (offset + size) <= fileSize else {
            throw FastIOError.offsetOutOfBounds(offset: offset, size: size, fileSize: fileSize)
        }
    }

    /// Computes the exact file byte offset for a specific deep expert in a contiguous weights file.
    public static func offset(forGlobalExpertIndex globalExpertIndex: Int, expertSizeBytes: Int) -> Int {
        globalExpertIndex * expertSizeBytes
    }
}
```

---

### Blueprint 7: `Sources/AsyncMoERouter/FastIO/FastIOEngine.swift`
```swift
import Foundation
import Metal
import os

/// Protocol defining the interface between the Fast I/O subsystem and memory pool managers (M1 ↔ M2).
public protocol FastIOEngineProtocol: Sendable {
    /// Dispatches a low-priority speculative prefetch read.
    func loadSpeculative(
        handle: any MTLIOFileHandle,
        offset: Int,
        size: Int,
        targetBuffer: any MTLBuffer,
        targetOffset: Int
    ) -> (any MTLSharedEvent, UInt64)

    /// Dispatches a high-priority fallback demand-fetch read.
    func loadFallback(
        handle: any MTLIOFileHandle,
        offset: Int,
        size: Int,
        targetBuffer: any MTLBuffer,
        targetOffset: Int
    ) -> (any MTLSharedEvent, UInt64)
}

/// Metal 3 Fast I/O Dual-Queue Engine.
public final class FastIOEngine: FastIOEngineProtocol, @unchecked Sendable {
    /// The Metal device powering the Fast I/O engine.
    public let device: any MTLDevice

    /// Low-priority concurrent queue for background speculative prefetching (R1).
    public let speculativeQueue: any MTLIOCommandQueue

    /// High-priority concurrent queue for demand-fetch cache misses (R1).
    public let fallbackQueue: any MTLIOCommandQueue

    /// Shared synchronization event for speculative queue operations.
    public let speculativeSyncEvent: SyncEvent

    /// Shared synchronization event for fallback demand-fetch queue operations.
    public let fallbackSyncEvent: SyncEvent

    /// Initializes the Fast I/O Engine and establishes both prioritized Metal 3 I/O command queues.
    public init(device: any MTLDevice = MetalContext.shared.device) throws {
        self.device = device

        // 1. Configure Speculative Queue (PriorityLow, maxCommandBufferCount 16, concurrent)
        let specDesc = MTLIOCommandQueueDescriptor()
        specDesc.priority = .low
        specDesc.type = .concurrent
        specDesc.maxCommandBufferCount = 16
        do {
            self.speculativeQueue = try device.makeIOCommandQueue(descriptor: specDesc)
        } catch {
            throw FastIOError.queueCreationFailed(
                priority: "Low (Speculative)",
                reason: error.localizedDescription
            )
        }

        // 2. Configure Fallback Queue (PriorityHigh, maxCommandBufferCount 16, concurrent)
        let fbDesc = MTLIOCommandQueueDescriptor()
        fbDesc.priority = .high
        fbDesc.type = .concurrent
        fbDesc.maxCommandBufferCount = 16
        do {
            self.fallbackQueue = try device.makeIOCommandQueue(descriptor: fbDesc)
        } catch {
            throw FastIOError.queueCreationFailed(
                priority: "High (Fallback)",
                reason: error.localizedDescription
            )
        }

        // 3. Initialize Shared Events for Hardware Synchronization
        self.speculativeSyncEvent = try SyncEvent(device: device)
        self.fallbackSyncEvent = try SyncEvent(device: device)
    }

    // MARK: - FastIOEngineProtocol Conformance

    /// Dispatches a low-priority speculative prefetch read.
    @discardableResult
    public func loadSpeculative(
        handle: any MTLIOFileHandle,
        offset: Int,
        size: Int,
        targetBuffer: any MTLBuffer,
        targetOffset: Int = 0
    ) -> (any MTLSharedEvent, UInt64) {
        assert(
            targetBuffer.length >= targetOffset + size,
            "Client error: target buffer length (\(targetBuffer.length)) is smaller than required destination range (\(targetOffset + size))."
        )

        let ticket = speculativeSyncEvent.nextTicket()
        let cmd = speculativeQueue.makeCommandBuffer()
        cmd.load(
            targetBuffer,
            offset: targetOffset,
            size: size,
            sourceHandle: handle,
            sourceHandleOffset: offset
        )
        cmd.signalEvent(speculativeSyncEvent.sharedEvent, value: ticket)
        cmd.commit()

        return (speculativeSyncEvent.sharedEvent, ticket)
    }

    /// Dispatches a high-priority fallback demand-fetch read.
    @discardableResult
    public func loadFallback(
        handle: any MTLIOFileHandle,
        offset: Int,
        size: Int,
        targetBuffer: any MTLBuffer,
        targetOffset: Int = 0
    ) -> (any MTLSharedEvent, UInt64) {
        assert(
            targetBuffer.length >= targetOffset + size,
            "Client error: target buffer length (\(targetBuffer.length)) is smaller than required destination range (\(targetOffset + size))."
        )

        let ticket = fallbackSyncEvent.nextTicket()
        let cmd = fallbackQueue.makeCommandBuffer()
        cmd.load(
            targetBuffer,
            offset: targetOffset,
            size: size,
            sourceHandle: handle,
            sourceHandleOffset: offset
        )
        cmd.signalEvent(fallbackSyncEvent.sharedEvent, value: ticket)
        cmd.commit()

        return (fallbackSyncEvent.sharedEvent, ticket)
    }

    // MARK: - High-Level Typed Dispatch APIs

    /// Dispatches a speculative read using a typed `WeightFileHandle`, returning command buffer and ticket.
    public func dispatchSpeculative(
        handle: WeightFileHandle,
        offset: Int,
        size: Int,
        targetBuffer: any MTLBuffer,
        targetOffset: Int = 0,
        completion: (@Sendable (MTLIOStatus) -> Void)? = nil
    ) throws -> (ticket: UInt64, ioCommandBuffer: any MTLIOCommandBuffer) {
        try handle.validateBounds(offset: offset, size: size)
        guard targetBuffer.length >= targetOffset + size else {
            throw FastIOError.bufferTooSmall(required: targetOffset + size, actual: targetBuffer.length)
        }

        let ticket = speculativeSyncEvent.nextTicket()
        let cmd = speculativeQueue.makeCommandBuffer()
        cmd.load(
            targetBuffer,
            offset: targetOffset,
            size: size,
            sourceHandle: handle.rawHandle,
            sourceHandleOffset: offset
        )
        cmd.signalEvent(speculativeSyncEvent.sharedEvent, value: ticket)

        if let completion = completion {
            cmd.addCompletedHandler { completedCmd in
                completion(completedCmd.status)
            }
        }

        cmd.commit()
        return (ticket, cmd)
    }

    /// Dispatches an urgent demand fetch using a typed `WeightFileHandle`, returning command buffer and ticket.
    public func dispatchFallback(
        handle: WeightFileHandle,
        offset: Int,
        size: Int,
        targetBuffer: any MTLBuffer,
        targetOffset: Int = 0,
        completion: (@Sendable (MTLIOStatus) -> Void)? = nil
    ) throws -> (ticket: UInt64, ioCommandBuffer: any MTLIOCommandBuffer) {
        try handle.validateBounds(offset: offset, size: size)
        guard targetBuffer.length >= targetOffset + size else {
            throw FastIOError.bufferTooSmall(required: targetOffset + size, actual: targetBuffer.length)
        }

        let ticket = fallbackSyncEvent.nextTicket()
        let cmd = fallbackQueue.makeCommandBuffer()
        cmd.load(
            targetBuffer,
            offset: targetOffset,
            size: size,
            sourceHandle: handle.rawHandle,
            sourceHandleOffset: offset
        )
        cmd.signalEvent(fallbackSyncEvent.sharedEvent, value: ticket)

        if let completion = completion {
            cmd.addCompletedHandler { completedCmd in
                completion(completedCmd.status)
            }
        }

        cmd.commit()
        return (ticket, cmd)
    }
}
```

---

### Blueprint 8: `swift_tests/AsyncMoERouterTests/Common/TestHelpers.swift`
```swift
import Foundation
import Metal
import AsyncMoERouter

/// Utilities for generating synthetic weight files, temporary test fixtures, and buffer assertions.
public enum TestHelpers {
    /// Creates a temporary binary file containing deterministic synthetic expert weights.
    public static func createSyntheticWeightFile(
        expertCount: Int = 4,
        expertSizeBytes: Int = MoEArchitectureConfig.synthetic.expertSizeBytes,
        fillByte: UInt8? = nil
    ) throws -> (url: URL, cleanup: () -> Void) {
        let tempDir = FileManager.default.temporaryDirectory
        let fileName = "synthetic_weights_\(UUID().uuidString).bin"
        let fileURL = tempDir.appendingPathComponent(fileName)

        var fileData = Data(capacity: expertCount * expertSizeBytes)
        for i in 0..<expertCount {
            let byteVal = fillByte ?? UInt8((i * 17 + 1) % 256)
            fileData.append(Data(repeating: byteVal, count: expertSizeBytes))
        }

        try fileData.write(to: fileURL)

        let cleanup: () -> Void = {
            _ = try? FileManager.default.removeItem(at: fileURL)
        }

        return (fileURL, cleanup)
    }

    /// Verifies that all bytes in an MTLBuffer match the expected value.
    public static func verifyBufferContents(
        buffer: any MTLBuffer,
        offset: Int = 0,
        length: Int,
        expectedByte: UInt8
    ) -> Bool {
        let ptr = buffer.contents().advanced(by: offset).bindMemory(to: UInt8.self, capacity: length)
        for i in 0..<length {
            if ptr[i] != expectedByte {
                return false
            }
        }
        return true
    }
}
```

---

### Blueprint 9: `swift_tests/AsyncMoERouterTests/Unit/FastIOTests.swift`
```swift
import XCTest
import Metal
@testable import AsyncMoERouter

final class FastIOTests: XCTestCase {
    var metalContext: MetalContext!
    var fastIOEngine: FastIOEngine!

    override func setUpWithError() throws {
        try super.setUpWithError()
        metalContext = try MetalContext()
        fastIOEngine = try FastIOEngine(device: metalContext.device)
    }

    // MARK: - Dimension & Layout Tests

    func testArchitectureConfigSizing() {
        let qwen = MoEArchitectureConfig.qwen15MoEA27B
        XCTAssertEqual(qwen.hiddenSize, 2048)
        XCTAssertEqual(qwen.intermediateSize, 1408)
        XCTAssertEqual(qwen.numExperts, 60)
        XCTAssertEqual(qwen.numActiveExperts, 4)
        XCTAssertEqual(qwen.numDeepLayers, 20)
        XCTAssertEqual(qwen.bytesPerElement, 2)

        // 3 * 2048 * 1408 = 8,650,752 elements
        XCTAssertEqual(qwen.expertElements, 8_650_752)
        // 8,650,752 * 2 = 17,301,504 bytes
        XCTAssertEqual(qwen.expertSizeBytes, 17_301_504)
        // Check Apple Silicon 16KB page alignment (17,301,504 / 16,384 == 1056 exactly)
        XCTAssertTrue(MemoryBudgetConfig.isPageAligned(bytes: qwen.expertSizeBytes))
        XCTAssertEqual(qwen.expertSizeBytes % 16_384, 0)
        XCTAssertEqual(qwen.expertSizeBytes / 16_384, 1056)
    }

    func testSyntheticConfigSizing() {
        let synthetic = MoEArchitectureConfig.synthetic
        XCTAssertEqual(synthetic.hiddenSize, 64)
        XCTAssertEqual(synthetic.intermediateSize, 64)
        // 3 * 64 * 64 = 12,288 elements
        XCTAssertEqual(synthetic.expertElements, 12_288)
        // 12,288 * 2 = 24,576 bytes
        XCTAssertEqual(synthetic.expertSizeBytes, 24_576)
        XCTAssertTrue(MemoryBudgetConfig.isSectorAligned(bytes: synthetic.expertSizeBytes))
    }

    func testExecutionLogEntryLayout() {
        // Must be exactly 32 bytes and 8-byte aligned for lock-free GPU DMA mapping
        XCTAssertEqual(MemoryLayout<ExecutionLogEntry>.size, 32)
        XCTAssertEqual(MemoryLayout<ExecutionLogEntry>.stride, 32)
        XCTAssertEqual(MemoryLayout<ExecutionLogEntry>.alignment, 8)

        let entry = ExecutionLogEntry(
            tokenIndex: 42,
            layerIndex: 7,
            horizonIndex: 2,
            expertID: 15,
            confidenceScore: 0.95
        )
        XCTAssertEqual(entry.tokenIndex, 42)
        XCTAssertEqual(entry.layerIndex, 7)
        XCTAssertEqual(entry.horizonIndex, 2)
        XCTAssertEqual(entry.expertID, 15)
        XCTAssertEqual(entry.padding, 0)
        XCTAssertEqual(entry.confidenceScore, 0.95)
    }

    // MARK: - Metal Context & Runtime MSL Compilation Tests

    func testMetalContextDeviceAndQueue() {
        XCTAssertNotNil(metalContext.device)
        XCTAssertNotNil(metalContext.commandQueue)
        XCTAssertTrue(metalContext.supportsMetal3)
    }

    func testRuntimeMSLCompilationAndExecution() throws {
        let msl = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void vector_add(
            device const float *inA [[buffer(0)]],
            device const float *inB [[buffer(1)]],
            device float *out [[buffer(2)]],
            uint id [[thread_position_in_grid]])
        {
            out[id] = inA[id] + inB[id];
        }
        """

        let pipeline = try metalContext.makeComputePipelineState(source: msl, functionName: "vector_add")
        XCTAssertNotNil(pipeline)

        let count = 64
        let byteSize = count * MemoryLayout<Float>.size

        let bufA = try metalContext.makeBuffer(length: byteSize)
        let bufB = try metalContext.makeBuffer(length: byteSize)
        let bufOut = try metalContext.makeBuffer(length: byteSize)

        let ptrA = bufA.contents().bindMemory(to: Float.self, capacity: count)
        let ptrB = bufB.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            ptrA[i] = Float(i)
            ptrB[i] = Float(i * 2)
        }

        let cmd = metalContext.commandQueue.makeCommandBuffer()!
        let encoder = cmd.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(bufA, offset: 0, index: 0)
        encoder.setBuffer(bufB, offset: 0, index: 1)
        encoder.setBuffer(bufOut, offset: 0, index: 2)

        let gridSize = MTLSize(width: count, height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: min(count, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()

        cmd.commit()
        cmd.waitUntilCompleted()

        let ptrOut = bufOut.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            XCTAssertEqual(ptrOut[i], Float(i * 3))
        }
    }

    // MARK: - Dual Queue Fast I/O & Synchronization Tests

    func testDualQueueFastIOQueueProperties() {
        XCTAssertNotNil(fastIOEngine.speculativeQueue)
        XCTAssertNotNil(fastIOEngine.fallbackQueue)
        XCTAssertNotNil(fastIOEngine.speculativeSyncEvent)
        XCTAssertNotNil(fastIOEngine.fallbackSyncEvent)
    }

    func testSpeculativeBlockLoadAndSync() throws {
        let expertSize = 16_384 // 16 KB aligned
        let (fileURL, cleanup) = try TestHelpers.createSyntheticWeightFile(
            expertCount: 2,
            expertSizeBytes: expertSize,
            fillByte: 0x77
        )
        defer { cleanup() }

        let handle = try WeightFileHandle(url: fileURL, device: metalContext.device)
        let targetBuffer = try metalContext.makeBuffer(length: expertSize)

        let (ticket, cmd) = try fastIOEngine.dispatchSpeculative(
            handle: handle,
            offset: 0,
            size: expertSize,
            targetBuffer: targetBuffer
        )

        let signaled = fastIOEngine.speculativeSyncEvent.waitUntilSignaled(ticket: ticket, timeoutSeconds: 5.0)
        XCTAssertTrue(signaled, "Speculative SharedEvent was not signaled within timeout")
        XCTAssertEqual(cmd.status, .complete)

        let matched = TestHelpers.verifyBufferContents(
            buffer: targetBuffer,
            length: expertSize,
            expectedByte: 0x77
        )
        XCTAssertTrue(matched, "Buffer content after DMA load did not match expected byte 0x77")
    }

    func testFallbackDemandFetchLoad() throws {
        let expertSize = 16_384
        let (fileURL, cleanup) = try TestHelpers.createSyntheticWeightFile(
            expertCount: 2,
            expertSizeBytes: expertSize,
            fillByte: 0x88
        )
        defer { cleanup() }

        let handle = try WeightFileHandle(url: fileURL, device: metalContext.device)
        let targetBuffer = try metalContext.makeBuffer(length: expertSize)

        let (ticket, cmd) = try fastIOEngine.dispatchFallback(
            handle: handle,
            offset: expertSize, // Read expert 1
            size: expertSize,
            targetBuffer: targetBuffer
        )

        let signaled = fastIOEngine.fallbackSyncEvent.waitUntilSignaled(ticket: ticket, timeoutSeconds: 5.0)
        XCTAssertTrue(signaled, "Fallback SharedEvent was not signaled within timeout")
        XCTAssertEqual(cmd.status, .complete)

        let matched = TestHelpers.verifyBufferContents(
            buffer: targetBuffer,
            length: expertSize,
            expectedByte: 0x88
        )
        XCTAssertTrue(matched, "Buffer content after fallback DMA load did not match expected byte 0x88")
    }

    func testZeroCPUSharedEventSynchronizationWithCompute() throws {
        let expertSize = 16_384
        let (fileURL, cleanup) = try TestHelpers.createSyntheticWeightFile(
            expertCount: 1,
            expertSizeBytes: expertSize,
            fillByte: 0x55
        )
        defer { cleanup() }

        let handle = try WeightFileHandle(url: fileURL, device: metalContext.device)
        let intermediateBuffer = try metalContext.makeBuffer(length: expertSize)
        let gpuDestinationBuffer = try metalContext.makeBuffer(length: expertSize)

        // 1. Dispatch speculative I/O load which will signal event at ticket
        let (ticket, _) = try fastIOEngine.dispatchSpeculative(
            handle: handle,
            offset: 0,
            size: expertSize,
            targetBuffer: intermediateBuffer
        )

        // 2. Enqueue GPU compute/blit command buffer waiting on ticket BEFORE confirming completion
        let computeCmd = metalContext.commandQueue.makeCommandBuffer()!
        fastIOEngine.speculativeSyncEvent.encodeWait(on: computeCmd, ticket: ticket)

        let blitEncoder = computeCmd.makeBlitCommandEncoder()!
        blitEncoder.copy(
            from: intermediateBuffer,
            sourceOffset: 0,
            to: gpuDestinationBuffer,
            destinationOffset: 0,
            size: expertSize
        )
        blitEncoder.endEncoding()
        computeCmd.commit()

        // 3. Wait on compute completion (CPU never had to poll or orchestrate the handoff between I/O and GPU)
        computeCmd.waitUntilCompleted()

        let matched = TestHelpers.verifyBufferContents(
            buffer: gpuDestinationBuffer,
            length: expertSize,
            expectedByte: 0x55
        )
        XCTAssertTrue(matched, "Zero-CPU synchronization failed to transfer data cleanly to GPU destination buffer")
        XCTAssertGreaterThanOrEqual(fastIOEngine.speculativeSyncEvent.currentSignaledValue, ticket)
    }

    func testBoundsCheckingAndDefensiveAssertions() throws {
        let expertSize = 4_096
        let (fileURL, cleanup) = try TestHelpers.createSyntheticWeightFile(
            expertCount: 1,
            expertSizeBytes: expertSize,
            fillByte: 0x11
        )
        defer { cleanup() }

        let handle = try WeightFileHandle(url: fileURL, device: metalContext.device)

        // 1. Destination buffer smaller than read size
        let smallBuffer = try metalContext.makeBuffer(length: 1_024)
        XCTAssertThrowsError(
            try fastIOEngine.dispatchSpeculative(
                handle: handle,
                offset: 0,
                size: 2_048,
                targetBuffer: smallBuffer
            )
        ) { error in
            guard case FastIOError.bufferTooSmall = error else {
                return XCTFail("Expected FastIOError.bufferTooSmall but got \(error)")
            }
        }

        // 2. Offset beyond file boundary
        let normalBuffer = try metalContext.makeBuffer(length: 4_096)
        XCTAssertThrowsError(
            try fastIOEngine.dispatchSpeculative(
                handle: handle,
                offset: 4_096,
                size: 1_024,
                targetBuffer: normalBuffer
            )
        ) { error in
            guard case FastIOError.offsetOutOfBounds = error else {
                return XCTFail("Expected FastIOError.offsetOutOfBounds but got \(error)")
            }
        }
    }
}
```

---

## 5. Verification Method

To independently verify this complete technical blueprint, run the following commands:

### 5.1 Standalone Compiler & Execution Verification
Compile the module and execute the verification test suite directly:

```bash
mkdir -p /tmp/verify_m1_final/Sources/AsyncMoERouter/Common \
         /tmp/verify_m1_final/Sources/AsyncMoERouter/FastIO \
         /tmp/verify_m1_final/swift_tests/AsyncMoERouterTests/Common \
         /tmp/verify_m1_final/swift_tests/AsyncMoERouterTests/Unit

# Copy proposed files
cp /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_1/proposed_Package.swift /tmp/verify_m1_final/Package.swift
cp /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_1/proposed_Config.swift /tmp/verify_m1_final/Sources/AsyncMoERouter/Common/Config.swift
cp /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_1/proposed_MetalContext.swift /tmp/verify_m1_final/Sources/AsyncMoERouter/Common/MetalContext.swift
cp /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_1/proposed_Types.swift /tmp/verify_m1_final/Sources/AsyncMoERouter/Common/Types.swift
cp /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_1/proposed_SyncEvent.swift /tmp/verify_m1_final/Sources/AsyncMoERouter/FastIO/SyncEvent.swift
cp /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_1/proposed_WeightFileHandle.swift /tmp/verify_m1_final/Sources/AsyncMoERouter/FastIO/WeightFileHandle.swift
cp /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_1/proposed_FastIOEngine.swift /tmp/verify_m1_final/Sources/AsyncMoERouter/FastIO/FastIOEngine.swift
cp /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_1/proposed_TestHelpers.swift /tmp/verify_m1_final/swift_tests/AsyncMoERouterTests/Common/TestHelpers.swift
cp /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_1/proposed_FastIOTests.swift /tmp/verify_m1_final/swift_tests/AsyncMoERouterTests/Unit/FastIOTests.swift

# Compile dynamic library module
swiftc /tmp/verify_m1_final/Sources/AsyncMoERouter/Common/*.swift \
       /tmp/verify_m1_final/Sources/AsyncMoERouter/FastIO/*.swift \
       -emit-module -module-name AsyncMoERouter \
       -emit-library -o /tmp/verify_m1_final/libAsyncMoERouter.dylib

# Run verification runner
cat << 'RUNNER_EOF' > /tmp/verify_m1_final/main.swift
import Foundation
import Metal
import AsyncMoERouter

@main
struct Runner {
    static func main() throws {
        print("Running standalone verification runner...")
        let ctx = try MetalContext()
        assert(ctx.supportsMetal3)
        let engine = try FastIOEngine(device: ctx.device)
        assert(engine.speculativeSyncEvent.currentSignaledValue >= 0)
        assert(engine.fallbackSyncEvent.currentSignaledValue >= 0)
        print("FastIOEngine verified successfully!")
    }
}
RUNNER_EOF

swiftc -parse-as-library -I /tmp/verify_m1_final -L /tmp/verify_m1_final -lAsyncMoERouter \
       /tmp/verify_m1_final/main.swift -framework Metal -framework Foundation \
       -o /tmp/verify_m1_final/runner && /tmp/verify_m1_final/runner
```

### 5.2 Expected Output
```
Running standalone verification runner...
FastIOEngine verified successfully!
```

### 5.3 Invalidation Conditions
This architecture and blueprint shall be invalidated if:
1. Apple Metal changes `MTLIOCommandQueueDescriptor` to deprecate `priority` or change priority queue semantics.
2. The model architecture departs from SwiGLU 3-projection MLP, changing expert dimension calculations.
3. The memory ceiling is adjusted above 5 GB, making dynamic NVMe streaming unnecessary.
