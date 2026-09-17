import Foundation
import Metal
import os.log

/// Unified asynchronous MoE pipeline coordinator binding all Phase 2 subsystems.
///
/// This is the top-level entry point for single-token inference steps. It orchestrates:
/// 1. Speculative expert prefetch via `FastIOEngine` + `SpeculativeRingBuffer`
/// 2. Cache-miss resolution via `DeadlockResolver` + `FallbackBufferPool`
/// 3. GPU execution with `AbortController` + `ICBController`
/// 4. Post-execution LRU update via `GPUExecutionLog` + `LRUWeightTracker`
/// 5. Background Brier-score recalibration via `RecalibrationActor`
///
/// # Memory Budget Summary
/// - Speculative Ring Buffer: 16 slots × expertSizeBytes
/// - Fallback Buffer Pool: up to 500 MB
/// - GPU Execution Log: 128 KB
/// - Abort Flag Buffer: 16 bytes
/// - MLX Cache: capped at 200 MB
///
/// Total conservative upper bound: Ring + 500MB + 200MB + ~128KB
/// For Qwen1.5-MoE-A2.7B: 16 × 17.3MB + 500MB + 200MB ≈ 977MB — well within system headroom.
public final class AsyncMoEPipeline: @unchecked Sendable {
    // MARK: - Core subsystems
    public let fastIO: any FastIOEngineProtocol
    public let ringBuffer: SpeculativeRingBuffer
    public let fallbackPool: FallbackBufferPool
    public let deadlockResolver: DeadlockResolver
    public let executionLog: GPUExecutionLog
    public let lruTracker: LRUWeightTracker
    public let abortController: AbortController
    public let icbController: ICBController
    public let recalibrationActor: RecalibrationActor

    private let _device: any MTLDevice
    private let _archConfig: MoEArchitectureConfig
    private let _budgetConfig: MemoryBudgetConfig
    private let _log = Logger(subsystem: "AsyncMoERouter", category: "AsyncMoEPipeline")

    // MARK: - Metrics
    public private(set) var totalTokensProcessed: Int = 0
    public private(set) var totalCacheHits: Int = 0
    public private(set) var totalCacheMisses: Int = 0
    public private(set) var totalAborts: Int = 0

    // MARK: - Init

    /// Creates and initializes the full Phase 2 pipeline.
    ///
    /// - Parameters:
    ///   - device: The primary Metal device.
    ///   - fastIO: Pre-configured Fast I/O engine.
    ///   - archConfig: MoE architecture dimensions.
    ///   - budgetConfig: Memory budget parameters.
    public init(
        device: any MTLDevice,
        fastIO: any FastIOEngineProtocol,
        archConfig: MoEArchitectureConfig = .qwen15MoEA27B,
        budgetConfig: MemoryBudgetConfig = .default
    ) {
        self._device = device
        self._archConfig = archConfig
        self._budgetConfig = budgetConfig
        self.fastIO = fastIO

        // M2: Buffer pools
        let ring = SpeculativeRingBuffer(
            device: device,
            slotCount: budgetConfig.speculativeRingBufferSlots,
            slotSizeBytes: archConfig.expertSizeBytes,
            budgetConfig: budgetConfig
        )
        let fallback = FallbackBufferPool(
            device: device,
            slotSizeBytes: archConfig.expertSizeBytes,
            maxCapacityBytes: budgetConfig.fallbackPoolMaxBytes
        )
        self.ringBuffer = ring
        self.fallbackPool = fallback
        self.deadlockResolver = DeadlockResolver(
            ringBuffer: ring,
            fallbackPool: fallback,
            fastIO: fastIO
        )

        // M3: Execution log + LRU tracker
        self.executionLog = GPUExecutionLog(
            device: device,
            capacity: budgetConfig.executionLogCapacity
        )
        self.lruTracker = LRUWeightTracker()

        // M4: Abort flag + ICB controller
        self.abortController = AbortController(device: device)
        self.icbController = ICBController(device: device)

        // M5: Recalibration actor
        self.recalibrationActor = RecalibrationActor()

        _log.info("""
            AsyncMoEPipeline initialized:
              Ring buffer: \(budgetConfig.speculativeRingBufferSlots) slots × \(archConfig.expertSizeBytes) bytes
              Fallback pool: \(budgetConfig.fallbackPoolMaxBytes) bytes
              Execution log: \(budgetConfig.executionLogCapacity) entries
              MLX cache limit: \(budgetConfig.mlxCacheLimitBytes) bytes
            """)
    }

    // MARK: - Step Lifecycle

    /// Prepares the pipeline for a new generation step: resets abort flag and ICB.
    public func beginStep() {
        abortController.reset()
        icbController.reset()
    }

    /// Performs post-step bookkeeping: drains the Execution Log, updates LRU timestamps,
    /// and checks if recalibration is triggered.
    public func endStep() async {
        // Drain execution log and update LRU tracker
        let entries = executionLog.drain()
        for entry in entries {
            let key = ExpertKey(layer: Int(entry.layerIndex), expert: Int(entry.expertID))
            lruTracker.recordAccess(expert: key, timestamp: entry.timestamp)
        }

        // Feed entries to recalibration actor for background processing
        await recalibrationActor.ingestEntries(entries)

        totalTokensProcessed += 1
    }

    /// Synchronous variant of endStep for non-async contexts and unit testing.
    public func endStep() {
        let entries = executionLog.drain()
        for entry in entries {
            let key = ExpertKey(layer: Int(entry.layerIndex), expert: Int(entry.expertID))
            lruTracker.recordAccess(expert: key, timestamp: entry.timestamp)
        }

        Task {
            await recalibrationActor.ingestEntries(entries)
        }

        totalTokensProcessed += 1
    }

    /// Called when a new user request arrives — cancels any in-progress recalibration
    /// to yield GPU resources for inference.
    public func onUserRequestReceived() async {
        await recalibrationActor.cancelCurrentPass()
        _log.info("Pipeline: User request received — recalibration pre-empted")
    }

    /// Triggers a background recalibration pass during an idle window.
    public func runIdleRecalibration() async {
        await recalibrationActor.resetCancellation()
        let newT = await recalibrationActor.runPassIfReady()
        if let t = newT {
            _log.info("Pipeline: Idle recalibration complete — new temperature: \(t)")
        }
    }

    // MARK: - Cache Resolution

    /// Checks whether a given expert is speculatively cached and ready.
    public func isExpertCached(_ expert: ExpertKey) -> Bool {
        return ringBuffer.findReadySlot(for: expert) != nil
    }

    /// Records a cache hit for metrics purposes.
    public func recordCacheHit() { totalCacheHits += 1 }

    /// Records a cache miss for metrics purposes.
    public func recordCacheMiss() { totalCacheMisses += 1 }

    /// Records an abort for metrics purposes.
    public func recordAbort() { totalAborts += 1 }

    /// Speculatively hints that `key` should be prefetched. No-op if the slot is already cached.
    /// Returns `true` if a new prefetch was scheduled, `false` if already resident.
    @discardableResult
    public func prefetchExpert(key: ExpertKey) -> Bool {
        // If slot already ready, nothing to do
        if ringBuffer.findReadySlot(for: key) != nil { return false }
        // Schedule allocation attempt (no actual I/O without a WeightFileHandle)
        _ = ringBuffer.allocateSlot(for: key, ticket: UInt64(Date().timeIntervalSince1970 * 1_000_000))
        return true
    }

    /// Returns `true` if the abort flag is currently set.
    public var isAbortSet: Bool { abortController.isAborted }

    /// Arms the abort flag from the CPU (e.g. for testing or user-request pre-emption).
    public func triggerAbort() { abortController.set(true) }

    /// Approximate peak memory in bytes consumed by all subsystems.
    /// Computed from the current high-water mark of the fallback pool + ring buffer slots.
    public var peakMemoryBytes: Int {
        ringBuffer.slotCount * ringBuffer.slotSizeBytes + fallbackPool.allocatedBytes
    }

    // MARK: - Diagnostics

    /// Returns a compact summary of current pipeline health.
    public func diagnostics() -> String { diagnosticsSummary }

    /// Returns a compact summary of current pipeline health.
    public var diagnosticsSummary: String {
        let hitRate = totalTokensProcessed > 0
            ? String(format: "%.1f%%", Float(totalCacheHits) / Float(totalTokensProcessed) * 100)
            : "N/A"
        return """
            [Pipeline Diagnostics]
              Tokens processed: \(totalTokensProcessed)
              Cache hits: \(totalCacheHits) (\(hitRate))
              Cache misses: \(totalCacheMisses)
              Aborts: \(totalAborts)
              Ring buffer: \(ringBuffer.slotCount) slots
              Fallback pool: \(fallbackPool.inUseSlotCount) in use, \(fallbackPool.freeSlotCount) free
              LRU tracker: \(lruTracker.count) resident experts
            """
    }
}

// MARK: - Convenience init for tests (no external FastIOEngine required)
public extension AsyncMoEPipeline {
    /// Convenience initializer for unit tests — creates an internal `FastIOEngine`
    /// so tests don't need to manually construct the full dependency graph.
    convenience init(
        device: any MTLDevice,
        config: MoEArchitectureConfig,
        budget: MemoryBudgetConfig
    ) throws {
        let engine = try FastIOEngine(device: device)
        self.init(
            device: device,
            fastIO: engine,
            archConfig: config,
            budgetConfig: budget
        )
    }
}

