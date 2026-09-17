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
