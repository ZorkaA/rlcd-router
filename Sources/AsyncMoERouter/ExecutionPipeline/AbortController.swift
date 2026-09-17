import Foundation
import Metal
import os.log

/// Manages the global 1-byte abort flag shared between GPU kernels.
///
/// # Correctness Invariant
/// The `abort_flag` buffer must NEVER be marked `.untracked` (`.resourceOptions` must
/// include the default hazard tracking mode). Metal's default hazard tracking on a
/// `MTLComputeCommandEncoder` guarantees that:
///
/// - A write by the **Gating Kernel** to `abort_flag` is visible to all subsequent
///   kernels dispatched on the **same encoder** without requiring explicit barriers.
/// - If you ever refactor to use `.untracked` for a performance win, you will silently
///   reintroduce the residual stream corruption bug (downstream kernels may observe
///   stale `abort_flag == false` and write garbage into `x += moe_output`).
///
/// # Metal Hazard Tracking Note
/// This relies on `MTLResourceHazardTrackingModeTracked` (the default). Never pass
/// `.resourceOptions = .storageModeShared` together with any mode that suppresses
/// hazard tracking for this buffer.
public final class AbortController: @unchecked Sendable {
    // MARK: - Public buffer (bind to Metal kernels)
    /// The 1-byte global abort flag buffer. Bind as `buffer(N)` in MSL with type `device bool*`.
    public let buffer: any MTLBuffer

    private let _log = Logger(subsystem: "AsyncMoERouter", category: "AbortController")
    private var _ptr: UnsafeMutablePointer<UInt8>

    /// Allocates a 16-byte page-aligned abort flag buffer (1 byte flag + 15 bytes padding).
    /// 16 bytes guarantees Apple Silicon page-alignment for cache-line isolation.
    public init(device: any MTLDevice) {
        // Allocate 16 bytes for cache-line alignment; use only byte [0] as the flag
        guard let buf = device.makeBuffer(length: 16, options: .storageModeShared) else {
            fatalError("AbortController: Failed to allocate abort_flag buffer")
        }
        buf.label = "AbortFlagBuffer"
        // ⚠️  MUST use default hazard tracking — do NOT add .hazardTrackingModeUntracked
        self.buffer = buf
        self._ptr = buf.contents().bindMemory(to: UInt8.self, capacity: 16)
        self._ptr[0] = 0  // Initialize to false
        _log.info("AbortController initialized — hazard tracking: tracked (default)")
    }

    // MARK: - CPU Control

    /// Resets the abort flag to `false` at the start of each generation step.
    /// Call this before encoding the Gating Kernel dispatch.
    public func reset() {
        _ptr[0] = 0
    }

    /// Explicitly sets the abort flag to `value` from the CPU.
    /// Primarily used in tests and abort-injection scenarios.
    /// The GPU sets this flag natively via the MSL Gating Kernel.
    public func set(_ value: Bool) {
        _ptr[0] = value ? 1 : 0
    }

    /// Returns `true` if the abort flag is currently set (either by GPU or CPU).
    public var isAborted: Bool {
        return _ptr[0] != 0
    }

    // MARK: - MSL Shader Strings

    /// MSL source for the Gating Kernel that conditionally writes to the abort flag
    /// and populates the ICB execution grid.
    ///
    /// # ICB Conditional Execution (Expert Kernels)
    /// If `*abort_flag != 0`, the kernel writes `uint3(0,0,0)` to the ICB's thread-grid
    /// size. The Metal driver natively skips zero-thread ICB commands without launching
    /// any GPU threads — saving the entire threadgroup spin-up cost.
    ///
    /// # Cascading No-Ops (Standard Layer Kernels)
    /// Standard layer kernels (norm, attention, etc.) NOT dispatched via ICB must check
    /// `*abort_flag` as their very first instruction and return immediately if set.
    /// This prevents garbage writes to the residual stream `x`.
    ///
    /// # Safety Requirement
    /// Metal hazard tracking on the MTLComputeCommandEncoder MUST be preserved
    /// (no `.untracked` buffers on `abort_flag`). Without it, downstream kernels
    /// may observe a stale `false` and corrupt `x`.
    public static let icbGatingKernelSource: String = """
    #include <metal_stdlib>
    using namespace metal;

    struct ICBContainer {
        command_buffer icb [[id(0)]];
    };

    kernel void gatingKernel(
        device bool*                  abort_flag    [[buffer(0)]],
        constant uint3&               gridSize      [[buffer(1)]],
        constant uint3&               threadgroupSz [[buffer(2)]],
        device ICBContainer&          icbContainer  [[buffer(3)]],
        constant float*               gateProbs     [[buffer(4)]],
        constant uint&                expertCount   [[buffer(5)]],
        constant float&               abortThreshold [[buffer(6)]],
        uint                          tid           [[thread_position_in_grid]]
    ) {
        if (tid != 0) return;

        // Check abort flag from previous step — no-op if already aborted
        if (*abort_flag) {
            // Write zero threads to ICB — Metal driver skips this command natively
            icbContainer.icb.concurrent_dispatch_threads(uint3(0, 0, 0), uint3(0, 0, 0));
            return;
        }

        // Determine top-scoring expert
        float maxProb = 0.0f;
        for (uint i = 0; i < expertCount; i++) {
            maxProb = max(maxProb, gateProbs[i]);
        }

        if (maxProb < abortThreshold) {
            // Cache miss — set abort flag and zero the ICB grid
            *abort_flag = true;
            icbContainer.icb.concurrent_dispatch_threads(uint3(0, 0, 0), uint3(0, 0, 0));
            return;
        }

        // Expert is resident — dispatch normal execution grid
        icbContainer.icb.concurrent_dispatch_threads(gridSize, threadgroupSz);
    }
    """

    /// MSL source for a standard layer kernel (attention, norm, etc.) that implements
    /// the cascading no-op pattern to protect residual stream `x`.
    ///
    /// Every standard kernel checks `abort_flag` **before doing any work**.
    /// This ensures nothing is ever written into `x += moe_output` if the pipeline aborted.
    public static let cascadingLayerKernelSource: String = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void standardLayerKernel(
        device bool*   abort_flag  [[buffer(0)]],
        device float*  residual_x  [[buffer(1)]],
        constant uint& elementCount [[buffer(2)]],
        uint           tid          [[thread_position_in_grid]]
    ) {
        // Cascading No-Op: Check abort_flag BEFORE any computation.
        // Metal hazard tracking guarantees abort_flag visibility from the gating kernel.
        // INVARIANT: Never mark abort_flag buffer without hazard tracking — doing so silently
        // reintroduces residual stream corruption (the exact bug this pattern prevents).
        if (*abort_flag != 0) return;

        if (tid >= elementCount) return;
        // ... actual layer computation would follow here ...
        residual_x[tid] += 0.0f; // placeholder
    }
    """
}
