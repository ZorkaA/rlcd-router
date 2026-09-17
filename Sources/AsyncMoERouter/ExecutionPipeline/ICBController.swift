import Foundation
import Metal
import os.log

/// Manages the Metal Indirect Command Buffer (ICB) for conditionally dispatching expert kernels.
///
/// # ICB Conditional Execution
/// When the Gating Kernel detects a cache miss (`*abort_flag == true`), it writes
/// `uint3(0,0,0)` to the ICB execution grid. The Metal driver natively skips
/// zero-thread ICB commands without launching any GPU threads.
///
/// This saves the GPU from spinning up threadgroups for ALL subsequent expert kernels
/// in the pipeline, avoiding the overhead of the cascading `return;` pattern for
/// dynamically-encoded ICB dispatches.
///
/// The cascading `return;` no-op pattern is complementary — it applies to **standard**
/// layer kernels (attention, norms, etc.) that are encoded directly on the
/// `MTLComputeCommandEncoder`, not via ICB.
public final class ICBController: @unchecked Sendable {
    // MARK: - Public buffers
    /// The argument buffer wrapping the ICB (bind as `buffer(N)` with struct type `ICBContainer`).
    public let icbArgumentBuffer: any MTLBuffer
    /// Alias for test convenience.
    public var argumentBuffer: (any MTLBuffer)? { icbArgumentBuffer }
    /// The underlying ICB (retain reference to prevent deallocation).
    public private(set) var icb: (any MTLIndirectCommandBuffer)?

    private let _device: any MTLDevice
    private let _log = Logger(subsystem: "AsyncMoERouter", category: "ICBController")
    private let _maxCommands: Int

    // MARK: - Init

    /// Creates the ICB controller for a given expert pipeline stage.
    ///
    /// - Parameters:
    ///   - device: Metal device.
    ///   - maxCommands: Maximum number of indirect commands (one per expert dispatch). Default: 64.
    public init(device: any MTLDevice, maxCommands: Int = 64) {
        _device = device
        _maxCommands = maxCommands

        // Create ICB descriptor
        let descriptor = MTLIndirectCommandBufferDescriptor()
        descriptor.commandTypes = [.concurrentDispatch]
        descriptor.inheritBuffers = false
        descriptor.inheritPipelineState = true
        descriptor.maxKernelBufferBindCount = 8

        guard let icbObj = device.makeIndirectCommandBuffer(
            descriptor: descriptor,
            maxCommandCount: maxCommands,
            options: .storageModeShared
        ) else {
            fatalError("ICBController: Failed to create MTLIndirectCommandBuffer")
        }
        self.icb = icbObj

        // Create the argument buffer that wraps the ICB
        // Size: pointer-sized handle (8 bytes) with padding
        guard let argBuf = device.makeBuffer(length: 64, options: .storageModeShared) else {
            fatalError("ICBController: Failed to create ICB argument buffer")
        }
        argBuf.label = "ICBArgumentBuffer"
        self.icbArgumentBuffer = argBuf

        _log.info("ICBController initialized: maxCommands=\(maxCommands)")
    }

    // MARK: - Reset

    /// Resets all ICB commands to zero-thread dispatches at the start of each generation step.
    /// Call before encoding the Gating Kernel.
    public func reset() {
        guard let icbObj = icb else { return }
        let resetEncoder = icbObj.indirectComputeCommandAt(0)
        resetEncoder.concurrentDispatchThreads(
            MTLSize(width: 0, height: 0, depth: 0),
            threadsPerThreadgroup: MTLSize(width: 0, height: 0, depth: 0)
        )
        _log.debug("ICBController: Reset to zero-thread dispatch")
    }

    // MARK: - Encoding

    /// Encodes a normal expert kernel dispatch into the ICB at the specified command index.
    ///
    /// This is called by the CPU when pre-encoding a speculative expert execution.
    /// The Gating Kernel will overwrite with `uint3(0,0,0)` if it detects a cache miss.
    public func encodeExpertDispatch(
        at commandIndex: Int,
        gridSize: MTLSize,
        threadgroupSize: MTLSize
    ) {
        guard let icbObj = icb, commandIndex < _maxCommands else {
            _log.error("ICBController: encodeExpertDispatch out of bounds (index=\(commandIndex), max=\(self._maxCommands))")
            return
        }
        let cmd = icbObj.indirectComputeCommandAt(commandIndex)
        cmd.concurrentDispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        _log.debug("ICBController: Expert dispatch encoded at index \(commandIndex)")
    }

    // MARK: - Execution

    /// Encodes the ICB execution on a compute command encoder.
    ///
    /// - Parameters:
    ///   - encoder: The active `MTLComputeCommandEncoder`.
    ///   - commandCount: Number of ICB commands to execute.
    public func execute(on encoder: any MTLComputeCommandEncoder, commandCount: Int) {
        guard let icbObj = icb else { return }
        encoder.executeCommandsInBuffer(icbObj, range: 0..<commandCount)
    }
}
