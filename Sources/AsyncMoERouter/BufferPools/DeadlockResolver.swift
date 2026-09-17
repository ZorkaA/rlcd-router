import Foundation
import Metal
import os.log

/// Resolves cache-miss deadlocks between the Speculative Ring Buffer and the Fallback Pool.
///
/// # Protocol
/// When the Speculative Ring Buffer is entirely occupied with in-flight or in-use slots
/// and the current token requires an expert that hasn't been prefetched, the pipeline
/// must never overwrite a speculative slot that is still receiving a DMA transfer (risk
/// of silent memory corruption). Instead:
///
/// 1. Allocate a slot from the isolated `FallbackBufferPool` (never the Ring Buffer).
/// 2. Issue a PriorityHigh I/O read on the `fallbackQueue`.
/// 3. Mark the speculative Ring Buffer slot as `.abandoned` and call `tryCancel()`
///    for a best-effort PCIe bandwidth saving — the physical DMA may have already
///    started and cannot be safely interrupted.
/// 4. When the abandoned slot's IO completion fires, drop the signal and reclaim to `.free`.
///
/// # Note on MTLIOCommandBuffer Cancellation
/// `MTLIOCommandBuffer.tryCancel()` submits a cancellation *request* to the queue
/// but the NVMe DMA controller may have already commenced the physical transfer.
/// Therefore we NEVER reuse a speculative slot's memory until its completion handler
/// fires — the Fallback Pool's separate memory region is the safe path.
public final class DeadlockResolver: @unchecked Sendable {
    private let _fallbackPool: FallbackBufferPool
    private let _ringBuffer: SpeculativeRingBuffer
    private let _fastIO: any FastIOEngineProtocol
    private let _log = Logger(subsystem: "AsyncMoERouter", category: "DeadlockResolver")

    // Diagnostics
    public private(set) var totalDeadlocksResolved: Int = 0

    public init(
        ringBuffer: SpeculativeRingBuffer,
        fallbackPool: FallbackBufferPool,
        fastIO: any FastIOEngineProtocol
    ) {
        _ringBuffer = ringBuffer
        _fallbackPool = fallbackPool
        _fastIO = fastIO
    }

    /// Resolves a cache-miss for `expert` by issuing a synchronous demand fetch into the Fallback Pool.
    ///
    /// - Parameters:
    ///   - expertID: The expert key that was not found in the Ring Buffer.
    ///   - fileHandle: The `MTLIOFileHandle` for the weight file.
    ///   - fileOffset: Byte offset of the expert in the weight file.
    ///   - expertSize: Byte size of the expert weight tensor.
    ///   - device: MTLDevice for growing the Fallback Pool if needed.
    ///   - stalledRingSlotIndex: Optional index of a speculative slot to abandon (if identified).
    /// - Returns: The `FallbackSlot` holding the fetched weights.
    /// - Throws: `FastIOError` if the Fallback Pool is exhausted or the I/O read fails.
    @discardableResult
    public func resolveDeadlock(
        expertID: ExpertKey,
        fileHandle: any MTLIOFileHandle,
        fileOffset: Int,
        expertSize: Int,
        device: any MTLDevice,
        stalledRingSlotIndex: Int? = nil
    ) async throws -> FallbackSlot {
        _log.warning("DeadlockResolver: Cache miss on \(expertID.description) — initiating fallback fetch")

        // Step 1: Abandon the stalled speculative slot if identified
        if let stallIdx = stalledRingSlotIndex {
            _ringBuffer.markAbandoned(slotIndex: stallIdx)
            _log.info("DeadlockResolver: Ring Buffer slot[\(stallIdx)] marked abandoned")
        }

        // Step 2: Allocate from the isolated Fallback Pool
        guard let slot = _fallbackPool.allocate(device: device) else {
            _log.error("DeadlockResolver: Fallback Pool exhausted — cannot resolve deadlock for \(expertID.description)")
            throw FastIOError.bufferTooSmall(required: expertSize, actual: 0)
        }

        // Step 3: Issue PriorityHigh demand fetch
        let (event, signalVal) = try _fastIO.loadFallback(
            handle: fileHandle,
            offset: fileOffset,
            size: expertSize,
            targetBuffer: slot.buffer,
            targetOffset: 0
        )
        slot.sharedEvent = event
        slot.signalValue = signalVal

        // Step 4: Wait for the high-priority fetch to complete
        // In the real pipeline this is handled by encoding an MTLSharedEvent wait
        // on the compute command buffer. For the Swift-only path we spin with a timeout.
        let deadline = Date().addingTimeInterval(5.0)
        while event.signaledValue < signalVal {
            if Date() > deadline {
                _log.error("DeadlockResolver: Fallback fetch timeout for \(expertID.description)")
                _fallbackPool.reclaim(slot)
                throw FastIOError.ioExecutionFailed(status: -1, description: "Fallback fetch timed out")
            }
            try await Task.sleep(nanoseconds: 100_000) // 0.1ms
        }

        totalDeadlocksResolved += 1
        _log.info("DeadlockResolver: Fallback fetch completed for \(expertID.description), slot[\(slot.index)]")
        return slot
    }
}
