import Foundation
import Metal
import os

/// Protocol defining the interface between the Fast I/O subsystem and memory pool managers (M1 ↔ M2).
public protocol FastIOEngineProtocol: Sendable {
    /// Dispatches a low-priority speculative prefetch read.
    ///
    /// - Parameters:
    ///   - handle: Source file handle on disk.
    ///   - offset: Source file offset in bytes.
    ///   - size: Number of bytes to transfer via DMA.
    ///   - targetBuffer: Destination MTLBuffer in unified memory.
    ///   - targetOffset: Offset within targetBuffer where DMA writes begin.
    /// - Returns: Tuple of `(MTLSharedEvent, ticket)` signaled when I/O completes.
    func loadSpeculative(
        handle: any MTLIOFileHandle,
        offset: Int,
        size: Int,
        targetBuffer: any MTLBuffer,
        targetOffset: Int
    ) -> (any MTLSharedEvent, UInt64)

    /// Dispatches a high-priority fallback demand-fetch read.
    ///
    /// - Parameters:
    ///   - handle: Source file handle on disk.
    ///   - offset: Source file offset in bytes.
    ///   - size: Number of bytes to transfer via DMA.
    ///   - targetBuffer: Destination MTLBuffer in unified memory (from isolated Fallback Pool).
    ///   - targetOffset: Offset within targetBuffer where DMA writes begin.
    /// - Returns: Tuple of `(MTLSharedEvent, ticket)` signaled when I/O completes.
    func loadFallback(
        handle: any MTLIOFileHandle,
        offset: Int,
        size: Int,
        targetBuffer: any MTLBuffer,
        targetOffset: Int
    ) -> (any MTLSharedEvent, UInt64)
}

/// Metal 3 Fast I/O Dual-Queue Engine.
///
/// Implements asynchronous NVMe-to-UMA streaming with priority segregation:
/// - `speculativeQueue`: `priority = .low`, `maxCommandBufferCount = 16`, `type = .concurrent`.
///   Handles speculative prefetching driven by speculative routing predictions without starving demand reads.
/// - `fallbackQueue`: `priority = .high`, `maxCommandBufferCount = 16`, `type = .concurrent`.
///   Handles urgent demand fetches on cache misses, preempting background reads at the storage controller.
/// - Hardware synchronization is handled via `MTLSharedEvent` (zero CPU overhead).
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
    ///
    /// - Parameter device: The Metal device to bind (defaults to `MetalContext.shared.device`).
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
    ///
    /// - Parameters:
    ///   - handle: Managed weight file handle.
    ///   - offset: File offset in bytes.
    ///   - size: Number of bytes to transfer.
    ///   - targetBuffer: Destination buffer.
    ///   - targetOffset: Buffer destination offset.
    ///   - completion: Optional async callback invoked upon command buffer completion on host.
    /// - Returns: `(ticket, ioCommandBuffer)`
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
    ///
    /// - Parameters:
    ///   - handle: Managed weight file handle.
    ///   - offset: File offset in bytes.
    ///   - size: Number of bytes to transfer.
    ///   - targetBuffer: Destination buffer.
    ///   - targetOffset: Buffer destination offset.
    ///   - completion: Optional async callback invoked upon command buffer completion on host.
    /// - Returns: `(ticket, ioCommandBuffer)`
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
