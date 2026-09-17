import Foundation
import Metal
import os

/// Encapsulates an `MTLSharedEvent` with monotonic ticket generation for zero-CPU hardware synchronization.
///
/// Enables hardware-level command synchronization between Metal 3 Fast I/O command buffers and GPU
/// compute command queues without CPU wakeups or thread scheduling latency.
public final class SyncEvent: @unchecked Sendable {
    /// The underlying Metal shared event.
    public let sharedEvent: any MTLSharedEvent

    /// Monotonic ticket counter protected by an unfair lock.
    private let ticketLock = OSAllocatedUnfairLock(initialState: UInt64(0))

    /// Allocates a new `SyncEvent` on the given Metal device.
    ///
    /// - Parameter device: The Metal device on which to create the event.
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
    ///
    /// When the I/O transfer completes, the storage controller hardware will automatically
    /// update the shared event to `ticket`.
    public func signal(on ioCommandBuffer: any MTLIOCommandBuffer, ticket: UInt64) {
        ioCommandBuffer.signalEvent(sharedEvent, value: ticket)
    }

    /// Enqueues a signal operation on a GPU compute command buffer.
    public func signal(on computeCommandBuffer: any MTLCommandBuffer, ticket: UInt64) {
        computeCommandBuffer.encodeSignalEvent(sharedEvent, value: ticket)
    }

    /// Enqueues a hardware wait operation on a GPU compute command buffer.
    ///
    /// The GPU command processor will pause execution of the compute command buffer
    /// until the shared event reaches or exceeds `ticket`, with exactly zero CPU intervention.
    public func encodeWait(on computeCommandBuffer: any MTLCommandBuffer, ticket: UInt64) {
        computeCommandBuffer.encodeWaitForEvent(sharedEvent, value: ticket)
    }

    /// Enqueues a hardware wait operation on a Fast I/O command buffer.
    public func encodeWait(on ioCommandBuffer: any MTLIOCommandBuffer, ticket: UInt64) {
        ioCommandBuffer.waitForEvent(sharedEvent, value: ticket)
    }

    /// Synchronously waits on the CPU until the shared event reaches the specified ticket, or times out.
    ///
    /// Note: Intended primarily for unit tests and synchronization barriers, not hot inference loops.
    /// - Parameters:
    ///   - ticket: The ticket value to wait for.
    ///   - timeoutSeconds: Maximum duration in seconds to wait.
    /// - Returns: `true` if the event was signaled at or above `ticket`, `false` if timed out.
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
