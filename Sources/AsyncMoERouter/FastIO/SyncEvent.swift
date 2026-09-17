import Foundation
import Metal
import os

/// High-performance zero-CPU synchronization wrapper around Metal's `MTLSharedEvent`.
/// Coordinates hardware-level synchronization between `MTLIOCommandBuffer` DMA transfers
/// and `MTLCommandBuffer` compute/blit command encoders without CPU stalls or thread spinning.
public final class SyncEvent: @unchecked Sendable {
    /// The underlying Metal shared event.
    public let sharedEvent: any MTLSharedEvent

    /// Alias for property convenience across blueprints.
    public var event: any MTLSharedEvent {
        sharedEvent
    }

    /// Monotonic ticket counter protected by an unfair lock.
    private let ticketLock = OSAllocatedUnfairLock(initialState: UInt64(0))

    /// Allocates a new `SyncEvent` on the given Metal device.
    public init(device: any MTLDevice) throws {
        guard let sharedEvent = device.makeSharedEvent() else {
            throw FastIOError.sharedEventCreationFailed
        }
        self.sharedEvent = sharedEvent
    }

    /// Allocates a new `SyncEvent` wrapping an existing `MTLSharedEvent`.
    public init(sharedEvent: any MTLSharedEvent, initialTicket: UInt64 = 0) {
        self.sharedEvent = sharedEvent
        self.ticketLock.withLock { $0 = initialTicket }
    }

    /// Convenience initializer using `event` parameter name.
    public convenience init(event: any MTLSharedEvent, initialTicket: UInt64 = 0) {
        self.init(sharedEvent: event, initialTicket: initialTicket)
    }

    /// Atomically increments and returns the next monotonic ticket value.
    @discardableResult
    public func nextTicket() -> UInt64 {
        ticketLock.withLock { counter in
            counter += 1
            return counter
        }
    }

    /// Returns the current local ticket allocated by this event manager.
    public var currentTicket: UInt64 {
        ticketLock.withLock { $0 }
    }

    /// Returns the highest ticket value signaled so far by GPU or Fast I/O hardware.
    public var signaledValue: UInt64 {
        sharedEvent.signaledValue
    }

    /// Alias for signaledValue.
    public var currentSignaledValue: UInt64 {
        sharedEvent.signaledValue
    }

    /// Non-blocking CPU check whether a specific ticket has been signaled by the GPU or I/O hardware.
    /// Executes an atomic 64-bit load from unified memory in ~150 nanoseconds.
    public func isSignaled(at ticket: UInt64) -> Bool {
        sharedEvent.signaledValue >= ticket
    }

    /// Enqueues a signal operation on a Fast I/O command buffer upon completion of DMA transfer.
    public func signal(on ioCommandBuffer: any MTLIOCommandBuffer, ticket: UInt64) {
        ioCommandBuffer.signalEvent(sharedEvent, value: ticket)
    }

    /// Enqueues a signal operation on a GPU compute/blit command buffer upon completion of passes.
    public func signal(on computeCommandBuffer: any MTLCommandBuffer, ticket: UInt64) {
        computeCommandBuffer.encodeSignalEvent(sharedEvent, value: ticket)
    }

    /// Alias matching encodeSignal naming.
    public func encodeSignal(on ioCommandBuffer: any MTLIOCommandBuffer, ticket: UInt64) {
        signal(on: ioCommandBuffer, ticket: ticket)
    }

    /// Alias matching encodeSignal naming.
    public func encodeSignal(on computeCommandBuffer: any MTLCommandBuffer, ticket: UInt64) {
        signal(on: computeCommandBuffer, ticket: ticket)
    }

    /// Enqueues a hardware wait operation on a GPU compute command buffer.
    /// Halts the GPU Command Processor (CP) at hardware level until `sharedEvent.signaledValue >= ticket`.
    /// Zero CPU cycles consumed.
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
        let sleepStepNanoseconds: UInt32 = 100_000 // 100 microseconds

        while Date() < deadline {
            if sharedEvent.signaledValue >= ticket {
                return true
            }
            usleep(sleepStepNanoseconds)
        }

        return sharedEvent.signaledValue >= ticket
    }

    /// Registers an asynchronous CPU callback via `MTLSharedEventListener` when the ticket is reached.
    public func notify(
        at ticket: UInt64,
        listener: MTLSharedEventListener,
        handler: @escaping (any MTLSharedEvent, UInt64) -> Void
    ) {
        sharedEvent.notify(listener, atValue: ticket, block: handler)
    }
}
