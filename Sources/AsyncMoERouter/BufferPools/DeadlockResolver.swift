//===----------------------------------------------------------------------===//
//
// This source file is part of the AsyncMoERouter open source project
//
// Copyright (c) 2026 Apple Inc. and the AsyncMoERouter project authors
// Licensed under Apache License v2.0
//
//===----------------------------------------------------------------------===//

import Foundation
import Metal
import os
import os.log

/// Errors specific to cache-miss deadlock resolution and fallback memory operations.
public enum DeadlockResolverError: Error, CustomStringConvertible, Sendable {
    case fallbackPoolExhausted(requestedBytes: Int, maxCapacityBytes: Int)
    case fallbackFetchTimeout(expert: ExpertKey, timeoutSeconds: Double)
    case invalidReadSize(required: Int, available: Int)
    case fileHandleUnavailable
    case ioExecutionFailed(status: Int, description: String)

    public var description: String {
        switch self {
        case .fallbackPoolExhausted(let req, let max):
            return "DeadlockResolverError: 500MB Fallback Pool exhausted (requested \(req) bytes, ceiling \(max) bytes)."
        case .fallbackFetchTimeout(let exp, let sec):
            return "DeadlockResolverError: Fallback fetch for \(exp.description) timed out after \(sec) seconds."
        case .invalidReadSize(let req, let avail):
            return "DeadlockResolverError: Required read size (\(req) bytes) exceeds buffer capacity (\(avail) bytes)."
        case .fileHandleUnavailable:
            return "DeadlockResolverError: MTLIOFileHandle is unavailable or closed."
        case .ioExecutionFailed(let status, let desc):
            return "DeadlockResolverError: Fallback I/O command failed with status \(status): \(desc)"
        }
    }
}

/// Thread-safe Cache-Miss Deadlock Resolution Protocol Coordinator.
/// Implements Requirement R2: Resolves cache-miss deadlocks between the Speculative Ring Buffer
/// and the strictly isolated 500MB Fallback Pool via PriorityHigh Fast I/O dispatch,
/// speculative slot abandonment, tryCancel(), and completedHandler signal dropping.
public final class DeadlockResolver: @unchecked Sendable {
    // MARK: - Core Dependencies
    public let ringBuffer: SpeculativeRingBuffer
    public let fallbackPool: FallbackBufferPool
    public let fastIO: any FastIOEngineProtocol
    public let device: any MTLDevice
    public var fileHandle: (any MTLIOFileHandle)?

    // MARK: - Synchronization & Logging
    private let _lock = OSAllocatedUnfairLock(initialState: ())
    private let _log = Logger(subsystem: "AsyncMoERouter", category: "DeadlockResolver")

    // MARK: - Telemetry & Diagnostic Metrics
    public private(set) var totalDeadlocksResolved: Int = 0
    public private(set) var totalSpeculativeCancellations: Int = 0
    public private(set) var totalSignalsDropped: Int = 0
    public private(set) var totalDirtySlotsReclaimed: Int = 0

    // MARK: - Initializer
    public init(
        ringBuffer: SpeculativeRingBuffer,
        fallbackPool: FallbackBufferPool,
        fastIO: any FastIOEngineProtocol,
        fileHandle: (any MTLIOFileHandle)? = nil,
        device: any MTLDevice = MetalContext.shared.device
    ) {
        self.ringBuffer = ringBuffer
        self.fallbackPool = fallbackPool
        self.fastIO = fastIO
        self.fileHandle = fileHandle
        self.device = device
    }

    // MARK: - Deadlock Resolution (Zero-CPU GPU Synchronization Path)
    /// Resolves a cache miss by immediately allocating from the Fallback Pool, dispatching to
    /// fallbackQueue (PriorityHigh), abandoning any speculative slot, and encoding a zero-CPU wait
    /// on the GPU compute command buffer.
    @discardableResult
    public func resolveCacheMissDeadlock(
        expertID: ExpertKey,
        fileOffset: Int,
        size: Int,
        computeCommandBuffer: any MTLCommandBuffer,
        fileHandle: (any MTLIOFileHandle)? = nil,
        stalledRingSlotIndex: Int? = nil
    ) throws -> FallbackSlot {
        guard let handle = fileHandle ?? self.fileHandle else {
            throw DeadlockResolverError.fileHandleUnavailable
        }
        _log.warning("DeadlockResolver: Cache miss on \(expertID.description) — triggering zero-CPU fallback resolution")

        // 1. Quarantining: If this expert was being loaded speculatively, abandon it and issue tryCancel()
        quarantineSpeculativeSlot(for: expertID, explicitSlotIndex: stalledRingSlotIndex)

        // 2. Fallback Pool Allocation: Allocate isolated buffer from 500MB pool
        let context = DemandFetchContext(expert: expertID, reason: .cacheMiss)
        let fallbackSlot: FallbackSlot
        do {
            fallbackSlot = try fallbackPool.allocateForDemand(context: context, device: device)
        } catch {
            _log.error("DeadlockResolver: Fallback pool 500MB ceiling exhausted for \(expertID.description)")
            throw DeadlockResolverError.fallbackPoolExhausted(
                requestedBytes: size,
                maxCapacityBytes: fallbackPool.maxCapacityBytes
            )
        }

        guard fallbackSlot.buffer.length >= size else {
            try? fallbackPool.reclaim(fallbackSlot)
            throw DeadlockResolverError.invalidReadSize(required: size, available: fallbackSlot.buffer.length)
        }

        // 3. Demand-Fetch Dispatch: Issue PriorityHigh read via FastIO fallbackQueue
        let (sharedEvent, ticket) = fastIO.loadFallback(
            handle: handle,
            offset: fileOffset,
            size: size,
            targetBuffer: fallbackSlot.buffer,
            targetOffset: 0
        )
        fallbackSlot.sharedEvent = sharedEvent
        fallbackSlot.signalValue = ticket

        // 4. Zero-CPU Synchronization: Encode hardware wait on compute command buffer
        computeCommandBuffer.encodeWaitForEvent(sharedEvent, value: ticket)

        _lock.withLock {
            totalDeadlocksResolved += 1
        }

        _log.info("DeadlockResolver: Zero-CPU demand fetch dispatched on PriorityHigh for \(expertID.description), ticket=\(ticket), fallbackSlot[\(fallbackSlot.index)]")
        return fallbackSlot
    }

    // MARK: - Deadlock Resolution (Asynchronous Swift Path)
    /// Convenience deadlock resolution method for tests and mock environments without active file handles.
    @discardableResult
    public func resolveDeadlock(
        for expertID: ExpertKey,
        staleSlot: RingBufferSlot? = nil
    ) async throws -> FallbackSlot? {
        if let stale = staleSlot {
            ringBuffer.markAbandoned(slotIndex: stale.index)
            _lock.withLock {
                totalSpeculativeCancellations += 1
            }
        } else {
            quarantineSpeculativeSlot(for: expertID, explicitSlotIndex: nil)
        }
        let context = DemandFetchContext(expert: expertID, reason: .cacheMiss)
        let slot = try? fallbackPool.allocateForDemand(context: context, device: device)
        if slot != nil {
            _lock.withLock {
                totalDeadlocksResolved += 1
            }
        }
        return slot
    }

    /// Resolves a cache miss asynchronously for CPU-driven execution loops and unit test harnesses.
    @discardableResult
    public func resolveDeadlock(
        expertID: ExpertKey,
        fileHandle: (any MTLIOFileHandle)? = nil,
        fileOffset: Int,
        expertSize: Int,
        device: any MTLDevice? = nil,
        stalledRingSlotIndex: Int? = nil
    ) async throws -> FallbackSlot {
        let activeHandle = fileHandle ?? self.fileHandle
        guard let handle = activeHandle else {
            throw DeadlockResolverError.fileHandleUnavailable
        }
        let targetDevice = device ?? self.device
        return try await resolveDeadlockAsync(
            expertID: expertID,
            fileHandle: handle,
            fileOffset: fileOffset,
            size: expertSize,
            device: targetDevice,
            stalledRingSlotIndex: stalledRingSlotIndex,
            timeoutSeconds: 5.0
        )
    }

    /// Extended asynchronous demand fetch with customizable timeout.
    @discardableResult
    public func resolveDeadlockAsync(
        expertID: ExpertKey,
        fileHandle: (any MTLIOFileHandle)? = nil,
        fileOffset: Int,
        size: Int,
        device: any MTLDevice? = nil,
        stalledRingSlotIndex: Int? = nil,
        timeoutSeconds: Double = 5.0
    ) async throws -> FallbackSlot {
        guard let handle = fileHandle ?? self.fileHandle else {
            throw DeadlockResolverError.fileHandleUnavailable
        }
        let targetDevice = device ?? self.device
        _log.warning("DeadlockResolver: Cache miss on \(expertID.description) — triggering async fallback resolution")

        // 1. Quarantining: Abandon any speculative slot and issue tryCancel()
        quarantineSpeculativeSlot(for: expertID, explicitSlotIndex: stalledRingSlotIndex)

        // 2. Fallback Pool Allocation
        let context = DemandFetchContext(expert: expertID, reason: .cacheMiss)
        let fallbackSlot: FallbackSlot
        do {
            fallbackSlot = try fallbackPool.allocateForDemand(context: context, device: targetDevice)
        } catch {
            _log.error("DeadlockResolver: Fallback pool 500MB ceiling exhausted for \(expertID.description)")
            throw DeadlockResolverError.fallbackPoolExhausted(
                requestedBytes: size,
                maxCapacityBytes: fallbackPool.maxCapacityBytes
            )
        }

        guard fallbackSlot.buffer.length >= size else {
            try? fallbackPool.reclaim(fallbackSlot)
            throw DeadlockResolverError.invalidReadSize(required: size, available: fallbackSlot.buffer.length)
        }

        // 3. Demand-Fetch Dispatch on fallbackQueue (PriorityHigh)
        let (sharedEvent, ticket) = fastIO.loadFallback(
            handle: handle,
            offset: fileOffset,
            size: size,
            targetBuffer: fallbackSlot.buffer,
            targetOffset: 0
        )
        fallbackSlot.sharedEvent = sharedEvent
        fallbackSlot.signalValue = ticket

        // 4. Non-blocking Asynchronous Wait
        let deadline = Date().addingTimeInterval(timeoutSeconds)
        while sharedEvent.signaledValue < ticket {
            if Date() > deadline {
                _log.error("DeadlockResolver: Fallback demand fetch timed out for \(expertID.description)")
                try? fallbackPool.reclaim(fallbackSlot)
                throw DeadlockResolverError.fallbackFetchTimeout(expert: expertID, timeoutSeconds: timeoutSeconds)
            }
            try await Task.sleep(nanoseconds: 100_000) // 100 microseconds poll
        }

        _lock.withLock {
            totalDeadlocksResolved += 1
        }

        _log.info("DeadlockResolver: Async demand fetch completed for \(expertID.description), fallbackSlot[\(fallbackSlot.index)]")
        return fallbackSlot
    }

    // MARK: - Quarantining & Speculative Invalidation
    /// Identifies and quarantines any in-flight speculative slot for `expertID`.
    private func quarantineSpeculativeSlot(for expertID: ExpertKey, explicitSlotIndex: Int?) {
        if let explicitIdx = explicitSlotIndex {
            ringBuffer.markAbandoned(slotIndex: explicitIdx)
            _lock.withLock {
                totalSpeculativeCancellations += 1
            }
            _log.info("DeadlockResolver: Explicit speculative slot[\(explicitIdx)] marked .abandoned with tryCancel()")
            return
        }

        // Look up by expert key in Ring Buffer
        if let slot = ringBuffer.findSlot(for: expertID) {
            if case .loading = slot.state {
                ringBuffer.markAbandoned(slotIndex: slot.index)
                _lock.withLock {
                    totalSpeculativeCancellations += 1
                }
                _log.info("DeadlockResolver: Found matching speculative slot[\(slot.index)] for \(expertID.description) — marked .abandoned with tryCancel()")
            }
        }
    }

    // MARK: - Completion Handler & Signal Dropping Protocol
    /// Invoked when a speculative MTLIOCommandBuffer finishes.
    /// Evaluates the slot state: if `.abandoned`, drops the signal and reclaims the dirty slot to `.free`.
    ///
    /// - Parameters:
    ///   - slotIndex: The slot index in the Speculative Ring Buffer.
    ///   - ticket: The expected ticket value of the speculative transfer.
    ///   - status: The Metal I/O completion status.
    /// - Returns: `true` if signal is propagated (valid prefetch); `false` if signal is dropped (abandoned slot).
    @discardableResult
    public func handleSpeculativeCompletion(
        slotIndex: Int,
        ticket: UInt64,
        status: MTLIOStatus
    ) -> Bool {
        let snapshot = ringBuffer.slotStateSnapshot()
        guard slotIndex >= 0 && slotIndex < snapshot.count else { return false }

        let state = snapshot[slotIndex]
        switch state {
        case .abandoned(let t, let exp):
            if t == ticket {
                _log.warning("DeadlockResolver: Speculative I/O callback fired for ABANDONED slot[\(slotIndex)] (\(exp.description), status=\(status.rawValue)). DROPPING SIGNAL & RECLAIMING.")

                // DROP SIGNAL: Do not update router, do not advance sharedEvent, do not mark ready.
                // Reclaim slot back to free list
                ringBuffer.reclaim(slotIndex: slotIndex)

                _lock.withLock {
                    totalSignalsDropped += 1
                    totalDirtySlotsReclaimed += 1
                }
                return false // Signal dropped
            }
        case .loading(let t, _):
            if t == ticket {
                if status == .complete {
                    ringBuffer.markReady(slotIndex: slotIndex, ticket: ticket)
                    return true // Signal propagated
                } else if status == .cancelled {
                    ringBuffer.reclaim(slotIndex: slotIndex)
                    return false
                }
            }
        default:
            break
        }

        return false
    }

    // MARK: - Fallback Slot Recycling
    /// Releases a fallback slot back to the Fallback Pool after GPU execution completes.
    public func releaseFallbackSlot(_ slot: FallbackSlot) {
        try? fallbackPool.reclaim(slot)
    }

    // MARK: - Diagnostics Reset (Testing)
    public func resetDiagnostics() {
        _lock.withLock {
            totalDeadlocksResolved = 0
            totalSpeculativeCancellations = 0
            totalSignalsDropped = 0
            totalDirtySlotsReclaimed = 0
        }
    }
}
