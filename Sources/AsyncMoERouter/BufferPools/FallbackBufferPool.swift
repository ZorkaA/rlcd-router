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

// MARK: - Demand Fetch Context & Reasons

/// Specific reason triggering a synchronous demand fetch into the Fallback Pool.
public enum DemandReason: String, Sendable, Equatable, CustomStringConvertible {
    case cacheMiss = "CacheMiss"
    case deadlineMiss = "DeadlineMiss"
    case ringBufferExhaustion = "RingBufferExhaustion"
    case speculativeDemoted = "SpeculativeDemoted"
    case legacyDemandFetch = "LegacyDemandFetch"

    public var description: String { rawValue }
}

/// Metadata uniquely identifying the token, layer, and expert requiring emergency fallback allocation.
public struct DemandFetchContext: Sendable, Equatable, CustomStringConvertible {
    public let expert: ExpertKey
    public let tokenIndex: UInt32
    public let reason: DemandReason

    public init(
        expert: ExpertKey,
        tokenIndex: UInt32 = 0,
        reason: DemandReason = .cacheMiss
    ) {
        self.expert = expert
        self.tokenIndex = tokenIndex
        self.reason = reason
    }

    public var description: String {
        "Demand[\(expert.description), token: \(tokenIndex), reason: \(reason)]"
    }
}

// MARK: - Allocation Intent (Strict Isolation Guard)

/// Explicit allocation intent passed by callers.
/// Speculative prefetching attempts are rejected at both compile-time and runtime.
public enum AllocationIntent: Sendable, Equatable {
    case demand(DemandFetchContext)
    case speculative(expert: ExpertKey)
}

// MARK: - Fallback Pool Errors

/// Errors raised during Fallback Pool allocation, lifecycle transitions, and memory validation.
public enum FallbackPoolError: Error, CustomStringConvertible, Equatable, Sendable {
    case speculativeRequestRejected(expert: ExpertKey)
    case capacityExhausted(requestedBytes: Int, allocatedBytes: Int, maxCapacityBytes: Int)
    case slotAlreadyFree(slotIndex: Int)
    case foreignSlot(slotIndex: Int)
    case metalAllocationFailed(sizeBytes: Int)
    case slotNotInUse(slotIndex: Int)

    public var description: String {
        switch self {
        case .speculativeRequestRejected(let exp):
            return "FallbackPoolError: Speculative prefetch rejected for \(exp.description). Fallback Buffer Pool is strictly isolated for demand fetches."
        case .capacityExhausted(let req, let alloc, let max):
            return "FallbackPoolError: 500MB hard ceiling exhausted. Requested: \(req) bytes, Allocated: \(alloc) bytes, Max: \(max) bytes."
        case .slotAlreadyFree(let idx):
            return "FallbackPoolError: Double-reclaim detected for FallbackSlot[\(idx)]. Slot is already in free list."
        case .foreignSlot(let idx):
            return "FallbackPoolError: Foreign slot rejected with index \(idx). Slot does not belong to this pool."
        case .metalAllocationFailed(let sz):
            return "FallbackPoolError: MTLDevice.makeBuffer failed for length \(sz) bytes (system out of memory)."
        case .slotNotInUse(let idx):
            return "FallbackPoolError: Slot[\(idx)] is not in active use."
        }
    }
}

// MARK: - Fallback Slot State Machine

/// Monotonic lifecycle states for a single slot in the Fallback Pool.
public enum FallbackSlotState: Sendable, Equatable {
    case free
    case reserved(context: DemandFetchContext, ticket: UInt64)
    case inFlightIO(context: DemandFetchContext, ticket: UInt64)
    case inCompute(context: DemandFetchContext, ticket: UInt64)
}

// MARK: - FallbackSlot Implementation

/// A single pre-allocated or dynamically grown unified memory buffer slot inside the Fallback Pool.
public final class FallbackSlot: @unchecked Sendable {
    /// Zero-based slot index in the pool array (0 ..< maxSlots).
    public let index: Int
    /// Dedicated MTLBuffer allocated in .storageModeShared.
    public let buffer: any MTLBuffer
    /// Current lifecycle state.
    public var state: FallbackSlotState
    /// Context of the demand fetch currently occupying this slot.
    public var context: DemandFetchContext?
    /// Hardware synchronization event for zero-CPU GPU-IO coordination.
    public var sharedEvent: (any MTLSharedEvent)?
    /// Ticket value signaled on `sharedEvent` when Fast I/O DMA finishes.
    public var signalValue: UInt64
    /// Timestamp (uptime nanoseconds) when the slot was allocated.
    public var allocationTimestamp: UInt64

    public init(index: Int, buffer: any MTLBuffer) {
        self.index = index
        self.buffer = buffer
        self.state = .free
        self.context = nil
        self.sharedEvent = nil
        self.signalValue = 0
        self.allocationTimestamp = 0
    }

    /// Resets all slot metadata to clean state prior to returning to free list.
    public func reset() {
        self.state = .free
        self.context = nil
        self.sharedEvent = nil
        self.signalValue = 0
        self.allocationTimestamp = 0
    }
}

// MARK: - DemandBufferPoolProtocol

/// Protocol defining the strictly isolated interface for demand-fetch buffer pools.
public protocol DemandBufferPoolProtocol: AnyObject, Sendable {
    var maxCapacityBytes: Int { get }
    var slotSizeBytes: Int { get }
    var maxSlots: Int { get }
    var allocatedBytes: Int { get }
    var inUseSlotCount: Int { get }
    var freeSlotCount: Int { get }

    func allocateForDemand(context: DemandFetchContext, device: any MTLDevice, ticket: UInt64) throws -> FallbackSlot
    func reclaim(_ slot: FallbackSlot) throws
}

// MARK: - FallbackBufferPool Implementation

/// A strictly isolated 500 MB buffer pool reserved exclusively for synchronous cache-miss demand fetches.
///
/// # Strict Memory Ceiling (Invariant 1)
/// The pool enforces an absolute hard capacity ceiling of 524,288,000 bytes (500 × 1024 × 1024).
/// Invariant: `totalAllocatedBytes <= maxCapacityBytes` at all times.
/// When capacity is exhausted, the pool throws `FallbackPoolError.capacityExhausted` rather than OOMing.
///
/// # Strict Isolation from Speculative Prefetching (Invariant 2)
/// This pool is physically and logically segregated from `SpeculativeRingBuffer`.
/// Speculative prefetching calls are rejected with `FallbackPoolError.speculativeRequestRejected`.
///
/// # Zero Fragmentation & Lifecycle Tracking (Invariant 3)
/// All slots have identical uniform byte size (`slotSizeBytes`). External fragmentation is mathematically zero.
/// Double-reclaim protection and foreign slot detection prevent free list corruption.
public final class FallbackBufferPool: DemandBufferPoolProtocol, @unchecked Sendable {
    // MARK: - Public Configuration
    public let maxCapacityBytes: Int
    public let slotSizeBytes: Int
    public let maxSlots: Int

    // MARK: - Internal State
    private let _device: any MTLDevice
    private var _freeSlots: [FallbackSlot] = []
    private var _inUseSlots: [Int: FallbackSlot] = [:]
    private var _totalAllocatedBytes: Int = 0
    private let _lock = OSAllocatedUnfairLock(initialState: ())
    private let _log = Logger(subsystem: "AsyncMoERouter", category: "FallbackBufferPool")

    // MARK: - Diagnostics & Telemetry
    public private(set) var totalFallbackAllocations: Int = 0
    public private(set) var totalReclaims: Int = 0
    public private(set) var totalSpeculativeRejections: Int = 0
    public private(set) var peakBytesInUse: Int = 0

    // MARK: - Initialization

    /// Creates the isolated Fallback Pool with a strict byte ceiling.
    ///
    /// - Parameters:
    ///   - device: Metal device for buffer allocations.
    ///   - slotSizeBytes: Uniform byte size of each expert buffer slot.
    ///   - maxCapacityBytes: Hard capacity ceiling (default: exactly 500 MB = 524,288,000 bytes).
    ///   - prewarmCount: Optional number of warm slots to pre-allocate conservatively (default: min(4, maxSlots)).
    public init(
        device: any MTLDevice,
        slotSizeBytes: Int,
        maxCapacityBytes: Int = 500 * 1024 * 1024,
        prewarmCount: Int? = nil
    ) {
        precondition(slotSizeBytes > 0, "Slot size must be positive")
        precondition(maxCapacityBytes >= slotSizeBytes, "Capacity must fit at least one slot")

        self._device = device
        self.slotSizeBytes = slotSizeBytes
        self.maxCapacityBytes = maxCapacityBytes
        self.maxSlots = maxCapacityBytes / slotSizeBytes

        // Determine prewarm count: default to min(4, maxSlots) to stay conservative on host RAM
        let count = min(prewarmCount ?? min(4, maxSlots), self.maxSlots)
        for i in 0..<count {
            guard let buf = device.makeBuffer(length: slotSizeBytes, options: .storageModeShared) else {
                _log.error("FallbackPool: Failed to pre-allocate warm slot[\(i)]")
                break
            }
            buf.label = "FallbackPool[slot=\(i)]"
            self._freeSlots.append(FallbackSlot(index: i, buffer: buf))
            self._totalAllocatedBytes += slotSizeBytes
        }
        _log.info("FallbackPool initialized: ceiling=\(maxCapacityBytes) bytes, maxSlots=\(self.maxSlots), preAllocated=\(self._freeSlots.count) slots")
    }

    // MARK: - Strict Isolation Allocation API

    /// Allocates a slot for a verified demand fetch, rejecting any speculative request.
    ///
    /// - Parameters:
    ///   - context: Demand context indicating the token, expert, and reason for the cache miss.
    ///   - device: Metal device for buffer allocation if pool growth is needed.
    ///   - ticket: Fast I/O hardware sync ticket to attach to the slot.
    /// - Returns: An allocated `FallbackSlot` in `.reserved` state.
    /// - Throws: `FallbackPoolError.capacityExhausted` if 500MB ceiling is reached,
    ///           `FallbackPoolError.metalAllocationFailed` if OS allocation fails.
    public func allocateForDemand(
        context: DemandFetchContext,
        device: any MTLDevice,
        ticket: UInt64 = 0
    ) throws -> FallbackSlot {
        try _lock.withLock {
            // 1. Check recycled free list (O(1))
            if let recycled = _freeSlots.popLast() {
                recycled.state = .reserved(context: context, ticket: ticket)
                recycled.context = context
                recycled.signalValue = ticket
                recycled.allocationTimestamp = DispatchTime.now().uptimeNanoseconds
                _inUseSlots[recycled.index] = recycled

                totalFallbackAllocations += 1
                let bytesInUse = _inUseSlots.count * slotSizeBytes
                if bytesInUse > peakBytesInUse { peakBytesInUse = bytesInUse }
                _log.debug("FallbackPool: Reused slot[\(recycled.index)] for \(context.description), inUse=\(self._inUseSlots.count)")
                return recycled
            }

            // 2. Enforce Hard 500MB Memory Ceiling Invariant
            guard _totalAllocatedBytes + slotSizeBytes <= maxCapacityBytes else {
                _log.warning("FallbackPool: Hard 500MB ceiling reached (\(self._totalAllocatedBytes)/\(self.maxCapacityBytes) bytes). Rejecting demand allocation for \(context.description).")
                throw FallbackPoolError.capacityExhausted(
                    requestedBytes: slotSizeBytes,
                    allocatedBytes: _totalAllocatedBytes,
                    maxCapacityBytes: maxCapacityBytes
                )
            }

            // 3. Grow pool dynamically within ceiling bounds
            guard let newIndex = (0..<maxSlots).first(where: { _inUseSlots[$0] == nil }) else {
                throw FallbackPoolError.capacityExhausted(
                    requestedBytes: slotSizeBytes,
                    allocatedBytes: _totalAllocatedBytes,
                    maxCapacityBytes: maxCapacityBytes
                )
            }

            guard let buf = device.makeBuffer(length: slotSizeBytes, options: .storageModeShared) else {
                _log.error("FallbackPool: MTLDevice.makeBuffer failed for new slot[\(newIndex)] of size \(self.slotSizeBytes)")
                throw FallbackPoolError.metalAllocationFailed(sizeBytes: slotSizeBytes)
            }
            buf.label = "FallbackPool[slot=\(newIndex)]"

            let newSlot = FallbackSlot(index: newIndex, buffer: buf)
            newSlot.state = .reserved(context: context, ticket: ticket)
            newSlot.context = context
            newSlot.signalValue = ticket
            newSlot.allocationTimestamp = DispatchTime.now().uptimeNanoseconds

            _totalAllocatedBytes += slotSizeBytes
            _inUseSlots[newSlot.index] = newSlot

            totalFallbackAllocations += 1
            let bytesInUse = _inUseSlots.count * slotSizeBytes
            if bytesInUse > peakBytesInUse { peakBytesInUse = bytesInUse }
            _log.debug("FallbackPool: Grew pool with slot[\(newIndex)] for \(context.description), allocated=\(self._totalAllocatedBytes) bytes")
            return newSlot
        }
    }

    /// Allocates with explicit intent validation, actively rejecting speculative prefetch requests.
    public func allocate(
        intent: AllocationIntent,
        device: any MTLDevice,
        ticket: UInt64 = 0
    ) throws -> FallbackSlot {
        switch intent {
        case .speculative(let expert):
            _lock.withLock { totalSpeculativeRejections += 1 }
            _log.error("FallbackPool: Speculative prefetch rejected for \(expert.description). Pool is strictly isolated for demand fetches.")
            throw FallbackPoolError.speculativeRequestRejected(expert: expert)

        case .demand(let context):
            return try allocateForDemand(context: context, device: device, ticket: ticket)
        }
    }

    // MARK: - Backward-Compatible Allocation API

    /// Backward-compatible allocate method for existing callers (`DeadlockResolver`, legacy tests).
    /// Defaults to `.legacyDemandFetch` demand context. Returns `nil` on capacity exhaustion.
    public func allocate(device: any MTLDevice) -> FallbackSlot? {
        let dummyExpert = ExpertKey(layer: 0, expert: 0)
        let context = DemandFetchContext(expert: dummyExpert, reason: .legacyDemandFetch)
        do {
            return try allocateForDemand(context: context, device: device, ticket: 0)
        } catch {
            return nil
        }
    }

    // MARK: - Reclaim & Clean Recycling API

    /// Returns a fallback slot to the free list after compute completes.
    ///
    /// - Parameter slot: The slot to reclaim.
    /// - Throws: `FallbackPoolError.slotAlreadyFree` if double-reclaim is attempted,
    ///           `FallbackPoolError.foreignSlot` if slot does not belong to this pool.
    public func reclaim(_ slot: FallbackSlot) throws {
        try _lock.withLock {
            // Guard against foreign slot
            guard slot.index < maxSlots else {
                _log.error("FallbackPool: Attempted to reclaim foreign slot with index \(slot.index) >= \(self.maxSlots)")
                throw FallbackPoolError.foreignSlot(slotIndex: slot.index)
            }

            // Guard against double reclaim
            guard _inUseSlots.removeValue(forKey: slot.index) != nil else {
                if _freeSlots.contains(where: { $0.index == slot.index }) {
                    _log.error("FallbackPool: Double-reclaim detected on slot[\(slot.index)]")
                    throw FallbackPoolError.slotAlreadyFree(slotIndex: slot.index)
                }
                _log.error("FallbackPool: Foreign slot[\(slot.index)] not found in active tracking")
                throw FallbackPoolError.foreignSlot(slotIndex: slot.index)
            }

            // Clean sanitization
            slot.reset()
            _freeSlots.append(slot)
            totalReclaims += 1
            _log.debug("FallbackPool: Cleanly reclaimed slot[\(slot.index)], freeCount=\(self._freeSlots.count)")
        }
    }

    /// Legacy non-throwing reclaim method for existing callers.
    public func reclaimLegacy(_ slot: FallbackSlot) {
        _ = try? reclaim(slot)
    }

    // MARK: - Maintenance & Teardown

    /// Purges all cached free buffers back to the operating system, reducing host RAM footprint.
    public func purge() {
        _lock.withLock {
            let freedCount = _freeSlots.count
            _freeSlots.removeAll(keepingCapacity: false)
            _totalAllocatedBytes = _inUseSlots.count * slotSizeBytes
            _log.info("FallbackPool: Purged \(freedCount) free buffers. Remaining allocated=\(self._totalAllocatedBytes) bytes")
        }
    }

    // MARK: - Telemetry & Invariant Properties

    /// Total bytes allocated via `MTLDevice.makeBuffer` across both free and in-use buffers.
    public var allocatedBytes: Int {
        _lock.withLock { _totalAllocatedBytes }
    }

    /// Number of free slots available immediately for reuse.
    public var freeSlotCount: Int {
        _lock.withLock { _freeSlots.count }
    }

    /// Number of slots currently in active use for demand fetches or GPU compute.
    public var inUseSlotCount: Int {
        _lock.withLock { _inUseSlots.count }
    }

    /// Current bytes occupied by in-use slots.
    public var currentBytesInUse: Int {
        _lock.withLock { _inUseSlots.count * slotSizeBytes }
    }

    /// Current bytes in idle free list.
    public var currentFreeBytes: Int {
        _lock.withLock { _freeSlots.count * slotSizeBytes }
    }

    /// Invariant verification: asserts accounting consistency and ceiling bounds.
    public func verifyInvariants() -> Bool {
        _lock.withLock {
            let totalSlots = _freeSlots.count + _inUseSlots.count
            let expectedBytes = totalSlots * slotSizeBytes
            let invariant1 = _totalAllocatedBytes <= maxCapacityBytes
            let invariant2 = _totalAllocatedBytes == expectedBytes
            let invariant3 = totalSlots <= maxSlots
            return invariant1 && invariant2 && invariant3
        }
    }

    /// A snapshot of pool health metrics.
    public struct PoolDiagnostics: Sendable {
        public let inUseCount: Int
        public let freeCount: Int
        public let allocatedBytes: Int
        public let maxCapacityBytes: Int
    }

    /// Returns a snapshot of current pool health metrics.
    public func diagnostics() -> PoolDiagnostics {
        _lock.withLock {
            PoolDiagnostics(
                inUseCount: _inUseSlots.count,
                freeCount: _freeSlots.count,
                allocatedBytes: _totalAllocatedBytes,
                maxCapacityBytes: maxCapacityBytes
            )
        }
    }
}
