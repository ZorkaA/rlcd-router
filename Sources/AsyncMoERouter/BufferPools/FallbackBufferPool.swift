import Foundation
import Metal
import os.log

/// A strictly isolated 500 MB buffer pool reserved exclusively for synchronous
/// cache-miss demand fetches.
///
/// This pool is **completely separate** from the `SpeculativeRingBuffer`.
/// When a cache-miss deadlock occurs, the CPU allocates a slot here, issues a
/// PriorityHigh I/O read via `fallbackQueue`, and never touches the Speculative
/// Ring Buffer until the abandoned slot's completion handler reclaims it.
///
/// # Memory Safety
/// The pool enforces a hard byte ceiling (`maxCapacityBytes`). If capacity is
/// exhausted it returns `nil` rather than OOM-crashing the process.
public final class FallbackBufferPool: @unchecked Sendable {
    // MARK: - Private state
    private let _lock = NSLock()
    private var _freeSlots: [FallbackSlot] = []
    private var _inUseSlots: [FallbackSlot] = []
    private let _log = Logger(subsystem: "AsyncMoERouter", category: "FallbackBufferPool")

    // MARK: - Public configuration
    public let maxCapacityBytes: Int
    public let slotSizeBytes: Int
    public let maxSlots: Int

    // MARK: - Diagnostics
    public private(set) var totalFallbackAllocations: Int = 0
    public private(set) var peakBytesInUse: Int = 0

    // MARK: - Init

    /// Creates the isolated Fallback Pool.
    ///
    /// - Parameters:
    ///   - device: Metal device for buffer allocation.
    ///   - slotSizeBytes: Byte size of each expert buffer slot.
    ///   - maxCapacityBytes: Hard capacity ceiling (default 500 MB).
    public init(
        device: any MTLDevice,
        slotSizeBytes: Int,
        maxCapacityBytes: Int = 500 * 1024 * 1024
    ) {
        precondition(slotSizeBytes > 0, "Slot size must be positive")
        precondition(maxCapacityBytes >= slotSizeBytes, "Capacity must fit at least one slot")

        self.maxCapacityBytes = maxCapacityBytes
        self.slotSizeBytes = slotSizeBytes
        self.maxSlots = maxCapacityBytes / slotSizeBytes

        // Pre-allocate conservatively: up to 4 slots or max budget, whichever is smaller
        let preAllocCount = min(4, maxSlots)
        for i in 0..<preAllocCount {
            guard let buf = device.makeBuffer(length: slotSizeBytes, options: .storageModeShared) else {
                _log.error("FallbackPool: Failed to pre-allocate slot \(i)")
                break
            }
            buf.label = "FallbackPool[slot=\(i)]"
            _freeSlots.append(FallbackSlot(index: i, buffer: buf))
        }
        _log.info("FallbackPool initialized: capacity=\(maxCapacityBytes) bytes, preAllocated=\(preAllocCount) slots")
    }

    // MARK: - Allocation

    /// Allocates a slot from the Fallback Pool for a synchronous cache-miss fetch.
    ///
    /// Returns `nil` if the hard capacity ceiling is reached.
    public func allocate(device: any MTLDevice) -> FallbackSlot? {
        _lock.lock()
        defer { _lock.unlock() }

        if let slot = _freeSlots.popLast() {
            _inUseSlots.append(slot)
            totalFallbackAllocations += 1
            let bytesInUse = _inUseSlots.count * slotSizeBytes
            if bytesInUse > peakBytesInUse { peakBytesInUse = bytesInUse }
            let inUseCount = _inUseSlots.count
            _log.debug("FallbackPool: Allocated existing slot[\(slot.index)], inUse=\(inUseCount)")
            return slot
        }

        // Grow pool if capacity allows
        let currentTotal = (_freeSlots.count + _inUseSlots.count)
        let cap = maxSlots
        let slotSz = slotSizeBytes
        let maxCap = maxCapacityBytes
        guard currentTotal < cap else {
            _log.warning("FallbackPool: Hard capacity ceiling reached (\(cap) slots × \(slotSz) bytes = \(maxCap) bytes). Allocation refused.")
            return nil
        }

        let newIndex = currentTotal
        guard let buf = device.makeBuffer(length: slotSizeBytes, options: .storageModeShared) else {
            _log.error("FallbackPool: MTLDevice.makeBuffer failed for new slot \(newIndex)")
            return nil
        }
        buf.label = "FallbackPool[slot=\(newIndex)]"
        let slot = FallbackSlot(index: newIndex, buffer: buf)
        _inUseSlots.append(slot)
        totalFallbackAllocations += 1
        let bytesInUse = _inUseSlots.count * slotSizeBytes
        if bytesInUse > peakBytesInUse { peakBytesInUse = bytesInUse }
        let inUseCount = _inUseSlots.count
        _log.debug("FallbackPool: Grew pool to slot[\(newIndex)], inUse=\(inUseCount)")
        return slot
    }

    /// Returns a fallback slot to the free list after the GPU has finished using its buffer.
    public func reclaim(_ slot: FallbackSlot) {
        _lock.lock()
        defer { _lock.unlock() }
        _inUseSlots.removeAll { $0.index == slot.index }
        _freeSlots.append(slot)
        let freeCount = _freeSlots.count
        _log.debug("FallbackPool: Reclaimed slot[\(slot.index)], free=\(freeCount)")
    }

    // MARK: - Diagnostics

    public var freeSlotCount: Int {
        _lock.lock(); defer { _lock.unlock() }
        return _freeSlots.count
    }

    public var inUseSlotCount: Int {
        _lock.lock(); defer { _lock.unlock() }
        return _inUseSlots.count
    }

    public var currentBytesInUse: Int {
        _lock.lock(); defer { _lock.unlock() }
        return _inUseSlots.count * slotSizeBytes
    }
}

/// A single buffer slot inside the Fallback Pool.
public final class FallbackSlot: @unchecked Sendable {
    public let index: Int
    public let buffer: any MTLBuffer
    public var sharedEvent: (any MTLSharedEvent)?
    public var signalValue: UInt64 = 0

    public init(index: Int, buffer: any MTLBuffer) {
        self.index = index
        self.buffer = buffer
    }
}
