import Foundation
import Metal
import os.log

/// Thread-safe speculative Ring Buffer Pool managing 16 pre-allocated `MTLBuffer` slots
/// for overlapping expert weight DMA with GPU compute.
///
/// # LRU Policy
/// Slot `lastAccessedTimestamp` values are updated **strictly** by the CPU after draining
/// the GPU Execution Log — never from pre-routing predictions. This maintains a single
/// deterministic source of truth for LRU state.
///
/// # Thread Safety
/// All public methods are protected by `_lock` (an `NSLock`). Callers must NOT hold
/// the lock when entering public API — the lock is not reentrant.
public final class SpeculativeRingBuffer: @unchecked Sendable {
    // MARK: - Private state
    private let _lock = NSLock()
    private var _slots: [RingBufferSlot]
    private let _log = Logger(subsystem: "AsyncMoERouter", category: "SpeculativeRingBuffer")
    private var _allocationEpoch: UInt64 = 0

    // MARK: - Public metrics (read under lock for accuracy)
    public private(set) var totalAllocations: Int = 0
    public private(set) var totalEvictions: Int = 0
    public private(set) var totalAbandonments: Int = 0

    // MARK: - Init

    /// Creates and pre-allocates a pool of `slotCount` Metal buffers.
    ///
    /// - Parameters:
    ///   - device: The `MTLDevice` to allocate buffers on.
    ///   - slotCount: Number of Ring Buffer slots (default: 16).
    ///   - slotSizeBytes: Byte size of each slot, must equal expert weight tensor size.
    ///   - budgetConfig: Memory budget configuration for alignment validation.
    public init(
        device: any MTLDevice,
        slotCount: Int = 16,
        slotSizeBytes: Int,
        budgetConfig: MemoryBudgetConfig = .default
    ) {
        precondition(slotCount > 0, "Ring Buffer must have at least 1 slot")
        precondition(slotSizeBytes > 0, "Slot size must be positive")

        var slots: [RingBufferSlot] = []
        slots.reserveCapacity(slotCount)
        for i in 0..<slotCount {
            guard let buf = device.makeBuffer(length: slotSizeBytes, options: .storageModeShared) else {
                fatalError("SpeculativeRingBuffer: Failed to allocate MTLBuffer for slot \(i) (size: \(slotSizeBytes) bytes).")
            }
            buf.label = "SpeculativeRingBuffer[slot=\(i)]"
            let evt = device.makeSharedEvent()
            evt?.label = "SpeculativeRingBuffer[slot=\(i)].sharedEvent"
            slots.append(RingBufferSlot(index: i, buffer: buf, sharedEvent: evt, signalValue: 0))
        }
        _slots = slots

        _log.info("SpeculativeRingBuffer initialized: \(slotCount) slots × \(slotSizeBytes) bytes = \(slotCount * slotSizeBytes) bytes")
    }

    // MARK: - Slot Lookup

    /// Looks up a ready slot containing the specified expert, if one exists.
    /// - Returns: The matching slot in `.ready` state, or `nil` if not cached.
    public func findReadySlot(for expert: ExpertKey) -> RingBufferSlot? {
        _lock.lock()
        defer { _lock.unlock() }
        return _slots.first { slot in
            if case .ready(_, let key) = slot.state { return key == expert }
            return false
        }
    }

    /// Looks up any slot containing the specified expert (loading OR ready).
    /// - Returns: The matching slot, or `nil`.
    public func findSlot(for expert: ExpertKey) -> RingBufferSlot? {
        _lock.lock()
        defer { _lock.unlock() }
        return _slots.first { slot in
            switch slot.state {
            case .loading(_, let key): return key == expert
            case .ready(_, let key): return key == expert
            case .inUse(_, let key, _): return key == expert
            default: return false
            }
        }
    }

    // MARK: - Slot Allocation

    /// Allocates the best free (or evictable) slot for an upcoming speculative I/O load.
    ///
    /// Allocation priority:
    /// 1. First `.free` slot found (O(n) scan).
    /// 2. Oldest `.ready` slot by `lastAccessedTimestamp` (LRU eviction).
    ///
    /// `.loading`, `.inUse`, and `.abandoned` slots are never evicted.
    ///
    /// - Returns: The allocated `RingBufferSlot` (state set to `.loading`), or `nil` if all
    ///   slots are unavailable (all in-use or loading).
    public func allocateSlot(for expert: ExpertKey, ticket: UInt64) -> RingBufferSlot? {
        _lock.lock()
        defer { _lock.unlock() }

        // 1. Find a free slot
        if let freeSlot = _slots.first(where: { if case .free = $0.state { return true }; return false }) {
            freeSlot.state = .loading(ticket: ticket, expert: expert)
            _allocationEpoch &+= 1
            totalAllocations += 1
            _log.debug("RingBuffer: Allocated free slot[\(freeSlot.index)] for \(expert.description)")
            return freeSlot
        }

        // 2. LRU evict the oldest .ready slot
        let evictable = _slots.filter {
            if case .ready = $0.state { return true }
            return false
        }
        guard let victim = evictable.min(by: { $0.lastAccessedTimestamp < $1.lastAccessedTimestamp }) else {
            let slotCount = _slots.count
            _log.warning("RingBuffer: All \(slotCount) slots occupied (loading/inUse/abandoned) — allocation failed for \(expert.description)")
            return nil
        }

        _log.debug("RingBuffer: LRU-evicting slot[\(victim.index)] (ts=\(victim.lastAccessedTimestamp)) for \(expert.description)")
        victim.state = .loading(ticket: ticket, expert: expert)
        totalEvictions += 1
        totalAllocations += 1
        return victim
    }

    // MARK: - State Transitions

    /// Transitions a slot from `.loading` to `.ready` after DMA completes.
    public func markReady(slotIndex: Int, ticket: UInt64) {
        _lock.lock()
        defer { _lock.unlock() }
        let slot = _slots[slotIndex]
        guard case .loading(let t, let key) = slot.state, t == ticket else {
            _log.warning("RingBuffer: markReady(slot=\(slotIndex)) ticket mismatch or wrong state — ignoring (likely abandoned)")
            return
        }
        slot.state = .ready(ticket: ticket, expert: key)
        _log.debug("RingBuffer: Slot[\(slotIndex)] marked ready for \(key.description)")
    }

    /// Marks a slot as `.abandoned` when a fallback fetch pre-empts the speculative slot.
    /// The slot remains physically intact; its completion handler will drop the signal.
    public func markAbandoned(slotIndex: Int) {
        _lock.lock()
        defer { _lock.unlock() }
        let slot = _slots[slotIndex]
        switch slot.state {
        case .loading(let ticket, let key):
            slot.state = .abandoned(ticket: ticket, expert: key)
            totalAbandonments += 1
            _ = slot.inFlightIOCommand?.tryCancel() // best-effort PCIe bandwidth save
            _log.info("RingBuffer: Slot[\(slotIndex)] abandoned (tryCancel issued) — will be reclaimed on callback fire")
        default:
            _log.warning("RingBuffer: markAbandoned(slot=\(slotIndex)) in unexpected state — ignoring")
        }
    }

    /// Called from the IO completion handler of an abandoned slot to reclaim it.
    public func reclaim(slotIndex: Int) {
        _lock.lock()
        defer { _lock.unlock() }
        let slot = _slots[slotIndex]
        if case .abandoned = slot.state {
            slot.state = .free
            slot.inFlightIOCommand = nil
            slot.signalValue = 0
            _log.debug("RingBuffer: Slot[\(slotIndex)] reclaimed to .free after abandoned callback")
        }
    }

    /// Called by the Execution Log drain to update LRU timestamps after real GPU execution.
    /// This is the **only** permitted path for updating `lastAccessedTimestamp`.
    public func updateLRUTimestamp(slotIndex: Int, timestamp: UInt64) {
        _lock.lock()
        defer { _lock.unlock() }
        _slots[slotIndex].lastAccessedTimestamp = timestamp
    }

    /// Marks a slot as `.inUse` when it is bound to an active GPU encode pass.
    public func markInUse(slotIndex: Int, ticket: UInt64) {
        _lock.lock()
        defer { _lock.unlock() }
        let slot = _slots[slotIndex]
        if case .ready(let t, let key) = slot.state, t == ticket {
            slot.state = .inUse(ticket: ticket, expert: key, retainCount: 1)
        }
    }

    /// Releases a slot from `.inUse` back to `.ready` (or `.free` if no longer needed).
    public func releaseFromUse(slotIndex: Int, ticket: UInt64) {
        _lock.lock()
        defer { _lock.unlock() }
        let slot = _slots[slotIndex]
        if case .inUse(let t, let key, let rc) = slot.state, t == ticket {
            if rc <= 1 {
                slot.state = .ready(ticket: ticket, expert: key)
            } else {
                slot.state = .inUse(ticket: ticket, expert: key, retainCount: rc - 1)
            }
        }
    }

    // MARK: - Diagnostics

    /// Returns a snapshot of all slot states (for testing & logging).
    public func slotStateSnapshot() -> [SlotState] {
        _lock.lock()
        defer { _lock.unlock() }
        return _slots.map(\.state)
    }

    public var slotCount: Int { _slots.count }
}
