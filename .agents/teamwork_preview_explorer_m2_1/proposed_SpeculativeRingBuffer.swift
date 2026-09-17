import Foundation
import Metal
import os
import os.log

/// Thread-safe speculative Ring Buffer Pool managing a fixed array of 16 pre-allocated `MTLBuffer` slots
/// in `.storageModeShared` for overlapping expert weight Fast I/O DMA transfers with GPU execution.
///
/// # Architecture & Design Principles (Phase 2 Requirement R2)
/// 1. **Fixed Array Pre-Allocation**:
///    Pre-allocates exactly 16 `MTLBuffer` slots in `.storageModeShared` at initialization.
///    Zero runtime buffer allocations occur during inference, eliminating memory fragmentation
///    and dynamic kernel allocation latency.
///    - Sizing: $17,301,504$ bytes ($16.50$ MiB, exactly 1056 pages of 16 KB) for FP16 Qwen1.5-MoE-A2.7B.
///    - Total memory: $16 \times 17.3\text{ MB} \approx 276.8\text{ MB}$, well within system limits.
///
/// 2. **Slot Lifecycle State Machine**:
///    Each slot cycles through a strict, deterministic state machine:
///    `.free` -> `.loading`/`.prefetching` -> `.ready` -> `.inUse` -> `.ready`/`.free`,
///    with safe `.abandoned` (dirty slot) preemption and signal dropping.
///
/// 3. **OSAllocatedUnfairLock Thread Safety**:
///    All mutable state is encapsulated inside `_state`, protected by `OSAllocatedUnfairLock`.
///    Replaces `NSLock` to eliminate Objective-C runtime overhead, heap allocation, and POSIX
///    kernel transitions on uncontended access (<15ns lock overhead).
///
/// 4. **Dedicated SyncEvent Per Slot**:
///    Every slot owns an independent `SyncEvent` (wrapping a dedicated `MTLSharedEvent`).
///    This guarantees **zero cross-slot ticket interference**: out-of-order I/O completions
///    on concurrent command buffers can never prematurely satisfy waits on unrelated slots.
///
/// 5. **Dispatch-Time LRU via GPU Execution Log (Requirement R3)**:
///    Slot `lastAccessedTimestamp` values are updated **strictly** by the CPU after draining
///    the GPU Execution Log — never from speculative pre-routing predictions.
///
/// 6. **Dirty Slot Recovery Protocol**:
///    When a cache miss prompts a demand fetch via `FallbackBufferPool`, the speculative slot
///    is marked `.abandoned` and `tryCancel()` is issued. Physical memory is preserved untouched.
///    When the Fast I/O completion callback fires, it drops the signal and resets the slot to `.free`.
public final class SpeculativeRingBuffer: @unchecked Sendable {

    // MARK: - Internal Pool State
    private struct PoolState {
        var slots: [RingBufferSlot]
        var allocationEpoch: UInt64 = 0
        var totalAllocations: Int = 0
        var totalEvictions: Int = 0
        var totalAbandonments: Int = 0
        var totalReclaims: Int = 0
    }

    // MARK: - Private Fields
    private let _state: OSAllocatedUnfairLock<PoolState>
    private let _log = Logger(subsystem: "AsyncMoERouter", category: "SpeculativeRingBuffer")

    // MARK: - Public Configuration
    public let slotCount: Int
    public let slotSizeBytes: Int
    public let device: any MTLDevice

    // MARK: - Public Metrics (Read under lock for atomic accuracy)
    public var totalAllocations: Int { _state.withLock { $0.totalAllocations } }
    public var totalEvictions: Int { _state.withLock { $0.totalEvictions } }
    public var totalAbandonments: Int { _state.withLock { $0.totalAbandonments } }
    public var totalReclaims: Int { _state.withLock { $0.totalReclaims } }
    public var allocationEpoch: UInt64 { _state.withLock { $0.allocationEpoch } }

    // MARK: - Initializer

    /// Creates and pre-allocates a pool of `slotCount` Metal buffers in `.storageModeShared`.
    ///
    /// - Parameters:
    ///   - device: The `MTLDevice` to allocate buffers on.
    ///   - slotCount: Number of Ring Buffer slots (default: 16).
    ///   - slotSizeBytes: Byte size of each slot, matching the expert weight tensor size.
    ///   - budgetConfig: Memory budget configuration for alignment and size validation.
    public init(
        device: any MTLDevice,
        slotCount: Int = 16,
        slotSizeBytes: Int,
        budgetConfig: MemoryBudgetConfig = .default
    ) {
        precondition(slotCount > 0, "Ring Buffer must have at least 1 slot")
        precondition(slotSizeBytes > 0, "Slot size must be positive")

        self.device = device
        self.slotCount = slotCount
        self.slotSizeBytes = slotSizeBytes

        // Defensive alignment verification: Apple Silicon page size is 16 KB (16,384 bytes)
        if slotSizeBytes % MemoryBudgetConfig.appleSiliconPageSizeBytes != 0 {
            let rem = slotSizeBytes % MemoryBudgetConfig.appleSiliconPageSizeBytes
            os_log(.default, "SpeculativeRingBuffer: Notice - slot size %{public}d has page remainder %{public}d", slotSizeBytes, rem)
        }

        var slots: [RingBufferSlot] = []
        slots.reserveCapacity(slotCount)

        for i in 0..<slotCount {
            guard let buf = device.makeBuffer(length: slotSizeBytes, options: .storageModeShared) else {
                fatalError("SpeculativeRingBuffer: Failed to allocate MTLBuffer for slot \(i) (size: \(slotSizeBytes) bytes).")
            }
            buf.label = "SpeculativeRingBuffer[slot=\(i)]"

            // Dedicated SyncEvent per slot to prevent cross-slot ticket interference
            let sync: SyncEvent
            do {
                sync = try SyncEvent(device: device)
                sync.sharedEvent.label = "SpeculativeRingBuffer[slot=\(i)].sharedEvent"
            } catch {
                fatalError("SpeculativeRingBuffer: Failed to allocate SyncEvent for slot \(i): \(error)")
            }

            let slot = RingBufferSlot(
                index: i,
                buffer: buf,
                sharedEvent: sync.sharedEvent,
                signalValue: 0,
                syncEvent: sync
            )
            slots.append(slot)
        }

        self._state = OSAllocatedUnfairLock(initialState: PoolState(slots: slots))

        _log.info("SpeculativeRingBuffer initialized: \(slotCount) slots × \(slotSizeBytes) bytes = \(slotCount * slotSizeBytes) bytes in .storageModeShared")
    }

    // MARK: - Slot Lookup APIs

    /// Looks up an expert slot that has completed DMA and is in `.ready` state.
    ///
    /// - Parameter expert: The expert key to query.
    /// - Returns: The matching `RingBufferSlot` if resident and ready, or `nil` if not cached.
    public func findReadySlot(for expert: ExpertKey) -> RingBufferSlot? {
        _state.withLock { state in
            state.slots.first { slot in
                if case .ready(_, let key) = slot.state { return key == expert }
                return false
            }
        }
    }

    /// Looks up any slot containing the specified expert across active states (`.loading`, `.ready`, `.inUse`).
    ///
    /// - Parameter expert: The expert key to query.
    /// - Returns: The matching `RingBufferSlot`, or `nil` if absent or free/abandoned.
    public func findSlot(for expert: ExpertKey) -> RingBufferSlot? {
        _state.withLock { state in
            state.slots.first { slot in
                switch slot.state {
                case .loading(_, let key): return key == expert
                case .ready(_, let key): return key == expert
                case .inUse(_, let key, _): return key == expert
                default: return false
                }
            }
        }
    }

    // MARK: - Slot Allocation

    /// Allocates the optimal free or evictable slot for an upcoming speculative Fast I/O load.
    ///
    /// # Allocation & Eviction Policy
    /// 0. **Idempotency**: If the requested expert is already loading, ready, or inUse, returns `nil`
    ///    to prevent duplicate in-flight DMA operations for the same weights.
    /// 1. **Free Slot Preference**: Returns the first available `.free` slot (O(n) linear scan, n <= 16).
    /// 2. **Dispatch-Time LRU Eviction**: If no slots are `.free`, selects the `.ready` slot with
    ///    the minimal `lastAccessedTimestamp` (updated strictly via the GPU Execution Log drain per R3).
    /// 3. **Immunity**: `.loading`/`.prefetching`, `.inUse`, and `.abandoned` slots are strictly immune
    ///    to eviction.
    ///
    /// - Parameters:
    ///   - expert: The target expert key to prefetch.
    ///   - ticket: Optional ticket override. If omitted, the slot's dedicated `SyncEvent.nextTicket()` is used.
    /// - Returns: The allocated `RingBufferSlot` (state transitioned to `.loading`), or `nil` if all
    ///   16 slots are saturated.
    public func allocateSlot(for expert: ExpertKey, ticket: UInt64? = nil) -> RingBufferSlot? {
        _state.withLock { state in
            // Step 0: Deduplication check
            for slot in state.slots {
                switch slot.state {
                case .loading(_, let key), .ready(_, let key), .inUse(_, let key, _):
                    if key == expert {
                        _log.debug("SpeculativeRingBuffer: Duplicate prefetch suppressed for \(expert.description)")
                        return nil
                    }
                default: break
                }
            }

            // Step 1: Scan for a .free slot
            if let freeSlot = state.slots.first(where: { if case .free = $0.state { return true }; return false }) {
                let activeTicket = ticket ?? freeSlot.syncEvent?.nextTicket() ?? (freeSlot.signalValue + 1)
                freeSlot.state = .loading(ticket: activeTicket, expert: expert)
                freeSlot.signalValue = activeTicket
                state.allocationEpoch &+= 1
                state.totalAllocations += 1
                _log.debug("SpeculativeRingBuffer: Allocated free slot[\(freeSlot.index)] for \(expert.description), ticket=\(activeTicket)")
                return freeSlot
            }

            // Step 2: LRU eviction among .ready slots
            let evictable = state.slots.filter { if case .ready = $0.state { return true }; return false }
            guard let victim = evictable.min(by: { $0.lastAccessedTimestamp < $1.lastAccessedTimestamp }) else {
                _log.warning("SpeculativeRingBuffer: All \(self.slotCount) slots saturated — allocation failed for \(expert.description)")
                return nil
            }

            let activeTicket = ticket ?? victim.syncEvent?.nextTicket() ?? (victim.signalValue + 1)
            _log.debug("SpeculativeRingBuffer: LRU evicting slot[\(victim.index)] (ts=\(victim.lastAccessedTimestamp)) for \(expert.description), ticket=\(activeTicket)")
            victim.state = .loading(ticket: activeTicket, expert: expert)
            victim.signalValue = activeTicket
            state.totalEvictions += 1
            state.totalAllocations += 1
            return victim
        }
    }

    // MARK: - State Transitions & Hardware Synchronization

    /// Transitions a slot from `.loading` to `.ready` after Fast I/O DMA transfer finishes.
    ///
    /// - Parameters:
    ///   - slotIndex: Slot index in the ring buffer.
    ///   - ticket: Expected completion ticket for verification.
    public func markReady(slotIndex: Int, ticket: UInt64) {
        _state.withLock { state in
            guard slotIndex >= 0 && slotIndex < state.slots.count else { return }
            let slot = state.slots[slotIndex]
            guard case .loading(let t, let key) = slot.state, t == ticket else {
                _log.warning("SpeculativeRingBuffer: markReady(slot=\(slotIndex)) ticket mismatch or state was abandoned — ignoring")
                return
            }
            slot.state = .ready(ticket: ticket, expert: key)
            slot.inFlightIOCommand = nil
            _log.debug("SpeculativeRingBuffer: Slot[\(slotIndex)] marked ready for \(key.description)")
        }
    }

    /// Primary completion callback invoked by the `MTLIOCommandBuffer.addCompletedHandler`.
    ///
    /// Handles both successful arrivals and dirty slot recoveries:
    /// - If the slot is in `.loading`, transitions to `.ready` and returns `true` (signal propagated).
    /// - If the slot was `.abandoned`, drops the signal, clears references, resets to `.free`,
    ///   and returns `false` (signal dropped).
    ///
    /// - Parameters:
    ///   - slotIndex: Index of the slot that completed I/O.
    ///   - ticket: Ticket value associated with the I/O command buffer.
    /// - Returns: `true` if the slot became ready; `false` if the signal was dropped or rejected.
    @discardableResult
    public func completeIO(slotIndex: Int, ticket: UInt64) -> Bool {
        _state.withLock { state in
            guard slotIndex >= 0 && slotIndex < state.slots.count else { return false }
            let slot = state.slots[slotIndex]
            switch slot.state {
            case .loading(let t, let expert):
                if t == ticket {
                    slot.state = .ready(ticket: ticket, expert: expert)
                    slot.inFlightIOCommand = nil
                    _log.debug("SpeculativeRingBuffer: completeIO slot[\(slotIndex)] -> .ready for \(expert.description)")
                    return true
                }
            case .abandoned(let t, let expert):
                if t == ticket {
                    // SIGNAL DROPPING PROTOCOL: Drop signal, reset directly to .free
                    slot.state = .free
                    slot.inFlightIOCommand = nil
                    slot.signalValue = 0
                    state.totalReclaims += 1
                    _log.info("SpeculativeRingBuffer: completeIO slot[\(slotIndex)] abandoned for \(expert.description) -> SIGNAL DROPPED, slot reclaimed to .free")
                    return false
                }
            default:
                break
            }
            return false
        }
    }

    /// Marks a slot as `.abandoned` when a fallback demand fetch preempts this speculative slot.
    ///
    /// The slot's buffer memory is preserved untouched while DMA may still be in-flight.
    /// Initiates a best-effort `tryCancel()` on the in-flight I/O command buffer.
    ///
    /// - Parameter slotIndex: Index of the speculative slot to abandon.
    public func markAbandoned(slotIndex: Int) {
        _state.withLock { state in
            guard slotIndex >= 0 && slotIndex < state.slots.count else { return }
            let slot = state.slots[slotIndex]
            switch slot.state {
            case .loading(let ticket, let expert):
                slot.state = .abandoned(ticket: ticket, expert: expert)
                state.totalAbandonments += 1
                _ = slot.inFlightIOCommand?.tryCancel()
                _log.info("SpeculativeRingBuffer: Slot[\(slotIndex)] abandoned for \(expert.description) (tryCancel issued) — pending completion reclaim")
            default:
                _log.warning("SpeculativeRingBuffer: markAbandoned(slot=\(slotIndex)) called in non-loading state — ignoring")
            }
        }
    }

    /// Marks any loading slot matching the specified expert as `.abandoned`.
    ///
    /// - Parameter expert: The expert key to locate and abandon.
    /// - Returns: The abandoned `RingBufferSlot`, or `nil` if not found in loading state.
    @discardableResult
    public func markAbandoned(for expert: ExpertKey) -> RingBufferSlot? {
        _state.withLock { state in
            for slot in state.slots {
                if case .loading(let ticket, let key) = slot.state, key == expert {
                    slot.state = .abandoned(ticket: ticket, expert: key)
                    state.totalAbandonments += 1
                    _ = slot.inFlightIOCommand?.tryCancel()
                    _log.info("SpeculativeRingBuffer: Slot[\(slot.index)] abandoned for \(expert.description) (tryCancel issued)")
                    return slot
                }
            }
            return nil
        }
    }

    /// Explicitly reclaims an abandoned slot back to `.free`.
    ///
    /// Called when the I/O completion block fires, guaranteeing DMA has ceased.
    ///
    /// - Parameter slotIndex: Index of the slot to reclaim.
    public func reclaim(slotIndex: Int) {
        _state.withLock { state in
            guard slotIndex >= 0 && slotIndex < state.slots.count else { return }
            let slot = state.slots[slotIndex]
            if case .abandoned = slot.state {
                slot.state = .free
                slot.inFlightIOCommand = nil
                slot.signalValue = 0
                state.totalReclaims += 1
                _log.debug("SpeculativeRingBuffer: Slot[\(slotIndex)] reclaimed to .free")
            }
        }
    }

    // MARK: - GPU Execution Binding & LRU

    /// Marks a `.ready` slot as `.inUse` when it is bound to an active GPU compute command encoder.
    /// Supports re-entrant acquisition across multiple token heads via `retainCount`.
    ///
    /// - Parameters:
    ///   - slotIndex: Index of the slot.
    ///   - ticket: Expected ticket matching the slot's current ready ticket.
    public func markInUse(slotIndex: Int, ticket: UInt64) {
        _state.withLock { state in
            guard slotIndex >= 0 && slotIndex < state.slots.count else { return }
            let slot = state.slots[slotIndex]
            if case .ready(let t, let key) = slot.state, t == ticket {
                slot.state = .inUse(ticket: ticket, expert: key, retainCount: 1)
            } else if case .inUse(let t, let key, let rc) = slot.state, t == ticket {
                slot.state = .inUse(ticket: ticket, expert: key, retainCount: rc + 1)
            }
        }
    }

    /// Releases a slot from GPU compute usage.
    ///
    /// Decrements `retainCount`. Once the count reaches 0, transitions the slot back to `.ready`.
    ///
    /// - Parameters:
    ///   - slotIndex: Index of the slot.
    ///   - ticket: Expected ticket matching the active ticket.
    public func releaseFromUse(slotIndex: Int, ticket: UInt64) {
        _state.withLock { state in
            guard slotIndex >= 0 && slotIndex < state.slots.count else { return }
            let slot = state.slots[slotIndex]
            if case .inUse(let t, let key, let rc) = slot.state, t == ticket {
                if rc <= 1 {
                    slot.state = .ready(ticket: ticket, expert: key)
                } else {
                    slot.state = .inUse(ticket: ticket, expert: key, retainCount: rc - 1)
                }
            }
        }
    }

    /// Updates the slot's LRU timestamp after confirmed GPU execution.
    ///
    /// **Authoritative Invariant (Requirement R3)**: This method must ONLY be called
    /// when the CPU drains the GPU Execution Log. It must NEVER be called based on
    /// speculative pre-routing predictions.
    ///
    /// - Parameters:
    ///   - slotIndex: Index of the slot to update.
    ///   - timestamp: Monotonic Mach timestamp from the GPU Execution Log entry.
    public func updateLRUTimestamp(slotIndex: Int, timestamp: UInt64) {
        _state.withLock { state in
            guard slotIndex >= 0 && slotIndex < state.slots.count else { return }
            state.slots[slotIndex].lastAccessedTimestamp = timestamp
        }
    }

    // MARK: - Diagnostics & Inspection

    /// Returns a point-in-time snapshot of all slot states (for telemetry and unit testing).
    public func slotStateSnapshot() -> [SlotState] {
        _state.withLock { $0.slots.map(\.state) }
    }

    /// Returns the number of slots currently in `.free` state.
    public var freeSlotCount: Int {
        _state.withLock { state in
            state.slots.filter { if case .free = $0.state { return true }; return false }.count
        }
    }

    /// Returns the number of slots currently loading or prefetching.
    public var loadingSlotCount: Int {
        _state.withLock { state in
            state.slots.filter { if case .loading = $0.state { return true }; return false }.count
        }
    }

    /// Returns the number of slots currently cached and `.ready`.
    public var readySlotCount: Int {
        _state.withLock { state in
            state.slots.filter { if case .ready = $0.state { return true }; return false }.count
        }
    }

    /// Returns the number of slots currently in active GPU use.
    public var inUseSlotCount: Int {
        _state.withLock { state in
            state.slots.filter { if case .inUse = $0.state { return true }; return false }.count
        }
    }

    /// Returns the number of slots currently abandoned (dirty, pending callback).
    public var abandonedSlotCount: Int {
        _state.withLock { state in
            state.slots.filter { if case .abandoned = $0.state { return true }; return false }.count
        }
    }
}
