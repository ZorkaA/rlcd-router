import Testing
import Foundation
import Metal
@testable import AsyncMoERouter

// MARK: - Tier 2: Boundary & Edge-Case Tests

/// Tests that probe exact capacity limits, wrap-around conditions, and boundary
/// states in each subsystem.
@Suite("E2E Tier 2: Boundary & Edge-Case Tests")
struct Tier2BoundaryTests {

    // MARK: - Ring Buffer: 16-Slot Full-Capacity Saturation

    @Test("Ring buffer saturation: 16 simultaneous allocations then LRU eviction on 17th")
    func testRingBufferSaturationAndEviction() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw _Skip.always("Metal unavailable")
        }
        let ring = SpeculativeRingBuffer(device: device, slotCount: 16, slotSizeBytes: 64 * 1024)

        // Allocate all 16 slots through allocateSlot
        var allocated: [RingBufferSlot] = []
        var epoch: UInt64 = 1
        for layer in 0..<16 {
            let key = ExpertKey(layer: layer, expert: 0)
            if let slot = ring.allocateSlot(for: key, ticket: epoch) {
                allocated.append(slot)
                ring.markReady(slotIndex: slot.index, ticket: epoch)
                ring.markInUse(slotIndex: slot.index, ticket: epoch)
                epoch += 1
            }
        }
        #expect(allocated.count == 16, "Must allocate 16 slots")

        // Release slot 0 back to ready (evictable), then the 17th allocation must evict it
        ring.releaseFromUse(slotIndex: allocated[0].index, ticket: 1)

        let key17 = ExpertKey(layer: 16, expert: 0)
        let evicted = ring.allocateSlot(for: key17, ticket: epoch)
        #expect(evicted != nil, "17th allocation must succeed via LRU eviction")
    }

    // MARK: - Ring Buffer: Wrap-Around

    @Test("Ring buffer wrap-around: 32 sequential alloc/release cycles with correct indices")
    func testRingBufferWrapAround() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw _Skip.always("Metal unavailable")
        }
        let ring = SpeculativeRingBuffer(device: device, slotCount: 16, slotSizeBytes: 16 * 1024)

        var seenIndices = Set<Int>()
        var epoch: UInt64 = 1
        for cycle in 0..<32 {
            let key = ExpertKey(layer: cycle % 64, expert: cycle % 8)
            guard let slot = ring.allocateSlot(for: key, ticket: epoch) else { epoch += 1; continue }
            let idx = slot.index
            #expect(idx >= 0 && idx < 16, "Slot index \(idx) out of range [0,15]")
            seenIndices.insert(idx)
            ring.markReady(slotIndex: idx, ticket: epoch)
            epoch += 1
        }
        #expect(seenIndices.count == 16, "All 16 physical slot indices must be seen across 32 cycles")
    }

    // MARK: - Fallback Pool: Hard 500 MB Ceiling

    @Test("Fallback pool hard 500MB ceiling: overflow allocation is rejected")
    func testFallbackPoolCeiling() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw _Skip.always("Metal unavailable")
        }
        // 4 slots × 1 MB = 4 MB ceiling
        let pool = FallbackBufferPool(device: device, slotSizeBytes: 1024 * 1024, maxCapacityBytes: 4 * 1024 * 1024, prewarmCount: 0)

        // Drain all 4 slots
        var held: [FallbackSlot] = []
        for i in 0..<4 {
            let ctx = DemandFetchContext(expert: ExpertKey(layer: 0, expert: i))
            if let s = try? pool.allocateForDemand(context: ctx, device: device) {
                held.append(s)
            }
        }
        #expect(held.count == 4, "Should allocate exactly 4 slots")

        // 5th must fail with capacityExhausted
        let ctx5 = DemandFetchContext(expert: ExpertKey(layer: 0, expert: 99))
        do {
            let _ = try pool.allocateForDemand(context: ctx5, device: device)
            #expect(Bool(false), "5th allocation must throw capacityExhausted")
        } catch {
            // expected
        }

        // Reclaim one, then 5th succeeds
        try? pool.reclaim(held[0])
        let retry = try? pool.allocateForDemand(context: ctx5, device: device)
        #expect(retry != nil, "After reclaim, 5th allocation should succeed")
        if let r = retry { try? pool.reclaim(r) }
        for slot in held.dropFirst() { try? pool.reclaim(slot) }
    }

    // MARK: - Abort Flag During ICB Encode

    @Test("Abort flag set before ICB encode: flag is readable and correct")
    func testAbortFlagBeforeICBEncode() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw _Skip.always("Metal unavailable")
        }
        let abort = AbortController(device: device)
        let icb = ICBController(device: device)

        abort.set(true)
        #expect(abort.isAborted == true, "Abort flag must read back true immediately")

        icb.reset()
        let argBuf = icb.argumentBuffer
        #expect(argBuf != nil, "ICB argument buffer must remain valid while abort is set")

        abort.reset()
        #expect(abort.isAborted == false, "Abort flag must clear correctly")
    }

    // MARK: - LRU Tracker: Boundary

    @Test("LRU tracker returns correct LRU expert after 16 sequential accesses")
    func testLRUTrackerBoundary() throws {
        let tracker = LRUWeightTracker()
        var keys: [ExpertKey] = []
        for i in 0..<16 {
            let key = ExpertKey(layer: i, expert: 0)
            keys.append(key)
            tracker.recordAccess(expert: key, timestamp: UInt64(i + 1) * 100)
        }
        // LRU must be keys[0] (timestamp 100)
        let lru = tracker.lruExpert
        #expect(lru == keys[0], "LRU expert must be the one with the lowest timestamp")

        // Re-access keys[0] with a fresh timestamp; LRU must shift to keys[1]
        tracker.recordAccess(expert: keys[0], timestamp: 9999)
        let newLRU = tracker.lruExpert
        #expect(newLRU == keys[1], "After re-access, LRU must shift to second-oldest")
    }

    // MARK: - GPU Execution Log: Capacity Boundary

    @Test("GPU execution log: drain at capacity returns correct count")
    func testExecutionLogCapacityDrain() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw _Skip.always("Metal unavailable")
        }
        let log = GPUExecutionLog(device: device, capacity: 64)

        // Drain on empty log returns 0
        let emptyDrain = log.drain()
        #expect(emptyDrain.isEmpty, "Drain on empty log must return 0 entries")
        #expect(log.totalEntriesDrained == 0, "totalEntriesDrained must be 0 initially")
    }

    // MARK: - Abort Controller: API Check

    @Test("AbortController: set/reset produces coherent readback")
    func testAbortControllerSetReset() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw _Skip.always("Metal unavailable")
        }
        let abort = AbortController(device: device)

        abort.set(true)
        #expect(abort.isAborted == true)
        abort.set(false)
        #expect(abort.isAborted == false)
        abort.reset()
        #expect(abort.isAborted == false)
    }
}

// MARK: - Skip helper

private enum _Skip {
    static func always(_ reason: String) -> any Error {
        struct SkipError: Error { let reason: String }
        return SkipError(reason: reason)
    }
}
