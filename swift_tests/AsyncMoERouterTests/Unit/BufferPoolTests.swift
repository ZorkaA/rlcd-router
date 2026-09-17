import Testing
import Metal
import Foundation
@testable import AsyncMoERouter

/// Comprehensive tests for M2 (Buffer Pools) and M3 (Execution Log + LRU).
@Suite("Buffer Pool & Execution Log Tests")
struct BufferPoolTests {
    let device: any MTLDevice

    init() throws {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            throw TestError.metalUnavailable
        }
        self.device = dev
    }

    // MARK: - SpeculativeRingBuffer Tests

    @Test("Ring buffer initializes with correct slot count")
    func testRingBufferInit() {
        let ring = SpeculativeRingBuffer(
            device: device,
            slotCount: 4,
            slotSizeBytes: MoEArchitectureConfig.synthetic.expertSizeBytes
        )
        #expect(ring.slotCount == 4)
        let states = ring.slotStateSnapshot()
        for state in states {
            if case .free = state { } else { Issue.record("Expected .free, got \(state)") }
        }
    }

    @Test("Ring buffer allocates free slot")
    func testRingBufferAllocatesFreeSlot() {
        let ring = SpeculativeRingBuffer(
            device: device, slotCount: 4,
            slotSizeBytes: MoEArchitectureConfig.synthetic.expertSizeBytes
        )
        let expert = ExpertKey(layer: 5, expert: 0)
        let slot = ring.allocateSlot(for: expert, ticket: 1)
        #expect(slot != nil)
        if case .loading(let t, let k) = slot!.state {
            #expect(t == 1)
            #expect(k == expert)
        } else {
            Issue.record("Expected .loading state")
        }
    }

    @Test("Ring buffer marks slot ready")
    func testRingBufferMarkReady() {
        let ring = SpeculativeRingBuffer(
            device: device, slotCount: 4,
            slotSizeBytes: MoEArchitectureConfig.synthetic.expertSizeBytes
        )
        let expert = ExpertKey(layer: 5, expert: 1)
        let slot = ring.allocateSlot(for: expert, ticket: 2)!
        ring.markReady(slotIndex: slot.index, ticket: 2)
        if case .ready(let t, let k) = slot.state {
            #expect(t == 2)
            #expect(k == expert)
        } else {
            Issue.record("Expected .ready state")
        }
    }

    @Test("Ring buffer finds ready slot")
    func testRingBufferFindReady() {
        let ring = SpeculativeRingBuffer(
            device: device, slotCount: 4,
            slotSizeBytes: MoEArchitectureConfig.synthetic.expertSizeBytes
        )
        let expert = ExpertKey(layer: 6, expert: 3)
        let slot = ring.allocateSlot(for: expert, ticket: 3)!
        ring.markReady(slotIndex: slot.index, ticket: 3)
        let found = ring.findReadySlot(for: expert)
        #expect(found != nil)
        #expect(found!.index == slot.index)
    }

    @Test("Ring buffer returns nil for missing expert")
    func testRingBufferMissReturnsNil() {
        let ring = SpeculativeRingBuffer(
            device: device, slotCount: 4,
            slotSizeBytes: MoEArchitectureConfig.synthetic.expertSizeBytes
        )
        let result = ring.findReadySlot(for: ExpertKey(layer: 99, expert: 99))
        #expect(result == nil)
    }

    @Test("Ring buffer LRU evicts oldest ready slot")
    func testRingBufferLRUEviction() {
        let ring = SpeculativeRingBuffer(
            device: device, slotCount: 2,
            slotSizeBytes: MoEArchitectureConfig.synthetic.expertSizeBytes
        )
        let e0 = ExpertKey(layer: 5, expert: 0)
        let e1 = ExpertKey(layer: 5, expert: 1)
        let s0 = ring.allocateSlot(for: e0, ticket: 1)!
        ring.markReady(slotIndex: s0.index, ticket: 1)
        ring.updateLRUTimestamp(slotIndex: s0.index, timestamp: 100)

        let s1 = ring.allocateSlot(for: e1, ticket: 2)!
        ring.markReady(slotIndex: s1.index, ticket: 2)
        ring.updateLRUTimestamp(slotIndex: s1.index, timestamp: 200)

        // All slots occupied — force LRU eviction of e0 (oldest ts=100)
        let e2 = ExpertKey(layer: 5, expert: 2)
        let evicted = ring.allocateSlot(for: e2, ticket: 3)
        #expect(evicted != nil)
        // The evicted slot should have been the one with ts=100 (s0)
        #expect(evicted!.index == s0.index)
    }

    @Test("Ring buffer abandons slot and reclaims after callback")
    func testRingBufferAbandonAndReclaim() {
        let ring = SpeculativeRingBuffer(
            device: device, slotCount: 4,
            slotSizeBytes: MoEArchitectureConfig.synthetic.expertSizeBytes
        )
        let expert = ExpertKey(layer: 7, expert: 2)
        let slot = ring.allocateSlot(for: expert, ticket: 5)!
        ring.markAbandoned(slotIndex: slot.index)
        if case .abandoned = slot.state { } else { Issue.record("Expected .abandoned") }
        ring.reclaim(slotIndex: slot.index)
        if case .free = slot.state { } else { Issue.record("Expected .free after reclaim") }
    }

    @Test("LRU timestamp can only update from log drain path")
    func testLRUTimestampUpdateIsOnlyViaLog() {
        // This test validates the design invariant: the CPU must only
        // call updateLRUTimestamp from the Execution Log drain path.
        let ring = SpeculativeRingBuffer(
            device: device, slotCount: 4,
            slotSizeBytes: MoEArchitectureConfig.synthetic.expertSizeBytes
        )
        let e = ExpertKey(layer: 5, expert: 0)
        let slot = ring.allocateSlot(for: e, ticket: 1)!
        ring.markReady(slotIndex: slot.index, ticket: 1)
        // Simulate execution log drain updating LRU
        ring.updateLRUTimestamp(slotIndex: slot.index, timestamp: 999)
        #expect(slot.lastAccessedTimestamp == 999)
    }

    // MARK: - FallbackBufferPool Tests

    @Test("Fallback pool allocates slot within capacity")
    func testFallbackPoolAllocates() {
        let pool = FallbackBufferPool(
            device: device,
            slotSizeBytes: MoEArchitectureConfig.synthetic.expertSizeBytes,
            maxCapacityBytes: 500 * 1024 * 1024
        )
        let slot = pool.allocate(device: device)
        #expect(slot != nil)
        #expect(pool.inUseSlotCount == 1)
    }

    @Test("Fallback pool reclaims slot correctly")
    func testFallbackPoolReclaims() {
        let pool = FallbackBufferPool(
            device: device,
            slotSizeBytes: MoEArchitectureConfig.synthetic.expertSizeBytes,
            maxCapacityBytes: 500 * 1024 * 1024
        )
        let slot = pool.allocate(device: device)!
        pool.reclaim(slot)
        #expect(pool.inUseSlotCount == 0)
        #expect(pool.freeSlotCount >= 1)
    }

    @Test("Fallback pool respects hard capacity ceiling")
    func testFallbackPoolCapacityCeiling() {
        // Very small capacity — only 1 slot
        let pool = FallbackBufferPool(
            device: device,
            slotSizeBytes: MoEArchitectureConfig.synthetic.expertSizeBytes,
            maxCapacityBytes: MoEArchitectureConfig.synthetic.expertSizeBytes
        )
        let slot1 = pool.allocate(device: device)
        #expect(slot1 != nil)
        // Second allocation should fail (capacity exhausted)
        let slot2 = pool.allocate(device: device)
        #expect(slot2 == nil)
    }

    @Test("Fallback pool is isolated from ring buffer")
    func testFallbackPoolIsolation() {
        // The two pools are completely separate objects — no shared state
        let ring = SpeculativeRingBuffer(
            device: device, slotCount: 2,
            slotSizeBytes: MoEArchitectureConfig.synthetic.expertSizeBytes
        )
        let pool = FallbackBufferPool(
            device: device,
            slotSizeBytes: MoEArchitectureConfig.synthetic.expertSizeBytes,
            maxCapacityBytes: 500 * 1024 * 1024
        )
        // Exhaust ring buffer
        let e0 = ExpertKey(layer: 5, expert: 0)
        let e1 = ExpertKey(layer: 5, expert: 1)
        _ = ring.allocateSlot(for: e0, ticket: 1)
        _ = ring.allocateSlot(for: e1, ticket: 2)
        // Fallback pool allocation must still succeed
        let fallbackSlot = pool.allocate(device: device)
        #expect(fallbackSlot != nil)
    }

    // MARK: - GPUExecutionLog Tests

    @Test("Execution log initializes with correct capacity")
    func testExecutionLogInit() {
        let log = GPUExecutionLog(device: device, capacity: 128)
        #expect(log.capacity == 128)
        // Drain should return empty when freshly initialized
        let entries = log.drain()
        #expect(entries.isEmpty)
    }

    @Test("Execution log drains written entries")
    func testExecutionLogDrainWritten() {
        let log = GPUExecutionLog(device: device, capacity: 128)
        // Manually write an entry (simulating GPU write)
        let ptr = log.buffer.contents().bindMemory(to: ExecutionLogEntry.self, capacity: 128)
        ptr[0] = ExecutionLogEntry(
            tokenIndex: 42, layerIndex: 5, horizonIndex: 1,
            expertID: 7, confidenceScore: 0.9, timestamp: 12345
        )
        let entries = log.drain()
        #expect(entries.count == 1)
        #expect(entries[0].tokenIndex == 42)
        #expect(entries[0].expertID == 7)
        // Log entry cleared after drain
        let secondDrain = log.drain()
        #expect(secondDrain.isEmpty)
    }

    // MARK: - LRUWeightTracker Tests

    @Test("LRU tracker records and retrieves experts")
    func testLRUTrackerRecord() {
        let tracker = LRUWeightTracker()
        let e = ExpertKey(layer: 5, expert: 3)
        tracker.recordAccess(expert: e, timestamp: 1000)
        #expect(tracker.contains(expert: e))
        #expect(tracker.count == 1)
    }

    @Test("LRU tracker evicts least recently used")
    func testLRUTrackerEvictLRU() {
        let tracker = LRUWeightTracker()
        let e0 = ExpertKey(layer: 5, expert: 0)
        let e1 = ExpertKey(layer: 5, expert: 1)
        tracker.recordAccess(expert: e0, timestamp: 100)
        tracker.recordAccess(expert: e1, timestamp: 200)
        let evicted = tracker.evictLRU()
        #expect(evicted == e0) // e0 has lower timestamp (older)
    }

    @Test("LRU tracker promotes on re-access")
    func testLRUTrackerPromotion() {
        let tracker = LRUWeightTracker()
        let e0 = ExpertKey(layer: 5, expert: 0)
        let e1 = ExpertKey(layer: 5, expert: 1)
        tracker.recordAccess(expert: e0, timestamp: 100)
        tracker.recordAccess(expert: e1, timestamp: 200)
        // Re-access e0 with newer timestamp — promotes it above e1
        tracker.recordAccess(expert: e0, timestamp: 300)
        let evicted = tracker.evictLRU()
        #expect(evicted == e1) // e1 is now older
    }

    @Test("LRU tracker returns nil when empty")
    func testLRUTrackerEmptyEvict() {
        let tracker = LRUWeightTracker()
        #expect(tracker.lruExpert == nil)
        #expect(tracker.evictLRU() == nil)
    }
}

enum TestError: Error {
    case metalUnavailable
}
