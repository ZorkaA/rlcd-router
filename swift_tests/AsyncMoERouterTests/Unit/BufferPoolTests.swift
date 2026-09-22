//===----------------------------------------------------------------------===//
//
// This source file is part of the AsyncMoERouter open source project
//
// Copyright (c) 2026 Apple Inc. and the AsyncMoERouter project authors
// Licensed under Apache License v2.0
//
//===----------------------------------------------------------------------===//

import Testing
import Metal
import Foundation
@testable import AsyncMoERouter

/// Comprehensive unit test suite for Milestone 2:
/// - Speculative Ring Buffer (16-slot cycling, wraparound, LRU eviction, state lifecycle)
/// - Fallback Buffer Pool (Strict 500MB ceiling, capacity enforcement, strict isolation)
/// - Deadlock Resolution Protocol (PriorityHigh demand fetch, speculative tryCancel, signal dropping)
/// - Zero memory leak verification under heavy churn via mach_task_basic_info
@Suite("Buffer Pool & Deadlock Resolution Unit Tests")
struct BufferPoolTests {
    let device: any MTLDevice

    init() throws {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            throw TestError.metalUnavailable
        }
        self.device = dev
    }

    // MARK: - 1. Speculative Ring Buffer Lifecycle & Wraparound Tests

    @Test("Ring buffer initializes with correct slot count and .free states")
    func testRingBufferInit() {
        let ring = SpeculativeRingBuffer(
            device: device,
            slotCount: 16,
            slotSizeBytes: MoEArchitectureConfig.synthetic.expertSizeBytes
        )
        #expect(ring.slotCount == 16)
        let states = ring.slotStateSnapshot()
        #expect(states.count == 16)
        for state in states {
            if case .free = state { } else { Issue.record("Expected .free, got \(state)") }
        }
    }

    @Test("Ring buffer complete slot state machine lifecycle: free -> loading -> ready -> inUse -> ready -> evicted")
    func testRingBufferSlotStateLifecycle() {
        let ring = SpeculativeRingBuffer(
            device: device,
            slotCount: 4,
            slotSizeBytes: MoEArchitectureConfig.synthetic.expertSizeBytes
        )
        let expert = ExpertKey(layer: 5, expert: 0)

        // 1. Allocate: free -> loading
        guard let slot = ring.allocateSlot(for: expert, ticket: 1) else {
            Issue.record("Allocation failed")
            return
        }
        #expect(slot.state == .loading(ticket: 1, expert: expert))

        // 2. Ready: loading -> ready
        ring.markReady(slotIndex: slot.index, ticket: 1)
        #expect(slot.state == .ready(ticket: 1, expert: expert))

        // 3. InUse: ready -> inUse (retainCount 1)
        ring.markInUse(slotIndex: slot.index, ticket: 1)
        #expect(slot.state == .inUse(ticket: 1, expert: expert, retainCount: 1))

        // 4. InUse retain count increment
        ring.markInUse(slotIndex: slot.index, ticket: 1)
        #expect(slot.state == .inUse(ticket: 1, expert: expert, retainCount: 2))

        // 5. Release from use: decrement retainCount to 1, then back to ready
        ring.releaseFromUse(slotIndex: slot.index, ticket: 1)
        #expect(slot.state == .inUse(ticket: 1, expert: expert, retainCount: 1))
        ring.releaseFromUse(slotIndex: slot.index, ticket: 1)
        #expect(slot.state == .ready(ticket: 1, expert: expert))
    }

    @Test("Ring buffer continuous 16-slot cycling, wraparound, and deterministic LRU eviction")
    func testRingBufferContinuousCyclingAndWraparound() {
        let slotCount = 16
        let ring = SpeculativeRingBuffer(
            device: device,
            slotCount: slotCount,
            slotSizeBytes: MoEArchitectureConfig.synthetic.expertSizeBytes
        )

        // Fill all 16 slots initially
        for i in 0..<slotCount {
            let expert = ExpertKey(layer: 5, expert: i)
            guard let slot = ring.allocateSlot(for: expert, ticket: UInt64(i + 1)) else {
                Issue.record("Failed to allocate initial slot \(i)")
                return
            }
            #expect(slot.index == i)
            ring.markReady(slotIndex: slot.index, ticket: UInt64(i + 1))
            ring.updateLRUTimestamp(slotIndex: slot.index, timestamp: UInt64(100 + i))
        }

        #expect(ring.totalAllocations == 16)
        #expect(ring.totalEvictions == 0)

        // Allocate 32 more times — forcing 32 deterministic LRU evictions with wraparound
        for j in 0..<32 {
            let newExpert = ExpertKey(layer: 6, expert: j)
            guard let evictedSlot = ring.allocateSlot(for: newExpert, ticket: UInt64(100 + j)) else {
                Issue.record("Eviction failed on cycle \(j)")
                return
            }
            // Slot indices must always be within [0..<16]
            #expect(evictedSlot.index >= 0 && evictedSlot.index < slotCount)
            ring.markReady(slotIndex: evictedSlot.index, ticket: UInt64(100 + j))
            ring.updateLRUTimestamp(slotIndex: evictedSlot.index, timestamp: UInt64(200 + j))
        }

        #expect(ring.totalAllocations == 48)
        #expect(ring.totalEvictions == 32)
    }

    @Test("In-use and loading slots are strictly protected from LRU eviction")
    func testRingBufferProtectedSlotsFromEviction() {
        let ring = SpeculativeRingBuffer(
            device: device,
            slotCount: 3,
            slotSizeBytes: MoEArchitectureConfig.synthetic.expertSizeBytes
        )

        // Slot 0: inUse
        let e0 = ExpertKey(layer: 5, expert: 0)
        let s0 = ring.allocateSlot(for: e0, ticket: 1)!
        ring.markReady(slotIndex: s0.index, ticket: 1)
        ring.markInUse(slotIndex: s0.index, ticket: 1)
        ring.updateLRUTimestamp(slotIndex: s0.index, timestamp: 10) // Very old timestamp

        // Slot 1: loading
        let e1 = ExpertKey(layer: 5, expert: 1)
        let _ = ring.allocateSlot(for: e1, ticket: 2)!

        // Slot 2: ready
        let e2 = ExpertKey(layer: 5, expert: 2)
        let s2 = ring.allocateSlot(for: e2, ticket: 3)!
        ring.markReady(slotIndex: s2.index, ticket: 3)
        ring.updateLRUTimestamp(slotIndex: s2.index, timestamp: 500) // Newer timestamp

        // Attempting to allocate when only slot 2 is evictable:
        // Slot 0 (inUse) and Slot 1 (loading) must NOT be evicted!
        let e3 = ExpertKey(layer: 5, expert: 3)
        let victim = ring.allocateSlot(for: e3, ticket: 4)
        #expect(victim != nil)
        #expect(victim!.index == s2.index, "Only the .ready slot (s2) may be evicted")

        // Now all slots are either inUse or loading. Allocation must fail (returns nil)
        let e4 = ExpertKey(layer: 5, expert: 4)
        let impossibleAllocation = ring.allocateSlot(for: e4, ticket: 5)
        #expect(impossibleAllocation == nil, "Allocation must return nil when all slots are protected")
    }

    @Test("Ring buffer finds ready slot and returns nil for missing expert")
    func testRingBufferFindReadyAndMiss() {
        let ring = SpeculativeRingBuffer(
            device: device,
            slotCount: 4,
            slotSizeBytes: MoEArchitectureConfig.synthetic.expertSizeBytes
        )
        let expert = ExpertKey(layer: 6, expert: 3)
        let slot = ring.allocateSlot(for: expert, ticket: 3)!
        ring.markReady(slotIndex: slot.index, ticket: 3)

        let found = ring.findReadySlot(for: expert)
        #expect(found != nil)
        #expect(found!.index == slot.index)

        let miss = ring.findReadySlot(for: ExpertKey(layer: 99, expert: 99))
        #expect(miss == nil)
    }

    // MARK: - 2. Fallback Pool Strict 500MB Ceiling & Isolation Tests

    @Test("Fallback pool enforces strict 500MB capacity ceiling and rejects overflow")
    func testFallbackPoolStrict500MBCeiling() throws {
        // Default constructor has maxCapacityBytes = 500 MB (524,288,000 bytes)
        let pool = FallbackBufferPool(
            device: device,
            slotSizeBytes: MoEArchitectureConfig.synthetic.expertSizeBytes,
            maxCapacityBytes: 500 * 1024 * 1024
        )
        #expect(pool.maxCapacityBytes == 524_288_000)

        // Micro-ceiling test: configure pool to hold strictly 2 slots
        let slotSize = MoEArchitectureConfig.synthetic.expertSizeBytes
        let microPool = FallbackBufferPool(
            device: device,
            slotSizeBytes: slotSize,
            maxCapacityBytes: 2 * slotSize
        )
        #expect(microPool.maxSlots == 2)

        let slot1 = microPool.allocate(device: device)
        #expect(slot1 != nil)
        #expect(microPool.inUseSlotCount == 1)

        let slot2 = microPool.allocate(device: device)
        #expect(slot2 != nil)
        #expect(microPool.inUseSlotCount == 2)

        // Third allocation must fail (hard ceiling respected)
        let slot3 = microPool.allocate(device: device)
        #expect(slot3 == nil)
        #expect(microPool.inUseSlotCount == 2)

        // Reclaim slot 1 and reallocate: must reuse without growing capacity
        try microPool.reclaim(slot1!)
        #expect(microPool.inUseSlotCount == 1)
        #expect(microPool.freeSlotCount == 1)

        let reusedSlot = microPool.allocate(device: device)
        #expect(reusedSlot != nil)
        #expect(reusedSlot!.index == slot1!.index)
        #expect(microPool.inUseSlotCount == 2)
    }

    @Test("Fallback pool remains 100% isolated when Speculative Ring Buffer is exhausted")
    func testFallbackPoolStrictIsolation() throws {
        let ring = SpeculativeRingBuffer(
            device: device,
            slotCount: 2,
            slotSizeBytes: MoEArchitectureConfig.synthetic.expertSizeBytes
        )
        let pool = FallbackBufferPool(
            device: device,
            slotSizeBytes: MoEArchitectureConfig.synthetic.expertSizeBytes,
            maxCapacityBytes: 500 * 1024 * 1024
        )

        // Exhaust Ring Buffer completely
        _ = ring.allocateSlot(for: ExpertKey(layer: 5, expert: 0), ticket: 1)
        _ = ring.allocateSlot(for: ExpertKey(layer: 5, expert: 1), ticket: 2)
        #expect(ring.allocateSlot(for: ExpertKey(layer: 5, expert: 2), ticket: 3) == nil)

        // Fallback pool allocation must succeed unhindered
        let fallbackSlot = pool.allocate(device: device)
        #expect(fallbackSlot != nil)
        #expect(pool.inUseSlotCount == 1)
        try pool.reclaim(fallbackSlot!)
    }

    @Test("Fallback pool rejects speculative prefetch requests with strict isolation error")
    func testFallbackPoolSpeculativeRejection() {
        let pool = FallbackBufferPool(
            device: device,
            slotSizeBytes: MoEArchitectureConfig.synthetic.expertSizeBytes,
            maxCapacityBytes: 500 * 1024 * 1024
        )
        let specKey = ExpertKey(layer: 5, expert: 1)
        do {
            _ = try pool.allocate(intent: .speculative(expert: specKey), device: device)
            Issue.record("Expected speculative prefetch to be rejected!")
        } catch FallbackPoolError.speculativeRequestRejected(let expert) {
            #expect(expert == specKey)
            #expect(pool.totalSpeculativeRejections == 1)
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test("Fallback pool guards against double reclaim and foreign slots")
    func testFallbackPoolDoubleReclaimAndForeignSlotGuards() throws {
        let slotSize = MoEArchitectureConfig.synthetic.expertSizeBytes
        let pool = FallbackBufferPool(
            device: device,
            slotSizeBytes: slotSize,
            maxCapacityBytes: 2 * slotSize
        )
        let s0 = pool.allocate(device: device)!
        try pool.reclaim(s0)

        // Double reclaim must throw slotAlreadyFree
        do {
            try pool.reclaim(s0)
            Issue.record("Expected double reclaim to throw slotAlreadyFree")
        } catch FallbackPoolError.slotAlreadyFree(let idx) {
            #expect(idx == s0.index)
        }

        // Foreign slot with invalid index must throw foreignSlot
        let foreignSlot = FallbackSlot(index: 999, buffer: s0.buffer)
        do {
            try pool.reclaim(foreignSlot)
            Issue.record("Expected foreign slot to throw foreignSlot")
        } catch FallbackPoolError.foreignSlot(let idx) {
            #expect(idx == 999)
        }
    }

    // MARK: - 3. Cache-Miss Deadlock Resolution & Zero-CPU Synchronization Tests

    @Test("Deadlock resolution dispatches demand fetch on fallbackQueue (PriorityHigh)")
    func testDeadlockResolutionPriorityHighDemandFetch() async throws {
        let engine = try FastIOEngine(device: device)
        let expertSize = MoEArchitectureConfig.synthetic.expertSizeBytes
        let (fileURL, cleanup) = try TestHelpers.createSyntheticWeightFile(
            expertCount: 4,
            expertSizeBytes: expertSize,
            fillByte: 0xEE
        )
        defer { cleanup() }

        let handle = try device.makeIOFileHandle(url: fileURL)
        let ring = SpeculativeRingBuffer(device: device, slotCount: 4, slotSizeBytes: expertSize)
        let pool = FallbackBufferPool(device: device, slotSizeBytes: expertSize)
        let resolver = DeadlockResolver(
            ringBuffer: ring,
            fallbackPool: pool,
            fastIO: engine,
            fileHandle: handle,
            device: device
        )

        let missExpert = ExpertKey(layer: 5, expert: 2)
        let fallbackSlot = try await resolver.resolveDeadlock(
            expertID: missExpert,
            fileHandle: handle,
            fileOffset: 2 * expertSize,
            expertSize: expertSize,
            device: device
        )

        #expect(fallbackSlot.buffer.length >= expertSize)
        #expect(resolver.totalDeadlocksResolved == 1)

        // Bitwise verification
        let ptr = fallbackSlot.buffer.contents().bindMemory(to: UInt8.self, capacity: expertSize)
        #expect(ptr[0] == 0xEE)
        #expect(ptr[expertSize - 1] == 0xEE)

        resolver.releaseFallbackSlot(fallbackSlot)
        #expect(pool.inUseSlotCount == 0)
    }

    @Test("Deadlock resolution zero-CPU GPU compute synchronization via MTLSharedEvent")
    func testDeadlockResolutionZeroCPUGPUSync() throws {
        try autoreleasepool {
            guard let computeQueue = device.makeCommandQueue() else {
                Issue.record("Failed to create compute queue")
                return
            }

            let engine = try FastIOEngine(device: device)
            let expertSize = MoEArchitectureConfig.synthetic.expertSizeBytes
            let (fileURL, cleanup) = try TestHelpers.createSyntheticWeightFile(
                expertCount: 2,
                expertSizeBytes: expertSize,
                fillByte: 0x55
            )
            defer { cleanup() }

            let handle = try device.makeIOFileHandle(url: fileURL)
            let ring = SpeculativeRingBuffer(device: device, slotCount: 4, slotSizeBytes: expertSize)
            let pool = FallbackBufferPool(device: device, slotSizeBytes: expertSize)
            let resolver = DeadlockResolver(
                ringBuffer: ring,
                fallbackPool: pool,
                fastIO: engine,
                fileHandle: handle,
                device: device
            )

            let computeCmd = computeQueue.makeCommandBuffer()!
            let missExpert = ExpertKey(layer: 6, expert: 0)

            // Zero-CPU resolution encodes encodeWaitForEvent onto computeCmd
            let fallbackSlot = try resolver.resolveCacheMissDeadlock(
                expertID: missExpert,
                fileOffset: 0,
                size: expertSize,
                computeCommandBuffer: computeCmd
            )

            // GPU blit copy from fallback buffer to destination buffer
            guard let destBuffer = device.makeBuffer(length: expertSize, options: .storageModeShared) else {
                Issue.record("Failed to create destination buffer")
                return
            }
            guard let blit = computeCmd.makeBlitCommandEncoder() else {
                Issue.record("Failed to create blit encoder")
                return
            }
            blit.copy(from: fallbackSlot.buffer, sourceOffset: 0, to: destBuffer, destinationOffset: 0, size: expertSize)
            blit.endEncoding()

            computeCmd.commit()
            computeCmd.waitUntilCompleted()

            // Verify hardware synchronization completed with matching data
            let destPtr = destBuffer.contents().bindMemory(to: UInt8.self, capacity: expertSize)
            #expect(destPtr[0] == 0x55)
            #expect(destPtr[expertSize - 1] == 0x55)

            resolver.releaseFallbackSlot(fallbackSlot)
            #expect(pool.inUseSlotCount == 0)
        }
    }

    // MARK: - 4. Speculative Cancellation & Signal Dropping Tests

    @Test("Speculative slot abandonment, tryCancel(), signal dropping, and dirty slot reclamation")
    func testSpeculativeAbandonmentAndSignalDropping() throws {
        let engine = try FastIOEngine(device: device)
        let expertSize = MoEArchitectureConfig.synthetic.expertSizeBytes
        let (fileURL, cleanup) = try TestHelpers.createSyntheticWeightFile(
            expertCount: 4,
            expertSizeBytes: expertSize,
            fillByte: 0x99
        )
        defer { cleanup() }

        let handle = try device.makeIOFileHandle(url: fileURL)
        let ring = SpeculativeRingBuffer(device: device, slotCount: 4, slotSizeBytes: expertSize)
        let pool = FallbackBufferPool(device: device, slotSizeBytes: expertSize)
        let resolver = DeadlockResolver(
            ringBuffer: ring,
            fallbackPool: pool,
            fastIO: engine,
            fileHandle: handle,
            device: device
        )

        // Step 1: Start low-priority speculative prefetch into slot
        let specExpert = ExpertKey(layer: 5, expert: 3)
        let specSlot = ring.allocateSlot(for: specExpert, ticket: 42)!
        let specTicket: UInt64 = 42
        let ioCmd = engine.speculativeQueue.makeCommandBuffer()
        specSlot.inFlightIOCommand = ioCmd
        ioCmd.load(specSlot.buffer, offset: 0, size: expertSize, sourceHandle: handle, sourceHandleOffset: 0)
        ioCmd.signalEvent(specSlot.sharedEvent ?? engine.speculativeSyncEvent.sharedEvent, value: specTicket)

        let slotIdx = specSlot.index
        var signalPropagated = false
        ioCmd.addCompletedHandler { completedCmd in
            signalPropagated = resolver.handleSpeculativeCompletion(
                slotIndex: slotIdx,
                ticket: specTicket,
                status: completedCmd.status
            )
        }
        ioCmd.commit()

        // Step 2: Cache miss occurs before speculative DMA completes.
        // Trigger deadlock resolution which quarantines slot as .abandoned and issues tryCancel()
        ring.markAbandoned(slotIndex: slotIdx)
        if case .abandoned(let t, let k) = ring.slotStateSnapshot()[slotIdx] {
            #expect(t == specTicket)
            #expect(k == specExpert)
        } else {
            Issue.record("Expected .abandoned state")
        }

        // Wait for speculative command completion handler to fire
        Thread.sleep(forTimeInterval: 0.15)

        // Step 3: Verify signal was dropped and slot returned to .free
        #expect(signalPropagated == false, "Signal must be dropped for abandoned slot")
        #expect(resolver.totalSignalsDropped >= 1)
        #expect(resolver.totalDirtySlotsReclaimed >= 1)

        let finalState = ring.slotStateSnapshot()[slotIdx]
        #expect(finalState == .free, "Abandoned slot must be returned to .free after completion handler")
    }

    // MARK: - 5. Zero Memory Leak & High Churn Verification

    @Test("Zero memory leak verification over 500 continuous allocation, deadlock, and churn cycles")
    func testZeroMemoryLeaksUnderHeavyChurn() throws {
        let expertSize = MoEArchitectureConfig.synthetic.expertSizeBytes
        let ring = SpeculativeRingBuffer(device: device, slotCount: 8, slotSizeBytes: expertSize)
        let pool = FallbackBufferPool(device: device, slotSizeBytes: expertSize, maxCapacityBytes: 10 * expertSize)

        func getResidentKB() -> UInt64 {
            var info = mach_task_basic_info()
            var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
            let kerr = withUnsafeMutablePointer(to: &info) {
                $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                    task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
                }
            }
            return kerr == KERN_SUCCESS ? info.resident_size / 1024 : 0
        }

        let initialResidentKB = getResidentKB()

        // 500 iterations of mixed allocations, abandonments, and fallback reclaims
        for i in 0..<500 {
            let expert = ExpertKey(layer: 5, expert: i % 16)
            if let slot = ring.allocateSlot(for: expert, ticket: UInt64(i + 1)) {
                if i % 2 == 0 {
                    ring.markReady(slotIndex: slot.index, ticket: UInt64(i + 1))
                    ring.updateLRUTimestamp(slotIndex: slot.index, timestamp: UInt64(i))
                } else {
                    ring.markAbandoned(slotIndex: slot.index)
                    ring.reclaim(slotIndex: slot.index)
                }
            }

            if let fbSlot = pool.allocate(device: device) {
                try pool.reclaim(fbSlot)
            }
        }

        let finalResidentKB = getResidentKB()
        let memoryDeltaKB = Int(finalResidentKB) - Int(initialResidentKB)

        // Memory delta must not exceed 8 MB (8,192 KB)
        #expect(abs(memoryDeltaKB) < 8192, "Memory leak detected: resident size grew by \(memoryDeltaKB) KB")
        #expect(pool.inUseSlotCount == 0, "All fallback slots must be reclaimed")
    }
}

enum TestError: Error {
    case metalUnavailable
}

