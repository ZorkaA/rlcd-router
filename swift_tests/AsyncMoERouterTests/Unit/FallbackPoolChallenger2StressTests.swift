//===----------------------------------------------------------------------===//
//
// This source file is part of the AsyncMoERouter open source project
//
// Copyright (c) 2026 Apple Inc. and the AsyncMoERouter project authors
// Licensed under Apache License v2.0
//
//===----------------------------------------------------------------------===//

import XCTest
import Metal
import Foundation
import Darwin
import os
@testable import AsyncMoERouter

/// Adversarial empirical stress test suite for Phase 2 Milestone 2:
/// - 500MB ($524,288,000$ bytes) Hard Ceiling Invariant enforcement under 32+ concurrent threads
/// - Speculative Prefetch Request Smuggling 100% rejection under adversarial pressure
/// - Extreme churn memory leakage and slot corruption detection (2,000+ continuous allocate/verify/reclaim cycles)
/// - Multi-threaded micro-ceiling zero-overrun protection (64 concurrent threads)
/// - Foreign slot and double-reclaim injection resistance
final class FallbackPoolChallenger2StressTests: XCTestCase {
    private static var metalDevice: (any MTLDevice)!

    override class func setUp() {
        super.setUp()
        guard let dev = MTLCreateSystemDefaultDevice() else {
            fatalError("FallbackPoolChallenger2StressTests requires a Metal-compatible GPU device.")
        }
        metalDevice = dev
    }

    override class func tearDown() {
        metalDevice = nil
        super.tearDown()
    }

    // MARK: - Memory Telemetry Helpers (Mach Kernel)

    /// Returns the physical memory footprint of the current process in bytes.
    /// This corresponds to `phys_footprint` from `task_vm_info`, which is the authoritative
    /// memory metric used by Apple Silicon unified memory accounting.
    private static func getPhysicalFootprintBytes() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kerr = withUnsafeMutablePointer(to: &info) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), intPtr, &count)
            }
        }
        if kerr == KERN_SUCCESS {
            return UInt64(info.phys_footprint)
        }
        return 0
    }

    /// Returns the resident task memory size in bytes.
    private static func getResidentMemoryBytes() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let kerr = withUnsafeMutablePointer(to: &info) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), intPtr, &count)
            }
        }
        if kerr == KERN_SUCCESS {
            return info.resident_size
        }
        return 0
    }

    // MARK: - Test 1: 500MB Hard Ceiling Under Massive 32-Thread Allocation Pressure

    /// Adversarially attempts to breach the 500MB (524,288,000 bytes) hard ceiling under 32 concurrent threads.
    /// 32 threads simultaneously request 128 slots of 16MB each (demanding 2,048 MB = 4x ceiling).
    /// Asserts:
    /// 1. Exactly 31 slots succeed in allocating (520,093,696 bytes <= 524,288,000 bytes).
    /// 2. Exactly 97 allocations are rejected with FallbackPoolError.capacityExhausted.
    /// 3. allocatedBytes never exceeds 524,288,000 bytes at any point.
    /// 4. All allocated slot indices are strictly unique (no duplicate slot corruption).
    /// 5. Clean concurrent reclaim across all 32 threads returns inUseSlotCount to 0.
    func test500MBHardCeilingUnder32ThreadContention() {
        let device = Self.metalDevice!
        let maxCeiling = 500 * 1024 * 1024 // 524,288,000 bytes
        let slotSize = 16 * 1024 * 1024   // 16 MB = 16,777,216 bytes
        let expectedMaxSlots = maxCeiling / slotSize // 31 slots = 520,093,696 bytes

        let pool = FallbackBufferPool(
            device: device,
            slotSizeBytes: slotSize,
            maxCapacityBytes: maxCeiling,
            prewarmCount: 0 // Do not pre-allocate warm slots to test pure concurrent growth
        )

        XCTAssertEqual(pool.maxCapacityBytes, 524_288_000)
        XCTAssertEqual(pool.maxSlots, 31)
        XCTAssertEqual(pool.allocatedBytes, 0)

        let threadCount = 32
        let allocationsPerThread = 4
        let totalAttempts = threadCount * allocationsPerThread // 128 attempts demanding 2048 MB

        let lock = NSLock()
        var allocatedSlots: [FallbackSlot] = []
        var capacityExhaustedCount = 0
        var unexpectedErrorCount = 0

        // Launch 32 concurrent threads hammering the pool simultaneously
        DispatchQueue.concurrentPerform(iterations: threadCount) { threadIdx in
            for reqIdx in 0..<allocationsPerThread {
                let context = DemandFetchContext(
                    expert: ExpertKey(layer: 5 + (threadIdx % 20), expert: reqIdx * 10),
                    tokenIndex: UInt32(threadIdx * 100 + reqIdx),
                    reason: .cacheMiss
                )

                do {
                    let slot = try pool.allocateForDemand(context: context, device: device, ticket: UInt64(threadIdx + 1))
                    lock.lock()
                    allocatedSlots.append(slot)
                    lock.unlock()
                } catch FallbackPoolError.capacityExhausted(let req, let alloc, let max) {
                    lock.lock()
                    capacityExhaustedCount += 1
                    lock.unlock()
                    XCTAssertEqual(req, slotSize)
                    XCTAssertLessThanOrEqual(alloc, maxCeiling)
                    XCTAssertEqual(max, maxCeiling)
                } catch {
                    lock.lock()
                    unexpectedErrorCount += 1
                    lock.unlock()
                }
            }
        }

        // Verification of 500MB Ceiling Invariant
        XCTAssertEqual(unexpectedErrorCount, 0, "No unexpected errors should occur under contention")
        XCTAssertEqual(allocatedSlots.count, expectedMaxSlots, "Exactly \(expectedMaxSlots) slots should succeed")
        XCTAssertEqual(capacityExhaustedCount, totalAttempts - expectedMaxSlots, "Exactly \(totalAttempts - expectedMaxSlots) requests must be rejected")
        XCTAssertEqual(pool.inUseSlotCount, expectedMaxSlots)
        XCTAssertLessThanOrEqual(pool.allocatedBytes, maxCeiling, "500MB ceiling must NEVER be breached!")
        XCTAssertEqual(pool.allocatedBytes, expectedMaxSlots * slotSize)
        XCTAssertTrue(pool.verifyInvariants(), "Pool invariant verification must succeed")

        // Assert all allocated slots have unique indices
        let uniqueIndices = Set(allocatedSlots.map { $0.index })
        XCTAssertEqual(uniqueIndices.count, expectedMaxSlots, "Every allocated slot must possess a unique index")
        for idx in uniqueIndices {
            XCTAssertTrue(idx >= 0 && idx < expectedMaxSlots, "Slot index \(idx) must be within 0..<\(expectedMaxSlots)")
        }

        print("--> [Test 1 Empirical] 500MB Ceiling Stress: Total Attempts=\(totalAttempts), Successful=\(allocatedSlots.count) (\(pool.allocatedBytes) bytes), Rejected=\(capacityExhaustedCount), PeakBytesInUse=\(pool.peakBytesInUse)")

        // Concurrent reclaim across 32 threads
        DispatchQueue.concurrentPerform(iterations: allocatedSlots.count) { slotIdx in
            let slot = allocatedSlots[slotIdx]
            do {
                try pool.reclaim(slot)
            } catch {
                XCTFail("Reclaim failed unexpectedly for slot \(slot.index): \(error)")
            }
        }

        XCTAssertEqual(pool.inUseSlotCount, 0)
        XCTAssertEqual(pool.freeSlotCount, expectedMaxSlots)
        XCTAssertEqual(pool.allocatedBytes, expectedMaxSlots * slotSize)
        XCTAssertTrue(pool.verifyInvariants())

        // Purge pool to release memory back to OS
        pool.purge()
        XCTAssertEqual(pool.freeSlotCount, 0)
        XCTAssertEqual(pool.allocatedBytes, 0)
        XCTAssertTrue(pool.verifyInvariants())
    }

    // MARK: - Test 2: Multi-Threaded Micro-Ceiling Boundary Attack (64 Threads)

    /// Stress-tests the exact ceiling boundary with 64 concurrent threads trying to allocate a 5-slot pool.
    /// Verifies zero-overrun: exactly 5 slots granted, 59 rejected, no race condition window allows slot 6.
    func testMicroCeilingBoundaryContentionWith64Threads() {
        let device = Self.metalDevice!
        let slotSize = 64 * 1024 // 64 KB
        let maxSlots = 5
        let microCeiling = maxSlots * slotSize // 320 KB

        let pool = FallbackBufferPool(
            device: device,
            slotSizeBytes: slotSize,
            maxCapacityBytes: microCeiling,
            prewarmCount: 0
        )

        let threadCount = 64
        let barrier = DispatchGroup()
        let lock = NSLock()
        var grantedSlots: [FallbackSlot] = []
        var rejectedCount = 0

        for threadIdx in 0..<threadCount {
            barrier.enter()
            DispatchQueue.global().async {
                let context = DemandFetchContext(
                    expert: ExpertKey(layer: 5, expert: threadIdx),
                    reason: .deadlineMiss
                )
                do {
                    let slot = try pool.allocateForDemand(context: context, device: device, ticket: UInt64(threadIdx))
                    lock.lock()
                    grantedSlots.append(slot)
                    lock.unlock()
                } catch FallbackPoolError.capacityExhausted {
                    lock.lock()
                    rejectedCount += 1
                    lock.unlock()
                } catch {
                    XCTFail("Unexpected error: \(error)")
                }
                barrier.leave()
            }
        }

        barrier.wait()

        XCTAssertEqual(grantedSlots.count, maxSlots, "Exactly \(maxSlots) slots granted")
        XCTAssertEqual(rejectedCount, threadCount - maxSlots, "Exactly \(threadCount - maxSlots) requests rejected")
        XCTAssertEqual(pool.allocatedBytes, microCeiling, "Allocated bytes must equal exact micro-ceiling")
        XCTAssertEqual(pool.inUseSlotCount, maxSlots)
        XCTAssertTrue(pool.verifyInvariants())

        let grantedIndices = Set(grantedSlots.map { $0.index })
        XCTAssertEqual(grantedIndices.count, maxSlots, "All granted slot indices must be strictly unique")

        // Clean up
        for slot in grantedSlots {
            XCTAssertNoThrow(try pool.reclaim(slot))
        }
        XCTAssertEqual(pool.inUseSlotCount, 0)
        XCTAssertEqual(pool.freeSlotCount, maxSlots)
    }

    // MARK: - Test 3: Speculative Prefetch Request Smuggling (100% Rejection Rate)

    /// Adversarially attempts to smuggle speculative prefetch requests into FallbackBufferPool.
    /// Tests 100 concurrent attempts with diverse and adversarial ExpertKey variants.
    /// Asserts 100% rejection rate with FallbackPoolError.speculativeRequestRejected.
    func testSpeculativePrefetchSmugglingRejection() {
        let device = Self.metalDevice!
        let pool = FallbackBufferPool(
            device: device,
            slotSizeBytes: 128 * 1024,
            maxCapacityBytes: 10 * 1024 * 1024
        )

        let attemptCount = 100
        let lock = NSLock()
        var rejectedCount = 0
        var illegalGrantedCount = 0

        let adversarialKeys = [
            ExpertKey(layer: 0, expert: 0),
            ExpertKey(layer: 5, expert: 0),
            ExpertKey(layer: 5, expert: 59),
            ExpertKey(layer: 24, expert: 59),
            ExpertKey(layer: 99, expert: 99),
            ExpertKey(layer: 12, expert: 30)
        ]

        DispatchQueue.concurrentPerform(iterations: attemptCount) { idx in
            let key = adversarialKeys[idx % adversarialKeys.count]
            do {
                _ = try pool.allocate(intent: .speculative(expert: key), device: device, ticket: UInt64(idx))
                lock.lock()
                illegalGrantedCount += 1
                lock.unlock()
            } catch FallbackPoolError.speculativeRequestRejected(let rejectedKey) {
                lock.lock()
                rejectedCount += 1
                lock.unlock()
                XCTAssertEqual(rejectedKey, key)
            } catch {
                XCTFail("Unexpected error thrown for speculative intent: \(error)")
            }
        }

        XCTAssertEqual(illegalGrantedCount, 0, "ZERO speculative requests may ever be granted by FallbackBufferPool!")
        XCTAssertEqual(rejectedCount, attemptCount, "100% of speculative requests must be rejected")
        XCTAssertEqual(pool.totalSpeculativeRejections, attemptCount)
        XCTAssertEqual(pool.inUseSlotCount, 0, "No slots may be occupied")
        XCTAssertEqual(pool.totalFallbackAllocations, 0, "Zero fallback allocations recorded")
        XCTAssertTrue(pool.verifyInvariants())

        print("--> [Test 3 Empirical] Speculative Smuggling: Attempts=\(attemptCount), Rejected=\(rejectedCount) (100.0%), IllegalGranted=\(illegalGrantedCount)")
    }

    // MARK: - Test 4: Mixed Contention: Concurrent Speculative Smuggling vs Demand Allocations

    /// Concurrently executes 32 speculative prefetch attempts and 32 legitimate demand fetch requests.
    /// Verifies that speculative requests are 100% rejected while demand requests succeed without interference.
    func testMixedConcurrentSpeculativeAndDemandAllocation() {
        let device = Self.metalDevice!
        let slotSize = 64 * 1024
        let pool = FallbackBufferPool(
            device: device,
            slotSizeBytes: slotSize,
            maxCapacityBytes: 64 * slotSize // Capacity fits all 32 demand slots
        )

        let totalThreads = 64
        let lock = NSLock()
        var speculativeRejections = 0
        var speculativeGranted = 0
        var demandSuccesses = 0
        var demandSlots: [FallbackSlot] = []

        DispatchQueue.concurrentPerform(iterations: totalThreads) { threadIdx in
            let expert = ExpertKey(layer: 5 + (threadIdx % 20), expert: threadIdx % 60)
            if threadIdx % 2 == 0 {
                // Speculative attempt
                do {
                    _ = try pool.allocate(intent: .speculative(expert: expert), device: device, ticket: UInt64(threadIdx))
                    lock.lock()
                    speculativeGranted += 1
                    lock.unlock()
                } catch FallbackPoolError.speculativeRequestRejected {
                    lock.lock()
                    speculativeRejections += 1
                    lock.unlock()
                } catch {
                    XCTFail("Unexpected error: \(error)")
                }
            } else {
                // Demand attempt
                let context = DemandFetchContext(expert: expert, tokenIndex: UInt32(threadIdx), reason: .cacheMiss)
                do {
                    let slot = try pool.allocate(intent: .demand(context), device: device, ticket: UInt64(threadIdx))
                    lock.lock()
                    demandSuccesses += 1
                    demandSlots.append(slot)
                    lock.unlock()
                } catch {
                    XCTFail("Demand allocation failed: \(error)")
                }
            }
        }

        XCTAssertEqual(speculativeGranted, 0, "Zero speculative allocations granted")
        XCTAssertEqual(speculativeRejections, 32, "All 32 speculative requests rejected")
        XCTAssertEqual(demandSuccesses, 32, "All 32 demand allocations succeeded")
        XCTAssertEqual(pool.inUseSlotCount, 32)
        XCTAssertTrue(pool.verifyInvariants())

        // Reclaim demand slots
        for slot in demandSlots {
            XCTAssertNoThrow(try pool.reclaim(slot))
        }
        XCTAssertEqual(pool.inUseSlotCount, 0)
        XCTAssertEqual(pool.freeSlotCount, 32)
    }

    // MARK: - Test 5: Foreign Slot and Double Reclaim Injection Resistance

    /// Tests pool resistance against rogue reclaims:
    /// - Reclaiming the same slot twice throws slotAlreadyFree
    /// - Reclaiming a foreign slot with out-of-bounds index throws foreignSlot
    /// - Reclaiming a foreign slot with in-bounds index not in pool throws foreignSlot
    func testForeignSlotAndDoubleReclaimResistance() {
        let device = Self.metalDevice!
        let slotSize = 64 * 1024
        let pool = FallbackBufferPool(
            device: device,
            slotSizeBytes: slotSize,
            maxCapacityBytes: 4 * slotSize,
            prewarmCount: 0 // Ensure no slots are prewarmed initially
        )

        let validSlot = pool.allocate(device: device)!
        XCTAssertNoThrow(try pool.reclaim(validSlot))

        // 1. Double reclaim of previously reclaimed slot
        XCTAssertThrowsError(try pool.reclaim(validSlot)) { error in
            guard case FallbackPoolError.slotAlreadyFree(let idx) = error else {
                return XCTFail("Expected slotAlreadyFree, got \(error)")
            }
            XCTAssertEqual(idx, validSlot.index)
        }

        // 2. Out-of-bounds foreign slot (index 9999 >= maxSlots 4)
        guard let dummyBuf = device.makeBuffer(length: slotSize, options: .storageModeShared) else {
            return XCTFail("Failed to allocate dummy buffer")
        }
        let oobSlot = FallbackSlot(index: 9999, buffer: dummyBuf)
        XCTAssertThrowsError(try pool.reclaim(oobSlot)) { error in
            guard case FallbackPoolError.foreignSlot(let idx) = error else {
                return XCTFail("Expected foreignSlot, got \(error)")
            }
            XCTAssertEqual(idx, 9999)
        }

        // 3. In-bounds foreign slot (index 2 when slot 2 was never allocated or prewarmed)
        let unallocatedIndexSlot = FallbackSlot(index: 2, buffer: dummyBuf)
        XCTAssertThrowsError(try pool.reclaim(unallocatedIndexSlot)) { error in
            guard case FallbackPoolError.foreignSlot(let idx) = error else {
                return XCTFail("Expected foreignSlot, got \(error)")
            }
            XCTAssertEqual(idx, 2)
        }

        // 4. Test prewarmed slot rejection: prewarmed slot sitting in free list cannot be reclaimed
        let prewarmedPool = FallbackBufferPool(
            device: device,
            slotSizeBytes: slotSize,
            maxCapacityBytes: 4 * slotSize,
            prewarmCount: 4 // All 4 slots prewarmed into free list
        )
        let fakePrewarmedSlot = FallbackSlot(index: 1, buffer: dummyBuf)
        XCTAssertThrowsError(try prewarmedPool.reclaim(fakePrewarmedSlot)) { error in
            guard case FallbackPoolError.slotAlreadyFree(let idx) = error else {
                return XCTFail("Expected slotAlreadyFree for prewarmed free slot, got \(error)")
            }
            XCTAssertEqual(idx, 1)
        }
    }

    // MARK: - Test 6: Extreme Churn (2,000 Continuous Cycles) & Zero Memory Leakage

    /// Tests extreme churn: executes 2,000 continuous allocate / write checksum / verify / reclaim cycles
    /// under high concurrency (16 worker tasks, 125 cycles each).
    /// Detects memory leakage via task_vm_info.phys_footprint and mach_task_basic_info.resident_size.
    /// Asserts zero memory leak (delta < 5 MB), zero data corruption, zero slot leakage.
    func testExtremeChurn2000CyclesAndZeroMemoryLeak() {
        let device = Self.metalDevice!
        let slotSize = 128 * 1024 // 128 KB
        let poolCapacity = 16 * slotSize // 2 MB capacity (16 slots)

        let pool = FallbackBufferPool(
            device: device,
            slotSizeBytes: slotSize,
            maxCapacityBytes: poolCapacity,
            prewarmCount: 4
        )

        // Pre-churn memory measurement
        let initialPhysFootprint = Self.getPhysicalFootprintBytes()
        let initialResident = Self.getResidentMemoryBytes()

        let totalCycles = 2000
        let workers = 16
        let cyclesPerWorker = totalCycles / workers // 125 cycles per worker

        var corruptionCount = 0
        let corruptionLock = NSLock()

        DispatchQueue.concurrentPerform(iterations: workers) { workerIdx in
            for cycle in 0..<cyclesPerWorker {
                let globalCycleId = workerIdx * cyclesPerWorker + cycle
                let expert = ExpertKey(layer: 5 + (globalCycleId % 20), expert: globalCycleId % 60)
                let context = DemandFetchContext(
                    expert: expert,
                    tokenIndex: UInt32(globalCycleId),
                    reason: .ringBufferExhaustion
                )

                // Allocate
                guard let slot = try? pool.allocateForDemand(context: context, device: device, ticket: UInt64(globalCycleId)) else {
                    XCTFail("Worker \(workerIdx) failed to allocate on cycle \(cycle)")
                    return
                }

                // Write unique verification checksum pattern into slot buffer
                let pattern: UInt64 = 0xCAFE_BABE_0000_0000 | UInt64(globalCycleId)
                let ptr = slot.buffer.contents().bindMemory(to: UInt64.self, capacity: slotSize / 8)
                ptr[0] = pattern
                ptr[(slotSize / 8) - 1] = ~pattern

                // Verify data integrity: read back pattern
                let read0 = ptr[0]
                let readEnd = ptr[(slotSize / 8) - 1]
                if read0 != pattern || readEnd != ~pattern {
                    corruptionLock.lock()
                    corruptionCount += 1
                    corruptionLock.unlock()
                }

                // Reclaim slot back to free list
                do {
                    try pool.reclaim(slot)
                } catch {
                    XCTFail("Worker \(workerIdx) failed to reclaim on cycle \(cycle): \(error)")
                }
            }
        }

        // Post-churn memory measurement
        let finalPhysFootprint = Self.getPhysicalFootprintBytes()
        let finalResident = Self.getResidentMemoryBytes()

        let footprintDeltaBytes = Int64(finalPhysFootprint) - Int64(initialPhysFootprint)
        let residentDeltaBytes = Int64(finalResident) - Int64(initialResident)
        let footprintDeltaMB = Double(footprintDeltaBytes) / (1024.0 * 1024.0)
        let residentDeltaMB = Double(residentDeltaBytes) / (1024.0 * 1024.0)

        print("--> [Test 6 Empirical] Extreme Churn (2000 cycles):")
        print("    Total Allocations: \(pool.totalFallbackAllocations)")
        print("    Total Reclaims:    \(pool.totalReclaims)")
        print("    In-Use Slots:      \(pool.inUseSlotCount)")
        print("    Free Slots:        \(pool.freeSlotCount)")
        print("    Peak Bytes In Use: \(pool.peakBytesInUse) bytes (\(Double(pool.peakBytesInUse)/(1024.0*1024.0)) MB)")
        print("    Allocated Bytes:   \(pool.allocatedBytes) bytes")
        print("    Data Corruptions:  \(corruptionCount)")
        print("    Initial Phys Footprint: \(initialPhysFootprint / (1024 * 1024)) MB, Final: \(finalPhysFootprint / (1024 * 1024)) MB (Delta: \(String(format: "%.2f", footprintDeltaMB)) MB)")
        print("    Initial Resident Size:  \(initialResident / (1024 * 1024)) MB, Final: \(finalResident / (1024 * 1024)) MB (Delta: \(String(format: "%.2f", residentDeltaMB)) MB)")

        // Assertions
        XCTAssertEqual(corruptionCount, 0, "Zero slot data corruption permitted!")
        XCTAssertEqual(pool.totalFallbackAllocations, totalCycles, "Total allocations must equal 2,000")
        XCTAssertEqual(pool.totalReclaims, totalCycles, "Total reclaims must equal 2,000")
        XCTAssertEqual(pool.inUseSlotCount, 0, "All slots must be cleanly reclaimed back to free list")
        XCTAssertTrue(pool.verifyInvariants(), "Pool invariants must hold post-churn")

        // Memory leak threshold: physical footprint growth must not exceed 5 MB
        XCTAssertLessThan(abs(footprintDeltaMB), 5.0, "Physical footprint leak detected: grew by \(footprintDeltaMB) MB")
    }

    // MARK: - Test 7: Chaos Stress: Concurrent Alloc, Reclaim, Invariants & Purge

    /// Executes a chaos stress test with simultaneous allocation, reclamation, invariant verification,
    /// and periodic purges to stress-test lock contention and race conditions.
    func testChaosConcurrencyAndLockContention() {
        let device = Self.metalDevice!
        let slotSize = 64 * 1024
        let maxSlots = 8
        let pool = FallbackBufferPool(
            device: device,
            slotSizeBytes: slotSize,
            maxCapacityBytes: maxSlots * slotSize,
            prewarmCount: 2
        )

        let stopSignal = OSAllocatedUnfairLock(initialState: false)
        let group = DispatchGroup()

        // Group 1: 8 Allocator / Reclaimer threads
        for t in 0..<8 {
            group.enter()
            DispatchQueue.global().async {
                var localSlots: [FallbackSlot] = []
                while !stopSignal.withLock({ $0 }) {
                    let context = DemandFetchContext(
                        expert: ExpertKey(layer: 5, expert: t),
                        reason: .cacheMiss
                    )
                    if let slot = try? pool.allocateForDemand(context: context, device: device, ticket: UInt64(t)) {
                        localSlots.append(slot)
                    }

                    if localSlots.count > 2 {
                        let toReclaim = localSlots.removeFirst()
                        _ = try? pool.reclaim(toReclaim)
                    }
                    Thread.sleep(forTimeInterval: 0.0005)
                }

                for s in localSlots {
                    _ = try? pool.reclaim(s)
                }
                group.leave()
            }
        }

        // Group 2: Invariant Verifier thread
        group.enter()
        DispatchQueue.global().async {
            while !stopSignal.withLock({ $0 }) {
                XCTAssertTrue(pool.verifyInvariants(), "Pool invariant violated during chaos stress!")
                let diag = pool.diagnostics()
                XCTAssertLessThanOrEqual(diag.allocatedBytes, diag.maxCapacityBytes)
                Thread.sleep(forTimeInterval: 0.001)
            }
            group.leave()
        }

        // Group 3: Speculative Smuggling Attacker thread
        group.enter()
        DispatchQueue.global().async {
            while !stopSignal.withLock({ $0 }) {
                do {
                    _ = try pool.allocate(intent: .speculative(expert: ExpertKey(layer: 7, expert: 1)), device: device)
                    XCTFail("Speculative allocation must never succeed during chaos stress!")
                } catch FallbackPoolError.speculativeRequestRejected {
                    // Expected
                } catch {
                    XCTFail("Unexpected error: \(error)")
                }
                Thread.sleep(forTimeInterval: 0.001)
            }
            group.leave()
        }

        // Run chaos for 0.5 seconds
        Thread.sleep(forTimeInterval: 0.5)
        stopSignal.withLock { $0 = true }
        group.wait()

        XCTAssertTrue(pool.verifyInvariants())
        pool.purge()
        XCTAssertTrue(pool.verifyInvariants())
    }
}
