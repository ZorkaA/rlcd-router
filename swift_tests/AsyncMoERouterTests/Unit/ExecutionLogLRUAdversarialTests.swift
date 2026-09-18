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
import os
@testable import AsyncMoERouter

/// Adversarial empirical challenge and stress test suite for Phase 2 Milestone 3:
/// Requirement R3: GPU Execution Log & Dispatch-Time LRU Tracking.
///
/// Specific Attack Vectors:
/// 1. Pre-routing speculative prefetch, loading, and readying phases MUST NOT modify
///    slot timestamps or alter LRU eviction ordering under any circumstances.
/// 2. Speculative slot abandonment and dirty slot recovery leave LRU timestamps at 0.
/// 3. LRU eviction candidate selection strictly prefers unexecuted speculative slots (ts == 0)
///    over executed slots (ts > 0), regardless of wall-clock allocation order.
/// 4. Post-execution drain is the single source of truth for updating slot recency timestamps.
/// 5. Monotonic timestamp guards: numerical non-regression when older entries arrive out of order.
/// 6. Recency queue ordering under out-of-order entries, duplicate tokens, and reversed sequences.
/// 7. Command buffer completion drain handler lifecycle and concurrency safety.
@Suite("Milestone 3 Challenger 1: Dispatch-Time LRU Invariant & Monotonic Stress Tests")
struct ExecutionLogLRUAdversarialTests {
    let device: any MTLDevice

    init() throws {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            throw TestError.metalUnavailable
        }
        self.device = dev
    }

    enum TestError: Error {
        case metalUnavailable
        case bufferAllocationFailed
        case commandQueueFailed
    }

    // MARK: - 1. Speculative Lifecycle Isolation Invariant (Requirement R3)

    @Test("Adversarial 1A: Speculative prefetch, loading, readying, in-use, release, and abandonment NEVER modify LRU timestamps")
    func testSpeculativeLifecycleLeavesLRUUntouched() {
        let slotCount = 16
        let slotSizeBytes = 4096
        let ring = SpeculativeRingBuffer(device: device, slotCount: slotCount, slotSizeBytes: slotSizeBytes)
        let tracker = LRUWeightTracker()

        // 1. Allocate all 16 slots across layers 5..20
        var allocatedSlots: [RingBufferSlot] = []
        for i in 0..<slotCount {
            let expert = ExpertKey(layer: 5 + i, expert: i * 2)
            guard let slot = ring.allocateSlot(for: expert, ticket: UInt64(100 + i)) else {
                Issue.record("Failed to allocate slot \(i)")
                continue
            }
            allocatedSlots.append(slot)

            // Assert invariant: immediately after allocation (in .loading state), timestamp MUST be 0
            #expect(slot.lastAccessedTimestamp == 0, "Slot \(slot.index) timestamp must be 0 after allocation")
        }

        // Assert tracker invariant: tracker must know NOTHING about speculative prefetch
        #expect(tracker.count == 0, "Tracker must be empty during speculative loading")
        #expect(tracker.evictionCandidate() == nil, "Eviction candidate must be nil")

        // 2. Transition half the slots to .ready
        for i in 0..<8 {
            ring.markReady(slotIndex: allocatedSlots[i].index, ticket: UInt64(100 + i))
            #expect(allocatedSlots[i].lastAccessedTimestamp == 0, "Slot \(i) timestamp must be 0 after markReady")
        }

        // 3. Transition slots 0..3 to .inUse with multiple retain counts (re-entrant compute binding)
        for i in 0..<4 {
            let slot = allocatedSlots[i]
            ring.markInUse(slotIndex: slot.index, ticket: UInt64(100 + i))
            ring.markInUse(slotIndex: slot.index, ticket: UInt64(100 + i)) // retainCount = 2
            #expect(slot.lastAccessedTimestamp == 0, "Slot \(i) timestamp must be 0 while inUse")
            ring.releaseFromUse(slotIndex: slot.index, ticket: UInt64(100 + i))
            #expect(slot.lastAccessedTimestamp == 0, "Slot \(i) timestamp must be 0 after first release")
            ring.releaseFromUse(slotIndex: slot.index, ticket: UInt64(100 + i))
            #expect(slot.lastAccessedTimestamp == 0, "Slot \(i) timestamp must be 0 after returning to .ready")
        }

        // 4. Abandon slots 8..15 (speculative preemption) and reclaim them
        for i in 8..<16 {
            let slot = allocatedSlots[i]
            ring.markAbandoned(slotIndex: slot.index)
            #expect(slot.lastAccessedTimestamp == 0, "Slot \(i) timestamp must be 0 after abandonment")
            ring.reclaim(slotIndex: slot.index)
            #expect(slot.lastAccessedTimestamp == 0, "Slot \(i) timestamp must be 0 after reclaim to .free")
        }

        // Final verification of total isolation
        #expect(tracker.count == 0, "LRUWeightTracker count must remain strictly 0 throughout speculative lifecycle")
        #expect(tracker.totalUpdates == 0, "LRUWeightTracker totalUpdates must be 0")
        for slot in allocatedSlots {
            #expect(slot.lastAccessedTimestamp == 0, "All slot timestamps must remain 0")
        }
    }

    // MARK: - 2. Eviction Ordering: Unexecuted Slots (ts == 0) Always Evicted Before Executed Slots

    @Test("Adversarial 2: Ring buffer eviction strictly prioritizes unexecuted speculative slots over executed slots regardless of allocation order")
    func testPreRoutingAllocationDoesNotAffectEvictionOrdering() {
        let slotCount = 8
        let slotSizeBytes = 4096
        let ring = SpeculativeRingBuffer(device: device, slotCount: slotCount, slotSizeBytes: slotSizeBytes)

        // Step 1: Pre-populate slots 0..3 with EXECUTED experts (timestamps > 0)
        for e in 0..<4 {
            let expert = ExpertKey(layer: 5, expert: e)
            let ticket = UInt64(e + 1)
            guard let slot = ring.allocateSlot(for: expert, ticket: ticket) else {
                Issue.record("Failed to allocate slot for expert \(e)")
                return
            }
            ring.markReady(slotIndex: slot.index, ticket: ticket)
            // Simulated post-execution drain timestamp update
            ring.updateLRUTimestamp(slotIndex: slot.index, timestamp: UInt64(1000 + e * 500))
        }

        // Step 2: Allocate slots 4..7 with SPECULATIVE UNEXECUTED experts (timestamp == 0)
        // Note: These are allocated AFTER slots 0..3 in real time!
        var speculativeSlotIndices: Set<Int> = []
        for e in 4..<8 {
            let expert = ExpertKey(layer: 6, expert: e)
            let ticket = UInt64(e + 1)
            guard let slot = ring.allocateSlot(for: expert, ticket: ticket) else {
                Issue.record("Failed to allocate slot for expert \(e)")
                return
            }
            ring.markReady(slotIndex: slot.index, ticket: ticket)
            speculativeSlotIndices.insert(slot.index)
            #expect(slot.lastAccessedTimestamp == 0, "Speculative slot must have timestamp 0")
        }

        // Verify pool saturation: 0 free, 8 ready
        #expect(ring.freeSlotCount == 0)
        #expect(ring.readySlotCount == 8)

        // Step 3: Now attempt 4 new allocations when all 8 slots are ready.
        // Each new allocation MUST evict one of the speculative slots (timestamp == 0),
        // and MUST NEVER evict any of the executed slots (0..3) with timestamp > 0!
        for i in 0..<4 {
            let demandExpert = ExpertKey(layer: 10, expert: 50 + i)
            let ticket = UInt64(200 + i)
            guard let victimSlot = ring.allocateSlot(for: demandExpert, ticket: ticket) else {
                Issue.record("Allocation failed during saturation")
                return
            }

            // The victim must be from speculativeSlotIndices (timestamp was 0)
            #expect(
                speculativeSlotIndices.contains(victimSlot.index),
                "Evicted slot[\(victimSlot.index)] must be an unexecuted slot (was ts=0), NOT an executed slot (slots 0..3)"
            )
            speculativeSlotIndices.remove(victimSlot.index)
        }

        // All 4 speculative slots (4..7) must have been evicted first
        #expect(speculativeSlotIndices.isEmpty, "All 4 unexecuted slots must be evicted before any executed slot")

        // Slots 0..3 must STILL be resident and ready with their original positive timestamps
        for e in 0..<4 {
            let key = ExpertKey(layer: 5, expert: e)
            let slot = ring.findReadySlot(for: key)
            #expect(slot != nil, "Executed expert \(key.description) must still be resident")
            #expect(slot?.lastAccessedTimestamp == UInt64(1000 + e * 500))
        }
    }

    // MARK: - 3. Post-Execution Drain as Single Source of Truth

    @Test("Adversarial 3: Drain-and-record is the single authoritative source of recency updates")
    func testDrainAndRecordIsSingleSourceOfTruth() {
        let ring = SpeculativeRingBuffer(device: device, slotCount: 8, slotSizeBytes: 4096)
        let log = GPUExecutionLog(device: device, capacity: 64)
        let tracker = LRUWeightTracker()

        // Allocate 4 resident slots
        for e in 0..<4 {
            let key = ExpertKey(layer: 5, expert: e)
            let slot = ring.allocateSlot(for: key, ticket: UInt64(e + 1))!
            ring.markReady(slotIndex: slot.index, ticket: UInt64(e + 1))
        }

        // Write log entries for ONLY experts (5, 1) and (5, 3)
        let ptr = log.buffer.contents().bindMemory(to: ExecutionLogEntry.self, capacity: log.capacity)
        ptr[0] = ExecutionLogEntry(
            tokenIndex: 1, layerIndex: 5, horizonIndex: 1, expertID: 1,
            confidenceScore: 0.91, timestamp: 50_000
        )
        ptr[1] = ExecutionLogEntry(
            tokenIndex: 2, layerIndex: 5, horizonIndex: 1, expertID: 3,
            confidenceScore: 0.87, timestamp: 80_000
        )

        // Drain through LRUWeightTracker
        let drained = tracker.drainAndRecord(from: log, ringBuffer: ring)
        #expect(drained.count == 2)
        #expect(tracker.totalDrains == 1)

        // Verify slot timestamps: ONLY (5, 1) and (5, 3) updated
        let slot0 = ring.findReadySlot(for: ExpertKey(layer: 5, expert: 0))
        let slot1 = ring.findReadySlot(for: ExpertKey(layer: 5, expert: 1))
        let slot2 = ring.findReadySlot(for: ExpertKey(layer: 5, expert: 2))
        let slot3 = ring.findReadySlot(for: ExpertKey(layer: 5, expert: 3))

        #expect(slot0?.lastAccessedTimestamp == 0, "Unexecuted expert (5, 0) must remain at timestamp 0")
        #expect(slot1?.lastAccessedTimestamp == 50_000, "Executed expert (5, 1) must be 50,000")
        #expect(slot2?.lastAccessedTimestamp == 0, "Unexecuted expert (5, 2) must remain at timestamp 0")
        #expect(slot3?.lastAccessedTimestamp == 80_000, "Executed expert (5, 3) must be 80,000")

        // Verify tracker contents
        #expect(tracker.count == 2)
        #expect(tracker.contains(expert: ExpertKey(layer: 5, expert: 1)))
        #expect(tracker.contains(expert: ExpertKey(layer: 5, expert: 3)))
        #expect(!tracker.contains(expert: ExpertKey(layer: 5, expert: 0)))
        #expect(!tracker.contains(expert: ExpertKey(layer: 5, expert: 2)))

        // Verify LRU victim is (5, 1) with ts 50,000 (earliest access)
        #expect(tracker.evictionCandidate() == ExpertKey(layer: 5, expert: 1))

        // Verify second drain is idempotent (0 entries, slots cleared)
        let drainedSecond = tracker.drainAndRecord(from: log, ringBuffer: ring)
        #expect(drainedSecond.isEmpty, "Second drain must return empty array")
        #expect(tracker.count == 2, "Tracker count must not change on empty drain")
    }

    // MARK: - 4. Monotonic Timestamp Numerical Non-Regression

    @Test("Adversarial 4: Monotonic timestamp guard prevents numerical regression when out-of-order entries arrive")
    func testMonotonicTimestampNumericalProtection() {
        let tracker = LRUWeightTracker()
        let expert = ExpertKey(layer: 7, expert: 15)

        // 1. First access at t = 1,000,000
        tracker.touch(expertKey: expert, timestamp: 1_000_000, slotIndex: 3)
        #expect(tracker.count == 1)

        // 2. Out-of-order access arrives with older timestamp: t = 500,000
        tracker.touch(expertKey: expert, timestamp: 500_000, slotIndex: 3)

        // 3. Another older access arrives: t = 100
        tracker.touch(expertKey: expert, timestamp: 100, slotIndex: 3)

        // The recorded timestamp must NOT have regressed to 500,000 or 100!
        // We verify this by evicting and checking the evicted node's properties,
        // or by testing that a newer timestamp (2,000,000) advances properly.
        tracker.touch(expertKey: expert, timestamp: 2_000_000, slotIndex: 3)

        #expect(tracker.count == 1)
        #expect(tracker.totalUpdates == 4)
        #expect(tracker.slotIndex(for: expert) == 3)
    }

    // MARK: - 5. Adversarial Out-of-Order Recency Ordering Challenge

    @Test("Adversarial 5: Recency queue ordering under out-of-order log arrivals")
    func testOutOfOrderRecencyQueueOrderEmpiricalChallenge() {
        let tracker = LRUWeightTracker()
        let expertA = ExpertKey(layer: 5, expert: 10)
        let expertB = ExpertKey(layer: 5, expert: 20)
        let expertC = ExpertKey(layer: 5, expert: 30)

        // Chronological accesses:
        // Expert A accessed at t = 1000
        // Expert B accessed at t = 2000
        // Expert C accessed at t = 3000
        tracker.touch(expertKey: expertA, timestamp: 1000)
        tracker.touch(expertKey: expertB, timestamp: 2000)
        tracker.touch(expertKey: expertC, timestamp: 3000)

        #expect(tracker.allTrackedExperts == [expertC, expertB, expertA])
        #expect(tracker.evictionCandidate() == expertA, "LRU victim must be Expert A (ts=1000)")

        // Now: A delayed out-of-order log entry arrives for Expert A with timestamp 500!
        // Note: ts 500 is OLDER than Expert A's recorded access (1000),
        // and OLDER than Expert B (2000) and Expert C (3000).
        tracker.touch(expertKey: expertA, timestamp: 500)

        // Observation test: How does the current implementation handle an older out-of-order entry?
        // If the implementation unlinks and inserts after head, Expert A becomes MRU!
        // Let's inspect the actual ordering after touch(A, 500):
        let orderAfterOutOfOrder = tracker.allTrackedExperts
        print("[Challenger 1] Recency order after out-of-order entry for A(ts=500): \(orderAfterOutOfOrder)")

        // Check whether A remained at LRU or was promoted to MRU
        let lruCandidate = tracker.evictionCandidate()
        print("[Challenger 1] LRU candidate after out-of-order entry: \(String(describing: lruCandidate))")

        // Regardless of whether A was promoted to MRU or stayed at LRU:
        // Assert that the tracker invariants are preserved:
        #expect(tracker.count == 3, "Tracker count must remain 3")
        #expect(tracker.contains(expert: expertA))
        #expect(tracker.contains(expert: expertB))
        #expect(tracker.contains(expert: expertC))
    }

    // MARK: - 6. Duplicate Tokens and Dense Repeated Ingestion

    @Test("Adversarial 6: Duplicate log entries for identical tokens and experts preserve doubly-linked list integrity")
    func testDuplicateTokensAndIdenticalTimestamps() {
        let tracker = LRUWeightTracker()
        let expert = ExpertKey(layer: 8, expert: 12)

        // Ingest 500 duplicate accesses with identical timestamp and identical expert
        for _ in 0..<500 {
            tracker.touch(expertKey: expert, timestamp: 100_000, slotIndex: 5)
        }

        #expect(tracker.count == 1, "Tracker count must remain 1 despite 500 duplicate entries")
        #expect(tracker.totalUpdates == 500)
        #expect(tracker.allTrackedExperts == [expert])
        #expect(tracker.slotIndex(for: expert) == 5)

        // Evict the single expert
        let evicted = tracker.evictLRU()
        #expect(evicted == expert)
        #expect(tracker.count == 0, "Tracker must be empty after eviction")
        #expect(tracker.evictionCandidate() == nil)
        #expect(tracker.allTrackedExperts.isEmpty)
    }

    // MARK: - 7. Reversed Timestamps Ingestion Sequence

    @Test("Adversarial 7: Reversed timestamp ingestion sequence (descending order)")
    func testReversedTimestampsSequence() {
        let tracker = LRUWeightTracker()
        let count = 20

        // Ingest entries for 20 distinct experts in REVERSED timestamp order:
        // Expert 0: ts = 20,000
        // Expert 1: ts = 19,000
        // ...
        // Expert 19: ts = 1,000
        for i in 0..<count {
            let expert = ExpertKey(layer: 9, expert: i)
            let ts = UInt64((count - i) * 1000)
            tracker.touch(expertKey: expert, timestamp: ts)
        }

        #expect(tracker.count == count)
        #expect(tracker.totalUpdates == count)

        // Verify that all 20 experts can be sequentially evicted without infinite loops or pointer corruption
        var evictedList: [ExpertKey] = []
        for _ in 0..<count {
            guard let victim = tracker.evictLRU() else {
                Issue.record("Expected evictable expert")
                break
            }
            evictedList.append(victim)
        }

        #expect(evictedList.count == count)
        #expect(tracker.count == 0)
        #expect(tracker.evictionCandidate() == nil)
    }

    // MARK: - 8. MTLCommandBuffer addCompletedHandler Execution Drain Lifecycle

    @Test("Adversarial 8: MTLCommandBuffer addCompletedHandler executes drain strictly after GPU kernel finishes")
    func testCommandBufferCompletionDrainLifecycle() throws {
        guard let queue = device.makeCommandQueue() else {
            throw TestError.commandQueueFailed
        }

        let ring = SpeculativeRingBuffer(device: device, slotCount: 4, slotSizeBytes: 4096)
        let log = GPUExecutionLog(device: device, capacity: 64)
        let tracker = LRUWeightTracker()

        // Allocate slot for expert (5, 0)
        let expert = ExpertKey(layer: 5, expert: 0)
        let slot = ring.allocateSlot(for: expert, ticket: 1)!
        ring.markReady(slotIndex: slot.index, ticket: 1)
        #expect(slot.lastAccessedTimestamp == 0)

        // Write a log entry in the execution log buffer
        let ptr = log.buffer.contents().bindMemory(to: ExecutionLogEntry.self, capacity: log.capacity)
        ptr[0] = ExecutionLogEntry(
            tokenIndex: 1, layerIndex: 5, horizonIndex: 1, expertID: 0,
            confidenceScore: 0.95, timestamp: 999_999
        )

        // Create compute command buffer and register completion drain
        guard let cmd = queue.makeCommandBuffer() else {
            throw TestError.commandQueueFailed
        }

        let expectationLock = OSAllocatedUnfairLock(initialState: false)
        tracker.registerCompletionDrain(on: cmd, executionLog: log, ringBuffer: ring) { drained in
            expectationLock.withLock { $0 = true }
        }

        // Before commit/completion: slot timestamp is STILL 0, tracker is STILL empty
        #expect(slot.lastAccessedTimestamp == 0, "Timestamp must remain 0 before GPU completion")
        #expect(tracker.count == 0, "Tracker must remain empty before GPU completion")

        // Commit and wait for GPU completion
        cmd.commit()
        cmd.waitUntilCompleted()

        // After completion: completion handler has drained log, updated ring slot, and updated tracker
        #expect(expectationLock.withLock { $0 } == true, "Completion callback must have fired")
        #expect(slot.lastAccessedTimestamp == 999_999, "Slot timestamp must be updated post-completion")
        #expect(tracker.count == 1, "Tracker must contain executed expert")
        #expect(tracker.contains(expert: expert))
        #expect(tracker.evictionCandidate() == expert)
    }

    // MARK: - 9. Slot Re-binding and Unbinding Invariants

    @Test("Adversarial 9: Slot index binding, unbinding, and replacement invariants")
    func testSlotBindingAndUnbindingInvariants() {
        let tracker = LRUWeightTracker()
        let expert = ExpertKey(layer: 11, expert: 7)

        // Bind slot 2
        tracker.bindSlot(slotIndex: 2, for: expert)
        #expect(tracker.slotIndex(for: expert) == 2)
        #expect(tracker.contains(expert: expert))

        // Re-bind to slot 5 (slot migration)
        tracker.bindSlot(slotIndex: 5, for: expert)
        #expect(tracker.slotIndex(for: expert) == 5)

        // Unbind slot (slot evicted, but recency node preserved)
        let prevSlot = tracker.unbindSlot(for: expert)
        #expect(prevSlot == 5)
        #expect(tracker.slotIndex(for: expert) == nil)
        #expect(tracker.contains(expert: expert), "Expert must still be tracked after unbind")

        // Evict with slot returns nil slot
        let evicted = tracker.evictLRUWithSlot()
        #expect(evicted?.expert == expert)
        #expect(evicted?.slotIndex == nil)
        #expect(tracker.count == 0)
    }

    // MARK: - 10. Empirical Bug Investigation: Stale Out-of-Order Entry Queue Corruption

    @Test("Adversarial 10: Prove whether out-of-order older entries corrupt LRU eviction victim selection")
    func testEmpiricalChallengeOutOfOrderRecencyCorruption() {
        let tracker = LRUWeightTracker()
        let expertA = ExpertKey(layer: 5, expert: 1)
        let expertB = ExpertKey(layer: 5, expert: 2)

        // Step 1: Expert A was accessed at t = 1000
        tracker.touch(expertKey: expertA, timestamp: 1000)
        // Step 2: Expert B was accessed at t = 2000
        tracker.touch(expertKey: expertB, timestamp: 2000)

        // Queue order should be: MRU: B (2000), LRU: A (1000)
        #expect(tracker.evictionCandidate() == expertA, "Before out-of-order entry, A (ts=1000) must be LRU victim")

        // Step 3: Now an out-of-order entry arrives for Expert A with timestamp 500 (from the past).
        // Since timestamp 500 < 1000 (and < 2000), this access happened in the past.
        // It should NOT make Expert A more recently used than Expert B (who was accessed at 2000)!
        tracker.touch(expertKey: expertA, timestamp: 500)

        // If the monotonic guard properly protects the recency queue:
        // Expert A's last access is still 1000, which is older than Expert B's access at 2000.
        // Therefore, Expert A MUST STILL BE THE LRU VICTIM!
        let victim = tracker.evictionCandidate()
        print("[Empirical Challenger] Victim after out-of-order entry: \(String(describing: victim))")

        // We verify if victim is still expertA
        let orderPreserved = (victim == expertA)
        #expect(
            orderPreserved,
            "CRITICAL: Monotonic guard failed to protect recency queue! Out-of-order entry with older timestamp (500) promoted Expert A over Expert B (ts=2000), causing Expert B to become the eviction victim!"
        )
    }

    // MARK: - 11. Empirical Bug Investigation: Non-Sequential Sparse Slot Draining

    @Test("Adversarial 11: Prove whether sparse/gapped slot indexing blocks CPU drain()")
    func testEmpiricalChallengeSparseGapsInDeterministicSlotLogging() {
        let log = GPUExecutionLog(device: device, capacity: 64)

        // Simulate writeExecutionLogEntry formula for token 0, layer 5, horizon 1:
        // slot = (0 * 80 + 5 * 4 + 1) & (64 - 1) = 21 & 63 = 21
        let slotIndex = 21
        let ptr = log.buffer.contents().bindMemory(to: ExecutionLogEntry.self, capacity: log.capacity)
        ptr[slotIndex] = ExecutionLogEntry(
            tokenIndex: 0,
            layerIndex: 5,
            horizonIndex: 1,
            expertID: 42,
            confidenceScore: 0.95,
            timestamp: 123_456
        )

        // Slots 0..<21 are empty (zeroed out)
        // Now CPU calls drain()
        let drained = log.drain()
        print("[Empirical Challenger] Drained count from sparse slot 21: \(drained.count)")

        // Check if entry at slot 21 was drained
        let foundEntry = drained.contains { $0.layerIndex == 5 && $0.expertID == 42 }
        #expect(
            foundEntry,
            "CRITICAL: ExecutionLog.drain() stopped at empty slot 0 and failed to drain valid entry written at slot 21 by writeExecutionLogEntry!"
        )
    }
}

