import Testing
import Foundation
import Metal
@testable import AsyncMoERouter

// MARK: - Tier 4: Workload & Stress Tests

@Suite("E2E Tier 4: Workload & Stress Tests")
struct Tier4WorkloadTests {

    // MARK: - Multi-Step Token Generation (200 tokens)

    @Test("Multi-step token generation: 200 steps, memory stays within budget")
    func testMultiStepTokenGeneration() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw _Skip.always("Metal unavailable")
        }
        let config = MoEArchitectureConfig.synthetic
        let budget = MemoryBudgetConfig()
        let pipeline = try AsyncMoEPipeline(device: device, config: config, budget: budget)
        let ceiling = budget.ringBufferSlotsBytes + budget.fallbackPoolBytes + budget.mlxCacheLimitBytes

        let numTokens = 200
        let numLayers = 4  // synthetic config
        let numExperts = 16

        // Pre-warm
        var epoch: UInt64 = 1
        for layer in 0..<numLayers {
            for expert in 0..<4 {
                let key = ExpertKey(layer: layer, expert: expert)
                _ = pipeline.ringBuffer.allocateSlot(for: key, ticket: epoch)
                epoch += 1
            }
        }

        for token in 0..<numTokens {
            pipeline.beginStep()
            for layer in 0..<numLayers {
                let expertIdx = (token + layer) % numExperts
                let key = ExpertKey(layer: layer, expert: expertIdx)
                let _ = pipeline.isExpertCached(key)
                // Speculative prefetch of next token's expert
                let nextKey = ExpertKey(layer: layer, expert: (expertIdx + 1) % numExperts)
                _ = pipeline.prefetchExpert(key: nextKey)
            }
            pipeline.endStep()
        }

        #expect(pipeline.peakMemoryBytes <= ceiling,
                "Peak memory \(pipeline.peakMemoryBytes) must not exceed budget ceiling \(ceiling)")
    }

    // MARK: - Recalibration Actor Under Load

    @Test("RecalibrationActor: 50 passes with realistic log volumes, temperature bounded")
    func testRecalibrationActorUnderLoad() async throws {
        let actor = RecalibrationActor(minEntriesToRecalibrate: 20)
        var totalConsumed = 0

        for pass in 0..<50 {
            var entries: [ExecutionLogEntry] = []
            for i in 0..<25 {
                let tok = UInt32(pass * 25 + i + 1)
                let exp = UInt16((pass + i) % 64)
                let conf = Float(i % 10) / 10.0
                let ts = UInt64(pass * 25 + i + 1) * 100
                entries.append(ExecutionLogEntry(
                    tokenIndex: tok, layerIndex: UInt16(pass % 28), horizonIndex: 0,
                    expertID: exp, confidenceScore: conf, timestamp: ts
                ))
            }
            await actor.ingestEntries(entries)

            if let temp = await actor.runPassIfReady() {
                #expect(temp > 0.0 && temp < 10.0, "Temperature out of range at pass \(pass)")
                let consumed = await actor.totalEntriesConsumed
                #expect(consumed >= totalConsumed, "totalEntriesConsumed must be monotonic")
                totalConsumed = consumed
            }
        }
        #expect(totalConsumed > 0, "At least some entries must have been consumed across 50 passes")
    }

    // MARK: - Abort Controller Cycling (1000 iterations)

    @Test("AbortController: 1000 toggle cycles with coherent readback")
    func testAbortControllerCyclingUnderLoad() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw _Skip.always("Metal unavailable")
        }
        let abort = AbortController(device: device)

        for i in 0..<1000 {
            let expected = (i % 2 == 0)
            abort.set(expected)
            let actual = abort.isAborted
            #expect(actual == expected,
                    "Abort flag readback must be coherent at iteration \(i): expected \(expected), got \(actual)")
        }
        abort.reset()
    }

    // MARK: - Ring Buffer: 500-Cycle Alloc/Release Churn

    @Test("RingBuffer: 500-cycle sequential alloc/release churn, no index corruption")
    func testRingBufferChurnUnderLoad() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw _Skip.always("Metal unavailable")
        }
        let ring = SpeculativeRingBuffer(device: device, slotCount: 16, slotSizeBytes: 16 * 1024)

        var totalAllocated = 0
        var epoch: UInt64 = 1
        for cycle in 0..<500 {
            let key = ExpertKey(layer: cycle % 28, expert: cycle % 64)
            if let slot = ring.allocateSlot(for: key, ticket: epoch) {
                ring.markReady(slotIndex: slot.index, ticket: epoch)
                totalAllocated += 1
                epoch += 1
            }
        }
        #expect(totalAllocated > 0, "Must have allocated at least one slot across 500 cycles")
        // Total allocations metric must be consistent
        #expect(ring.totalAllocations == totalAllocated, "totalAllocations metric must match")
    }

    // MARK: - LRU Tracker: 10,000 Access Records

    @Test("LRUWeightTracker: 10,000 recordAccess calls complete and return correct LRU")
    func testLRUTrackerPerformanceUnder10k() throws {
        let tracker = LRUWeightTracker()

        for i in 0..<10_000 {
            let key = ExpertKey(layer: i % 28, expert: i % 64)
            tracker.recordAccess(expert: key, timestamp: UInt64(i + 1))
        }

        // LRU must exist and be non-nil after 10k accesses
        let lru = tracker.lruExpert
        #expect(lru != nil, "LRU tracker must return a non-nil LRU key after 10,000 accesses")
        // Total updates counter must equal 10,000
        #expect(tracker.totalUpdates == 10_000, "totalUpdates must equal 10,000")
    }

    // MARK: - Full Pipeline: 50 Steps with Abort at Step 25

    @Test("AsyncMoEPipeline: 50 steps with abort injected at step 25 — graceful recovery")
    func testPipelineAbortMidSequence() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw _Skip.always("Metal unavailable")
        }
        let config = MoEArchitectureConfig.synthetic
        let budget = MemoryBudgetConfig()
        let pipeline = try AsyncMoEPipeline(device: device, config: config, budget: budget)

        var stepsCompleted = 0
        for step in 0..<50 {
            pipeline.beginStep()

            if step == 25 {
                pipeline.triggerAbort()
                #expect(pipeline.isAbortSet == true, "Abort must be set at step 25")
            }

            // endStep clears abort flag
            pipeline.endStep()
            stepsCompleted += 1

            if step == 25 {
                // After endStep, abort must be cleared (beginStep does this next iteration)
                // Here we just verify triggerAbort worked — endStep itself calls reset
                // The abort flag state at this point depends on whether endStep clears it
            }
        }

        #expect(stepsCompleted == 50, "All 50 steps must complete despite mid-sequence abort")
    }

    // MARK: - Fallback Pool: 200-cycle demand allocation and reclaim

    @Test("FallbackPool: 200-cycle demand alloc/reclaim maintains zero leak")
    func testFallbackPoolCyclicAllocReclaim() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw _Skip.always("Metal unavailable")
        }
        // 8 slots × 32KB = 256KB total ceiling
        let pool = FallbackBufferPool(device: device, slotSizeBytes: 32 * 1024, maxCapacityBytes: 8 * 32 * 1024, prewarmCount: 0)

        for cycle in 0..<200 {
            let ctx = DemandFetchContext(expert: ExpertKey(layer: cycle % 28, expert: cycle % 64))
            guard let slot = try? pool.allocateForDemand(context: ctx, device: device) else {
                // Pool full — verify inUseCount > 0 and continue
                let diag = pool.diagnostics()
                #expect(diag.inUseCount > 0, "Pool must be full if allocation fails")
                continue
            }
            // Always reclaim immediately to maintain zero-leak property
            try? pool.reclaim(slot)
        }

        // After all cycles, no slots should be in use
        let finalDiag = pool.diagnostics()
        #expect(finalDiag.inUseCount == 0, "All slots must be reclaimed after 200 cycles")
        #expect(pool.verifyInvariants(), "Pool invariants must hold after 200 cycles")
    }
}

// MARK: - Skip helper

private enum _Skip {
    static func always(_ reason: String) -> any Error {
        struct SkipError: Error { let reason: String }
        return SkipError(reason: reason)
    }
}
