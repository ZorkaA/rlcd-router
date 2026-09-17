import Testing
import Metal
import Foundation
@testable import AsyncMoERouter

/// Tier 1 E2E tests: Full pipeline initialization and lifecycle validation.
@Suite("E2E Tier 1: Pipeline Initialization & Lifecycle")
struct Tier1_FeatureTests {
    let device: any MTLDevice

    init() throws {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            throw TestError.metalUnavailable
        }
        self.device = dev
    }

    @Test("Pipeline initializes all subsystems without error")
    func testPipelineFullInit() throws {
        let engine = try FastIOEngine(device: device)
        let pipeline = AsyncMoEPipeline(
            device: device,
            fastIO: engine,
            archConfig: .synthetic,
            budgetConfig: .default
        )
        // Verify all subsystems are initialized
        #expect(pipeline.ringBuffer.slotCount == 16)
        #expect(pipeline.fallbackPool.maxCapacityBytes == 500 * 1024 * 1024)
        #expect(pipeline.executionLog.capacity == 4096)
        #expect(!pipeline.abortController.isAborted)
        #expect(pipeline.icbController.icb != nil)
    }

    @Test("Pipeline step lifecycle: begin → end does not crash")
    func testPipelineStepLifecycle() async throws {
        let engine = try FastIOEngine(device: device)
        let pipeline = AsyncMoEPipeline(
            device: device, fastIO: engine,
            archConfig: .synthetic, budgetConfig: .default
        )
        pipeline.beginStep()
        #expect(!pipeline.abortController.isAborted)
        await pipeline.endStep()
        #expect(pipeline.totalTokensProcessed == 1)
    }

    @Test("Pipeline abort flag resets between steps")
    func testAbortFlagResetsPerStep() async throws {
        let engine = try FastIOEngine(device: device)
        let pipeline = AsyncMoEPipeline(
            device: device, fastIO: engine,
            archConfig: .synthetic, budgetConfig: .default
        )
        // Simulate GPU setting abort flag
        let ptr = pipeline.abortController.buffer.contents()
            .bindMemory(to: UInt8.self, capacity: 16)
        pipeline.beginStep()
        ptr[0] = 1  // Simulate abort
        #expect(pipeline.abortController.isAborted)
        // Next step resets flag
        pipeline.beginStep()
        #expect(!pipeline.abortController.isAborted)
    }

    @Test("Pipeline expert cache check returns false when empty")
    func testPipelineCacheCheckEmpty() throws {
        let engine = try FastIOEngine(device: device)
        let pipeline = AsyncMoEPipeline(
            device: device, fastIO: engine,
            archConfig: .synthetic, budgetConfig: .default
        )
        let expert = ExpertKey(layer: 5, expert: 0)
        #expect(!pipeline.isExpertCached(expert))
    }

    @Test("Pipeline diagnostics summary is non-empty")
    func testPipelineDiagnostics() throws {
        let engine = try FastIOEngine(device: device)
        let pipeline = AsyncMoEPipeline(
            device: device, fastIO: engine,
            archConfig: .synthetic, budgetConfig: .default
        )
        let summary = pipeline.diagnosticsSummary
        #expect(!summary.isEmpty)
        #expect(summary.contains("Pipeline Diagnostics"))
    }

    @Test("Memory budget stays within conservative ceiling for synthetic config")
    func testMemoryBudgetConservative() throws {
        let engine = try FastIOEngine(device: device)
        let pipeline = AsyncMoEPipeline(
            device: device, fastIO: engine,
            archConfig: .synthetic, budgetConfig: .default
        )
        let budget = MemoryBudgetConfig.default
        let totalBytes = budget.totalConservativeBudgetBytes(for: .synthetic)
        // For synthetic: 16 × 24KB + 500MB + 200MB + 128KB ≈ 701MB — well under 8GB
        #expect(totalBytes < 2_000_000_000)
        // Must not exceed available RAM (guard against full-model accidental config)
        _ = pipeline.diagnosticsSummary  // Ensure pipeline is alive
    }

    @Test("Execution log post-step drain updates LRU tracker")
    func testEndStepDrainsLog() async throws {
        let engine = try FastIOEngine(device: device)
        let pipeline = AsyncMoEPipeline(
            device: device, fastIO: engine,
            archConfig: .synthetic, budgetConfig: .default
        )
        // Manually write an execution log entry (simulating GPU write)
        let ptr = pipeline.executionLog.buffer.contents()
            .bindMemory(to: ExecutionLogEntry.self, capacity: 4096)
        ptr[0] = ExecutionLogEntry(
            tokenIndex: 1, layerIndex: 5, horizonIndex: 1,
            expertID: 7, confidenceScore: 0.9, timestamp: 12345
        )
        pipeline.beginStep()
        await pipeline.endStep()

        // LRU tracker should now contain the expert
        let expert = ExpertKey(layer: 5, expert: 7)
        #expect(pipeline.lruTracker.contains(expert: expert))
        #expect(pipeline.totalTokensProcessed == 1)
    }

    @Test("Recalibration pre-empted by user request")
    func testRecalibrationPreemptedByRequest() async throws {
        let engine = try FastIOEngine(device: device)
        let pipeline = AsyncMoEPipeline(
            device: device, fastIO: engine,
            archConfig: .synthetic, budgetConfig: .default
        )
        await pipeline.onUserRequestReceived()
        // Trying to run recalibration should return nil (cancelled)
        let result = await pipeline.recalibrationActor.runPassIfReady()
        #expect(result == nil)
    }
}
