import Testing
import Metal
import Foundation
@testable import AsyncMoERouter

/// Tests for M5 (MLXCacheController + RecalibrationActor).
@Suite("Recalibration & MLX Cache Tests")
struct RecalibrationTests {
    let device: any MTLDevice

    init() throws {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            throw TestError.metalUnavailable
        }
        self.device = dev
    }

    // MARK: - MLXCacheController Tests

    @Test("MLX cache controller reports correct limit")
    func testMLXCacheLimit() {
        #expect(MLXCacheController.cacheLimitBytes == 200 * 1024 * 1024)
    }

    @Test("MLX cache controller applyCacheLimit does not crash")
    func testMLXApplyCacheLimitNoCrash() {
        // Even without MLX linked, this must be a no-op, not a crash
        MLXCacheController.applyCacheLimit()
    }

    // MARK: - RecalibrationActor Tests

    @Test("Recalibration actor skips pass with insufficient data")
    func testRecalibrationSkipsInsufficient() async {
        let actor = RecalibrationActor(minEntriesToRecalibrate: 256)
        let entries: [ExecutionLogEntry] = (0..<10).map { i in
            ExecutionLogEntry(
                tokenIndex: UInt32(i), layerIndex: 5, horizonIndex: 1,
                expertID: 3, confidenceScore: 0.8, timestamp: UInt64(i * 100)
            )
        }
        await actor.ingestEntries(entries)
        let result = await actor.runPassIfReady()
        #expect(result == nil)  // Insufficient data
    }

    @Test("Recalibration actor runs pass with sufficient data")
    func testRecalibrationRunsWithData() async {
        let actor = RecalibrationActor(minEntriesToRecalibrate: 10)
        let entries: [ExecutionLogEntry] = (0..<20).map { i in
            let tok = UInt32(i + 1)
            let exp = UInt16(i % 16)
            let conf = Float(i % 10) / 10.0
            let ts = UInt64((i + 1) * 100)
            return ExecutionLogEntry(tokenIndex: tok, layerIndex: 5, horizonIndex: 1,
                                     expertID: exp, confidenceScore: conf, timestamp: ts)
        }
        await actor.ingestEntries(entries)
        let result = await actor.runPassIfReady()
        #expect(result != nil)
        if let t = result {
            #expect(t >= 0.5 && t <= 5.0)  // Temperature bounds from actor
        }
    }

    @Test("Recalibration actor cancels in-progress pass")
    func testRecalibrationCancels() async {
        let actor = RecalibrationActor(minEntriesToRecalibrate: 10)
        let entries: [ExecutionLogEntry] = (0..<20).map { i in
            ExecutionLogEntry(
                tokenIndex: UInt32(i + 1), layerIndex: 5, horizonIndex: 1,
                expertID: 3, confidenceScore: 0.7, timestamp: UInt64((i + 1) * 50)
            )
        }
        await actor.ingestEntries(entries)
        await actor.cancelCurrentPass()
        let result = await actor.runPassIfReady()
        #expect(result == nil)  // Cancelled
    }

    @Test("Recalibration actor temperature stays within bounds")
    func testRecalibrationTemperatureBounds() async {
        let actor = RecalibrationActor(minEntriesToRecalibrate: 5)
        // Feed extremely high-confidence entries to push temperature low
        let entries: [ExecutionLogEntry] = (0..<20).map { i in
            ExecutionLogEntry(
                tokenIndex: UInt32(i + 1), layerIndex: 5, horizonIndex: 1,
                expertID: 0, confidenceScore: 0.99, timestamp: UInt64((i + 1) * 10)
            )
        }
        await actor.ingestEntries(entries)
        for _ in 0..<10 {
            await actor.resetCancellation()
            _ = await actor.runPassIfReady()
        }
        let t = await actor.currentTemperature
        #expect(t >= 0.5 && t <= 5.0)
    }

    @Test("Recalibration actor counts passes and entries")
    func testRecalibrationMetrics() async {
        let actor = RecalibrationActor(minEntriesToRecalibrate: 10)
        let entries: [ExecutionLogEntry] = (0..<20).map { i in
            ExecutionLogEntry(
                tokenIndex: UInt32(i + 1), layerIndex: 5, horizonIndex: 1,
                expertID: 2, confidenceScore: 0.6, timestamp: UInt64((i + 1) * 100)
            )
        }
        await actor.ingestEntries(entries)
        let passes0 = await actor.totalPassesCompleted
        _ = await actor.runPassIfReady()
        let passes1 = await actor.totalPassesCompleted
        #expect(passes1 == passes0 + 1)
        let consumed = await actor.totalEntriesConsumed
        #expect(consumed >= 10)
    }
}
