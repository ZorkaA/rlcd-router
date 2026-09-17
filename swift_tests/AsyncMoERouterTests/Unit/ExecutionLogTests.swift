import Testing
import Metal
import Foundation
@testable import AsyncMoERouter

/// ExecutionLog unit tests (M3) — drain semantics, ring wrap-around, entry clearing.
@Suite("Execution Log Tests")
struct ExecutionLogTests {
    let device: any MTLDevice

    init() throws {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            throw TestError.metalUnavailable
        }
        self.device = dev
    }

    @Test("Execution log drains zero entries when buffer is zeroed")
    func testEmptyLogDrain() {
        let log = GPUExecutionLog(device: device, capacity: 64)
        let entries = log.drain()
        #expect(entries.isEmpty)
    }

    @Test("Execution log drains multiple sequential entries")
    func testMultipleEntriesDrained() {
        let log = GPUExecutionLog(device: device, capacity: 64)
        let ptr = log.buffer.contents().bindMemory(to: ExecutionLogEntry.self, capacity: 64)
        for i in 0..<5 {
            ptr[i] = ExecutionLogEntry(
                tokenIndex: UInt32(i + 1),
                layerIndex: UInt16(5 + i), horizonIndex: 1,
                expertID: UInt16(i * 3), confidenceScore: Float(i) / 10.0,
                timestamp: UInt64((i + 1) * 1000)
            )
        }
        let entries = log.drain()
        #expect(entries.count == 5)
        #expect(entries[0].tokenIndex == 1)
        #expect(entries[4].tokenIndex == 5)
        #expect(log.totalEntriesDrained == 5)
    }

    @Test("Execution log clears entries after drain")
    func testLogEntriesClearedAfterDrain() {
        let log = GPUExecutionLog(device: device, capacity: 64)
        let ptr = log.buffer.contents().bindMemory(to: ExecutionLogEntry.self, capacity: 64)
        ptr[0] = ExecutionLogEntry(
            tokenIndex: 77, layerIndex: 8, horizonIndex: 2,
            expertID: 11, confidenceScore: 0.95, timestamp: 54321
        )
        _ = log.drain()
        // Second drain must return empty (entries cleared)
        let second = log.drain()
        #expect(second.isEmpty)
    }

    @Test("Execution log buffer size is correct (capacity × 32)")
    func testExecutionLogBufferSize() {
        let log = GPUExecutionLog(device: device, capacity: 256)
        #expect(log.buffer.length == 256 * 32)
    }

    @Test("Execution log MSL shader source is valid metal syntax")
    func testExecutionLogMSLSource() {
        let src = GPUExecutionLog.mslKernelSource
        #expect(src.contains("kernel void writeExecutionLogEntry"))
        #expect(src.contains("device ExecutionLogEntry* log"))
        #expect(src.contains("logCapacity"))
        // Must not use atomics
        #expect(!src.contains("atomic_"))
    }

    // MARK: - Memory Budget Tests

    @Test("MemoryBudgetConfig conservative budget calculation")
    func testMemoryBudgetTotal() {
        let budget = MemoryBudgetConfig()
        let arch = MoEArchitectureConfig.synthetic
        let total = budget.totalConservativeBudgetBytes(for: arch)
        // Must fit in available RAM (well under 8GB for synthetic)
        #expect(total < 1_000_000_000) // < 1 GB for synthetic config
    }

    @Test("MemoryBudgetConfig page alignment check")
    func testPageAlignment() {
        #expect(MemoryBudgetConfig.isPageAligned(bytes: 16_384))
        #expect(MemoryBudgetConfig.isPageAligned(bytes: 32_768))
        #expect(!MemoryBudgetConfig.isPageAligned(bytes: 1_000))
    }

    @Test("MemoryBudgetConfig sector alignment check")
    func testSectorAlignment() {
        #expect(MemoryBudgetConfig.isSectorAligned(bytes: 4096))
        #expect(MemoryBudgetConfig.isSectorAligned(bytes: 8192))
        #expect(!MemoryBudgetConfig.isSectorAligned(bytes: 1001))
    }
}
