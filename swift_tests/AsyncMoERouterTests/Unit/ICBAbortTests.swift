import Testing
import Metal
import Foundation
@testable import AsyncMoERouter

/// Tests for M4 (AbortController + ICBController) correctness and safety invariants.
@Suite("ICB Abort & Execution Safety Tests")
struct ICBAbortTests {
    let device: any MTLDevice

    init() throws {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            throw TestError.metalUnavailable
        }
        self.device = dev
    }

    // MARK: - AbortController Tests

    @Test("Abort flag initializes to false")
    func testAbortFlagInitiallyFalse() {
        let ctrl = AbortController(device: device)
        #expect(!ctrl.isAborted)
    }

    @Test("Abort flag reset clears value")
    func testAbortFlagReset() {
        let ctrl = AbortController(device: device)
        // Manually set flag via pointer (simulating GPU kernel write)
        let ptr = ctrl.buffer.contents().bindMemory(to: UInt8.self, capacity: 16)
        ptr[0] = 1
        #expect(ctrl.isAborted)
        ctrl.reset()
        #expect(!ctrl.isAborted)
    }

    @Test("Abort flag buffer is allocated with storageModeShared")
    func testAbortFlagStorageMode() {
        let ctrl = AbortController(device: device)
        // storageModeShared = 0 in MTLStorageMode
        #expect(ctrl.buffer.storageMode == .shared)
    }

    @Test("Abort flag buffer is exactly 16 bytes")
    func testAbortFlagBufferSize() {
        let ctrl = AbortController(device: device)
        #expect(ctrl.buffer.length == 16)
    }

    @Test("MSL gating kernel source contains abort_flag check")
    func testICBGatingKernelContainsAbortCheck() {
        let src = AbortController.icbGatingKernelSource
        #expect(src.contains("abort_flag"))
        #expect(src.contains("uint3(0, 0, 0)"))
        #expect(src.contains("concurrent_dispatch_threads"))
    }

    @Test("MSL cascading layer kernel checks abort_flag before residual write")
    func testCascadingLayerKernelAbortFirst() {
        let src = AbortController.cascadingLayerKernelSource
        // The abort check must appear before any write to residual_x
        let abortIdx = src.range(of: "if (*abort_flag != 0) return;")?.lowerBound
        let writeIdx = src.range(of: "residual_x[tid]")?.lowerBound
        #expect(abortIdx != nil && writeIdx != nil)
        if let ai = abortIdx, let wi = writeIdx {
            #expect(ai < wi)
        }
    }

    @Test("MSL kernel source does NOT use .untracked buffers")
    func testNoUntrackedBuffersInShaders() {
        // Hazard tracking invariant: no shader or setup should mention .untracked
        // (This test catches accidental regressions in shader source strings)
        let gating = AbortController.icbGatingKernelSource
        let cascading = AbortController.cascadingLayerKernelSource
        #expect(!gating.contains("untracked"))
        #expect(!cascading.contains("untracked"))
    }

    @Test("MSL execution log kernel source uses deterministic slot assignment")
    func testExecutionLogKernelDeterministicSlots() {
        let src = GPUExecutionLog.mslKernelSource
        #expect(src.contains("writeExecutionLogEntry"))
        #expect(src.contains("logCapacity"))
        // Must not use atomic operations for log writes
        #expect(!src.contains("atomic_store"))
        #expect(!src.contains("atomic_fetch_add"))
    }

    // MARK: - ICBController Tests

    @Test("ICB controller initializes successfully")
    func testICBControllerInit() {
        let ctrl = ICBController(device: device, maxCommands: 16)
        #expect(ctrl.icb != nil)
        #expect(ctrl.icbArgumentBuffer.length >= 8)
    }

    @Test("ICB controller resets without crash")
    func testICBControllerReset() {
        let ctrl = ICBController(device: device, maxCommands: 8)
        ctrl.reset()  // Should not throw or crash
        ctrl.reset()  // Idempotent
    }

    @Test("ICB controller encodes dispatch within bounds")
    func testICBControllerEncodeDispatch() {
        let ctrl = ICBController(device: device, maxCommands: 8)
        // Should not crash for valid indices
        ctrl.encodeExpertDispatch(
            at: 0,
            gridSize: MTLSize(width: 64, height: 1, depth: 1),
            threadgroupSize: MTLSize(width: 32, height: 1, depth: 1)
        )
        ctrl.encodeExpertDispatch(
            at: 7,
            gridSize: MTLSize(width: 128, height: 1, depth: 1),
            threadgroupSize: MTLSize(width: 64, height: 1, depth: 1)
        )
    }

    @Test("ICB controller argument buffer uses shared storage")
    func testICBArgumentBufferStorageMode() {
        let ctrl = ICBController(device: device, maxCommands: 16)
        #expect(ctrl.icbArgumentBuffer.storageMode == .shared)
    }

    // MARK: - Execution Log Entry Layout Tests

    @Test("ExecutionLogEntry is exactly 32 bytes")
    func testExecutionLogEntrySize() {
        #expect(MemoryLayout<ExecutionLogEntry>.stride == 32)
    }

    @Test("ExecutionLogEntry field offsets are correct")
    func testExecutionLogEntryOffsets() {
        // tokenIndex at byte 0, layerIndex at byte 4, etc.
        #expect(MemoryLayout<ExecutionLogEntry>.offset(of: \.tokenIndex) == 0)
        #expect(MemoryLayout<ExecutionLogEntry>.offset(of: \.layerIndex) == 4)
        #expect(MemoryLayout<ExecutionLogEntry>.offset(of: \.horizonIndex) == 6)
        #expect(MemoryLayout<ExecutionLogEntry>.offset(of: \.expertID) == 8)
        #expect(MemoryLayout<ExecutionLogEntry>.offset(of: \.confidenceScore) == 12)
        #expect(MemoryLayout<ExecutionLogEntry>.offset(of: \.timestamp) == 16)
        #expect(MemoryLayout<ExecutionLogEntry>.offset(of: \.reserved) == 24)
    }
}
