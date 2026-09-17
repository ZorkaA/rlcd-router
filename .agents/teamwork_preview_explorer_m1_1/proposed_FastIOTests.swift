import XCTest
import Metal
@testable import AsyncMoERouter

final class FastIOTests: XCTestCase {
    var metalContext: MetalContext!
    var fastIOEngine: FastIOEngine!

    override func setUpWithError() throws {
        try super.setUpWithError()
        metalContext = try MetalContext()
        fastIOEngine = try FastIOEngine(device: metalContext.device)
    }

    // MARK: - Dimension & Layout Tests

    func testArchitectureConfigSizing() {
        let qwen = MoEArchitectureConfig.qwen15MoEA27B
        XCTAssertEqual(qwen.hiddenSize, 2048)
        XCTAssertEqual(qwen.intermediateSize, 1408)
        XCTAssertEqual(qwen.numExperts, 60)
        XCTAssertEqual(qwen.numActiveExperts, 4)
        XCTAssertEqual(qwen.numDeepLayers, 20)
        XCTAssertEqual(qwen.bytesPerElement, 2)

        // 3 * 2048 * 1408 = 8,650,752 elements
        XCTAssertEqual(qwen.expertElements, 8_650_752)
        // 8,650,752 * 2 = 17,301,504 bytes
        XCTAssertEqual(qwen.expertSizeBytes, 17_301_504)
        // Check Apple Silicon 16KB page alignment (17,301,504 / 16,384 == 1056 exactly)
        XCTAssertTrue(MemoryBudgetConfig.isPageAligned(bytes: qwen.expertSizeBytes))
        XCTAssertEqual(qwen.expertSizeBytes % 16_384, 0)
        XCTAssertEqual(qwen.expertSizeBytes / 16_384, 1056)
    }

    func testSyntheticConfigSizing() {
        let synthetic = MoEArchitectureConfig.synthetic
        XCTAssertEqual(synthetic.hiddenSize, 64)
        XCTAssertEqual(synthetic.intermediateSize, 64)
        // 3 * 64 * 64 = 12,288 elements
        XCTAssertEqual(synthetic.expertElements, 12_288)
        // 12,288 * 2 = 24,576 bytes
        XCTAssertEqual(synthetic.expertSizeBytes, 24_576)
        XCTAssertTrue(MemoryBudgetConfig.isSectorAligned(bytes: synthetic.expertSizeBytes))
    }

    func testExecutionLogEntryLayout() {
        // Must be exactly 32 bytes and 8-byte aligned for lock-free GPU DMA mapping
        XCTAssertEqual(MemoryLayout<ExecutionLogEntry>.size, 32)
        XCTAssertEqual(MemoryLayout<ExecutionLogEntry>.stride, 32)
        XCTAssertEqual(MemoryLayout<ExecutionLogEntry>.alignment, 8)

        let entry = ExecutionLogEntry(
            tokenIndex: 42,
            layerIndex: 7,
            horizonIndex: 2,
            expertID: 15,
            confidenceScore: 0.95
        )
        XCTAssertEqual(entry.tokenIndex, 42)
        XCTAssertEqual(entry.layerIndex, 7)
        XCTAssertEqual(entry.horizonIndex, 2)
        XCTAssertEqual(entry.expertID, 15)
        XCTAssertEqual(entry.padding, 0)
        XCTAssertEqual(entry.confidenceScore, 0.95)
    }

    // MARK: - Metal Context & Runtime MSL Compilation Tests

    func testMetalContextDeviceAndQueue() {
        XCTAssertNotNil(metalContext.device)
        XCTAssertNotNil(metalContext.commandQueue)
        XCTAssertTrue(metalContext.supportsMetal3)
    }

    func testRuntimeMSLCompilationAndExecution() throws {
        let msl = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void vector_add(
            device const float *inA [[buffer(0)]],
            device const float *inB [[buffer(1)]],
            device float *out [[buffer(2)]],
            uint id [[thread_position_in_grid]])
        {
            out[id] = inA[id] + inB[id];
        }
        """

        let pipeline = try metalContext.makeComputePipelineState(source: msl, functionName: "vector_add")
        XCTAssertNotNil(pipeline)

        let count = 64
        let byteSize = count * MemoryLayout<Float>.size

        let bufA = try metalContext.makeBuffer(length: byteSize)
        let bufB = try metalContext.makeBuffer(length: byteSize)
        let bufOut = try metalContext.makeBuffer(length: byteSize)

        let ptrA = bufA.contents().bindMemory(to: Float.self, capacity: count)
        let ptrB = bufB.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            ptrA[i] = Float(i)
            ptrB[i] = Float(i * 2)
        }

        let cmd = metalContext.commandQueue.makeCommandBuffer()!
        let encoder = cmd.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(bufA, offset: 0, index: 0)
        encoder.setBuffer(bufB, offset: 0, index: 1)
        encoder.setBuffer(bufOut, offset: 0, index: 2)

        let gridSize = MTLSize(width: count, height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: min(count, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()

        cmd.commit()
        cmd.waitUntilCompleted()

        let ptrOut = bufOut.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            XCTAssertEqual(ptrOut[i], Float(i * 3))
        }
    }

    // MARK: - Dual Queue Fast I/O & Synchronization Tests

    func testDualQueueFastIOQueueProperties() {
        XCTAssertNotNil(fastIOEngine.speculativeQueue)
        XCTAssertNotNil(fastIOEngine.fallbackQueue)
        XCTAssertNotNil(fastIOEngine.speculativeSyncEvent)
        XCTAssertNotNil(fastIOEngine.fallbackSyncEvent)
    }

    func testSpeculativeBlockLoadAndSync() throws {
        let expertSize = 16_384 // 16 KB aligned
        let (fileURL, cleanup) = try TestHelpers.createSyntheticWeightFile(
            expertCount: 2,
            expertSizeBytes: expertSize,
            fillByte: 0x77
        )
        defer { cleanup() }

        let handle = try WeightFileHandle(url: fileURL, device: metalContext.device)
        let targetBuffer = try metalContext.makeBuffer(length: expertSize)

        let (ticket, cmd) = try fastIOEngine.dispatchSpeculative(
            handle: handle,
            offset: 0,
            size: expertSize,
            targetBuffer: targetBuffer
        )

        let signaled = fastIOEngine.speculativeSyncEvent.waitUntilSignaled(ticket: ticket, timeoutSeconds: 5.0)
        XCTAssertTrue(signaled, "Speculative SharedEvent was not signaled within timeout")
        XCTAssertEqual(cmd.status, .complete)

        let matched = TestHelpers.verifyBufferContents(
            buffer: targetBuffer,
            length: expertSize,
            expectedByte: 0x77
        )
        XCTAssertTrue(matched, "Buffer content after DMA load did not match expected byte 0x77")
    }

    func testFallbackDemandFetchLoad() throws {
        let expertSize = 16_384
        let (fileURL, cleanup) = try TestHelpers.createSyntheticWeightFile(
            expertCount: 2,
            expertSizeBytes: expertSize,
            fillByte: 0x88
        )
        defer { cleanup() }

        let handle = try WeightFileHandle(url: fileURL, device: metalContext.device)
        let targetBuffer = try metalContext.makeBuffer(length: expertSize)

        let (ticket, cmd) = try fastIOEngine.dispatchFallback(
            handle: handle,
            offset: expertSize, // Read expert 1
            size: expertSize,
            targetBuffer: targetBuffer
        )

        let signaled = fastIOEngine.fallbackSyncEvent.waitUntilSignaled(ticket: ticket, timeoutSeconds: 5.0)
        XCTAssertTrue(signaled, "Fallback SharedEvent was not signaled within timeout")
        XCTAssertEqual(cmd.status, .complete)

        let matched = TestHelpers.verifyBufferContents(
            buffer: targetBuffer,
            length: expertSize,
            expectedByte: 0x88
        )
        XCTAssertTrue(matched, "Buffer content after fallback DMA load did not match expected byte 0x88")
    }

    func testZeroCPUSharedEventSynchronizationWithCompute() throws {
        let expertSize = 16_384
        let (fileURL, cleanup) = try TestHelpers.createSyntheticWeightFile(
            expertCount: 1,
            expertSizeBytes: expertSize,
            fillByte: 0x55
        )
        defer { cleanup() }

        let handle = try WeightFileHandle(url: fileURL, device: metalContext.device)
        let intermediateBuffer = try metalContext.makeBuffer(length: expertSize)
        let gpuDestinationBuffer = try metalContext.makeBuffer(length: expertSize)

        // 1. Dispatch speculative I/O load which will signal event at `ticket`
        let (ticket, _) = try fastIOEngine.dispatchSpeculative(
            handle: handle,
            offset: 0,
            size: expertSize,
            targetBuffer: intermediateBuffer
        )

        // 2. Enqueue GPU compute/blit command buffer waiting on `ticket` BEFORE confirming completion
        let computeCmd = metalContext.commandQueue.makeCommandBuffer()!
        fastIOEngine.speculativeSyncEvent.encodeWait(on: computeCmd, ticket: ticket)

        let blitEncoder = computeCmd.makeBlitCommandEncoder()!
        blitEncoder.copy(
            from: intermediateBuffer,
            sourceOffset: 0,
            to: gpuDestinationBuffer,
            destinationOffset: 0,
            size: expertSize
        )
        blitEncoder.endEncoding()
        computeCmd.commit()

        // 3. Wait on compute completion (CPU never had to poll or orchestrate the handoff between I/O and GPU)
        computeCmd.waitUntilCompleted()

        let matched = TestHelpers.verifyBufferContents(
            buffer: gpuDestinationBuffer,
            length: expertSize,
            expectedByte: 0x55
        )
        XCTAssertTrue(matched, "Zero-CPU synchronization failed to transfer data cleanly to GPU destination buffer")
        XCTAssertGreaterThanOrEqual(fastIOEngine.speculativeSyncEvent.currentSignaledValue, ticket)
    }

    func testBoundsCheckingAndDefensiveAssertions() throws {
        let expertSize = 4_096
        let (fileURL, cleanup) = try TestHelpers.createSyntheticWeightFile(
            expertCount: 1,
            expertSizeBytes: expertSize,
            fillByte: 0x11
        )
        defer { cleanup() }

        let handle = try WeightFileHandle(url: fileURL, device: metalContext.device)

        // 1. Destination buffer smaller than read size
        let smallBuffer = try metalContext.makeBuffer(length: 1_024)
        XCTAssertThrowsError(
            try fastIOEngine.dispatchSpeculative(
                handle: handle,
                offset: 0,
                size: 2_048,
                targetBuffer: smallBuffer
            )
        ) { error in
            guard case FastIOError.bufferTooSmall = error else {
                return XCTFail("Expected FastIOError.bufferTooSmall but got \(error)")
            }
        }

        // 2. Offset beyond file boundary
        let normalBuffer = try metalContext.makeBuffer(length: 4_096)
        XCTAssertThrowsError(
            try fastIOEngine.dispatchSpeculative(
                handle: handle,
                offset: 4_096,
                size: 1_024,
                targetBuffer: normalBuffer
            )
        ) { error in
            guard case FastIOError.offsetOutOfBounds = error else {
                return XCTFail("Expected FastIOError.offsetOutOfBounds but got \(error)")
            }
        }
    }
}
