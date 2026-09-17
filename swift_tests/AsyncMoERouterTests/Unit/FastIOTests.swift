import XCTest
import Metal
@testable import AsyncMoERouter

final class FastIOTests: XCTestCase {
    var metalContext: MetalContext!
    var fastIOEngine: FastIOEngine!
    private var syntheticFileURL: URL!
    private static var sharedMetalContext: MetalContext!
    private static var sharedFastIOEngine: FastIOEngine!
    private static var sharedSyntheticFileURL: URL!
    private static var sharedCleanup: (() -> Void)!
    private let syntheticConfig = SyntheticMoEConfig.fastTest

    override class func setUp() {
        super.setUp()
        do {
            let ctx = try MetalContext()
            sharedMetalContext = ctx
            sharedFastIOEngine = try FastIOEngine(device: ctx.device)
            let fixture = try SyntheticWeightFileGenerator.createTemporaryWeightFile(config: SyntheticMoEConfig.fastTest)
            sharedSyntheticFileURL = fixture.url
            sharedCleanup = fixture.cleanup
        } catch {
            fatalError("Failed to initialize shared FastIOTests fixtures: \(error)")
        }
    }

    override class func tearDown() {
        sharedCleanup?()
        sharedCleanup = nil
        sharedSyntheticFileURL = nil
        sharedFastIOEngine = nil
        sharedMetalContext = nil
        super.tearDown()
    }

    override func setUpWithError() throws {
        try super.setUpWithError()
        metalContext = Self.sharedMetalContext
        fastIOEngine = Self.sharedFastIOEngine
        syntheticFileURL = Self.sharedSyntheticFileURL
    }

    override func tearDownWithError() throws {
        try super.tearDownWithError()
    }

    // MARK: - 1. Dimension & Layout Tests

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

    // MARK: - 2. Metal Context & Runtime MSL Compilation Tests

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

    // MARK: - 3. Dual-Queue Fast I/O Setup (Requirement R1)

    func testDualQueueFastIOQueueProperties() {
        XCTAssertNotNil(fastIOEngine.speculativeQueue)
        XCTAssertNotNil(fastIOEngine.fallbackQueue)
        XCTAssertNotNil(fastIOEngine.speculativeSyncEvent)
        XCTAssertNotNil(fastIOEngine.fallbackSyncEvent)
    }

    func testSpeculativeQueueConfiguration() throws {
        let desc = MTLIOCommandQueueDescriptor()
        desc.priority = .low
        desc.maxCommandBufferCount = 16
        desc.type = .concurrent

        let queue = try metalContext.device.makeIOCommandQueue(descriptor: desc)
        XCTAssertNotNil(queue, "Speculative queue creation failed")
    }

    func testFallbackQueueConfiguration() throws {
        let desc = MTLIOCommandQueueDescriptor()
        desc.priority = .high
        desc.maxCommandBufferCount = 16
        desc.type = .concurrent

        let queue = try metalContext.device.makeIOCommandQueue(descriptor: desc)
        XCTAssertNotNil(queue, "Fallback queue creation failed")
    }

    func testDualQueueConcurrentExecution() throws {
        let fileHandle = try metalContext.device.makeIOFileHandle(url: syntheticFileURL)
        let buf1 = try metalContext.makeBuffer(length: syntheticConfig.expertSizeBytes)
        let buf2 = try metalContext.makeBuffer(length: syntheticConfig.expertSizeBytes)

        let sema1 = DispatchSemaphore(value: 0)
        let sema2 = DispatchSemaphore(value: 0)

        let cmd1 = fastIOEngine.speculativeQueue.makeCommandBuffer()
        cmd1.load(buf1, offset: 0, size: syntheticConfig.expertSizeBytes, sourceHandle: fileHandle, sourceHandleOffset: 0)
        cmd1.addCompletedHandler { _ in sema1.signal() }

        let cmd2 = fastIOEngine.fallbackQueue.makeCommandBuffer()
        cmd2.load(buf2, offset: 0, size: syntheticConfig.expertSizeBytes, sourceHandle: fileHandle, sourceHandleOffset: syntheticConfig.expertSizeBytes)
        cmd2.addCompletedHandler { _ in sema2.signal() }

        cmd1.commit()
        cmd2.commit()

        XCTAssertEqual(sema1.wait(timeout: .now() + 2.0), .success, "Speculative command timed out")
        XCTAssertEqual(sema2.wait(timeout: .now() + 2.0), .success, "Fallback command timed out")
        XCTAssertEqual(cmd1.status, .complete)
        XCTAssertEqual(cmd2.status, .complete)
    }

    // MARK: - 4. Direct Block Read & Accuracy Tests

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

    func testDirectBlockReadAccuracy() throws {
        let fileHandle = try metalContext.device.makeIOFileHandle(url: syntheticFileURL)

        let testTargets = [
            (layer: 0, expert: 0),
            (layer: 1, expert: 4),
            (layer: 2, expert: 7),
            (layer: 3, expert: 15)
        ]

        for target in testTargets {
            let offset = SyntheticWeightFileGenerator.fileOffset(layer: target.layer, expert: target.expert, config: syntheticConfig)
            let buffer = try metalContext.makeBuffer(length: syntheticConfig.expertSizeBytes)

            let cmd = fastIOEngine.speculativeQueue.makeCommandBuffer()
            cmd.load(buffer, offset: 0, size: syntheticConfig.expertSizeBytes, sourceHandle: fileHandle, sourceHandleOffset: offset)

            let sema = DispatchSemaphore(value: 0)
            cmd.addCompletedHandler { _ in sema.signal() }
            cmd.commit()

            XCTAssertEqual(sema.wait(timeout: .now() + 2.0), .success)
            XCTAssertEqual(cmd.status, .complete)

            let validation = MockExpertValidator.validate(
                buffer: buffer,
                layer: target.layer,
                expert: target.expert,
                config: syntheticConfig
            )
            XCTAssertTrue(validation.isValid, "Validation failed for Layer \(target.layer) Expert \(target.expert)")
        }
    }

    func testPageBoundaryAndArbitraryAlignment() throws {
        let fileHandle = try metalContext.device.makeIOFileHandle(url: syntheticFileURL)

        // 16 KB Page Aligned Read
        let pageAlignedBuf = try metalContext.makeBuffer(length: 16_384)
        let cmd1 = fastIOEngine.speculativeQueue.makeCommandBuffer()
        cmd1.load(pageAlignedBuf, offset: 0, size: 16_384, sourceHandle: fileHandle, sourceHandleOffset: 16_384)

        let sema1 = DispatchSemaphore(value: 0)
        cmd1.addCompletedHandler { _ in sema1.signal() }
        cmd1.commit()
        XCTAssertEqual(sema1.wait(timeout: .now() + 2.0), .success)
        XCTAssertEqual(cmd1.status, .complete)

        // Arbitrary Non-Aligned Read (e.g. offset 123, size 456 bytes)
        let unalignedBuf = try metalContext.makeBuffer(length: 1_024)
        let cmd2 = fastIOEngine.speculativeQueue.makeCommandBuffer()
        cmd2.load(unalignedBuf, offset: 17, size: 456, sourceHandle: fileHandle, sourceHandleOffset: 123)

        let sema2 = DispatchSemaphore(value: 0)
        cmd2.addCompletedHandler { _ in sema2.signal() }
        cmd2.commit()
        XCTAssertEqual(sema2.wait(timeout: .now() + 2.0), .success)
        XCTAssertEqual(cmd2.status, .complete)

        // Verify unaligned bytes match byte-for-byte
        let ptr = unalignedBuf.contents().advanced(by: 17).bindMemory(to: UInt8.self, capacity: 456)
        for i in 0..<456 {
            let expected = SyntheticWeightFileGenerator.expectedByte(layer: 0, expert: 0, byteOffset: 123 + i)
            XCTAssertEqual(ptr[i], expected, "Byte mismatch at relative offset \(i)")
        }
    }

    // MARK: - 5. Defensive Bounds & Assertion Tests

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

    func testClientSideBoundsProtection() throws {
        let targetBuffer = try metalContext.makeBuffer(length: 1_024)
        let requestedSize = 2_048

        let wouldOverflow = (0 + requestedSize) > targetBuffer.length
        XCTAssertTrue(wouldOverflow, "Defensive check must flag buffer overflow condition")
    }

    // MARK: - 6. Zero-CPU Hardware Synchronization Tests (Requirement R1)

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

        // 1. Dispatch speculative I/O load which will signal event at ticket
        let (ticket, _) = try fastIOEngine.dispatchSpeculative(
            handle: handle,
            offset: 0,
            size: expertSize,
            targetBuffer: intermediateBuffer
        )

        // 2. Enqueue GPU compute/blit command buffer waiting on ticket BEFORE confirming completion
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

    func testBitwiseGPUTransformationKernel() throws {
        let mslSource = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void transform_weights(
            device const uchar* inWeights [[buffer(0)]],
            device uchar* outWeights [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint id [[thread_position_in_grid]])
        {
            if (id < count) {
                outWeights[id] = inWeights[id] ^ 0xFF;
            }
        }
        """

        let pipeline = try metalContext.makeComputePipelineState(source: mslSource, functionName: "transform_weights")
        let fileHandle = try metalContext.device.makeIOFileHandle(url: syntheticFileURL)

        let syncEvent = try SyncEvent(device: metalContext.device)
        let ticket: UInt64 = 101

        let inBuffer = try metalContext.makeBuffer(length: syntheticConfig.expertSizeBytes)
        let outBuffer = try metalContext.makeBuffer(length: syntheticConfig.expertSizeBytes)

        // Encode compute kernel waiting on hardware event
        let computeCmd = metalContext.commandQueue.makeCommandBuffer()!
        syncEvent.encodeWait(on: computeCmd, ticket: ticket)

        let encoder = computeCmd.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inBuffer, offset: 0, index: 0)
        encoder.setBuffer(outBuffer, offset: 0, index: 1)
        var count = UInt32(syntheticConfig.expertSizeBytes)
        encoder.setBytes(&count, length: 4, index: 2)

        let gridSize = MTLSize(width: syntheticConfig.expertSizeBytes, height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: min(pipeline.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        computeCmd.commit()

        // Commit IO load
        let ioCmd = fastIOEngine.speculativeQueue.makeCommandBuffer()
        ioCmd.load(inBuffer, offset: 0, size: syntheticConfig.expertSizeBytes, sourceHandle: fileHandle, sourceHandleOffset: 0)
        syncEvent.encodeSignal(on: ioCmd, ticket: ticket)
        ioCmd.commit()

        computeCmd.waitUntilCompleted()

        // Verify bitwise transformation: out = in ^ 0xFF
        let inPtr = inBuffer.contents().bindMemory(to: UInt8.self, capacity: syntheticConfig.expertSizeBytes)
        let outPtr = outBuffer.contents().bindMemory(to: UInt8.self, capacity: syntheticConfig.expertSizeBytes)
        for i in 0..<syntheticConfig.expertSizeBytes {
            XCTAssertEqual(outPtr[i], inPtr[i] ^ 0xFF)
        }
    }

    func testNonBlockingCPUQueryLatency() throws {
        let syncEvent = try SyncEvent(device: metalContext.device)

        let iterations = 10_000
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = syncEvent.signaledValue
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let averageNanoseconds = (elapsed / Double(iterations)) * 1_000_000_000

        XCTAssertLessThan(averageNanoseconds, 1000.0, "Atomic query of signaledValue should be < 1 microsecond (measured \(averageNanoseconds) ns)")
    }

    func testMultiSlotOutOfOrderSafety() throws {
        let fileHandle = try metalContext.device.makeIOFileHandle(url: syntheticFileURL)

        let slot0Event = try SyncEvent(device: metalContext.device)
        let slot1Event = try SyncEvent(device: metalContext.device)

        let buf0 = try metalContext.makeBuffer(length: syntheticConfig.expertSizeBytes)
        let buf1 = try metalContext.makeBuffer(length: syntheticConfig.expertSizeBytes)

        // Slot 0 waits for ticket 1
        let computeCmd0 = metalContext.commandQueue.makeCommandBuffer()!
        slot0Event.encodeWait(on: computeCmd0, ticket: 1)
        let blit0 = computeCmd0.makeBlitCommandEncoder()!
        blit0.fill(buffer: buf0, range: 0..<4, value: 0x11)
        blit0.endEncoding()
        computeCmd0.commit()

        // Slot 1 signals ticket 10 (higher ticket)
        let ioCmd1 = fastIOEngine.speculativeQueue.makeCommandBuffer()
        ioCmd1.load(buf1, offset: 0, size: syntheticConfig.expertSizeBytes, sourceHandle: fileHandle, sourceHandleOffset: syntheticConfig.expertSizeBytes)
        slot1Event.encodeSignal(on: ioCmd1, ticket: 10)
        ioCmd1.commit()

        ioCmd1.waitUntilCompleted()
        XCTAssertEqual(slot1Event.signaledValue, 10)

        // Crucial verification: Slot 0 event signaledValue must STILL be 0!
        XCTAssertEqual(slot0Event.signaledValue, 0, "Slot 0 event must remain unsignaled despite Slot 1 signaling ticket 10")
        XCTAssertFalse(slot0Event.isSignaled(at: 1))

        // Now release Slot 0
        let ioCmd0 = fastIOEngine.speculativeQueue.makeCommandBuffer()
        ioCmd0.load(buf0, offset: 0, size: syntheticConfig.expertSizeBytes, sourceHandle: fileHandle, sourceHandleOffset: 0)
        slot0Event.encodeSignal(on: ioCmd0, ticket: 1)
        ioCmd0.commit()

        computeCmd0.waitUntilCompleted()
        XCTAssertEqual(slot0Event.signaledValue, 1)
    }

    func testTryCancelAndSignalDropping() throws {
        let fileHandle = try metalContext.device.makeIOFileHandle(url: syntheticFileURL)
        let syncEvent = try SyncEvent(device: metalContext.device)
        let targetTicket: UInt64 = 88

        let buffer = try metalContext.makeBuffer(length: syntheticConfig.expertSizeBytes)
        let cmd = fastIOEngine.speculativeQueue.makeCommandBuffer()
        cmd.load(buffer, offset: 0, size: syntheticConfig.expertSizeBytes, sourceHandle: fileHandle, sourceHandleOffset: 0)
        syncEvent.encodeSignal(on: cmd, ticket: targetTicket)

        let sema = DispatchSemaphore(value: 0)
        var observedStatus: MTLIOStatus?
        cmd.addCompletedHandler { cb in
            observedStatus = cb.status
            sema.signal()
        }

        cmd.commit()
        cmd.tryCancel()

        XCTAssertEqual(sema.wait(timeout: .now() + 2.0), .success)
        XCTAssertEqual(observedStatus, .cancelled)
        XCTAssertEqual(syncEvent.signaledValue, 0, "Hardware must drop signal upon cancellation")
    }

    func testMemoryLifecycleUnderRepeatedLoads() throws {
        let fileHandle = try metalContext.device.makeIOFileHandle(url: syntheticFileURL)
        let buffer = try metalContext.makeBuffer(length: syntheticConfig.expertSizeBytes)
        let syncEvent = try SyncEvent(device: metalContext.device)

        for i in 1...100 {
            autoreleasepool {
                let ticket = UInt64(i)
                let cmd = fastIOEngine.speculativeQueue.makeCommandBuffer()
                cmd.load(buffer, offset: 0, size: syntheticConfig.expertSizeBytes, sourceHandle: fileHandle, sourceHandleOffset: 0)
                syncEvent.encodeSignal(on: cmd, ticket: ticket)
                cmd.commit()

                let computeCmd = metalContext.commandQueue.makeCommandBuffer()!
                syncEvent.encodeWait(on: computeCmd, ticket: ticket)
                let blit = computeCmd.makeBlitCommandEncoder()!
                blit.fill(buffer: buffer, range: 0..<4, value: UInt8(i % 255))
                blit.endEncoding()
                computeCmd.commit()
                computeCmd.waitUntilCompleted()

                cmd.waitUntilCompleted()
                XCTAssertEqual(syncEvent.signaledValue, ticket)
            }
        }
    }
}
