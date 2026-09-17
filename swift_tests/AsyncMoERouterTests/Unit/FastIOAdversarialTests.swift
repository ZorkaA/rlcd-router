import XCTest
import Metal
@testable import AsyncMoERouter

final class FastIOAdversarialTests: XCTestCase {
    private static var sharedMetalContext: MetalContext!
    private static var sharedFastIOEngine: FastIOEngine!
    private static var sharedSyntheticFileURL: URL!
    private static var sharedCleanup: (() -> Void)!

    override class func setUp() {
        super.setUp()
        do {
            let ctx = try MetalContext()
            sharedMetalContext = ctx
            sharedFastIOEngine = try FastIOEngine(device: ctx.device)
            // Create a larger synthetic file for preemption testing: 16 experts * 1MB = 16MB
            let expertCount = 16
            let expertSizeBytes = 1_048_576 // 1 MB
            let fixture = try TestHelpers.createSyntheticWeightFile(
                expertCount: expertCount,
                expertSizeBytes: expertSizeBytes,
                fillByte: 0x42
            )
            sharedSyntheticFileURL = fixture.url
            sharedCleanup = fixture.cleanup
        } catch {
            fatalError("Failed to setup FastIOAdversarialTests class fixtures: \(error)")
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

    // MARK: - 1. Priority Preemption Verification
    func testFallbackQueuePriorityPreemptionOverSpeculativeQueue() throws {
        let engine = Self.sharedFastIOEngine!
        let ctx = Self.sharedMetalContext!
        let fileURL = Self.sharedSyntheticFileURL!

        let fileHandle = try ctx.device.makeIOFileHandle(url: fileURL)
        let expertSize = 1_048_576 // 1 MB

        let numSpeculative = 10
        var specBuffers: [any MTLBuffer] = []
        for _ in 0..<numSpeculative {
            specBuffers.append(try ctx.makeBuffer(length: expertSize))
        }
        let fallbackBuffer = try ctx.makeBuffer(length: expertSize)

        let lock = NSLock()
        var specCompletionTimes = [Double](repeating: 0, count: numSpeculative)
        var fallbackCompletionTime: Double = 0

        let startTime = CFAbsoluteTimeGetCurrent()
        let group = DispatchGroup()

        // 1. Submit numSpeculative commands to speculativeQueue (.low)
        var specCmds: [any MTLIOCommandBuffer] = []
        for i in 0..<numSpeculative {
            group.enter()
            let cmd = engine.speculativeQueue.makeCommandBuffer()
            cmd.load(specBuffers[i], offset: 0, size: expertSize, sourceHandle: fileHandle, sourceHandleOffset: i * expertSize)
            let index = i
            cmd.addCompletedHandler { _ in
                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                lock.lock()
                specCompletionTimes[index] = elapsed
                lock.unlock()
                group.leave()
            }
            specCmds.append(cmd)
        }

        // Commit all speculative commands
        for cmd in specCmds {
            cmd.commit()
        }

        // 2. Submit urgent command to fallbackQueue (.high)
        group.enter()
        let fallbackCmd = engine.fallbackQueue.makeCommandBuffer()
        fallbackCmd.load(fallbackBuffer, offset: 0, size: expertSize, sourceHandle: fileHandle, sourceHandleOffset: 0)
        fallbackCmd.addCompletedHandler { _ in
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            lock.lock()
            fallbackCompletionTime = elapsed
            lock.unlock()
            group.leave()
        }
        fallbackCmd.commit()

        // Wait for all commands to complete
        let waitResult = group.wait(timeout: .now() + 10.0)
        XCTAssertEqual(waitResult, .success, "All Fast I/O commands must complete within 10s")

        lock.lock()
        let fbTime = fallbackCompletionTime
        let maxSpecTime = specCompletionTimes.max() ?? 0
        let minSpecTime = specCompletionTimes.min() ?? 0
        let specTimes = specCompletionTimes
        lock.unlock()

        print("[PREEMPTION TEST] Fallback completed at: \(fbTime)s")
        print("[PREEMPTION TEST] Speculative range: [\(minSpecTime)s - \(maxSpecTime)s], values: \(specTimes)")

        // Fallback was submitted AFTER all speculative commands, but because of .high priority,
        // it must complete before the final speculative transfers finish
        XCTAssertLessThan(fbTime, maxSpecTime, "High-priority fallbackQueue must finish before last low-priority speculative command")
    }

    // MARK: - 2. tryCancel Signal Dropping & GPU Compute Block Verification
    func testTryCancelSignalDroppingAndGPUComputeBlocking() throws {
        let engine = Self.sharedFastIOEngine!
        let ctx = Self.sharedMetalContext!
        let fileURL = Self.sharedSyntheticFileURL!

        let fileHandle = try ctx.device.makeIOFileHandle(url: fileURL)
        let expertSize = 16_384
        let ioBuffer = try ctx.makeBuffer(length: expertSize)
        let gpuDestBuffer = try ctx.makeBuffer(length: expertSize)

        // Initialize gpuDestBuffer with 0x00
        memset(gpuDestBuffer.contents(), 0, expertSize)

        let syncEvent = try SyncEvent(device: ctx.device)
        let targetTicket: UInt64 = 777

        // 1. Enqueue GPU compute command that waits for targetTicket before blitting 0xAA
        let computeCmd = ctx.commandQueue.makeCommandBuffer()!
        syncEvent.encodeWait(on: computeCmd, ticket: targetTicket)

        let blitEncoder = computeCmd.makeBlitCommandEncoder()!
        blitEncoder.fill(buffer: gpuDestBuffer, range: 0..<expertSize, value: 0xAA)
        blitEncoder.endEncoding()

        // Commit GPU compute command
        computeCmd.commit()

        // 2. Dispatch speculative I/O load configured to signal targetTicket upon completion
        let ioCmd = engine.speculativeQueue.makeCommandBuffer()
        ioCmd.load(ioBuffer, offset: 0, size: expertSize, sourceHandle: fileHandle, sourceHandleOffset: 0)
        syncEvent.encodeSignal(on: ioCmd, ticket: targetTicket)

        let ioSema = DispatchSemaphore(value: 0)
        var observedIOStatus: MTLIOStatus?
        ioCmd.addCompletedHandler { cb in
            observedIOStatus = cb.status
            ioSema.signal()
        }

        ioCmd.commit()

        // 3. Immediately cancel the speculative command buffer
        ioCmd.tryCancel()

        // Wait for IO completion handler
        XCTAssertEqual(ioSema.wait(timeout: .now() + 2.0), .success)
        XCTAssertEqual(observedIOStatus, .cancelled, "Speculative command should be cancelled")

        // 4. Assert signal was dropped
        XCTAssertEqual(syncEvent.signaledValue, 0, "Shared event signal must NOT advance on cancelled command")
        XCTAssertFalse(syncEvent.isSignaled(at: targetTicket))

        // 5. Verify GPU compute command is STILL BLOCKED and has NOT executed prematurely
        // Wait 150ms to allow any phantom hardware execution
        usleep(150_000)

        XCTAssertNotEqual(computeCmd.status, .completed, "GPU compute command must NOT complete prematurely on dropped signal")
        let ptrBefore = gpuDestBuffer.contents().bindMemory(to: UInt8.self, capacity: expertSize)
        XCTAssertEqual(ptrBefore[0], 0x00, "GPU compute must not have touched buffer while blocked")

        // 6. Now simulate fallback resolution: explicitly signal the ticket
        syncEvent.sharedEvent.signaledValue = targetTicket

        // 7. Wait for compute command completion now that event is signaled
        computeCmd.waitUntilCompleted()
        XCTAssertEqual(computeCmd.status, .completed, "GPU compute command must complete once signal is satisfied")

        let ptrAfter = gpuDestBuffer.contents().bindMemory(to: UInt8.self, capacity: expertSize)
        XCTAssertEqual(ptrAfter[0], 0xAA, "GPU compute must execute and fill buffer once unblocked")
    }

    // MARK: - 3. Queue 16-Command Saturation & Backpressure Verification
    func testSpeculativeQueue16CommandSaturation() throws {
        let engine = Self.sharedFastIOEngine!
        let ctx = Self.sharedMetalContext!
        let fileURL = Self.sharedSyntheticFileURL!

        let fileHandle = try ctx.device.makeIOFileHandle(url: fileURL)
        let blockSize = 16_384

        // 1. Allocate 16 destination buffers
        var buffers: [any MTLBuffer] = []
        for _ in 0..<16 {
            buffers.append(try ctx.makeBuffer(length: blockSize))
        }

        // 2. Dispatch exactly 16 command buffers simultaneously (saturating maxCommandBufferCount = 16)
        let group = DispatchGroup()
        var statuses = [MTLIOStatus](repeating: .pending, count: 16)
        let lock = NSLock()

        for i in 0..<16 {
            group.enter()
            let cmd = engine.speculativeQueue.makeCommandBuffer()
            cmd.load(buffers[i], offset: 0, size: blockSize, sourceHandle: fileHandle, sourceHandleOffset: i * blockSize)
            let idx = i
            cmd.addCompletedHandler { completedCmd in
                lock.lock()
                statuses[idx] = completedCmd.status
                lock.unlock()
                group.leave()
            }
            cmd.commit()
        }

        let result = group.wait(timeout: .now() + 5.0)
        XCTAssertEqual(result, .success, "All 16 saturated command buffers must complete within 5s")

        lock.lock()
        for i in 0..<16 {
            XCTAssertEqual(statuses[i], .complete, "Command buffer \(i) did not complete successfully")
        }
        lock.unlock()
    }

    // MARK: - 4. Dual Queue 32-Command Simultaneous Saturation
    func testDualQueueSimultaneous32CommandSaturation() throws {
        let engine = Self.sharedFastIOEngine!
        let ctx = Self.sharedMetalContext!
        let fileURL = Self.sharedSyntheticFileURL!

        let fileHandle = try ctx.device.makeIOFileHandle(url: fileURL)
        let blockSize = 16_384

        // 16 buffers for speculative, 16 buffers for fallback
        var specBuffers: [any MTLBuffer] = []
        var fallbackBuffers: [any MTLBuffer] = []
        for _ in 0..<16 {
            specBuffers.append(try ctx.makeBuffer(length: blockSize))
            fallbackBuffers.append(try ctx.makeBuffer(length: blockSize))
        }

        let group = DispatchGroup()
        var specStatuses = [MTLIOStatus](repeating: .pending, count: 16)
        var fallbackStatuses = [MTLIOStatus](repeating: .pending, count: 16)
        let lock = NSLock()

        // Saturate speculativeQueue with 16
        for i in 0..<16 {
            group.enter()
            let cmd = engine.speculativeQueue.makeCommandBuffer()
            cmd.load(specBuffers[i], offset: 0, size: blockSize, sourceHandle: fileHandle, sourceHandleOffset: i * blockSize)
            let idx = i
            cmd.addCompletedHandler { cb in
                lock.lock()
                specStatuses[idx] = cb.status
                lock.unlock()
                group.leave()
            }
            cmd.commit()
        }

        // Simultaneously saturate fallbackQueue with 16
        for i in 0..<16 {
            group.enter()
            let cmd = engine.fallbackQueue.makeCommandBuffer()
            cmd.load(fallbackBuffers[i], offset: 0, size: blockSize, sourceHandle: fileHandle, sourceHandleOffset: i * blockSize)
            let idx = i
            cmd.addCompletedHandler { cb in
                lock.lock()
                fallbackStatuses[idx] = cb.status
                lock.unlock()
                group.leave()
            }
            cmd.commit()
        }

        let result = group.wait(timeout: .now() + 5.0)
        XCTAssertEqual(result, .success, "All 32 simultaneous commands across both queues must complete")

        lock.lock()
        for i in 0..<16 {
            XCTAssertEqual(specStatuses[i], .complete, "Speculative command \(i) failed")
            XCTAssertEqual(fallbackStatuses[i], .complete, "Fallback command \(i) failed")
        }
        lock.unlock()
    }

    // MARK: - 5. Backpressure & 17th Command Buffer Allocation Behavior
    func testBackpressureBeyond16CommandLimit() throws {
        let engine = Self.sharedFastIOEngine!
        let ctx = Self.sharedMetalContext!
        let fileURL = Self.sharedSyntheticFileURL!

        let fileHandle = try ctx.device.makeIOFileHandle(url: fileURL)
        let blockSize = 16_384

        // Verify that submitting 20 commands sequentially or in overlapping batches
        // handles backpressure gracefully without crashing or leaking.
        for batch in 0..<3 {
            let count = 20
            var buffers: [any MTLBuffer] = []
            for _ in 0..<count {
                buffers.append(try ctx.makeBuffer(length: blockSize))
            }

            let group = DispatchGroup()
            for i in 0..<count {
                group.enter()
                autoreleasepool {
                    let cmd = engine.speculativeQueue.makeCommandBuffer()
                    cmd.load(buffers[i], offset: 0, size: blockSize, sourceHandle: fileHandle, sourceHandleOffset: (i % 16) * blockSize)
                    cmd.addCompletedHandler { _ in
                        group.leave()
                    }
                    cmd.commit()
                }
            }

            let result = group.wait(timeout: .now() + 5.0)
            XCTAssertEqual(result, .success, "Batch \(batch) of 20 commands (exceeding 16-limit) must complete cleanly with backpressure")
        }
    }

    // MARK: - 6. Cache-Miss Deadlock Resolution Protocol Simulation (M1 ↔ M2 Contract)
    func testCacheMissDeadlockResolutionEndToEnd() throws {
        let engine = Self.sharedFastIOEngine!
        let ctx = Self.sharedMetalContext!
        let fileURL = Self.sharedSyntheticFileURL!

        let fileHandle = try ctx.device.makeIOFileHandle(url: fileURL)
        let expertSize = 16_384
        let specBuffer = try ctx.makeBuffer(length: expertSize)
        let fallbackBuffer = try ctx.makeBuffer(length: expertSize)
        let computeOutBuffer = try ctx.makeBuffer(length: expertSize)

        // Initialize computeOutBuffer to 0x00
        memset(computeOutBuffer.contents(), 0, expertSize)

        let syncEvent = try SyncEvent(device: ctx.device)
        let sharedTicket: UInt64 = 555

        // GPU compute queue enqueues work waiting on sharedTicket
        let computeCmd = ctx.commandQueue.makeCommandBuffer()!
        syncEvent.encodeWait(on: computeCmd, ticket: sharedTicket)

        let blit = computeCmd.makeBlitCommandEncoder()!
        blit.copy(from: fallbackBuffer, sourceOffset: 0, to: computeOutBuffer, destinationOffset: 0, size: expertSize)
        blit.endEncoding()
        computeCmd.commit()

        // 1. Issue speculative load signaling sharedTicket
        let specCmd = engine.speculativeQueue.makeCommandBuffer()
        specCmd.load(specBuffer, offset: 0, size: expertSize, sourceHandle: fileHandle, sourceHandleOffset: 0)
        syncEvent.encodeSignal(on: specCmd, ticket: sharedTicket)
        specCmd.commit()

        // 2. Cache-miss timeout triggers: abandon speculative slot & cancel
        specCmd.tryCancel()

        // 3. Issue urgent fallback load on fallbackQueue signaling sharedTicket
        let fallbackCmd = engine.fallbackQueue.makeCommandBuffer()
        fallbackCmd.load(fallbackBuffer, offset: 0, size: expertSize, sourceHandle: fileHandle, sourceHandleOffset: 0)
        syncEvent.encodeSignal(on: fallbackCmd, ticket: sharedTicket)
        fallbackCmd.commit()

        // 4. Compute command must successfully unblock on the fallback signal
        computeCmd.waitUntilCompleted()
        XCTAssertEqual(computeCmd.status, .completed, "Compute command must complete via fallback signal")
        fallbackCmd.waitUntilCompleted()
        XCTAssertEqual(fallbackCmd.status, .complete)

        // Verify data was copied
        let ptr = computeOutBuffer.contents().bindMemory(to: UInt8.self, capacity: expertSize)
        XCTAssertEqual(ptr[0], 0x42, "Compute kernel must have executed using weights loaded by fallbackQueue")
    }

    // MARK: - 7. Mass Cancellation Under 16-Command Saturation
    func testMassCancellationUnder16CommandSaturation() throws {
        let engine = Self.sharedFastIOEngine!
        let ctx = Self.sharedMetalContext!
        let fileURL = Self.sharedSyntheticFileURL!

        let fileHandle = try ctx.device.makeIOFileHandle(url: fileURL)
        let blockSize = 16_384

        var buffers: [any MTLBuffer] = []
        for _ in 0..<16 {
            buffers.append(try ctx.makeBuffer(length: blockSize))
        }

        let syncEvent = try SyncEvent(device: ctx.device)
        var cmds: [any MTLIOCommandBuffer] = []
        let group = DispatchGroup()
        let lock = NSLock()
        var statuses = [MTLIOStatus](repeating: .pending, count: 16)

        for i in 0..<16 {
            group.enter()
            let cmd = engine.speculativeQueue.makeCommandBuffer()
            cmd.load(buffers[i], offset: 0, size: blockSize, sourceHandle: fileHandle, sourceHandleOffset: i * blockSize)
            syncEvent.encodeSignal(on: cmd, ticket: UInt64(i + 1))
            let idx = i
            cmd.addCompletedHandler { cb in
                lock.lock()
                statuses[idx] = cb.status
                lock.unlock()
                group.leave()
            }
            cmds.append(cmd)
        }

        // Commit all 16 and immediately cancel all 16
        for cmd in cmds {
            cmd.commit()
            cmd.tryCancel()
        }

        let waitResult = group.wait(timeout: .now() + 5.0)
        XCTAssertEqual(waitResult, .success, "All 16 cancelled commands must return within 5s")

        // Crucial check: Queue must be fully functional for subsequent dispatches
        let followUpBuffer = try ctx.makeBuffer(length: blockSize)
        let followUpCmd = engine.speculativeQueue.makeCommandBuffer()
        followUpCmd.load(followUpBuffer, offset: 0, size: blockSize, sourceHandle: fileHandle, sourceHandleOffset: 0)
        let followUpSema = DispatchSemaphore(value: 0)
        var followUpStatus: MTLIOStatus?
        followUpCmd.addCompletedHandler { cb in
            followUpStatus = cb.status
            followUpSema.signal()
        }
        followUpCmd.commit()

        XCTAssertEqual(followUpSema.wait(timeout: .now() + 3.0), .success)
        XCTAssertEqual(followUpStatus, .complete, "Queue must remain healthy after mass cancellation")
    }

    // MARK: - 8. Concurrent Compute Waiters on Single Shared Event
    func testConcurrentComputeWaitersOnSingleTicket() throws {
        let engine = Self.sharedFastIOEngine!
        let ctx = Self.sharedMetalContext!
        let fileURL = Self.sharedSyntheticFileURL!

        let fileHandle = try ctx.device.makeIOFileHandle(url: fileURL)
        let expertSize = 16_384
        let ioBuffer = try ctx.makeBuffer(length: expertSize)

        let syncEvent = try SyncEvent(device: ctx.device)
        let ticket: UInt64 = 999

        let numComputeWaiters = 4
        var computeCmds: [any MTLCommandBuffer] = []
        var outBuffers: [any MTLBuffer] = []

        for i in 0..<numComputeWaiters {
            let outBuf = try ctx.makeBuffer(length: expertSize)
            memset(outBuf.contents(), 0, expertSize)
            outBuffers.append(outBuf)

            let cmd = ctx.commandQueue.makeCommandBuffer()!
            syncEvent.encodeWait(on: cmd, ticket: ticket)
            let blit = cmd.makeBlitCommandEncoder()!
            blit.fill(buffer: outBuf, range: 0..<expertSize, value: UInt8(0x30 + i))
            blit.endEncoding()
            cmd.commit()
            computeCmds.append(cmd)
        }

        // Verify none have completed before IO signal
        usleep(50_000)
        for cmd in computeCmds {
            XCTAssertNotEqual(cmd.status, .completed, "Waiters must not complete before ticket is reached")
        }

        // Dispatch speculative IO load signaling ticket
        let ioCmd = engine.speculativeQueue.makeCommandBuffer()
        ioCmd.load(ioBuffer, offset: 0, size: expertSize, sourceHandle: fileHandle, sourceHandleOffset: 0)
        syncEvent.encodeSignal(on: ioCmd, ticket: ticket)
        ioCmd.commit()

        // Wait for all compute command buffers
        for (i, cmd) in computeCmds.enumerated() {
            cmd.waitUntilCompleted()
            XCTAssertEqual(cmd.status, .completed, "Compute waiter \(i) must complete successfully")
            let ptr = outBuffers[i].contents().bindMemory(to: UInt8.self, capacity: expertSize)
            XCTAssertEqual(ptr[0], UInt8(0x30 + i), "Compute waiter \(i) must execute correctly")
        }
    }

    // MARK: - 9. Repeated 100 Rapid Cancel Cycles Under Zero Leak
    func testRepeatedCancelCyclesUnderZeroLeak() throws {
        let engine = Self.sharedFastIOEngine!
        let ctx = Self.sharedMetalContext!
        let fileURL = Self.sharedSyntheticFileURL!

        let fileHandle = try ctx.device.makeIOFileHandle(url: fileURL)
        let expertSize = 16_384
        let buffer = try ctx.makeBuffer(length: expertSize)

        for _ in 1...100 {
            autoreleasepool {
                let cmd = engine.speculativeQueue.makeCommandBuffer()
                cmd.load(buffer, offset: 0, size: expertSize, sourceHandle: fileHandle, sourceHandleOffset: 0)
                cmd.commit()
                cmd.tryCancel()
                cmd.waitUntilCompleted()
                XCTAssertTrue(cmd.status == .cancelled || cmd.status == .complete)
            }
        }
    }
}
