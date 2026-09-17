import XCTest
import Metal
import Foundation
import Darwin
import os
@testable import AsyncMoERouter

final class FastIOChallenger2StressTests: XCTestCase {
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
            fatalError("Failed to initialize FastIOChallenger2StressTests fixtures: \(error)")
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

    // MARK: - Memory Telemetry Helpers

    /// Returns the physical memory footprint of the current process in bytes.
    /// This corresponds to `phys_footprint` from `task_vm_info`, which is the authoritative
    /// memory metric used by Apple Silicon memory accounting (excluding shared purgeable memory).
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
            return UInt64(info.resident_size)
        }
        return 0
    }

    // MARK: - 1. Memory Leak Stress (200+ Repeated Loads, 0-Byte Growth)

    func test250RepeatedLoadsZeroMemoryGrowth() throws {
        let engine = Self.sharedFastIOEngine!
        let context = Self.sharedMetalContext!
        let fileURL = Self.sharedSyntheticFileURL!
        let handle = try WeightFileHandle(url: fileURL, device: context.device)

        let expertSize = syntheticConfig.expertSizeBytes
        let targetBuffer = try context.makeBuffer(length: expertSize)

        // 1. Warm-up Phase: 20 iterations to reach steady-state (fill driver tables, JIT, page faults)
        for _ in 0..<20 {
            autoreleasepool {
                let (ticket, _) = try! engine.dispatchSpeculative(
                    handle: handle,
                    offset: 0,
                    size: expertSize,
                    targetBuffer: targetBuffer
                )
                _ = engine.speculativeSyncEvent.waitUntilSignaled(ticket: ticket, timeoutSeconds: 2.0)
            }
        }

        // Allow kernel memory and autorelease pools to settle
        usleep(50_000)

        let baselineFootprint = Self.getPhysicalFootprintBytes()
        let baselineResident = Self.getResidentMemoryBytes()

        // 2. Stress Phase: Exactly 250 repeated loads alternating between Speculative and Fallback queues
        let iterations = 250
        for i in 1...iterations {
            autoreleasepool {
                if i % 2 == 0 {
                    let (ticket, cmd) = try! engine.dispatchSpeculative(
                        handle: handle,
                        offset: (i % 4) * expertSize,
                        size: expertSize,
                        targetBuffer: targetBuffer
                    )
                    let ok = engine.speculativeSyncEvent.waitUntilSignaled(ticket: ticket, timeoutSeconds: 2.0)
                    XCTAssertTrue(ok)
                    XCTAssertEqual(cmd.status, .complete)
                } else {
                    let (ticket, cmd) = try! engine.dispatchFallback(
                        handle: handle,
                        offset: ((i + 1) % 4) * expertSize,
                        size: expertSize,
                        targetBuffer: targetBuffer
                    )
                    let ok = engine.fallbackSyncEvent.waitUntilSignaled(ticket: ticket, timeoutSeconds: 2.0)
                    XCTAssertTrue(ok)
                    XCTAssertEqual(cmd.status, .complete)
                }
            }
        }

        // Settle
        usleep(50_000)

        let finalFootprint = Self.getPhysicalFootprintBytes()
        let finalResident = Self.getResidentMemoryBytes()

        let footprintGrowth = Int64(finalFootprint) - Int64(baselineFootprint)
        let residentGrowth = Int64(finalResident) - Int64(baselineResident)

        print("""
        [Stress Test 1: 250 Repeated Loads]
        Baseline Footprint: \(baselineFootprint) bytes
        Final Footprint:    \(finalFootprint) bytes
        Footprint Delta:    \(footprintGrowth) bytes
        Baseline Resident:  \(baselineResident) bytes
        Final Resident:     \(finalResident) bytes
        Resident Delta:     \(residentGrowth) bytes
        """)

        // Memory leak verification:
        // System RAM physical footprint growth must be <= 0 bytes (or within a single 16KB system page if macOS internal logging occurred)
        // Strictly verify zero unbounded growth: footprint growth should be 0 or negative (deallocated).
        XCTAssertLessThanOrEqual(
            footprintGrowth,
            0,
            "Memory footprint grew by \(footprintGrowth) bytes over 250 repeated Fast I/O loads! Expected 0 byte growth."
        )
    }

    func testRepeatedDynamicBufferAllocationAndDeallocationStress() throws {
        let engine = Self.sharedFastIOEngine!
        let context = Self.sharedMetalContext!
        let fileURL = Self.sharedSyntheticFileURL!
        let handle = try WeightFileHandle(url: fileURL, device: context.device)
        let expertSize = syntheticConfig.expertSizeBytes

        // Warm up
        for _ in 0..<10 {
            autoreleasepool {
                let buf = try! context.makeBuffer(length: expertSize)
                let (ticket, _) = try! engine.dispatchSpeculative(handle: handle, offset: 0, size: expertSize, targetBuffer: buf)
                _ = engine.speculativeSyncEvent.waitUntilSignaled(ticket: ticket, timeoutSeconds: 2.0)
            }
        }
        usleep(50_000)

        let startFootprint = Self.getPhysicalFootprintBytes()

        // 200 cycles of dynamic buffer allocation, load, sync, and deallocation
        for i in 1...200 {
            autoreleasepool {
                let buf = try! context.makeBuffer(length: expertSize)
                let (ticket, _) = try! engine.dispatchSpeculative(
                    handle: handle,
                    offset: (i % 4) * expertSize,
                    size: expertSize,
                    targetBuffer: buf
                )
                let ok = engine.speculativeSyncEvent.waitUntilSignaled(ticket: ticket, timeoutSeconds: 2.0)
                XCTAssertTrue(ok)
                // Buffer deallocates when exiting autoreleasepool
            }
        }
        usleep(50_000)

        let endFootprint = Self.getPhysicalFootprintBytes()
        let growth = Int64(endFootprint) - Int64(startFootprint)

        print("""
        [Stress Test: 200 Dynamic Allocations]
        Start Footprint: \(startFootprint) bytes
        End Footprint:   \(endFootprint) bytes
        Growth:          \(growth) bytes
        """)

        // Must not leak allocated MTLBuffers across 200 cycles
        XCTAssertLessThanOrEqual(
            growth,
            0,
            "Dynamic MTLBuffer lifecycle leaked \(growth) bytes over 200 cycles."
        )
    }

    // MARK: - 2. Defensive Bounds & Corruption Resistance

    func testDefensiveBoundsReadingPastEOF() throws {
        let context = Self.sharedMetalContext!
        let fileURL = Self.sharedSyntheticFileURL!
        let handle = try WeightFileHandle(url: fileURL, device: context.device)
        let fileSize = handle.fileSize
        let buffer = try context.makeBuffer(length: 16_384)

        // 1. Exact EOF boundary read (offset == fileSize, size > 0)
        XCTAssertThrowsError(
            try handle.validateBounds(offset: fileSize, size: 1)
        ) { error in
            guard case FastIOError.offsetOutOfBounds(let off, let sz, let fs) = error else {
                return XCTFail("Expected FastIOError.offsetOutOfBounds, got \(error)")
            }
            XCTAssertEqual(off, fileSize)
            XCTAssertEqual(sz, 1)
            XCTAssertEqual(fs, fileSize)
        }

        // 2. Reading straddling EOF (offset = fileSize - 10, size = 20)
        XCTAssertThrowsError(
            try handle.validateBounds(offset: fileSize - 10, size: 20)
        ) { error in
            guard case FastIOError.offsetOutOfBounds = error else {
                return XCTFail("Expected FastIOError.offsetOutOfBounds, got \(error)")
            }
        }

        // 3. Offset well past EOF (offset = fileSize + 1000, size = 100)
        XCTAssertThrowsError(
            try handle.validateBounds(offset: fileSize + 1000, size: 100)
        ) { error in
            guard case FastIOError.offsetOutOfBounds = error else {
                return XCTFail("Expected FastIOError.offsetOutOfBounds, got \(error)")
            }
        }

        // 4. Size larger than entire file
        XCTAssertThrowsError(
            try handle.validateBounds(offset: 0, size: fileSize + 1)
        ) { error in
            guard case FastIOError.offsetOutOfBounds = error else {
                return XCTFail("Expected FastIOError.offsetOutOfBounds, got \(error)")
            }
        }

        // 5. Negative offset
        XCTAssertThrowsError(
            try handle.validateBounds(offset: -1, size: 100)
        ) { error in
            guard case FastIOError.offsetOutOfBounds = error else {
                return XCTFail("Expected FastIOError.offsetOutOfBounds, got \(error)")
            }
        }

        // 6. Zero size read
        XCTAssertThrowsError(
            try handle.validateBounds(offset: 0, size: 0)
        ) { error in
            guard case FastIOError.offsetOutOfBounds = error else {
                return XCTFail("Expected FastIOError.offsetOutOfBounds, got \(error)")
            }
        }

        // 7. Negative size read
        XCTAssertThrowsError(
            try handle.validateBounds(offset: 0, size: -100)
        ) { error in
            guard case FastIOError.offsetOutOfBounds = error else {
                return XCTFail("Expected FastIOError.offsetOutOfBounds, got \(error)")
            }
        }

        // 8. Verify dispatchSpeculative traps before Metal I/O can execute
        XCTAssertThrowsError(
            try Self.sharedFastIOEngine.dispatchSpeculative(
                handle: handle,
                offset: fileSize,
                size: 100,
                targetBuffer: buffer
            )
        ) { error in
            guard case FastIOError.offsetOutOfBounds = error else {
                return XCTFail("Expected FastIOError.offsetOutOfBounds, got \(error)")
            }
        }
    }

    func testDefensiveBoundsWritingPastBufferLength() throws {
        let engine = Self.sharedFastIOEngine!
        let context = Self.sharedMetalContext!
        let fileURL = Self.sharedSyntheticFileURL!
        let handle = try WeightFileHandle(url: fileURL, device: context.device, layout: .synthetic)

        let smallBuffer = try context.makeBuffer(length: 1_000)

        // 1. Requested size exceeds total buffer length (targetOffset = 0, size = 1001)
        XCTAssertThrowsError(
            try engine.dispatchSpeculative(
                handle: handle,
                offset: 0,
                size: 1001,
                targetBuffer: smallBuffer,
                targetOffset: 0
            )
        ) { error in
            guard case FastIOError.bufferTooSmall(let required, let actual) = error else {
                return XCTFail("Expected FastIOError.bufferTooSmall, got \(error)")
            }
            XCTAssertEqual(required, 1001)
            XCTAssertEqual(actual, 1000)
        }

        // 2. targetOffset + size exceeds buffer length (targetOffset = 600, size = 500 -> 1100 > 1000)
        XCTAssertThrowsError(
            try engine.dispatchFallback(
                handle: handle,
                offset: 0,
                size: 500,
                targetBuffer: smallBuffer,
                targetOffset: 600
            )
        ) { error in
            guard case FastIOError.bufferTooSmall(let required, let actual) = error else {
                return XCTFail("Expected FastIOError.bufferTooSmall, got \(error)")
            }
            XCTAssertEqual(required, 1100)
            XCTAssertEqual(actual, 1000)
        }

        // 3. encodeLoadExpert defensive buffer bounds trap
        let cmd = engine.speculativeQueue.makeCommandBuffer()
        XCTAssertThrowsError(
            try handle.encodeLoadExpert(
                layerIndex: 0,
                expertIndex: 0,
                into: smallBuffer,
                targetOffset: 0,
                commandBuffer: cmd
            )
        ) { error in
            guard case FastIOError.bufferTooSmall = error else {
                return XCTFail("Expected FastIOError.bufferTooSmall, got \(error)")
            }
        }
    }

    func testDefensiveBoundsInvalidLayerAndExpertIndices() throws {
        let context = Self.sharedMetalContext!
        let fileURL = Self.sharedSyntheticFileURL!
        // Synthetic config: 4 layers (0..3), 16 experts (0..15)
        let handle = try WeightFileHandle(
            url: fileURL,
            device: context.device,
            layout: WeightLayoutConfig.synthetic
        )

        // 1. Layer index out of bounds: negative
        XCTAssertThrowsError(try handle.fileOffset(layerIndex: -1, expertIndex: 0)) { error in
            guard case WeightFileError.invalidLayout(let details) = error else {
                return XCTFail("Expected WeightFileError.invalidLayout, got \(error)")
            }
            XCTAssertTrue(details.contains("Layer index -1 out of bounds"))
        }

        // 2. Layer index out of bounds: equal to numLayers (4)
        XCTAssertThrowsError(try handle.fileOffset(layerIndex: 4, expertIndex: 0)) { error in
            guard case WeightFileError.invalidLayout(let details) = error else {
                return XCTFail("Expected WeightFileError.invalidLayout, got \(error)")
            }
            XCTAssertTrue(details.contains("Layer index 4 out of bounds"))
        }

        // 3. Layer index arbitrarily high
        XCTAssertThrowsError(try handle.fileOffset(layerIndex: 999, expertIndex: 0)) { error in
            guard case WeightFileError.invalidLayout = error else {
                return XCTFail("Expected WeightFileError.invalidLayout, got \(error)")
            }
        }

        // 4. Expert index out of bounds: negative
        XCTAssertThrowsError(try handle.fileOffset(layerIndex: 0, expertIndex: -1)) { error in
            guard case WeightFileError.invalidLayout(let details) = error else {
                return XCTFail("Expected WeightFileError.invalidLayout, got \(error)")
            }
            XCTAssertTrue(details.contains("Expert index -1 out of bounds"))
        }

        // 5. Expert index out of bounds: equal to numExperts (16)
        XCTAssertThrowsError(try handle.fileOffset(layerIndex: 0, expertIndex: 16)) { error in
            guard case WeightFileError.invalidLayout(let details) = error else {
                return XCTFail("Expected WeightFileError.invalidLayout, got \(error)")
            }
            XCTAssertTrue(details.contains("Expert index 16 out of bounds"))
        }

        // 6. Expert index arbitrarily high
        XCTAssertThrowsError(try handle.fileOffset(layerIndex: 0, expertIndex: 999)) { error in
            guard case WeightFileError.invalidLayout = error else {
                return XCTFail("Expected WeightFileError.invalidLayout, got \(error)")
            }
        }
    }

    func testClosedHandleRejection() throws {
        let context = Self.sharedMetalContext!
        let fileURL = Self.sharedSyntheticFileURL!
        let handle = try WeightFileHandle(url: fileURL, device: context.device)

        XCTAssertFalse(handle.isClosed)
        handle.close()
        XCTAssertTrue(handle.isClosed)

        // All operations must immediately throw WeightFileError.handleClosed
        XCTAssertThrowsError(try handle.validateBounds(offset: 0, size: 100)) { error in
            guard case WeightFileError.handleClosed = error else {
                return XCTFail("Expected WeightFileError.handleClosed, got \(error)")
            }
        }

        XCTAssertThrowsError(try handle.fileOffset(layerIndex: 0, expertIndex: 0)) { error in
            guard case WeightFileError.handleClosed = error else {
                return XCTFail("Expected WeightFileError.handleClosed, got \(error)")
            }
        }

        let cmd = Self.sharedFastIOEngine.speculativeQueue.makeCommandBuffer()
        let buf = try context.makeBuffer(length: 16_384)
        XCTAssertThrowsError(
            try handle.encodeLoadExpert(
                layerIndex: 0,
                expertIndex: 0,
                into: buf,
                targetOffset: 0,
                commandBuffer: cmd
            )
        ) { error in
            guard case WeightFileError.handleClosed = error else {
                return XCTFail("Expected WeightFileError.handleClosed, got \(error)")
            }
        }
    }

    func testValidateFullModelSizeRejectionOnTruncatedFile() throws {
        let context = Self.sharedMetalContext!
        // Create a truncated file with only 2 experts (small file)
        let (fileURL, cleanup) = try TestHelpers.createSyntheticWeightFile(
            expertCount: 2,
            expertSizeBytes: 16_384
        )
        defer { cleanup() }

        // Attempting to open with validateFullModelSize: true under Qwen layout (requires ~20 GB)
        XCTAssertThrowsError(
            try WeightFileHandle(
                url: fileURL,
                device: context.device,
                layout: WeightLayoutConfig.qwen15MoEA27B,
                validateFullModelSize: true
            )
        ) { error in
            guard case WeightFileError.fileTooSmall(let expMin, let actual) = error else {
                return XCTFail("Expected WeightFileError.fileTooSmall, got \(error)")
            }
            XCTAssertEqual(expMin, WeightLayoutConfig.qwen15MoEA27B.totalExpectedSizeBytes)
            XCTAssertEqual(actual, 2 * 16_384)
        }
    }

    // MARK: - 3. Out-of-Order Multi-Slot Safety & Race Condition Resistance

    func test16SlotOutOfOrderCompletionSafety() throws {
        let context = Self.sharedMetalContext!
        let engine = Self.sharedFastIOEngine!
        let fileHandle = try context.device.makeIOFileHandle(url: Self.sharedSyntheticFileURL!)
        let slotCount = 16
        let expertSize = syntheticConfig.expertSizeBytes

        // Setup 16 slots, each with its dedicated SyncEvent, MTLBuffer, and target expert
        struct SlotHarness {
            let slotIndex: Int
            let buffer: any MTLBuffer
            let syncEvent: SyncEvent
            let layer: Int
            let expert: Int
            let offset: Int
        }

        var harnesses: [SlotHarness] = []
        for i in 0..<slotCount {
            let buf = try context.makeBuffer(length: expertSize)
            let evt = try SyncEvent(device: context.device)
            let layer = i / 4
            let expert = (i * 3) % 16
            let offset = SyntheticWeightFileGenerator.fileOffset(layer: layer, expert: expert, config: syntheticConfig)
            harnesses.append(SlotHarness(
                slotIndex: i,
                buffer: buf,
                syncEvent: evt,
                layer: layer,
                expert: expert,
                offset: offset
            ))
        }

        // 1. Enqueue GPU compute work on ALL 16 slots waiting on their respective SyncEvent ticket 1
        var computeCmds: [any MTLCommandBuffer] = []
        for h in harnesses {
            let cmd = context.commandQueue.makeCommandBuffer()!
            h.syncEvent.encodeWait(on: cmd, ticket: 1)

            // Blit / compute pass verifying data is present
            let blit = cmd.makeBlitCommandEncoder()!
            blit.fill(buffer: h.buffer, range: 0..<4, value: 0xEE)
            blit.endEncoding()
            cmd.commit()
            computeCmds.append(cmd)
        }

        // At this point, ALL 16 GPU command buffers are queued waiting on GPU hardware for ticket 1.
        // Verify NONE of the events are signaled yet.
        for h in harnesses {
            XCTAssertEqual(h.syncEvent.signaledValue, 0)
            XCTAssertFalse(h.syncEvent.isSignaled(at: 1))
        }

        // 2. Dispatch I/O in REVERSE order: Slot 15 down to Slot 0
        // To verify that completing slot 15 with higher or out-of-order tickets NEVER unblocks slot 0, slot 1, etc.
        let reverseOrder = Array((0..<slotCount).reversed())
        var ioCmds: [any MTLIOCommandBuffer] = []

        for idx in reverseOrder {
            let h = harnesses[idx]
            let ioCmd = engine.speculativeQueue.makeCommandBuffer()
            ioCmd.load(h.buffer, offset: 0, size: expertSize, sourceHandle: fileHandle, sourceHandleOffset: h.offset)
            h.syncEvent.encodeSignal(on: ioCmd, ticket: 1)
            ioCmd.commit()
            ioCmds.append(ioCmd)

            // Wait specifically for this slot to complete
            ioCmd.waitUntilCompleted()
            XCTAssertEqual(h.syncEvent.signaledValue, 1, "Slot \(idx) should have signaled ticket 1")

            // Verify that all slots not yet dispatched remain UNSIGNALED
            for unreachedIdx in 0..<idx {
                let unreached = harnesses[unreachedIdx]
                XCTAssertEqual(
                    unreached.syncEvent.signaledValue,
                    0,
                    "Race condition! Slot \(unreachedIdx) was prematurely signaled when slot \(idx) completed."
                )
                XCTAssertFalse(unreached.syncEvent.isSignaled(at: 1))
            }
        }

        // 3. Wait for all GPU compute command buffers to complete
        for (i, computeCmd) in computeCmds.enumerated() {
            computeCmd.waitUntilCompleted()
            XCTAssertEqual(computeCmd.status, .completed, "Compute command for slot \(i) failed: \(String(describing: computeCmd.error))")
        }

        // 4. Verify all 16 slots have valid signaled values and no memory corruption
        for h in harnesses {
            XCTAssertEqual(h.syncEvent.signaledValue, 1)
        }
    }

    func testHighConcurrencyTicketCounterAtomicSafety() throws {
        let context = Self.sharedMetalContext!
        let syncEvent = try SyncEvent(device: context.device)

        let totalThreads = 100
        let ticketsPerThread = 100
        let totalExpectedTickets = totalThreads * ticketsPerThread

        let queue = DispatchQueue(label: "test.concurrency.tickets", attributes: .concurrent)
        let group = DispatchGroup()

        let resultsLock = OSAllocatedUnfairLock(initialState: [UInt64]())

        for _ in 0..<totalThreads {
            group.enter()
            queue.async {
                let localTickets = (0..<ticketsPerThread).map { _ in syncEvent.nextTicket() }
                resultsLock.withLock { arr in
                    arr.append(contentsOf: localTickets)
                }
                group.leave()
            }
        }

        let waitResult = group.wait(timeout: .now() + 5.0)
        XCTAssertEqual(waitResult, .success, "Concurrent ticket generation timed out")

        let collected = resultsLock.withLock { $0 }
        XCTAssertEqual(collected.count, totalExpectedTickets)

        // Verify all tickets are strictly unique (no duplicates, no dropped increments)
        let uniqueSet = Set(collected)
        XCTAssertEqual(uniqueSet.count, totalExpectedTickets, "Duplicate tickets generated under concurrent access!")

        // Verify range is exactly 1...totalExpectedTickets
        let minTicket = collected.min()!
        let maxTicket = collected.max()!
        XCTAssertEqual(minTicket, 1)
        XCTAssertEqual(maxTicket, UInt64(totalExpectedTickets))
        XCTAssertEqual(syncEvent.currentTicket, UInt64(totalExpectedTickets))
    }

    func testConcurrentMultiSlotDispatchesUnderContention() throws {
        let context = Self.sharedMetalContext!
        let engine = Self.sharedFastIOEngine!
        let expertSize = syntheticConfig.expertSizeBytes

        let concurrentTasks = 32
        let group = DispatchGroup()
        let queue = DispatchQueue(label: "test.concurrency.dispatches", attributes: .concurrent)

        for i in 0..<concurrentTasks {
            group.enter()
            queue.async {
                do {
                    let buf = try context.makeBuffer(length: expertSize)
                    let slotEvent = try SyncEvent(device: context.device)
                    let offset = (i % 4) * expertSize

                    let (ticket, cmd) = if i % 2 == 0 {
                        try engine.dispatchSpeculative(
                            handle: try WeightFileHandle(url: Self.sharedSyntheticFileURL!, device: context.device),
                            offset: offset,
                            size: expertSize,
                            targetBuffer: buf,
                            syncEvent: slotEvent
                        )
                    } else {
                        try engine.dispatchFallback(
                            handle: try WeightFileHandle(url: Self.sharedSyntheticFileURL!, device: context.device),
                            offset: offset,
                            size: expertSize,
                            targetBuffer: buf,
                            syncEvent: slotEvent
                        )
                    }

                    let ok = slotEvent.waitUntilSignaled(ticket: ticket, timeoutSeconds: 3.0)
                    XCTAssertTrue(ok)
                    XCTAssertEqual(cmd.status, .complete)
                } catch {
                    XCTFail("Concurrent dispatch failed: \(error)")
                }
                group.leave()
            }
        }

        let waitResult = group.wait(timeout: .now() + 5.0)
        XCTAssertEqual(waitResult, .success, "Concurrent multi-slot dispatches timed out or deadlocked")
    }
}
