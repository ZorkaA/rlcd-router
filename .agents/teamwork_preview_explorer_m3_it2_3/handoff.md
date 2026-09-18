# Handoff Report: Phase 2 Milestone 3 Iteration 2 — Explorer 3
## ExecutionLogTests Genuine Compilation & Production GPU Execution Remediation

**Explorer Agent**: `teamwork_preview_explorer_m3_it2_3` (Explorer 3)  
**Assigned Directory**: `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m3_it2_3`  
**Milestone**: Phase 2 Milestone 3 (GPU Execution Log & Dispatch-Time LRU — Requirement R3) Iteration 2  
**Date**: 2026-09-18  
**Parent Orchestrator**: `913b8328-6b64-4881-a075-c0057bc23d84`  
**Target File**: `swift_tests/AsyncMoERouterTests/Unit/ExecutionLogTests.swift`  

---

## 1. Observation

### 1.1 Forensic Auditor Findings in `ExecutionLogTests.swift`
In `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_auditor_m3_1/handoff.md`, the Forensic Auditor recorded a veto verdict (`INTEGRITY VIOLATION`) citing facade tests and test evasion in `ExecutionLogTests.swift`:
1. **Facade Syntax Check in `testExecutionLogMSLSource`** (`ExecutionLogTests.swift:689-698`):
   ```swift
   @Test("Execution log MSL shader source is valid metal syntax")
   func testExecutionLogMSLSource() {
       let src = GPUExecutionLog.mslKernelSource
       #expect(src.contains("kernel void writeExecutionLogEntry"))
       #expect(src.contains("device ExecutionLogEntry* log"))
       #expect(src.contains("logCapacity"))
       // Must not use atomics
       #expect(!src.contains("atomic_"))
   }
   ```
   **Observation**: The test titled `"Execution log MSL shader source is valid metal syntax"` performs zero syntax validation and zero Metal compilation. It performs only string containment checks (`src.contains(...)`), which pass even if the MSL shader source contains syntax errors, invalid Metal types, or malformed kernel signatures.

2. **Synthetic Shader Substitution Concealing Production Kernel Incompatibility**:
   In `testGPUExecutionLogKernelWraparound` (lines 240-287) and `testSyntheticGatingAndLogKernelCompilation` (lines 589-639), unit tests bypass production kernels in `ExecutionLog.mslKernelSource` (`writeExecutionLogEntry`, `writeTokenGatingLog`, `writeExecutionLogBatch`) and substitute synthetic shaders (`parallelLogWriterSource`, `mockGatingAndLogSource`) using contiguous sequential indexing (`slot = token % logCapacity`). Consequently, no unit test in `ExecutionLogTests.swift` ever executed the production MSL kernels on GPU or verified that `ExecutionLog.drain()` could drain entries written by `writeExecutionLogEntry`.

### 1.2 Empirical Failure in `ExecutionLogLRUAdversarialTests.swift`
When executing `swift test --filter ExecutionLogLRUAdversarialTests`:
```text
✘ Test "Adversarial 10: Prove whether out-of-order older entries corrupt LRU eviction victim selection" recorded an issue at ExecutionLogLRUAdversarialTests.swift:477:9: Expectation failed: orderPreserved
↳ CRITICAL: Monotonic guard failed to protect recency queue! Out-of-order entry with older timestamp (500) promoted Expert A over Expert B (ts=2000), causing Expert B to become the eviction victim!
✘ Test "Adversarial 11: Prove whether sparse/gapped slot indexing blocks CPU drain()" recorded an issue at ExecutionLogLRUAdversarialTests.swift:509:9: Expectation failed: foundEntry
↳ CRITICAL: ExecutionLog.drain() stopped at empty slot 0 and failed to drain valid entry written at slot 21 by writeExecutionLogEntry!
```
- **Test 10 Failure**: In `LRUWeightTracker.swift:95-108`, `Self._unlink(node: node)` and `Self._insertAfterHead(node: node, head: state.head)` execute unconditionally even when `timestamp < node.timestamp`.
- **Test 11 Failure**: In `ExecutionLog.swift:157-164`, `drain()` encounters empty slot 0, evaluates `guard entry.tokenIndex != 0 || entry.timestamp != 0 || entry.confidenceScore != 0 else { break }`, breaks immediately, and strands valid entries written at non-zero slots (e.g., slot 21).

### 1.3 Empirical Hardware Compilation & Execution of Production MSL Kernels
We executed an empirical Metal 3 compilation and dispatch probe on the Apple Silicon GPU using `GPUExecutionLog.mslKernelSource`:
1. `device.makeLibrary(source: GPUExecutionLog.mslKernelSource, options: nil)` compiled successfully with 0 errors, exposing functions `["writeExecutionLogBatch", "writeTokenGatingLog", "writeExecutionLogEntry"]`.
2. `device.makeComputePipelineState(function:)` succeeded for all 3 kernels with `maxTotalThreadsPerThreadgroup == 1024`.
3. GPU dispatch of `writeExecutionLogEntry` for `tokenIndex = 42, layerIndex = 5, horizonIndex = 1, expertID = 17, confidence = 0.94, timestamp = 1234567` computed slot `(42 * 80 + 5 * 4 + 1) % 4096 = 3381` and wrote exact 32-byte fields into unified memory.
4. GPU dispatch of `writeTokenGatingLog` (topK = 4) wrote parallel expert entries into consecutive slots `[40, 41, 42, 43]`.
5. GPU dispatch of `writeExecutionLogBatch` (numExperts = 3, baseSlot = 128) wrote consecutive entries into slots `[128, 129, 130]`.

---

## 2. Logic Chain

1. **Premise 1 (Remediation of Facade String Checks)**:
   A test asserting MSL syntax validity must invoke the Apple Metal compiler runtime (`device.makeLibrary(source:options:)` and `MetalContext.shared.makeComputePipelineState(...)`). String containment is an anti-pattern that conceals compilation regressions.
2. **Premise 2 (Mandatory Zero-Atomic Invariant)**:
   Requirement R3 strictly forbids GPU atomic timestamp updates or slot counters (`atomic_fetch_add_explicit`). The test suite must assert both positive compilation of all 3 production kernels AND negative absence of atomic keywords (`!src.contains("atomic_")` and `!src.contains("atomic_fetch_add")`).
3. **Premise 3 (Direct Production Kernel GPU Execution)**:
   To ensure the pipeline is genuine and eliminate the gap flagged by Check 2, unit tests must dispatch the exact production MSL kernels (`writeExecutionLogEntry`, `writeTokenGatingLog`, `writeExecutionLogBatch`) on GPU and verify that CPU `drain()` successfully extracts the records, validates all 8 fields of `ExecutionLogEntry`, and confirms that drained slots are zeroed.
4. **Premise 4 (Full Test Suite Interoperability & 100% Pass Rate)**:
   - When Explorer 1's monotonic guard fix is applied to `LRUWeightTracker.swift`, Test 10 passes.
   - When Spec Miner 2's full-capacity drain scanning fix is applied to `ExecutionLog.swift`, Test 11 passes, sparse entries (e.g. slot 21 or slot 3381) are drained without truncation, and `state.totalEntriesDrained` increments accurately by `entries.count`.
   - All 6 adversarial stress tests in `ExecutionLogChallenger2StressTests.swift` pass (validated in 0.084s).
   - Therefore, the remediated `ExecutionLogTests.swift` provides full coverage without introducing regressions into existing unit, adversarial, or stress suites.

---

## 3. Caveats

- **Device Requirement**: Unit tests requiring Metal compilation and GPU execution require an Apple Silicon Metal device (`MTLCreateSystemDefaultDevice() != nil`). When run in environments without Metal GPU support, `init() throws` cleanly skips with `TestError.metalUnavailable`.
- **Worker Execution Dependency**: The unit tests executing `writeExecutionLogEntry` on GPU with non-zero layer indices write to sparse slots (e.g., slot 21 or 3381). Draining these entries depends on `ExecutionLog.drain()` scanning across capacity without prematurely breaking at empty slot 0 (as authored by Spec Miner 2 and implemented by the worker).
- No other caveats.

---

## 4. Conclusion

`ExecutionLogTests.swift` is completely remediated to eliminate all facade string checks and provide comprehensive, genuine production GPU execution testing for Phase 2 Milestone 3:
1. `testExecutionLogMSLSource` now compiles `GPUExecutionLog.mslKernelSource` at runtime via `device.makeLibrary` and compiles pipeline states for `writeExecutionLogEntry`, `writeTokenGatingLog`, and `writeExecutionLogBatch` via `MetalContext.shared`.
2. Four new production GPU kernel unit tests are added:
   - `testProductionWriteExecutionLogEntryDirectExecution`: GPU execution of single `writeExecutionLogEntry` and drain verification.
   - `testProductionWriteExecutionLogEntryMultipleEntriesAndRecencyDrain`: GPU execution of multiple entries across layers/tokens, drained into `LRUWeightTracker` with LRU eviction candidate verification.
   - `testProductionWriteTokenGatingLogDirectExecution`: GPU execution of parallel `writeTokenGatingLog` for top-K routed experts and drain verification.
   - `testProductionWriteExecutionLogBatchDirectExecution`: GPU execution of `writeExecutionLogBatch` with base slot offset and drain verification.
3. All 15 existing tests (layout, alignment, wraparound, canary protection, budget calculations, and Requirement R3 pre-routing isolation) are preserved 100%.
4. Together with Explorer 1's and Spec Miner 2's blueprints, 100% of tests across all suites (`ExecutionLogTests`, `ExecutionLogLRUAdversarialTests`, `ExecutionLogChallenger2StressTests`) will pass cleanly.

---

## 5. Verification Method

### 5.1 Verification Commands
Run the targeted test suites in order:
```bash
# 1. Build project
swift build

# 2. Run remediated ExecutionLogTests (19 tests)
swift test --filter ExecutionLogTests

# 3. Run Challenger 1 Adversarial Tests (11 tests)
swift test --filter ExecutionLogLRUAdversarialTests

# 4. Run Challenger 2 Stress Tests (6 tests)
swift test --filter ExecutionLogChallenger2StressTests

# 5. Run full test suite regression
swift test
```

### 5.2 Success Criteria
1. `ExecutionLogTests` runs 19 tests with 0 failures, confirming genuine Metal MSL compilation and GPU execution of all 3 production kernels.
2. `ExecutionLogLRUAdversarialTests` runs 11 tests with 0 failures (Tests 10 and 11 pass).
3. `ExecutionLogChallenger2StressTests` runs 6 tests with 0 failures.
4. Total test run passes 100% with exit code 0.

---

## 6. Complete Production Blueprint for `ExecutionLogTests.swift`

Target path: `swift_tests/AsyncMoERouterTests/Unit/ExecutionLogTests.swift`

```swift
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

/// Synthetic MSL shader sources for Milestone 3 runtime execution and stress tests.
public enum SyntheticExecutionLogShaders {
    /// Realistic synthetic gating and execution logging kernel.
    /// Computes deterministic expert routing and calibrated confidence without atomics,
    /// writing a 32-byte ExecutionLogEntry directly to the circular execution log.
    public static let mockGatingAndLogSource: String = """
    #include <metal_stdlib>
    using namespace metal;

    struct ExecutionLogEntry {
        uint   tokenIndex;
        ushort layerIndex;
        ushort horizonIndex;
        ushort expertID;
        ushort padding;
        float  confidenceScore;
        ulong  timestamp;
        ulong  reserved;
    };

    kernel void syntheticGatingAndLog(
        device ExecutionLogEntry* logBuffer     [[buffer(0)]],
        constant uint&            logCapacity   [[buffer(1)]],
        constant uint&            tokenOffset   [[buffer(2)]],
        constant ushort&          layerIndex    [[buffer(3)]],
        constant ushort&          horizonIndex  [[buffer(4)]],
        constant uint&            numExperts    [[buffer(5)]],
        constant ulong&           baseTimestamp [[buffer(6)]],
        uint                      tid           [[thread_position_in_grid]]
    ) {
        uint globalToken = tokenOffset + tid;
        // Zero-atomic deterministic slot assignment via token stripe
        uint slot = globalToken % logCapacity;

        // Deterministic synthetic gating decision
        ushort expertID = (ushort)((globalToken * 7u + layerIndex * 13u + horizonIndex * 3u) % numExperts);
        // Calibrated confidence score in [0.50, 0.99]
        float confidence = 0.50f + 0.49f * fract(sin((float)globalToken * 12.9898f + (float)layerIndex * 78.233f) * 43758.5453f);

        device ExecutionLogEntry& entry = logBuffer[slot];
        entry.tokenIndex      = globalToken;
        entry.layerIndex      = layerIndex;
        entry.horizonIndex    = horizonIndex;
        entry.expertID        = expertID;
        entry.padding         = 0;
        entry.confidenceScore = confidence;
        entry.timestamp       = baseTimestamp + (ulong)tid * 10ULL;
        entry.reserved        = 0xCAFEBABEDEADBEEFULL;
    }
    """

    /// High-throughput parallel log writer for wraparound and concurrent stress testing.
    public static let parallelLogWriterSource: String = """
    #include <metal_stdlib>
    using namespace metal;

    struct ExecutionLogEntry {
        uint   tokenIndex;
        ushort layerIndex;
        ushort horizonIndex;
        ushort expertID;
        ushort padding;
        float  confidenceScore;
        ulong  timestamp;
        ulong  reserved;
    };

    kernel void parallelLogWriter(
        device ExecutionLogEntry* logBuffer     [[buffer(0)]],
        constant uint&            logCapacity   [[buffer(1)]],
        constant uint&            startToken    [[buffer(2)]],
        constant ulong&           baseTimestamp [[buffer(3)]],
        constant uint&            expertBase    [[buffer(4)]],
        uint                      tid           [[thread_position_in_grid]]
    ) {
        uint token = startToken + tid;
        uint slot = token % logCapacity;

        device ExecutionLogEntry& entry = logBuffer[slot];
        entry.tokenIndex      = token;
        entry.layerIndex      = (ushort)(5 + (tid % 20));
        entry.horizonIndex    = (ushort)(1 + (tid % 3));
        entry.expertID        = (ushort)((expertBase + tid) % 60);
        entry.padding         = 0;
        entry.confidenceScore = 0.88f;
        entry.timestamp       = baseTimestamp + (ulong)tid;
        entry.reserved        = 0x55AA55AA00000000ULL | (ulong)tid;
    }
    """
}

/// Comprehensive unit test suite for Milestone 3 (Phase 2 Requirement R3):
/// - Exact 32-byte layout and field offset verification for `ExecutionLogEntry`
/// - 4,096-entry circular wraparound and zero-atomic bitwise indexing math
/// - Post-execution CPU log draining into `LRUWeightTracker`
/// - Strict verification that pre-routing predictions do NOT modify LRU timestamps (Requirement R3)
/// - Concurrent GPU write and CPU drain stress testing with canary bounds verification
/// - Genuine runtime MSL compilation of production kernels (`writeExecutionLogEntry`, `writeTokenGatingLog`, `writeExecutionLogBatch`)
/// - Direct GPU execution and drain verification for all production MSL kernels
@Suite("Execution Log Tests")
struct ExecutionLogTests {
    let device: any MTLDevice

    init() throws {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            throw TestError.metalUnavailable
        }
        self.device = dev
    }

    enum TestError: Error {
        case metalUnavailable
        case computeEncoderCreationFailed
        case bufferAllocationFailed
    }

    // MARK: - 1. 32-Byte Layout & Alignment Verification

    @Test("ExecutionLogEntry exact 32-byte size, stride, alignment, and field offsets")
    func testExecutionLogEntryAlignmentAndLayout() {
        // Assert struct byte dimensions
        #expect(MemoryLayout<ExecutionLogEntry>.size == 32, "ExecutionLogEntry size must be exactly 32 bytes")
        #expect(MemoryLayout<ExecutionLogEntry>.stride == 32, "ExecutionLogEntry stride must be exactly 32 bytes")
        #expect(MemoryLayout<ExecutionLogEntry>.alignment == 8, "ExecutionLogEntry alignment must be 8 bytes (due to UInt64)")

        // Assert field offsets matching C / MSL struct layout
        #expect(MemoryLayout<ExecutionLogEntry>.offset(of: \.tokenIndex) == 0, "tokenIndex must be at byte offset 0")
        #expect(MemoryLayout<ExecutionLogEntry>.offset(of: \.layerIndex) == 4, "layerIndex must be at byte offset 4")
        #expect(MemoryLayout<ExecutionLogEntry>.offset(of: \.horizonIndex) == 6, "horizonIndex must be at byte offset 6")
        #expect(MemoryLayout<ExecutionLogEntry>.offset(of: \.expertID) == 8, "expertID must be at byte offset 8")
        #expect(MemoryLayout<ExecutionLogEntry>.offset(of: \.padding) == 10, "padding must be at byte offset 10")
        #expect(MemoryLayout<ExecutionLogEntry>.offset(of: \.confidenceScore) == 12, "confidenceScore must be at byte offset 12")
        #expect(MemoryLayout<ExecutionLogEntry>.offset(of: \.timestamp) == 16, "timestamp must be at byte offset 16")
        #expect(MemoryLayout<ExecutionLogEntry>.offset(of: \.reserved) == 24, "reserved must be at byte offset 24")

        // Bitwise serialization test
        let rawBuffer = UnsafeMutableRawPointer.allocate(byteCount: 32, alignment: 8)
        defer { rawBuffer.deallocate() }
        rawBuffer.initializeMemory(as: UInt8.self, repeating: 0xFF, count: 32)

        let entryPtr = rawBuffer.bindMemory(to: ExecutionLogEntry.self, capacity: 1)
        entryPtr.pointee = ExecutionLogEntry(
            tokenIndex: 0x12345678,
            layerIndex: 0x0506,
            horizonIndex: 0x0102,
            expertID: 0x003B, // 59
            confidenceScore: 0.75,
            timestamp: 0x0102030405060708,
            reserved: 0xAABBCCDDEEFF0011
        )

        // Validate byte serialization
        let bytePtr = rawBuffer.bindMemory(to: UInt8.self, capacity: 32)
        // Check tokenIndex (0x12345678 in little-endian: 78 56 34 12)
        #expect(bytePtr[0] == 0x78)
        #expect(bytePtr[1] == 0x56)
        #expect(bytePtr[2] == 0x34)
        #expect(bytePtr[3] == 0x12)
        // Check padding is zeroed
        #expect(bytePtr[10] == 0x00)
        #expect(bytePtr[11] == 0x00)
        // Check float 0.75 in IEEE-754 (0x3F400000 in little-endian: 00 00 40 3F)
        #expect(bytePtr[12] == 0x00)
        #expect(bytePtr[13] == 0x00)
        #expect(bytePtr[14] == 0x40)
        #expect(bytePtr[15] == 0x3F)
        // Check timestamp (little endian: 08 07 06 05 04 03 02 01)
        #expect(bytePtr[16] == 0x08)
        #expect(bytePtr[23] == 0x01)
    }

    // MARK: - 2. 4,096-Entry Circular Wraparound & Indexing Math

    @Test("Zero-atomic circular log buffer 5,000 entries wraparound and indexing math")
    func testZeroAtomicCircularLogBufferWrap() throws {
        let capacity = 4096
        let totalEntries = 5000
        let entrySize = MemoryLayout<ExecutionLogEntry>.stride
        let totalBufferBytes = capacity * entrySize

        // Allocate buffer with 64-byte guard canary memory past capacity
        let canarySize = 64
        guard let mtlBuffer = device.makeBuffer(length: totalBufferBytes + canarySize, options: .storageModeShared) else {
            Issue.record("Failed to allocate test buffer with canary")
            return
        }
        // Fill canary area with 0xAA
        let canaryPtr = mtlBuffer.contents().advanced(by: totalBufferBytes).bindMemory(to: UInt8.self, capacity: canarySize)
        for i in 0..<canarySize {
            canaryPtr[i] = 0xAA
        }

        let entryPtr = mtlBuffer.contents().bindMemory(to: ExecutionLogEntry.self, capacity: capacity)
        memset(mtlBuffer.contents(), 0, totalBufferBytes)

        // Write 5,000 entries into 4,096 circular buffer
        for i in 0..<totalEntries {
            let slot = i % capacity
            #expect(slot >= 0 && slot < capacity, "Slot index must be within [0, 4095]")
            entryPtr[slot] = ExecutionLogEntry(
                tokenIndex: UInt32(i),
                layerIndex: UInt16(5 + (i % 20)),
                horizonIndex: UInt16(1 + (i % 3)),
                expertID: UInt16(i % 60),
                confidenceScore: 0.9,
                timestamp: UInt64(1000 + i)
            )
        }

        // Verify clean circular indexing state:
        // Entries 0..<904 were overwritten by entries 4096..<5000.
        // Slots 0..<904 must contain tokens 4096..<5000.
        for slot in 0..<904 {
            let expectedToken = UInt32(4096 + slot)
            #expect(entryPtr[slot].tokenIndex == expectedToken, "Slot \(slot) should have wrapped token \(expectedToken)")
            #expect(entryPtr[slot].timestamp == UInt64(1000 + 4096 + slot))
        }

        // Slots 904..<4096 must contain first-pass tokens 904..<4096.
        for slot in 904..<4096 {
            let expectedToken = UInt32(slot)
            #expect(entryPtr[slot].tokenIndex == expectedToken, "Slot \(slot) should have retained token \(expectedToken)")
            #expect(entryPtr[slot].timestamp == UInt64(1000 + slot))
        }

        // Verify canary guard region was NOT corrupted (zero out-of-bounds writes)
        for i in 0..<canarySize {
            #expect(canaryPtr[i] == 0xAA, "Canary corrupted at offset \(i) — out of bounds write detected!")
        }
    }

    @Test("GPU runtime kernel circular wraparound via MetalContext")
    func testGPUExecutionLogKernelWraparound() throws {
        let ctx = MetalContext.shared
        let pipeline = try ctx.makeComputePipelineState(
            source: SyntheticExecutionLogShaders.parallelLogWriterSource,
            functionName: "parallelLogWriter"
        )

        let capacity = 4096
        let log = GPUExecutionLog(device: ctx.device, capacity: capacity)

        func dispatchBatch(startToken: UInt32, count: Int, baseTimestamp: UInt64) {
            var sToken = startToken
            var bTime = baseTimestamp
            var expBase: UInt32 = 0
            var cap = UInt32(capacity)

            guard let cmdBuf = ctx.commandQueue.makeCommandBuffer(),
                  let enc = cmdBuf.makeComputeCommandEncoder() else {
                Issue.record("Failed to create Metal command encoder")
                return
            }
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(log.buffer, offset: 0, index: 0)
            enc.setBytes(&cap, length: 4, index: 1)
            enc.setBytes(&sToken, length: 4, index: 2)
            enc.setBytes(&bTime, length: 8, index: 3)
            enc.setBytes(&expBase, length: 4, index: 4)

            let gridSize = MTLSize(width: count, height: 1, depth: 1)
            let threadgroupSize = MTLSize(width: min(pipeline.maxTotalThreadsPerThreadgroup, count), height: 1, depth: 1)
            enc.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }

        // Pass 1: Write 4,096 tokens (0..<4096)
        dispatchBatch(startToken: 0, count: 4096, baseTimestamp: 10_000)
        // Pass 2: Write 904 tokens (4096..<5000), wrapping around into slots 0..<904
        dispatchBatch(startToken: 4096, count: 904, baseTimestamp: 50_000)

        let ptr = log.buffer.contents().bindMemory(to: ExecutionLogEntry.self, capacity: capacity)
        #expect(ptr[0].tokenIndex == 4096)
        #expect(ptr[903].tokenIndex == 4999)
        #expect(ptr[904].tokenIndex == 904)
        #expect(ptr[4095].tokenIndex == 4095)
    }

    // MARK: - 3. Post-Execution CPU Log Draining & LRU Weight Tracker

    @Test("Post-execution CPU log draining integrates with LRUWeightTracker and resets slots")
    func testCPULogDrainPostExecution() {
        let capacity = 512
        let log = GPUExecutionLog(device: device, capacity: capacity)
        let tracker = LRUWeightTracker()

        let ptr = log.buffer.contents().bindMemory(to: ExecutionLogEntry.self, capacity: capacity)

        // Write batch of 64 entries
        let batch1Count = 64
        for i in 0..<batch1Count {
            ptr[i] = ExecutionLogEntry(
                tokenIndex: UInt32(i + 1),
                layerIndex: UInt16(5 + (i % 4)),
                horizonIndex: UInt16(1 + (i % 3)),
                expertID: UInt16(i % 16),
                confidenceScore: 0.85,
                timestamp: UInt64(10_000 + i * 100)
            )
        }

        // Drain batch 1
        let drained1 = log.drain()
        #expect(drained1.count == batch1Count)
        #expect(log.totalEntriesDrained == batch1Count)

        // Verify drained slots are zeroed out (idempotent second drain)
        let drainedEmpty = log.drain()
        #expect(drainedEmpty.isEmpty, "Second drain must return empty array")

        // Update LRU tracker with drained entries
        for entry in drained1 {
            let key = ExpertKey(layer: Int(entry.layerIndex), expert: Int(entry.expertID))
            tracker.recordAccess(expert: key, timestamp: entry.timestamp)
        }

        #expect(tracker.count == 16, "16 unique experts should be resident in tracker")
        #expect(tracker.totalUpdates == batch1Count)

        // LRU expert must be the one with the lowest timestamp
        guard let lru = tracker.lruExpert else {
            Issue.record("Expected resident LRU expert")
            return
        }
        #expect(lru == ExpertKey(layer: 5, expert: 0), "L5E0 was accessed earliest at ts=10000")

        // Write batch 2 (32 entries starting at slot 64)
        let batch2Count = 32
        for i in 0..<batch2Count {
            let slot = 64 + i
            ptr[slot] = ExecutionLogEntry(
                tokenIndex: UInt32(100 + i),
                layerIndex: 5,
                horizonIndex: 1,
                expertID: UInt16(i % 4),
                confidenceScore: 0.92,
                timestamp: UInt64(50_000 + i * 100)
            )
        }

        let drained2 = log.drain()
        #expect(drained2.count == batch2Count)
        #expect(log.totalEntriesDrained == batch1Count + batch2Count)

        // Update tracker with batch 2
        for entry in drained2 {
            let key = ExpertKey(layer: Int(entry.layerIndex), expert: Int(entry.expertID))
            tracker.recordAccess(expert: key, timestamp: entry.timestamp)
        }

        // L5E0 timestamp was updated to 50,000, so it is no longer the LRU victim!
        let newLRU = tracker.lruExpert
        #expect(newLRU != ExpertKey(layer: 5, expert: 0), "L5E0 was refreshed and cannot be LRU victim")
    }

    // MARK: - 4. Strict Requirement R3 Post-Execution LRU Invariant Verification

    @Test("Strict Requirement R3: Speculative pre-routing does NOT update LRU timestamps, only log drain does")
    func testLRUUpdatedStrictlyPostExecution() {
        let slotCount = 8
        let slotSizeBytes = 4096
        let ring = SpeculativeRingBuffer(device: device, slotCount: slotCount, slotSizeBytes: slotSizeBytes)
        let tracker = LRUWeightTracker()
        let log = GPUExecutionLog(device: device, capacity: 128)

        // Step 1: Pre-routing speculative prefetch phase
        // Router speculates 4 candidate experts: L5E0, L5E1, L5E2, L5E3
        var allocatedSlots: [RingBufferSlot] = []
        for e in 0..<4 {
            let expert = ExpertKey(layer: 5, expert: e)
            guard let slot = ring.allocateSlot(for: expert, ticket: UInt64(e + 1)) else {
                Issue.record("Failed to allocate speculative slot for \(expert)")
                continue
            }
            allocatedSlots.append(slot)
            // Complete DMA
            ring.markReady(slotIndex: slot.index, ticket: UInt64(e + 1))
            // Bind to compute encoder
            ring.markInUse(slotIndex: slot.index, ticket: UInt64(e + 1))
            // Release from compute encoder
            ring.releaseFromUse(slotIndex: slot.index, ticket: UInt64(e + 1))
        }

        // ASSERT THE INVARIANT:
        // Pre-routing predictions, loading, and readying MUST NOT modify slot lastAccessedTimestamp!
        for slot in allocatedSlots {
            #expect(slot.lastAccessedTimestamp == 0, "Slot \(slot.index) lastAccessedTimestamp must remain 0 during pre-routing")
        }

        // LRUWeightTracker MUST have zero knowledge of pre-routed experts
        #expect(tracker.count == 0, "Tracker must have 0 resident experts before execution log drain")
        #expect(tracker.lruExpert == nil, "Tracker must have no LRU victim")
        for e in 0..<4 {
            let expert = ExpertKey(layer: 5, expert: e)
            #expect(!tracker.contains(expert: expert), "Tracker must NOT contain pre-routed \(expert)")
        }

        // Step 2: Speculative branch cancellation / abandonment
        // Speculative prefetch for L6E10 that is never dispatched
        guard let cancelledSlot = ring.allocateSlot(for: ExpertKey(layer: 6, expert: 10), ticket: 99) else {
            Issue.record("Failed to allocate cancelled slot")
            return
        }
        ring.markAbandoned(slotIndex: cancelledSlot.index)
        ring.reclaim(slotIndex: cancelledSlot.index)
        #expect(cancelledSlot.lastAccessedTimestamp == 0)
        #expect(!tracker.contains(expert: ExpertKey(layer: 6, expert: 10)))

        // Step 3: GPU Execution Log commits ONLY actually executed decisions
        // Suppose native gating actually selected ONLY L5E1 and L5E3 (L5E0 and L5E2 were bypassed)
        let logPtr = log.buffer.contents().bindMemory(to: ExecutionLogEntry.self, capacity: log.capacity)
        logPtr[0] = ExecutionLogEntry(
            tokenIndex: 1, layerIndex: 5, horizonIndex: 1, expertID: 1,
            confidenceScore: 0.95, timestamp: 100_000
        )
        logPtr[1] = ExecutionLogEntry(
            tokenIndex: 2, layerIndex: 5, horizonIndex: 1, expertID: 3,
            confidenceScore: 0.98, timestamp: 200_000
        )

        // Step 4: CPU Drains Execution Log post-execution
        let drained = log.drain()
        #expect(drained.count == 2)

        for entry in drained {
            let expert = ExpertKey(layer: Int(entry.layerIndex), expert: Int(entry.expertID))
            tracker.recordAccess(expert: expert, timestamp: entry.timestamp)
            if let slot = ring.findReadySlot(for: expert) {
                ring.updateLRUTimestamp(slotIndex: slot.index, timestamp: entry.timestamp)
            }
        }

        // Verify post-drain invariants:
        // ONLY L5E1 and L5E3 have non-zero timestamps
        let readySlot1 = ring.findReadySlot(for: ExpertKey(layer: 5, expert: 1))
        let readySlot3 = ring.findReadySlot(for: ExpertKey(layer: 5, expert: 3))
        let readySlot0 = ring.findReadySlot(for: ExpertKey(layer: 5, expert: 0))
        let readySlot2 = ring.findReadySlot(for: ExpertKey(layer: 5, expert: 2))

        #expect(readySlot1?.lastAccessedTimestamp == 100_000)
        #expect(readySlot3?.lastAccessedTimestamp == 200_000)
        #expect(readySlot0?.lastAccessedTimestamp == 0, "Unexecuted speculative slot must remain at timestamp 0")
        #expect(readySlot2?.lastAccessedTimestamp == 0, "Unexecuted speculative slot must remain at timestamp 0")

        #expect(tracker.count == 2)
        #expect(tracker.contains(expert: ExpertKey(layer: 5, expert: 1)))
        #expect(tracker.contains(expert: ExpertKey(layer: 5, expert: 3)))
        #expect(!tracker.contains(expert: ExpertKey(layer: 5, expert: 0)))
        #expect(!tracker.contains(expert: ExpertKey(layer: 5, expert: 2)))

        // Step 5: LRU Eviction ordering preference
        // When ring buffer saturates, allocateSlot must evict slots with timestamp 0 before slots with timestamp > 0!
        for i in 4..<8 {
            let expert = ExpertKey(layer: 7, expert: i)
            let s = ring.allocateSlot(for: expert, ticket: UInt64(10 + i))
            #expect(s != nil)
            ring.markReady(slotIndex: s!.index, ticket: UInt64(10 + i))
        }

        // All 8 slots are now .ready. The next allocation MUST evict a slot with lastAccessedTimestamp == 0!
        let newExpert = ExpertKey(layer: 8, expert: 42)
        guard let victimSlot = ring.allocateSlot(for: newExpert, ticket: 100) else {
            Issue.record("Ring buffer eviction failed")
            return
        }
        // Victim MUST have been an unexecuted slot (timestamp 0), NOT slot 1 or 3!
        #expect(victimSlot.index != readySlot1?.index, "Executed slot 1 must not be evicted ahead of unexecuted slots")
        #expect(victimSlot.index != readySlot3?.index, "Executed slot 3 must not be evicted ahead of unexecuted slots")
    }

    // MARK: - 5. Concurrent GPU Write & CPU Drain Stress Testing

    @Test("Concurrent GPU parallel writes and CPU drain stress testing with zero data corruption")
    func testConcurrentLogWriteAndDrain() async throws {
        let ctx = MetalContext.shared
        let pipeline = try ctx.makeComputePipelineState(
            source: SyntheticExecutionLogShaders.parallelLogWriterSource,
            functionName: "parallelLogWriter"
        )

        let capacity = 4096
        let log = GPUExecutionLog(device: ctx.device, capacity: capacity)
        let tracker = LRUWeightTracker()

        let numBatches = 25
        let batchSize = 64
        let totalExpected = numBatches * batchSize

        let stateLock = OSAllocatedUnfairLock(initialState: (drained: [ExecutionLogEntry](), finished: false))

        // Background GPU Dispatch Worker
        let gpuTask = Task.detached(priority: .userInitiated) {
            for b in 0..<numBatches {
                var startToken = UInt32(b * batchSize)
                var baseTime = UInt64(b * 10_000 + 100)
                var expBase = UInt32(b * 2)
                var cap = UInt32(capacity)

                guard let cmdBuf = ctx.commandQueue.makeCommandBuffer(),
                      let enc = cmdBuf.makeComputeCommandEncoder() else {
                    break
                }
                enc.setComputePipelineState(pipeline)
                enc.setBuffer(log.buffer, offset: 0, index: 0)
                enc.setBytes(&cap, length: 4, index: 1)
                enc.setBytes(&startToken, length: 4, index: 2)
                enc.setBytes(&baseTime, length: 8, index: 3)
                enc.setBytes(&expBase, length: 4, index: 4)

                let gridSize = MTLSize(width: batchSize, height: 1, depth: 1)
                let threadgroupSize = MTLSize(width: min(pipeline.maxTotalThreadsPerThreadgroup, batchSize), height: 1, depth: 1)
                enc.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
                enc.endEncoding()
                cmdBuf.commit()
                cmdBuf.waitUntilCompleted()

                try? await Task.sleep(nanoseconds: 150_000) // 150 us
            }

            stateLock.withLock { $0.finished = true }
        }

        // Concurrent CPU Drain Worker
        let cpuTask = Task.detached(priority: .medium) {
            while true {
                let batch = log.drain()
                if !batch.isEmpty {
                    stateLock.withLock { $0.drained.append(contentsOf: batch) }
                    for e in batch {
                        let key = ExpertKey(layer: Int(e.layerIndex), expert: Int(e.expertID))
                        tracker.recordAccess(expert: key, timestamp: e.timestamp)
                    }
                }

                let done = stateLock.withLock { $0.finished }
                if done { break }
                try? await Task.sleep(nanoseconds: 100_000) // 100 us
            }

            // Final drain sweeps to drain any residual entries
            for _ in 0..<5 {
                let batch = log.drain()
                if !batch.isEmpty {
                    stateLock.withLock { $0.drained.append(contentsOf: batch) }
                    for e in batch {
                        let key = ExpertKey(layer: Int(e.layerIndex), expert: Int(e.expertID))
                        tracker.recordAccess(expert: key, timestamp: e.timestamp)
                    }
                }
                try? await Task.sleep(nanoseconds: 50_000) // 50 us
            }
        }

        _ = await gpuTask.value
        _ = await cpuTask.value

        let drainedEntries = stateLock.withLock { $0.drained }

        // Assert all entries were drained with zero loss
        #expect(drainedEntries.count == totalExpected)

        // Assert zero data corruption across drained entries
        for entry in drainedEntries {
            #expect(entry.confidenceScore == 0.88)
            #expect(entry.layerIndex >= 5 && entry.layerIndex <= 24)
            #expect(entry.horizonIndex >= 1 && entry.horizonIndex <= 3)
            #expect(entry.expertID < 60)
            #expect(entry.padding == 0)
            #expect((entry.reserved & 0xFFFFFFFF00000000) == 0x55AA55AA00000000)
        }

        #expect(tracker.totalUpdates == totalExpected)
        #expect(tracker.count > 0 && tracker.count <= 1200)
        #expect(tracker.lruExpert != nil)
    }

    // MARK: - 6. Synthetic MSL Gating & Logging Kernel Compilation

    @Test("Synthetic MSL gating and logging kernel compiles cleanly via MetalContext")
    func testSyntheticGatingAndLogKernelCompilation() throws {
        let ctx = MetalContext.shared
        let pipeline = try ctx.makeComputePipelineState(
            source: SyntheticExecutionLogShaders.mockGatingAndLogSource,
            functionName: "syntheticGatingAndLog"
        )
        #expect(pipeline.maxTotalThreadsPerThreadgroup > 0)

        // Execute synthetic gating for 64 tokens
        let count = 64
        let log = GPUExecutionLog(device: ctx.device, capacity: 256)
        var cap = UInt32(log.capacity)
        var tokenOffset: UInt32 = 0
        var layerIndex: UInt16 = 8
        var horizonIndex: UInt16 = 2
        var numExperts: UInt32 = 60
        var baseTimestamp: UInt64 = 1_000_000

        guard let cmdBuf = ctx.commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else {
            Issue.record("Failed to create Metal command buffer or encoder")
            return
        }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(log.buffer, offset: 0, index: 0)
        enc.setBytes(&cap, length: 4, index: 1)
        enc.setBytes(&tokenOffset, length: 4, index: 2)
        enc.setBytes(&layerIndex, length: 2, index: 3)
        enc.setBytes(&horizonIndex, length: 2, index: 4)
        enc.setBytes(&numExperts, length: 4, index: 5)
        enc.setBytes(&baseTimestamp, length: 8, index: 6)

        let gridSize = MTLSize(width: count, height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: min(pipeline.maxTotalThreadsPerThreadgroup, count), height: 1, depth: 1)
        enc.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let drained = log.drain()
        #expect(drained.count == count)
        for (i, entry) in drained.enumerated() {
            #expect(entry.tokenIndex == UInt32(i))
            #expect(entry.layerIndex == 8)
            #expect(entry.horizonIndex == 2)
            #expect(entry.confidenceScore >= 0.50 && entry.confidenceScore <= 1.0)
            #expect(entry.reserved == 0xCAFEBABEDEADBEEF)
        }
    }

    // MARK: - 7. Basic Buffer & Drain Semantics

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

    // MARK: - 8. Genuine MSL Runtime Compilation Verification (Remediating Facade / Evasion)

    @Test("Execution log MSL shader source is valid metal syntax and compiles via MetalContext")
    func testExecutionLogMSLSource() throws {
        let src = GPUExecutionLog.mslKernelSource

        // 1. Mandatory zero-atomic invariant check (Requirement R3)
        #expect(!src.contains("atomic_"), "MSL shader source must not contain GPU atomic operations")
        #expect(!src.contains("atomic_fetch_add"), "MSL shader source must not contain atomic_fetch_add")

        // 2. Genuine runtime Metal compilation via MTLDevice.makeLibrary(source:options:)
        let library = try device.makeLibrary(source: src, options: nil)
        #expect(library.functionNames.contains("writeExecutionLogEntry"), "Library must contain writeExecutionLogEntry")
        #expect(library.functionNames.contains("writeTokenGatingLog"), "Library must contain writeTokenGatingLog")
        #expect(library.functionNames.contains("writeExecutionLogBatch"), "Library must contain writeExecutionLogBatch")

        // 3. Genuine pipeline state compilation for all 3 production kernels via MetalContext
        let ctx = MetalContext.shared
        let entryPipeline = try ctx.makeComputePipelineState(source: src, functionName: "writeExecutionLogEntry")
        #expect(entryPipeline.maxTotalThreadsPerThreadgroup > 0, "writeExecutionLogEntry pipeline must have valid threadgroup size")

        let gatingPipeline = try ctx.makeComputePipelineState(source: src, functionName: "writeTokenGatingLog")
        #expect(gatingPipeline.maxTotalThreadsPerThreadgroup > 0, "writeTokenGatingLog pipeline must have valid threadgroup size")

        let batchPipeline = try ctx.makeComputePipelineState(source: src, functionName: "writeExecutionLogBatch")
        #expect(batchPipeline.maxTotalThreadsPerThreadgroup > 0, "writeExecutionLogBatch pipeline must have valid threadgroup size")
    }

    // MARK: - 9. Direct Production MSL Kernel GPU Execution & Drain Verification

    @Test("Production MSL writeExecutionLogEntry executes on GPU and drains accurately")
    func testProductionWriteExecutionLogEntryDirectExecution() throws {
        let ctx = MetalContext.shared
        let pipeline = try ctx.makeComputePipelineState(
            source: GPUExecutionLog.mslKernelSource,
            functionName: "writeExecutionLogEntry"
        )

        let capacity = 4096
        let log = GPUExecutionLog(device: ctx.device, capacity: capacity)

        var cap = UInt32(capacity)
        var tokenIndex: UInt32 = 42
        var layerIndex: UInt16 = 5
        var horizonIndex: UInt16 = 1
        var expertID: UInt16 = 17
        var confidence: Float32 = 0.94
        var timestamp: UInt64 = 1_234_567

        guard let cmdBuf = ctx.commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else {
            throw TestError.computeEncoderCreationFailed
        }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(log.buffer, offset: 0, index: 0)
        enc.setBytes(&cap, length: MemoryLayout<UInt32>.size, index: 1)
        enc.setBytes(&tokenIndex, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&layerIndex, length: MemoryLayout<UInt16>.size, index: 3)
        enc.setBytes(&horizonIndex, length: MemoryLayout<UInt16>.size, index: 4)
        enc.setBytes(&expertID, length: MemoryLayout<UInt16>.size, index: 5)
        enc.setBytes(&confidence, length: MemoryLayout<Float32>.size, index: 6)
        enc.setBytes(&timestamp, length: MemoryLayout<UInt64>.size, index: 7)

        let gridSize = MTLSize(width: 1, height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: 1, height: 1, depth: 1)
        enc.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Drain from execution log and verify exact record extracted
        let drained = log.drain()
        #expect(drained.count == 1, "Exactly one entry must be drained")

        guard let entry = drained.first else {
            Issue.record("Expected drained entry from production kernel write")
            return
        }

        #expect(entry.tokenIndex == 42)
        #expect(entry.layerIndex == 5)
        #expect(entry.horizonIndex == 1)
        #expect(entry.expertID == 17)
        #expect(abs(entry.confidenceScore - 0.94) < 1e-5)
        #expect(entry.timestamp == 1_234_567)
        #expect(entry.padding == 0)
        #expect(entry.reserved == 0)

        // Idempotent second drain
        let secondDrain = log.drain()
        #expect(secondDrain.isEmpty, "Second drain must be empty as slot was cleared")
    }

    @Test("Production MSL writeExecutionLogEntry dispatches multiple entries drained into LRUWeightTracker")
    func testProductionWriteExecutionLogEntryMultipleEntriesAndRecencyDrain() throws {
        let ctx = MetalContext.shared
        let pipeline = try ctx.makeComputePipelineState(
            source: GPUExecutionLog.mslKernelSource,
            functionName: "writeExecutionLogEntry"
        )

        let capacity = 4096
        let log = GPUExecutionLog(device: ctx.device, capacity: capacity)
        let tracker = LRUWeightTracker()

        struct TestSpec {
            let token: UInt32
            let layer: UInt16
            let horizon: UInt16
            let expert: UInt16
            let conf: Float32
            let ts: UInt64
        }

        let specs: [TestSpec] = [
            TestSpec(token: 0, layer: 5, horizon: 1, expert: 10, conf: 0.95, ts: 1000),
            TestSpec(token: 0, layer: 6, horizon: 1, expert: 20, conf: 0.88, ts: 1100),
            TestSpec(token: 1, layer: 5, horizon: 1, expert: 30, conf: 0.91, ts: 1200),
            TestSpec(token: 1, layer: 7, horizon: 2, expert: 15, conf: 0.82, ts: 1300),
            TestSpec(token: 2, layer: 8, horizon: 3, expert: 45, conf: 0.77, ts: 1400)
        ]

        var cap = UInt32(capacity)

        for spec in specs {
            guard let cmdBuf = ctx.commandQueue.makeCommandBuffer(),
                  let enc = cmdBuf.makeComputeCommandEncoder() else {
                throw TestError.computeEncoderCreationFailed
            }

            var t = spec.token
            var l = spec.layer
            var h = spec.horizon
            var e = spec.expert
            var c = spec.conf
            var ts = spec.ts

            enc.setComputePipelineState(pipeline)
            enc.setBuffer(log.buffer, offset: 0, index: 0)
            enc.setBytes(&cap, length: MemoryLayout<UInt32>.size, index: 1)
            enc.setBytes(&t, length: MemoryLayout<UInt32>.size, index: 2)
            enc.setBytes(&l, length: MemoryLayout<UInt16>.size, index: 3)
            enc.setBytes(&h, length: MemoryLayout<UInt16>.size, index: 4)
            enc.setBytes(&e, length: MemoryLayout<UInt16>.size, index: 5)
            enc.setBytes(&c, length: MemoryLayout<Float32>.size, index: 6)
            enc.setBytes(&ts, length: MemoryLayout<UInt64>.size, index: 7)

            let gridSize = MTLSize(width: 1, height: 1, depth: 1)
            let threadgroupSize = MTLSize(width: 1, height: 1, depth: 1)
            enc.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }

        // Drain all written entries
        let drained = log.drain()
        #expect(drained.count == specs.count, "Must drain all \(specs.count) entries written by production kernel")

        for entry in drained {
            guard let match = specs.first(where: {
                $0.token == entry.tokenIndex &&
                $0.layer == entry.layerIndex &&
                $0.horizon == entry.horizonIndex &&
                $0.expert == entry.expertID
            }) else {
                Issue.record("Unexpected entry drained: \(entry)")
                continue
            }
            #expect(abs(entry.confidenceScore - match.conf) < 1e-4)
            #expect(entry.timestamp == match.ts)

            let key = ExpertKey(layer: Int(entry.layerIndex), expert: Int(entry.expertID))
            tracker.recordAccess(expert: key, timestamp: entry.timestamp)
        }

        #expect(tracker.count == specs.count)
        // The LRU victim must be the earliest accessed expert: Layer 5, Expert 10 (ts = 1000)
        #expect(tracker.lruExpert == ExpertKey(layer: 5, expert: 10))
    }

    @Test("Production MSL writeTokenGatingLog executes parallel top-K logging on GPU and drains accurately")
    func testProductionWriteTokenGatingLogDirectExecution() throws {
        let ctx = MetalContext.shared
        let pipeline = try ctx.makeComputePipelineState(
            source: GPUExecutionLog.mslKernelSource,
            functionName: "writeTokenGatingLog"
        )

        let capacity = 4096
        let log = GPUExecutionLog(device: ctx.device, capacity: capacity)

        let topK: UInt32 = 4
        let selectedExperts: [UInt16] = [12, 25, 38, 49]
        let routingWeights: [Float32] = [0.45, 0.25, 0.18, 0.12]
        var dispatchTime: UInt64 = 7_000_000

        guard let expertsBuffer = ctx.device.makeBuffer(
                bytes: selectedExperts,
                length: selectedExperts.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared
              ),
              let weightsBuffer = ctx.device.makeBuffer(
                bytes: routingWeights,
                length: routingWeights.count * MemoryLayout<Float32>.stride,
                options: .storageModeShared
              ) else {
            throw TestError.bufferAllocationFailed
        }

        var cap = UInt32(capacity)
        var tokenIndex: UInt32 = 10
        var layerIndex: UInt16 = 6
        var k = topK

        guard let cmdBuf = ctx.commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else {
            throw TestError.computeEncoderCreationFailed
        }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(log.buffer, offset: 0, index: 0)
        enc.setBytes(&cap, length: 4, index: 1)
        enc.setBytes(&tokenIndex, length: 4, index: 2)
        enc.setBytes(&layerIndex, length: 2, index: 3)
        enc.setBuffer(expertsBuffer, offset: 0, index: 4)
        enc.setBuffer(weightsBuffer, offset: 0, index: 5)
        enc.setBytes(&dispatchTime, length: 8, index: 6)
        enc.setBytes(&k, length: 4, index: 7)

        let gridSize = MTLSize(width: Int(topK), height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: min(pipeline.maxTotalThreadsPerThreadgroup, Int(topK)), height: 1, depth: 1)
        enc.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let drained = log.drain()
        #expect(drained.count == Int(topK))

        for (rank, entry) in drained.sorted(by: { $0.timestamp < $1.timestamp }).enumerated() {
            #expect(entry.tokenIndex == 10)
            #expect(entry.layerIndex == 6)
            #expect(entry.horizonIndex == 1)
            #expect(entry.expertID == selectedExperts[rank])
            #expect(abs(entry.confidenceScore - routingWeights[rank]) < 1e-5)
            #expect(entry.timestamp == dispatchTime + UInt64(rank))
            #expect(entry.padding == 0)
            #expect(entry.reserved == 0)
        }
    }

    @Test("Production MSL writeExecutionLogBatch executes sequential batch logging on GPU and drains accurately")
    func testProductionWriteExecutionLogBatchDirectExecution() throws {
        let ctx = MetalContext.shared
        let pipeline = try ctx.makeComputePipelineState(
            source: GPUExecutionLog.mslKernelSource,
            functionName: "writeExecutionLogBatch"
        )

        let capacity = 4096
        let log = GPUExecutionLog(device: ctx.device, capacity: capacity)

        let numExperts: UInt32 = 3
        let expertIDs: [UInt16] = [3, 7, 19]
        let confidences: [Float32] = [0.85, 0.90, 0.78]
        var timestamp: UInt64 = 8_888_000

        guard let expertsBuffer = ctx.device.makeBuffer(
                bytes: expertIDs,
                length: expertIDs.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared
              ),
              let confBuffer = ctx.device.makeBuffer(
                bytes: confidences,
                length: confidences.count * MemoryLayout<Float32>.stride,
                options: .storageModeShared
              ) else {
            throw TestError.bufferAllocationFailed
        }

        var cap = UInt32(capacity)
        var baseSlot: UInt32 = 128
        var tokenIndex: UInt32 = 55
        var layerIndex: UInt16 = 9
        var n = numExperts

        guard let cmdBuf = ctx.commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else {
            throw TestError.computeEncoderCreationFailed
        }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(log.buffer, offset: 0, index: 0)
        enc.setBytes(&cap, length: 4, index: 1)
        enc.setBytes(&baseSlot, length: 4, index: 2)
        enc.setBytes(&tokenIndex, length: 4, index: 3)
        enc.setBytes(&layerIndex, length: 2, index: 4)
        enc.setBuffer(expertsBuffer, offset: 0, index: 5)
        enc.setBuffer(confBuffer, offset: 0, index: 6)
        enc.setBytes(&timestamp, length: 8, index: 7)
        enc.setBytes(&n, length: 4, index: 8)

        let gridSize = MTLSize(width: Int(numExperts), height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: min(pipeline.maxTotalThreadsPerThreadgroup, Int(numExperts)), height: 1, depth: 1)
        enc.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let drained = log.drain()
        #expect(drained.count == Int(numExperts))

        for (i, entry) in drained.sorted(by: { $0.timestamp < $1.timestamp }).enumerated() {
            #expect(entry.tokenIndex == 55)
            #expect(entry.layerIndex == 9)
            #expect(entry.horizonIndex == 1)
            #expect(entry.expertID == expertIDs[i])
            #expect(abs(entry.confidenceScore - confidences[i]) < 1e-5)
            #expect(entry.timestamp == timestamp + UInt64(i))
            #expect(entry.padding == 0)
            #expect(entry.reserved == 0)
        }
    }

    // MARK: - 10. Memory Budget & Alignment Checks

    @Test("MemoryBudgetConfig conservative budget calculation")
    func testMemoryBudgetTotal() {
        let budget = MemoryBudgetConfig()
        let arch = MoEArchitectureConfig.synthetic
        let total = budget.totalConservativeBudgetBytes(for: arch)
        #expect(total < 1_000_000_000)
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
```
