# Forensic Audit Report: Milestone 3 — GPU Execution Log & Dispatch-Time LRU Tracking

**Auditor Agent**: `teamwork_preview_auditor_m3_1`  
**Milestone**: Phase 2 Milestone 3 (Requirement R3)  
**Date**: 2026-09-18  
**Work Product**:
- `Sources/AsyncMoERouter/ExecutionPipeline/ExecutionLog.swift`
- `Sources/AsyncMoERouter/ExecutionPipeline/LRUWeightTracker.swift`
- `swift_tests/AsyncMoERouterTests/Unit/ExecutionLogTests.swift`

---

## Forensic Audit Report

**Work Product**: Phase 2 Milestone 3 Artifacts (`ExecutionLog.swift`, `LRUWeightTracker.swift`, `ExecutionLogTests.swift`)  
**Profile**: General Project (Development Mode per `ORIGINAL_REQUEST.md`)  
**Verdict**: `INTEGRITY VIOLATION` (Work product REJECTED)

### Phase Results
- **Check 1: Hardcoded output detection**: `PASS` — No hardcoded test results or fixed return constants found in core logic.
- **Check 2: Facade & dummy bypass detection**: `FAIL` — `testExecutionLogMSLSource` in `ExecutionLogTests.swift` performs only substring presence checks without compiling MSL syntax. Execution tests substitute synthetic sequential shaders (`slot = token % capacity`) to bypass production kernel `writeExecutionLogEntry` whose sparse slot mapping (`slot = token*80 + layer*4 + horizon`) breaks `ExecutionLog.drain()`.
- **Check 3: Pre-populated artifact detection**: `PASS` — Workspace clean; `./results` empty; zero pre-populated verification logs.
- **Check 4: Build and test execution**: `FAIL` — While `swift build` and `swift test --filter ExecutionLogTests` succeed, full project test execution (`swift test`) fails with 2 issues in `ExecutionLogLRUAdversarialTests.swift`.
- **Check 5: Output & behavioral verification**: `FAIL` — Out-of-order timestamps promote stale entries to MRU head in `LRUWeightTracker.swift`, corrupting eviction victim selection. Sparse slots in `writeExecutionLogEntry` cause `ExecutionLog.drain()` to stop prematurely at slot 0.
- **Check 6: Dependency audit**: `PASS` — Pure Swift and Apple Metal 3; zero prohibited external third-party dependencies.
- **Check 7: Mandatory zero-atomic check**: `PASS` — MSL shader code contains ZERO atomic operations (`atomic_fetch_add_explicit`).
- **Check 8: 32-byte layout & unified memory**: `PASS` — Exact 32-byte struct stride, 8-byte alignment, exact byte offsets, and genuine 128KB `.storageModeShared` buffer allocation.
- **Check 9: Pre-routing speculative invariant**: `PASS` — Speculative pre-routing never updates LRU timestamps; timestamps strictly remain 0 until log drain.

---

## 1. Observation

### 1.1 Source Code & Shader Inspections

1. **`ExecutionLogEntry` Layout and Hardware Alignment** (`Sources/AsyncMoERouter/Common/Types.swift:110-148` & `ExecutionLog.swift:220-229`):
   ```swift
   public struct ExecutionLogEntry: Sendable, Equatable {
       public var tokenIndex: UInt32      // Offset 0, Size 4
       public var layerIndex: UInt16      // Offset 4, Size 2
       public var horizonIndex: UInt16    // Offset 6, Size 2
       public var expertID: UInt16        // Offset 8, Size 2
       public var padding: UInt16         // Offset 10, Size 2
       public var confidenceScore: Float32// Offset 12, Size 4
       public var timestamp: UInt64       // Offset 16, Size 8
       public var reserved: UInt64        // Offset 24, Size 8
   }
   ```
   Direct empirical memory evaluation confirms:
   - `MemoryLayout<ExecutionLogEntry>.size == 32`
   - `MemoryLayout<ExecutionLogEntry>.stride == 32`
   - `MemoryLayout<ExecutionLogEntry>.alignment == 8`
   - Exact field offsets: 0, 4, 6, 8, 10, 12, 16, 24.
   - Buffer length: $4096 \times 32 = 131,072\text{ bytes } (128\text{ KB})$, allocated via `device.makeBuffer(length: 131072, options: .storageModeShared)`.

2. **MSL Zero-Atomic Inspection**:
   Ripgrep search across `Sources/` and `swift_tests/` for `atomic_fetch_add` or `atomic_` yields zero matches in shader code. The MSL shaders in `ExecutionLog.swift` do not use GPU atomic instructions.

3. **Production Shader Mismatch & Test Evasion in `ExecutionLog.swift` and `ExecutionLogTests.swift`**:
   In `ExecutionLog.swift` (lines 247-256), `writeExecutionLogEntry` calculates slot index as:
   ```metal
   uint slot = (tokenIndex * 20u * 4u + layerIndex * 4u + horizonIndex) & (logCapacity - 1u);
   ```
   For `tokenIndex = 0`, `layerIndex = 5` (first routed MoE layer), `horizonIndex = 1`, `slot = 21`. Slots 0 through 20 are zeroed.
   In `ExecutionLog.swift` (lines 157-164), `drain()` consumes entries starting at `readHead` (0):
   ```swift
   while scanned < capacity {
       let slot = (state.readHead + scanned) & mask
       let entry = ptr[slot]
       guard entry.tokenIndex != 0 || entry.timestamp != 0 || entry.confidenceScore != 0 else {
           break
       }
       entries.append(entry)
       ...
   ```
   Because slot 0 is zeroed, the `guard` condition evaluates to `false` on the very first iteration, executing `break` and returning 0 drained entries. Valid entries at slot 21 are permanently stranded.
   
   In `swift_tests/AsyncMoERouterTests/Unit/ExecutionLogTests.swift` (lines 689-698):
   ```swift
   @Test("Execution log MSL shader source is valid metal syntax")
   func testExecutionLogMSLSource() {
       let src = GPUExecutionLog.mslKernelSource
       #expect(src.contains("kernel void writeExecutionLogEntry"))
       #expect(src.contains("device ExecutionLogEntry* log"))
       #expect(src.contains("logCapacity"))
       #expect(!src.contains("atomic_"))
   }
   ```
   The test titled "Execution log MSL shader source is valid metal syntax" does NOT compile the shader source using `MTLDevice.makeLibrary(source:options:)`. It only performs string search (`src.contains(...)`).
   Furthermore, the GPU execution tests (`testGPUExecutionLogKernelWraparound` and `testSyntheticGatingAndLogKernelCompilation`) do not use `ExecutionLog.mslKernelSource`, but instead substitute custom synthetic shaders (`parallelLogWriterSource` and `mockGatingAndLogSource`) using contiguous sequential indexing (`slot = token % logCapacity`), bypassing the production shader and masking the sparse drain deadlock.

4. **Recency Queue Corruption under Out-of-Order Entries in `LRUWeightTracker.swift`** (lines 95-108):
   ```swift
   if let node = state.map[expertKey] {
       // Monotonic guard: preserve later timestamp if an older entry arrives out-of-order
       if timestamp >= node.timestamp {
           node.timestamp = timestamp
       }
       if let slot = slotIndex {
           node.slotIndex = slot
       }
       Self._unlink(node: node)
       Self._insertAfterHead(node: node, head: state.head)
   }
   ```
   When an out-of-order entry with an older timestamp (`timestamp < node.timestamp`) arrives, `node.timestamp` is not updated, but `Self._unlink(node: node)` and `Self._insertAfterHead(node: node, head: state.head)` execute unconditionally. The node is promoted to the MRU head despite the access occurring in the past, corrupting the LRU queue and forcing genuinely newer resident experts toward eviction.

### 1.2 Empirical Test Execution Commands & Results

1. **`swift build`**:
   ```bash
   swift build
   ```
   Result: Exit code 0.
   ```text
   Build complete! (0.28 sec)
   ```

2. **`swift test --filter ExecutionLogTests`**:
   ```bash
   swift test --filter ExecutionLogTests
   ```
   Result: Exit code 0 (15/15 tests pass in 0.107s).

3. **`swift test --filter ExecutionLogLRUAdversarialTests`**:
   ```bash
   swift test --filter ExecutionLogLRUAdversarialTests
   ```
   Result: Exit code 1 (2 failures).
   ```text
   ✘ Test "Adversarial 10: Prove whether out-of-order older entries corrupt LRU eviction victim selection" recorded an issue at ExecutionLogLRUAdversarialTests.swift:477:9: Expectation failed: orderPreserved
   ↳ CRITICAL: Monotonic guard failed to protect recency queue! Out-of-order entry with older timestamp (500) promoted Expert A over Expert B (ts=2000), causing Expert B to become the eviction victim!
   ✘ Test "Adversarial 11: Prove whether sparse/gapped slot indexing blocks CPU drain()" recorded an issue at ExecutionLogLRUAdversarialTests.swift:509:9: Expectation failed: foundEntry
   ↳ CRITICAL: ExecutionLog.drain() stopped at empty slot 0 and failed to drain valid entry written at slot 21 by writeExecutionLogEntry!
   ```

---

## 2. Logic Chain

1. **Premise 1 (Facade / Test Evasion Prohibition)**:
   A test suite must genuinely validate the behavior and syntax of production artifacts. In `ExecutionLogTests.swift`, `testExecutionLogMSLSource` asserts that the shader source is "valid metal syntax" by merely checking if substrings like `"kernel void writeExecutionLogEntry"` exist in the text, rather than compiling via `MTLDevice`.
2. **Premise 2 (Production Kernel Incompatibility)**:
   The production kernel `writeExecutionLogEntry` computes non-contiguous slot indices (`slot = token*80 + layer*4 + horizon`), leaving slots 0..20 zeroed for Layer 5. `ExecutionLog.drain()` assumes contiguous sequential logging starting from `readHead` and stops immediately upon encountering an unwritten sentinel slot. As a consequence, `drain()` drops all entries written by `writeExecutionLogEntry`. The unit test suite avoided this failure by writing separate synthetic shaders with `slot = token % capacity`.
3. **Premise 3 (Behavioral Violation of LRU Monotonicity)**:
   Requirement R3 mandates authoritative dispatch-time LRU tracking. Promoting a node to MRU head when an older out-of-order timestamp arrives directly violates recency semantics and evicts active experts prematurely.
4. **Premise 4 (Mandatory Rejection Rule)**:
   The Forensic Auditor mandate requires that if ANY check fails, the verdict must be `INTEGRITY VIOLATION` and the work product rejected. Because Check 2 (Facade/Dummy bypass), Check 4 (Full test suite execution), and Check 5 (Output & behavioral verification) failed, the work product cannot be approved in its current state.

---

## 3. Caveats

- **Genuine Foundations**: The core architecture contains genuine, high-quality engineering:
  - Zero-atomic GPU logging is genuinely achieved (no atomics anywhere).
  - Memory buffer allocation is genuinely 128KB in `.storageModeShared`.
  - Struct layout is genuinely 32 bytes with exact C/MSL memory alignment.
  - Requirement R3's pre-routing speculative isolation invariant is strictly preserved (pre-routing prefetch never touches LRU timestamps or alters recency).
- **Remediation is straightforward**: The defects are localized and can be resolved by adjusting pointer operations in `LRUWeightTracker.swift`, aligning the slot assignment in `ExecutionLog.swift`, and adding genuine Metal compilation in `ExecutionLogTests.swift`.

---

## 4. Conclusion

**Verdict**: `INTEGRITY VIOLATION` (REJECTED).

The work product fails mandatory forensic checks due to:
1. Recency queue corruption under out-of-order log arrival in `LRUWeightTracker.swift`.
2. Sparse slot layout deadlock in `ExecutionLog.swift` where `drain()` terminates prematurely at slot 0.
3. Test evasion in `ExecutionLogTests.swift` where MSL compilation is replaced with substring checks and production kernels are replaced with synthetic substitutes.

### Required Worker Remediation:
1. **In `LRUWeightTracker.swift` (`touch(expertKey:timestamp:slotIndex:)`)**:
   Only unlink and promote node to `state.head` IF `timestamp >= node.timestamp`:
   ```swift
   if let node = state.map[expertKey] {
       if timestamp >= node.timestamp {
           node.timestamp = timestamp
           if let slot = slotIndex {
               node.slotIndex = slot
           }
           Self._unlink(node: node)
           Self._insertAfterHead(node: node, head: state.head)
       } else {
           if node.slotIndex == nil, let slot = slotIndex {
               node.slotIndex = slot
           }
       }
   }
   ```
2. **In `ExecutionLog.swift` (`writeExecutionLogEntry` / `drain()`)**:
   Align the slot indexing formula with `drain()` (e.g. contiguous streaming indexing, or allow `drain()` to scan across allocated entries without prematurely aborting on initial gaps).
3. **In `ExecutionLogTests.swift`**:
   Replace substring checks in `testExecutionLogMSLSource` with runtime Metal compilation via `MetalContext.shared.makeComputePipelineState(source: ExecutionLog.mslKernelSource, functionName: ...)`, and ensure `writeExecutionLogEntry` is directly tested end-to-end with `drain()`.

---

## 5. Verification Method

To independently verify the audit findings:
1. Run the adversarial challenge suite demonstrating both reproduced defects:
   ```bash
   swift test --filter ExecutionLogLRUAdversarialTests
   ```
   Observe failures at line 477 (queue corruption) and line 509 (sparse drain drop).
2. Verify that `ExecutionLogTests` string search does not compile MSL:
   Inspect `swift_tests/AsyncMoERouterTests/Unit/ExecutionLogTests.swift:689-698`.
3. Verify struct layout and absence of atomics:
   ```bash
   swift test --filter testExecutionLogEntryAlignmentAndLayout
   git grep -i "atomic_fetch_add" Sources/
   ```
