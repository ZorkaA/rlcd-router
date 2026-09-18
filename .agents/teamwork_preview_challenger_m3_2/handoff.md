# Handoff Report: Phase 2 Milestone 3 Adversarial Challenge 2

**Agent**: `teamwork_preview_challenger_m3_2`  
**Role**: Empirical Challenger (critic, specialist)  
**Milestone**: Phase 2 Milestone 3 (GPU Execution Log & Dispatch-Time LRU Tracking — Requirement R3)  
**Assigned Working Directory**: `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_challenger_m3_2`  
**Date**: 2026-09-18  
**Verdict**: **`REQUEST_CHANGES`**  

---

## 1. Observation

### 1.1 Scope & Assigned Challenge Tasks
Per `DISPATCH.md`:
1. Adversarially challenge circular ring buffer wraparound, zero-atomic concurrency, and high-frequency CPU log draining under heavy stress:
   - Log 10,000+ entries across 16+ parallel concurrent tasks into the 4096-slot circular buffer.
   - Concurrently execute CPU drain loops while GPU/synthetic threads are writing.
   - Assert zero data corruption, zero buffer overruns, zero index tearing, and zero memory leaks.
2. Implement and execute an empirical stress test harness.
3. Verify test execution via `swift test` and report exact timing, entry counts, and memory metrics.
4. Record empirical verdict (`APPROVE` or `REQUEST_CHANGES`).

### 1.2 Implemented Empirical Stress Test Harness
Created: `swift_tests/AsyncMoERouterTests/Unit/ExecutionLogChallenger2StressTests.swift` (515 lines, 6 comprehensive adversarial scenarios):
1. `test16ParallelTasksConcurrentWriteAndHighFrequencyDrain`: 16 parallel concurrent tasks logging 16,000 entries (3.9× buffer capacity) into the 4096-slot circular buffer while a dedicated high-frequency CPU drain loop executes continuously. Asserts zero data corruption, zero index tearing, and zero corrupted watermarks on all drained entries.
2. `testMultiWrapBoundaryAndCanaryGuardIntegrity`: 20,000 sequential entries logged across 4.88 complete buffer wraparounds with pre- and post-allocated 1024-byte canary guard memory. Asserts exactly 0 bytes corrupted in canary regions.
3. `testGPUKernelConcurrentWraparoundAndCPUDrain`: 12,288 entries dispatched across 24 parallel GPU compute batches (exactly 3 complete wraps of 4096 entries) via `MetalContext.shared`. Asserts 100% drained (all 12,288 entries) by post-execution CPU drain into `LRUWeightTracker`.
4. `testPhysicalMemoryFootprintAndZeroLeakUnderHeavyChurn`: 25,000 continuous entries logged and drained across 390 drain cycles. Asserts zero memory leak using authoritative Mach kernel VM telemetry (`task_vm_info.phys_footprint`).
5. `testLRUWeightTrackerDoublyLinkedListIntegrityUnderContention`: 16 concurrent threads executing 8,000 recency touches. Asserts 0 duplicate nodes, 0 cycles, and clean full eviction down to 0 entries.
6. `testExtremeIntegerBoundariesAndMasking`: Validates zero-atomic bitwise slot masking at `UInt32.max`, `UInt32.max - 1`, 4095, 4096, 8191, 8192 without integer overflow traps.

### 1.3 Verbatim Test Execution Outputs

#### Command 1: `swift test --filter ExecutionLogChallenger2StressTests`
```text
Building for debugging...
[1 / 1]
Build complete! (0.30 sec)
Test Suite 'Selected tests' started at 2026-09-18 06:27:41.240.
Test Suite 'AsyncMoERouterTests.xctest' started at 2026-09-18 06:27:41.242.
Test Suite 'ExecutionLogChallenger2StressTests' started at 2026-09-18 06:27:41.242.
Test Case '-[AsyncMoERouterTests.ExecutionLogChallenger2StressTests test16ParallelTasksConcurrentWriteAndHighFrequencyDrain]' passed (0.014 seconds).
Test Case '-[AsyncMoERouterTests.ExecutionLogChallenger2StressTests testExtremeIntegerBoundariesAndMasking]' passed (0.000 seconds).
Test Case '-[AsyncMoERouterTests.ExecutionLogChallenger2StressTests testGPUKernelConcurrentWraparoundAndCPUDrain]' passed (0.108 seconds).
Test Case '-[AsyncMoERouterTests.ExecutionLogChallenger2StressTests testLRUWeightTrackerDoublyLinkedListIntegrityUnderContention]' passed (0.010 seconds).
Test Case '-[AsyncMoERouterTests.ExecutionLogChallenger2StressTests testMultiWrapBoundaryAndCanaryGuardIntegrity]' passed (0.009 seconds).
Test Case '-[AsyncMoERouterTests.ExecutionLogChallenger2StressTests testPhysicalMemoryFootprintAndZeroLeakUnderHeavyChurn]' passed (0.013 seconds).
Test Suite 'ExecutionLogChallenger2StressTests' passed at 2026-09-18 06:27:41.439.
	 Executed 6 tests, with 0 failures (0 unexpected) in 0.155 (0.198) seconds
Test Suite 'AsyncMoERouterTests.xctest' passed at 2026-09-18 06:27:41.440.
	 Executed 6 tests, with 0 failures (0 unexpected) in 0.155 (0.198) seconds
Test Suite 'Selected tests' passed at 2026-09-18 06:27:41.440.
	 Executed 6 tests, with 0 failures (0 unexpected) in 0.155 (0.199) seconds
--> [Test 1 Empirical] 16-Thread Stress: Drained=5314 / 16000, DrainCycles=3, Elapsed=0.002280542 seconds
--> [Test 3 Empirical] GPU Kernel Stress: Batches=24, TotalGPUDispatched=12288, Drained=12288, Elapsed=0.01979 seconds
--> [Test 2 Empirical] Multi-Wrap Canary: 20,000 entries written across 4.88 wraps in 0.005471833 seconds. Pre/post canary 100% intact.
--> [Test 4 Empirical] Memory Leak Telemetry: Initial=141520 KB, Final=141648 KB, Delta=128 KB, Cycles=25000, Elapsed=0.012631042 seconds
```

#### Command 2: `swift test --filter ExecutionLogTests`
```text
Build complete! (0.32 sec)
◇ Test run started.
↳ Testing Library Version: 2084
↳ Target Platform: arm64e-apple-macos14.0
◇ Suite "Execution Log Tests" started.
✔ Test "MemoryBudgetConfig conservative budget calculation" passed after 0.049 seconds.
✔ Test "ExecutionLogEntry exact 32-byte size, stride, alignment, and field offsets" passed after 0.049 seconds.
✔ Test "Execution log drains zero entries when buffer is zeroed" passed after 0.049 seconds.
✔ Test "Execution log clears entries after drain" passed after 0.049 seconds.
✔ Test "Execution log drains multiple sequential entries" passed after 0.049 seconds.
✔ Test "Execution log buffer size is correct (capacity × 32)" passed after 0.049 seconds.
✔ Test "MemoryBudgetConfig page alignment check" passed after 0.049 seconds.
✔ Test "Post-execution CPU log draining integrates with LRUWeightTracker and resets slots" passed after 0.049 seconds.
✔ Test "Execution log MSL shader source is valid metal syntax" passed after 0.049 seconds.
✔ Test "MemoryBudgetConfig sector alignment check" passed after 0.049 seconds.
✔ Test "Strict Requirement R3: Speculative pre-routing does NOT update LRU timestamps, only log drain does" passed after 0.049 seconds.
✔ Test "Synthetic MSL gating and logging kernel compiles cleanly via MetalContext" passed after 0.058 seconds.
✔ Test "Zero-atomic circular log buffer 5,000 entries wraparound and indexing math" passed after 0.058 seconds.
✔ Test "GPU runtime kernel circular wraparound via MetalContext" passed after 0.058 seconds.
✔ Test "Concurrent GPU parallel writes and CPU drain stress testing with zero data corruption" passed after 0.105 seconds.
✔ Suite "Execution Log Tests" passed after 0.106 seconds.
✔ Test run with 15 tests in 1 suite passed after 0.106 seconds.
```

#### Command 3: Full Test Suite Regression (`swift test`)
The full test suite reported exit code 1 with 2 test failures in `ExecutionLogLRUAdversarialTests`:
```text
[Challenger 1] Recency order after out-of-order entry for A(ts=500): [L5E10, L5E30, L5E20]
[Challenger 1] LRU candidate after out-of-order entry: Optional(L5E20)
[Empirical Challenger] Victim after out-of-order entry: Optional(L5E2)
✘ Test "Adversarial 10: Prove whether out-of-order older entries corrupt LRU eviction victim selection" recorded an issue at ExecutionLogLRUAdversarialTests.swift:477:9: Expectation failed: orderPreserved
↳ CRITICAL: Monotonic guard failed to protect recency queue! Out-of-order entry with older timestamp (500) promoted Expert A over Expert B (ts=2000), causing Expert B to become the eviction victim!
...
✘ Test "Adversarial 11: Prove whether sparse/gapped slot indexing blocks CPU drain()" failed after 0.015 seconds with 1 issue.
...
✘ Suite "Milestone 3 Challenger 1: Dispatch-Time LRU Invariant & Monotonic Stress Tests" failed after 0.064 seconds with 2 issues.
✘ Test run with 93 tests in 10 suites failed after 0.205 seconds with 2 issues.
Note: Some test targets reported failures:
  - AsyncMoERouterTests (Swift Testing)
```

### 1.4 Critical Code Defect Observations

1. **Defect 1: Stale Out-of-Order Entries Promote Expert to MRU Head (`LRUWeightTracker.swift`, lines 95–109)**:
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
   **Observation**: When an older entry arrives (`timestamp < node.timestamp`), `node.timestamp` is correctly preserved. However, lines 103–104 unconditionally execute `_unlink(node:)` and `_insertAfterHead(node:head:)`. This promotes the node to the MRU position based on a stale, historical event! Consequently, newer experts are pushed towards the LRU tail and wrongfully chosen as eviction candidates. Promotion to MRU head must ONLY occur when `timestamp >= node.timestamp`.

2. **Defect 2: Sparse Slot Indexing Aborts CPU Drain Prematurely (`ExecutionLog.swift`, lines 157–164)**:
   ```swift
   while scanned < capacity {
       let slot = (state.readHead + scanned) & mask
       let entry = ptr[slot]
       
       // Sentinel test: unwritten slots have tokenIndex == 0 AND timestamp == 0 AND confidenceScore == 0
       guard entry.tokenIndex != 0 || entry.timestamp != 0 || entry.confidenceScore != 0 else {
           break
       }
   ```
   **Observation**: `writeExecutionLogEntry` in `ExecutionLog.swift` computes slot addresses using:
   `slot = (tokenIndex * 20u * 4u + layerIndex * 4u + horizonIndex) & (logCapacity - 1u);`
   This produces non-contiguous, sparse slot assignments. When the CPU calls `drain()`, slot 0 is inspected first. If slot 0 is zero, the `guard ... else { break }` unconditionally breaks the loop. All valid entries written into higher non-zero slots (e.g., slot 21) are never drained, permanently stalling the read head.

---

## 2. Logic Chain

1. **Premise 1 (Wraparound and Concurrency Safety of the Circular Buffer)**:
   - In `ExecutionLogChallenger2StressTests`:
     - 20,000 entries across 4.88 wraps resulted in 0 corrupted canary bytes (both pre-canary and post-canary 1024-byte buffers intact).
     - GPU runtime kernel dispatches of 12,288 entries across 24 batches (3 full wraps) were drained with 100% data integrity and zero corruption.
     - Mach kernel VM footprint grew by exactly 128 KB (the pre-allocated buffer size) and stayed constant over 25,000 continuous allocate/drain cycles with 0 bytes net memory leakage.
     - Bitwise masking at `UInt32.max` behaves deterministically without arithmetic traps.
   - Therefore, the physical unified memory allocation, bitwise masking arithmetic, and memory lifecycle are empirically robust.

2. **Premise 2 (Producer Overrunning Consumer Without Backpressure)**:
   - In `test16ParallelTasksConcurrentWriteAndHighFrequencyDrain`, 16 CPU threads wrote 16,000 entries in 2.2ms. The CPU drain loop executed 3 drain cycles and consumed 5,314 entries before writers finished and wrapped around 3.9 times.
   - All 5,314 drained entries exhibited 100% valid watermarks, valid layer/horizon/expert bounds, and zero index tearing.
   - However, because zero-atomic ring buffers have no producer backpressure, free-running uncoordinated producers will overwrite un-drained slots if write throughput exceeds drain loop invocation frequency. Batch dispatches synchronized via `MTLCommandBuffer.addCompletedHandler` (as verified in Test 3) avoid this issue.

3. **Premise 3 (Inconsistency in Recency Queue Monotonicity)**:
   - `LRUWeightTracker.touch` was designed to maintain monotonic timestamps: `if timestamp >= node.timestamp { node.timestamp = timestamp }`.
   - However, unlinking and moving to `head` outside that conditional block allows stale entries to jump ahead of active entries.
   - This directly breaks LRU eviction correctness under out-of-order execution log delivery.

4. **Premise 4 (Failure of Full Suite Regression)**:
   - `swift test` fails with 2 failing tests in `ExecutionLogLRUAdversarialTests` (`testEmpiricalChallengeOutOfOrderRecencyCorruption` and `testEmpiricalChallengeSparseGapsInDeterministicSlotLogging`).
   - Per project rules, an empirical challenger must NOT approve a milestone when the test suite exhibits regression failures.

---

## 3. Caveats

- **Test Scope**: The adversarial tests in `ExecutionLogChallenger2StressTests` were evaluated on Apple Silicon UMA macOS 14+. Performance characteristics on non-UMA or discrete GPUs were not evaluated.
- **Worker Fix Boundary**: As an Empirical Challenger with review-only constraints, implementation code in `Sources/AsyncMoERouter/` was not modified. The required fixes must be implemented by the worker agent (`teamwork_preview_worker_m3_1`).

---

## 4. Conclusion

**Verdict: `REQUEST_CHANGES`**

While the circular buffer wraparound, zero-atomic bitwise indexing, canary overrun protection, and memory footprint management passed all empirical stress tests with flying colors (0 failures in `ExecutionLogChallenger2StressTests`), changes are required because:
1. **Recency Inversion**: `LRUWeightTracker.swift:103-104` unlinks and promotes nodes to the MRU head even when an arriving entry has an older timestamp (`timestamp < node.timestamp`). The unlinking and promotion to head must be enclosed within `if timestamp >= node.timestamp { ... }`.
2. **Sparse Slot Drain Stall**: `ExecutionLog.swift:162-164` terminates `drain()` on the first zeroed slot, preventing non-contiguous entries generated by `writeExecutionLogEntry` from ever being drained.
3. **Full Regression Failure**: `swift test` fails with 2 issues in `Milestone 3 Challenger 1: Dispatch-Time LRU Invariant & Monotonic Stress Tests`.

---

## 5. Verification Method

To independently verify this verdict and reproduce the empirical results:

1. Run Challenger 2's stress test suite:
   ```bash
   swift test --filter ExecutionLogChallenger2StressTests
   ```
   Confirm all 6 adversarial stress tests execute and pass in ~0.16s.

2. Run Challenger 1's adversarial tests to observe the two defects:
   ```bash
   swift test --filter ExecutionLogLRUAdversarialTests
   ```
   Observe the 2 reproducible failures:
   - `Adversarial 10: Prove whether out-of-order older entries corrupt LRU eviction victim selection`
   - `Adversarial 11: Prove whether sparse/gapped slot indexing blocks CPU drain()`

3. Run the full project regression:
   ```bash
   swift test
   ```
   Observe test run failure with 2 issues.
