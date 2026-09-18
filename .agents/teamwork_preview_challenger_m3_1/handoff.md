# Handoff Report: Milestone 3 Challenger 1 — Empirical LRU Invariant & Monotonic Guard Challenge

**Agent**: `teamwork_preview_challenger_m3_1` (Challenger 1)  
**Milestone**: Phase 2 Milestone 3 (GPU Execution Log & Dispatch-Time LRU Tracking — Requirement R3)  
**Assigned Working Directory**: `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_challenger_m3_1`  
**Verdict**: `REQUEST_CHANGES`  
**Date**: 2026-09-18  

---

## Challenge Summary

**Overall risk assessment**: **HIGH**  
While the core architecture of pre-routing speculative isolation, ring buffer eviction preference for unexecuted slots, zero-atomic circular GPU logging, and completion handler draining is well-constructed and passed 9 of 11 adversarial tests, **two critical defects were empirically proven and reproduced**:
1. **Monotonic Guard Queue Corruption**: In `LRUWeightTracker.swift` (`touch(expertKey:timestamp:slotIndex:)`), while `node.timestamp` is protected from numerical regression, `_unlink(node)` and `_insertAfterHead(node, head: state.head)` execute unconditionally. Stale out-of-order log entries from the past promote the expert to the MRU head, causing genuinely newer, recently executed experts to be demoted to the LRU tail and evicted.
2. **Sparse/Gapped Slot Indexing Drain Deadlock**: In `ExecutionLog.swift`, `writeExecutionLogEntry` MSL shader uses `slot = (tokenIndex * 20u * 4u + layerIndex * 4u + horizonIndex) & (logCapacity - 1u)`. For token 0, layer 5, horizon 1, this writes to slot 21. Slots 0..20 are empty (all zeroed). When the CPU calls `drain()`, the sentinel test (`guard entry.tokenIndex != 0 || entry.timestamp != 0 || entry.confidenceScore != 0 else { break }`) triggers immediately on slot 0 and aborts with 0 entries drained. `readHead` is permanently stuck at 0, making entries written by `writeExecutionLogEntry` permanently lost and unreadable.

---

## 1. Observation

### 1.1 Source Code Observations

1. **`Sources/AsyncMoERouter/ExecutionPipeline/LRUWeightTracker.swift` (lines 91-111)**:
   ```swift
   public func touch(expertKey: ExpertKey, timestamp: UInt64, slotIndex: Int? = nil) {
       _state.withLock { state in
           state.totalUpdates += 1

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
           } else {
               let node = Node(key: expertKey, timestamp: timestamp, slotIndex: slotIndex)
               state.map[expertKey] = node
               Self._insertAfterHead(node: node, head: state.head)
           }
       }
   }
   ```
   **Observation**: Notice lines 97-99 vs 103-104. When `node` is found, the conditional check `if timestamp >= node.timestamp` ONLY guards `node.timestamp = timestamp`. The pointer operations `Self._unlink(node: node)` and `Self._insertAfterHead(node: node, head: state.head)` are outside the `if` block and execute unconditionally on every call to `touch`, even when `timestamp < node.timestamp`.

2. **`Sources/AsyncMoERouter/ExecutionPipeline/ExecutionLog.swift` (lines 157-180 and 247)**:
   ```swift
   while scanned < capacity {
       let slot = (state.readHead + scanned) & mask
       let entry = ptr[slot]
       
       // Sentinel test: unwritten slots have tokenIndex == 0 AND timestamp == 0 AND confidenceScore == 0
       guard entry.tokenIndex != 0 || entry.timestamp != 0 || entry.confidenceScore != 0 else {
           break
       }
       
       entries.append(entry)
       ...
       scanned += 1
   }
   ```
   And in MSL kernel `writeExecutionLogEntry` (line 247):
   ```metal
   // Deterministic slot calculation: zero atomics, single-cycle bitwise masking
   uint slot = (tokenIndex * 20u * 4u + layerIndex * 4u + horizonIndex) & (logCapacity - 1u);
   ```
   **Observation**: In Qwen1.5-MoE-A2.7B, routed layers are 5..24 and lookahead horizons are 1..3. For `tokenIndex = 0, layerIndex = 5, horizonIndex = 1`, `slot = (0 + 20 + 1) = 21`. Slots 0..20 are all zeros. When `drain()` starts at `readHead = 0`, slot 0 evaluates the sentinel guard to false, breaks immediately (`scanned = 0`), returns 0 entries, and leaves `readHead = 0`.

### 1.2 Test Execution Observations

We authored and executed `swift_tests/AsyncMoERouterTests/Unit/ExecutionLogLRUAdversarialTests.swift`:
```bash
swift test --filter ExecutionLogLRUAdversarialTests
```

**Verbatim Test Output**:
```text
◇ Suite "Milestone 3 Challenger 1: Dispatch-Time LRU Invariant & Monotonic Stress Tests" started.
[Empirical Challenger] Drained count from sparse slot 21: 0
✔ Test "Adversarial 4: Monotonic timestamp guard prevents numerical regression when out-of-order entries arrive" passed after 0.043 seconds.
[Empirical Challenger] Victim after out-of-order entry: Optional(L5E2)
[Challenger 1] Recency order after out-of-order entry for A(ts=500): [L5E10, L5E30, L5E20]
[Challenger 1] LRU candidate after out-of-order entry: Optional(L5E20)
✘ Test "Adversarial 11: Prove whether sparse/gapped slot indexing blocks CPU drain()" recorded an issue at ExecutionLogLRUAdversarialTests.swift:509:9: Expectation failed: foundEntry
↳ CRITICAL: ExecutionLog.drain() stopped at empty slot 0 and failed to drain valid entry written at slot 21 by writeExecutionLogEntry!
↳ foundEntry → <not evaluated>
✔ Test "Adversarial 7: Reversed timestamp ingestion sequence (descending order)" passed after 0.043 seconds.
✔ Test "Adversarial 9: Slot index binding, unbinding, and replacement invariants" passed after 0.043 seconds.
✘ Test "Adversarial 11: Prove whether sparse/gapped slot indexing blocks CPU drain()" failed after 0.043 seconds with 1 issue.
✔ Test "Adversarial 6: Duplicate log entries for identical tokens and experts preserve doubly-linked list integrity" passed after 0.043 seconds.
✔ Test "Adversarial 5: Recency queue ordering under out-of-order log arrivals" passed after 0.043 seconds.
✘ Test "Adversarial 10: Prove whether out-of-order older entries corrupt LRU eviction victim selection" recorded an issue at ExecutionLogLRUAdversarialTests.swift:477:9: Expectation failed: orderPreserved
↳ CRITICAL: Monotonic guard failed to protect recency queue! Out-of-order entry with older timestamp (500) promoted Expert A over Expert B (ts=2000), causing Expert B to become the eviction victim!
↳ orderPreserved → <not evaluated>
✘ Test "Adversarial 10: Prove whether out-of-order older entries corrupt LRU eviction victim selection" failed after 0.043 seconds with 1 issue.
✔ Test "Adversarial 3: Drain-and-record is the single authoritative source of recency updates" passed after 0.043 seconds.
✔ Test "Adversarial 2: Ring buffer eviction strictly prioritizes unexecuted speculative slots over executed slots regardless of allocation order" passed after 0.044 seconds.
✔ Test "Adversarial 1A: Speculative prefetch, loading, readying, in-use, release, and abandonment NEVER modify LRU timestamps" passed after 0.044 seconds.
✔ Test "Adversarial 8: MTLCommandBuffer addCompletedHandler executes drain strictly after GPU kernel finishes" passed after 0.045 seconds.
✘ Suite "Milestone 3 Challenger 1: Dispatch-Time LRU Invariant & Monotonic Stress Tests" failed after 0.045 seconds with 2 issues.
✘ Test run with 11 tests in 1 suite failed after 0.045 seconds with 2 issues.
```

---

## 2. Logic Chain

### 2.1 Challenge 1: Recency Queue Corruption on Out-of-Order Log Entries (Defect 1)

1. **Premise 1**: In `LRUWeightTracker`, `head.next` represents the Most Recently Used (MRU) expert, and `tail.prev` represents the Least Recently Used (LRU) expert.
2. **Premise 2**: When `Expert A` is accessed at $t = 1000$ and `Expert B` is accessed at $t = 2000$, `Expert B` is more recently used than `Expert A`. The recency list is `Head -> B(2000) -> A(1000) -> Tail`, and `evictionCandidate()` correctly returns `Expert A`.
3. **Premise 3**: Suppose an out-of-order log entry arrives for `Expert A` with an older timestamp $t = 500$ (e.g., from an earlier command buffer or delayed completion callback).
4. **Premise 4**: In `LRUWeightTracker.swift`, line 97 checks `if timestamp >= node.timestamp`. Since $500 < 1000$, this condition is false, so `node.timestamp` is correctly preserved at $1000$.
5. **Premise 5**: However, lines 103-104 execute unconditionally:
   `Self._unlink(node: node)`
   `Self._insertAfterHead(node: node, head: state.head)`
   This unlinks `Expert A` and places it at the MRU head position directly after `state.head`.
6. **Deduction 6**: The recency list becomes `Head -> A(1000) -> B(2000) -> Tail`.
7. **Deduction 7**: `evictionCandidate()` (which reads `state.tail.prev`) now returns `Expert B` (`Optional(L5E2)`).
8. **Conclusion 8**: A delayed, stale log entry from the past ($t = 500$) causes the router to evict `Expert B` (which was legitimately used at $t = 2000$) instead of `Expert A` (last used at $t = 1000$). The monotonic timestamp guard fails to protect the recency queue from out-of-order corruption.

### 2.2 Challenge 2: Sentinel Break Blocks Draining on Gapped/Sparse Slot Formulations (Defect 2)

1. **Premise 1**: In `ExecutionLog.swift`, `writeExecutionLogEntry` MSL shader implements:
   `uint slot = (tokenIndex * 20u * 4u + layerIndex * 4u + horizonIndex) & (logCapacity - 1u);`
2. **Premise 2**: MoE deep layers are defined as layers 5 through 24, and horizons are 1 through 3.
3. **Premise 3**: For `tokenIndex = 0, layerIndex = 5, horizonIndex = 1`, `slot = 0 + 20 + 1 = 21`.
4. **Premise 4**: Slots 0 through 20 are initialized to zero (`memset(0)`) and remain 0.
5. **Premise 5**: When CPU calls `drain()`, the loop begins at `readHead = 0` and inspects slot 0:
   `guard entry.tokenIndex != 0 || entry.timestamp != 0 || entry.confidenceScore != 0 else { break }`
6. **Premise 6**: Because slot 0 is zero, the guard condition evaluates to false, immediately triggering `break`.
7. **Deduction 7**: The loop terminates with `scanned = 0`. `state.readHead` is updated to `(0 + 0) & mask = 0`. `drain()` returns `[]` (0 entries).
8. **Conclusion 8**: Valid entries written by `writeExecutionLogEntry` at non-zero base slots (such as slot 21) are never drained, `readHead` never advances, and the execution log remains permanently stuck, violating Requirement R3.

---

## 3. Stress Test Results

| Test Case | Scenario | Expected Behavior | Actual Behavior | Result |
|---|---|---|---|---|
| **Adversarial 1A** | Speculative prefetch lifecycle (.loading, .ready, .inUse, release, .abandoned, reclaim) | All slot timestamps remain 0; `LRUWeightTracker` remains empty (`count == 0`) | All timestamps remain 0, tracker count 0 | **PASS** |
| **Adversarial 2** | Full ring buffer (4 executed slots ts>0, 4 speculative slots ts=0) + new allocation | Evicts speculative slots (ts=0) first, NEVER executed slots | Speculative slots evicted first; executed slots preserved | **PASS** |
| **Adversarial 3** | Post-execution drain with `drainAndRecord` | Only executed slots updated; unexecuted remain ts=0; idempotent second drain | Drained 2 entries; unexecuted slots remain 0; second drain empty | **PASS** |
| **Adversarial 4** | Out-of-order older entries (t=1,000,000 then t=500,000) | `node.timestamp` does not regress numerically | Preserved at 1,000,000; advanced to 2,000,000 on newer | **PASS** |
| **Adversarial 5** | Recency queue ordering under out-of-order log arrivals | Diagnostics inspection of queue state | Logged ordering transition | **PASS** |
| **Adversarial 6** | 500 duplicate entries with identical token/expert/timestamp | No cycle leaks in DLL; count stays 1; clean single eviction | Count remains 1; cleanly evicted | **PASS** |
| **Adversarial 7** | 20 entries ingested with descending timestamps (reversed order) | Sequential eviction drains all 20 without crash or hang | All 20 evicted cleanly; count returns to 0 | **PASS** |
| **Adversarial 8** | `MTLCommandBuffer.addCompletedHandler` drain lifecycle | Timestamp remains 0 until GPU completes; updated post-completion | Pre-completion ts=0; post-completion ts=999,999 | **PASS** |
| **Adversarial 9** | Slot binding, migration, and unbinding invariants | Slot index updated/cleared while node recency is preserved | Slot bound, rebound, and unbound cleanly | **PASS** |
| **Adversarial 10** | Stale out-of-order entry ($t = 500$) delivered for Expert A ($t = 1000$) after Expert B ($t = 2000$) | Expert A remains LRU victim because $1000 < 2000$ | Expert A promoted to MRU head; Expert B (ts=2000) evicted | **FAIL (Defect 1)** |
| **Adversarial 11** | Sparse slot 21 written by `writeExecutionLogEntry`, slots 0..20 empty | CPU `drain()` successfully returns entry from slot 21 | `drain()` stops at empty slot 0; returns 0 entries; slot 21 lost | **FAIL (Defect 2)** |

---

## 4. Caveats

- **Scope of Challenge**: Tested purely within the Swift/Metal subsystem contracts for Milestone 3 (ExecutionLog and LRUWeightTracker).
- **Concurrency Test Lock Contention**: Running full `swift test` across all suites concurrently on Apple Silicon can cause Fast I/O command queue saturation if earlier suites (such as M2 Challenger 1's 250-iteration test) commit command buffers without awaiting GPU execution. This is a known harness constraint and does not affect the correctness of M3.
- No other caveats.

---

## 5. Conclusion & Recommendations

### Verdict: `REQUEST_CHANGES`

The worker must address the two identified defects before Milestone 3 can be approved:

### Actionable Remediation Steps for Worker:

1. **Fix Defect 1 in `Sources/AsyncMoERouter/ExecutionPipeline/LRUWeightTracker.swift`**:
   In `touch(expertKey:timestamp:slotIndex:)`, move `slotIndex` assignment, `_unlink`, and `_insertAfterHead` INSIDE the monotonic timestamp check:
   ```swift
   if let node = state.map[expertKey] {
       // Monotonic guard: preserve later timestamp and only promote to MRU if newer or equal
       if timestamp >= node.timestamp {
           node.timestamp = timestamp
           if let slot = slotIndex {
               node.slotIndex = slot
           }
           Self._unlink(node: node)
           Self._insertAfterHead(node: node, head: state.head)
       }
       // Stale out-of-order entries (timestamp < node.timestamp) are ignored,
       // preserving both node.timestamp and the legitimate recency queue order!
   } else {
       let node = Node(key: expertKey, timestamp: timestamp, slotIndex: slotIndex)
       state.map[expertKey] = node
       Self._insertAfterHead(node: node, head: state.head)
   }
   ```

2. **Fix Defect 2 in `Sources/AsyncMoERouter/ExecutionPipeline/ExecutionLog.swift`**:
   Align the slot indexing formula in `writeExecutionLogEntry` and `drain()`:
   - **Option A (Recommended)**: Make `writeExecutionLogEntry` use contiguous indexing matching `writeTokenGatingLog`, e.g.:
     `uint slot = (tokenIndex * topK + rank) & (logCapacity - 1u);` or `(baseSlotIndex + thread_position_in_grid) & (logCapacity - 1u)`.
   - **Option B**: If sparse slot formulas like `(tokenIndex * 80 + layerIndex * 4 + horizonIndex)` are desired, `drain()` cannot use an immediate `break` on the first zero slot encountered; it must scan through the known active slot range or use a ring occupancy bitmap / generation counter.

---

## 6. Verification Method

To independently reproduce the failures:
```bash
swift test --filter ExecutionLogLRUAdversarialTests
```
Observe that:
1. Test 10 fails with: `CRITICAL: Monotonic guard failed to protect recency queue! Out-of-order entry with older timestamp (500) promoted Expert A over Expert B (ts=2000), causing Expert B to become the eviction victim!`
2. Test 11 fails with: `CRITICAL: ExecutionLog.drain() stopped at empty slot 0 and failed to drain valid entry written at slot 21 by writeExecutionLogEntry!`

After applying the mitigations, re-run:
```bash
swift test --filter ExecutionLogLRUAdversarialTests
```
Confirm that all 11 adversarial tests pass cleanly in <0.05s.
