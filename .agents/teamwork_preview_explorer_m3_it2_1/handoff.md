# Handoff Report: Explorer 1 — LRUWeightTracker Monotonic Queue Remediation

**Agent**: `teamwork_preview_explorer_m3_it2_1` (Explorer 1)  
**Milestone**: Phase 2 Milestone 3 Iteration 2 (GPU Execution Log & Dispatch-Time LRU Remediation — Requirement R3)  
**Assigned Working Directory**: `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m3_it2_1`  
**Date**: 2026-09-18  

---

## 1. Observation

### 1.1 Source Code Defect Location in `LRUWeightTracker.swift`

In `Sources/AsyncMoERouter/ExecutionPipeline/LRUWeightTracker.swift` (lines 91–111):

```swift
91:     public func touch(expertKey: ExpertKey, timestamp: UInt64, slotIndex: Int? = nil) {
92:         _state.withLock { state in
93:             state.totalUpdates += 1
94: 
95:             if let node = state.map[expertKey] {
96:                 // Monotonic guard: preserve later timestamp if an older entry arrives out-of-order
97:                 if timestamp >= node.timestamp {
98:                     node.timestamp = timestamp
99:                 }
100:                 if let slot = slotIndex {
101:                     node.slotIndex = slot
102:                 }
103:                 Self._unlink(node: node)
104:                 Self._insertAfterHead(node: node, head: state.head)
105:             } else {
106:                 let node = Node(key: expertKey, timestamp: timestamp, slotIndex: slotIndex)
107:                 state.map[expertKey] = node
108:                 Self._insertAfterHead(node: node, head: state.head)
109:             }
110:         }
111:     }
```

**Observation Details**:
- Line 97: The monotonic check `if timestamp >= node.timestamp` ONLY encloses line 98 (`node.timestamp = timestamp`).
- Lines 100–104: The slot index mutation (`node.slotIndex = slot`), node unlinking (`Self._unlink(node: node)`), and MRU insertion (`Self._insertAfterHead(node: node, head: state.head)`) execute **unconditionally**.
- When an out-of-order execution log entry with an older timestamp (`timestamp < node.timestamp`) arrives:
  1. `node.timestamp` is correctly preserved at the higher numerical value.
  2. However, the node is unlinked and moved to `head.next` (MRU head) as if it had been accessed at the current time.
  3. Consequently, other resident nodes that were legitimately accessed more recently than this expert's latest access are pushed toward the tail (`tail.prev`, the LRU eviction position).

---

### 1.2 Adversarial Test Failure Reproduction

In `swift_tests/AsyncMoERouterTests/Unit/ExecutionLogLRUAdversarialTests.swift` (lines 450–481):

```swift
450:     @Test("Adversarial 10: Prove whether out-of-order older entries corrupt LRU eviction victim selection")
451:     func testEmpiricalChallengeOutOfOrderRecencyCorruption() {
452:         let tracker = LRUWeightTracker()
453:         let expertA = ExpertKey(layer: 5, expert: 1)
454:         let expertB = ExpertKey(layer: 5, expert: 2)
455: 
456:         // Step 1: Expert A was accessed at t = 1000
457:         tracker.touch(expertKey: expertA, timestamp: 1000)
458:         // Step 2: Expert B was accessed at t = 2000
459:         tracker.touch(expertKey: expertB, timestamp: 2000)
460: 
461:         // Queue order should be: MRU: B (2000), LRU: A (1000)
462:         #expect(tracker.evictionCandidate() == expertA, "Before out-of-order entry, A (ts=1000) must be LRU victim")
463: 
464:         // Step 3: Now an out-of-order entry arrives for Expert A with timestamp 500 (from the past).
465:         // Since timestamp 500 < 1000 (and < 2000), this access happened in the past.
466:         // It should NOT make Expert A more recently used than Expert B (who was accessed at 2000)!
467:         tracker.touch(expertKey: expertA, timestamp: 500)
468: 
469:         // If the monotonic guard properly protects the recency queue:
470:         // Expert A's last access is still 1000, which is older than Expert B's access at 2000.
471:         // Therefore, Expert A MUST STILL BE THE LRU VICTIM!
472:         let victim = tracker.evictionCandidate()
473:         print("[Empirical Challenger] Victim after out-of-order entry: \(String(describing: victim))")
474: 
475:         // We verify if victim is still expertA
476:         let orderPreserved = (victim == expertA)
477:         #expect(
478:             orderPreserved,
479:             "CRITICAL: Monotonic guard failed to protect recency queue! Out-of-order entry with older timestamp (500) promoted Expert A over Expert B (ts=2000), causing Expert B to become the eviction victim!"
480:         )
481:     }
```

**Verbatim Execution Command & Output**:
```bash
$ swift test --filter ExecutionLogLRUAdversarialTests
```
Output:
```text
[Empirical Challenger] Victim after out-of-order entry: Optional(L5E2)
[Challenger 1] Recency order after out-of-order entry for A(ts=500): [L5E10, L5E30, L5E20]
[Challenger 1] LRU candidate after out-of-order entry: Optional(L5E20)
...
✘ Test "Adversarial 10: Prove whether out-of-order older entries corrupt LRU eviction victim selection" recorded an issue at ExecutionLogLRUAdversarialTests.swift:477:9: Expectation failed: orderPreserved
↳ CRITICAL: Monotonic guard failed to protect recency queue! Out-of-order entry with older timestamp (500) promoted Expert A over Expert B (ts=2000), causing Expert B to become the eviction victim!
↳ orderPreserved → <not evaluated>
✘ Test "Adversarial 10: Prove whether out-of-order older entries corrupt LRU eviction victim selection" failed after 0.048 seconds with 1 issue.
```

---

## 2. Logic Chain

1. **Recency List Invariant**:
   In `LRUWeightTracker`, the doubly-linked list maintains recency order from MRU (`state.head.next`) to LRU (`state.tail.prev`).
   `evictionCandidate()` returns `state.tail.prev.key`.

2. **State Before Out-of-Order Entry**:
   - `touch(expertA, 1000)`: List is `head <-> A(1000) <-> tail`.
   - `touch(expertB, 2000)`: List is `head <-> B(2000) <-> A(1000) <-> tail`.
   - At this point, `evictionCandidate()` yields `A` because $1000 < 2000$.

3. **Out-of-Order Entry Arrival**:
   - `touch(expertA, 500)` arrives.
   - `node.timestamp` is $1000$. The incoming timestamp is $500$.
   - The comparison `timestamp >= node.timestamp` evaluates to `false` ($500 < 1000$).
   - `node.timestamp` remains $1000$ (numerical regression is avoided).

4. **Queue Inversion (The Defect)**:
   - Because `Self._unlink(node: node)` and `Self._insertAfterHead(node: node, head: state.head)` are outside the `if` block, they execute unconditionally.
   - `A` is unlinked from its tail position and re-inserted directly after `head`.
   - The list becomes `head <-> A(1000) <-> B(2000) <-> tail`.
   - `evictionCandidate()` now returns `B` (`L5E2`).

5. **Impact on MoE Router (Requirement R3 Violation)**:
   - When GPU completion handlers drain out-of-order execution log batches (e.g. from concurrent command buffers or multi-stream execution), an older batch arriving slightly later causes active, recently utilized experts (like Expert B at $t = 2000$) to be prematurely selected as eviction victims in `SpeculativeRingBuffer`.
   - This causes cache thrashing, destroys prefetch hit rates, and violates the dispatch-time hardware truth invariant required by Phase 2 R3.

6. **Remediation Logic**:
   - Unlinking and moving a node to `state.head` represents an update to its recency. An access event that occurred in the past cannot make an expert more recently used than its existing latest access.
   - Therefore, `_unlink` and `_insertAfterHead` must **only** execute when `timestamp >= node.timestamp`.
   - When `timestamp < node.timestamp`:
     - The node's position in the doubly-linked list must remain completely undisturbed.
     - `node.timestamp` must not change.
     - To handle late slot discoveries safely, if `node.slotIndex == nil`, an incoming `slotIndex` can be bound without disturbing queue order (`if node.slotIndex == nil, let slot = slotIndex { node.slotIndex = slot }`). If `node.slotIndex` already has a value, it must not be overwritten by an older entry's stale slot index.

---

## 3. Caveats

1. **First-Time Out-of-Order Arrivals**:
   If an expert key is encountered for the very first time with an older timestamp (i.e. `state.map[expertKey] == nil`), the tracker must insert it at the MRU head because it has no prior timestamp history for that key. In practice, all entries for a given expert arrive after kernel dispatch, so initial insertions establish baseline recency.

2. **Total Updates Metric**:
   `state.totalUpdates` must continue to increment unconditionally on every `touch` call, regardless of whether the entry was out-of-order. This maintains parity with `totalUpdates` accounting tested in `Adversarial 4` (asserting `totalUpdates == 4` after 4 calls, including 2 out-of-order).

3. **Separation from Defect 2 (ExecutionLog.swift Sparse Slots)**:
   `ExecutionLogLRUAdversarialTests.swift:509` (Test 11) is an independent failure caused by sparse slot hashing in `ExecutionLog.swift`'s MSL shader (`writeExecutionLogEntry`) combined with sentinel breaking in `ExecutionLog.drain()`. That defect is in `ExecutionLog.swift`, which is assigned to Explorer 2 / Worker for Iteration 2.

4. **Scope of Explorer 1**:
   In strict accordance with the read-only explorer archetype, no changes have been applied to `Sources/AsyncMoERouter/ExecutionPipeline/LRUWeightTracker.swift`. Complete blueprints and patch artifacts are provided in this directory for Worker application.

---

## 4. Conclusion & Production Blueprint

### 4.1 Assessment
The defect in `LRUWeightTracker.swift` is surgical and completely contained within `touch(expertKey:timestamp:slotIndex:)`. By moving the DLL unlinking, insertion, and slot updating inside the monotonic guard branch, and adding safe nil-only slot binding for stale entries, the recency queue is guaranteed to be strictly monotonic.

### 4.2 Code Blueprint Diff

```diff
--- a/Sources/AsyncMoERouter/ExecutionPipeline/LRUWeightTracker.swift
+++ b/Sources/AsyncMoERouter/ExecutionPipeline/LRUWeightTracker.swift
@@ -26,7 +26,7 @@
 ///    recency metadata with minimal lock contention (<15ns lock acquisition via `OSAllocatedUnfairLock`).
 ///
 /// 3. **O(1) Recency Operations**:
-///    - `touch(expertKey:timestamp:slotIndex:)`: O(1) recency promotion and timestamp update.
+///    - `touch(expertKey:timestamp:slotIndex:)`: O(1) recency promotion and timestamp update with monotonic queue protection.
 ///    - `evictionCandidate()`: O(1) non-mutating peek at LRU tail victim.
 ///    - `evictLRU()`: O(1) unlinking and removal of least-recently-used expert.
 ///    - `remove(expertKey:)`: O(1) arbitrary key invalidation.
@@ -81,8 +81,10 @@
 
     /// Records or updates the recency of an expert key following verified GPU execution.
     ///
-    /// Promotes the node to the MRU head position, updates its timestamp monotonically,
-    /// and optionally updates the associated ring buffer slot index.
+    /// Promotes the node to the MRU head position ONLY if its timestamp is monotonically non-decreasing.
+    /// If an older out-of-order entry arrives (`timestamp < node.timestamp`), its position in the
+    /// recency queue is strictly preserved to prevent stale log arrivals from demoting active experts.
+    /// If `node.slotIndex == nil`, an incoming `slotIndex` will be bound without altering queue recency.
     ///
     /// - Parameters:
     ///   - expertKey: The expert uniquely identified across layers.
@@ -93,15 +95,23 @@
             state.totalUpdates += 1
 
             if let node = state.map[expertKey] {
-                // Monotonic guard: preserve later timestamp if an older entry arrives out-of-order
+                // Monotonic guard: only update timestamp and promote to MRU if the incoming
+                // entry is newer or at the same timestamp as the expert's recorded execution.
                 if timestamp >= node.timestamp {
                     node.timestamp = timestamp
-                }
-                if let slot = slotIndex {
-                    node.slotIndex = slot
+                    if let slot = slotIndex {
+                        node.slotIndex = slot
+                    }
+                    Self._unlink(node: node)
+                    Self._insertAfterHead(node: node, head: state.head)
+                } else {
+                    // Out-of-order stale entry (timestamp < node.timestamp):
+                    // Preserve the node's position in the recency queue and protect node.timestamp from regression.
+                    // Safely bind slotIndex only if the node currently has no slot assigned.
+                    if node.slotIndex == nil, let slot = slotIndex {
+                        node.slotIndex = slot
+                    }
                 }
-                Self._unlink(node: node)
-                Self._insertAfterHead(node: node, head: state.head)
             } else {
                 let node = Node(key: expertKey, timestamp: timestamp, slotIndex: slotIndex)
                 state.map[expertKey] = node
```

### 4.3 Provided Artifacts in Working Directory
1. **Validated Patch File**:
   `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m3_it2_1/LRUWeightTracker_monotonic_guard.patch`
   (Verified via `git apply --check`)
2. **Full Replacement Blueprint**:
   `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m3_it2_1/proposed_LRUWeightTracker.swift`

### 4.4 Full Source Blueprint for `LRUWeightTracker.swift`

```swift
//===----------------------------------------------------------------------===//
//
// This source file is part of the AsyncMoERouter open source project
//
// Copyright (c) 2026 Apple Inc. and the AsyncMoERouter project authors
// Licensed under Apache License v2.0
//
//===----------------------------------------------------------------------===//

import Foundation
import Metal
import os
import os.log

/// O(1) CPU Dispatch-Time LRU Weight Tracker backed by a doubly-linked list and hash table.
///
/// # Design Principles (Phase 2 Requirement R3)
/// 1. **Post-Execution Invariant Enforcement**:
///    LRU metadata and timestamps are updated **strictly** by draining the `GPUExecutionLog`
///    after GPU kernel completion. The CPU NEVER mutates LRU timestamps based on speculative
///    pre-routing predictions. This guarantees a single, authoritative hardware truth.
///
/// 2. **Lock-Free / High-Efficiency Log Draining**:
///    Synchronizes with GPU execution via `MTLCommandBuffer.addCompletedHandler`. When the GPU
///    finishes execution, the completion block drains newly committed log entries and updates
///    recency metadata with minimal lock contention (<15ns lock acquisition via `OSAllocatedUnfairLock`).
///
/// 3. **O(1) Recency Operations**:
///    - `touch(expertKey:timestamp:slotIndex:)`: O(1) recency promotion and timestamp update with monotonic queue protection.
///    - `evictionCandidate()`: O(1) non-mutating peek at LRU tail victim.
///    - `evictLRU()`: O(1) unlinking and removal of least-recently-used expert.
///    - `remove(expertKey:)`: O(1) arbitrary key invalidation.
///
/// 4. **Speculative Ring Buffer Integration**:
///    Coordinates with `SpeculativeRingBuffer.updateLRUTimestamp(for:timestamp:)` and provides
///    safe eviction candidate cross-referencing ensuring only `.ready` slots are evicted.
public final class LRUWeightTracker: @unchecked Sendable {

    // MARK: - Doubly-Linked List Node
    private final class Node {
        let key: ExpertKey
        var timestamp: UInt64
        var slotIndex: Int?
        var prev: Node?
        var next: Node?

        init(key: ExpertKey, timestamp: UInt64, slotIndex: Int? = nil) {
            self.key = key
            self.timestamp = timestamp
            self.slotIndex = slotIndex
        }
    }

    // MARK: - Internal Synchronized State
    private struct TrackerState {
        var map: [ExpertKey: Node] = [:]
        let head: Node // Sentinel MRU head
        let tail: Node // Sentinel LRU tail
        var totalUpdates: Int = 0
        var totalEvictions: Int = 0
        var totalDrains: Int = 0

        init() {
            self.head = Node(key: ExpertKey(layer: -1, expert: -1), timestamp: .max)
            self.tail = Node(key: ExpertKey(layer: -2, expert: -2), timestamp: 0)
            self.head.next = self.tail
            self.tail.prev = self.head
        }
    }

    // MARK: - Private Properties
    private let _state: OSAllocatedUnfairLock<TrackerState>
    private let _log = Logger(subsystem: "AsyncMoERouter", category: "LRUWeightTracker")

    // MARK: - Initializer
    public init() {
        self._state = OSAllocatedUnfairLock(initialState: TrackerState())
    }

    // MARK: - O(1) Recency APIs

    /// Records or updates the recency of an expert key following verified GPU execution.
    ///
    /// Promotes the node to the MRU head position ONLY if its timestamp is monotonically non-decreasing.
    /// If an older out-of-order entry arrives (`timestamp < node.timestamp`), its position in the
    /// recency queue is strictly preserved to prevent stale log arrivals from demoting active experts.
    /// If `node.slotIndex == nil`, an incoming `slotIndex` will be bound without altering queue recency.
    ///
    /// - Parameters:
    ///   - expertKey: The expert uniquely identified across layers.
    ///   - timestamp: Authoritative continuous clock or GPU cycle timestamp.
    ///   - slotIndex: Optional ring buffer slot index currently holding these weights.
    public func touch(expertKey: ExpertKey, timestamp: UInt64, slotIndex: Int? = nil) {
        _state.withLock { state in
            state.totalUpdates += 1

            if let node = state.map[expertKey] {
                // Monotonic guard: only update timestamp and promote to MRU if the incoming
                // entry is newer or at the same timestamp as the expert's recorded execution.
                if timestamp >= node.timestamp {
                    node.timestamp = timestamp
                    if let slot = slotIndex {
                        node.slotIndex = slot
                    }
                    Self._unlink(node: node)
                    Self._insertAfterHead(node: node, head: state.head)
                } else {
                    // Out-of-order stale entry (timestamp < node.timestamp):
                    // Preserve the node's position in the recency queue and protect node.timestamp from regression.
                    // Safely bind slotIndex only if the node currently has no slot assigned.
                    if node.slotIndex == nil, let slot = slotIndex {
                        node.slotIndex = slot
                    }
                }
            } else {
                let node = Node(key: expertKey, timestamp: timestamp, slotIndex: slotIndex)
                state.map[expertKey] = node
                Self._insertAfterHead(node: node, head: state.head)
            }
        }
    }

    /// Backward-compatible alias for `touch(expertKey:timestamp:slotIndex: nil)`.
    ///
    /// - Parameters:
    ///   - expert: The expert key to record.
    ///   - timestamp: Authoritative execution timestamp.
    public func recordAccess(expert: ExpertKey, timestamp: UInt64) {
        touch(expertKey: expert, timestamp: timestamp, slotIndex: nil)
    }

    /// Returns the least-recently-used expert key without removing it from the tracker.
    ///
    /// Complexity: O(1).
    ///
    /// - Returns: The LRU `ExpertKey`, or `nil` if the tracker is empty.
    public func evictionCandidate() -> ExpertKey? {
        _state.withLock { state in
            guard let node = state.tail.prev, node !== state.head else { return nil }
            return node.key
        }
    }

    /// Returns the least-recently-used expert key and its mapped slot index without removing it.
    ///
    /// Complexity: O(1).
    public func evictionCandidateWithSlot() -> (expert: ExpertKey, slotIndex: Int?)? {
        _state.withLock { state in
            guard let node = state.tail.prev, node !== state.head else { return nil }
            return (node.key, node.slotIndex)
        }
    }

    /// Backward-compatible property alias for `evictionCandidate()`.
    public var lruExpert: ExpertKey? {
        evictionCandidate()
    }

    /// Removes and returns the least-recently-used expert key from the recency queue.
    ///
    /// Complexity: O(1).
    ///
    /// - Returns: The evicted `ExpertKey`, or `nil` if empty.
    @discardableResult
    public func evictLRU() -> ExpertKey? {
        _state.withLock { state in
            guard let node = state.tail.prev, node !== state.head else { return nil }
            Self._unlink(node: node)
            state.map.removeValue(forKey: node.key)
            state.totalEvictions += 1
            _log.debug("LRU evicted: \(node.key.description) (ts=\(node.timestamp))")
            return node.key
        }
    }

    /// Removes and returns the least-recently-used expert key along with its associated slot index.
    ///
    /// Complexity: O(1).
    @discardableResult
    public func evictLRUWithSlot() -> (expert: ExpertKey, slotIndex: Int?)? {
        _state.withLock { state in
            guard let node = state.tail.prev, node !== state.head else { return nil }
            Self._unlink(node: node)
            state.map.removeValue(forKey: node.key)
            state.totalEvictions += 1
            _log.debug("LRU evicted: \(node.key.description) slot=\(String(describing: node.slotIndex))")
            return (node.key, node.slotIndex)
        }
    }

    /// Removes a specific expert key from the tracker (e.g. on cache invalidation or slot reclaim).
    ///
    /// Complexity: O(1).
    ///
    /// - Parameter expertKey: The expert key to remove.
    /// - Returns: The previously mapped slot index, or `nil` if not tracked.
    @discardableResult
    public func remove(expertKey: ExpertKey) -> Int? {
        _state.withLock { state in
            guard let node = state.map.removeValue(forKey: expertKey) else { return nil }
            Self._unlink(node: node)
            return node.slotIndex
        }
    }

    /// Checks if a given expert key is currently tracked in the recency queue.
    ///
    /// Complexity: O(1).
    public func contains(expert: ExpertKey) -> Bool {
        _state.withLock { $0.map[expert] != nil }
    }

    /// Retrieves the active ring buffer slot index mapped to the specified expert key.
    ///
    /// Complexity: O(1).
    public func slotIndex(for expert: ExpertKey) -> Int? {
        _state.withLock { $0.map[expert]?.slotIndex }
    }

    /// Associates a ring buffer slot index with an expert key.
    public func bindSlot(slotIndex: Int, for expert: ExpertKey) {
        _state.withLock { state in
            if let node = state.map[expert] {
                node.slotIndex = slotIndex
            } else {
                let node = Node(key: expert, timestamp: 0, slotIndex: slotIndex)
                state.map[expert] = node
                Self._insertAfterHead(node: node, head: state.head)
            }
        }
    }

    /// Dissociates a slot index from an expert key without removing it from recency order.
    @discardableResult
    public func unbindSlot(for expert: ExpertKey) -> Int? {
        _state.withLock { state in
            guard let node = state.map[expert] else { return nil }
            let prev = node.slotIndex
            node.slotIndex = nil
            return prev
        }
    }

    /// Clears all tracked experts from the recency queue.
    public func clear() {
        _state.withLock { state in
            state.map.removeAll(keepingCapacity: true)
            state.head.next = state.tail
            state.tail.prev = state.head
        }
    }

    // MARK: - Post-Execution GPU Log Draining

    /// Drains all newly committed entries from the `GPUExecutionLog`, updates recency
    /// metadata for each recorded expert, and synchronizes with `SpeculativeRingBuffer`.
    ///
    /// **Authoritative Invariant (Requirement R3)**: This method is the single source of truth
    /// for mutating slot recency timestamps. Pre-routing predictions never invoke this.
    ///
    /// - Parameters:
    ///   - executionLog: The circular GPU execution log buffer.
    ///   - ringBuffer: Optional speculative ring buffer to update.
    /// - Returns: Batch array of drained `ExecutionLogEntry` records.
    @discardableResult
    public func drainAndRecord(
        from executionLog: GPUExecutionLog,
        ringBuffer: SpeculativeRingBuffer? = nil
    ) -> [ExecutionLogEntry] {
        let entries = executionLog.drain()
        guard !entries.isEmpty else { return [] }

        for entry in entries {
            let key = ExpertKey(layer: Int(entry.layerIndex), expert: Int(entry.expertID))
            let timestamp = entry.timestamp

            var mappedSlot: Int? = nil
            if let ring = ringBuffer {
                if let slot = ring.findSlot(for: key) {
                    mappedSlot = slot.index
                    ring.updateLRUTimestamp(slotIndex: slot.index, timestamp: timestamp)
                }
            }

            touch(expertKey: key, timestamp: timestamp, slotIndex: mappedSlot)
        }

        _state.withLock { $0.totalDrains += 1 }
        _log.debug("LRUWeightTracker: Drained and recorded \(entries.count) execution log entries")
        return entries
    }

    /// Attaches an asynchronous completion drain handler to an in-flight `MTLCommandBuffer`.
    ///
    /// When the GPU finishes kernel execution, Metal invokes the completion block on a background
    /// queue. The handler immediately drains the `GPUExecutionLog`, updates LRU recency metadata,
    /// updates the ring buffer, and forwards the drained entries to `onCompletion`.
    ///
    /// - Parameters:
    ///   - commandBuffer: The Metal command buffer executing compute kernels.
    ///   - executionLog: The GPU execution log written by the kernels.
    ///   - ringBuffer: Optional speculative ring buffer pool.
    ///   - onCompletion: Optional callback invoked with the batch of drained entries.
    public func registerCompletionDrain(
        on commandBuffer: any MTLCommandBuffer,
        executionLog: GPUExecutionLog,
        ringBuffer: SpeculativeRingBuffer? = nil,
        onCompletion: (@Sendable ([ExecutionLogEntry]) -> Void)? = nil
    ) {
        commandBuffer.addCompletedHandler { [weak self, weak executionLog, weak ringBuffer] buffer in
            guard let self = self, let log = executionLog else { return }

            if buffer.status == .error {
                let desc = buffer.error?.localizedDescription ?? "unknown error"
                Logger(subsystem: "AsyncMoERouter", category: "LRUWeightTracker")
                    .error("MTLCommandBuffer completed with error: \(desc)")
            }

            let drained = self.drainAndRecord(from: log, ringBuffer: ringBuffer)
            onCompletion?(drained)
        }
    }

    /// Selects the optimal eviction victim slot from `SpeculativeRingBuffer` based on
    /// the authoritative post-execution LRU history.
    ///
    /// Scans from the LRU tail toward MRU head to find the first expert currently resident
    /// in a `.ready` state in the ring buffer. `.loading`, `.inUse`, and `.abandoned` slots
    /// are strictly protected from eviction per Requirement R2.
    ///
    /// - Parameter ringBuffer: The speculative ring buffer pool.
    /// - Returns: The optimal evictable `RingBufferSlot`, or `nil` if no resident slot is ready.
    public func findEvictionCandidateSlot(in ringBuffer: SpeculativeRingBuffer) -> RingBufferSlot? {
        _state.withLock { state in
            var curr = state.tail.prev
            while let node = curr, node !== state.head {
                if let readySlot = ringBuffer.findReadySlot(for: node.key) {
                    return readySlot
                }
                curr = node.prev
            }
            return nil
        }
    }

    // MARK: - Diagnostics & Metrics

    /// Total number of active experts tracked in the recency queue.
    public var count: Int {
        _state.withLock { $0.map.count }
    }

    /// Cumulative count of recency updates performed.
    public var totalUpdates: Int {
        _state.withLock { $0.totalUpdates }
    }

    /// Cumulative count of LRU evictions performed.
    public var totalEvictions: Int {
        _state.withLock { $0.totalEvictions }
    }

    /// Cumulative count of log drain operations executed.
    public var totalDrains: Int {
        _state.withLock { $0.totalDrains }
    }

    /// Returns an ordered snapshot of tracked expert keys from MRU to LRU.
    public var allTrackedExperts: [ExpertKey] {
        _state.withLock { state in
            var list: [ExpertKey] = []
            list.reserveCapacity(state.map.count)
            var curr = state.head.next
            while let node = curr, node !== state.tail {
                list.append(node.key)
                curr = node.next
            }
            return list
        }
    }

    // MARK: - Internal Doubly-Linked List Pointer Operations (Unsynchronized)

    @inline(__always)
    private static func _unlink(node: Node) {
        node.prev?.next = node.next
        node.next?.prev = node.prev
        node.prev = nil
        node.next = nil
    }

    @inline(__always)
    private static func _insertAfterHead(node: Node, head: Node) {
        node.next = head.next
        node.prev = head
        head.next?.prev = node
        head.next = node
    }
}

// MARK: - SpeculativeRingBuffer Extension for Requirement R3

extension SpeculativeRingBuffer {
    /// Updates the slot's LRU timestamp for the specified expert after confirmed GPU execution.
    ///
    /// **Authoritative Invariant (Requirement R3)**: This method must ONLY be called
    /// when draining the GPU Execution Log. It must NEVER be called during speculative
    /// pre-routing predictions.
    ///
    /// - Parameters:
    ///   - expert: The expert key to locate in the ring buffer.
    ///   - timestamp: Authoritative continuous clock timestamp from the execution log entry.
    /// - Returns: `true` if an active resident slot was found and updated; `false` otherwise.
    @discardableResult
    public func updateLRUTimestamp(for expert: ExpertKey, timestamp: UInt64) -> Bool {
        if let slot = findSlot(for: expert) {
            updateLRUTimestamp(slotIndex: slot.index, timestamp: timestamp)
            return true
        }
        return false
    }
}
```

---

## 5. Verification Method

### 5.1 Patch Application & Dry Run Verification
```bash
git apply --check .agents/teamwork_preview_explorer_m3_it2_1/LRUWeightTracker_monotonic_guard.patch
```
Expected output: exit code 0 (no conflicts or errors).

### 5.2 Specific Test Verification Command
Once applied by the Worker, run:
```bash
swift test --filter testEmpiricalChallengeOutOfOrderRecencyCorruption
```
Expected output:
```text
✔ Test "Adversarial 10: Prove whether out-of-order older entries corrupt LRU eviction victim selection" passed
```

### 5.3 Full Adversarial & Regression Suite Execution
```bash
swift test --filter ExecutionLogLRUAdversarialTests
swift test --filter ExecutionLogTests
swift test --filter ExecutionLogChallenger2StressTests
swift test --filter Tier3_PairwiseTests
swift test --filter Tier4_WorkloadTests
```

### 5.4 Invalidation Conditions
The fix would be invalidated if:
1. An out-of-order entry with an older timestamp (`t_older < t_latest`) causes `evictionCandidate()` to change to any key other than the one with the lowest recorded timestamp.
2. `node.slotIndex` is overwritten by a stale `slotIndex` when `node.slotIndex != nil`.
3. Calls to `touch` fail to increment `totalUpdates`.
4. DLL integrity is broken under concurrent access (e.g. cycles or orphaned nodes).
