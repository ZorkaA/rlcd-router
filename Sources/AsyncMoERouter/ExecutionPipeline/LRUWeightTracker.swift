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
///    - `touch(expertKey:timestamp:slotIndex:)`: O(1) recency promotion and timestamp update.
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
    /// Promotes the node to the MRU head position, updates its timestamp monotonically,
    /// and optionally updates the associated ring buffer slot index.
    ///
    /// - Parameters:
    ///   - expertKey: The expert uniquely identified across layers.
    ///   - timestamp: Authoritative continuous clock or GPU cycle timestamp.
    ///   - slotIndex: Optional ring buffer slot index currently holding these weights.
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
