import Foundation
import os.log

/// O(1) LRU tracker for expert weight slots backed by a doubly-linked list + hash table.
///
/// # Update Policy
/// Timestamps are updated **strictly** by draining the `GPUExecutionLog` after each
/// generation step. The CPU never writes LRU state based on its own speculative
/// pre-routing predictions. This preserves a single, deterministic source of truth.
///
/// # Complexity
/// - `recordAccess`: O(1)
/// - `lruExpert`: O(1)
/// - `evictLRU`: O(1)
public final class LRUWeightTracker: @unchecked Sendable {
    // MARK: - Doubly linked list node
    private final class Node {
        var key: ExpertKey
        var timestamp: UInt64
        var prev: Node?
        var next: Node?
        init(key: ExpertKey, timestamp: UInt64) {
            self.key = key
            self.timestamp = timestamp
        }
    }

    // MARK: - Private state
    private var _map: [ExpertKey: Node] = [:]
    private let _head: Node      // sentinel (MRU side)
    private let _tail: Node      // sentinel (LRU side)
    private let _lock = NSLock()
    private let _log = Logger(subsystem: "AsyncMoERouter", category: "LRUWeightTracker")
    private var _totalUpdates: Int = 0

    // MARK: - Init
    public init() {
        _head = Node(key: ExpertKey(layer: -1, expert: -1), timestamp: .max)
        _tail = Node(key: ExpertKey(layer: -2, expert: -2), timestamp: 0)
        _head.next = _tail
        _tail.prev = _head
    }

    // MARK: - Public API

    /// Records that `expert` was actually consumed by the GPU, using the timestamp from
    /// the Execution Log entry. This is the **only** permitted path for mutating LRU state.
    public func recordAccess(expert: ExpertKey, timestamp: UInt64) {
        _lock.lock()
        defer { _lock.unlock() }
        _totalUpdates += 1

        if let node = _map[expert] {
            // Update timestamp and promote to MRU head
            node.timestamp = timestamp
            _removeNode(node)
            _insertAfterHead(node)
        } else {
            let node = Node(key: expert, timestamp: timestamp)
            _map[expert] = node
            _insertAfterHead(node)
        }
    }

    /// Returns the least-recently-used expert key (the LRU victim), or `nil` if empty.
    public var lruExpert: ExpertKey? {
        _lock.lock()
        defer { _lock.unlock() }
        return _tail.prev === _head ? nil : _tail.prev?.key
    }

    /// Removes and returns the LRU expert from the tracker.
    @discardableResult
    public func evictLRU() -> ExpertKey? {
        _lock.lock()
        defer { _lock.unlock() }
        guard let node = _tail.prev, node !== _head else { return nil }
        _removeNode(node)
        _map.removeValue(forKey: node.key)
        _log.debug("LRU evicted: \(node.key.description) (ts=\(node.timestamp))")
        return node.key
    }

    /// Returns whether a given expert key is currently tracked (i.e., resident).
    public func contains(expert: ExpertKey) -> Bool {
        _lock.lock()
        defer { _lock.unlock() }
        return _map[expert] != nil
    }

    public var count: Int {
        _lock.lock(); defer { _lock.unlock() }
        return _map.count
    }

    public var totalUpdates: Int {
        _lock.lock(); defer { _lock.unlock() }
        return _totalUpdates
    }

    // MARK: - Linked list internals (must be called under _lock)
    private func _removeNode(_ node: Node) {
        node.prev?.next = node.next
        node.next?.prev = node.prev
        node.prev = nil
        node.next = nil
    }

    private func _insertAfterHead(_ node: Node) {
        node.next = _head.next
        node.prev = _head
        _head.next?.prev = node
        _head.next = node
    }
}
