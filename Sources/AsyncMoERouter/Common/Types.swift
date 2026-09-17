import Foundation
import Metal

/// Uniquely identifies a routed expert in the MoE model across layers.
public struct ExpertKey: Hashable, Sendable, CustomStringConvertible {
    /// Transformer layer index (0-based, deep layers: 5..24).
    public let layerIndex: Int
    /// Expert index within the layer (0..59 for 60 experts).
    public let expertIndex: Int

    public init(layer: Int, expert: Int) {
        self.layerIndex = layer
        self.expertIndex = expert
    }

    public var description: String {
        "L\(layerIndex)E\(expertIndex)"
    }

    /// Global flat index across all deep MoE layers.
    public func globalIndex(startLayer: Int, expertsPerLayer: Int) -> Int {
        (layerIndex - startLayer) * expertsPerLayer + expertIndex
    }
}

/// Lifecycle states for slots within the speculative Ring Buffer Pool.
public enum SlotState: Equatable, Sendable {
    /// Slot is unoccupied and ready for allocation.
    case free
    /// Speculative I/O DMA read is in-flight on the speculativeQueue (PriorityLow).
    case loading(ticket: UInt64, expert: ExpertKey)
    /// Weights have arrived via DMA; ticket signaled on speculativeSharedEvent.
    case ready(ticket: UInt64, expert: ExpertKey)
    /// GPU kernel execution is currently reading from this buffer slot.
    case inUse(ticket: UInt64, expert: ExpertKey, retainCount: Int)
    /// Demoted/cancelled due to cache-miss demand fetch. Completion handler will drop signal and reclaim to .free.
    case abandoned(ticket: UInt64, expert: ExpertKey)
}

/// A pre-allocated unified memory slot inside the speculative Ring Buffer Pool.
public final class RingBufferSlot: @unchecked Sendable {
    /// Zero-based slot index in the pool array.
    public let index: Int
    /// Alias matching PROJECT.md interface contract.
    public var slotIndex: Int { index }
    /// Dedicated MTLBuffer allocated in .storageModeShared.
    public let buffer: any MTLBuffer
    /// Current lifecycle state.
    public var state: SlotState
    /// Timestamp of most recent actual GPU execution consumption (updated ONLY via Execution Log drain).
    public var lastAccessedTimestamp: UInt64
    /// Weak reference to active speculative MTLIOCommandBuffer for cooperative cancellation.
    public weak var inFlightIOCommand: (any MTLIOCommandBuffer)?
    /// Dedicated MTLSharedEvent for this slot to eliminate out-of-order race conditions.
    public var sharedEvent: (any MTLSharedEvent)?
    /// Signal ticket value expected for the active or most recent DMA transfer into this slot.
    public var signalValue: UInt64

    public init(
        index: Int,
        buffer: any MTLBuffer,
        sharedEvent: (any MTLSharedEvent)? = nil,
        signalValue: UInt64 = 0
    ) {
        self.index = index
        self.buffer = buffer
        self.state = .free
        self.lastAccessedTimestamp = 0
        self.inFlightIOCommand = nil
        self.sharedEvent = sharedEvent
        self.signalValue = signalValue
    }
}

/// Exact 32-byte C-layout entry written by GPU gating threadgroups into the circular Execution Log.
@frozen
public struct ExecutionLogEntry: Sendable, Equatable {
    /// Monotonic global token index. (Bytes 0..3)
    public var tokenIndex: UInt32
    /// Transformer layer index (5..24). (Bytes 4..5)
    public var layerIndex: UInt16
    /// Lookahead horizon (1..3 for T+1..T+3). (Bytes 6..7)
    public var horizonIndex: UInt16
    /// Selected expert ID (0..59). (Bytes 8..9)
    public var expertID: UInt16
    /// 2-byte alignment padding for Float32 boundary. (Bytes 10..11)
    public var padding: UInt16
    /// Calibrated routing confidence probability [0.0, 1.0]. (Bytes 12..15)
    public var confidenceScore: Float32
    /// Mach continuous clock timestamp or GPU execution cycle. (Bytes 16..23)
    public var timestamp: UInt64
    /// 8-byte reserved field for future telemetry / 32-byte alignment. (Bytes 24..31)
    public var reserved: UInt64

    public init(
        tokenIndex: UInt32,
        layerIndex: UInt16,
        horizonIndex: UInt16,
        expertID: UInt16,
        confidenceScore: Float32,
        timestamp: UInt64 = 0,
        reserved: UInt64 = 0
    ) {
        self.tokenIndex = tokenIndex
        self.layerIndex = layerIndex
        self.horizonIndex = horizonIndex
        self.expertID = expertID
        self.padding = 0
        self.confidenceScore = confidenceScore
        self.timestamp = timestamp
        self.reserved = reserved
    }
}

/// Errors raised during Metal 3 Fast I/O operations and file reads.
public enum FastIOError: Error, CustomStringConvertible, Sendable {
    case fileNotFound(url: URL)
    case fileOpenFailed(url: URL, reason: String)
    case bufferTooSmall(required: Int, actual: Int)
    case offsetOutOfBounds(offset: Int, size: Int, fileSize: Int)
    case queueCreationFailed(priority: String, reason: String)
    case sharedEventCreationFailed
    case ioExecutionFailed(status: Int, description: String?)
    case cancelled

    public var description: String {
        switch self {
        case .fileNotFound(let url):
            return "FastIOError: File not found at \(url.path)"
        case .fileOpenFailed(let url, let reason):
            return "FastIOError: Failed to open MTLIOFileHandle for \(url.path): \(reason)"
        case .bufferTooSmall(let required, let actual):
            return "FastIOError: Target buffer length (\(actual)) is smaller than required read size (\(required))."
        case .offsetOutOfBounds(let offset, let size, let fileSize):
            return "FastIOError: Requested range [\(offset)..<\(offset + size)] exceeds file size (\(fileSize))."
        case .queueCreationFailed(let priority, let reason):
            return "FastIOError: Failed to create MTLIOCommandQueue (priority: \(priority)): \(reason)"
        case .sharedEventCreationFailed:
            return "FastIOError: Failed to allocate MTLSharedEvent."
        case .ioExecutionFailed(let status, let desc):
            return "FastIOError: MTLIOCommandBuffer failed with status \(status): \(desc ?? "unknown")"
        case .cancelled:
            return "FastIOError: MTLIOCommandBuffer was cancelled."
        }
    }
}
