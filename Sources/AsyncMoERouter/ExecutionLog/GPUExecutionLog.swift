import Foundation
import Metal
import os.log

/// GPU-side circular ring buffer (4096 × 32-byte entries = 128 KB) written by
/// gating threadgroups to record actual expert routing decisions.
///
/// # Design Principles
/// - The CPU drains this log after each generation step to update the `LRUWeightTracker`.
/// - The GPU uses zero-atomic deterministic slot assignment (token-level stripe writes).
/// - The buffer is `.storageModeShared` for direct CPU readback.
/// - This is the **single, authoritative source** for the CPU's LRU registry.
public final class GPUExecutionLog: @unchecked Sendable {
    // MARK: - Public constants
    public let capacity: Int
    public let buffer: any MTLBuffer

    // MARK: - Private state
    private var _readHead: Int = 0
    private let _lock = NSLock()
    private let _log = Logger(subsystem: "AsyncMoERouter", category: "GPUExecutionLog")

    // MARK: - Diagnostics
    public private(set) var totalEntriesDrained: Int = 0

    // MARK: - Init

    /// Creates the shared execution log buffer.
    ///
    /// - Parameters:
    ///   - device: Metal device.
    ///   - capacity: Number of log entries (default: 4096, = 128 KB).
    public init(device: any MTLDevice, capacity: Int = 4096) {
        self.capacity = capacity
        let sizeBytes = capacity * MemoryLayout<ExecutionLogEntry>.stride
        precondition(MemoryLayout<ExecutionLogEntry>.stride == 32, "ExecutionLogEntry must be exactly 32 bytes")
        guard let buf = device.makeBuffer(length: sizeBytes, options: .storageModeShared) else {
            fatalError("GPUExecutionLog: Failed to allocate \(sizeBytes) byte execution log buffer")
        }
        buf.label = "GPUExecutionLog"
        // Zero-initialize so stale data is distinguishable (tokenIndex == 0)
        memset(buf.contents(), 0, sizeBytes)
        self.buffer = buf
        _log.info("GPUExecutionLog initialized: \(capacity) entries × 32 bytes = \(sizeBytes) bytes")
    }

    // MARK: - CPU Drain

    /// Drains all newly written entries from the GPU Execution Log.
    ///
    /// Entries are consumed from the current `readHead` until a zero-token entry is found
    /// (indicating no new GPU writes beyond that point in the ring).
    ///
    /// - Returns: Array of valid `ExecutionLogEntry` records written since last drain.
    public func drain() -> [ExecutionLogEntry] {
        _lock.lock()
        defer { _lock.unlock() }

        let ptr = buffer.contents().bindMemory(to: ExecutionLogEntry.self, capacity: capacity)
        var entries: [ExecutionLogEntry] = []

        // Scan forward from readHead in ring order, stopping at unwritten slots
        var scanned = 0
        while scanned < capacity {
            let idx = (_readHead + scanned) % capacity
            let entry = ptr[idx]
            // tokenIndex == 0 with timestamp == 0 → not written by GPU in this step
            guard entry.tokenIndex != 0 || entry.timestamp != 0 else { break }
            entries.append(entry)
            // Clear entry after reading so next drain doesn't re-process it
            ptr[idx] = ExecutionLogEntry(tokenIndex: 0, layerIndex: 0, horizonIndex: 0, expertID: 0, confidenceScore: 0)
            scanned += 1
        }

        _readHead = (_readHead + scanned) % capacity
        totalEntriesDrained += scanned
        if scanned > 0 {
            let total = totalEntriesDrained
            _log.debug("GPUExecutionLog: Drained \(scanned) entries (total=\(total))")
        }
        return entries
    }

    // MARK: - GPU MSL Shader String

    /// Embedded MSL kernel string for writing to the execution log from GPU gating threadgroups.
    ///
    /// **NOTE**: This kernel writes deterministically via a per-token stripe, avoiding atomics.
    /// Each thread's write slot is derived from its token and layer indices, ensuring zero contention.
    ///
    /// **Safety**: Buffers using the execution log must NOT be marked `.untracked`.
    /// Metal's default hazard tracking guarantees visibility ordering.
    public static let mslKernelSource: String = """
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

    kernel void writeExecutionLogEntry(
        device ExecutionLogEntry* log           [[buffer(0)]],
        constant uint&            logCapacity   [[buffer(1)]],
        constant uint&            tokenIndex    [[buffer(2)]],
        constant ushort&          layerIndex    [[buffer(3)]],
        constant ushort&          horizonIndex  [[buffer(4)]],
        constant ushort&          expertID      [[buffer(5)]],
        constant float&           confidence    [[buffer(6)]],
        uint                      tid           [[thread_position_in_grid]]
    ) {
        if (tid != 0) return; // single-thread write per invocation

        uint slot = (tokenIndex * 20u * 4u + layerIndex * 4u + horizonIndex) % logCapacity;
        device ExecutionLogEntry& entry = log[slot];
        entry.tokenIndex     = tokenIndex;
        entry.layerIndex     = layerIndex;
        entry.horizonIndex   = horizonIndex;
        entry.expertID       = expertID;
        entry.padding        = 0;
        entry.confidenceScore = confidence;
        entry.timestamp      = clock();
        entry.reserved       = 0;
    }
    """
}
