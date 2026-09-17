import Foundation
import Metal
import os

// MARK: - Weight Layout Configuration
public struct WeightLayoutConfig: Sendable, Equatable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numExperts: Int
    public let numLayers: Int
    public let bytesPerElement: Int
    public let pageAlignment: Int

    public init(
        hiddenSize: Int = 2048,
        intermediateSize: Int = 1408,
        numExperts: Int = 60,
        numLayers: Int = 20,
        bytesPerElement: Int = 2, // FP16
        pageAlignment: Int = 16_384 // 16 KB Apple Silicon page size
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numExperts = numExperts
        self.numLayers = numLayers
        self.bytesPerElement = bytesPerElement
        self.pageAlignment = pageAlignment
    }

    /// SwiGLU: gate_proj + up_proj + down_proj parameter count
    public var expertElements: Int {
        3 * (intermediateSize * hiddenSize)
    }

    /// Size in bytes of a single expert's weights (17,301,504 bytes for FP16 Qwen-MoE)
    public var expertSizeBytes: Int {
        expertElements * bytesPerElement
    }

    /// Total file size required for all layers and experts
    public var totalExpectedSizeBytes: Int {
        numLayers * numExperts * expertSizeBytes
    }

    /// Production configuration matching Qwen/Qwen1.5-MoE-A2.7B
    public static let qwen15MoEA27B = WeightLayoutConfig()

    /// Synthetic configuration for fast CI/CD and unit testing
    public static let synthetic = WeightLayoutConfig(
        hiddenSize: 64,
        intermediateSize: 64,
        numExperts: 16,
        numLayers: 4,
        bytesPerElement: 2,
        pageAlignment: 16_384
    )
}

// MARK: - Weight File Errors
public enum WeightFileError: Error, LocalizedError, Sendable {
    case fileNotFound(URL)
    case notARegularFile(URL)
    case unreadableFile(URL)
    case fileTooSmall(expectedMin: Int, actual: Int)
    case invalidLayout(details: String)
    case outOfBoundsRead(offset: Int, size: Int, fileSize: Int)
    case targetBufferOverflow(required: Int, bufferLength: Int)
    case metalIOError(String)
    case handleClosed

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let url):
            return "Weight file not found at: \(url.path)"
        case .notARegularFile(let url):
            return "Path at \(url.path) is a directory or special file, not a regular file."
        case .unreadableFile(let url):
            return "Weight file at \(url.path) is not readable (check permissions)."
        case .fileTooSmall(let exp, let act):
            return "Weight file size (\(act) bytes) is smaller than required minimum (\(exp) bytes)."
        case .invalidLayout(let details):
            return "Invalid weight file layout: \(details)"
        case .outOfBoundsRead(let offset, let size, let fileSize):
            return "Out-of-bounds read: offset \(offset) + size \(size) exceeds file size \(fileSize)."
        case .targetBufferOverflow(let req, let len):
            return "Target buffer overflow: required \(req) bytes exceeds buffer length \(len) bytes."
        case .metalIOError(let desc):
            return "Metal Fast I/O error: \(desc)"
        case .handleClosed:
            return "Operation attempted on a closed WeightFileHandle."
        }
    }
}

// MARK: - WeightFileHandle
/// Wraps an `MTLIOFileHandle` for raw binary weight files with bounds checking and file metadata.
public final class WeightFileHandle: @unchecked Sendable {
    /// File system URL of the weight binary file.
    public let fileURL: URL

    /// Alias matching URL property.
    public var url: URL {
        fileURL
    }

    /// Total size of the file on disk in bytes.
    public let fileSize: Int

    /// Layout configuration specifying dimensions and alignment.
    public let layout: WeightLayoutConfig

    /// Native Metal Fast I/O file handle.
    public let ioFileHandle: any MTLIOFileHandle

    /// Alias for native handle.
    public var rawHandle: any MTLIOFileHandle {
        ioFileHandle
    }

    private let stateLock = OSAllocatedUnfairLock(initialState: false) // tracks isClosed

    /// Opens a weight file for Fast I/O DMA access on the given Metal device.
    public init(
        url: URL,
        device: any MTLDevice,
        layout: WeightLayoutConfig = .qwen15MoEA27B,
        validateFullModelSize: Bool = false
    ) throws {
        self.fileURL = url
        self.layout = layout

        let path = url.path
        let fm = FileManager.default

        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: path, isDirectory: &isDir) else {
            throw FastIOError.fileNotFound(url: url)
        }
        guard !isDir.boolValue else {
            throw FastIOError.fileOpenFailed(url: url, reason: "Path is a directory, not a regular file.")
        }
        guard fm.isReadableFile(atPath: path) else {
            throw FastIOError.fileOpenFailed(url: url, reason: "File is not readable.")
        }

        do {
            let attrs = try fm.attributesOfItem(atPath: path)
            guard let size = attrs[.size] as? NSNumber else {
                throw FastIOError.fileOpenFailed(url: url, reason: "Unable to determine file size attribute.")
            }
            let actualSize = size.intValue
            if validateFullModelSize && actualSize < layout.totalExpectedSizeBytes {
                throw WeightFileError.fileTooSmall(expectedMin: layout.totalExpectedSizeBytes, actual: actualSize)
            }
            self.fileSize = actualSize
        } catch let err as FastIOError {
            throw err
        } catch let err as WeightFileError {
            throw err
        } catch {
            throw FastIOError.fileOpenFailed(url: url, reason: error.localizedDescription)
        }

        do {
            self.ioFileHandle = try device.makeIOFileHandle(url: url)
            self.ioFileHandle.label = "WeightFileHandle_\(url.lastPathComponent)"
        } catch {
            throw FastIOError.fileOpenFailed(url: url, reason: error.localizedDescription)
        }
    }

    /// Convenience initializer supporting alternative argument order: `(device:url:layout:)`.
    public convenience init(
        device: any MTLDevice,
        url: URL,
        layout: WeightLayoutConfig = .qwen15MoEA27B,
        validateFullModelSize: Bool = false
    ) throws {
        try self.init(url: url, device: device, layout: layout, validateFullModelSize: validateFullModelSize)
    }

    /// Returns true if the handle has been explicitly closed.
    public var isClosed: Bool {
        stateLock.withLock { $0 }
    }

    /// Validates that a requested byte offset and size fall strictly within the file boundaries.
    public func validateBounds(offset: Int, size: Int) throws {
        guard !isClosed else {
            throw WeightFileError.handleClosed
        }
        guard offset >= 0, size > 0, (offset + size) <= fileSize else {
            throw FastIOError.offsetOutOfBounds(offset: offset, size: size, fileSize: fileSize)
        }
    }

    /// Computes the exact file byte offset for a specific deep expert in a contiguous weights file.
    public static func offset(forGlobalExpertIndex globalExpertIndex: Int, expertSizeBytes: Int) -> Int {
        globalExpertIndex * expertSizeBytes
    }

    /// Computes the exact file byte offset for a given layer and expert index.
    public func fileOffset(layerIndex: Int, expertIndex: Int) throws -> Int {
        guard !isClosed else {
            throw WeightFileError.handleClosed
        }
        guard layerIndex >= 0 && layerIndex < layout.numLayers else {
            throw WeightFileError.invalidLayout(
                details: "Layer index \(layerIndex) out of bounds [0, \(layout.numLayers))."
            )
        }
        guard expertIndex >= 0 && expertIndex < layout.numExperts else {
            throw WeightFileError.invalidLayout(
                details: "Expert index \(expertIndex) out of bounds [0, \(layout.numExperts))."
            )
        }

        let globalIndex = (layerIndex * layout.numExperts) + expertIndex
        let offset = globalIndex * layout.expertSizeBytes
        try validateBounds(offset: offset, size: layout.expertSizeBytes)
        return offset
    }

    /// Encodes a direct DMA load of an expert weight block into the destination `MTLBuffer`.
    public func encodeLoadExpert(
        layerIndex: Int,
        expertIndex: Int,
        into targetBuffer: any MTLBuffer,
        targetOffset: Int = 0,
        commandBuffer: any MTLIOCommandBuffer
    ) throws {
        guard !isClosed else {
            throw WeightFileError.handleClosed
        }

        let sourceOffset = try fileOffset(layerIndex: layerIndex, expertIndex: expertIndex)
        let size = layout.expertSizeBytes

        guard targetOffset + size <= targetBuffer.length else {
            throw FastIOError.bufferTooSmall(
                required: targetOffset + size,
                actual: targetBuffer.length
            )
        }

        commandBuffer.load(
            targetBuffer,
            offset: targetOffset,
            size: size,
            sourceHandle: ioFileHandle,
            sourceHandleOffset: sourceOffset
        )
    }

    /// Explicitly closes the handle and invalidates future read encoding.
    public func close() {
        stateLock.withLock { $0 = true }
    }

    deinit {
        close()
    }
}
