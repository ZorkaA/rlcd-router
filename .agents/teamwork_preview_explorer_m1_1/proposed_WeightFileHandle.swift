import Foundation
import Metal

/// Wraps an `MTLIOFileHandle` for raw binary weight files with bounds checking and file metadata.
///
/// Enables zero-copy Direct Memory Access (DMA) streaming from external or internal NVMe storage
/// directly into unified memory `MTLBuffer`s without intermediate POSIX kernel copies.
public final class WeightFileHandle: @unchecked Sendable {
    /// File system URL of the weight binary file.
    public let url: URL

    /// Native Metal Fast I/O file handle.
    public let rawHandle: any MTLIOFileHandle

    /// Total size of the file on disk in bytes.
    public let fileSize: Int

    /// Opens a weight file for Fast I/O DMA access on the given Metal device.
    ///
    /// - Parameters:
    ///   - url: File URL to the binary weights.
    ///   - device: Metal device to associate with the I/O handle.
    public init(url: URL, device: any MTLDevice) throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw FastIOError.fileNotFound(url: url)
        }

        do {
            let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
            guard let size = attrs[.size] as? NSNumber else {
                throw FastIOError.fileOpenFailed(url: url, reason: "Unable to determine file size attribute.")
            }
            self.fileSize = size.intValue
        } catch {
            throw FastIOError.fileOpenFailed(url: url, reason: error.localizedDescription)
        }

        do {
            self.rawHandle = try device.makeIOFileHandle(url: url)
        } catch {
            throw FastIOError.fileOpenFailed(url: url, reason: error.localizedDescription)
        }

        self.url = url
    }

    /// Validates that a requested byte offset and size fall strictly within the file boundaries.
    ///
    /// - Parameters:
    ///   - offset: File offset in bytes.
    ///   - size: Number of bytes to read.
    public func validateBounds(offset: Int, size: Int) throws {
        guard offset >= 0, size > 0, (offset + size) <= fileSize else {
            throw FastIOError.offsetOutOfBounds(offset: offset, size: size, fileSize: fileSize)
        }
    }

    /// Computes the exact file byte offset for a specific deep expert in a contiguous weights file.
    ///
    /// - Parameters:
    ///   - globalExpertIndex: Flattened index across deep layers (0..totalDeepExperts-1).
    ///   - expertSizeBytes: Byte size per expert (17,301,504 for FP16 Qwen-MoE).
    /// - Returns: Byte offset from start of file.
    public static func offset(forGlobalExpertIndex globalExpertIndex: Int, expertSizeBytes: Int) -> Int {
        globalExpertIndex * expertSizeBytes
    }
}
