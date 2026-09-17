import Foundation
import Metal
import AsyncMoERouter

/// Utilities for generating synthetic weight files, temporary test fixtures, and buffer assertions.
public enum TestHelpers {
    /// Creates a temporary binary file containing deterministic synthetic expert weights.
    ///
    /// - Parameters:
    ///   - expertCount: Number of experts to write into the file.
    ///   - expertSizeBytes: Byte size of each expert.
    ///   - fillByte: Optional base byte to fill each expert (if nil, uses expert index).
    /// - Returns: URL to the created temporary file and cleanup closure.
    public static func createSyntheticWeightFile(
        expertCount: Int = 4,
        expertSizeBytes: Int = MoEArchitectureConfig.synthetic.expertSizeBytes,
        fillByte: UInt8? = nil
    ) throws -> (url: URL, cleanup: () -> Void) {
        let tempDir = FileManager.default.temporaryDirectory
        let fileName = "synthetic_weights_\(UUID().uuidString).bin"
        let fileURL = tempDir.appendingPathComponent(fileName)

        var fileData = Data(capacity: expertCount * expertSizeBytes)
        for i in 0..<expertCount {
            let byteVal = fillByte ?? UInt8((i * 17 + 1) % 256)
            fileData.append(Data(repeating: byteVal, count: expertSizeBytes))
        }

        try fileData.write(to: fileURL)

        let cleanup: () -> Void = {
            _ = try? FileManager.default.removeItem(at: fileURL)
        }

        return (fileURL, cleanup)
    }

    /// Verifies that all bytes in an MTLBuffer match the expected value.
    public static func verifyBufferContents(
        buffer: any MTLBuffer,
        offset: Int = 0,
        length: Int,
        expectedByte: UInt8
    ) -> Bool {
        let ptr = buffer.contents().advanced(by: offset).bindMemory(to: UInt8.self, capacity: length)
        for i in 0..<length {
            if ptr[i] != expectedByte {
                return false
            }
        }
        return true
    }
}
