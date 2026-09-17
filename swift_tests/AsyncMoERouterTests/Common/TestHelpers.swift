import Foundation
import Metal
import AsyncMoERouter

/// Architectural dimensions and parameters for synthetic and production MoE configurations.
public struct SyntheticMoEConfig: Sendable, Equatable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numExperts: Int
    public let numActiveExperts: Int
    public let numDeepLayers: Int
    public let bytesPerElement: Int

    public init(
        hiddenSize: Int = 64,
        intermediateSize: Int = 64,
        numExperts: Int = 16,
        numActiveExperts: Int = 4,
        numDeepLayers: Int = 4,
        bytesPerElement: Int = 2 // FP16
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numExperts = numExperts
        self.numActiveExperts = numActiveExperts
        self.numDeepLayers = numDeepLayers
        self.bytesPerElement = bytesPerElement
    }

    /// Number of elements in a SwiGLU 3-projection expert (gate, up, down).
    public var expertElements: Int {
        3 * (hiddenSize * intermediateSize)
    }

    /// Total byte footprint of a single expert MLP.
    public var expertSizeBytes: Int {
        expertElements * bytesPerElement
    }

    /// Total byte footprint of all deep layers and experts.
    public var totalModelSizeBytes: Int {
        numDeepLayers * numExperts * expertSizeBytes
    }

    /// Fast synthetic configuration designed for CI unit testing (1.57 MB total on disk).
    public static let fastTest = SyntheticMoEConfig(
        hiddenSize: 64,
        intermediateSize: 64,
        numExperts: 16,
        numActiveExperts: 4,
        numDeepLayers: 4,
        bytesPerElement: 2
    )

    /// Full-scale Qwen1.5-MoE-A2.7B configuration (17.3 MB per expert, 20.76 GB full model).
    public static let qwen15MoE = SyntheticMoEConfig(
        hiddenSize: 2048,
        intermediateSize: 1408,
        numExperts: 60,
        numActiveExperts: 4,
        numDeepLayers: 20,
        bytesPerElement: 2
    )
}

/// Utility for generating deterministic synthetic weight files and validating Fast I/O DMA accuracy.
public enum SyntheticWeightFileGenerator {
    /// Computes the deterministic byte value for an expert block at a given relative byte offset.
    @inline(__always)
    public static func expectedByte(layer: Int, expert: Int, byteOffset: Int) -> UInt8 {
        UInt8((layer * 17 + expert * 31 + byteOffset) & 0xFF)
    }

    /// Calculates the byte offset of a specific expert within the contiguous binary file.
    public static func fileOffset(layer: Int, expert: Int, config: SyntheticMoEConfig) -> Int {
        (layer * config.numExperts + expert) * config.expertSizeBytes
    }

    /// Generates an isolated temporary binary weight file populated with deterministic expert blocks.
    /// Streaming is performed in 64 KB chunks to guarantee < 2 MB RAM usage during test fixture generation.
    public static func createTemporaryWeightFile(
        config: SyntheticMoEConfig = .fastTest,
        prefix: String = "synthetic_weights"
    ) throws -> (url: URL, cleanup: () -> Void) {
        let tempDir = FileManager.default.temporaryDirectory
        let fileURL = tempDir.appendingPathComponent("\(prefix)_\(UUID().uuidString).bin")

        FileManager.default.createFile(atPath: fileURL.path, contents: nil)
        let handle = try FileHandle(forWritingTo: fileURL)

        let expertSize = config.expertSizeBytes
        let chunkSize = 65_536 // 64 KB streaming buffer
        var chunk = [UInt8](repeating: 0, count: chunkSize)

        for l in 0..<config.numDeepLayers {
            for e in 0..<config.numExperts {
                var bytesWrittenForExpert = 0
                while bytesWrittenForExpert < expertSize {
                    let toWrite = min(chunkSize, expertSize - bytesWrittenForExpert)
                    for i in 0..<toWrite {
                        chunk[i] = expectedByte(layer: l, expert: e, byteOffset: bytesWrittenForExpert + i)
                    }
                    handle.write(Data(bytes: chunk, count: toWrite))
                    bytesWrittenForExpert += toWrite
                }
            }
        }

        try handle.close()

        let cleanup: () -> Void = {
            try? FileManager.default.removeItem(at: fileURL)
        }

        return (fileURL, cleanup)
    }
}

/// Bitwise validator checking that `MTLBuffer` contents match the expected synthetic pattern.
public enum MockExpertValidator {
    public struct ValidationResult: Sendable {
        public let isValid: Bool
        public let totalBytesChecked: Int
        public let firstMismatchOffset: Int?
        public let expectedByte: UInt8?
        public let actualByte: UInt8?
    }

    /// Validates an `MTLBuffer` against the expected synthetic expert pattern.
    public static func validate(
        buffer: any MTLBuffer,
        bufferOffset: Int = 0,
        layer: Int,
        expert: Int,
        length: Int? = nil,
        config: SyntheticMoEConfig = .fastTest
    ) -> ValidationResult {
        let size = length ?? config.expertSizeBytes
        precondition(buffer.length >= bufferOffset + size, "Buffer length insufficient for validation")

        let ptr = buffer.contents().advanced(by: bufferOffset).bindMemory(to: UInt8.self, capacity: size)
        for i in 0..<size {
            let expected = SyntheticWeightFileGenerator.expectedByte(layer: layer, expert: expert, byteOffset: i)
            let actual = ptr[i]
            if actual != expected {
                return ValidationResult(
                    isValid: false,
                    totalBytesChecked: i + 1,
                    firstMismatchOffset: i,
                    expectedByte: expected,
                    actualByte: actual
                )
            }
        }

        return ValidationResult(
            isValid: true,
            totalBytesChecked: size,
            firstMismatchOffset: nil,
            expectedByte: nil,
            actualByte: nil
        )
    }
}

/// Utilities for generating synthetic weight files, temporary test fixtures, and buffer assertions.
public enum TestHelpers {
    /// Creates a temporary binary file containing deterministic synthetic expert weights.
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
