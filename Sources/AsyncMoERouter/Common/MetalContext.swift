import Foundation
import Metal
import os

/// Errors originating from Metal device initialization and runtime shader compilation.
public enum MetalContextError: Error, CustomStringConvertible, Sendable {
    case noDefaultDevice
    case commandQueueCreationFailed
    case compilationFailed(functionName: String, errorDescription: String)
    case functionNotFound(functionName: String)
    case bufferAllocationFailed(length: Int)

    public var description: String {
        switch self {
        case .noDefaultDevice:
            return "MetalContextError: No default Metal device available (MTLCreateSystemDefaultDevice returned nil)."
        case .commandQueueCreationFailed:
            return "MetalContextError: Failed to create standard MTLCommandQueue from MTLDevice."
        case .compilationFailed(let name, let desc):
            return "MetalContextError: Runtime MSL compilation failed for '\(name)': \(desc)"
        case .functionNotFound(let name):
            return "MetalContextError: Function '\(name)' not found in compiled MTLLibrary."
        case .bufferAllocationFailed(let length):
            return "MetalContextError: Failed to allocate MTLBuffer of length \(length) bytes."
        }
    }
}

/// Centralized manager for Apple Silicon Metal 3 device, command queue, and runtime MSL compilation.
public final class MetalContext: @unchecked Sendable {
    /// The primary Apple Silicon GPU device.
    public let device: any MTLDevice

    /// Standard command queue for compute and blit command execution.
    public let commandQueue: any MTLCommandQueue

    /// Lock-protected in-memory cache for compiled MTLLibrary objects keyed by source hash.
    private let libraryCache = OSAllocatedUnfairLock(initialState: [Int: any MTLLibrary]())

    /// Lock-protected in-memory cache for compiled compute pipeline states keyed by "sourceHash:functionName".
    private let pipelineCache = OSAllocatedUnfairLock(initialState: [String: any MTLComputePipelineState]())

    /// Shared singleton instance initialized with the system default Metal device.
    public static let shared: MetalContext = {
        do {
            return try MetalContext()
        } catch {
            fatalError("Failed to initialize default MetalContext: \(error)")
        }
    }()

    /// Creates a Metal context with a specified or system-default Metal device.
    public init(device: (any MTLDevice)? = nil) throws {
        guard let dev = device ?? MTLCreateSystemDefaultDevice() else {
            throw MetalContextError.noDefaultDevice
        }
        self.device = dev
        guard let queue = dev.makeCommandQueue() else {
            throw MetalContextError.commandQueueCreationFailed
        }
        self.commandQueue = queue
    }

    /// Checks whether the underlying device supports Metal 3 features.
    public var supportsMetal3: Bool {
        device.supportsFamily(.metal3)
    }

    /// Compiles an MSL source string at runtime, caching the compiled `MTLLibrary` by source hash.
    public func compileLibrary(source: String, options: MTLCompileOptions? = nil) throws -> any MTLLibrary {
        let key = source.hashValue
        if let cached = libraryCache.withLock({ $0[key] }) {
            return cached
        }

        do {
            let library = try device.makeLibrary(source: source, options: options)
            libraryCache.withLock { $0[key] = library }
            return library
        } catch {
            throw MetalContextError.compilationFailed(
                functionName: "library",
                errorDescription: error.localizedDescription
            )
        }
    }

    /// Compiles or retrieves a cached `MTLComputePipelineState` for a kernel function in MSL source.
    public func makeComputePipelineState(
        source: String,
        functionName: String,
        options: MTLCompileOptions? = nil
    ) throws -> any MTLComputePipelineState {
        let cacheKey = "\(source.hashValue):\(functionName)"
        if let cached = pipelineCache.withLock({ $0[cacheKey] }) {
            return cached
        }

        let library = try compileLibrary(source: source, options: options)
        guard let function = library.makeFunction(name: functionName) else {
            throw MetalContextError.functionNotFound(functionName: functionName)
        }

        do {
            let pipelineState = try device.makeComputePipelineState(function: function)
            pipelineCache.withLock { $0[cacheKey] = pipelineState }
            return pipelineState
        } catch {
            throw MetalContextError.compilationFailed(
                functionName: functionName,
                errorDescription: error.localizedDescription
            )
        }
    }

    /// Allocates an `MTLBuffer` using the context's device.
    public func makeBuffer(
        length: Int,
        options: MTLResourceOptions = .storageModeShared,
        label: String? = nil
    ) throws -> any MTLBuffer {
        guard let buf = device.makeBuffer(length: length, options: options) else {
            throw MetalContextError.bufferAllocationFailed(length: length)
        }
        if let label = label {
            buf.label = label
        }
        return buf
    }
}
