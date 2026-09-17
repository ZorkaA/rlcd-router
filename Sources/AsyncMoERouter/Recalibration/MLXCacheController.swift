import Foundation
import os.log

/// Enforces the 200MB MLX Metal cache ceiling to prevent background recalibration
/// from triggering OS-level memory compression of the Speculative Ring Buffer.
///
/// # Why This Matters
/// MLX's Metal backend maintains an aggressive memory recycling cache. Without an explicit
/// limit, MLX will greedily consume Unified Memory, potentially causing the OS to
/// WKdm-compress the Ring Buffer's `storageModeShared` pages during a background
/// Brier-score optimization pass.
///
/// By calling `set(cacheLimit:)` before the first MLX operation, we fence MLX's
/// memory footprint to 200MB, making it functionally invisible to the main inference pipeline.
public enum MLXCacheController {
    private static let _log = Logger(subsystem: "AsyncMoERouter", category: "MLXCacheController")
    private static let maxCacheLimitBytes = 200 * 1024 * 1024  // 200 MB

    /// Applies the 200MB MLX Metal cache ceiling.
    ///
    /// Call this **once** before initializing any MLX graph or model. Subsequent
    /// MLX cache allocations beyond 200MB will be reclaimed rather than retained.
    ///
    /// Note: This calls `mlx_metal_set_cache_limit` if the MLX C API is linked.
    /// In a Swift-only test environment without MLX, this is a documented no-op.
    public static func applyCacheLimit() {
        #if canImport(MLX)
        MLX.GPU.set(cacheLimit: maxCacheLimitBytes)
        _log.info("MLXCacheController: MLX Metal cache limit set to \(maxCacheLimitBytes) bytes (200MB)")
        #else
        _log.info("MLXCacheController: MLX not linked — cache limit call is a no-op in this build target")
        #endif
    }

    public static var cacheLimitBytes: Int { maxCacheLimitBytes }
}
