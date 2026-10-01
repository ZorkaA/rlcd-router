import Foundation
import Metal
import AsyncMoERouter

@main
struct BenchmarkE2E {
    static func main() async throws {
        print("Starting E2E Benchmark...")
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal device not found")
        }
        
        let fastIO = try FastIOEngine(device: device)
        let config = MoEArchitectureConfig.mixtral8x22B 
        
        let ringBufferBytes = MemoryBudgetConfig.default.speculativeRingBufferBytes(for: config)
        let mlxCacheLimit = 200 * 1024 * 1024
        let maxMemory = Int(30.6 * 1024 * 1024 * 1024)
        let fallbackMax = maxMemory - ringBufferBytes - mlxCacheLimit - (4096 * 32) - 16
        
        let budgetConfig = MemoryBudgetConfig(
            speculativeRingBufferSlots: 16,
            fallbackPoolMaxBytes: fallbackMax,
            mlxCacheLimitBytes: mlxCacheLimit,
            executionLogCapacity: 4096
        )

        let pipeline = AsyncMoEPipeline(
            device: device,
            fastIO: fastIO,
            archConfig: config,
            budgetConfig: budgetConfig
        )
        
        let expertsDir = URL(fileURLWithPath: "/Volumes/SSK SSD/partitioned_experts")
        guard FileManager.default.fileExists(atPath: expertsDir.path) else {
            fatalError("Partitioned experts directory not found. Please run scripts/partition_model.py first.")
        }
        
        let files = try FileManager.default.contentsOfDirectory(atPath: expertsDir.path)
        guard let firstExpertFile = files.first(where: { $0 == "mixtral_dummy.bin" }) else {
            fatalError("mixtral_dummy.bin not found in partitioned_experts")
        }
        let firstExpertURL = expertsDir.appendingPathComponent(firstExpertFile)
        
        let weightLayout = WeightLayoutConfig.mixtral8x22B
        let weightHandle = try WeightFileHandle(device: device, url: firstExpertURL, layout: weightLayout)
        
        print("Running benchmark loop for 100 tokens...")
        let startTime = CFAbsoluteTimeGetCurrent()
        let numTokens = 100
        
        for i in 0..<numTokens {
            pipeline.beginStep()
            
            // Loop across all layers to ensure we thrash the OS page cache and hit the SSD
            let layerIndex = i % config.numTotalLayers
            let expertIndex = (i * 7) % config.numExperts
            let expertKey = ExpertKey(layer: layerIndex, expert: expertIndex)
            let _ = pipeline.prefetchExpert(key: expertKey)
            
            if !pipeline.isExpertCached(expertKey) {
                let context = DemandFetchContext(expert: expertKey, tokenIndex: UInt32(i), reason: .cacheMiss)
                let slot = try pipeline.fallbackPool.allocateForDemand(context: context, device: device, ticket: UInt64(i))
                
                let sourceOffset = try weightHandle.fileOffset(layerIndex: expertKey.layerIndex, expertIndex: expertKey.expertIndex)
                
                let (syncEvent, ticket) = fastIO.loadFallback(
                    handle: weightHandle.ioFileHandle,
                    offset: sourceOffset,
                    size: config.expertSizeBytes,
                    targetBuffer: slot.buffer,
                    targetOffset: 0
                )
                
                // wait asynchronously
                await withCheckedContinuation { continuation in
                    let listener = MTLSharedEventListener()
                    syncEvent.notify(listener, atValue: ticket) { _, _ in
                        continuation.resume()
                    }
                }
                
                try pipeline.fallbackPool.reclaim(slot)
                pipeline.recordCacheMiss()
            } else {
                pipeline.recordCacheHit()
            }
            
            await pipeline.endStep()
        }
        
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        let tokPerSec = Double(numTokens) / elapsed
        print(String(format: "Benchmark finished: %.2f tok/sec", tokPerSec))
        print(pipeline.diagnostics())
    }
}
