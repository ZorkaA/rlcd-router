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
        let config = MoEArchitectureConfig.qwen15MoEA27B 
        let pipeline = AsyncMoEPipeline(
            device: device,
            fastIO: fastIO,
            archConfig: config,
            budgetConfig: .default
        )
        
        let expertsDir = URL(fileURLWithPath: "partitioned_experts")
        guard FileManager.default.fileExists(atPath: expertsDir.path) else {
            fatalError("Partitioned experts directory not found. Please run scripts/partition_model.py first.")
        }
        
        let files = try FileManager.default.contentsOfDirectory(atPath: expertsDir.path)
        guard let firstExpertFile = files.first(where: { $0.hasSuffix(".bin") }) else {
            fatalError("No expert files found in partitioned_experts")
        }
        let firstExpertURL = expertsDir.appendingPathComponent(firstExpertFile)
        
        let weightLayout = WeightLayoutConfig.qwen15MoEA27B
        let weightHandle = try WeightFileHandle(device: device, url: firstExpertURL, layout: weightLayout)
        
        print("Running benchmark loop for 100 tokens...")
        let startTime = CFAbsoluteTimeGetCurrent()
        let numTokens = 100
        
        for i in 0..<numTokens {
            pipeline.beginStep()
            
            let expertKey = ExpertKey(layer: 0, expert: i % config.numExperts)
            let _ = pipeline.prefetchExpert(key: expertKey)
            
            if !pipeline.isExpertCached(expertKey) {
                let context = DemandFetchContext(expert: expertKey, tokenIndex: UInt32(i), reason: .cacheMiss)
                let slot = try pipeline.fallbackPool.allocateForDemand(context: context, device: device, ticket: UInt64(i))
                
                let (syncEvent, ticket) = fastIO.loadFallback(
                    handle: weightHandle.ioFileHandle,
                    offset: 0,
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
