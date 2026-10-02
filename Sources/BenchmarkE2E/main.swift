import Foundation
import Metal
import AsyncMoERouter

@main
struct BenchmarkE2E {
    static func main() async throws {
        print("Starting E2E Router Server...")
        fflush(stdout)
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal device not found")
        }
        
        let fastIO = try FastIOEngine(device: device)
        let config = MoEArchitectureConfig.mixtral8x7B 
        
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
        
        let expertsDir = URL(fileURLWithPath: "/Users/jack/Downloads/rlcd-router/partitioned_experts")
        guard FileManager.default.fileExists(atPath: expertsDir.path) else {
            fatalError("Partitioned experts directory not found. Please run scripts/partition_model.py first.")
        }
        
        let files = try FileManager.default.contentsOfDirectory(atPath: expertsDir.path)
        guard let firstExpertFile = files.first(where: { $0 == "expert_0.bin" }) else {
            fatalError("expert_0.bin not found in partitioned_experts")
        }
        let firstExpertURL = expertsDir.appendingPathComponent(firstExpertFile)
        
        let weightLayout = WeightLayoutConfig.mixtral8x7B
        let weightHandle = try WeightFileHandle(device: device, url: firstExpertURL, layout: weightLayout)
        
        print("READY")
        fflush(stdout)
        
        while let line = readLine() {
            guard let data = line.data(using: .utf8),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Int],
                  let layer = json["layer"],
                  let expert = json["expert"] else {
                continue
            }
            
            let expertKey = ExpertKey(layer: layer, expert: expert)
            
            if !pipeline.isExpertCached(expertKey) {
                let context = DemandFetchContext(expert: expertKey, tokenIndex: 0, reason: .cacheMiss)
                if let slot = try? pipeline.fallbackPool.allocateForDemand(context: context, device: device, ticket: 0) {
                    
                    let sourceOffset = 0 // Dummy fetch to force bounded I/O paging
                    
                    let maxChunkSize = 32 * 1024 * 1024
                    var currentOffset = 0
                    var remainingSize = config.expertSizeBytes
                    
                    while remainingSize > 0 {
                        let chunkSize = min(remainingSize, maxChunkSize)
                        let (syncEvent, ticket) = fastIO.loadFallback(
                            handle: weightHandle.ioFileHandle,
                            offset: sourceOffset + currentOffset,
                            size: chunkSize,
                            targetBuffer: slot.buffer,
                            targetOffset: currentOffset
                        )
                        
                        await withCheckedContinuation { continuation in
                            let listener = MTLSharedEventListener()
                            syncEvent.notify(listener, atValue: ticket) { _, _ in
                                continuation.resume()
                            }
                        }
                        currentOffset += chunkSize
                        remainingSize -= chunkSize
                    }
                    try? pipeline.fallbackPool.reclaim(slot)
                }
            }
            
            print("{\"status\": \"ok\", \"layer\": \(layer), \"expert\": \(expert)}")
            fflush(stdout)
        }
    }
}
