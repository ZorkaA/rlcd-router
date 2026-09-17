# Dispatch Assignment: Milestone 3 Spec Miner 2 (CPU Drain & Dispatch-Time LRU Tracker)

**Assigned Agent**: `teamwork_preview_spec_miner_m3_2`  
**Milestone**: Phase 2 Milestone 3 (GPU Execution Log & Dispatch-Time LRU Tracking — Requirement R3)  
**Assigned Working Directory**: `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_spec_miner_m3_2`  
**Date**: 2026-09-18  

---

## 1. Objective
Design the complete architecture and production Swift blueprint for the CPU Dispatch-Time LRU Weight Tracker:
- Target File: `Sources/AsyncMoERouter/ExecutionPipeline/LRUWeightTracker.swift`
- Mechanism: Lock-free draining of the GPU Execution Log and $O(1)$ LRU cache metadata updates.

---

## 2. Technical Specifications (Requirement R3)
1. **Post-Execution Invariant Enforcement**:
   - The CPU MUST update slot and expert LRU metadata *only* by draining the GPU Execution Log after kernel execution completion.
   - Slot `lastAccessedTimestamp` MUST NEVER be updated during speculative pre-routing predictions.
2. **Lock-Free / High-Efficiency Log Draining**:
   - Track read head index in CPU memory.
   - Synchronize with GPU execution via `MTLCommandBuffer.addCompletedHandler` or explicit CPU-GPU completion fences.
   - Safely drain newly committed log entries into a batch buffer.
3. **$O(1)$ LRU Weight Cache**:
   - Track active expert keys and their mapped ring buffer slots.
   - Doubly-linked list or dictionary-backed recency queue providing $O(1)$ `touch(expertKey:timestamp:)`, $O(1)$ `evictionCandidate() -> ExpertKey?`, and $O(1)$ `remove(expertKey:)`.
   - Update `SpeculativeRingBuffer.updateLRUTimestamp(for:timestamp:)` directly.

---

## 3. Deliverables
- Maintain `progress.md` in your working directory.
- Deliver a comprehensive handoff report with a complete, production-grade Swift code blueprint for `LRUWeightTracker.swift` to:
  `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_spec_miner_m3_2/handoff.md`
- Send a completion message to the orchestrator.

## 2026-09-17T21:41:24Z
You are Spec Miner 2 for Phase 2 Milestone 3: GPU Execution Log & Dispatch-Time LRU Tracking (Requirement R3).
Your assigned working directory is: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_spec_miner_m3_2
Read your dispatch assignment at: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_spec_miner_m3_2/DISPATCH.md
Read the authoritative user requirements at: /Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md
Read Phase 2 architecture at: /Users/jack/Downloads/rlcd-router/.agents/orchestrator_phase2/PROJECT.md

Tasks:
1. Design the CPU Dispatch-Time LRU Weight Tracker:
   - File: `Sources/AsyncMoERouter/ExecutionPipeline/LRUWeightTracker.swift`
   - Invariant: LRU metadata is updated strictly by draining the GPU Execution Log, NEVER via pre-routing prediction.
   - Lock-free / high-performance draining synchronized with command buffer completion handlers.
   - O(1) doubly-linked list / recency queue for expert key tracking and eviction candidates.
   - Integration with SpeculativeRingBuffer.updateLRUTimestamp(for:timestamp:).
   - Provide complete, compilable Swift code for LRUWeightTracker.swift.
2. Deliver your handoff report to:
   /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_spec_miner_m3_2/handoff.md
3. Update progress.md and send a message to the orchestrator when finished.

