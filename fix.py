import sys

file_path = '/Users/jack/.gemini/antigravity/brain/7c0a1b5d-9de5-4e5e-a0cc-e24308e9273d/implementation_plan.md'
with open(file_path, 'r') as f:
    content = f.read()

target = '''- **Ring Buffer and Fallback Pool:** Pre-allocate a fixed-size pool of `MTLBuffer` slots in Unified Memory (~3-4 GB limit) for the Speculative Ring Buffer, alongside a smaller, completely isolated Fallback Buffer Pool (e.g., 500MB).
  - **Deadlock Failsafe:** If extreme speculative divergence fills 100% of the Ring Buffer slots and the GPU hits a cache miss for the *current* token, a circular deadlock can occur. To fix this, the CPU ignores the speculative Ring Buffer entirely. It allocates a slot from the isolated Fallback Pool, encodes the fetch on the `fallbackQueue`, and dispatches. The CPU then marks the in-flight speculative slot in the Ring Buffer as "abandoned/dirty." When the abandoned speculative IO callback eventually fires, the CPU simply drops the signal, marks the slot as clean, and makes it available for the next speculative cycle.
- **Execution Log, LRU, & Runtime Recalibration (MLX-Swift):** 
  - The Metal shader appends the ID of every routed expert, its predicted confidence, and its actual correct outcome into a dedicated `MTLBuffer` Execution Log. 
  - **Host-Side LRU:** The CPU maintains a "Dispatch-Time" LRU registry. Crucially, the CPU updates this LRU metadata *only* by draining the GPU Execution Log, never from its own pre-routing speculative predictions. This ensures the per-slot LRU field remains a single deterministic source of truth reflecting actual execution, avoiding any divergence between speculative fetch intent and actual expert consumption.'''

replacement = '''- **Ring Buffer and Fallback Pool:** Pre-allocate a fixed-size pool of `MTLBuffer` slots in Unified Memory (~3-4 GB limit) for the Speculative Ring Buffer, alongside a smaller, completely isolated Fallback Buffer Pool (e.g., 500MB).
  - **Deadlock Failsafe:** If extreme speculative divergence fills 100% of the Ring Buffer slots and the GPU hits a cache miss for the *current* token, a circular deadlock can occur. To fix this, the CPU ignores the speculative Ring Buffer entirely. It allocates a slot from the isolated Fallback Pool, encodes the fetch on the `fallbackQueue`, and dispatches. The CPU then marks the in-flight speculative slot in the Ring Buffer as "abandoned/dirty." While the Metal Fast I/O API does provide a `tryCancel()` method on `MTLIOCommandBuffer`, this is purely a best-effort request and cannot guarantee that the NVMe DMA controller hasn't already started streaming bytes to that physical RAM address. Therefore, forceful preemption (overwriting an in-flight slot) risks kernel panics and silent memory corruption. Instead, the system calls `tryCancel()` to potentially save PCIe bandwidth, but MUST leave the slot alone. When the abandoned speculative IO callback eventually fires, the CPU simply drops the signal, marks the slot as clean, and makes it available for the next speculative cycle.
- **Execution Log, LRU, & Runtime Recalibration (MLX-Swift):** 
  - The Metal shader appends the ID of every routed expert, its predicted confidence, and its actual correct outcome into a dedicated `MTLBuffer` Execution Log. 
  - **Host-Side LRU:** The CPU maintains a cached LRU registry. Crucially, the CPU updates this LRU metadata *only* by draining the GPU Execution Log, never from its own pre-routing speculative predictions. This ensures the per-slot LRU field remains a single deterministic source of truth reflecting actual execution, avoiding any divergence between speculative fetch intent and actual expert consumption. The GPU does not need to atomically update a separate timestamp buffer; the execution log naturally preserves the access order.'''

if target in content:
    content = content.replace(target, replacement)
    with open(file_path, 'w') as f:
        f.write(content)
    print('Replaced successfully')
else:
    print('Target not found')
    sys.exit(1)
