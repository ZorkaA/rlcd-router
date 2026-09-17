# Milestone 1 Explorer 2: Zero-OOM Streaming Extractor & Hook Architecture

## 2026-09-17T07:31:42Z

You are teamwork_preview_explorer_m1_2.
Working directory: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_2
Parent: orchestrator (ce5bc762-f633-465c-9133-7ec43d0b5719)

MANDATORY INPUT:
- Read /Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md
- Read /Users/jack/Downloads/rlcd-router/PROJECT.md
- Read /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_2/DISPATCH.md

YOUR ROLE & OBJECTIVE:
Investigate and design the exact technical specification and code blueprint for `src/data/stream_extractor.py`:
1. 100k-token corpus streaming without OOM crashes:
   - Token streaming generator with batch size $B=1$, sequence length $L=1024$ (98 sequences total).
   - Execution context: `torch.inference_mode()`, disabling KV cache (`use_cache=False`).
   - Memory management: Immediate `.detach().to("cpu", dtype=torch.float16)` of harvested tensors, explicit `del outputs`, and periodic `torch.mps.empty_cache()` / `gc.collect()`.
2. Activation & Routing Extraction:
   - Extract Layer N (default Layer 3, index 2 in `outputs.hidden_states[3]`) hidden states: shape $(B, L, d_{\text{model}})$.
   - Extract native router logits across all 24 layers: tuple of 24 tensors, each $(B \cdot L, 60)$ or $(B, L, 60)$.
   - Compute native top-4 expert assignments and probabilities per token per layer.
3. Design clean class and function interfaces, progress reporting, and memory tracking hooks.
Write `analysis.md` and `handoff.md` in your working directory, and notify parent.
