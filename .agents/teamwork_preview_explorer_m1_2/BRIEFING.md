# BRIEFING — 2026-09-17T07:39:30Z

## Mission
Investigate and design the exact technical specification and code blueprint for `src/data/stream_extractor.py` (Zero-OOM streaming generator for 100k tokens, Layer N hidden states extraction, 24-layer router logits harvesting, memory hygiene and hooks).

## 🔒 My Identity
- Archetype: teamwork_preview_explorer
- Roles: explorer, investigator, architect
- Working directory: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_2
- Original parent: ce5bc762-f633-465c-9133-7ec43d0b5719
- Milestone: M1 (Data Partitioning & Generation)

## 🔒 Key Constraints
- Read-only investigation — do NOT implement directly in src/
- Only write metadata, reports, and blueprints within /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_2/
- Target model: Qwen/Qwen1.5-MoE-A2.7B (24 layers, d_model=2048, 60 routed experts, top-4 routed)
- Token streaming generator with batch size B=1, sequence length L=1024 (98 sequences total for 100k tokens)
- Execution context: torch.inference_mode(), use_cache=False
- Memory management: Immediate .detach().to("cpu", dtype=torch.float16) of harvested tensors, explicit `del outputs`, periodic `torch.mps.empty_cache()` and `gc.collect()`
- Layer N extraction: Default Layer 3 (1-indexed Layer 3, index 3 in HF outputs.hidden_states), shape (B, L, d_model)
- Router logits extraction: All 24 layers, shape (B, L, 60) per layer
- Native top-4 expert indices and probabilities calculation

## Current Parent
- Conversation ID: ce5bc762-f633-465c-9133-7ec43d0b5719
- Updated: 2026-09-17T07:39:30Z

## Investigation State
- **Explored paths**:
  - `ORIGINAL_REQUEST.md` (R1, AC1)
  - `PROJECT.md` (Features 2 & 3, M1-M2 contract)
  - `DISPATCH.md` (detailed task instructions)
  - `teamwork_preview_explorer_m1_1/proposed_config.py` & `proposed_model_loader.py`
  - `teamwork_preview_explorer_m1_3/analysis.md` & `handoff.md` (contract alignment with dataset.py)
  - Empirical memory & forward execution benchmarks on Apple Silicon MPS and CPU
- **Key findings**:
  - In `transformers`, `outputs.hidden_states[3]` provides Layer 3 representation ($B, L, 2048$) in float16.
  - `outputs.router_logits` contains 24 tensors each of shape $(B \cdot L, 60)$, reshaped to $(B, L, 60)$ and stacked to $(B, L, 24, 60)$ in CPU float16.
  - Native top-4 expert indices `(B, L, 24, 4)` and probabilities `(B, L, 24, 4)` computed via softmax in float32 + `torch.topk(k=4)`.
  - Zero-OOM streaming verified: PyTorch MPS allocated memory remains flat at 14.99 MB across 50-98 iterations; CPU RSS stabilizes at 414.97 MB (delta <0.72 MB).
  - Corpus tokenization verified: local `wikitext-2-raw-v1` yields 100,352 tokens in ~1.0 s offline.
- **Unexplored areas**: None for M1_2 scope. Complete blueprint delivered and verified.

## Key Decisions Made
- Generator-based design: `StreamExtractor` yields `ExtractionBatch` instances one sequence at a time ($B=1, L=1024$).
- Built-in `MemoryTracker` and `ExtractionHook` callbacks for leak detection and progress logging.
- Support 3 streaming modes: `stream_token_tensor`, `stream_synthetic`, `stream_text_corpus` (`wikitext-2-raw-v1`).
- Created working reference implementation `proposed_stream_extractor.py` and validated with automated test script.

## Artifact Index
- DISPATCH.md — Task log from orchestrator
- BRIEFING.md — Working memory
- progress.md — Liveness heartbeat
- analysis.md — Technical specification & code blueprint for `src/data/stream_extractor.py`
- proposed_stream_extractor.py — Production-grade reference implementation
- handoff.md — 5-component hard handoff report
