# Milestone 1 Explorer 1: Model Loading & Environment Architecture

You are teamwork_preview_explorer_m1_1.
Working directory: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_1
Parent: orchestrator (ce5bc762-f633-465c-9133-7ec43d0b5719)

MANDATORY INPUT:
- Read /Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md
- Read /Users/jack/Downloads/rlcd-router/PROJECT.md

YOUR ROLE & OBJECTIVE:
Investigate and design the exact technical specification and code blueprint for `src/data/model_loader.py` and `src/config.py`:
1. Model loading for `Qwen/Qwen1.5-MoE-A2.7B`:
   - Device handling: Automatic detection of MPS vs CPU. Critical finding from Survey 2: PyTorch 2.2.2 on MPS crashes on `bfloat16` (`RuntimeError: BFloat16 is not supported on MPS`). Therefore on MPS, dtype MUST be `torch.float16`. On CPU, `torch.bfloat16` or `torch.float32`.
   - Forward pass configuration: `output_hidden_states=True`, `output_router_logits=True`, `use_cache=False`.
2. Fast Synthetic Model Fixture:
   - Provide a zero-download synthetic `Qwen2MoeForCausalLM` factory (`get_synthetic_model()`) using a scaled-down `Qwen2MoeConfig` ($d=64$, 6 layers, 16 experts, top_k=4) with exact output structure matching the genuine model for unit testing and CI.
3. Design clear function signatures, class definitions, and error handling.
Write `analysis.md` and `handoff.md` in your working directory.

## 2026-09-17T07:31:42Z
You are teamwork_preview_explorer_m1_1.
Your working directory is: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_1
Your parent is orchestrator (conversation ID: ce5bc762-f633-465c-9133-7ec43d0b5719).

MANDATORY INPUT:
- Read /Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md
- Read /Users/jack/Downloads/rlcd-router/PROJECT.md
- Read /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_1/DISPATCH.md

YOUR ROLE & OBJECTIVE:
Investigate and design the exact technical specification and code blueprint for `src/data/model_loader.py` and `src/config.py`:
- Device selection (MPS vs CPU), float16 constraint on MPS (preventing BFloat16 crash).
- Model loading mechanics with `output_hidden_states=True`, `output_router_logits=True`, `use_cache=False`.
- Synthetic model fixture for fast testing (`get_synthetic_model`).
Write analysis.md and handoff.md in your working directory, and notify parent.
