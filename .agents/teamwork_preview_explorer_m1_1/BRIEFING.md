# BRIEFING — 2026-09-17T07:37:30Z

## Mission
Investigate and design the exact technical specification and code blueprint for `src/data/model_loader.py` and `src/config.py` (device selection, FP16 MPS constraint, model loading, synthetic fixture).

## 🔒 My Identity
- Archetype: teamwork_preview_explorer
- Roles: explorer, investigator, architect
- Working directory: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_1
- Original parent: ce5bc762-f633-465c-9133-7ec43d0b5719
- Milestone: M1 (Data Partitioning & Generation)

## 🔒 Key Constraints
- Read-only investigation — do NOT implement directly in src/
- Only write metadata, reports, and blueprints within .agents/teamwork_preview_explorer_m1_1/
- No bfloat16 on MPS (PyTorch 2.2.2 MPS crash workaround: use torch.float16)
- Model forward pass must configure output_hidden_states=True, output_router_logits=True, use_cache=False
- Zero-download synthetic model fixture required for testing and CI

## Current Parent
- Conversation ID: ce5bc762-f633-465c-9133-7ec43d0b5719
- Updated: 2026-09-17T07:37:30Z

## Investigation State
- **Explored paths**: ORIGINAL_REQUEST.md, PROJECT.md, survey_2 handoff, local HF cache, PyTorch 2.2.2 MPS runtime, Transformers Qwen2Moe architecture.
- **Key findings**: 
  1. MPS raises `TypeError: BFloat16 is not supported on MPS`. FP16 strictly enforced.
  2. Base model config has `"torch_dtype": "bfloat16"`, so naive auto-loading crashes on MPS. Explicit override to `torch.float16` implemented.
  3. `outputs.hidden_states` has length 25 (Layer 3 output is index 3). `outputs.router_logits` has length 24 (Deep layers 5..24 are indices 4..23, shape `(B*L, 60)`).
  4. Created zero-download synthetic fixture `get_synthetic_model()` with $d=64$, 6 layers, 16 experts, top-4 (~1.56M params, <6 MB RAM, <0.06s init, <0.6s fwd).
- **Unexplored areas**: None for M1.1 scope. Fully characterized.

## Key Decisions Made
- `src/config.py`: Defined comprehensive constants table, device resolution helpers (`resolve_device`, `resolve_dtype`), and structured configuration dataclasses.
- `src/data/model_loader.py`: Implemented `load_model`, `load_tokenizer`, `load_model_and_tokenizer`, `get_synthetic_model`, `get_synthetic_tokenizer`, `get_synthetic_model_and_tokenizer`.
- Validated all functions and fixtures via automated test suite in working directory.

## Artifact Index
- DISPATCH.md — Task log from orchestrator
- BRIEFING.md — Working memory
- progress.md — Liveness heartbeat
- analysis.md — Technical specification & blueprint
- handoff.md — 5-component hard handoff report
- proposed_config.py — Machine-applicable proposed implementation for src/config.py
- proposed_model_loader.py — Machine-applicable proposed implementation for src/data/model_loader.py
