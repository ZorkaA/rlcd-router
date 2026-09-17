# Milestone 1 Worker Task: Data Partitioning & Generation

You are teamwork_preview_worker_m1_1.
Working directory: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_worker_m1_1
Parent: orchestrator (ce5bc762-f633-465c-9133-7ec43d0b5719)

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

MANDATORY INPUTS:
- /Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md
- /Users/jack/Downloads/rlcd-router/PROJECT.md
- /Users/jack/Downloads/rlcd-router/TEST_READY.md
- Explorer 1 blueprints: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_1/handoff.md, proposed_config.py, proposed_model_loader.py
- Explorer 2 blueprints: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_2/handoff.md, proposed_stream_extractor.py
- Explorer 3 blueprints: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_3/handoff.md, analysis.md

EXCLUSIVE FILE OWNERSHIP:
- /Users/jack/Downloads/rlcd-router/src/__init__.py
- /Users/jack/Downloads/rlcd-router/src/config.py
- /Users/jack/Downloads/rlcd-router/src/data/__init__.py
- /Users/jack/Downloads/rlcd-router/src/data/model_loader.py
- /Users/jack/Downloads/rlcd-router/src/data/stream_extractor.py
- /Users/jack/Downloads/rlcd-router/src/data/dataset.py

YOUR DELIVERABLES:
1. Implement production-grade modules in `src/` and `src/data/`:
   - `src/__init__.py`
   - `src/config.py`: Device and dtype resolution (MPS with float16, CPU with float32/bfloat16, strict ValueError on MPS bfloat16), paths, MoE architecture constants (24 layers, 60 routed experts, top-4, default tap Layer 3, deep layers 5-24, early 5-10, late 11-24).
   - `src/data/__init__.py`: Export key loaders, extractors, datasets.
   - `src/data/model_loader.py`: `load_model()`, `load_tokenizer()`, `get_synthetic_model()`, `SyntheticTokenizer` ensuring 100% parity and fast zero-download execution.
   - `src/data/stream_extractor.py`: `StreamExtractor`, `StreamExtractorConfig`, `ExtractionBatch`, `MemoryTracker` implementing zero-OOM memory hygiene (`torch.inference_mode()`, `use_cache=False`, CPU offload, periodic cache clear).
   - `src/data/dataset.py`: `partition_and_align()`, `save_calibration_datasets()`, `load_calibration_dataset()`, `MoECalibrationDataset(Dataset)` with 80/20 sequence-level isolation, multi-horizon targets (T+1, T+2, T+3) for deep layers (5-24), trailing token boundary masks, and safetensors I/O.
2. Verify all implementations by running pytest:
   `pytest -v tests/`
   Verify that tests covering Features 1-5 pass cleanly without regressions.
3. Write `handoff.md` in your working directory and notify parent.
## 2026-09-17T11:40:00Z
<USER_REQUEST>
You are teamwork_preview_worker_m1_1.
Your working directory is: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_worker_m1_1
Your parent is orchestrator (conversation ID: ce5bc762-f633-465c-9133-7ec43d0b5719).

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

MANDATORY INPUTS:
- /Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md
- /Users/jack/Downloads/rlcd-router/PROJECT.md
- /Users/jack/Downloads/rlcd-router/TEST_READY.md
- /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_worker_m1_1/DISPATCH.md
- Explorer 1 blueprints: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_1/handoff.md, proposed_config.py, proposed_model_loader.py
- Explorer 2 blueprints: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_2/handoff.md, proposed_stream_extractor.py
- Explorer 3 blueprints: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_3/handoff.md, analysis.md

EXCLUSIVE FILE WRITE OWNERSHIP:
- /Users/jack/Downloads/rlcd-router/src/__init__.py
- /Users/jack/Downloads/rlcd-router/src/config.py
- /Users/jack/Downloads/rlcd-router/src/data/__init__.py
- /Users/jack/Downloads/rlcd-router/src/data/model_loader.py
- /Users/jack/Downloads/rlcd-router/src/data/stream_extractor.py
- /Users/jack/Downloads/rlcd-router/src/data/dataset.py

YOUR OBJECTIVE:
Implement the complete, production-grade Milestone 1 (Data Partitioning & Generation) modules.
Run build/test commands to verify your implementation:
- `pytest -v tests/`
Ensure all tests for Features 1-5 pass without regressions.
Produce a comprehensive handoff report in `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_worker_m1_1/handoff.md` and send a completion message to parent.
</USER_REQUEST>
