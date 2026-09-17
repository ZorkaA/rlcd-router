# BRIEFING — 2026-09-17T07:31:00Z

## Mission
Investigate local environment, Qwen1.5-MoE-A2.7B architecture, streaming/memory management for 100k tokens, and clean 15-20% train/calibration split.

## 🔒 My Identity
- Archetype: explorer
- Roles: investigation, synthesis
- Working directory: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_survey_2
- Original parent: ce5bc762-f633-465c-9133-7ec43d0b5719
- Milestone: Phase 1 Preview / Discovery Survey 2

## 🔒 Key Constraints
- Read-only investigation — do NOT implement production or test code files
- Write outputs to /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_survey_2/analysis.md and handoff.md
- Maintain progress.md as heartbeat
- Send completion message to parent (ce5bc762-f633-465c-9133-7ec43d0b5719)

## Current Parent
- Conversation ID: ce5bc762-f633-465c-9133-7ec43d0b5719
- Updated: 2026-09-17T07:24:32Z

## Investigation State
- **Explored paths**:
  - Hardware & Host Environment: macOS Apple Silicon, 14 cores, 36 GB RAM (~13 GB avail), 187 GB free disk
  - ML Stack: PyTorch 2.2.2, Transformers 4.44.0, Safetensors 0.6.2, Datasets 2.16.0, Pytest 8.3.4, Scikit-learn 1.3.2, Scipy 1.14.1
  - Qwen/Qwen1.5-MoE-A2.7B: config.json, Qwen2MoeSparseMoeBlock forward, router logits tensors, shared expert gate, top-4 routing
  - Token Streaming: Wikitext-2 (2M words), Qwen tokenizer (151k vocab), batch sizing (B=1, L=1024), 98 sequences
  - Memory Management: Zero-OOM streaming protocol, torch.inference_mode(), use_cache=False, immediate .detach().cpu(), safetensors/memmap
  - Data Partitioning: 80% train / 20% calib sequence-level split, lookahead horizon trimming (T+1..T+3)
- **Key findings**:
  - `torch.bfloat16` fails on Apple Silicon MPS in PyTorch 2.2.2; must use `torch.float16` on MPS.
  - Model weights total 26.67 GB in FP16 (14.3B params total, 2.7B active). Automated testing must use synthetic `Qwen2MoeConfig`.
  - Layer N tap: Layer 3 (1-indexed; index 2) recommended for optimal representation and ~20ms async lead time before Layer 5.
  - Extracted 100k-token dataset is compact (~736 MB FP16 / ~1.43 GB FP32) and fits easily in RAM/disk once generated.
- **Unexplored areas**: None for survey 2 scope. All 4 target areas thoroughly investigated and verified.

## Key Decisions Made
- Recommended Layer 3 (1-indexed) as default tap layer, configurable via CLI `--tap-layer 3`.
- Mandated sequence-level 80/20 data partitioning with trailing 3-token horizon trimming to prevent leakage.
- Selected `safetensors` / `np.memmap` for streaming persistence.

## Artifact Index
- DISPATCH.md — task instructions and prompt record
- BRIEFING.md — persistent state memory
- progress.md — liveness heartbeat
- analysis.md — comprehensive 7-section technical analysis report
- handoff.md — self-contained 5-component handoff report
