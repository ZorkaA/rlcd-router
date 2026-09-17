# Spec Miner Survey Task

You are teamwork_preview_spec_miner_survey_1.
Working directory: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_spec_miner_survey_1
Parent: orchestrator (ce5bc762-f633-465c-9133-7ec43d0b5719)

Your task:
Mine and document all formal specifications, functional and mathematical requirements, interfaces, acceptance criteria, and edge cases from /Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md.
Produce a comprehensive analysis.md and handoff.md in your working directory.

## 2026-09-17T07:24:32Z
You are teamwork_preview_spec_miner_survey_1.
Your working directory is: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_spec_miner_survey_1
Your parent is orchestrator (conversation ID: ce5bc762-f633-465c-9133-7ec43d0b5719).

MANDATORY INPUT: Read /Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md thoroughly before starting.
Also read your /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_spec_miner_survey_1/DISPATCH.md.

YOUR ROLE & OBJECTIVE:
You are a read-only specification investigator. Probe the authoritative source of truth (/Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md and related MoE / Medusa / calibration literature and specifications) to extract every single functional requirement, mathematical requirement, non-functional requirement, interface contract, acceptance criteria, boundary condition, and edge case for:
1. R1: Data Partitioning & Generation (Qwen/Qwen1.5-MoE-A2.7B, streaming 100k-token corpus, Layer N hidden states logging, all-layer native gating decisions, 15-20% held-out calibration split strictly isolated, OOM avoidance).
2. R2: Linear Speculative Head & Training (Medusa-style linear speculative head on Layer N, deep-layer routing distributions for layers 5-24, horizons T+1, T+2, T+3, Cross-Entropy loss, tunable MMCE penalty formulation CE + \lambda * MMCE).
3. R3: Grid-Based Temperature Scaling (grid of temperature scalars indexed by {early layers 5-10, late layers 11-24} x {T+1, T+2, T+3}, strictly NLL minimization via LBFGS, L2 regularization towards T=1.0 for sparse buckets, strictly NO ECE in optimization).
4. R4: Automated Testing & Verification (logging, test suite, memory leak monitoring, Targeted Gating criteria on held-out split with Targeted ECE at 0.05 and 0.85 decision boundaries broken down by layer bucket and horizon).

SCOPE BOUNDARIES:
- DO NOT write any production or test code.
- Write your findings to /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_spec_miner_survey_1/analysis.md and /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_spec_miner_survey_1/handoff.md.
- Maintain progress.md in your working directory.
- Send a completion message to parent when done.
