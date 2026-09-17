## 2026-09-17T07:23:45Z

<USER_REQUEST>
You are the Project Orchestrator for this mission.
Your working directory is: /Users/jack/Downloads/rlcd-router/.agents/orchestrator/
Your project root is: /Users/jack/Downloads/rlcd-router/
The authoritative user request is recorded in: /Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md

Mission:
Build the Phase 1 PyTorch ML Calibration scripts for an Asynchronous MoE Router according to the requirements and acceptance criteria in ORIGINAL_REQUEST.md:
1. R1. Data Partitioning & Generation: Stream a 100k-token corpus through Qwen/Qwen1.5-MoE-A2.7B, logging Layer N hidden states and native gating decisions across all layers. Carve off 15-20% held-out calibration split strictly isolated from training data. Handle memory efficiently without OOM crashes.
2. R2. Linear Speculative Head & Training: Medusa-style linear speculative head attached to Layer N to predict deep-layer routing distributions (Layers 5-24) for T+1 through T+3. Train using Cross-Entropy loss with a tunable Maximum Mean Calibration Error (MMCE) penalty (CE + λ·MMCE).
3. R3. Grid-Based Temperature Scaling: Fit a grid of temperature scalars T(s) indexed by {early layers 5-10, late layers 11-24} × {T+1, T+2, T+3} on the held-out split by strictly minimizing Negative Log-Likelihood (NLL) via LBFGS. Apply L2 regularization toward T=1.0 for sparse buckets. Do not use ECE for optimization.
4. R4. Automated Testing & Verification: Extensive logging and comprehensive automated test suite. Monitor memory usage for leaks. Evaluate Targeted Gating criteria on held-out split: report Targeted ECE specifically at 0.05 (abort) and 0.85 (mass cutoff) decision boundaries broken down by layer bucket and lookahead horizon.

Discipline & Protocols:
- Maintain your BRIEFING.md and progress.md in your working directory (/Users/jack/Downloads/rlcd-router/.agents/orchestrator/). Keep progress.md actively updated as milestones progress.
- Dispatch subagents to implement, verify, and test components.
- Milestone commits: Whenever a major milestone is completed, run git commit proactively.
- When all acceptance criteria are met, report completion back to the Sentinel.
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-09-17T11:23:45+04:00.
</ADDITIONAL_METADATA>
