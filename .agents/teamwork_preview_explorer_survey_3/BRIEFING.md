# BRIEFING — 2026-09-17T07:25:00Z

## Mission
Investigate mathematical calibration, Medusa-style speculative head formulation, RKHS MMCE penalty, grid-based temperature scaling via LBFGS, and targeted gating ECE metrics for the asynchronous MoE router.

## 🔒 My Identity
- Archetype: teamwork_preview_explorer
- Roles: explorer, analyst, surveyor
- Working directory: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_survey_3
- Original parent: ce5bc762-f633-465c-9133-7ec43d0b5719
- Milestone: Survey Phase (Calibration Math & Speculative Head)

## 🔒 Key Constraints
- Read-only investigation — do NOT implement production or test code files.
- Deliver analysis.md and handoff.md in .agents/teamwork_preview_explorer_survey_3/.
- Adhere strictly to the 5-component handoff report protocol.
- Do NOT use ECE for temperature optimization (non-differentiable).
- Use send_message to report results to parent.

## Current Parent
- Conversation ID: ce5bc762-f633-465c-9133-7ec43d0b5719
- Updated: not yet

## Investigation State
- **Explored paths**:
  - `transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeSparseMoeBlock` and `Qwen2MoeConfig`.
  - Medusa speculative head design for MoE routers (3 horizon linear heads: 7.38M params, 14.75 MB).
  - RKHS MMCE penalty formulation with Gaussian RBF kernel, detached indicators, and autograd validation.
  - Grid temperature scaling (2 layer buckets x 3 horizons = 6 buckets) via LBFGS with L2 penalty to 1.0.
  - Mathematical proof of ECE non-differentiability and degeneracy.
  - Targeted ECE at 0.05 (abort) and 0.85 (mass cutoff) decision boundaries.
  - Memory leak monitoring via `tracemalloc`, `psutil` RSS, and active tensor tracking.
- **Key findings**: All mathematical models, loss functions, optimization loops, and metrics fully formulated, documented, and empirically verified in PyTorch.
- **Unexplored areas**: None within survey scope. Ready for handoff to implementation workers.

## Key Decisions Made
- Scoped analysis strictly to mathematical calibration, head architecture, loss functions, temperature scaling, targeted ECE, and memory leak detection.
- Selected Layer 4 (0-based index 3) as the tap layer $N$ to maximize speculative lookahead window.
- Formulated Medusa head as `nn.ModuleList` of 3 independent horizon projection heads ($2048 \to 1200$).
- Recommended soft Cross-Entropy against native router softmax distribution ($q_{\text{soft}}$) with weighted RKHS MMCE penalty ($\lambda = 2.0, \sigma = 0.2$).
- Proved mathematical impossibility and statistical degeneracy of using ECE for temperature optimization; formalized NLL minimization with $\alpha (T - 1.0)^2$ prior via LBFGS.
- Formulated Window Targeted ECE and Cumulative Mass Cutoff metrics for operational boundaries 0.05 and 0.85.

## Artifact Index
- /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_survey_3/DISPATCH.md — Task instructions and updates
- /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_survey_3/progress.md — Liveness and progress tracking
- /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_survey_3/analysis.md — Comprehensive mathematical and architectural analysis (completed)
- /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_survey_3/handoff.md — 5-component handoff report (completed)

