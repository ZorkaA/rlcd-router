# Explorer Survey Calib Task

You are teamwork_preview_explorer_survey_3.
Working directory: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_survey_3
Parent: orchestrator (ce5bc762-f633-465c-9133-7ec43d0b5719)

Your task:
Investigate the mathematical calibration and speculative head formulation:
1. Medusa-style linear speculative head attached to Layer N predicting deep-layer routing distributions (layers 5-24) for T+1, T+2, T+3.
2. Cross-entropy loss and tunable Maximum Mean Calibration Error (MMCE) penalty (CE + \lambda * MMCE) using reproducing kernel Hilbert space (RKHS) Gaussian kernel.
3. Post-hoc grid-based temperature scaling: grid indexed by {early layers 5-10, late layers 11-24} x {T+1, T+2, T+3} fitting strictly Negative Log-Likelihood (NLL) via LBFGS with L2 regularization towards T=1.0 for sparse buckets (no ECE in optimization).
4. Targeted Gating evaluation on held-out split: Targeted ECE at 0.05 (abort) and 0.85 (mass cutoff) decision boundaries broken down by layer bucket and lookahead horizon.
5. Testing strategy and verification metrics.
Produce a comprehensive analysis.md and handoff.md in your working directory.
Reference /Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md.

## 2026-09-17T07:24:32Z
Investigate the mathematical calibration, speculative head formulation, and evaluation metrics:
1. Medusa-style Linear Speculative Head:
   - Architecture: Linear projection heads from Layer N hidden state h_N(t) to predict router logits / distributions for layers 5-24 at future token steps t+1, t+2, t+3.
   - Formulation: Single or multi-head linear mapping, target distributions (ground truth router softmax probabilities or top-k labels/one-hot/soft distributions).
   - Loss function: Cross-Entropy loss over deep-layer expert selections across T+1, T+2, T+3.
   - Tunable MMCE penalty: Exact formulation of Maximum Mean Calibration Error penalty (Kumar et al., RKHS kernel calibration error, differentiable surrogate for ECE, bandwidth selection, weighting parameter \lambda).
2. Grid-Based Temperature Scaling:
   - Grid definition: 2 layer buckets ({early layers 5-10, late layers 11-24}) x 3 lookahead horizons ({T+1, T+2, T+3}) = 6 temperature scalar buckets T(s).
   - Optimization: Strict Negative Log-Likelihood (NLL) minimization using torch.optim.LBFGS on the held-out calibration split.
   - Regularization: L2 penalty towards T=1.0 for sparse buckets: L = NLL + \alpha * (T - 1.0)^2.
   - Non-differentiability of ECE and why ECE must NOT be used for temperature optimization.
3. Targeted Gating Evaluation & Metrics:
   - Targeted ECE definition at decision boundaries: specifically at 0.05 (abort) and 0.85 (mass cutoff).
   - How Targeted ECE should be formulated and calculated per bucket and horizon.
   - Memory leak monitoring methodology for testing.

