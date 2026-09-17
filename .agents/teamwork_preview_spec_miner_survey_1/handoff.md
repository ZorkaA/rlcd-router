# 5-Component Handoff Report: Specification Mining Survey

**Agent ID**: `teamwork_preview_spec_miner_survey_1`  
**Working Directory**: `/Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_spec_miner_survey_1`  
**Parent Agent**: `orchestrator` (`ce5bc762-f633-465c-9133-7ec43d0b5719`)  
**Task**: Comprehensive Specification Mining for Phase 1 PyTorch ML Calibration scripts for an Asynchronous MoE Router.

---

## 1. Observation

1. **User Requirements Specification**:
   From `/Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md`:
   - Line 11: *"Build the Phase 1 PyTorch ML Calibration scripts for an Asynchronous MoE Router. This pipeline generates calibration data, trains a linear speculative Medusa head, and executes post-hoc mathematical calibration (NLL Temperature Scaling Grid & Targeted ECE evaluation)."*
   - Line 19 (R1): *"Implement a PyTorch script to load `Qwen/Qwen1.5-MoE-A2.7B`. Stream a 100k-token corpus through it, logging the Layer N hidden states and the native gating decisions across all layers. Carve off a 15-20% held-out calibration split strictly isolated from training data."*
   - Line 22 (R2): *"Implement a Medusa-style linear speculative head attached to Layer N to predict deep-layer routing distributions (Layers 5-24) for $T+1$ through $T+3$. Train using Cross-Entropy loss. If confidence ranking exhibits flaws, implement a tunable Maximum Mean Calibration Error (MMCE) penalty (CE + λ·MMCE)."*
   - Line 25 (R3): *"Implement a post-hoc calibration pass. Fit a grid of temperature scalars $T(s)$ indexed by `{early layers 5-10, late layers 11-24} × {T+1, T+2, T+3}`. Fit these scalars on the held-out split by strictly minimizing Negative Log-Likelihood (NLL) via LBFGS. Apply L2 regularization toward $T=1.0$ for sparse buckets. **Do not use ECE for optimization**, as it is non-differentiable."*
   - Line 28 (R4): *"Implement extensive logging and a comprehensive automated test suite. Monitor memory usage to catch leaks. Evaluate the "Targeted Gating" criteria on the held-out split: report Targeted ECE specifically at the 0.05 (abort) and 0.85 (mass cutoff) decision boundaries, broken down by layer bucket and lookahead horizon."*
   - Lines 33–37 (Acceptance Criteria):
     - Pipeline generates data from Qwen1.5-MoE-A2.7B without OOM crashes.
     - Speculative head trains successfully, logging loss curves.
     - LBFGS temperature scaling outputs grid of scalars minimizing NLL.
     - Test suite runs end-to-end without errors.
     - Targeted ECE at 0.05 and 0.85 explicitly computed and printed in report.

2. **Base Model Architecture Inspection**:
   Execution of `AutoConfig.from_pretrained('Qwen/Qwen1.5-MoE-A2.7B')` observed:
   - `model_type`: `"qwen2_moe"`
   - `hidden_size`: `2048`
   - `num_hidden_layers`: `24`
   - `num_experts`: `60` routed experts
   - `num_experts_per_tok`: `4`
   - `shared_expert_intermediate_size`: `5632`
   - `output_router_logits`: `False` (can be enabled via forward parameter `output_router_logits=True`)
   - `torch_dtype`: `"bfloat16"`

3. **Transformers Source Code Inspection**:
   In `transformers.models.qwen2_moe.modeling_qwen2_moe`:
   - `Qwen2MoeSparseMoeBlock.forward` returns `(final_hidden_states, router_logits)`.
   - `router_logits` is computed via `self.gate(hidden_states)` of shape `(batch * seq_len, 60)` with linear projection `nn.Linear(2048, 60, bias=False)`.
   - `Qwen2MoeModel.forward` outputs `all_router_logits` as a tuple of 24 tensors, and `all_hidden_states` as a tuple of 25 tensors when flags are enabled.

4. **Runtime Environment Probed**:
   - Python: 3.10
   - PyTorch: 2.2.2 (MPS available: True, CUDA available: False)
   - Transformers: 4.44.0
   - Unified RAM: 36 GB (38,654,705,664 bytes)
   - CPU: 14 cores

---

## 2. Logic Chain

1. **From R1 and Model Architecture to Data Generation Pipeline**:
   - Observation 2 & 3 establish that `Qwen/Qwen1.5-MoE-A2.7B` has 24 decoder layers, $d_{model}=2048$, and 60 routed experts per layer with top-4 selection.
   - Forward pass with `output_hidden_states=True` and `output_router_logits=True` produces Layer $N$ hidden states ($B \times L \times 2048$) and 24 layers of router logits ($B \times L \times 60$).
   - Because 100k tokens at float16 across 24 layers would exceed active memory if held un-detached in RAM, streaming with batching, `torch.no_grad()`, `.detach().cpu()`, and chunked disk writing is strictly necessary to prevent OOM (AC1).
   - Strict split isolation requires deterministic sequence-level partitioning (80k train / 20k calib) so that no tokens or context from the calibration set leak into training.

2. **From R2 and Medusa Architecture to Speculative Head**:
   - Medusa-style heads operate on hidden state $h_T^{(N)}$ at Layer $N$ to predict future routing decisions.
   - R2 specifies prediction of deep layers (Layers 5–24, total 20 layers) for horizons $T+1, T+2, T+3$.
   - This defines the output tensor dimension as $(B, L, 3, 20, 60)$ using a linear head parameterized by $W \in \mathbb{R}^{2048 \to (3 \times 20 \times 60)}$.
   - The primary objective is Cross-Entropy ($\mathcal{L}_{CE}$) against native top-1 expert targets (or soft routing distributions).
   - When confidence ranking exhibits miscalibration, Kumar et al.'s RKHS formulation defines the empirical MMCE penalty: $\text{MMCE} = \sqrt{\frac{1}{m^2} \mathbf{e}^T \mathbf{K} \mathbf{e} + \epsilon}$, yielding total loss $\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \cdot \text{MMCE}$.

3. **From R3 and Calibration Theory to Grid-Based Temperature Scaling**:
   - The grid is strictly indexed by 2 layer buckets (`early`: 5–10, `late`: 11–24) $\times$ 3 horizons ($T+1, T+2, T+3$), yielding exactly 6 temperature scalars $T_{b, h}$.
   - Calibration must be fitted on the strictly isolated held-out split by minimizing NLL via LBFGS.
   - To handle potential sparsity in buckets (e.g. late layers at $T+3$), an L2 regularizer $\frac{\gamma}{2}(T - 1.0)^2$ must be added to the NLL objective.
   - ECE is non-differentiable; thus, optimizing ECE directly is mathematically unsound and explicitly forbidden by R3.

4. **From R4 to Verification & Targeted Gating**:
   - Targeted Gating assesses model calibration at critical routing operational points: $0.05$ (aborting speculative prefetch) and $0.85$ (mass cutoff).
   - Targeted ECE evaluates localized calibration error $|\text{acc}(B^*) - \text{conf}(B^*)|$ within a narrow window around these decision boundaries.
   - Evaluating across 2 layer buckets and 3 horizons pre- and post-calibration confirms whether temperature scaling successfully reduced calibration error at the target operational thresholds.
   - Memory leak monitoring ensures peak memory remains bounded over iterations.

---

## 3. Caveats

1. **Layer Indexing Convention**:
   - In standard human/spec terminology, "Layers 5–24" is 1-based indexing (total 20 layers). In 0-based PyTorch indexing, this maps to `model.layers[4:24]`. Similarly, "early layers 5–10" maps to 1-based 5–10 (0-based 4–9), and "late layers 11–24" maps to 1-based 11–24 (0-based 10–23). The implementation must be completely consistent across components.
2. **Layer N Selection**:
   - The specification references "Layer N" generically. It must be an intermediate layer before the deep layers (5–24) begin, such as Layer 3 or 4 (1-based), so that speculative routing can execute asynchronously while earlier layers are processing.
3. **Weight Download vs. Synthetic Testing**:
   - Downloading the full 14B parameter weights (~28 GB) of `Qwen1.5-MoE-A2.7B` takes significant bandwidth and time. For the automated test suite (`pytest`) to run quickly and reliably in offline or CI environments, a synthetic data generator matching the exact tensor shapes and interfaces must be provided alongside the real model loader.

---

## 4. Conclusion

All functional, mathematical, interface, and non-functional requirements of ORIGINAL_REQUEST.md have been probed, verified against authoritative sources, and exhaustively documented in `analysis.md`.
The pipeline cleanly decomposes into four modular components:
1. `data_generation`: Streaming 100k corpus, Layer N extraction, native router logging, isolated 15–20% split.
2. `speculative_head`: Medusa linear head, forward pass $(B, L, 3, 20, 60)$, CE loss, RKHS MMCE penalty.
3. `temperature_scaling`: $2 \times 3$ grid container, LBFGS NLL minimization with L2 regularization to $T=1.0$.
4. `verification`: Targeted ECE at $0.05$ and $0.85$, memory leak tracker, and end-to-end test suite.

---

## 5. Verification Method

To independently verify the findings in this report and `analysis.md`:
1. **Inspect Analysis and Dispatch Documents**:
   - `view_file /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_spec_miner_survey_1/analysis.md`
   - `view_file /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_spec_miner_survey_1/DISPATCH.md`
   - `view_file /Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md`
2. **Verify Base Model Configuration**:
   ```bash
   python3 -c "from transformers import AutoConfig; cfg = AutoConfig.from_pretrained('Qwen/Qwen1.5-MoE-A2.7B'); print('Layers:', cfg.num_hidden_layers, 'Hidden:', cfg.hidden_size, 'Experts:', cfg.num_experts, 'TopK:', cfg.num_experts_per_tok)"
   ```
   *Expected Output*: `Layers: 24 Hidden: 2048 Experts: 60 TopK: 4`
3. **Verify RKHS MMCE and Temperature Scaling Mathematical Contracts**:
   - Verify that in `analysis.md` Section 3.1.2, the kernel embedding formulation $\mathbf{e}^T \mathbf{K} \mathbf{e}$ includes $\epsilon=10^{-8}$ numerical stabilizer.
   - Verify that Section 3.2 explicitly rejects ECE as an optimization target and defines NLL + L2 regularizer.
