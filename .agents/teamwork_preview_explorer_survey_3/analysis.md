# Mathematical Calibration, Speculative Head Formulation, and Targeted Gating Metrics for Asynchronous MoE Routing

**Author**: `teamwork_preview_explorer_survey_3`  
**Target Architecture**: `Qwen/Qwen1.5-MoE-A2.7B`  
**Date**: September 17, 2026  
**Status**: Technical Survey & Analytical Formulation  

---

## 1. Executive Summary

This report delivers the complete mathematical formulation, architectural specification, loss functions, post-hoc temperature scaling grid, targeted calibration metrics, and memory verification methodology for the Phase 1 PyTorch ML Calibration Pipeline of the Asynchronous MoE Router.

Key findings and architectural decisions:
1. **Model Dimensions**: `Qwen/Qwen1.5-MoE-A2.7B` has 24 decoder layers, hidden dimension $d_{\text{model}} = 2048$, 60 routed experts per layer, and activates top-4 experts per token. Deep layers 5–24 (1-based, 20 layers total) require routing predictions across lookahead horizons $T+1, T+2, T+3$.
2. **Medusa-style Speculative Head**: Formulated as 3 independent horizon projection heads tapped from Layer $N$ hidden state ($h_N(t) \in \mathbb{R}^{2048}$, with $N=4$ recommended). Each horizon head maps $d_{\text{model}} \to 20 \times 60 = 1200$ logits. Total parameter count is $3 \times (2048 \times 1200 + 1200) \approx 7.38\text{M}$ parameters (~14.75 MB in BF16), ensuring sub-microsecond latency.
3. **Loss Formulation & Target Distributions**: We formalize soft Cross-Entropy matching the full native router softmax distribution ($q_{\text{soft}}$) or top-4 normalized active routing weights ($q_{\text{top4}}$). To resolve confidence misranking, we integrate the Maximum Mean Calibration Error (MMCE) penalty (Kumar et al., ICML 2018) using an RKHS universal Gaussian RBF kernel with bandwidth $\sigma = 0.2$, full batch vectorization, and detached correctness indicators.
4. **Grid-Based Post-Hoc Temperature Scaling**: A $2 \times 3$ grid comprising 6 temperature scalar buckets $\{\text{early: 5-10}, \text{late: 11-24}\} \times \{T+1, T+2, T+3\}$ is fitted on the held-out calibration split. Optimization strictly minimizes Negative Log-Likelihood (NLL) regularized with an L2 penalty towards $T=1.0$ using `torch.optim.LBFGS` with strong Wolfe line search. We provide a rigorous mathematical proof demonstrating that Expected Calibration Error (ECE) is piecewise constant, non-differentiable ($\nabla_T \text{ECE} = 0$ almost everywhere), and leads to degenerate solutions, making its use in optimization fundamentally invalid.
5. **Targeted Gating Evaluation Metrics**: We formulate Targeted ECE directly mapped to operational runtime decisions: $\theta_{\text{abort}} = 0.05$ (speculative execution abort) and $\theta_{\text{mass}} = 0.85$ (cumulative mass cutoff for speculative expert set selection).
6. **Memory Leak Prevention & Verification**: We establish a multi-tier test harness combining `tracemalloc`, `psutil` Resident Set Size (RSS) monitoring across loop iterations, and `gc.get_objects()` tensor tracking to guarantee zero memory accumulation on Mac MPS/CPU runtimes.

---

## 2. Medusa-style Linear Speculative Head

### 2.1 Layer N Tap Selection & Input Dimensionality
In `Qwen/Qwen1.5-MoE-A2.7B`:
- Total transformer decoder layers: $L_{\text{total}} = 24$ (indexed 0 to 23 in 0-based, or 1 to 24 in 1-based).
- Hidden state dimension: $d_{\text{model}} = 2048$.
- Deep layers to predict: Layers 5 through 24 ($L_{\text{deep}} = 20$ layers).
- Number of routed experts per layer: $E = 60$.
- Native expert selection per token: $k = 4$.

**Layer Tap Selection ($N$)**:
The speculative head requires an intermediate representation early in the network before deep routing occurs. We recommend **Layer 4** (0-based index 3):
- Layer 4 captures early lexical and syntactic context while executing before Layer 5.
- By tapping the hidden state $h_N(t) \in \mathbb{R}^{d_{\text{model}}}$ immediately following Layer 4's post-attention and MLP residual addition, downstream asynchronous worker queues have the maximum possible lookahead window to prefetch weights for Layers 5–24.

### 2.2 Projection Head Architecture
Following the Medusa paradigm (Cai et al., 2024), we deploy $K = 3$ distinct linear projection heads, where each head $\tau \in \{1, 2, 3\}$ corresponds to lookahead horizon $T+\tau$:

$$\hat{Z}^{(\tau)}(t) = W^{(\tau)} h_N(t) + b^{(\tau)}$$

where:
- $h_N(t) \in \mathbb{R}^{d_{\text{model}}} = \mathbb{R}^{2048}$
- $W^{(\tau)} \in \mathbb{R}^{(L_{\text{deep}} \times E) \times d_{\text{model}}} = \mathbb{R}^{1200 \times 2048}$
- $b^{(\tau)} \in \mathbb{R}^{1200}$
- $\hat{Z}^{(\tau)}(t) \in \mathbb{R}^{1200}$ is reshaped into $\mathbb{R}^{L_{\text{deep}} \times E} = \mathbb{R}^{20 \times 60}$.

```
Input: h_N(t) [Batch, SeqLen, 2048]
   │
   ├─── Head 1 (T+1) ─── Linear(2048, 1200) ─── Reshape [Batch, SeqLen, 20, 60] -> Logits for T+1
   ├─── Head 2 (T+2) ─── Linear(2048, 1200) ─── Reshape [Batch, SeqLen, 20, 60] -> Logits for T+2
   └─── Head 3 (T+3) ─── Linear(2048, 1200) ─── Reshape [Batch, SeqLen, 20, 60] -> Logits for T+3
```

#### Parameter Footprint & Computational Budget
- Parameters per head: $(2048 \times 1200) + 1200 = 2,458,800$ parameters (~2.46M).
- Total parameters for 3 heads: $3 \times 2,458,800 = 7,376,400$ parameters (~7.38M).
- Storage size:
  - FP32: $7,376,400 \times 4 \text{ bytes} \approx 29.5 \text{ MB}$.
  - BF16 / FP16: $7,376,400 \times 2 \text{ bytes} \approx 14.75 \text{ MB}$.
- Compute cost per token: $2 \times 2048 \times 3600 \approx 14.7 \text{ MFLOPs}$, executing in $< 0.1 \text{ ms}$ on Apple Silicon MPS or CPU.

#### Modularity Comparison: Per-Horizon Heads vs Unified Projection
| Aspect | Independent Horizon Heads (`nn.ModuleList`) | Single Unified Projection (`nn.Linear(2048, 3600)`) |
|---|---|---|
| Parameter Count | 7,376,400 | 7,376,400 (Identical) |
| FLOPs | 14.7 MFLOPs | 14.7 MFLOPs (Identical) |
| Horizon Ablation / Freezing | Trivial: each head can be frozen or trained independently | Harder: requires weight slicing or masked gradients |
| Code Readability | High: mirrors Medusa paper structure | Moderate |
| **Recommendation** | **Adopt `nn.ModuleList` of 3 Linear Heads** | Alternative fallback |

---

### 2.3 Target Distributions Formulation

Let $g_l(t+\tau) \in \mathbb{R}^{60}$ denote the ground truth router logits produced by `Qwen1.5-MoE-A2.7B` at layer $l \in \{5, \dots, 24\}$ for future token $t+\tau$. Three candidate target distributions exist:

#### 1. Full Native Router Softmax Distribution ($q_{\text{soft}}$)
$$q_{\text{soft}}(e) = \frac{\exp(g_l(t+\tau)[e])}{\sum_{j=1}^{60} \exp(g_l(t+\tau)[j])}, \quad e \in \{1, \dots, 60\}$$
- **Pros**: Contains complete dark knowledge of the teacher router's relative expert rankings and confidence spread across all 60 experts. Soft labels prevent overconfident delta spikes.
- **Cons**: Assigns tiny probabilities to inactive experts ($e \notin \text{Top4}$).

#### 2. Top-$k$ Normalized Distribution ($q_{\text{top4}}$)
Let $\mathcal{T} = \text{Top4}(g_l(t+\tau))$ be the set of 4 experts selected by the native router:
$$q_{\text{top4}}(e) = \begin{cases} \frac{\exp(g_l(t+\tau)[e])}{\sum_{j \in \mathcal{T}} \exp(g_l(t+\tau)[j])}, & \text{if } e \in \mathcal{T} \\ 0, & \text{if } e \notin \mathcal{T} \end{cases}$$
- **Pros**: Matches the exact routing weights used during forward execution. Forces the head to assign zero mass to unselected experts.
- **Cons**: Semi-sparse target (56 zeros). Requires soft Cross-Entropy or KL divergence.

#### 3. Hard Top-1 Label ($y_{\text{top1}}$)
$$y_{\text{top1}} = \arg\max_{e \in \{1, \dots, 60\}} g_l(t+\tau)[e]$$
- **Pros**: Standard multi-class classification (`nn.CrossEntropyLoss`).
- **Cons**: Fails to account for experts 2, 3, and 4 which are also dispatched and executed by the MoE block.

**Analytical Conclusion**:
We recommend supporting **$q_{\text{soft}}$ (Teacher-Student Distillation)** as primary, and **$q_{\text{top4}}$** as an ablation. For evaluation against standard cross-entropy benchmarks, top-1 accuracy and top-4 recall should both be tracked.

---

### 2.4 Loss Function: Multi-Horizon Deep-Layer Cross-Entropy

For a batch of sequences of length $S$, let predicted logits for horizon $\tau \in \{1, 2, 3\}$ and layer $l \in \{5, \dots, 24\}$ at token position $t$ be:
$$\hat{z}_{\tau, l}(t) \in \mathbb{R}^{60}$$
The predicted probability vector is:
$$\hat{p}_{\tau, l}(t) = \text{Softmax}(\hat{z}_{\tau, l}(t)) \in \Delta^{60}$$
$$\log \hat{p}_{\tau, l}(t) = \text{LogSoftmax}(\hat{z}_{\tau, l}(t))$$

The multi-horizon deep-layer Cross-Entropy loss is formulated as:
$$\mathcal{L}_{\text{CE}} = \frac{1}{|\mathcal{D}|} \sum_{t \in \mathcal{D}} \sum_{\tau=1}^3 \sum_{l=5}^{24} w_\tau \cdot \mathcal{L}_{\text{CE}}^{(\tau, l)}(t)$$

where $w_\tau$ is the horizon weighting factor (default $w_1 = w_2 = w_3 = \frac{1}{3 \times 20} = \frac{1}{60}$), and:
- Under soft target distribution $q_{\tau, l}(t) \in \Delta^{60}$:
  $$\mathcal{L}_{\text{CE}}^{(\tau, l)}(t) = -\sum_{e=1}^{60} q_{\tau, l}(t)[e] \log \hat{p}_{\tau, l}(t)[e]$$
- Under hard top-1 target $y_{\tau, l}(t) \in \{1, \dots, 60\}$:
  $$\mathcal{L}_{\text{CE}}^{(\tau, l)}(t) = -\log \hat{p}_{\tau, l}(t)[y_{\tau, l}(t)]$$

---

### 2.5 Tunable Maximum Mean Calibration Error (MMCE) Penalty

#### 2.5.1 Theoretical Background (Kumar et al., ICML 2018)
Standard Cross-Entropy training suffers from the "overconfidence phenomenon" because Cross-Entropy is minimized only when logit magnitudes approach infinity. Post-hoc calibration can scale logits, but cannot alter the confidence ordering of predictions.

To learn intrinsically well-calibrated representations during training, Kumar et al. (2018) introduced the **Maximum Mean Calibration Error (MMCE)**. MMCE maps calibration residuals into a Reproducing Kernel Hilbert Space (RKHS), providing a continuous, differentiable surrogate for Expected Calibration Error (ECE).

Let:
- $r_i = \max_e \hat{p}_i(e) \in [0, 1]$ be the predicted confidence for instance $i$.
- $c_i \in \{0, 1\}$ be the correctness indicator. In MoE routing:
  $$c_i = \mathbb{I}\left( \arg\max_e \hat{p}_i(e) \in \mathcal{T}_i \right)$$
  where $\mathcal{T}_i$ is the ground-truth top-4 native expert set. (Alternatively, for top-1: $c_i = \mathbb{I}(\arg\max_e \hat{p}_i(e) = \arg\max_e q_i(e))$).
- The calibration residual is:
  $$e_i = c_i - r_i$$

In an RKHS $\mathcal{H}$ associated with a universal kernel $k(\cdot, \cdot)$, the calibration error embedding is:
$$\mu_{\text{cal}} = \mathbb{E}[(c - r) \phi(r)] \in \mathcal{H}, \quad \text{where } \langle \phi(r), \phi(r') \rangle_{\mathcal{H}} = k(r, r')$$

The Maximum Mean Calibration Error is the RKHS norm:
$$\text{MMCE} = \|\mu_{\text{cal}}\|_{\mathcal{H}} = \sup_{f \in \mathcal{H}, \|f\|_{\mathcal{H}} \le 1} \mathbb{E}[(c - r) f(r)]$$
$$\text{MMCE}^2 = \mathbb{E}_{(r, c), (r', c')}\left[ (c - r)(c' - r') k(r, r') \right]$$

#### 2.5.2 Empirical Batch Estimator
For a mini-batch of $M$ predictions (aggregated across tokens, layers, and horizons):
$$\widehat{\text{MMCE}}^2 = \frac{1}{M^2} \sum_{i=1}^M \sum_{j=1}^M (c_i - r_i)(c_j - r_j) k(r_i, r_j)$$

In matrix notation:
Let residual vector $\mathbf{e} = [c_1 - r_1, \dots, c_M - r_M]^T \in \mathbb{R}^M$.  
Let Gram matrix $K \in \mathbb{R}^{M \times M}$ have entries $K_{i, j} = k(r_i, r_j)$.  
Then:
$$\widehat{\text{MMCE}}^2 = \frac{1}{M^2} \mathbf{e}^T K \mathbf{e}$$
$$\widehat{\text{MMCE}} = \sqrt{\max\left(\widehat{\text{MMCE}}^2, \epsilon\right)}, \quad \epsilon = 10^{-8}$$

#### 2.5.3 Weighted MMCE ($\text{MMCE}_w$) for Imbalanced Accuracy
In trained models, the correctness indicator $c_i$ is typically skewed (e.g. 70–90% correct, 10–30% incorrect). In unweighted MMCE, the double sum is dominated by pairs where both samples are correct ($c_i = 1, c_j = 1$).

Kumar et al. derived the weighted estimator $\text{MMCE}_w$:
Let $S_1 = \{i : c_i = 1\}$ with count $M_1 = |S_1|$, and $S_0 = \{i : c_i = 0\}$ with count $M_0 = |S_0|$.
Define sample weights:
$$w_i = \begin{cases} \frac{1}{2 M_1}, & \text{if } c_i = 1 \\ \frac{1}{2 M_0}, & \text{if } c_i = 0 \end{cases}$$
*(If $M_0 = 0$ or $M_1 = 0$, fall back to $w_i = \frac{1}{M}$)*.

The weighted empirical estimator is:
$$\widehat{\text{MMCE}}_w^2 = \sum_{i=1}^M \sum_{j=1}^M w_i w_j (c_i - r_i)(c_j - r_j) k(r_i, r_j) = (\mathbf{w} \odot \mathbf{e})^T K (\mathbf{w} \odot \mathbf{e})$$
$$\widehat{\text{MMCE}}_w = \sqrt{\max\left(\widehat{\text{MMCE}}_w^2, \epsilon\right)}$$

#### 2.5.4 Kernel Function & Bandwidth Selection
We employ the universal Gaussian (Radial Basis Function) kernel:
$$k(r_i, r_j) = \exp\left( -\frac{(r_i - r_j)^2}{2\sigma^2} \right)$$

- **Bandwidth $\sigma$**:
  Since confidence scores $r_i \in [0, 1]$, distances $|r_i - r_j| \in [0, 1]$.
  - If $\sigma$ is too small ($\sigma < 0.05$): $K \approx I$, and the penalty degenerates into point-wise variance penalty.
  - If $\sigma$ is too large ($\sigma > 1.0$): $K \approx \mathbf{1}\mathbf{1}^T$, and the penalty only measures the global mean gap $\mathbb{E}[c - r]$.
  - **Optimal recommendation**: Fixed bandwidth $\sigma = 0.2$ (standard in Kumar et al., 2018), or a multi-scale mixture kernel:
    $$k_{\text{multi}}(r_i, r_j) = \frac{1}{3} \left[ \exp\left(-\frac{(r_i - r_j)^2}{2(0.1)^2}\right) + \exp\left(-\frac{(r_i - r_j)^2}{2(0.2)^2}\right) + \exp\left(-\frac{(r_i - r_j)^2}{2(0.4)^2}\right) \right]$$

#### 2.5.5 Autograd Gradient Flow & Weighting Parameter $\lambda$
- **Gradient Separation**:
  The indicator $c_i = \mathbb{I}(\hat{y}_i \in \mathcal{T}_i)$ is a discrete step function. In autograd, $c_i$ **must be detached**:
  $$c_i = c_i\text{.detach()}$$
  Gradients $\frac{\partial \text{MMCE}}{\partial \theta}$ flow strictly through confidence $r_i = \max_e \text{Softmax}(\hat{z}_i)[e]$:
  $$\frac{\partial \text{MMCE}}{\partial r_i} = \frac{1}{\text{MMCE}} \cdot \frac{1}{M^2} \sum_{j=1}^M \left[ -e_j k(r_i, r_j) - e_i e_j \frac{(r_i - r_j)}{\sigma^2} k(r_i, r_j) \right]$$
  This pulls overconfident incorrect predictions ($c_i = 0, r_i \approx 1$) towards lower confidence and pushes underconfident correct predictions ($c_i = 1, r_i \approx 0$) towards higher confidence.
- **Total Objective**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \cdot \widehat{\text{MMCE}}_w$$
  We recommend tuning $\lambda \in [1.0, 5.0]$ (default $\lambda = 2.0$). If validation ECE remains elevated after epoch 1, $\lambda$ can be increased up to $10.0$.

#### 2.5.6 Vectorized PyTorch Implementation
```python
import torch
import torch.nn.functional as F

def compute_mmce_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    sigma: float = 0.2,
    weighted: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Vectorized Maximum Mean Calibration Error (MMCE) penalty.
    
    Args:
        probs: (M, E) Predicted softmax probability distribution across E experts.
        targets: (M, k) Ground truth active top-k expert indices (or (M,) top-1 index).
        sigma: Kernel bandwidth parameter for Gaussian RBF kernel.
        weighted: Whether to apply class-imbalance weighting (MMCE_w).
        eps: Numerical stability epsilon.
    Returns:
        Scalar MMCE loss tensor with autograd gradients enabled.
    """
    # 1. Predicted confidence and top-1 expert
    conf, pred_top1 = torch.max(probs, dim=-1) # (M,)
    
    # 2. Correctness indicator c (detached from computational graph)
    if targets.dim() == 1:
        c = (pred_top1 == targets).float().detach()
    else:
        # Check if predicted top-1 is within ground truth top-k experts
        c = (pred_top1.unsqueeze(-1) == targets).any(dim=-1).float().detach() # (M,)
        
    e = c - conf # Residual (M,)
    M = probs.size(0)
    
    # 3. Weights calculation
    if weighted:
        m1 = torch.sum(c)
        m0 = M - m1
        w1 = 1.0 / (2.0 * m1) if m1 > 0 else 1.0 / M
        w0 = 1.0 / (2.0 * m0) if m0 > 0 else 1.0 / M
        weights = torch.where(c == 1.0, w1, w0) # (M,)
    else:
        weights = torch.full((M,), 1.0 / M, device=probs.device, dtype=probs.dtype)
        
    we = weights * e # (M,)
    
    # 4. Pairwise kernel matrix
    diff = conf.unsqueeze(1) - conf.unsqueeze(0) # (M, M)
    K = torch.exp(- (diff ** 2) / (2.0 * (sigma ** 2))) # (M, M)
    
    # 5. Quadratic form (we^T * K * we)
    mmce_sq = torch.sum(we.unsqueeze(1) * K * we.unsqueeze(0))
    return torch.sqrt(torch.clamp(mmce_sq, min=eps))
```

---

## 3. Grid-Based Temperature Scaling

### 3.1 Grid Architecture & Partitioning
Post-hoc calibration scales the speculative head's raw logits $z \in \mathbb{R}^{60}$ by a learned scalar temperature $T > 0$. Because prediction entropy naturally varies across network depth and temporal horizons, a single scalar is insufficient, while fitting 60 independent temperatures risks overfitting.

We construct a **$2 \times 3$ Grid** yielding **6 temperature scalar buckets**:
- **Layer Buckets (2 partitions)**:
  - $\mathcal{B}_{\text{early}} = \{5, 6, 7, 8, 9, 10\}$ (Layers 5–10, 6 layers)
  - $\mathcal{B}_{\text{late}} = \{11, 12, \dots, 24\}$ (Layers 11–24, 14 layers)
- **Lookahead Horizons (3 partitions)**:
  - $\mathcal{H}_1 = T+1$ ($\Delta t = 1$)
  - $\mathcal{H}_2 = T+2$ ($\Delta t = 2$)
  - $\mathcal{H}_3 = T+3$ ($\Delta t = 3$)

$$\mathcal{S} = \{ (\mathcal{B}_{\text{early}}, T+1), (\mathcal{B}_{\text{early}}, T+2), (\mathcal{B}_{\text{early}}, T+3), (\mathcal{B}_{\text{late}}, T+1), (\mathcal{B}_{\text{late}}, T+2), (\mathcal{B}_{\text{late}}, T+3) \}$$

| Bucket ID $s$ | Layer Group | Horizon | Expected Calibration Dynamic |
|---|---|---|---|
| $s_1$ | Early (5–10) | $T+1$ | Lowest entropy, highest predictive accuracy |
| $s_2$ | Early (5–10) | $T+2$ | Moderate entropy |
| $s_3$ | Early (5–10) | $T+3$ | High entropy |
| $s_4$ | Late (11–24) | $T+1$ | Deeper features, moderate entropy |
| $s_5$ | Late (11–24) | $T+2$ | Higher entropy |
| $s_6$ | Late (11–24) | $T+3$ | Highest uncertainty and dispersion |

For any logit vector $z \in \mathbb{R}^{60}$ at layer $l$ and horizon $\tau$, calibrated logits are computed as:
$$z_{\text{cal}} = \frac{z}{T_{s(l, \tau)}}, \quad \hat{p}_{\text{cal}} = \text{Softmax}\left(\frac{z}{T_{s(l, \tau)}}\right)$$

---

### 3.2 Optimization: Strict NLL Minimization with L2 Regularization

Let $\mathcal{D}_s = \{(z_i, y_i)\}_{i=1}^{M_s}$ denote the held-out calibration split data for bucket $s$.

#### Objective Function
The temperature $T_s$ is optimized by strictly minimizing Negative Log-Likelihood (NLL) with an L2 regularization penalty pulling towards $T = 1.0$:

$$\mathcal{L}(T_s) = \text{NLL}(T_s) + \alpha \cdot (T_s - 1.0)^2$$

where:
- For hard top-1 targets $y_i \in \{1, \dots, 60\}$:
  $$\text{NLL}(T_s) = -\frac{1}{M_s} \sum_{i=1}^{M_s} \left[ \frac{z_{i, y_i}}{T_s} - \log \sum_{e=1}^{60} \exp\left(\frac{z_{i, e}}{T_s}\right) \right]$$
- For soft target distribution $q_i \in \Delta^{60}$:
  $$\text{NLL}(T_s) = -\frac{1}{M_s} \sum_{i=1}^{M_s} \sum_{e=1}^{60} q_{i, e} \log \left( \frac{\exp(z_{i, e} / T_s)}{\sum_{j=1}^{60} \exp(z_{i, j} / T_s)} \right)$$

#### Rationale for Regularization towards $T = 1.0$
- **Bayesian Gaussian Prior**: The penalty $\alpha (T_s - 1.0)^2$ corresponds to a Gaussian prior $p(T_s) \sim \mathcal{N}\left(1.0, \frac{1}{2\alpha}\right)$.
- **Sparse Bucket Stability**: If a bucket contains limited calibration samples or noisy predictions, unregularized NLL can drift towards extreme values ($T \to 0$ or $T \to \infty$). Regularization ensures sparse buckets safely default to the unscaled identity temperature $T = 1.0$.
- **Hyperparameter $\alpha$**: Recommended default $\alpha = 0.05$ (or $\alpha = \frac{10}{M_s}$).

---

### 3.3 Optimization with `torch.optim.LBFGS`

LBFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) is a quasi-Newton optimization algorithm that uses second-order curvature information. Because fitting a scalar temperature $T$ is a strictly convex 1D optimization problem, LBFGS converges in 15–30 iterations with machine precision.

```python
import torch
import torch.nn.functional as F

class GridTemperatureScaler:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        # 6 temperature scalars initialized to 1.0
        # Index: [early/late (2), horizon (3)]
        self.T_grid = torch.ones(2, 3)

    def fit_bucket(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Fits a single temperature scalar T for a bucket via LBFGS.
        logits: (M, 60)
        targets: (M,)
        """
        T = torch.tensor([1.0], requires_grad=True, device=logits.device)
        optimizer = torch.optim.LBFGS(
            [T],
            lr=0.1,
            max_iter=50,
            line_search_fn="strong_wolfe"
        )
        
        def closure():
            optimizer.zero_grad()
            t_clamped = torch.clamp(T, min=1e-3, max=100.0)
            scaled_logits = logits / t_clamped
            nll = F.cross_entropy(scaled_logits, targets)
            reg = self.alpha * ((t_clamped - 1.0) ** 2)
            loss = nll + reg
            loss.backward()
            return loss
            
        optimizer.step(closure)
        return torch.clamp(T, min=1e-3, max=100.0).item()
```

---

### 3.4 Mathematical Proof: Why ECE Must NOT Be Used for Optimization

The user prompt mandates: *"Do not use ECE for optimization, as it is non-differentiable."* Below is the rigorous mathematical justification.

#### 1. Expected Calibration Error Definition
Given $M_{\text{bins}}$ disjoint probability bins $B_m = [b_m, b_{m+1}) \subset [0, 1]$, empirical ECE is defined as:
$$\text{ECE}(T) = \sum_{m=1}^{M_{\text{bins}}} \frac{|B_m(T)|}{N} \left| \text{acc}(B_m(T)) - \text{conf}(B_m(T)) \right|$$
where:
$$B_m(T) = \left\{ i \in \{1, \dots, N\} : r_i(T) \in [b_m, b_{m+1}) \right\}$$
$$r_i(T) = \max_e \frac{\exp(z_{i, e} / T)}{\sum_j \exp(z_{i, j} / T)}$$
$$\text{acc}(B_m(T)) = \frac{1}{|B_m(T)|} \sum_{i \in B_m(T)} c_i, \quad \text{conf}(B_m(T)) = \frac{1}{|B_m(T)|} \sum_{i \in B_m(T)} r_i(T)$$

Rewriting using indicator functions:
$$\text{ECE}(T) = \frac{1}{N} \sum_{m=1}^{M_{\text{bins}}} \left| \sum_{i=1}^N \mathbb{I}\left(r_i(T) \in [b_m, b_{m+1})\right) \cdot (c_i - r_i(T)) \right|$$

#### 2. Vanishing Gradients Almost Everywhere
Consider the gradient with respect to temperature $T$:
$$\frac{\partial}{\partial T} \mathbb{I}\left(r_i(T) \in [b_m, b_{m+1})\right) = \left[ \delta(r_i(T) - b_m) - \delta(r_i(T) - b_{m+1}) \right] \cdot \frac{\partial r_i(T)}{\partial T}$$

For any dataset where sample confidences do not fall exactly on the infinitesimal boundary points $\{b_m\}$:
$$\frac{\partial}{\partial T} \mathbb{I}\left(r_i(T) \in [b_m, b_{m+1})\right) = 0 \quad \text{almost everywhere (a.e.)}$$

Consequently, across almost the entire real line $T \in \mathbb{R}^+$:
$$\nabla_T \text{ECE}(T) = \mathbf{0}$$
The objective surface consists of flat step-plateaus separated by discontinuities. Gradient-based optimizers (SGD, Adam, LBFGS) receive zero gradients and terminate immediately at the initialization point $T=1.0$ without performing optimization.

#### 3. Non-Convexity, Discontinuity, and Degenerate Minima
- **Non-Convexity**: Even if smoothed with mollifiers, ECE is non-convex with numerous poor local minima.
- **Pathology & Lack of Strict Propriety**: ECE is not a strictly proper scoring rule (Gneiting & Raftery, 2007). A scoring rule $S(p, y)$ is strictly proper if and only if the true distribution $p^*$ uniquely minimizes the expected score. ECE permits degenerate solutions: a model predicting uniform confidence $r_i = \bar{c}$ across a bin achieves $\text{ECE} = 0$ while providing zero discriminative capability.
- **Strict Propriety and Convexity of NLL**:
  Negative Log-Likelihood is a strictly proper scoring rule. Furthermore, let $\beta = 1/T$. The NLL function:
  $$\text{NLL}(\beta) = -\frac{1}{N} \sum_{i=1}^N \left[ \beta z_{i, y_i} - \log \sum_e \exp(\beta z_{i, e}) \right]$$
  has Hessian:
  $$\frac{\partial^2 \text{NLL}}{\partial \beta^2} = \frac{1}{N} \sum_{i=1}^N \text{Var}_{p(e \mid \beta)}[z_{i, \cdot}] \ge 0$$
  Since the variance of logits is strictly positive for non-constant logits, NLL is **strictly convex** in $\beta = 1/T$. It possesses a unique global minimum, smooth continuous gradients everywhere, and cannot degenerate.

---

## 4. Targeted Gating Evaluation & Metrics

### 4.1 Operational Context in Asynchronous MoE Routing

In an asynchronous MoE execution engine, speculative routing heads drive real-time hardware dispatch decisions. Two decision thresholds are vital:

1. **Abort Decision Boundary ($\theta_{\text{abort}} = 0.05$)**:
   - When speculative routing determines that an expert's probability is below 0.05 ($\hat{p} < 0.05$), it issues an **abort signal**. The worker thread drops speculative prefetching of that expert's parameter matrices from host RAM to SRAM/HBM.
   - If probabilities near 0.05 are miscalibrated:
     - **False Abort**: Predicted $p = 0.04$, but true probability is $0.15$. The engine aborts, causing a pipeline stall and synchronous reload when the expert is requested.
     - **Wasted Prefetch**: Predicted $p = 0.06$, but true probability is $0.005$. The engine prefetches an unnecessary expert, saturating PCIe/memory bus bandwidth.
2. **Mass Cutoff Decision Boundary ($\theta_{\text{mass}} = 0.85$)**:
   - In speculative MoE dispatch, candidate experts are sorted by predicted probability and accumulated until total probability mass reaches 85%:
     $$\sum_{e \in S} \hat{p}(e) \ge 0.85$$
   - This minimal set $S$ is dispatched speculatively.
   - If probabilities are overconfident, the sum hits 0.85 with only 1 or 2 experts, missing genuine active experts (poor recall).
   - If underconfident, $S$ balloons to 8–10 experts, causing compute contention.

---

### 4.2 Targeted ECE Formulations

We formulate three rigorous metrics to evaluate calibration at these decision boundaries:

#### Metric 1: Window-Based Targeted ECE ($\text{T-ECE}_{\text{window}}(\theta, \delta)$)
Evaluates calibration within a localized confidence window centered at threshold $\theta$:
$$\mathcal{W}_\theta = [\theta - \delta, \theta + \delta]$$
- For $\theta = 0.05$: $\delta = 0.025 \implies \mathcal{W}_{0.05} = [0.025, 0.075]$.
- For $\theta = 0.85$: $\delta = 0.05 \implies \mathcal{W}_{0.85} = [0.80, 0.90]$.

Let $\mathcal{I}_\theta = \{ (i, e) : \hat{p}_i(e) \in \mathcal{W}_\theta \}$, with sample count $N_\theta = |\mathcal{I}_\theta|$.
$$\text{T-ECE}_{\text{window}}(\theta) = \left| \frac{1}{N_\theta} \sum_{(i, e) \in \mathcal{I}_\theta} y_i(e) - \frac{1}{N_\theta} \sum_{(i, e) \in \mathcal{I}_\theta} \hat{p}_i(e) \right|$$
where $y_i(e) = \mathbb{I}(e \in \mathcal{T}_i)$ indicates whether expert $e$ was actively executed by the native router.

#### Metric 2: Kernel-Smoothed Localized Calibration Error ($\text{T-ECE}_{\text{kernel}}(\theta, h)$)
To eliminate sensitivity to window width $\delta$, we evaluate the continuous calibration curve at exact point $\theta$ using a Gaussian kernel:
$$\widehat{\text{acc}}(\theta) = \frac{\sum_{i, e} y_i(e) \cdot \exp\left( - \frac{(\hat{p}_i(e) - \theta)^2}{2 h^2} \right)}{\sum_{i, e} \exp\left( - \frac{(\hat{p}_i(e) - \theta)^2}{2 h^2} \right)}$$
$$\text{T-ECE}_{\text{kernel}}(\theta) = |\widehat{\text{acc}}(\theta) - \theta|$$
Recommended bandwidth: $h = 0.02$ for $\theta = 0.05$, and $h = 0.04$ for $\theta = 0.85$.

#### Metric 3: Cumulative Mass Cutoff Calibration ($\text{T-ECE}_{\text{mass}}(0.85)$)
For each sample $i$, sort predicted expert probabilities: $\hat{p}_{(1)} \ge \hat{p}_{(2)} \ge \dots \ge \hat{p}_{(60)}$.  
Find the smallest expert subset $S_i = \{ (1), \dots, (k_i^*) \}$ such that:
$$\widehat{\text{Mass}}_i = \sum_{j=1}^{k_i^*} \hat{p}_{(j)} \ge 0.85$$

We evaluate:
1. **Mass Calibration Gap**:
   $$\text{Gap}_{\text{mass}} = \left| \frac{1}{N} \sum_{i=1}^N \left( \sum_{e \in S_i} q_i(e) \right) - \frac{1}{N} \sum_{i=1}^N \widehat{\text{Mass}}_i \right|$$
   where $q_i(e)$ is the ground-truth native router probability.
2. **Top-4 Native Expert Recall**:
   $$\text{Recall}_{0.85} = \frac{1}{N} \sum_{i=1}^N \frac{|S_i \cap \mathcal{T}_i|}{4}$$
   This directly measures the percentage of true executed experts captured when truncating at 85% cumulative mass.

---

### 4.3 Matrix Reporting Breakdown

Evaluation results on the held-out calibration split must be printed in the final evaluation report broken down across all 6 buckets:

| Layer Bucket | Horizon | Fitted $T(s)$ | Uncal NLL | Calib NLL | Standard ECE | T-ECE @ 0.05 (Abort) | T-ECE @ 0.85 (Marginal) | Recall @ 0.85 Mass |
|---|---|---|---|---|---|---|---|---|
| Early (5–10) | $T+1$ | $T_{1,1}$ | $\dots$ | $\dots$ | $\dots$ | $\dots$ | $\dots$ | $\dots$ |
| Early (5–10) | $T+2$ | $T_{1,2}$ | $\dots$ | $\dots$ | $\dots$ | $\dots$ | $\dots$ | $\dots$ |
| Early (5–10) | $T+3$ | $T_{1,3}$ | $\dots$ | $\dots$ | $\dots$ | $\dots$ | $\dots$ | $\dots$ |
| Late (11–24) | $T+1$ | $T_{2,1}$ | $\dots$ | $\dots$ | $\dots$ | $\dots$ | $\dots$ | $\dots$ |
| Late (11–24) | $T+2$ | $T_{2,2}$ | $\dots$ | $\dots$ | $\dots$ | $\dots$ | $\dots$ | $\dots$ |
| Late (11–24) | $T+3$ | $T_{2,3}$ | $\dots$ | $\dots$ | $\dots$ | $\dots$ | $\dots$ | $\dots$ |
| **Aggregate** | **All** | — | **$\dots$** | **$\dots$** | **$\dots$** | **$\dots$** | **$\dots$** | **$\dots$** |

---

## 5. Memory Leak Monitoring Methodology for Automated Testing

### 5.1 Root Causes of Memory Leaks in PyTorch Calibration Pipelines
1. **Computational Graph Retention**: Appending tensors to metric lists without `.item()` or `.detach()`, retaining the entire backward autograd graph across thousands of tokens.
2. **Activation Offloading Leaks**: Forward hooks caching activations during 100k token streaming without detaching and moving to CPU or freeing references.
3. **MPS Allocator Cache Bloat**: PyTorch's Metal Performance Shaders (MPS) allocator retains pooled memory blocks unless `torch.mps.empty_cache()` is called.
4. **Circular References in Generators**: Python generators holding references to large tensors or DataLoader workers.

### 5.2 Multi-Tier Verification Test Harness

We specify a 3-tier automated test harness to catch memory leaks during test execution:

```python
import gc
import psutil
import torch
import tracemalloc
import pytest

def test_pipeline_zero_memory_leak():
    process = psutil.Process()
    
    # 1. Warmup Phase (Initializes PyTorch CUDA/MPS context, allocator buffers, and kernels)
    for _ in range(3):
        run_speculative_calibration_step()
        
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        
    # Baseline snapshots
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()
    rss_before = process.memory_info().rss
    tensor_count_before = sum(1 for obj in gc.get_objects() if isinstance(obj, torch.Tensor))
    
    # 2. Execution Loop
    NUM_ITERATIONS = 20
    for _ in range(NUM_ITERATIONS):
        run_speculative_calibration_step()
        
    # 3. Post-execution Cleanup & Measurement
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        
    snapshot_after = tracemalloc.take_snapshot()
    rss_after = process.memory_info().rss
    tensor_count_after = sum(1 for obj in gc.get_objects() if isinstance(obj, torch.Tensor))
    
    # 4. Strict Assertions
    # Assertion A: Active tensor count delta MUST be zero
    tensor_delta = tensor_count_after - tensor_count_before
    assert tensor_delta == 0, f"Leaked {tensor_delta} PyTorch tensors across iterations!"
    
    # Assertion B: RSS Growth must be under 2 MB threshold (accounting for Python interpreter jitter)
    rss_growth_mb = (rss_after - rss_before) / (1024 * 1024)
    assert rss_growth_mb < 2.0, f"Detected memory leak: RSS grew by {rss_growth_mb:.2f} MB over {NUM_ITERATIONS} iterations!"
    
    # Assertion C: Tracemalloc heap difference analysis
    top_stats = snapshot_after.compare_to(snapshot_before, "lineno")
    leaked_heap_bytes = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
    assert leaked_heap_bytes < 1024 * 1024, f"Tracemalloc detected {leaked_heap_bytes / 1024:.1f} KB uncollected heap memory."
    
    tracemalloc.stop()
```

---

## 6. Summary Specification Table

| Component | Mathematical / Technical Specification | Default Hyperparameter |
|---|---|---|
| **Base Model** | `Qwen/Qwen1.5-MoE-A2.7B` | 24 layers, $d_{\text{model}} = 2048$, 60 experts, top-4 |
| **Tap Layer $N$** | Layer 4 (0-based index 3) | Immediate post-attention/MLP hidden state |
| **Speculative Head** | 3 Linear projection heads (`nn.ModuleList`) | Input: 2048, Output: $20 \times 60 = 1200$ logits |
| **Target Distribution** | Full native router softmax distribution $q_{\text{soft}} \in \Delta^{60}$ | Soft Cross-Entropy (Teacher-Student Distillation) |
| **Calibration Loss** | Maximum Mean Calibration Error (MMCE, Kumar et al.) | Universal Gaussian RBF kernel, $\sigma = 0.2$ |
| **Loss Objective** | $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \cdot \widehat{\text{MMCE}}_w$ | $\lambda = 2.0$, autograd detached on $c_i$ |
| **Temperature Grid** | $2 \times 3 = 6$ buckets: $\{\text{early 5-10}, \text{late 11-24}\} \times \{T+1, T+2, T+3\}$ | Scalar $T_s > 0$ per bucket, initialized to $1.0$ |
| **Scaling Objective** | $\mathcal{L}(T_s) = \text{NLL}(T_s) + \alpha \cdot (T_s - 1.0)^2$ | Regularization strength $\alpha = 0.05$ |
| **Optimizer** | `torch.optim.LBFGS` | `max_iter=50`, `line_search_fn='strong_wolfe'` |
| **Targeted Gating 0.05** | Window ECE $\mathcal{W}_{0.05} = [0.025, 0.075]$ & Kernel ECE ($h=0.02$) | Evaluates speculative abort accuracy |
| **Targeted Gating 0.85** | Cumulative Mass Cutoff ($\sum \hat{p} \ge 0.85$) & Recall of top-4 | Evaluates speculative candidate set coverage |
| **Memory Monitoring** | `tracemalloc` + `psutil` RSS + `gc.get_objects()` tensor tracking | Leak threshold: 0 tensor delta, $< 2 \text{ MB}$ RSS growth |

---
