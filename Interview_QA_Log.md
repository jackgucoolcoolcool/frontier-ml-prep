# Interview Prep — Q&A Log

A running log of the technical questions I've asked and the answers, captured for review.
Newest entries are added at the bottom. Deeper write-ups are cross-linked where they exist.

**Index**
1. [Multivariate sigmoid function](#q1--write-down-the-multivariate-sigmoid-function)
2. [Why do sigmoid and tanh saturate?](#q2--why-do-sigmoid-and-tanh-saturate)
3. [Compare ReLU and GELU](#q3--compare-relu-and-gelu)
4. [What is GELU / what is Φ?](#q4--what-is-gelu-again-what-is-%CF%86)
5. [Why is GELU z·Φ(z)? (input × CDF motivation)](#q5--why-is-gelu-zφz--the-input--cdf-motivation)
6. [Why does the stable sigmoid need np.where?](#q6--why-does-the-stable-sigmoid-need-npwhere)
7. [Bounded (tanh/sigmoid) vs unbounded (ReLU) activations](#q7--bounded-tanhsigmoid-vs-unbounded-relu-activations)
8. [Derive ∂L/∂z₂ = ŷ − y (softmax + cross-entropy)](#q8--derive-lz₂--ŷ--y-softmax--cross-entropy)
9. [MLE vs MAP](#q9--mle-vs-map)
10. [What is the chain rule? (and why a sum)](#q10--what-is-the-chain-rule-and-why-a-sum)
11. [Concrete example: error = prediction − truth](#q11--concrete-example-error--prediction--truth)
12. [How to write the outer product (δ aᵀ)](#q12--how-to-write-the-outer-product-δ-aᵀ)
13. [The product-of-Jacobians lens (vanish/explode)](#q13--the-product-of-jacobians-lens-vanishexplode)
14. [Which part is the Jacobian?](#q14--which-part-is-the-jacobian)
15. [Why constant variance, and where the 2 in He init comes from](#q15--why-constant-variance-and-where-the-2-in-he-init-comes-from)
16. [What is fan_in?](#q16--what-is-fan_in)
17. [Why scale residual branches by 1/√N_layers](#q17--why-scale-residual-branches-by-1√n_layers)
18. [What are LN/RMSNorm, and why are BN stats noisy?](#q18--what-are-lnrmsnorm-and-why-are-bn-stats-noisy)
19. [Pre-norm vs post-norm (residuals, equations, why pre-norm is stable)](#q19--pre-norm-vs-post-norm-residuals-equations-why-pre-norm-is-stable)
20. [Why does RMSNorm match LayerNorm? (theory or empirical)](#q20--why-does-rmsnorm-match-layernorm-theory-or-empirical)
21. [Optimizers in detail + how AdamW runs in production](#q21--optimizers-in-detail--how-adamw-runs-in-production)

---

## Q1 — Write down the multivariate sigmoid function

**Short answer:** The multivariate generalization of the sigmoid is the **softmax**.

- Scalar sigmoid: $\sigma(z) = \dfrac{1}{1+e^{-z}} \in (0,1)$ — binary.
- Softmax (multiclass), for logits $\mathbf{z}=(z_1,\dots,z_K)$:
$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$
  Components are in $(0,1)$ and **sum to 1**; it's **shift-invariant** (subtract $\max_j z_j$ for numerical stability).
- For $K=2$ it **reduces to the sigmoid**: $\text{softmax}(\mathbf z)_1 = \sigma(z_1-z_2)$.
- Distinct from the **elementwise sigmoid** $\sigma(z_i)$ (multi-label: independent yes/no per class, does *not* sum to 1).

📄 Full write-up: `Notes_Activation_Functions.md` / `.html`, §1.

---

## Q2 — Why do sigmoid and tanh saturate?

**Short answer:** Both are **bounded squashing functions** that compress all of $\mathbb{R}$ into a finite interval. Mapping an infinite domain into a finite range *must* flatten at the extremes, and a flat region means the derivative → 0. That near-zero gradient is "saturation."

- Sigmoid: $\sigma'(z)=\sigma(z)(1-\sigma(z))$; max $0.25$ at $z=0$, → 0 as $|z|\to\infty$.
- Tanh: $\tanh'(z)=1-\tanh^2(z)$; max $1$ at $z=0$, → 0 as $\tanh\to\pm1$. (Better than sigmoid — bigger max gradient, zero-centered — but same flat-tail flaw.)
- **Why it matters:** backprop multiplies these small per-layer factors; saturated layers contribute ≈ 0, so gradients **vanish** in deep nets → early layers stop learning. Motivates ReLU (derivative 1 on active branch) and normalization (keeps pre-activations near 0, the high-gradient region).

📄 Full write-up: `Notes_Activation_Functions.md` / `.html`, §2.

---

## Q3 — Compare ReLU and GELU

**Short answer:** ReLU $=\max(0,z)$ is a hard, cheap gate (gradient 1 or 0) but suffers **dead neurons**. GELU $=z\,\Phi(z)$ ($\Phi$ = Gaussian CDF) is a smooth, probabilistic gate that passes a small nonzero signal for negative inputs, avoiding dead neurons and giving smoother optimization at higher compute cost. Both avoid positive-side saturation.

| | ReLU | GELU |
|---|---|---|
| Formula | $\max(0,z)$ | $z\,\Phi(z)$ |
| Smoothness | Hard kink at 0 | Smooth everywhere |
| Negatives | Hard 0 | Small negative dip → 0 |
| Dead neurons | Yes | Largely avoided |
| Compute | Cheapest | Pricier (CDF/tanh/erf) |
| Used in | CNNs, MLPs | Transformers |

**Why transformers use GELU:** smoothness + nonzero negative gradient give a small but reliable quality gain that compounds at scale (worth the extra FLOPs). **Dying-ReLU symptom:** a growing fraction of permanently-zero activations → suspect LR/init, or switch to GELU/LeakyReLU.

📄 Full write-up: `Notes_Activation_Functions.md` / `.html`, §3.

---

## Q4 — What is GELU again, what is Φ?

**Short answer:** GELU is the activation $\text{GELU}(z) = z\cdot\Phi(z)$. The symbol is **Φ ("Phi"), not θ (theta)** — it's the **standard Gaussian CDF**.

- $\Phi(z) = \Pr[Z \le z]$ for $Z\sim\mathcal N(0,1)$ — the probability a standard normal is $\le z$. Ranges 0→1, S-shaped. $\Phi(0)=0.5$. Equivalently $\Phi(z)=\tfrac12\big(1+\operatorname{erf}(z/\sqrt2)\big)$.
- **Soft gate intuition:** GELU multiplies $z$ by the probability it should be "kept":
  - large $+z$ → $\Phi\approx1$ → output ≈ $z$ (on)
  - large $-z$ → $\Phi\approx0$ → output ≈ 0 (off)
  - near 0 → $\Phi\approx0.5$ → output ≈ $z/2$ (the smooth part)
- vs **ReLU** = a hard step (keep if $z>0$ else 0). GELU's smoothness avoids the kink / dead-neuron problem.

**Symbol guide:** Φ (Phi) = Gaussian **CDF** (the GELU gate); φ (lowercase phi) = Gaussian **PDF** (bell curve, $=\Phi'$); θ (theta) = unrelated, usually model **parameters**.

📄 Related: `Notes_Activation_Functions.md` §3.

---

## Q5 — Why is GELU z·Φ(z)? (the input × CDF motivation)

**Short answer:** Every activation is `input × gate`: $z\cdot g(z)$ with $g\in[0,1]$. ReLU's gate is the **hard Heaviside step** $\mathbf 1[z>0]$; GELU's gate is the **Gaussian CDF** $\Phi(z)$ — the *smooth, probabilistic* version of that step. A CDF is the natural soft step (monotone 0→1, differentiable).

**Why that CDF specifically:**
- **Gaussian-smoothed step:** convolving the hard step with a Gaussian gives exactly $\Phi$ → GELU = ReLU gate averaged over Gaussian input noise.
- **Stochastic-regularizer derivation (the real motivation):** mask the neuron with $m\sim\text{Bernoulli}(\Phi(z))$ (keep-prob depends on input; bigger inputs kept more often), then take the expectation: $\mathbb E[z\,m]=z\,\Phi(z)=\text{GELU}(z)$. So GELU blends dropout-style stochastic gating with the data — the CDF appears because it *is* the probability of keeping the unit.

**Consequences:** self-gating (input gates itself) — same family as Swish/SiLU $z\,\sigma(\beta z)$, and $\text{GELU}(z)\approx z\,\sigma(1.702 z)$. Derivative $\Phi(z)+z\varphi(z)$ → slightly negative for small negative z → **non-monotonic**, a touch more expressive.

**One-liner:** "GELU is input × Gaussian-CDF gate — the smooth/probabilistic ReLU. It's the expectation of randomly keeping a neuron with probability Φ(z), unifying dropout-style regularization with input-dependent gating."

📄 Related: `Notes_Activation_Functions.md` §3, [[Q3]], [[Q4]].

---

## Q6 — Why does the stable sigmoid need np.where?

**Short answer:** It's not math — the two forms $\frac{1}{1+e^{-z}}$ and $\frac{e^z}{1+e^z}$ are algebraically identical. It's **floating-point overflow**: you must never compute `exp(large positive)`.

- **Form A `1/(1+exp(-z))`:** clean for z≥0; for z≪0, `exp(-z)=inf` (overflow).
- **Form B `exp(z)/(1+exp(z))`:** clean for z<0; for z≫0, `exp(z)=inf` → `inf/inf = NaN`.
- Each form blows up on one side. `np.where(z>=0, A, B)` picks the form whose exponent is ≤0 (so `exp` stays bounded in (0,1]). Same function, different rearrangement.

**Caveats:**
- `np.where` evaluates **both** branches (no short-circuit), so the discarded branch can still emit an overflow *warning* — the selected value is still correct.
- A genuine single stable equation: $\sigma(z)=\tfrac12(1+\tanh(z/2))$ — `0.5*(1+np.tanh(0.5*z))`.
- In practice use `scipy.special.expit` / `torch.sigmoid` / `jax.nn.sigmoid` (stable internally). The from-scratch branch is to *show* you understand the overflow issue.

📄 Related: `Notes_Activation_Functions.md` §4 (code).

---

## Q7 — Bounded (tanh/sigmoid) vs unbounded (ReLU) activations

**Short answer:** Bounded activations self-limit their output range; unbounded ones don't. Practical consequences:

| | Bounded (sigmoid, tanh) | Unbounded (ReLU, GELU) |
|---|---|---|
| Range | (0,1) / (−1,1) | (0,∞) |
| Tails | saturate → grad→0 | no positive saturation → grad ~1 |
| Deep nets | vanishing gradients | trains deep nets |
| Scale | self-limiting (stable) | can grow → needs normalization |
| Sparsity | dense | ReLU → exact zeros (sparse) |
| Cost | exp (pricier) | max (cheap) |
| Failure | saturation | dead neurons |

**Key practical points:**
- **Gradient flow:** bounded saturate → vanishing gradients in depth; unbounded keep gradients alive → why deep nets use ReLU-family.
- **Scale:** bounded = built-in stability; unbounded = rely on LayerNorm/BatchNorm + residual scaling to control magnitude.
- **Info at extremes:** bounded discard magnitude (sigmoid(5)≈sigmoid(50)≈1); unbounded preserve "how big."
- **Sparsity/compute:** ReLU = exact zeros + cheap max; bounded = dense + exp.
- **Where used:** unbounded → hidden layers (representation, gradient flow). Bounded → where the range is the point: **gates** (LSTM/GRU, GLU — sigmoid = "how much to pass"), **probabilities** (sigmoid/softmax output), bounded regression (tanh).
- **Zero-centering:** tanh zero-centered (cleaner updates) vs sigmoid all-positive (biased/zig-zag).

**One-liner:** "Bounded self-limit → ideal as gates/probabilities but saturate and vanish gradients in depth; unbounded preserve magnitude and keep gradients alive (deep nets) but need normalization and risk dead neurons. Bounded at gates/outputs, unbounded in hidden layers."

📄 Related: `Notes_Activation_Functions.md`, [[Q2]] (saturation), [[Q3]] (ReLU vs GELU).

---

## Q8 — Derive ∂L/∂z₂ = ŷ − y (softmax + cross-entropy)

**Setup:** logits $z_2\in\mathbb R^K$, one-hot label $y$ (true class $c$).
$$\hat y_i = \frac{e^{z_{2,i}}}{\sum_j e^{z_{2,j}}}, \qquad L = -\sum_k y_k\log\hat y_k = -\log\hat y_c$$

**Elegant derivation (substitute softmax first):**
$$L = -\log\frac{e^{z_{2,c}}}{\sum_j e^{z_{2,j}}} = -z_{2,c} + \log\sum_j e^{z_{2,j}}$$
Differentiate w.r.t. $z_{2,i}$: first term gives $-y_i$ (=−1 at $i=c$); second term (log-sum-exp) gives $\frac{e^{z_{2,i}}}{\sum_j e^{z_{2,j}}}=\hat y_i$. Sum: $\boxed{\partial L/\partial z_{2,i}=\hat y_i-y_i}$.

**Full chain rule (alternative):** $\partial L/\partial z_i = \sum_k(\partial L/\partial\hat y_k)(\partial\hat y_k/\partial z_i)$ with $\partial L/\partial\hat y_k=-y_k/\hat y_k$ and softmax Jacobian $\partial\hat y_k/\partial z_i=\hat y_k(\delta_{ki}-\hat y_i)$. The $\hat y_k$ cancels → $-\sum_k y_k(\delta_{ki}-\hat y_i) = -y_i + \hat y_i\sum_k y_k = \hat y_i - y_i$ (using $\sum_k y_k=1$).

**Why it matters:** gradient = prediction − truth (same form as logistic regression); clean + stable → frameworks fuse softmax+CE into one op.

📄 Related: Day 1 §2.2; [[Q1]] (softmax).

---

## Q9 — MLE vs MAP

**Short answer:** Both pick a single best $\theta$. **MLE** maximizes the likelihood; **MAP** maximizes the posterior = likelihood × prior. So **MAP = MLE + a prior term**.

$$\theta_{MLE} = \arg\max_\theta \sum_i \log p(x_i\mid\theta) \qquad \theta_{MAP} = \arg\max_\theta \Big[\sum_i \log p(x_i\mid\theta) + \log p(\theta)\Big]$$

**Key insight — a prior is a regularizer:**
- Gaussian prior $\theta\sim\mathcal N(0,\sigma^2)$ → $\log p(\theta)=-\frac{1}{2\sigma^2}\|\theta\|^2$ → **L2 / weight decay**. So L2 reg = MAP with a Gaussian prior.
- Laplace prior → **L1** (sparsity).
- In DL: CE loss alone = MLE; CE + weight decay = MAP.

**Relationships:**
- MLE = MAP with a **flat/uniform prior** (prior drops out).
- As data grows, **MAP → MLE** (likelihood sum dominates the single prior term; prior washes out). Prior matters most with little data.
- MLE overfits small data; MAP regularizes.
- Both are **point estimates** (mode), not the full posterior (vs full Bayesian, which integrates over $p(\theta\mid D)$).
- Gotcha: MLE is reparameterization-invariant; MAP is **not** (prior density transforms).

**One-liner:** "MAP = MLE + log-prior; the prior is a regularizer (Gaussian→L2, Laplace→L1). MLE is MAP with a flat prior, and MAP→MLE as data grows."

📄 Related: Day 1 §5.2 (weight decay), [[Q8]] (CE=MLE).

---

## Q10 — What is the chain rule? (and why a sum)

**Short answer:** The chain rule differentiates a **composition** by multiplying local derivatives along the path.

- **Single-variable:** if $L=f(u)$ and $u=g(z)$, then $\frac{dL}{dz}=\frac{dL}{du}\frac{du}{dz}$ (sensitivities multiply).
- **Multivariable:** if $z_i$ reaches $L$ through several intermediates $\hat y_1,\dots,\hat y_K$, multiply along each path **and sum over all paths**: $\frac{\partial L}{\partial z_i}=\sum_k \frac{\partial L}{\partial \hat y_k}\frac{\partial \hat y_k}{\partial z_i}$.
- **Why the sum:** softmax is coupled — nudging one logit $z_i$ changes *every* $\hat y_k$ — so there are $K$ paths from $z_i$ to $L$, and independent contributions to the same quantity add.
- **Picture:** graph $z_i\to\{\hat y_1,\dots,\hat y_K\}\to L$; trace every path, multiply local derivatives along it, sum the paths.
- **Backprop = this rule applied across the whole computation graph** (multiply along edges, sum over paths, reuse shared sub-results).

📄 Related: Day 1 §2.2 (Method 2), §2.3 (backprop); [[Q8]].

---

## Q11 — Concrete example: error = prediction − truth

**Multiclass (softmax + CE):** cat/dog/bird classifier, logits `z=[2.0,1.0,0.1]`, true = cat → `y=[1,0,0]`. Softmax (`S=e²+e¹+e^0.1=11.21`) → `ŷ=[0.659,0.242,0.099]`.
$$\frac{\partial L}{\partial z} = \hat y - y = [-0.341,\ +0.242,\ +0.099]$$
- True class → **negative** gradient → its logit goes **up** (step `z←z−η∇`).
- Wrong classes → **positive** gradients → their logits go **down**.
- Magnitude = how much probability sat in the wrong place; entries **sum to 0** (`Σŷ=Σy=1`).

**Same form across models** (each loss paired with its matching output activation):

| Model | Output | Loss | Output gradient |
|---|---|---|---|
| Linear reg | `wx+b` | MSE | `ŷ−y` (residual; e.g. 250−300 = −50) |
| Logistic reg | `σ(z)` | BCE | `ŷ−y` (e.g. y=1, z=0.5⇒ŷ=0.62 → −0.38) |
| Softmax | `softmax(z)` | CE | `ŷ−y` (vector) |

**Why:** these are GLMs with canonical loss↔activation pairing (MSE↔identity, BCE↔sigmoid, CE↔softmax) — that pairing is what collapses the gradient to prediction − target. Mismatch it (MSE on sigmoid) and you lose it.

📄 Related: Day 1 §2.2; [[Q8]] (ŷ−y derivation).

---

## Q12 — How to write the outer product (δ aᵀ)

**Definition:** outer product of column vectors `u∈ℝᵐ`, `v∈ℝⁿ` is the **m×n** matrix `u vᵀ` with `(u vᵀ)_{ij}=u_i v_j` (every element of u × every element of v). Vs the **inner/dot** product `uᵀv` = a scalar (needs equal lengths).

**Example:** `[1,2,3]ᵀ·[4,5] = [[4,5],[8,10],[12,15]]` (3×1 times 1×2 → 3×2).

**Why it's the weight gradient:** `δ₂∈ℝ^{d_out}` (output error), `a₁∈ℝ^{d_in}` (layer input) → `δ₂ a₁ᵀ` is `d_out×d_in` = shape of `W₂`; entry `(∂L/∂W₂)_{ij}=δ₂_i·a₁_j` = (error at output i)×(input from j).

**Code:**
```python
np.outer(d2, a1)              # or  d2[:, None] * a1[None, :]   (broadcasting)
torch.outer(d2, a1)          # or  d2.unsqueeze(1) @ a1.unsqueeze(0)
jnp.outer(d2, a1)            # einsum: np.einsum('i,j->ij', d2, a1)
```

**Mini-batch:** sum outer products over examples = matmul `Δ Aᵀ` (Δ: d_out×N, A: d_in×N). The batch dim contracts → backprop is matmuls, not loops.

📄 Related: Day 1 §2.3.

---

## Q13 — The product-of-Jacobians lens (vanish/explode)

**Idea:** going backward through `L` layers, each layer multiplies the gradient by a factor `r ≈ ‖W‖·|φ'|`, so the gradient at layer 1 is `≈ r^L`. A **power** → small deviations from 1 compound:

| factor `r` | `r¹⁰` | regime |
|---|---|---|
| 0.5 | 0.001 | vanish fast |
| 0.8 | 0.107 | vanish slow |
| 1.0 | 1.0 | **stable** |
| 1.1 | 2.6 | explode |
| 1.5 | 57.7 | explode |

**Every stabilizing trick keeps `r ≈ 1`:**
- **Init** (Xavier/He): `‖W‖≈1` → starting factor ~1.
- **Normalization:** activations unit-scale → `φ'` stays responsive (non-saturated) → factor ~1 during training.
- **Residuals** `y=x+F(x)`: Jacobian `I+∂F/∂x` → a literal ×1 path → product can't collapse to 0.
- **Warmup/clipping:** stop `r` transiently spiking >1 early.

**Concretely:** sigmoid `φ'≤0.25` → factor ≤0.25 → vanishes (why deep sigmoid nets won't train); ReLU active `φ'=1` + `‖W‖≈1` → factor ~1 → stable.

📄 Related: Day 1 §2.4 (with bar diagram); §3 (init), §4 (norm/residuals), §5 (warmup/clipping).

---

## Q14 — Which part is the Jacobian?

**Definition:** a Jacobian is the matrix of all first-order partials of a vector function `f:ℝⁿ→ℝᵐ` → `m×n` matrix `J_{ij}=∂f_i/∂x_j`.

**Each layer's local derivative is a Jacobian:**

| Layer | Function | Jacobian |
|---|---|---|
| Linear | `z=Wx+b` | `∂z/∂x = W` → backward "× Wᵀ" |
| Activation | `a=φ(z)` | `∂a/∂z = diag(φ'(z))` → backward "⊙ φ'" |
| Softmax | `ŷ=softmax(z)` | `diag(ŷ) − ŷŷᵀ` |

So the backprop rules ARE multiplications by (transposed) layer Jacobians:
$$\delta_1 = \underbrace{\operatorname{diag}(\phi'(z_1))}_{\text{activation Jac}}\,\underbrace{W_2^\top}_{\text{linear Jac}}\,(\hat y - y)$$

The chain rule **stacks** them → the net's input→output derivative is the **product of per-layer Jacobians**; `r ≈ ‖W‖·|φ'|` is the magnitude of one. Backprop computes a **vector–Jacobian product** `vᵀJ` (e.g. `δᵀW`, or elementwise `×φ'`), never the full matrix.

📄 Related: Day 1 §2.3–§2.4; [[Q13]] (product-of-Jacobians), [[Q10]] (chain rule).

---

## Q15 — Why constant variance, and where the 2 in He init comes from

**Why constant variance:** `z=Wx` sums over `fan_in` inputs; if each layer scales the signal variance by a factor ≠1, over L layers it goes like `factor^L` (same compounding as §2.4). <1 → activations vanish to 0 over depth; >1 → explode. Same recursion for backward gradients (layer Jacobian = W). So preserve variance → keep forward signal & backward gradient at O(1) scale through depth.

**Recursion (gives 1/fan_in = Xavier):** `z_i = Σ_j W_ij x_j`, W & x i.i.d. mean-0, independent → variances add:
$$\text{Var}(z) = \text{fan\_in}\cdot\text{Var}(W)\cdot\text{Var}(x)$$
Preserve → `Var(W) = 1/fan_in` → `std = 1/√fan_in`. (Xavier; assumes linear/tanh-near-0.)

**Where the 2 comes from (He, ReLU):** ReLU `max(0,z)` zeros the negative half, so the second moment passed forward is **halved**:
$$E[a^2]=E[\max(0,z)^2]=\tfrac12 E[z^2]=\tfrac12\text{Var}(z)$$
Redo: `Var(z^l) = fan_in·Var(W)·½·Var(z^{l-1})`; preserve → `Var(W)=2/fan_in` → `std=√(2/fan_in)`.

**The 2 exactly cancels ReLU's ½.** Use Xavier's 1/fan_in on a ReLU net and the signal shrinks ~½ each layer → vanishes over depth. He doubles the weight scale to compensate for the half ReLU kills.

📄 Related: Day 1 §3.1; [[Q13]] (compounding over depth).

---

## Q16 — What is fan_in?

**Definition:** `fan_in` = number of inputs feeding into a neuron (input dimension; how many weighted connections get summed for one output). `fan_out` = number of outgoing connections (output dimension).

- **Linear** 512→256 (`W` is 256×512): `fan_in=512`, `fan_out=256`.
- **Conv:** `fan_in = in_channels × k_h × k_w`; `fan_out = out_channels × k_h × k_w`.

**Why it appears in init:** a neuron sums `fan_in` products, so `Var(z) = fan_in · Var(W) · Var(x)` — output variance grows linearly with fan_in. Summing 1000 inputs has ~100× the variance of summing 10. To keep the scale constant, weight variance ∝ `1/fan_in` (Xavier) or `2/fan_in` (He). More inputs → bigger sum → smaller weights.

📄 Related: Day 1 §3.1; [[Q15]] (variance recursion).

---

## Q17 — Why scale residual branches by 1/√N_layers

**Setup:** a residual block adds into a running stream `x_l = x_{l-1} + F_l(x_{l-1})`, so the stream is a *sum* over all N blocks. Variances add → if each block contributes `≈σ²`:
$$\text{Var}(x_N) \approx \text{Var}(x_0) + N\sigma^2$$
The residual stream variance **grows linearly with depth** (std ~ √N) → late-layer activations dwarf early ones → instability, and each new block's relative contribution shrinks like 1/l.

**Fix:** scale each branch by α: `x_l = x_{l-1} + α·F_l(...)` → `Var(x_N) ≈ Var(x_0) + N·α²σ²`. Set `α = 1/√N`:
$$N\cdot\tfrac1N\sigma^2 = \sigma^2 \quad\text{(constant, depth-independent)}$$

**Why √N not 1/N:** *variances* add, not std. A sum of N independent terms has variance ∝ N → divide std by √N. Same `1/√(count)` pattern as `1/√fan_in` (fan_in normalizes the sum *within* a layer; 1/√N normalizes the sum *across* depth). Used by GPT-2 init; DeepNorm/Fixup/T-Fixup do depth-dependent residual scaling.

📄 Related: Day 1 §3.2; [[Q16]] (fan_in), [[Q15]] (variance preservation).

---

## Q18 — What are LN/RMSNorm, and why are BN stats noisy?

**LayerNorm** (per example, over the d features):
$$\mu=\tfrac1d\textstyle\sum_i x_i,\ \sigma^2=\tfrac1d\textstyle\sum_i(x_i-\mu)^2,\ \hat x_i=\tfrac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}},\ y_i=\gamma_i\hat x_i+\beta_i$$
**RMSNorm** (drop mean): `y_i = γ_i · x_i / RMS(x)`, `RMS(x)=√(mean_i(x_i²)+ε)`. Cheaper, ~equal quality.
**BatchNorm** (per feature, over batch): `μ_i=mean_b(x_{b,i})`, `σ²_i` over B examples.

**Why BN stats are noisy:** `μ_i, σ²_i` are **sample estimates** from only B examples. SE of the mean ∝ 1/√B (variance estimate noisier) → with small B they jump batch-to-batch → an example's normalization depends on its random batch-mates (coupling + noise). Worse for transformers: small effective batch (long seqs), ragged variable-length batches, B=1 decoding (stats undefined).

**LN/RMSNorm** use the d features of ONE example (d large, fixed) → batch-independent, deterministic, **identical train/inference**. (BN: batch stats at train, running EMA at test → different function each mode → bugs.)

📄 Related: Day 1 §4.1–4.2; [[Q15]] (variance), [[Q16]] (fan_in).

---

## Q19 — Pre-norm vs post-norm (residuals, equations, why pre-norm is stable)

**Residual in practice:** ResNet `y = x + F(x)` (F = conv-bn-relu-conv-bn); transformer wraps attention & FFN in residuals.

**Equations** (sublayer `S`):
$$\text{Post-norm: } x_{l+1}=\text{LN}(x_l + S(x_l)) \qquad \text{Pre-norm: } x_{l+1}=x_l + S(\text{LN}(x_l))$$
```python
def postnorm(x): x = ln1(x + attn(x)); x = ln2(x + ffn(x)); return x   # LN on trunk
def prenorm(x):  x = x + attn(ln1(x)); x = x + ffn(ln2(x)); return x   # LN on branch
# pre-norm: one final ln_final before the head
```

**Why pre-norm cleaner/stabler:** unroll pre-norm `x_L = x_0 + Σ_l S_l(LN(x_l))` — a pure additive identity path with **nothing on it** (LN on the branch) → clean ×1 gradient highway at every depth. Post-norm puts LN on the **trunk**, so the backward gradient crosses L LayerNorm Jacobians (≠ identity) → clean path broken → gradients grow/shrink with depth → needs warmup, unstable deep (Xiong et al. 2020).

**Trade-off:** pre-norm's clean trunk → residual stream variance grows with depth (§3.2, fixed by 1/√N + final LN). Post-norm keeps activations bounded but gradients fragile. LLMs pick pre-norm: trainability > bounded activations.

📄 Related: Day 1 §4.3–4.4; [[Q17]] (residual variance), [[Q18]] (LN).

---

## Q20 — Why does RMSNorm match LayerNorm? (theory or empirical)

**Both.** LN = re-center (−μ) + re-scale (/σ); RMSNorm = re-scale only (/RMS).

**Theory (Zhang & Sennrich 2019 hypothesis):** LN's benefit comes mainly from **re-scaling invariance, not re-centering**. Normalization helps optimization by controlling activation/gradient **magnitude** (bounds them, smooths the landscape, stabilizes per-layer effective LR) — that's the /RMS part. Mean subtraction is near-redundant because (a) the learned affine + next linear layer absorb a constant offset, (b) in high dimensions the mean is small relative to scale.

**Scale-invariance argument:** both LN & RMSNorm satisfy `f(αx)=f(x)` → self-stabilizing gradient (roughly orthogonal to weight direction, bounded). RMSNorm keeps this; only drops the less-important *shift*-invariance.

**Verdict:** well-motivated hypothesis + strong empirical confirmation (RMSNorm matches LN, ~7–64% faster; LLaMA/T5/Gemma use it). Not a theorem, but more than "just empirical."

📄 Related: Day 1 §4.2; [[Q18]] (LN/RMSNorm definitions).

---

## Q21 — Optimizers in detail + how AdamW runs in production

**Ladder (g = ∇L):**
- **Momentum:** `v = βv + g`, `θ −= η·v` (β≈0.9) — accumulate consistent dirs, damp oscillation.
- **RMSProp:** `s = ρs + (1−ρ)g²`, `θ −= η·g/(√s+ε)` — per-param adaptive LR.
- **Adam:** `m = β₁m+(1−β₁)g`, `v = β₂v+(1−β₂)g²`; bias-correct `m̂=m/(1−β₁ᵗ)`, `v̂=v/(1−β₂ᵗ)` (m,v start at 0); `θ −= η·m̂/(√v̂+ε)`.
- **AdamW:** `θ −= η·m̂/(√v̂+ε) − η·λ·θ` (decay on weights directly).

**Production transformers:** AdamW default; β₁=0.9, **β₂=0.95** (faster 2nd moment, robust to spikes), ε=1e-8. LR linear **warmup → cosine decay** to ~10% peak (~1–6e-4). **Weight decay ~0.1** decoupled, **only on matmul weights** (exclude bias / LN-RMSNorm gains / embeddings). Clip global norm 1.0. **Memory:** m+v per param = 2× params; mixed-precision ≈ **16 bytes/param** (2 bf16 w + 2 grad + 4 fp32 m + 4 fp32 v + 4 fp32 master) → 7B ≈ 112 GB states → **why FSDP/ZeRO shard optimizer state**. Alts: Adafactor (factorized 2nd moment), Lion (sign-based), Shampoo (preconditioned), Muon.

**One-liner:** momentum (1st moment) for speed + RMSProp (2nd moment) for adaptive LR + bias-correction = Adam; AdamW decays weights directly. 2× optimizer-state memory drives sharding.

📄 Related: Day 1 §5.1–5.2.

---

## Q22 — How is fuzzy dedup done? What is Jaccard? (MinHash + LSH)

**The goal — Jaccard similarity.** Represent each doc as a **set of n-gram shingles** (slide an n-token window, collect distinct shingles). Similarity = overlap fraction:

```
J(A,B) = |A ∩ B| / |A ∪ B|   ∈ [0,1]
```

Example: `A={the cat sat, cat sat on, sat on mat}`, `B={…, …, sat on rug}` → 2/4 = **0.5**.

**Why not compute J directly?** Billions of docs → all-pairs is `O(N²)`, sets are huge. Two cheats:

**1. MinHash — estimate J from k numbers.** With random hash `h`, `minhash_h(A)=min_{x∈A} h(x)`. The min over `A∪B` falls in the intersection iff it's in both sets, so `P[minhash(A)=minhash(B)] = J(A,B)`. Use `k` hashes → length-`k` signature; fraction of matching positions is an **unbiased** estimate of J with `SE=sqrt(J(1−J)/k)`. `k≈128` → any doc is 128 ints, J known to a few %.

**2. LSH banding — only compare likely matches.** Split the signature into `b` bands of `r` rows (`k=b·r`). Two docs are **candidate pairs** if they collide in *any* band:

```
P[candidate] = 1 − (1 − s^r)^b      # S-curve, cliff near t ≈ (1/b)^(1/r)
```

Tune `(b,r)` so the cliff sits at the dedup threshold (e.g. drop J≥0.8). Above the cliff → almost always caught; below → almost never a candidate. Exact-Jaccard verification then runs on a tiny shortlist, not N² pairs.

**Pipeline:** shingle → MinHash signature → band → bucket → candidate pairs → verify exact J → drop. Knobs: *n* (shingle size), *k* (estimate precision), `(b,r)` (cliff location). Same three ideas in `datasketch`, Spark MinHashLSH, FineWeb/SlimPajama stacks.

**One-liner:** Jaccard = set-overlap similarity of shingle sets; MinHash compresses each set to a fixed signature whose match-rate *is* an unbiased Jaccard estimate; LSH banding turns the all-pairs search into a bucket lookup so you only verify a shortlist.

📄 Related: Day 3 §1.2 (full code + S-curve + pipeline diagram).

---

## Q23 — How can you "feed image patches straight into one transformer" (early fusion)?

**Key realization: a Transformer doesn't "know" about text.** Its input is a matrix `[seq_len × d_model]` — a list of `d`-dim vectors — and self-attention mixes them regardless of origin. The *only* text-specific step in an LLM is the front-end: a token id → **embedding-table lookup** → a `d`-dim vector. Early fusion keeps the whole transformer identical and swaps in a different front-end for images, so image and text positions are the same kind of vector in one shared sequence.

**Two ways to build the image front-end:**

**1. Continuous patches — Fuyu (truly encoder-free).** Split the image into patches (e.g. 30×30 px), flatten each to `x ∈ ℝ^(p²·3)`, and apply **one linear layer** `e = Wx + b`, `W ∈ ℝ^(d×p²·3)` → a token-sized vector. That single matmul *is* the entire vision system — no ViT, no CLIP; the transformer's own layers learn to see (gradients flow into `W`). An **image-newline token** after each patch row encodes the 2-D layout.

**2. Discrete tokens — Chameleon (VQ).** A **VQ tokenizer** (VQ-GAN/VQ-VAE) maps the image to a grid of *integer* codebook indices (e.g. 1024 tokens from an 8192 codebook). Add those ids to the **same vocabulary** as text BPE tokens → an image is *literally* a token sequence in the same embedding table. Training = ordinary next-token prediction over the mixed stream; because images are tokens, the model can also **generate** images.

**Then interleave** `[text][image][text]…` into one sequence and train a standard decoder-only transformer with the usual autoregressive loss.

**Why "hardest to train, data-hungry":** no pretrained-encoder head start (learns perception from scratch → more data/compute); VQ **quantizes away** fine detail (hurts OCR); mixing modalities can destabilize training (Chameleon needed **QK-norm** + reordered norms). Payoff: one clean architecture that scales and can both read *and* generate — the bet for the omni/unified future.

**One-liner:** a transformer just consumes `d`-dim vectors, so replace the text embedding-lookup front-end with a patch→linear projection (Fuyu) or a VQ image-tokenizer into a shared vocab (Chameleon); interleave the tokens and train next-token as usual — no separate vision encoder needed.

📄 Related: Day 4 §1.4 (diagram + Fuyu/Chameleon recipes).

---

## Q24 — What did SigLIP 2 add over SigLIP?

**Core unchanged:** the **sigmoid pairwise loss** is still the backbone. SigLIP 2 (Google DeepMind, 2025) wraps it in a *unified recipe* with extra objectives:

1. **Caption/decoder loss (LocCa-style)** — a text decoder trained with captioning + *grounded* objectives (referring-expression / region captioning) → generative supervision + **localization** (where, not just what).
2. **Self-distillation + masked prediction (SILC/TIPS)** — EMA-teacher local-to-global loss + masked image modeling → strong **dense per-patch features** (segmentation/depth), the weak spot of pure contrastive encoders.
3. **Multilingual** — multilingual data mixture (Gemma tokenizer) + fairness **de-biasing**, keeping English perf.
4. **NaFlex variant** — one checkpoint with **native aspect ratio + variable resolution** (no squashing) → big for OCR/docs/charts. Ships FixRes *and* NaFlex.
5. **Drop-in** — B/L/So400m/g sizes, same API.

**Net:** better zero-shot + retrieval *and* markedly better localization, dense-feature, OCR, and multilingual results.

**One-liner:** SigLIP = the sigmoid loss; SigLIP 2 = sigmoid loss + caption/localization (LocCa) + self-distillation dense features (SILC/TIPS) + multilingual + native-resolution (NaFlex), unified.

📄 Related: Day 4 §1.1 (SigLIP → SigLIP 2).

---

## Q25 — Explain Flamingo and BLIP / BLIP-2 in detail

**Flamingo (DeepMind, 2022)** — the canonical *cross-attention* VLM; origin of "freeze the big models, train only a bridge." Keeps a **frozen vision encoder** + **frozen LLM** (Chinchilla) and adds two trainable pieces:
- **Perceiver Resampler** — learnable latent queries cross-attend the encoder's *variable* features → a **fixed 64 visual tokens** (attentional pooling), decoupling encoder output from LLM cost.
- **Gated cross-attention (GATED XATTN-DENSE)** layers interleaved *between* frozen LLM layers; text reads visual tokens. Output scaled by a **tanh gate init 0**, so at init it's exactly the frozen LLM, then opens during training (stability).
Trained with LM loss on **interleaved** image-text web pages. Payoff: **few-shot, in-context** VL (GPT-3-style, multimodal). Only resampler + gated layers train.

**BLIP (Salesforce, 2022)** — unify understanding + generation and clean own data. **MED** (Multimodal mixture of Encoder-Decoder): one weight-sharing model, three modes/losses — **ITC** (contrastive), **ITM** (image-text matching, binary, hard negatives), **LM** (image-grounded captioning). Signature trick **CapFilt**: a *captioner* writes synthetic captions, a *filter* (ITM) drops mismatches → retrain on cleaned data (precursor to self-improvement).

**BLIP-2 (Salesforce, 2023)** — efficient successor; **freezes both** image encoder and LLM, bridges with a tiny **Q-Former**:
- ~32 **learnable query tokens** cross-attend frozen image features → 32 embeddings → projected → LLM **prefix**.
- **Two-stage:** Stage 1 (Q-Former + frozen encoder) ITC+ITM+ITG → queries learn text-relevant features; Stage 2 (+ frozen LLM) generation. Only the ~188M Q-Former trains; plugs into any LLM.

**Common thread:** Flamingo, BLIP-2, and CoCa's poolers all use **attentional pooling / learned queries** to turn variable visual features into a small fixed set. Flamingo & BLIP-2 freeze the big models, differing in entry: Flamingo = **gated cross-attn inside the LLM**; BLIP-2 = **prefix** to an unmodified LLM; LLaVA = even simpler (linear projector + prepend).

**One-liner:** Flamingo = frozen encoder + frozen LLM + Perceiver Resampler + gated cross-attention (few-shot, interleaved); BLIP = MED with ITC/ITM/LM + CapFilt data bootstrapping; BLIP-2 = freeze both, bridge with a 2-stage-trained Q-Former.

📄 Related: Day 4 §1.4 (Flamingo + BLIP-2 diagrams).

---

## Q26 — In L_aux = α·n·Σ f_i·P_i, what's the difference between f_i and P_i?

Both measure "how much expert *i* gets used," but at different points in the routing pipeline and with different differentiability:

**f_i — hard, discrete, post-top-k.** `f_i = (1/T) Σ_x 1[expert i ∈ top-k(x)]` — the fraction of tokens *actually routed* to expert i after the top-k selection. It comes from an argmax, so it's **piecewise constant → no gradient** (nudging router weights usually flips no token's assignment).

**P_i — soft, continuous, pre-top-k.** `P_i = (1/T) Σ_x p_i(x)` — the mean softmax probability the router puts on expert i, averaged over all tokens *before* any truncation. Fully **differentiable**.

**Why you need both:** `Σ f_i²` would measure real imbalance but can't be backpropped; `Σ P_i²` is differentiable but only balances *probabilities* — the router could keep probs near-uniform while top-k still dumps tokens on one expert. The product couples them: gradients flow only through `P_i`, with `f_i` acting as a fixed per-expert weight. Since `∂L_aux/∂p_i(x) ∝ α·n·f_i`, the most-overloaded experts get the strongest pressure to lower their router probability.

**Details that impress:** (1) f_i uses the *post*-top-k hard assignment while P_i uses the *pre*-top-k softmax, so the loss ties the discrete decision to a continuous handle; (2) for top-1 routing `E[f_i] ≈ P_i`, so the minimum is exactly uniform load; (3) at balance `f_i = P_i = 1/n` ⇒ `L_aux = n·Σ(1/n²) = 1` regardless of n — that's what the `n_experts` prefactor is for, so one α transfers across expert counts.

**One-liner:** f_i = hard fraction of tokens routed to expert i (post-top-k, non-differentiable); P_i = mean router softmax prob on expert i (pre-top-k, differentiable); f_i tells the loss *where* the imbalance is, P_i is the differentiable *handle* the gradient uses to fix it.

📄 Related: LLM_Architecture_Frontier §1.2 (load balancing).

---

## Q27 — Why is attention O(T²) in compute/memory, and why does the KV cache grow O(T) and dominate inference memory?

**Attention O(T²) — the score matrix.** Attention is *all-pairs* interaction: T queries each take a dot product with T keys → `QKᵀ` is a **T×T matrix**. Compute = `T²·d` multiply-adds per head per layer (double the context ⇒ 4× the attention FLOPs). Memory = naively you materialize that T×T matrix per head to softmax it (T=32k → 1B floats per head). **FlashAttention** tiles the softmax so the full matrix is never stored — memory falls to O(T) — but **compute stays O(T²)**: the dot products still have to happen. The quadratic is the price of letting every token attend to every other.

**KV cache O(T) — pay once, store forever.** Decoding is autoregressive: token T+1's query must attend over the keys/values of all T previous tokens. Recomputing prefix K,V every step would cost O(T²) *per generated token*, so each token's K,V is computed **once and cached**:

```
KV bytes = 2 (K&V) · n_layers · n_kv_heads · d_head · T · bytes_per_elem
```

Every factor is an architecture constant except T → **linear growth**, per sequence in the batch.

**Why it dominates:** weights are a *fixed* memory cost; the cache scales with **T × batch_size**. Concretely for a 7B model (32 layers, 4096 hidden, fp16): ≈ **0.5 MB/token** → one 128k context ≈ **64 GB of cache vs ~14 GB of weights**. It also hurts bandwidth, not just capacity: each decode step reads the *entire* cache from HBM to emit one token → decode is **memory-bandwidth-bound**. That's the motivation for GQA/MQA (fewer KV heads), MLA (compress d_head), KV quantization (fewer bytes), sliding window (cap T) — each attacks one factor in the formula.

**One-liner:** T queries × T keys = a T×T score matrix → O(T²) compute (Flash removes the memory, not the FLOPs); caching one K,V per token per layer avoids prefix recomputation = O(T) storage scaling with context×batch while weights stay fixed — so at long context the cache, not the model, fills the GPU.

📄 Related: LLM_Architecture_Frontier §2.1 (two costs of long context), Day 2 §3 (GQA/MQA).

---

## Q28 — "Given X more compute: bigger-dense, MoE, longer context, test-time compute, or better data?" (the architecture-taste meta-question)

**The expected answer is a decision procedure, not a pick.** Structure:

**Step 0 — Clarify (this IS the taste signal).** "X more compute" is underspecified: *training* compute or *lifecycle* compute? MoE spends training FLOPs to buy cheap inference; test-time compute is the opposite trade — free at training, multiplies every query's cost. Also pin the target metric (general loss vs reasoning vs agentic) and deployment profile (latency, serving memory, queries/lifetime).

**Step 1 — Diagnose the bottleneck.** Each option fixes a different failure: bigger-dense = predictable scaling gains but inference cost ∝ params and loses if data-bound (Chinchilla); MoE = quality per training-FLOP but pays in serving memory + expert-parallel infra + routing instability; longer context = new *capabilities* not raw quality, wasted unless evals show truncation failures; test-time compute = big reasoning gains with no retraining but cost ∝ usage and only works where verification/search applies; better data = shifts the scaling curve's *constant* (often highest ROI) but hard to measure and risks diversity collapse. Ground the choice in **eval error analysis**: knowledge misses → capacity/data; reasoning misses → test-time compute; truncation → context; loss still falling fast at train end → more data.

**Step 2 — Cheap proxies, cheapest first, designed to shift a curve not a point:**
1. **~Free:** pass@k / best-of-n + verifier on the *current* model = upper bound on test-time-compute gains before spending anything (Snell et al. 2024: compute-optimal test-time scaling can beat a ~14× bigger model — where a verifier exists).
2. **Cheap:** data ablation at fixed model size, matched tokens (quality effects show at small scale).
3. **Moderate:** iso-training-FLOP MoE-vs-dense pair; note the memory bill (Krajewski et al. 2024: the MoE edge grows with scale).
4. **Decider:** scaling ladder — 3–4 sizes per intervention, fit `L(C)=a·C^(−b)+c`, extrapolate whichever moves the exponent/offset (Chinchilla methodology; MM1 does the same for multimodal mixture ratios).

**Step 3 — Commit with a flip condition.** "Default: better data + MoE for the pretraining spend, test-time compute layered at deployment for reasoning tasks (its cost scales with usage, not X); bigger-dense only if serving simplicity dominates; longer context only on truncation evidence. I'd flip to test-time-compute-first if the pass@k gap on target tasks is large and verifiable."

**One-liner:** clarify training-vs-lifecycle compute → diagnose bottleneck from eval errors → proxies cheapest-first (pass@k free → data ablation → iso-FLOP MoE pair → scaling ladder decides) → commit to data+MoE at training and test-time compute at deployment, naming what would flip you.

📄 Related: Day 5 §1.4 (full decision procedure + diagram), LLM_Architecture_Frontier §1 (MoE), §2 (long context).

---

## Q29 — Is there research treating the data mixture as a hyperparameter that changes with scale and with training progress t?

**Yes — both axes have dedicated literatures**, and the field has moved from a constant w* to **w*(N, D, t)**.

**Axis 1 — mixture as a function of scale.**
- **AutoScale** (Kang et al., COLM 2025; arXiv 2407.20177) — the most direct answer: shows the compute-optimal composition is *scale-dependent*, derives an analytical model for how optimal weights shift with training scale, finds the small-scale optimum via bilevel optimization (DDO), then *predicts* the optimum at target scale (≥25% faster validation-perplexity decrease at GPT-2-Large/RedPajama scale).
- **Data Mixing Laws** (Ye et al. 2024) — fits loss-vs-mixture functional forms and *nests them inside* model-size/step scaling laws → extrapolate a small-scale mixture choice to the large run.
- **BiMix** (Ge et al. 2024) — bivariate scaling law in (domain proportion × data volume).
- **Scaling Laws for Data Filtering** (Goyal et al., CVPR 2024) — "curation cannot be compute-agnostic": how hard you filter / how much you repeat quality data should change with the compute budget.
- Contrast: **DoReMi** (2023) / **RegMix** (2024) assume the small-proxy optimum transfers (rank/scale-invariance) — the papers above are precisely the evidence that this assumption breaks.

**Axis 2 — mixture as a function of training progress t.**
- **ADO — Adaptive Data Optimization** (Jiang et al., ICLR 2025; arXiv 2410.11820) — fits *per-domain scaling laws online during the run* and re-weights domains by current learning potential; <0.4% wall-clock overhead; best average downstream at both scales tested (124M and 1.3B on the Pile), and its learned curricula differ across scales. Full TLDR: Q30.
- **Aioli** (Chen et al. 2024) — unifies mixing-law approaches in one online optimization framework.
- Earlier online/curriculum methods: **ODM** (bandit-based online mixing, Albalak et al. 2023), **Skill-It** (loss-driven skill curriculum, Chen et al. 2023), **DoGE** (gradient-based generalization estimates, Fan et al. 2024).
- **Production practice is already w(t):** Llama 3, MiniCPM (WSD decay phase), OLMo 2 (mid-training), and NVIDIA's two-phase pretraining all up-weight high-quality + math/code data in the final annealing phase — a coarse step-function mixture schedule.

**Interview one-liner:** the static-mixture view is outdated — AutoScale shows w* shifts with scale, ADO shows it should shift *within* a run, and every frontier lab ships a two-phase anneal; treat the mixture as a schedule w(N, t) fitted by nested scaling laws, not a constant vector.

📄 Related: Resume_Interview_Prep §2 (new deep card + w*(scale)/w(t) figure), Day 3 (data curation), Day 5 §1.4 (compute-allocation).

---

## Q30 — TLDR: Adaptive Data Optimization (ADO), Jiang et al., ICLR 2025

**Problem:** how to weight data domains during pretraining without the proxy-model tax. DoReMi/DoGE/mixing-laws need proxy models or multi-stage pipelines (one DoReMi round ≈ 760 A100-hours), and proxy weights are brittle (tokenizer-sensitive; loss extrapolation from a few small Pythia models has high variance → small-scale mixtures don't reliably transfer up).

**Core idea:** during the *actual* run, fit a tiny power law **per domain** to that run's own loss curve, refit periodically:

```
L̂_k(n) = ε_k + β_k · n^(−α_k)          # ε_k = domain's irreducible loss ("entropy")
−dL̂_k/dn = (1/n) · α_k · (L̂_k − ε_k)   # learning potential = speed × reducible loss
```

Prioritize domains with high *information gain per sample*. This fixes ODM's raw-high-loss heuristic, which starves low-entropy domains like code (GitHub val loss: ADO 1.40 vs ODM 1.54).

**Supporting machinery:** (1) **credit assignment** λ_k = smoothed EMA of recent sampling history — a domain only gets credit for its own loss drop if it was actually sampled recently (else the drop is cross-domain transfer); (2) **temporal averaging** of the policy + token-proportional prior μ_k + 1% probability floor. Final weight: `ρ_k ∝ μ_k · λ_k · α_k(L̂_k − ε_k)/n`.

**Results (Pile; 124M/15B tokens, 1.3B/125B tokens):** best average zero-shot downstream at *both* scales with one fixed hyperparameter set; beats Natural on SlimPajama/FineWeb val loss (drifts toward "quality" data) though slightly behind Natural on Pile val loss. Overhead: +20 min on a 3.5-day 1.3B run (~0.4%); cost independent of model size.

**Quotable findings:** (1) **Natural (token-proportional) is a shockingly strong baseline** — second-best downstream at both scales; almost no data-selection paper benchmarks it. (2) **Learned curricula differ across scales** (GitHub weight decays at 124M, dips-then-recovers at 1.3B) and shift over training — direct evidence for mixture-as-w(N, t) (→ Q29). (3) Meta-optimized data *orderings* beat random even for logistic regression → good curricula exist, they're just expensive to find; ADO is the cheap approximation. (4) Online scaling-law fits are locally accurate but overestimate final loss (LR decay isn't modeled).

**Limitations:** only self-interactions (no cross-domain transfer term), 1.3B max, LR schedule outside the law, downstream-agnostic (undervalues code-for-reasoning; fixable via prior).

**One-liner:** ADO replaces proxy-model mixture search with per-domain power laws fitted online to the live run — sample each domain ∝ prior × recent-usage credit × predicted information gain per sample — matching or beating DoReMi at ~0.4% overhead, and showing the optimal mixture is a function of both scale and training time.

📄 Paper: arXiv 2410.11820 · Related: Q29 (mixture as w(N,t)), Resume_Interview_Prep §2 deep card.

---

## Q31 — Deep dive: AutoScale (arXiv 2407.20177) — big idea, machinery, pros/cons + head-to-head vs ADO

**Big idea:** not just "the optimal mixture is scale-dependent" — the scale-dependence is itself **law-like and predictable**: optimal per-domain token counts are log-log linear in total budget (R²=0.998), so measure the optimum at two small scales and extrapolate geometrically to the target scale.

**Machinery (3 layers):**
1. **DDO:** the exact problem is bi-level (outer: weights; inner: ERM training). Collapse to single level via a per-domain transfer-style power law `L(N'_i) = (N₀ⁱ + N'_i)^(−γᵢ) + ℓᵢ`, where `N₀ⁱ` = "equivalent data" from all *other* domains (static stand-in for transfer; Hernandez et al. 2021 form). Fit each domain from 3 runs (baseline; domain-i tokens ×3; ÷3 → OLS). Reduced objective `Σᵢ(N₀ⁱ + wᵢN)^(−γᵢ)` is **convex** → projected GD, global optimum; 2m+1 retrains total.
2. **Theorem:** with additive per-domain power laws, optimizers at any budgets obey `Nᵢ⁽³⁾* = (Nᵢ⁽²⁾*)²/Nᵢ⁽¹⁾*` — a geometric recursion; two anchor solves generate the whole w*(N) trajectory (extends to a latent-skills version without independence).
3. **AutoScale:** run DDO at two small *data* scales with the **full-size model** (774M — no smaller proxy), fit the recursion, iterate to target.

**Results:** GPT-2 Large on RedPajama (3B/5B/10B tokens): val PPL drops ≥28% faster than every baseline (DoReMi, Data Mixing Laws, LLaMA weights, uniform), up to 38% vs unweighted; best downstream average. BERT: only ~10–17% — effect is architecture/objective-dependent. **Scientific payload:** standardized "high-quality" domains (Wikipedia, arXiv) matter most at small scale then sharply saturate; diverse sources (CC, C4, Books) keep paying as scale grows — explains why CC-heavy LLaMA weights beat uniform only at large scale.

**Pros:** kills the proxy-model confound (same model size; only data scale varies); principled (convex → global optimum; falsifiable closed-form prediction); drop-in static weights, no pipeline changes; opens the w*(N) scaling-law axis.

**Cons:** extrapolates data scale at **fixed model size** — joint (params, tokens) scaling unmodeled (biggest gap to practice); independence assumption freezes cross-domain transfer into static N₀ⁱ; geometric recursion **squares** its inputs → anchor-fit noise (3 points/domain, r=3 heuristic) compounds exponentially with extrapolation distance; DDO = 2m+1 retrains per anchor; static within-run; validated only at 774M/≤10B tokens.

**Head-to-head vs ADO (Q30):** AutoScale = **offline planner** for w*(N) across scales (theory-strong: convexity + closed-form recursion; practice-weak: fixed model size, static in-run). ADO = **online controller** for w(t) within the run (practice-strong: 0.4% overhead, adapts to real tokenizer/arch/optimizer; theory-weak: greedy, no transfer modeling, LR-blind). **Convergent finding from both:** diverse web data gains importance with scale while standardized high-quality text saturates early — two methodologies, same physics. **Synthesis / proposed experiment:** use AutoScale-style prediction to set ADO's prior μ at target scale, let ADO adapt within-run → full w(N,t); test whether the combination beats either alone at iso-compute.

**One-liner:** AutoScale proves w* shifts predictably with data scale and extrapolates it via a convex small-scale solver + geometric recursion (planner); ADO adapts w online from the live run's per-domain scaling laws (controller); plan the prior with one, steer the run with the other.

📄 Papers: arXiv 2407.20177 (COLM 2025), arXiv 2410.11820 (ICLR 2025) · Related: Q29, Q30, Resume_Interview_Prep §2.

---

## Q32 — Can PyTorch and JAX estimate FLOPs?

**Yes — both, natively.**

**PyTorch:**
1. **`torch.utils.flop_counter.FlopCounterMode`** (≥2.0) — the rigorous one. Intercepts ops via `__torch_dispatch__`, counts matmul/conv/SDPA **including backward**:
```python
from torch.utils.flop_counter import FlopCounterMode
with FlopCounterMode(display=True) as fcm:
    loss = model(x).sum(); loss.backward()
total = fcm.get_total_flops()
```
2. **`torch.profiler(with_flops=True)`** — per-op estimates (matmul/conv family only); good for hotspots, not totals.
3. Third-party for module breakdowns: `fvcore FlopCountAnalysis`, DeepSpeed flops profiler, `calflops`.

**JAX — ask XLA's cost model:**
```python
lowered  = jax.jit(train_step).lower(params, batch)
lowered.cost_analysis()["flops"]     # pre-optimization HLO
lowered.compile().cost_analysis()["flops"]  # post-fusion (more faithful)
```
`jit` the whole train step and backward is included automatically (it's just part of the traced graph).

**Caveats / interview flavor:**
- MAC = 2 FLOPs convention in these tools; some papers report MACs (÷2) — always say which.
- Counters skip/approximate elementwise + norm ops (<1% for transformers; matmul-only counting is standard).
- **Sanity anchor:** dense transformer ≈ `6·N·D` (2ND fwd + 4ND bwd) + attention `≈ 12·L·T²·d_head·n_heads` per pass. If the tool disagrees wildly with 6ND, suspect the count.
- These give *algorithmic* FLOPs, not achieved hardware FLOPs. **MFU = counted FLOPs / (wall-clock × peak hardware FLOPS)** — the counter supplies the numerator.

**One-liner:** PyTorch's `FlopCounterMode` counts dispatched ops (fwd+bwd) and JAX's `lower().compile().cost_analysis()` asks XLA's post-fusion cost model; both give algorithmic FLOPs for the MFU numerator, sanity-checked against 6ND.

📄 Related: Resume_Interview_Prep §3 (hero compute / scaling ladder), Day 1 (backprop cost).

---

## Q33 — How exactly does patchify work in ViT?

**Big idea:** patchify is ViT's tokenizer — it converts a 2D image into a 1D sequence of token embeddings so a vanilla transformer can consume it. No convolution stack, no feature pyramid: cut, flatten, project.

**The 4 steps (ViT-Base defaults: 224×224×3 image, P=16, D=768):**
1. **Cut** the image into non-overlapping P×P patches: `N = HW/P² = (224/16)² = 14×14 = 196` patches.
2. **Flatten** each patch to a raw pixel vector: `x_p ∈ ℝ^(P²·C) = ℝ^(16·16·3) = ℝ⁷⁶⁸` (the 768 here is coincidence — it's pixel count, not model width).
3. **Linearly project** with a single shared weight matrix `E ∈ ℝ^(P²C × D)`: `z_p = x_p·E + b`. Same E for all 196 patches — it's a per-patch "embedding lookup" analog.
4. **Prepend [CLS]** (a learned vector) and **add learned positional embeddings** `pos ∈ ℝ^((N+1)×D)` → sequence of 197 tokens into the encoder.

**Two equivalent implementations** (identical math, conv is what everyone ships):
```python
# einops view — makes the "cut + flatten" explicit
x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=16, p2=16)
z = x @ E + b                      # (B, 196, D)

# conv view — Conv2d with kernel = stride = P is exactly patchify + projection
z = nn.Conv2d(3, D, kernel_size=16, stride=16)(img)  # (B, D, 14, 14)
z = z.flatten(2).transpose(1, 2)                     # (B, 196, D)
```
A stride-P conv with kernel P touches each patch exactly once with shared weights — that *is* the shared linear projection.

**Interview flavor:**
- **P is the compute dial:** sequence length N = HW/P², attention is O(N²) → halving P quadruples tokens and ~16×'s attention FLOPs. ViT-L/14 vs /16 vs /32 is exactly this trade (CLIP/DINOv2 use P=14).
- **Resolution change at fine-tune:** patches stay P×P so N grows; you 2D-interpolate the positional embeddings (the standard trick from the ViT paper).
- **Patchify is the "weak inductive bias" move:** within a patch, spatial structure is destroyed (flattened); across patches, locality must be *learned* via attention + pos embeddings — this is why ViTs need more data/augmentation than CNNs at small scale but win at large scale.
- **Modern variants:** FlexiViT (random P at train time → one model, many patch sizes), NaViT (native aspect ratio, pack variable-N sequences), conv-stem hybrids (a few 3×3 convs before patchify stabilize training).

**One-liner:** patchify = reshape the image into N = HW/P² non-overlapping P×P patches, flatten each to P²C values, and push all of them through one shared linear map into D dims — equivalently a Conv2d(kernel=P, stride=P) — then add [CLS] + positional embeddings; P sets the token budget and thus the O(N²) attention cost.

📄 Related: Q13 (Jacobians), LLM_Architecture_Frontier (attention cost), Apple NMM scaling-law interest (vision-encoder token budgets).

---

## Q34 — After patchify, how does the encoder turn patch tokens into "soft" representations?

**Big idea:** the encoder is a stack of L identical pre-LN transformer blocks with **full bidirectional attention** (no causal mask). Each block lets every patch token softly pull in information from every other patch; after L rounds of mixing, each output token is no longer "pixels of patch i" but "patch i in the context of the whole image." They're "soft" in two precise senses: the mixing weights are a softmax (a soft, differentiable selection, never a hard pick) and the outputs are continuous D-dim vectors (not discrete codebook ids).

**One block (pre-LN, the ViT default):**
- `z̃ = z + MHSA(LN(z))` — attention sublayer + residual
- `z' = z̃ + MLP(LN(z̃))` — MLP sublayer + residual
- after block L: one **final LayerNorm**, then read out.

**Inside MHSA (per head, d_h = D/h):**
1. Project: `Q = XW_Q, K = XW_K, V = XW_V` (X is the 197×D token matrix).
2. Soft weights: `A = softmax(QKᵀ/√d_h)` — row i is a probability distribution over all 197 tokens.
3. Blend: `head = A·V` — token i's update is a **convex combination** of all value vectors.
4. Concat h heads, project with `W_O`. Bidirectional: A is a full 197×197 matrix, no mask — global receptive field from layer 1 (vs CNNs that grow it layer by layer).

**Inside MLP:** `Linear(D→4D) → GELU → Linear(4D→D)`, applied per token (no cross-token mixing). Attention moves information *between* tokens; the MLP transforms it *within* each token. The MLP holds ~2/3 of the params.

**What emerges across depth:** early layers = local/texture-like attention (some heads attend to neighbors, mimicking convs); later layers = global, semantic attention; [CLS] progressively aggregates an image-level summary. Positions are injected only once at the input (layer 0) — modern ViTs (DINOv2 variants, NaViT-era) often add 2D-RoPE inside attention instead.

**Where the soft tokens go:**
- **Classification:** [CLS] (or mean-pooled patches) → linear head.
- **Contrastive (CLIP):** pooled token → projection → shared embedding space.
- **Multimodal LLM (the "soft token" usage that matters for NMM/LLaVA-style stacks):** the 196 patch outputs are continuous **soft visual tokens** — passed through a connector (linear/MLP in LLaVA; Perceiver resampler/Q-Former to *compress* 196→64/32) and spliced into the LLM's input sequence like word embeddings. "Soft" here contrasts with discrete VQ tokens (VQ-VAE/VQGAN codes): no quantization, fully differentiable end-to-end, but not directly usable for autoregressive image *generation*.

**Interview flavor:**
- Compute split: attention is O(N²·D) (pairwise mixing) vs MLP O(N·D²) (per-token transform); with N=197, D=768 the **MLP dominates FLOPs** — "attention is quadratic" only bites at large N (high res / small P, back to Q33's token-budget dial).
- The encoder is order-equivariant except for positional embeddings — shuffle patches + their pos embeds and the output shuffles identically.
- Softmax temperature 1/√d_h keeps logits' variance ~1 at init so attention starts diffuse (near-uniform soft mixing) rather than saturated.
- Bidirectional ⇒ great encoder, but no causal factorization ⇒ can't autoregressively generate pixels/tokens without a decoder or discrete tokenizer on top.

**One-liner:** the encoder alternates soft cross-token mixing (softmax(QKᵀ/√d)·V, all 197↔197, plus residual) with per-token MLP transforms, ×L with pre-LN and a final LN; the result is 197 continuous, contextualized "soft tokens" — [CLS] for classification, the 196 patch tokens as the soft visual prompt a connector feeds into an LLM.

📄 Related: Q33 (patchify), Q19 (pre-norm vs post-norm), Q18 (LN/RMSNorm), Q32 (FLOP counting), LLM_Architecture_Frontier (attention variants).

---

## Q35 — Is centering in DINOv3 like batch norm on logits?

**Short answer:** yes in spirit — with two precision points. (1) *DINO v1* centering is literally the **mean-subtraction half of BN applied to teacher logits** (EMA statistics, no variance division, no affine, teacher-only). (2) *DINOv3* (inheriting DINOv2) doesn't use that anymore — it replaced centering with **Sinkhorn-Knopp**, which the DINOv2 paper itself calls "the Sinkhorn-Knopp (SK) **batch normalization** of SwAV." So the papers bless the analogy — but the *purpose* is anti-collapse target balancing, not optimization conditioning.

**DINO v1 mechanics (the thing being analogized):** teacher logits `g_t(x) ∈ ℝ^K` over K prototypes; center `c` updated per step as `c ← m·c + (1−m)·mean_batch(g_t(x))` (m ≈ 0.9); teacher targets = `softmax((g_t(x) − c)/τ_t)` with τ_t ≈ 0.04–0.07 (**sharpening**). Centering alone → uniform collapse; sharpening alone → single-prototype collapse; their balance keeps targets confident *and* spread across prototypes.

**Why mean subtraction prevents collapse:** softmax is invariant to a per-*sample* constant shift, but `c` is a per-*prototype* mean across the batch. If one prototype's logit is high for everyone (the collapse mode), its entry in `c` grows and gets subtracted away — a negative-feedback loop on prototype dominance. First-order statistics only, which is exactly the "running-mean BN" flavor.

**What DINOv3 actually does — SK centering (both DINO and iBOT heads):** run 3 Sinkhorn-Knopp iterations on the batch's teacher score matrix, alternately renormalizing rows (per-sample distributions) and columns (per-prototype usage). It's the entropic-OT projection onto ~doubly-stochastic assignments: **exact equipartition pressure** (each prototype gets ≈ B/K mass) instead of v1's soft first-order mean shift. Think "iterated two-sided batch norm on the softmax matrix." DINOv3 keeps this from v2 and adds KoLeo (uniform feature span in batch) and Gram anchoring (fixes dense-feature degradation over long training — separate problem).

**Where the BN analogy breaks — the checklist:**

| | BatchNorm (train) | DINO v1 centering | SK (DINOv2/v3) |
|---|---|---|---|
| statistic | current-batch mean **and** var | **EMA** of batch mean only | full row+col marginals |
| transform | (x−μ)/σ · γ + β | x − c only | iterative rescaling |
| learnable params | γ, β | none | none |
| gradient through statistic | **yes** (source of BN's batch coupling) | **no** (teacher, stop-grad) | no |
| applied where | every layer, both passes | teacher logits only | teacher logits only |
| goal | conditioning/optimization | anti-collapse (with sharpening) | hard assignment balancing |

The gradient row is the deep one: BN backprops through μ, σ, coupling samples in the *gradient*; centering never does — it shapes **targets**, not gradients (teacher is EMA + stop-grad).

**Interview flavor:** this puts centering in the "batch statistics against collapse" family — BN's implicated role in BYOL's stability (the "BYOL works without batch statistics" debate), VICReg's variance term, W-MSE whitening, SwAV equipartition, KoLeo. If asked "why did v2/v3 switch to SK?": v1 centering only constrains the mean and balances slowly via EMA; SK enforces balanced usage *within each batch*, more stable at scale (DINOv3: 7B params, 1.7B images) — and it costs only 3 cheap normalization iterations.

**One-liner:** v1 centering ≈ BN's running-mean half on teacher logits (mean-only, EMA, no gradient through the statistic, anti-collapse not conditioning); DINOv3 upgraded to Sinkhorn-Knopp — literally "SK batch normalization" per the DINOv2 paper — which rebalances rows *and* columns of the assignment matrix for hard equipartition, paired against low-τ sharpening.

📄 Papers: DINO (arXiv 2104.14294), DINOv2 (arXiv 2304.07193), DINOv3 (arXiv 2508.10104) · Related: Q34 (encoder/soft tokens), Q18 (BN stats noise), Q2 (softmax saturation).

---

## Q36 — Write out DINOv3's overall loss. Is there any contrastive term — do patches from different images ever interact?

**Total loss (paper Eq. 1, pretraining phase):**
`L_pre = L_DINO + L_iBOT + 0.1·L_KoLeo`
**Refinement phase (Eq. 3, after 1M iterations):**
`L_ref = w_D·L_DINO + L_iBOT + w_DK·L_KoLeo + w_Gram·L_Gram`
Setup: 2 global crops (256²) + 8 local crops (112²) per image; student sees all 10 crops plus masked versions of the globals; teacher (EMA of student, stop-grad) sees only the global crops, with SK centering + sharpening on its outputs (Q35).

**The four terms:**
1. **L_DINO (image-level distillation):** `−Σ_{g∈globals} Σ_{v≠g} Σ_k P_t^(g)[k] · log P_s^(v)[k]` — CE between teacher's (SK-centered, sharpened) prototype distribution for a global crop's [CLS] and the student's distribution for **every other crop of the same image** (global↔global cross + local→global). This is the "local-to-global correspondence" pressure.
2. **L_iBOT (patch-level MIM):** `−Σ_{i: masked} Σ_k P_t^patch(x_i)[k] · log P_s^patch(x̂_i)[k]` — student sees a masked global crop, teacher sees it unmasked; CE at each masked position against the teacher's token for the **same patch of the same view**. Separate head from DINO.
3. **L_KoLeo (uniformity):** `−(1/n) Σ_i log d_i`, `d_i = min_{j≠i} ‖f_i − f_j‖` — Kozachenko-Leonenko entropy estimator on student [CLS] features, pushing each feature away from its nearest neighbor **in the batch** (distributed variant: groups of 16 samples). Weight 0.1.
4. **L_Gram (v3's addition, late phase):** `‖X_S·X_Sᵀ − X_G·X_Gᵀ‖_F²` — X = L2-normalized patch features of a global crop; match the student's **patch-pair similarity structure** (Gram matrix) to a "Gram teacher" (snapshot refreshed every 10k iters). Fixes dense-feature degradation in long 7B-scale training. Same image only.

**Now the actual question — is anything contrastive?** **No InfoNCE, no negative pairs, anywhere.** Every CE term (DINO, iBOT) and the Gram term compares representations **of the same image** — student view vs teacher view. Patches or crops from *different* images never appear together inside any loss numerator/denominator the way SimCLR/CLIP negatives do. If you deleted every other image from the batch, L_DINO/L_iBOT/L_Gram for image x would be unchanged.

**But pure "student follows teacher" alone would collapse** — both nets emit one constant distribution and every CE term hits zero. What replaces negatives:
- **SK centering (batch-level, teacher side):** targets are rebalanced so each prototype gets ≈B/K mass *across the batch* — other images shape *your* target implicitly (Q35).
- **Sharpening (τ_t ≈ 0.05):** counters centering's pull toward uniform.
- **L_KoLeo (batch-level, student side):** the only *explicit* cross-image force — a repulsion/uniformity term, "negatives-lite" without logits.
- **Architecture asymmetries:** EMA teacher, stop-grad, student-only masked/local views.

**Interview framing (Wang & Isola):** contrastive InfoNCE = alignment (positives together) + uniformity (negatives apart) *bundled in one loss*. The DINO family **unbundles** them: CE distillation supplies alignment; SK centering + KoLeo supply uniformity via batch statistics instead of pairwise negatives. So "is there a contrastive loss?" → no; "is there cross-image signal?" → yes, exactly two places: SK centering and KoLeo.

**One-liner:** `L_pre = L_DINO + L_iBOT + 0.1·L_KoLeo` (+ Gram anchoring after 1M iters); DINO and iBOT are same-image teacher→student CE (cross-view [CLS], masked patches), Gram is same-image patch-pair structure — nothing contrastive; other images enter only through SK-centered targets and KoLeo's nearest-neighbor repulsion, which replace InfoNCE's negatives as the uniformity force.

📄 Papers: DINOv3 (arXiv 2508.10104, Eq. 1–3), iBOT (arXiv 2111.07832), KoLeo from Sablayrolles et al. (spreading vectors) · Related: Q35 (centering/SK), Q34 (soft tokens), Q29–31 (data-mixture scaling).

---

## Q37 — What exactly is the [CLS] token, and how do you build it in a ViT?

**What it is:** a single learned parameter vector `cls ∈ ℝ^D` — a free `nn.Parameter`, *not* derived from any pixels. At input it's **identical for every image**; it becomes image-specific only inside the encoder, by attending to the patch tokens. Think of it as a learned blank slate whose output slot is reserved as the image-level readout. Borrowed directly from BERT's [CLS].

**How to build it (the whole thing is ~4 lines):**
```python
self.cls_token = nn.Parameter(torch.zeros(1, 1, D))
nn.init.trunc_normal_(self.cls_token, std=0.02)
self.pos_embed = nn.Parameter(torch.zeros(1, 197, D))   # 196 patches + CLS at position 0

cls = self.cls_token.expand(B, -1, -1)   # (B, 1, D) — broadcast, same vector per image
x = torch.cat([cls, patch_tokens], dim=1)  # (B, 197, D)
x = x + self.pos_embed
```
It's trained by ordinary backprop — gradients reach it through every attention interaction it participates in.

**Why a dedicated token instead of pooling:**
- It's spatially neutral: attends to all patches with no fixed location bias, so no single patch is privileged as "the summary."
- It gives losses/heads a stable slot: classification head reads `x[:, 0]`; **DINO's image-level loss is defined on exactly this token** (Q36).
- The alternative works too: ViT paper ablates **mean-pooling the patch tokens (GAP)** and gets ≈ equal accuracy — CLS is a convention, not a necessity. CLIP-style models often use attention pooling instead.

**Interview flavor:**
- Attention is bidirectional (Q34), so patches can also *read from* [CLS] — global information leaks into patch tokens. At scale this produces **high-norm artifact patches** that hijack unused patches as global scratch space; the fix is **registers** ("Vision Transformers Need Registers"): extra CLS-like learned tokens, prepended the same way, that absorb the scratch role and are discarded at output. DINOv2 (retrofit) and DINOv3 use them.
- [CLS] adds one token: 196 → 197. Sequence-length cost is negligible; the pos-embed table just gets one extra row (kept, not interpolated, at resolution changes).
- In DINO-family, [CLS] output → DINO head → prototypes; patch outputs → iBOT head (Q38). Same backbone, different readouts.

**One-liner:** [CLS] is a learned 1×D `nn.Parameter`, identical across images, concatenated in front of the patch tokens with its own position-0 embedding; the bidirectional encoder turns its output slot into a query-built summary of the whole image — functionally a learned alternative to mean-pooling, and the anchor for BERT-style heads and DINO's image-level loss.

📄 Related: Q33 (patchify), Q34 (encoder/soft tokens), Q36 (DINO loss on [CLS]) · Registers: arXiv 2309.16588.

---

## Q38 — What is the iBOT loss?

**Big idea:** iBOT (**i**mage **B**ERT pre-training with **O**nline **T**okenizer) = masked-language-modeling for images, done **in latent space with the teacher as a live tokenizer**. BERT needs a discrete vocabulary to define its fill-in-the-blank targets; images don't have one. BEiT solved this with a *frozen pretrained* dVAE tokenizer; iBOT's move is to let the **EMA teacher generate the targets online** — soft prototype distributions per patch (soft tokens again, Q34) that co-evolve with the student.

**Mechanics (as used in DINOv2/v3):**
1. Take a global crop; sample a patch mask m (blockwise, ~10–50% of positions).
2. **Student input:** masked view — masked positions' patch embeddings are replaced by a single learned `[MASK]` embedding (post-patchify, pre-encoder).
3. **Teacher input:** the same view, unmasked. EMA weights, stop-grad.
4. Both encode; **patch head** (separate from the DINO head in v2/v3) maps every patch token to K prototype logits; teacher's are SK-centered + sharpened (Q35).
5. CE **only at masked positions**, matching position i to position i:
`L_iBOT = − Σ_{i: m_i=1} Σ_k P_t^(i)[k] · log P_s^(i)[k]`
Same image, same view, same position — the student must infer what the teacher "saw" at patch i from surrounding visible context. Pure spatial-context reasoning, which is exactly what dense tasks need.

**Why DINO needs it:** L_DINO supervises only [CLS] → strong global features, mediocre patch-level features. L_iBOT supervises every masked patch token → segmentation/depth/correspondence quality. That's the division of labor in `L_pre = L_DINO + L_iBOT + 0.1·L_KoLeo` (Q36).

**MIM family placement (know this table):**
| method | target | tokenizer |
|---|---|---|
| BEiT | discrete dVAE code | frozen, pretrained offline |
| MAE | raw pixels | none (regression, no teacher) |
| **iBOT** | **soft prototype distribution** | **online = EMA teacher** |
| data2vec | latent features (regression) | online = EMA teacher |

**Interview flavor:** the "online tokenizer" is chicken-and-egg by design — targets improve as the student improves; collapse is held off by the same SK centering + sharpening as the DINO head. v1 iBOT shared the head with DINO; DINOv2 found **separate heads** scale better (the two losses want different feature specializations). Masking is student-only — the teacher never sees [MASK], so targets are always computed from clean context.

**One-liner:** iBOT = BERT-style masked-patch prediction where the "vocabulary" is a set of K prototypes and the labels are the EMA teacher's SK-centered soft distributions for the same patches of the same image, CE'd at masked positions only — latent MIM with an online tokenizer (vs BEiT's frozen dVAE, MAE's pixels), and the term that gives DINOv2/v3 their dense-feature quality.

📄 Papers: iBOT (arXiv 2111.07832), BEiT (2106.08254), MAE (2111.06377), data2vec (2202.03555) · Related: Q36 (role in DINOv3 loss), Q37 ([CLS] vs patch readouts), Q35 (SK centering).

---

## Q39 — DINO, iBOT, Gram, KoLeo: what is each loss, exactly?

**The organizing insight:** the four losses split into two families. **Two CE distillation losses do the learning** (DINO = image level, iBOT = patch level — same machinery, different token). **Two geometry losses shape the space** (KoLeo = first-order, don't clump; Gram = second-order, don't drift). Combined: `L_pre = L_DINO + L_iBOT + 0.1·L_KoLeo`, plus `w_G·L_Gram` from 1M iterations (Q36).

**1. L_DINO — image-level self-distillation (on [CLS], cross-view):**
`L_DINO = − Σ_{g∈{g1,g2}} Σ_{v≠g} Σ_k Pₜ^(g)[k] · log Pₛ^(v)[k]`
- Pₜ = teacher [CLS]′ of a **global** crop → DINO head → K prototype logits → **SK-centered + sharpened** (τₜ≈0.05) softmax: a sharp, near-one-hot "which visual concept is this" distribution.
- Pₛ = student [CLS]′ of **every other crop** (incl. 112² locals) → same head → softmax at τₛ=0.1 (softer).
- CE pulls the student's belief toward the teacher's, across views — a local crop must recognize the whole-image concept from a fragment ("local-to-global correspondence"). Same image always.
- *Learns:* global semantic invariance. *Failure it prevents:* none by itself — it's the term that would collapse without SK+sharpening (Q35/Q36).

**2. L_iBOT — patch-level masked distillation (same view, same position):**
`L_iBOT = − Σ_{i: m_i=1} Σ_k Pₜ^(i)[k] · log Pₛ^(i)[k]`
- Same skeleton as DINO but: token = patch i (not [CLS]); separate **iBOT head**; student's input had patch i replaced by learned [MASK]; teacher saw the full view; CE **only at masked positions**.
- The student must infer patch content from spatial context — BERT for images with the teacher as online tokenizer (Q38).
- *Learns:* dense/local features (segmentation, depth, correspondence).

**3. L_KoLeo — batch uniformity (the only cross-image term):**
`L_KoLeo = −(1/n) Σᵢ log dᵢ,  dᵢ = min_{j≠i} ‖fᵢ − fⱼ‖`
- fᵢ = ℓ2-normalized student [CLS] features; computed in groups of 16 (distributed). Weight 0.1.
- It's the Kozachenko-Leonenko **differential-entropy estimator**: maximizing log nearest-neighbor distance ≈ maximizing the entropy of the feature distribution → features spread uniformly over the hypersphere.
- Gradient picture: each feature is pushed away from its **single nearest neighbor** only — the cheapest possible repulsion (no O(n²) pairs, no InfoNCE denominator).
- *Shapes:* first-order geometry — don't clump. Supplies the "uniformity" half that contrastive negatives would otherwise provide (Q36); also improves retrieval-style tasks.

**4. L_Gram — second-order structure anchoring (v3's addition, late phase):**
`L_Gram = ‖X_S·X_Sᵀ − X_G·X_Gᵀ‖²_F`
- X_S = P×d matrix of ℓ2-normalized student patch features of a global crop → X·Xᵀ is the P×P matrix of **pairwise patch cosine similarities** (the Gram matrix). X_G = same from the **Gram teacher**, a periodic snapshot (refreshed every 10k iters, becomes the main EMA teacher).
- Key subtlety: it constrains **relations, not features**. Features may keep improving/rotating globally as long as which-patch-resembles-which is preserved. That's why it can be added late without freezing progress.
- *Shapes:* second-order geometry — patch similarity structure must not drift. *Failure it fixes:* over very long training (7B model, ~1M+ iters) the CLS-level objectives dominate and **dense feature maps degrade** (noisy patch similarities, worse segmentation); Gram anchoring restores/preserves them.

**Side-by-side (the interview table):**

| | L_DINO | L_iBOT | L_KoLeo | L_Gram |
|---|---|---|---|---|
| level | image ([CLS]) | patch | image ([CLS]) | patch-pairs |
| compares | student crop v ↔ teacher other global | student masked i ↔ teacher same i | student fᵢ ↔ nearest fⱼ in batch | student Gram ↔ snapshot Gram |
| views | cross-view | same view | — (batch) | same view (global) |
| cross-image? | no | no | **yes** | no |
| target from | EMA teacher (SK+sharpen) | EMA teacher (SK+sharpen) | none (repulsion) | Gram teacher (10k snapshot) |
| type | CE on prototypes | CE on prototypes | entropy estimator | Frobenius regression |
| role | global semantics | dense features | anti-clumping / uniformity | dense stability at scale |
| weight | 1 | 1 | 0.1 | w_G, from 1M iters |

**One-liner:** DINO and iBOT are the same teacher→student CE on prototype distributions — once on [CLS] across views (global semantics), once on masked patches within a view (local context); KoLeo is a nearest-neighbor entropy bonus spreading [CLS] features across the batch (the only cross-image force); Gram is a Frobenius match of the student's patch-similarity matrix to a snapshot teacher's (relations, not features) — two losses learn, two losses shape.

📄 Related: Q36 (total loss + weights), Q38 (iBOT deep dive), Q37 ([CLS]), Q35 (SK centering) · Papers: DINOv3 (2508.10104), iBOT (2111.07832), KoLeo/spreading vectors (1806.03198).

---

## Q40 — DINOv3 is great at structure/detail, SigLIP at semantics — can you combine them into a better visual encoder?

**Big idea:** Yes — and it works because each model's weakness is **objective-level, not incidental**. SigLIP's sigmoid image–text loss only constrains the *global* embedding to match caption semantics; captions rarely describe layout, orientation, counts, or fine texture, so nothing forces those into the features — the "CLIP-blind pairs" of *Eyes Wide Shut* (MMVP): image pairs CLIP/SigLIP embed nearly identically that DINO features separate easily. DINOv3's DINO + iBOT (+ Gram) losses supervise every patch token for spatial consistency (Q36, Q38) but tie nothing to language. Complementary supervision → real gains from combining. Three recipes:

**Recipe 1 — Feature fusion (ensemble of encoders; easiest, most common).**
Run both, interpolate to a common patch grid, normalize each (features live at very different scales), channel-concat, one projector into the LLM:
$$h_i = W\,[\,\mathrm{LN}(f^{\mathrm{DINO}}_i)\,;\,\mathrm{LN}(f^{\mathrm{SigLIP}}_i)\,]$$
- *Eyes Wide Shut*: adding DINOv2 features to CLIP fixes MMVP failures in a VLM.
- **Cambrian-1**: SVA — learnable queries cross-attend to CLIP + SigLIP + DINOv2 + ConvNeXt.
- **Eagle** (NVIDIA): systematic fusion study; **simple channel concat beats sequence-append and interleaved fusion**.
- DeepSeek-VL: same logic with SigLIP (semantics) + SAM-B (high-res detail).
- Costs: 2× vision FLOPs; patch-size mismatch (interpolate grids); per-encoder normalization is load-bearing.

**Recipe 2 — Multi-teacher distillation into one backbone (the "better single encoder" answer).**
**AM-RADIO** (NVIDIA): distill CLIP + DINOv2 + SAM into one student with per-teacher heads —
$$\mathcal{L} = \sum_{T} \lambda_T\Big[\,1-\cos\big(g_T(s_{\mathrm{cls}}),\,T_{\mathrm{cls}}\big) + \tfrac{1}{N}\sum_i \big\|g_T(s_i)-T_i\big\|\,\Big]$$
(summary-token matching + per-patch spatial matching). The student **matches or beats each teacher on its home turf**, one forward pass at inference. This is the answer when the constraint is serving cost. Same family: Theia (robotics), UNIC.

**Recipe 3 — Joint objective / graft language onto DINO.**
- One tower, both losses: **SILC** (contrastive + local self-distillation → better dense tasks), **EVA** (MIM with CLIP features as targets — MIM and CLIP semantics in one backbone).
- Post-hoc, LiT-style: **freeze DINOv3, train only a text encoder against it** (dino.txt). The DINOv3 paper ships exactly this — competitive zero-shot classification while keeping the dense features. Cheap: image tower frozen.

| recipe | inference cost | training cost | when to pick |
|---|---|---|---|
| fusion (concat/SVA) | 2 encoders | low (projector only) | research default; hard baseline |
| distillation (RADIO) | 1 encoder | high (retrain student) | serving-cost bound |
| LiT / dino.txt | 1 encoder + text tower | low (text tower only) | want zero-shot from DINO features |

**Interview flavor:**
- **The counterargument to know:** Meta's **Perception Encoder** (2025) argues a single well-trained contrastive model already *contains* strong dense features at intermediate layers — with alignment tuning you may not need the ensemble; the global/dense dichotomy may be partly a *readout* artifact, not an objective one.
- **Projector-bottleneck trap:** concat 2D→D then immediately compress and you can discard exactly what you added; Cambrian's SVA and Eagle's channel concat deliberately keep capacity at the fusion point.
- Scaling angle: semantic gaps between fused and single encoders shrink as LLM/data scale grows, but the **spatial/grounding gains persist** — that's why Cambrian/Eagle-style fusion survives at scale.

**One-liner:** yes — SigLIP's loss never asks for geometry and DINOv3's never asks for language, so fusing them (channel-concat à la Eagle/Cambrian), distilling both into one student (AM-RADIO), or LiT-ing a text tower onto frozen DINOv3 (dino.txt) each yield an encoder with both semantics and structure; know Perception Encoder as the "one good contrastive model may suffice" rebuttal.

📄 Papers: [Eyes Wide Shut / MMVP (2401.06209)](https://arxiv.org/abs/2401.06209), [Cambrian-1 (2406.16860)](https://arxiv.org/abs/2406.16860), [Eagle (2408.15998)](https://arxiv.org/abs/2408.15998), [AM-RADIO (2312.06709)](https://arxiv.org/abs/2312.06709), [SILC (2310.13355)](https://arxiv.org/abs/2310.13355), [EVA (2211.07636)](https://arxiv.org/abs/2211.07636), [SigLIP (2303.15343)](https://arxiv.org/abs/2303.15343), [Perception Encoder (2504.13181)](https://arxiv.org/abs/2504.13181) · Related: Q36 (DINOv3 losses), Q38 (iBOT dense supervision).

---

## Q41 — In the iBOT loss, what exactly is "patch i, student"? What does "input was [MASK]" mean concretely?

**Two different tensors, don't conflate them:**
- **Input side:** at masked position i, the student's input is NOT the patch's pixels. The pixel-derived embedding `z_i` (from patchify) is thrown away and **replaced by a single learned vector** `e_[MASK] ∈ ℝ^D` — an `nn.Parameter`, built exactly like the [CLS] token (Q37). The image itself is never blacked out; the swap happens in embedding space, post-patchify, pre-encoder.
- **Output side:** "patch i, student" in the loss = `h_i`, the **encoder's output at position i** — what the [M] slot has become after L rounds of attending to the visible patches. That's what goes into the iBOT head → `P_s^(i)`.

**The code (this is the whole trick):**
```python
self.mask_token = nn.Parameter(torch.zeros(1, 1, D))   # ONE vector, shared

z = patch_embed(img)                   # (B, 196, D) — from real pixels
z[mask_bool] = self.mask_token         # replace embeddings at masked positions
z = torch.cat([cls_tok, z], dim=1) + pos_embed         # pos added AFTER masking
h = encoder(z)                         # h[:, i] = "patch i, student"
P_s = softmax(ibot_head(h[:, 1:][mask_bool]) / tau_s)  # only masked slots
```

**Three details that matter:**
1. **One shared vector for all masked slots, all images.** If 50 patches are masked, all 50 slots contain the *same* `e_[MASK]` at input. What distinguishes them is only the positional embedding: slot i = `e_[MASK] + pos_i`. So the input says "something is hidden *here*" — where, not what.
2. **The content is manufactured by attention, not by the input.** At layer 1 the [M] slot's query (built from `e_[MASK] + pos_i`) pulls in features from visible neighbors; by layer L, `h_i` is a context-built hypothesis of what patch i should contain — a soft "inpainted" token (Q34). All information in `h_i` came from *other* positions.
3. **`e_[MASK]` is itself trained.** Gradients flow into it through every masked slot; it learns to be a good universal "please infer this position" query. A learned token also lets the model distinguish "masked" from a genuinely black/uniform patch — which zeroing the embedding would confuse.

**Teacher side, for contrast:** teacher gets the unmasked view, so its position i embedding comes from **real pixels**; its `h_i^t` → patch head → SK-center + sharpen → `P_t^(i)` = the "answer key" for the CE.

**BERT/MAE placement:** this is literally BERT's [MASK] mechanics transplanted to patch embeddings. Contrast **MAE**, which *deletes* masked positions from the encoder input entirely (encoder sees only ~25% visible tokens, cheap) and reintroduces mask tokens only in a light decoder. iBOT keeps placeholders in the full-length sequence because student and teacher outputs must stay position-aligned for the per-position CE.

**One-liner:** "input was [MASK]" = the patchify embedding at position i is replaced (pixels untouched) by one shared learned D-dim vector plus that position's pos-embed; "patch i, student" = the encoder's output at that slot — a prediction assembled entirely from visible context — whose head distribution is CE'd against the teacher's view of the real patch.

📄 Related: Q38 (iBOT loss), Q37 ([CLS] — same nn.Parameter construction), Q34 (attention builds the content), Q33 (patchify) · MAE contrast: arXiv 2111.06377.

---

## Q42 — Why doesn't DINO use mutual information to avoid representation collapse?

**The reframe that makes this a great answer:** collapse *is* an information-theoretic event — constant output ⇔ I(X; Z) = 0 ⇔ H(Z) = 0. So "maximize MI" is the right *objective*; the question is really about the *estimator*. DINO's designers rejected MI **estimation**, not MI — and in fact the DINO machinery secretly implements an MI maximization in decomposed form.

**Why direct MI estimation is a bad deal (the three standard reasons):**
1. **The log-n curse.** The tractable MI lower bound is InfoNCE, and it saturates at `log(batch size)` (Poole et al., "On variational bounds of MI"). Want to certify ≥ 11 nats of MI → need ~60k negatives. Worse, McAllester & Stratos: *any* distribution-free high-confidence MI lower bound needs samples exponential in the MI value. That's why SimCLR-era contrastive methods needed 4k–8k batches — exactly the dependency DINO was built to escape.
2. **Estimator pathology.** Neural estimators (MINE, NWJ) have variance that grows exponentially with true MI; adversarial/critic training is unstable at scale.
3. **MI underdetermines geometry.** MI is invariant under bijections — an encoder followed by any invertible scrambling has identical MI but useless geometry. Tschannen et al. ("On MI maximization for representation learning"): the success of "MI-based" methods comes from estimator bias + architecture, not the MI value. Chasing the number is chasing the wrong thing.

**The punchline — DINO maximizes MI anyway, without an estimator.** For the discrete prototype assignment k:
`I(x; k) = H(𝔼ₓ[P(k|x)]) − 𝔼ₓ[H(P(k|x))]`  (marginal entropy − mean conditional entropy)
- **Centering / SK** pushes the *batch-mean* prediction toward uniform → maximizes the first term `H(𝔼[P])`. SK makes this exact: balanced assignments = max-entropy marginal (and SK's lineage — SeLa, Asano et al. — literally *derives* it from maximizing I(label; index)).
- **Sharpening (τₜ ≈ 0.05)** makes each *individual* target near-one-hot → minimizes the second term `𝔼[H(P)]`.
- Together: confident per sample, uniform on average = both halves of the MI, each controlled by a cheap closed-form operation, no critic, no negatives, no log-n bound. The two collapse modes are exactly the two terms failing: one-prototype collapse (H(𝔼[P]) → 0, centering's job) and uniform collapse (𝔼[H(P)] → max, sharpening's job).
- **KoLeo** (Q39) covers the continuous side: collapse ⇔ feature entropy H(Z) low; KoLeo *is* a nonparametric entropy estimator (Kozachenko-Leonenko) being maximized directly.

**So the honest one-line answer:** DINO doesn't *estimate* MI because high-dim MI estimation is statistically cursed (log-n saturation, exponential variance) and the MI number doesn't guarantee good geometry anyway — instead it decomposes the discrete MI into two entropies it can control exactly (SK centering ↑ marginal entropy, sharpening ↓ conditional entropy) and adds a direct entropy estimator (KoLeo) on the features. Information-theoretic goal, estimator-free implementation.

**Contrast column (methods that do go the explicit-MI route):** CPC/InfoNCE (contrastive, log-n bound), Deep InfoMax (critic-based), Barlow Twins/VICReg (Gaussian-surrogate decorrelation ≈ redundancy reduction), MCR² (coding-rate). All pay either the batch tax, the critic tax, or the Gaussian-assumption tax; DINO's CE-on-soft-targets pays none and scales to 7B/1.7B images (DINOv3).

📄 Papers: Poole et al. (1905.06922), McAllester & Stratos (1811.04251), Tschannen et al. (1907.13625), SeLa (1911.05371), DINO (2104.14294) · Related: Q35 (centering/SK), Q36 (no contrastive terms), Q39 (KoLeo as entropy estimator).

---

## Q43 — What exactly is RMSNorm? Visualize it and implement it.

**Definition (one token at a time).** For one token's feature vector `x ∈ ℝ^d` — never the batch, never other tokens:

`RMS(x) = √(1/d Σᵢ xᵢ² + ε)`,  `yᵢ = γᵢ · xᵢ / RMS(x)`

One scalar statistic per token (the root-mean-square of its d features), divide by it, apply learned per-feature gain γ (init 1). Versus LayerNorm it deletes the mean subtraction (−μ) and the bias (β): LN = re-center + re-scale; RMSNorm = re-scale only (Zhang & Sennrich 2019).

**The geometric picture.** `RMS(x) = ‖x‖₂/√d`, so `x/RMS(x) = √d · x/‖x‖`. RMSNorm is a **radial projection onto the hypersphere of radius √d**: direction kept exactly, magnitude forced constant (then γ stretches per axis). LN does one extra step first — project onto the hyperplane `Σxᵢ = 0`, *then* onto the sphere. Read the invariances off the picture:
- `x` and `αx` lie on the same ray → same output (**scale-invariant**, `f(αx)=f(x)`) — this is the part that stabilizes training (controls activation/gradient magnitude; gradient ⊥ x).
- `x + c·1` changes the ray → different output (**not shift-invariant**) — the invariance RMSNorm gives up, argued near-redundant because the next linear layer absorbs offsets and in high-d the mean is small vs the scale.
- Output always satisfies `RMS(y)=1`, `‖y‖=√d` → residual stream enters every sublayer at fixed magnitude.

**Implementation (LLaMA-style — the interview version):**

```python
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))    # gamma only — no beta

    def forward(self, x):                            # x: (B, T, d)
        x32 = x.float()                              # stats in fp32 (bf16-safe)
        inv = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * (x32 * inv).type_as(x)
```

NumPy core: `y = g * x / np.sqrt((x**2).mean(-1, keepdims=True) + eps)`.

**Probe-depth details:** (1) fp32 inside because squaring bf16 loses precision / can overflow — compute the statistic in fp32, `type_as` back (LLaMA does exactly this). (2) ε inside the sqrt, 1e-6 (LLaMA) or 1e-5. (3) Cost win: one reduction instead of two (no mean pass), no subtraction, d fewer params (no β) — norm kernels are memory-bound, so ~10–60% faster, and it matters at LLM scale. (4) Placement: pre-norm `x + Sublayer(RMSNorm(x))` in LLaMA/Mistral/Qwen; Gemma also norms sublayer outputs. `torch.nn.RMSNorm` since PyTorch 2.4. (5) Why it matches LN: well-motivated hypothesis (re-scaling invariance is the useful half) + strong empirical parity — LLaMA, T5, Gemma, Qwen, DeepSeek all ship it.

**One-liner:** RMSNorm normalizes each token's feature vector by its root-mean-square — a radial projection onto the √d-sphere with a learned per-feature gain — keeping LayerNorm's magnitude control (the part that matters) while dropping the mean-centering (the part that doesn't), one reduction cheaper.

📄 Paper: Zhang & Sennrich, "Root Mean Square Layer Normalization" (arXiv 1910.07467) · Related: Day 1 §4.2 (LN vs BN, why re-scaling is the useful half), Coding_Implementations §6.

---

## Q44 — Explain this stable cross-entropy line by line (why `logits − logsumexp(logits)`?)

```python
def cross_entropy(logits, targets):
    # logits: (B, C), targets: (B,) int class indices
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)  # log-softmax, stable
    return -logp[torch.arange(len(targets)), targets].mean()
```

**Line 1 is log-softmax via an identity — not `log(softmax(x))`.** Take the log of softmax and split the quotient:

`log softmax(z)_i = log(e^{z_i} / Σ_j e^{z_j}) = z_i − log Σ_j e^{z_j} = z_i − LSE(z)`

So log-softmax = **each logit minus the logsumexp of all logits**, one subtraction. Why written this way:
- Naive `log(softmax(z))` breaks in **both** directions: a large logit (~90 in fp32, ~11 in fp16) overflows `e^z` → `inf/inf = nan`; a very negative one underflows softmax to exact `0.0` → `log(0) = −inf`.
- `torch.logsumexp` applies the **max trick**: `LSE(z) = m + log Σ_j e^{z_j − m}`, `m = max_j z_j`. After subtracting `m` every exponent is ≤ 0 (exp can't overflow) and the largest term is `e^0 = 1` (sum ≥ 1, so log never sees 0). Mathematically identical, finite in floating point.
- Shapes: `dim=-1` reduces over classes; `keepdim=True` keeps `(B, 1)` so `(B, C) − (B, 1)` broadcasts the per-example normalizer across its row. Each row of `logp` is a valid log-distribution (row logsumexp = 0 ⇔ probs sum to 1).

**Line 2 is NLL via advanced indexing.** Hard-label CE collapses to `−log ŷ_c`, so per example we need `logp[b, targets[b]]`. `logp[torch.arange(B), targets]` pairs row indices `[0..B−1]` with each row's target column → `(B,)` vector of "log-prob assigned to the true class"; negate (log-probs ≤ 0, we minimize) and mean: `L = −(1/B) Σ_b logp[b, y_b]`. The `gather` spelling generalizes to extra dims (e.g. `(B, T, V)` next-token loss): `-logp.gather(-1, targets[..., None]).squeeze(-1).mean()`.

**Probe-depth follow-ups:** (1) gradient w.r.t. logits of the whole function is `(softmax(z) − onehot(y))/B` — the ŷ−y identity, which is why the fused backward is one subtraction. (2) `F.cross_entropy` = exactly this + `ignore_index` (pad masking) + optional label smoothing/weights, and it expects **raw logits** — softmaxing first is the classic double-softmax bug. (3) Sanity checks: `logp.logsumexp(-1)` ≈ 0 per row; uniform logits ⇒ loss `= log C`.

**One-liner:** log-softmax is `z − logsumexp(z)` because the log of the softmax quotient splits into exactly that, and logsumexp's internal max-shift makes it overflow/underflow-proof; the second line just gathers each example's true-class log-prob and averages the negatives — which is all `F.cross_entropy` does.

Related: Day 1 §2.2 (ŷ−y identity), §2.5 (this implementation + visualizations), Q43-style stability pattern (compute the statistic stably, subtract).

---

## Q45 — Why two different LayerNorms (`ln1`, `ln2`) for attention and MLP in a block?

```python
self.ln1, self.ln2 = LayerNorm(d_model), LayerNorm(d_model)
...
x = x + self.attn(self.ln1(x), mask)   # ln1 → attention's view of the stream
x = x + self.mlp(self.ln2(x))          # ln2 → MLP's view of the stream
```

**It's not two kinds of norm — it's two instances with independently learned parameters.** The normalization math (subtract μ, divide σ) is parameter-free, but each `LayerNorm(d_model)` owns a learned per-feature gain γ and bias β (2·d each). Separate instances because:

1. **Different inputs, different statistics.** `ln1` sees raw `x`; `ln2` sees `x + attn_out` — a different distribution (the residual stream's norm grows and its direction rotates as each sublayer writes into it). Each norm's γ, β adapt to the statistics at its own point in the stream.
2. **Each sublayer gets its own learned "read" of the residual stream.** In pre-norm, `LN(x)` is *everything* the sublayer sees, so γ acts as a per-feature gate: attention's γ₁ can amplify the features worth mixing across tokens, while the MLP's γ₂ selects a different subspace for per-token processing. Tying them forces both sublayers to read the stream with the same feature weighting.
3. **Sharing one module = weight tying, not code dedup.** One shared `LayerNorm` called twice means the same γ, β with gradients summed from both call sites — an actual modeling constraint, empirically worse. (With `elementwise_affine=False` the norm is stateless like `nn.ReLU`, and reusing one instance *would* be fine — the learned params are exactly why it isn't here.)
4. **Cost is negligible.** 4·d params per block vs ~12·d² in the attn+MLP weights — for d=4096 that's 16K vs ~200M. Sharing saves nothing measurable.
5. **Every production model does this:** GPT-2 `ln_1`/`ln_2`, LLaMA `attention_norm`/`ffn_norm` (RMSNorms), Gemma 2 goes further with pre- *and* post-norms (4 per block).

**One-liner:** LayerNorm carries learned per-feature gain/bias; the two norms sit at different points of the residual stream (different statistics) and feed different consumers (attention vs MLP), so each gets its own parameters — sharing would be weight-tying two unrelated roles to save 4·d params out of ~12·d².

Related: Coding_Implementations §6 (LN/RMSNorm impl) & §7 (this block + new visualization), Q43 (RMSNorm), Day 1 §4.2 (LN vs BN).

---

## Q46 — In `decode_step`, why `q = (x_t @ Wq)[None, :]`? Isn't that already a vector?

```python
q = (x_t @ Wq)[None, :]   # (d_k,) -> (1, d_k)
```

**Yes — it's already a vector, and that's exactly the problem.** `x_t` is 1-D `(d_model,)`, so `x_t @ Wq` is 1-D `(d_k,)`. `[None, :]` inserts a leading axis, promoting it to a **(1, d_k) one-row matrix**. The data doesn't change; the *rank* does. Three reasons the 2-D shape is load-bearing:

**1. The cache invariant: K/V must be a (t, d_k) stack of rows, one per past token — from step one.** On the first decode step the cache is seeded directly with `K = k`. If `k` were 1-D, the cache would start life as a vector and every op that assumes "axis 0 = tokens" breaks:
- `np.sqrt(K.shape[1])` → `IndexError` immediately (a 1-D array has no second axis).
- In the sliding-window variant, `K[-window:]` on a 1-D `K` slices the last `window` **features of one key** instead of the last `window` **tokens** — a silent shape bug, the worst kind.

**2. The attention algebra stays identical to the full/batched formula.** With 2-D `q`:
`q @ K.T` → `(1, t)` — literally row $t$ of the full attention matrix $\mathrm{softmax}(QK^\top/\sqrt{d_k})$, just with a 1-row $Q$. `softmax(axis=-1)` is the same code, `A @ V` → `(1, d_v)`, and the final `(A @ V)[0]` strips the dummy row back to `(d_v,)`. One code path, no first-step special case.
With 1-D `q` it *mostly* still runs — `q @ K.T` gives `(t,)`, `A @ V` gives `(d_v,)` — but then `(A @ V)[0]` silently returns a **scalar** (the first feature) instead of the output vector. Everything downstream gets garbage with no error.

**3. NumPy 1-D arrays have no row/column identity.** `.T` on a 1-D array is a no-op; `vstack` guesses orientation for you. `[None, :]` is the explicit statement "this is one *row* of a token-major matrix" — the same reason math notation distinguishes $x \in \mathbb{R}^d$ from $x^\top \in \mathbb{R}^{1 \times d}$.

Equivalent spellings: `(x_t @ Wq).reshape(1, -1)`, `np.atleast_2d(x_t @ Wq)`, or projecting after promotion: `x_t[None, :] @ Wq`.

**One-liner:** `[None, :]` doesn't reshape data, it changes the object's rank from "a vector" to "a 1-row matrix" so the KV cache is token-major 2-D from its very first entry — `vstack`, `[-window:]`, `.shape[1]`, and the batched attention formula then all mean what they say, with no step-1 special case.

Related: mock_interview/study.html Stage 4 (KV-cache decoding), Q44 (shape-driven line-by-line reading), Coding_Implementations §7.

---

## Q47 — What does indexing with `None` actually mean?

```python
x = np.arange(5)     # shape (5,)
x[None, :]           # shape (1, 5)
```

**`None` inside square brackets is `np.newaxis`** — literally the same object: `np.newaxis is None` → `True`. It does not *select* anything; it **inserts a brand-new axis of length 1** at that position in the result.

**The mental model: an index tuple is consumed left-to-right against the array's axes.** Each element of the tuple does one of three things:

| index element | existing axes consumed | axes in output |
|---|---|---|
| `:` (slice) | 1 | 1 (kept, possibly shortened) |
| integer `i` | 1 | 0 (axis removed — this is why `x[0]` drops rank) |
| `None` | **0** | **1 (new, length 1)** |

So the rank arithmetic is `out.ndim = in.ndim − (#integers) + (#Nones)`, and **where the `None` sits in the tuple is where the new axis appears**:

```python
x.shape == (5,)
x[None, :]         # (1, 5)   new axis first  -> a row matrix
x[:, None]         # (5, 1)   new axis second -> a column matrix
x[None, :, None]   # (1, 5, 1)
x[None]            # (1, 5)   trailing ':' is implicit
x[..., None]       # (5, 1)   Ellipsis eats "all remaining axes", None appends at the end
```

**It's a zero-copy view** (no data movement, just new strides metadata) — exactly equivalent to `np.expand_dims(x, axis)` or the right `reshape`. `np.newaxis` is the self-documenting spelling; `None` is the terse one. PyTorch supports the identical syntax: `t[None]` ≡ `t.unsqueeze(0)`.

**Why you constantly want a size-1 axis** — two reasons:

1. **Rank promotion** (Q46): turn a `(d_k,)` vector into a `(1, d_k)` row so it obeys matrix code paths (`vstack`, `.shape[1]`, `@`-as-matmul).
2. **Broadcasting setup**: size-1 axes stretch to match. The classic idiom — all pairwise differences:
```python
x[:, None] - x[None, :]        # (5,1) - (1,5) -> (5,5), diff[i,j] = x[i] - x[j]
```
The same trick builds a causal mask (`np.arange(T)[:, None] >= np.arange(T)[None, :]`), distance matrices, and outer products.

**Traps:**
- `x[None]` vs `x[0]` are opposites: one adds an axis, the other removes one.
- `None` in *indexing* has nothing to do with `None` as a Python value elsewhere — inside `[]`, NumPy's `__getitem__` special-cases it as newaxis.
- Since it's a view, writing through it writes the original.

**One-liner:** in an index, `None` is `np.newaxis` — it matches no existing axis and instead injects a length-1 axis at exactly that slot (ndim: −1 per integer, +1 per `None`), a zero-copy way to promote rank or stage broadcasting.

Related: Q46 (why `decode_step` promotes q/k/v with `[None, :]`), Q12 (outer product — same broadcasting idiom), mock_interview/study.html Stage 4.

---

## Q48 — How do CLIP / DINO / JEPA relate to unified MMU+MMGen ("omni") models, diffusion, and RAE?

**Short answer:** three representation programs — contrastive (CLIP/SigLIP: global, language-aligned semantics), self-distillation SSL (DINOv1–v3: dense, spatial, language-free), and latent prediction (I-JEPA/V-JEPA 2: predictive world models) — used to feed *separate* consumers, but are converging into one stack. VLMs (MMU) use CLIP/SigLIP (+DINO for spatial gaps) as eyes; generation (MMGen) historically ran on semantically empty reconstruction latents (VAE/VQ, ~8% linear probe), which is why unified models struggled. **REPA** (align DiT features to DINOv2 → ~17.5× faster convergence) and **RAE** (replace the VAE with a frozen DINOv2/SigLIP encoder + trained decoder; diffuse in that semantic latent → SOTA FID; RAEv2 / MeanFlow-RAE in 2026) close the loop: *understanding's encoders become generation's latent space*.

**Unified design space (know the four patterns):** (A) discrete AR everything — Chameleon, Emu3 (tokenizer is the bottleneck); (B) decoupled und/gen encoders, one AR core — Janus(-Pro); (C) AR + diffusion in one transformer — Transfusion, BAGEL (MoT, emergent editing); (D) LLM + external diffusion decoder via queries — MetaQuery/MetaMorph, GPT-4o image gen / Nano Banana in production flavor. **Omni** adds audio/speech (RVQ codec tokens = the VQGAN of audio) with streaming Thinker-Talker splits (Qwen-Omni), full-duplex (Moshi). **VLA** is the same convergence pointed at actions: π0 generates action chunks with the *same flow-matching loss* FLUX uses for pixels; OpenVLA's encoder is DINOv2+SigLIP; V-JEPA 2-AC plans by latent MPC (world model instead of policy).

**Synthesis sentence:** a truly omni model is an AR reasoning core over a shared semantic latent space (built by CLIP/DINO/JEPA-style pretraining) with diffusion/flow decoders realizing any output modality — pixels, audio, or actions; JEPA is that same machinery pointed at the future. Full treatment with equations, figures, and both-sides arguments: **VLM_VLA_Unified_Omni.md / .html** (new deep-dive doc).

---

## Q49 — Is representation learning just (1) world models (JEPA), (2) language alignment (CLIP), (3) generation (diffusion)? How do they unify?

**Yes, with two sharpenings.** (a) CLIP is not a language model — it's **cross-modal alignment**: it distills the abstraction humans already encoded in language; the LM proper is a fourth object that becomes the reasoning core. (b) The three camps are really points on three orthogonal axes: **target space** (pixels vs representation), **supervision source** (self vs cross-modal), **uncertainty modeled** (contrastive vs regression vs full distribution). JEPA = latent + self + regression; CLIP = latent + cross-modal + contrastive; diffusion = data + self + full distribution; DINO = latent + self + invariance. One template covers all: L = E D(f_θ(context), τ(target)) — choose the (context, target) pair, target transform τ (identity / EMA encoder / other-modality encoder), and divergence D (InfoNCE / L2 / NLL-score).

**Entropy-budget framing (memorize):** generation models **all bits** (densest signal, zero abstraction); JEPA models **only the predictable bits** (abstraction free; but a point-estimate predictor can't represent multimodal futures and can't emit outputs); CLIP models **only the nameable bits** (semantic; blind to what captions omit).

**Unification routes (all real):** (1) loss stacking on the encoder — SigLIP 2, DINOv2 (iBOT), AIMv2; (2) representation bridging — run generation *inside* the SSL latent: REPA/RAE, DINO-WM, V-JEPA 2-AC; key identity: **latent diffusion over SSL features ≈ JEPA + a distribution** (fixes multimodal futures, keeps abstraction, decoder optional); (3) architectural — omni models: AR core absorbs alignment, diffusion/flow decoders realize outputs (incl. actions, π0), latent predictor = world model; (4) convergence at scale — Platonic Representation Hypothesis, Web-SSL.

**If at all:** *against* — LeCun: data-space generation is only needed for output, never for representation/planning; plus loss interference & tokenizer compromises in unified models. *For* — planning under uncertainty needs distributions over futures; generation is the densest supervision; interleaved tasks need all three. **Synthesis: unify the representation (one semantic latent), stack objectives on the encoder, model distributions as heads over that latent — pixel-space unification is the skippable part.** Full treatment: VLM_VLA_Unified_Omni §9.1 (Fig. 4).

---

## Q50 — How does JEPA do planning (LeCun's world model / V-JEPA 2-AC)?

**Short answer:** JEPA alone is not a planner — it yields an encoder and a latent predictor. Planning = wrap them in LeCun's world-model blueprint: **perception** (frozen JEPA encoder), **world model** (make the predictor *action-conditioned*: $\hat z_{t+1} = P_\phi(z_t, a_t)$), a **cost/energy** over imagined futures, and an **actor** that *optimizes actions at inference time* instead of being a trained policy.

**V-JEPA 2-AC concretely:** freeze the V-JEPA 2 encoder; train a ~300M action-conditioned predictor on ~62h unlabeled DROID robot video (teacher forcing + multi-step rollout loss, L1 in latent space). Then plan by **receding-horizon MPC with CEM**:

1. encode current frame → $z_t$ and goal *image* → $z_g$;
2. sample K action sequences $a_{t:t+H} \sim \mathcal N(\mu,\sigma)$;
3. roll each out through the predictor purely in latent space;
4. energy $E = \lVert \hat z_{t+H} - z_g \rVert_1$;
5. CEM: refit $(\mu,\sigma)$ on top-k elites, iterate;
6. execute **only the first action**, observe, replan.

$$a^* = \arg\min_{a_{t:t+H}} \lVert P_\phi(\cdots P_\phi(z_t,a_t)\cdots) - \bar s(x_{\text{goal}})\rVert_1$$

No reward, no RL, no task demos — the "policy" is inference-time energy minimization against learned dynamics, which is why it transfers zero-shot to new labs. **Why latent not pixels:** encoder already dropped the unpredictable bits (no blur tax for a deterministic predictor), latent distance ≈ semantic progress, rollouts don't render. Same recipe on DINOv2 features = **DINO-WM**.

**Gaps vs the full LeCun blueprint (limitations to volunteer):** no latent variable for multimodal futures (point-estimate predictor), hand-coded cost instead of learned critic, no H-JEPA hierarchy (flat short horizons because latent rollouts drift → replan every step), no Mode-1 distillation into a fast reactive policy; practical issues: camera-pose sensitivity, goal-as-image vs language, CEM cost in high-dim action spaces.

Full treatment with the MPC-loop figure: **VLM_VLA_Unified_Omni.md / .html §3.3**.

---

## Q51 — Can I build on LeWorldModel (LeWM) and add captioning by connecting it to an LLM?

**Yes — direct precedent + clean recipe; the caveat is what the latent supports.** LeWM (Mar 2026, LeCun/AMI line after LeJEPA): first end-to-end-from-pixels JEPA, two losses only (next-embedding MSE + SIGReg isotropic-Gaussian regularizer → collapse provably impossible), ~15M params, **one 192-d token per frame**, action-conditioned predictor, CEM planning; trained per-environment (Push-T, Reacher, OGBench-Cube).

**Precedent:** V-JEPA 2 → LLM alignment for video QA. **Scale mismatch:** LeWM's latent keeps only the control-relevant predictable bits of one environment → expect state-level captions, no cross-env transfer (mechanism study, not a general captioner). **Recipe (LLaVA pattern, 1 GPU):** freeze encoder; projector P: R^192 → LLM embeddings; caption NLL −Σ log p(w_t | w_<t, P(z_1:T)); 0.5–3B LLM + LoRA. 1 token/frame → a trajectory is only T tokens (cheap temporal captioning). **Free labels:** auto-generate program-checkable captions from simulator state (doubles as verifiable reward). **Do first:** linear-probe the latent for every fact you want captioned — if a probe can't read it, no LLM will. **SIGReg bonus:** Gaussian-regularized latent = well-conditioned projector source.

**Four experiments (ascending value):** (1) observed-trajectory captioning, program-checked; (2) **narrated imagination** — caption predictor rollouts under a plan; horizon-k caption accuracy = language-space world-model metric + interpretability; (3) **language-conditioned goals** — text → z_goal via inverse projector, CEM plans against it instead of a goal image; (4) **objective-interference test** — captioning as auxiliary loss *during* LeWM training: do nameable bits help or fight predictable bits (direct test of the Q49 entropy-budget thesis)? Watch: unfreezing the encoder distorts dynamics → planning degrades; track caption accuracy and planning success together.

Full treatment + pipeline figure: VLM_VLA_Unified_Omni §3.4 (Fig. 2b). Builds on Q50 (JEPA planning).

---

## Q52 — Why does the log-sum-exp term differentiate back into softmax itself?

**Short answer:** Because it's the chain rule on a `log` of a sum, and the pieces reassemble softmax exactly. Write CE for true class `c` as `L = −z_c + LSE(z)` where `LSE(z) = log Σ_j e^{z_j}`. Then

```
∂/∂z_i LSE(z) = (1/Σ_j e^{z_j})·(∂/∂z_i Σ_j e^{z_j}) = e^{z_i}/Σ_j e^{z_j} = ŷ_i
```

The `1/Σ` from the outer `log` is the softmax **denominator**; the surviving `e^{z_i}` (only the `j=i` term of the sum has nonzero derivative) is the **numerator** → the quotient *is* softmax. So **LSE is the antiderivative / scalar potential of softmax**: `∇_z LSE(z) = softmax(z)`. (LSE is the smooth relaxation of `max`; its gradient is the "soft-argmax.")

**Why it matters:** this is exactly why Method 1 of the `ŷ − y` derivation is clean — the loss splits into `−z_c + LSE(z)`, the LSE term regenerates `ŷ_i`, the linear `−z_c` term supplies `−y_i`, and the messy softmax Jacobian (`ŷ_k(δ_ki − ŷ_i)`) never has to appear. Same result Method 2 gets by explicit chaining, but for free.

Full treatment + LSE-vs-softmax figure: **Day1_Deep_Learning_Fundamentals §2.2 (Method 1)**. Related: Q on `logits − logsumexp` = log-softmax (Day1 §2.5).

---

## Q53 — How do modern VLMs self-improve, and what does a concrete data-engine project look like?

**One mental model:** every scheme is **Generate → Verify → Update**, closed at three timescales (inference-time best-of-n/self-critique; post-training STaR/ReST/iterative-DPO/RLVR; pretraining-time — this model curates/relabels/synthesizes the corpus for the *next generation*, the biggest compounding). Binding constraint: **verifier quality** — the loop mines the generator–verifier gap, and a closed loop with no external verifier only redistributes competence. Verifier hierarchy: programmatic (render-and-compare, OCR exact-match) > cross-modal consistency > model judge > heuristic.

**By stage:** vision = SSL EMA-teachers (DINO), SAM-style data engines, recaptioning flywheels (fuse, don't replace — CapsFusion), rendered-with-source synthetic (the program *is* the label → infinite verifiable supervision). Pretraining = DFN-style distilled filters, RHO learnability, DoReMi/RegMix mixtures (the optimal image:text:interleaved ratio is itself a scaling-law object). Post-training = rejection sampling, RLVR (strongest — programmatic verifiers), RLAIF/self-rewarding, hallucination DPO loops (POVID/HA-DPO), agentic rollouts with environment reward.

**Data-side zoom — six verbs:** FILTER, RELABEL, SYNTHESIZE, ACQUIRE, MIX/SCHEDULE, RECYCLE; value(x) ≈ quality × learnability × marginal-diversity × verifiability (selection on any single factor fails). Cross-cutting: dedup, decontamination, the **anchor ratio** (cap the model-touched fraction — real data is the entropy source).

**Mock project (self-improving MM web scraper, Gemini-setting):** five subsystems — bandit acquisition policy, tiered render/extract, fused-caption labeling arm, multi-signal value scoring (DFN-style distilled scorer + learnability + diversity, kept separate), weekly proxy-pretrain feedback loop. Research-taste trio among the ten problems: **P1** credit assignment (data value is scale-dependent → scaling ladder, validate *rank transfer* not scores), **P4** feedback-loop distribution collapse (data value is a *set function* → exploration budget, cluster caps; same math as RLHF entropy collapse), **P9** acquisition policy reward-hacks the scorer (Goodhart → asynchronous verifier refresh + KL cap on intake shift). One-theme summary: *every proxy breaks under optimization pressure; the real job is the verifier hierarchy and refresh schedule that keeps proxies honest.*

Full treatment (all ten problems, figures, delivery notes): **Self_Improving_VLMs.md / .html**.

---

## Q54 — Debugging my own buggy RMSNorm / LayerNorm implementations

I wrote both from memory and made real mistakes. Logging so I re-look before coding rounds.

**RMSNorm bugs:** put `eps` outside the sqrt (`g/(eps + sqrt(...))`), wrote `x^2/x.shape(-1)` — three errors in one: `^` is XOR not power (`**`), no reduction over the feature dim (RMS = `sqrt(mean(x²))` = `sqrt(sum(x²)/d)`), and `x.shape(-1)` should be `x.shape[-1]`.

**LayerNorm bugs:** added `eps` *twice* and *outside* the sqrt (`np.std(...) + eps` then `+ eps` again). Correct is `(x−μ)/sqrt(var+eps)·g + b` with eps inside the sqrt.

**The three rules to remember:**
1. **eps lives inside the sqrt** — `sqrt(var+eps)`, `sqrt(mean(x²)+eps)`. It guards against `var→0`; outside it doesn't, and is a different quantity.
2. **The normalizer is a reduction over the last dim**, not elementwise — `mean(x**2, axis=-1, keepdims=True)`.
3. **LN re-centers + re-scales; RMSNorm only re-scales** (drops μ). Var/std default to population (`ddof=0`/`unbiased=False`), which is correct.

Corrected reference + ⚠️ callout folded into **Coding_Implementations_From_Scratch §6**.

---

## Q55 — Debugging my own softmax cross-entropy (forward + backward) in NumPy

Wrote `softmax_xent(logits, y) -> (loss, dlogits)` from memory. Loss was mostly right; the **gradient was fully broken**.

**Bugs:**
- Built `dlogits` from **raw logits** (`logits[arange, y].copy()`) instead of from **softmax probabilities** — and it came out shape `(B,)` instead of `(B, C)`.
- Then tried to 2D-index that 1-D array (`dlogits[arange, y] = ...`) → IndexError.
- **No division by `B`** — the loss is a mean, so the grad must be averaged too.
- Loss `np.log(np.exp(logits).sum(-1))` has **no max-subtraction** → overflow.

**Correct pattern (`ŷ − y`):**
```python
m = logits.max(-1, keepdims=True); exp = np.exp(logits - m)
p = exp / exp.sum(-1, keepdims=True)          # softmax (B, C)
loss = -np.log(p[np.arange(B), y]).mean()
dlogits = p.copy(); dlogits[np.arange(B), y] -= 1; dlogits /= B
```

**Remember:** the softmax-CE gradient is `(softmax(logits) − onehot(y)) / B` — a full `(B, C)` tensor built from **probabilities**, averaged by batch. I was differentiating the logits, not the softmax. Folded into **Coding_Implementations §1**. (Ties to the LSE-is-the-potential-of-softmax point in Q52.)

**Follow-up (2026-07-04) — I re-attempted this from scratch several times and hit a different bug each pass. The full list of traps, in the order I fell into them:**
1. **Differentiate the softmax, not the logits.** First `dlogits` was built from raw `logits` → wrong values *and* wrong shape `(B,)`.
2. **Keep log-softmax full-width `(B, C)`.** I collapsed `logp = logits[arange,y] − lse` to `(B,)`, so `p = exp(logp)` was only `p_y`. The gradient needs the whole distribution — compute `logp = logits − logsumexp` over *every* logit, gather `[arange,y]` only for the loss.
3. **`keepdims=True` rides on the reduction (`.sum`/`.mean`/`.max`), not on the function wrapped around it.** Two variants of this: (a) dropping it entirely → `(B,)` subtracted from `(B,C)` broadcasts on the wrong axis (silently wrong if `C==B`, else error); (b) putting it on `np.log(...)` → `TypeError`, `log` has no `keepdims`. Correct: `np.exp(z).sum(axis=-1, keepdims=True)`.
4. **`p − onehot` is an *indexed* subtraction** `dlogits[arange(B), y] -= 1`, NOT a scalar `dlogits -= 1` across the whole tensor.
5. **Divide the grad by `B`** — the loss is a mean, so the backward carries the same `1/B`.

**Final clean version:**
```python
def softmax_xent(logits, y):
    m = logits.max(axis=-1, keepdims=True)
    logits = logits - m
    B = len(logits)
    logp = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))  # full (B,C) log-softmax
    loss = -np.sum(logp[np.arange(B), y]) / B
    p = np.exp(logp)
    dlogits = p.copy()
    dlogits[np.arange(B), y] -= 1.0     # p - onehot
    dlogits /= B                        # grad of the MEAN
    return loss, dlogits
```
Meta-pattern across Q54/Q55: my recurring errors are all **shape/broadcasting discipline** (keepdims, full-width vs gathered) and **differentiating the wrong quantity** — worth a timed re-drill.

---

## Q56 — Debugging my own 2-layer MLP backprop (batched NumPy)

Wrote `mlp_grads(X, y, W1, b1, W2, b2)` returning the four param grads. Forward: `h_pre = X@W1+b1`, `h = relu(h_pre)`, `logits = h@W2+b2`, then `loss, dlogits = softmax_xent(...)` (dlogits already `/B`). Three bugs:

1. **`db2 = logits.sum(0)`** — used the forward activation instead of the incoming grad. Bias grad = `dlogits.sum(0)`.
2. **`dW1 = h_pre.T @ dhpre`** — used `h_pre` as layer-1 input. The input feeding `W1` is `X` → `dW1 = X.T @ dhpre`.
3. **ReLU backward `dh * np.where(dh>0, dh, 0)`** — doubly wrong: gated on `dh` instead of the pre-activation `h_pre`, and the positive branch returned `dh` (not `1`), so it squared the grad. Correct: `dhpre = dh * (h_pre > 0)`.

`dW2 = h.T @ dlogits`, `dh = dlogits @ W2.T`, `db1 = dhpre.sum(0)` were fine.

**The reusable rule — for every linear layer `out = in @ W + b`:**
```
dW  = in.T @ d_out       # (in_dim, out_dim)
db  = d_out.sum(axis=0)  # sum over batch
d_in = d_out @ W.T       # push back through linear
```
and an elementwise nonlinearity multiplies by its local derivative (**ReLU: mask on the pre-activation** `(z > 0)`). Same meta-pattern as Q54/Q55: my errors are **grad-quantity-vs-forward-quantity** confusions and **shape discipline**. Folded into **Coding_Implementations §14**.

---

## Q54 — What is a nat?

**The unit of information/entropy when you use the natural log** (base $e$) instead of base 2. 1 nat = $1/\ln 2 \approx 1.443$ bits; 1 bit = $\ln 2 \approx 0.693$ nats. Cross-entropy losses and KL divergences in ML are computed with $\ln$, so they're **in nats by default** — "loss 2.3 nats/token" means perplexity $e^{2.3} \approx 10$.

**The context it appeared in:** the best-of-n KL bound. Selecting the best of $n$ samples shifts the policy away from the base distribution by at most

$$\mathrm{KL}(\pi_{\text{BoN}} \,\Vert\, \pi_{\text{base}}) \;\le\; \log n - \tfrac{n-1}{n} \;\;\text{nats}$$

— about 2.35 nats around $n \approx 28$, ~2.5 nats at $n=32$. The point: best-of-n is a **bounded, logarithmically-growing** optimizer (doubling $n$ buys only ~0.69 more nats of drift), which is why it's the gentle, analyzable baseline — whereas RL with a weak KL penalty can move the policy arbitrarily far from the reference. Related: Post_Training_and_RL_Deep_Dive (BoN vs RL, reward hacking).

---

## Q55 — How does the crawler itself self-improve? (render-vs-extraction critic)

**Key idea: every page ships its own ground truth.** The rendered screenshot (what a human sees) vs the extracted interleaved doc (what the model eats) — any divergence is a pipeline bug, detectable with **no labels** because the page supplies both views. Render-and-compare turned *inward*: the verifier checks the pipeline, not the model.

**Mechanism:** a VLM critic takes screenshot tiles + extracted doc (+ DOM) and emits structured discrepancy reports — taxonomy code (MISSING / PHANTOM / ORDER / FIDELITY / COVERAGE / SEMANTIC), severity, bbox ↔ doc-span. **Stage attribution** routes each report before fixing: DOM-declared vs painted → rendering bug; DOM-text vs doc-text → extraction bug; screenshot vs captions → labeling bug.

**Loop A (fix the component):** template-clustered rule patching via code-gen + golden regression gates; distill critic corrections into a **screenshot-grounded extractor** (end state: extraction *is* a VLM reading the page image, DOM as hint); render-policy bandit rewarded by critic-score-per-render-dollar; extraction fixability as an acquisition-policy feature. **Loop B (fix the data):** severity-weighted quarantine/repair; **blast-radius backfill via lineage metadata** (extractor version + template hash + render config on every doc — lineage makes pipeline bugs reversible); confirmed failures graduate into an adversarial golden suite.

**New failure modes:** P11 shared blind spot (broken render → both views agree and both are wrong → multi-render disagreement + DOM-declared-resource checks; consistency verifies *agreement*, not truth); P12 critic cost → distillation cascade; P13 false positives on reachable-but-not-visible content (measure critic precision before automating); P14 genuinely ambiguous reading order (mark, don't force); P15 Goodhart round three — the extractor inherits the critic's blind spots → refresh critic each generation + critic-free human goldens. The closing flywheel: better VLM → sharper critic → better extractor → cleaner interleaved data → better VLM.

Full treatment + Fig. 3: **Self_Improving_VLMs.md / .html §3.6**.

---

## Q56 — Why the log-derivative trick? Can't we differentiate P(τ|θ) directly?

You *can* — the issue is what you're left with. $∇θJ = ∫ ∇θP(τ|θ)R(τ)dτ$ is a correct integral, but $∇θP$ is **not a density** (negative values, doesn't sum to 1), so the integral is not an expectation and **cannot be estimated by sampling** — you'd have to enumerate the exponential trajectory space. The trick $∇θP = P·∇θ\log P$ moves $P$ back into the density slot: $∫P·[∇θ\log P·R] = E_{τ\sim π_θ}[∇θ\log P·R]$ — same value, now a Monte Carlo average over rollouts you already run. **Both sides are integrals; only one is an expectation** (density × function), i.e. samplable. Bonus: log turns the trajectory product into a sum and the dynamics terms (no θ) vanish → model-free. Folded into **Post_Training_and_RL_Deep_Dive §2**.

---

## Q57 — Is REINFORCE == policy gradient? What else is in the family?

REINFORCE is the **vanilla policy gradient** — the bare MC estimator of $∇θJ$ with $Ψ = R(τ)$. "Policy gradient" also names the *family*: all members estimate the **same** $∇θJ$ and differ only in the weight $Ψ$ on $∇θ\log π$: $R(τ)$ (REINFORCE) → $R−b(s)$ (baseline) → $A(s,a)$ (A2C) → clipped-ratio·A (PPO). Analogy: REINFORCE : policy gradient :: plain SGD : gradient-based optimization. ($J(θ)=E[R(τ)]$ itself is just the **expected return / performance objective**; the letter J is control-theory convention for a cost functional — no eponymous name.)

---

## Q58 — Why can any action-independent baseline be subtracted without bias?

Because **the expected score is zero**: $E_{a\sim π}[∇θ\log π(a|s)] = ∇θΣ_aπ(a|s) = ∇θ1 = 0$. So $E[∇θ\log π·b(s)] = E_s[b(s)·0] = 0$ for *any* $b(s)$. Key clarifications: (1) the condition is **input signature** — $b$ must not take $a_t$ as an argument; a value head sharing the policy's transformer trunk is still legal since $V_φ(s_t)$ is computed from the prefix before $a_t$ is sampled; (2) **accuracy affects only variance, never bias** ($b=42$ is unbiased); (3) the payoff is recentering — the gradient signal becomes "better/worse than average for this state."

---

## Q59 — Why is reward-to-go the Q function and the baseline the V function?

Reward-to-go $\hat R_t = Σ_{t'≥t}r_{t'}$ is a single-sample MC estimate of $Q(s_t,a_t)$ — it's conditioned on the **action taken**, which the gradient weight must be (else it can't teach which action was good). The baseline must be action-**independent** (Q58), and the most informative legal choice is the action-average of Q — which *is* $V(s) = E_{a\sim π}[Q(s,a)]$. Difference = advantage $A = Q − V$: "how much better than the policy's average." The pairing is forced by the bias contract, not convention. Chain: $R(τ)$ →(reward-to-go)→ $Q$ →(subtract V)→ $A$ = actor-critic.

---

## Q60 — Why learn V with a NN but get Q/A from samples, and how is V trained?

Three reasons. (1) **Only V has an unbiased regression target:** train on pairs $(s_t, \hat R_t)$; each target is a Q-sample for the sampled action, but MSE regression converges to the **mean** of its targets = $E_a[Q] = V$. Nothing observable *is* an advantage sample. (2) **A learned action-dependent weight injects bias** — its errors correlate with actions (the illegal $b(s,a)$ case); the split keeps action-dependence in the unbiased sampled return and the learned part action-free. (3) **Identifiability + cost:** $Q=V+A$ is only identified under $E_a[A]=0$ — enforcing that *is* computing V; and a direct A-target needs rollouts of **every action from the same state** — $O(|A|)$ resets per state, exponential over the horizon. The value net **amortizes the branching** via regression-to-the-mean across states. Training: MSE to MC / TD(0) / **GAE** targets (the bias-variance dial), targets stop-gradiented per batch, linear value head on the shared trunk, value-clipping in PPO. **GRPO is the brute-force alternative made affordable:** LLM prompts reset for free + whole-response-as-one-action → group mean reward = empirical V, no critic. Folded into **Post_Training_and_RL_Deep_Dive §2**.

---

## Q61 — Where does DPO sit relative to the policy-gradient family? (the *PO landscape)

**Outside it.** Two axes: on-policy-ness × machinery. On-policy PG (PPO/TRPO/A2C with critic; GRPO/RLOO/REINFORCE critic-free) needs fresh rollouts; off-policy actor-critic (SAC/TD3/DQN) reuses replay *because* a bootstrapped critic can evaluate stale data; **DPO/IPO/KTO/SimPO/ORPO are fully offline and direct** — closed-form supervised loss on preference pairs, no RM, no critic, no rollouts. The correlation: top-left→bottom-right sheds machinery and on-policy sampling together (RLHF-PPO → GRPO → DPO). DPO and GRPO are **diagonal opposites** — both "simpler than PPO" in opposite directions. Family deltas: IPO regularizes; KTO takes unpaired labels; ORPO/SimPO drop the reference model. Landscape map in **Post_Training_and_RL_Deep_Dive §6**.

---

## Q62 — How do world-model methods (sample actions, reweight by reward) relate to policy gradient?

That procedure is **CEM/MPPI planning**: $q_{k+1}(a_{0:H}) ∝ q_k·\exp(R/η)$ — sample action sequences (in a *learned* model, "in imagination"), exponentially reweight by return, refit; the fixed point concentrates on high-return trajectories. It's the **orthogonal axis** the on/off-policy map is silent on: model-based (dynamics learned explicitly) vs model-free (dynamics cancelled out of $∇θ\log P$ — the REINFORCE derivation itself). Deep link (**RL-as-inference**): PG and reward-weighted refitting optimize the same $J=E[R]$ — first-order (differentiate) vs zeroth-order (reweight); the reweighting is the EM view whose gradient recovers the PG form. Hybrids: Dreamer/TD-MPC run actor-critic PG *inside* imagined rollouts. Directly relevant to the LatentFusion proposal — an LLM predicting in frozen JEPA latent space is a world model; acting with it = CEM-style latent planning or Dreamer-style amortization.

---

*Log started 2026-06-21. New Q&A appended below as asked.*
