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

*Log started 2026-06-21. New Q&A appended below as asked.*
