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

*Log started 2026-06-21. New Q&A appended below as asked.*
