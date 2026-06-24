# Notes — Activation Functions (Q&A Deep Dive)

> Companion notes to Day 1 §1.3. Covers: the multivariate sigmoid (softmax), why sigmoid/tanh saturate, and ReLU vs GELU.
> GitHub renders the LaTeX below. For local viewing with rendered math, open the matching `.html` version.

---

## 1. The "multivariate sigmoid" = softmax

### Scalar sigmoid (binary)

$$\sigma(z) = \frac{1}{1+e^{-z}} = \frac{e^z}{1+e^z}, \qquad z \in \mathbb{R},\ \ \sigma(z)\in(0,1)$$

Maps one real number to a probability. Used for **binary** classification.

### Softmax (multiclass)

For a logit vector $\mathbf{z} = (z_1, \dots, z_K) \in \mathbb{R}^K$:

$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}, \qquad i = 1,\dots,K$$

Properties:
- Each component lies in $(0,1)$ and they **sum to 1** — a valid distribution over $K$ classes.
- **Shift-invariant:** $\text{softmax}(\mathbf{z} + c\mathbf{1}) = \text{softmax}(\mathbf{z})$ for any scalar $c$. This is why the numerically stable implementation subtracts $\max_j z_j$ before exponentiating.

### Softmax reduces to sigmoid when $K=2$

$$\text{softmax}(\mathbf{z})_1 = \frac{e^{z_1}}{e^{z_1}+e^{z_2}} = \frac{1}{1+e^{-(z_1-z_2)}} = \sigma(z_1 - z_2)$$

Binary softmax depends only on the logit **difference** and equals the sigmoid of that difference — which is why a 2-class softmax and a single-sigmoid classifier are equivalent.

### Elementwise sigmoid (multi-label) — a different object

$$\sigma(\mathbf{z})_i = \frac{1}{1+e^{-z_i}}$$

Applied **independently** per component. Outputs do **not** sum to 1 — each class is an independent yes/no (multi-label), not a single choice among classes (multiclass).

---

## 2. Why sigmoid and tanh saturate

**Saturation** = the output flattens against its asymptote, so the function is nearly constant there → the **local derivative is near zero**. That near-zero gradient is the problem.

### Sigmoid

$$\sigma(z) = \frac{1}{1+e^{-z}}, \qquad \sigma'(z) = \sigma(z)\,\big(1-\sigma(z)\big)$$

- $z \to +\infty:\ e^{-z}\to 0 \Rightarrow \sigma(z)\to 1$ (flat tail).
- $z \to -\infty:\ e^{-z}\to \infty \Rightarrow \sigma(z)\to 0$ (flat tail).

The derivative is a product of "distance from 0" and "distance from 1":

$$\sigma'(0) = 0.5 \times 0.5 = 0.25 \quad (\text{maximum}), \qquad \sigma'(z)\xrightarrow[|z|\to\infty]{} 0$$

So once $|z|$ is even moderately large, the unit sits on a flat tail, output pinned near 0 or 1, gradient ≈ 0 → **saturated**.

### Tanh

$$\tanh(z) = \frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}, \qquad \tanh'(z) = 1-\tanh^2(z)$$

- $z\to+\infty \Rightarrow \tanh\to 1$; $z\to-\infty \Rightarrow \tanh\to -1$ (both tails flatten).
- $\tanh'(0) = 1$ (maximum), and $\tanh'(z)\to 0$ as $\tanh\to\pm 1$.

Tanh improves on sigmoid in two ways — max gradient **1** (vs 0.25) and **zero-centered** output — but has the **same flaw**: flat tails → derivative → 0 for large $|z|$.

### Root cause (one-liner)

Both are **bounded squashing functions**: they compress all of $\mathbb{R}$ into a finite interval. Mapping an infinite domain into a finite range *must* flatten at the extremes, and a flat region means $\frac{d}{dz}\approx 0$. Saturation is the unavoidable price of squashing.

### Why it matters (vanishing gradients, Day 1 §2.4)

Backprop multiplies local derivatives layer by layer. Every sigmoid layer contributes a factor $\le 0.25$; a saturated layer contributes ≈ 0. Across many layers the product **vanishes** → early layers barely update. This is why **ReLU** (derivative exactly 1 on its active branch) took over, and why **normalization** helps — it keeps pre-activations near 0, the high-gradient region, away from the saturated tails.

---

## 3. ReLU vs GELU

### Definitions

$$\text{ReLU}(z) = \max(0, z) = \begin{cases} z & z > 0 \\ 0 & z \le 0 \end{cases}$$

$$\text{GELU}(z) = z \cdot \Phi(z)$$

where $\Phi(z)$ is the **standard Gaussian CDF** ($\Pr[\mathcal{N}(0,1) \le z]$). Common tanh approximation used in practice:

$$\text{GELU}(z) \approx 0.5\,z\left(1 + \tanh\!\Big[\sqrt{\tfrac{2}{\pi}}\big(z + 0.044715\,z^3\big)\Big]\right)$$

**Intuition:** ReLU makes a **hard** 0/1 gate on the sign of $z$. GELU makes a **soft, probabilistic** gate — it scales $z$ by the probability that a standard normal is below it ("how likely is this input to be kept").

### Side-by-side

| | **ReLU** | **GELU** |
|---|---|---|
| Formula | $\max(0,z)$ | $z\,\Phi(z)$ |
| Shape | Piecewise linear, hard kink at 0 | Smooth, curved |
| Differentiable at 0? | No (kink; use subgradient) | Yes, everywhere |
| Negative inputs | Hard zero | Small **negative** dip for slightly-negative $z$; → 0 for very negative |
| Derivative | $1$ if $z>0$, $0$ if $z<0$ (a mask) | Smooth; can be slightly negative; → 1 for large $z$, → 0 for very negative |
| Output range | $[0, \infty)$ | $\approx(-0.17, \infty)$ |
| Compute | Cheapest possible | More expensive (CDF / tanh / erf) |
| Dead neurons | Yes — can stick at 0 with zero gradient | Largely avoided (nonzero negative-side gradient) |
| Typical use | CNNs, classic MLPs, RL | Transformers (BERT, GPT) |

### Key differences

**1. Smoothness.** ReLU has a non-differentiable kink and a hard cutoff; GELU is smooth everywhere → smoother gradients/loss landscape, which helps optimization in deep transformers. GELU's main selling point.

**2. Negative region / "dying ReLU" (the big practical one).**
- ReLU: a unit whose pre-activation is **always negative** outputs 0 with gradient 0 → gets no signal and **never recovers** (dead neuron). A too-high LR or bad init can permanently kill part of a layer.
- GELU: passes a **small nonzero (slightly negative)** signal for negative inputs, so its gradient there is **not exactly zero** → units recover, capacity isn't lost the same way. The non-monotonic dip also makes GELU a bit more expressive than strictly-nonnegative ReLU.

**3. Compute.** ReLU = one comparison, essentially free. GELU evaluates a CDF (or tanh/erf approximation) — meaningfully pricier per element. Negligible at small scale; a real cost across trillions of activations at frontier scale (usually still worth it).

**4. Large $z$.** Both → derivative $1$ for large positive $z$ (no positive-side saturation — this is what they share, and why both beat sigmoid/tanh). They differ only near and below 0.

### Why transformers chose GELU

A "marginal gains compound at scale" story: GELU's smoothness and nonzero negative gradient give a **small but reliable** quality bump in large LMs, worth the extra FLOPs at scale. CNNs and most non-transformer nets keep ReLU for speed; the difference is usually negligible there.

> **In practice — the distinguishing symptom:** a large, *growing* fraction of exactly-zero activations in a ReLU layer that never recovers = dying ReLU → suspect LR/init, or switch to GELU/LeakyReLU. GELU essentially lacks this failure mode.

---

## Interview-ready summaries

- **Saturation:** "Sigmoid and tanh are bounded squashing functions, so outputs flatten at the extremes; a flat output means derivative → 0. Sigmoid's gradient maxes at 0.25, tanh's at 1, but both collapse to ~0 for large $|z|$. In deep nets, backprop multiplies these small factors across layers, so gradients vanish and early layers stop learning — hence ReLU and normalization."
- **ReLU vs GELU:** "ReLU is a hard, cheap gate `max(0,z)` with gradient 1 or 0; weakness is dead neurons. GELU `z·Φ(z)` is a smooth probabilistic gate passing a small nonzero signal for negatives, avoiding dead neurons and giving smoother optimization at higher compute cost. Both avoid positive-side saturation. Transformers use GELU because the small reliable gain compounds at scale."
