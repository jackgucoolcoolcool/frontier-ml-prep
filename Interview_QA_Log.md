# Interview Prep — Q&A Log

A running log of the technical questions I've asked and the answers, captured for review.
Newest entries are added at the bottom. Deeper write-ups are cross-linked where they exist.

**Index**
1. [Multivariate sigmoid function](#q1--write-down-the-multivariate-sigmoid-function)
2. [Why do sigmoid and tanh saturate?](#q2--why-do-sigmoid-and-tanh-saturate)
3. [Compare ReLU and GELU](#q3--compare-relu-and-gelu)
4. [What is GELU / what is Φ?](#q4--what-is-gelu-again-what-is-%CF%86)

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

*Log started 2026-06-21. New Q&A appended below as asked.*
