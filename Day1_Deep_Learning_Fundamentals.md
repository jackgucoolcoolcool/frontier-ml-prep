# Day 1 — Deep Learning Fundamentals

**Target:** Research Scientist loop at a frontier lab (Meta TBD / MSL style).
**Mode:** Concepts & derivations first, light code, explanation + self-test Q&A.
**Time budget:** ~8 hours.
**The bar for today:** You should be able to *derive* backprop on a whiteboard, *explain why* every standard trick (residuals, normalization, AdamW, warmup, clipping) exists, and *reason through a training failure* out loud — "I see X, which means Y is likely wrong, so I'd look at Z."

> **How to use this doc.** Each section opens with a short "watch first" video pointer, then explanation + intuition, then a **In Practice** box (what this looks like in real training), and ends with **Self-test** questions with model answers. Don't just read the answers — cover them, say your answer out loud, then check. Talking is what the interview tests.

---

## Suggested 8-hour schedule

| Block | Time | Topic | Section |
|-------|------|-------|---------|
| 1 | 0:00–1:30 | Neural net mechanics: MLPs, nonlinearity, activations | §1 |
| 2 | 1:30–3:00 | Backpropagation, derived from scratch | §2 |
| — | 3:00–3:15 | Break | |
| 3 | 3:15–4:30 | Initialization & gradient flow | §3 |
| 4 | 4:30–5:45 | Normalization & residual connections | §4 |
| — | 5:45–6:00 | Break | |
| 5 | 6:00–7:15 | Optimizers, LR schedules, warmup, clipping, mixed precision | §5 |
| 6 | 7:15–8:00 | **The debugging playbook** + full self-test | §6, §7 |

The single highest-leverage parts for a research interview are **§2 (backprop)** and **§6 (debugging)**. If you fall behind, protect those two.

---

## §1 — Neural network mechanics

**Watch first (intuition layer):**
- 3Blue1Brown, *"But what is a neural network?"* (Ch. 1) — https://www.3blue1brown.com/lessons/neural-networks
- 3Blue1Brown, *"Gradient descent, how neural networks learn"* (Ch. 2) — https://www.3blue1brown.com/lessons/gradient-descent

### 1.1 What a layer actually computes

A single fully-connected (dense) layer is an **affine map followed by a nonlinearity**:

```
z = W x + b          # affine: linear map + shift
a = φ(z)             # elementwise nonlinearity
```

- `x ∈ R^{d_in}` is the input vector, `W ∈ R^{d_out × d_in}`, `b ∈ R^{d_out}`.
- `φ` is the activation function (ReLU, GELU, etc.).
- An **MLP** (multi-layer perceptron) stacks these: `a₁ = φ(W₁x+b₁)`, `a₂ = φ(W₂a₁+b₂)`, … and a final linear "head" producing logits.

### 1.2 Why the nonlinearity is non-negotiable

This is a classic warm-up question. **Without `φ`, a deep network collapses to a single linear layer.**

Compose two linear layers: `W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂)`. That is just `W'x + b'` — one affine map. Stacking 100 linear layers still gives you a linear function: it can only draw straight decision boundaries and cannot represent XOR, let alone language. **The nonlinearity is what lets depth buy you expressive power** (composition of simple nonlinear pieces → arbitrarily complex functions; this is the universal approximation intuition).

### 1.3 Activation functions — what and *why*

| Activation | Formula (intuition) | Why / when | Watch out for |
|---|---|---|---|
| **Sigmoid** | `1/(1+e^{-z})`, squashes to (0,1) | Historical; output gates, probabilities | Saturates → vanishing gradients; not zero-centered |
| **Tanh** | squashes to (−1,1) | Zero-centered version of sigmoid | Still saturates at extremes |
| **ReLU** | `max(0, z)` | Cheap, no saturation for z>0, sparse activations. Default for CNNs/MLPs | **Dead neurons** (stuck at 0 forever if always negative) |
| **GELU** | `z · Φ(z)` (Φ = Gaussian CDF) | Smooth, slightly negative for small negative z; **default in transformers** (BERT, GPT) | Slightly more compute than ReLU |
| **SwiGLU** | gated: `(W₁x) ⊙ swish(W₂x)` | Gating improves quality per-param; used in LLaMA, PaLM | Uses ~1.5× params for the FFN (3 matrices, not 2) |

**The key intuitions an interviewer wants:**

- **ReLU killed vanishing gradients** for positive activations: its derivative is exactly 1 for `z>0` (vs sigmoid's derivative which maxes at 0.25 and is near 0 when saturated). Stacking many sigmoids multiplies many <1 numbers → gradient vanishes. ReLU avoids that on the active path.
- **Why GELU/SwiGLU over ReLU in transformers?** They're smooth (everywhere differentiable) and empirically give a small but reliable quality bump at scale. SwiGLU's *gating* lets the network modulate information flow multiplicatively — more expressive per parameter, which matters when you're paying for every FLOP. This is a "marginal gains compound at scale" story.

> **In Practice — dead ReLUs.** If a chunk of your ReLU units output 0 for *every* input in a batch and their gradient is therefore 0, they never recover ("dead neurons"). Symptom: a layer's activation statistics show a large fraction of exact zeros that only grows over training; effective capacity silently drops. Common cause: learning rate too high caused a big negative bias update, or bad init. Fixes people reach for: lower LR, better init, or a leaky/GELU variant. **This is a good concrete example of the "see X → suspect Y → look at Z" pattern** you asked about: *see* rising fraction of zero activations → *suspect* dead ReLUs from too-aggressive updates → *look at* the LR and the bias gradients.

### Self-test §1
1. **Why do neural nets need nonlinearities?** → Because composing affine maps yields an affine map; depth only adds expressive power if a nonlinearity sits between layers. Without it the whole net is one linear function and can't model non-linear decision boundaries (e.g. XOR).
2. **Why did ReLU help training deep nets vs sigmoid?** → ReLU's gradient is 1 on the active branch, so it doesn't shrink gradients on the forward-active path; sigmoid's derivative is ≤0.25 and ~0 when saturated, so deep stacks multiply many small numbers and the gradient vanishes.
3. **Why do transformers use GELU/SwiGLU instead of ReLU?** → Smoothness + empirical quality gains at scale; SwiGLU adds multiplicative gating that is more expressive per parameter. The cost (more FLOPs/params) is justified by better loss-per-compute.

---

## §2 — Backpropagation, from scratch

**Watch first (the canonical explanation):**
- 3Blue1Brown Ch. 3 *"Backpropagation, intuitively"* and Ch. 4 *"Backpropagation calculus"* — https://www.3blue1brown.com/topics/neural-networks
- Andrej Karpathy, *"The spelled-out intro to backpropagation"* (micrograd) — https://karpathy.ai/zero-to-hero.html (watch the first ~hour for the derivation; the rest is coding)

Backprop is **just the chain rule applied efficiently to a computation graph**, reusing intermediate results so you compute all gradients in one backward pass instead of re-deriving each from scratch. If you can derive it for a 2-layer net, you understand it.

### 2.1 Setup: a 2-layer MLP for classification

Forward pass:
```
z₁ = W₁ x + b₁
a₁ = φ(z₁)
z₂ = W₂ a₁ + b₂          # logits
ŷ  = softmax(z₂)
L  = cross_entropy(ŷ, y) = −Σ_k y_k log ŷ_k
```
where `y` is a one-hot label. Goal: compute `∂L/∂W₁, ∂L/∂b₁, ∂L/∂W₂, ∂L/∂b₂`.

**Aside — what softmax and cross-entropy are (and the KL connection).**

**Softmax** turns logits into a probability distribution (all positive, sums to 1):
```
ŷ_i = softmax(z₂)_i = e^{z₂_i} / Σ_j e^{z₂_j}
```
**Cross-entropy (CE)** measures how well a predicted distribution `q` matches a true distribution `p`:
```
H(p, q) = −Σ_k p_k log q_k
```
In classification, `p = y` (one-hot label) and `q = ŷ`, so `H(y,ŷ) = −Σ_k y_k log ŷ_k = −log ŷ_c` (only the true class survives). Intuition: CE is the **expected surprise** (in nats) of the true labels under the model's predictions — large when the model assigns low probability to what actually happens; minimized when `q = p`.

**Is CE directional? Yes — not symmetric:** `H(p,q) ≠ H(q,p)`. The first argument `p` is the target you average over; the second `q` is the model distribution inside the log. ML always uses `H(true, model)`.

**Connection to KL divergence:**
```
H(p, q) = H(p) + D_KL(p ‖ q),    D_KL(p ‖ q) = Σ_k p_k log(p_k / q_k)
```
`H(p)` (entropy of the truth) is constant w.r.t. the model, so **minimizing CE = minimizing D_KL(p‖q)**. For one-hot labels `H(p)=0`, so **CE equals the KL divergence**. KL is directional too (`D_KL(p‖q) ≠ D_KL(q‖p)`); the forward KL minimized here is **mass-covering**. This is why next-token training = MLE = minimizing forward KL between data and model.

### 2.2 The one identity that makes it clean: softmax + cross-entropy

A beautiful, must-know result: **the gradient of softmax-cross-entropy w.r.t. the logits is just `ŷ − y`.**

```
∂L/∂z₂ = ŷ − y          # (predicted probabilities minus one-hot truth)
```

Why this matters: it's clean, numerically stable (which is why frameworks fuse `softmax` + `cross_entropy` into one op — see the In Practice box), and it gives the whole backward pass a simple starting point. Call this `δ₂ = ŷ − y` (the "error at the output").

**Derivation, two ways.**

*Method 1 — substitute softmax first (elegant).* With a one-hot label, `L = −log ŷ_c`. Substitute softmax and split the log:
```
L = −log( e^{z₂_c} / Σ_j e^{z₂_j} ) = −z₂_c + log Σ_j e^{z₂_j}
```
Differentiate w.r.t. `z₂_i`: the `−z₂_c` term gives `−1` at `i=c` else `0` (= `−y_i`); the log-sum-exp term differentiates back into softmax itself (`= ŷ_i`). Add → `∂L/∂z₂_i = ŷ_i − y_i`. Clean because log-sum-exp is the function whose gradient *is* the softmax.

*The chain rule (reminder).* It differentiates a composition by multiplying local derivatives along the path. Single-variable: if `L=f(u)`, `u=g(z)`, then `dL/dz = (dL/du)(du/dz)`. Multivariable: if `z_i` reaches `L` through several intermediates `ŷ_1..ŷ_K`, multiply along each path **and sum over all paths** — the sum appears because softmax is coupled (nudging one logit `z_i` changes every `ŷ_k`), so there are K paths from `z_i` to `L`. Backprop = this rule applied across the whole computation graph.

*Method 2 — full chain rule.* `∂L/∂z_i = Σ_k (∂L/∂ŷ_k)(∂ŷ_k/∂z_i)` with `∂L/∂ŷ_k = −y_k/ŷ_k` and softmax Jacobian `∂ŷ_k/∂z_i = ŷ_k(δ_{ki} − ŷ_i)` (diagonal `ŷ_i(1−ŷ_i)`, off-diagonal `−ŷ_k ŷ_i`). The `ŷ_k` cancels → `−Σ_k y_k(δ_{ki} − ŷ_i) = −y_i + ŷ_i·Σ_k y_k = ŷ_i − y_i`, using `Σ_k y_k δ_{ki} = y_i` and `Σ_k y_k = 1`. Either way: the output error is **prediction minus truth**.

**Concrete example.** A cat/dog/bird classifier outputs logits `z=[2.0, 1.0, 0.1]`, true class = cat, so `y=[1,0,0]`. Softmax (`S=11.21`) → `ŷ=[0.659, 0.242, 0.099]`, so `∂L/∂z = ŷ−y = [−0.341, +0.242, +0.099]`. The true class gets a *negative* gradient (its logit goes up under `z←z−η∇`); wrong classes get positive gradients (logits go down); magnitude = how much mass sat in the wrong place; entries sum to 0. Same form elsewhere: **linear reg** (MSE) → residual `ŷ−y` (predict 250k vs true 300k → −50); **logistic reg** (BCE, `ŷ=σ(z)`) → `ŷ−y` (spam `y=1`, `z=0.5⇒ŷ=0.62` → −0.38). These are GLMs with canonical loss↔activation pairing (MSE↔identity, BCE↔sigmoid, CE↔softmax), which is what makes the gradient collapse to prediction − target.

### 2.3 Propagate backward (the chain rule, layer by layer)

Let `δ₂ = ∂L/∂z₂ = ŷ − y`. Then:

```
∂L/∂W₂ = δ₂ a₁ᵀ          # outer product: error × the input that produced it
∂L/∂b₂ = δ₂
∂L/∂a₁ = W₂ᵀ δ₂          # push the error back through the linear layer
δ₁    = ∂L/∂z₁ = (W₂ᵀ δ₂) ⊙ φ'(z₁)   # gate by the local activation derivative
∂L/∂W₁ = δ₁ xᵀ
∂L/∂b₁ = δ₁
```

**Read the pattern — this is the whole of backprop:**
1. **Gradient w.r.t. a weight = (error at its output) × (input it multiplied).** That's why it's an outer product `δ aᵀ`.
2. **To move the error to the previous layer, multiply by `Wᵀ`** (the transpose appears because you're sending signal the opposite direction through the same linear map).
3. **To cross a nonlinearity, multiply elementwise by its local derivative `φ'`.** For ReLU, `φ'(z)` is 1 where z>0 and 0 elsewhere — so ReLU just *masks* the gradient (passes it through or kills it).

**Outer product (writing `δ aᵀ`).** The outer product of column vectors `u` (m-dim) and `v` (n-dim) is the `m×n` matrix `u vᵀ` with `(u vᵀ)_{ij} = u_i v_j` — every element of u times every element of v (vs the inner product `uᵀv`, a scalar). Example: `[1,2,3]ᵀ·[4,5] = [[4,5],[8,10],[12,15]]` (3×1 times 1×2 → 3×2). Here `δ₂` is `d_out`-dim and `a₁` is `d_in`-dim, so `δ₂ a₁ᵀ` is `d_out × d_in` = shape of `W₂`, with entry `δ₂_i · a₁_j` = (error at output i)×(input from j). Code: `np.outer(d2,a1)` / `d2[:,None]*a1[None,:]` / `torch.outer(d2,a1)` / `jnp.outer(d2,a1)` / `einsum('i,j->ij',d2,a1)`. Mini-batch: summing outer products over examples is the matmul `Δ Aᵀ`.

That's it. Forward = matmul then activation; backward = transpose-matmul then multiply-by-derivative. Every deep-learning framework is automating exactly this.

### 2.4 Why "vanishing/exploding gradients" fall right out of this

Look at `δ₁ = (W₂ᵀ δ₂) ⊙ φ'(z₁)`. For an L-layer net, the gradient at layer 1 is a **product of L−1 weight matrices and L−1 activation-derivative factors**. Two failure modes:

- If those factors are consistently **< 1** (e.g. saturated sigmoids with φ'≈small, or weight matrices with spectral norm < 1), the product shrinks geometrically → **vanishing gradient**, early layers barely learn.
- If they're consistently **> 1** (large weights), the product blows up → **exploding gradient**, you get NaNs/loss spikes.

This single observation motivates **almost every architectural trick in §3–§4**: careful initialization (keep factors ~1), normalization (control the scale of activations), and residual connections (provide a "×1" gradient path). Keep this product-of-Jacobians picture in your head — it's the unifying lens.

**Seeing it.** As the gradient flows backward through `L` layers, each layer multiplies it by one number — its Jacobian factor `r ≈ ‖W‖·|φ'|` — so the gradient at layer 1 is `≈ r^L`. Because it's a **power**, small deviations from 1 compound:

| factor `r` | after 10 layers (`r¹⁰`) | regime |
|---|---|---|
| 0.5 | 0.001 | vanish (fast) |
| 0.8 | 0.107 | vanish (slow) |
| 1.0 | 1.0 | **stable** |
| 1.1 | 2.6 | explode |
| 1.5 | 57.7 | explode |

Every §3–§5 trick just keeps `r ≈ 1`: **init** (Xavier/He) sets `‖W‖≈1` so the starting factor is ~1; **normalization** keeps activations unit-scale so `φ'` stays responsive (factor ~1 during training); **residuals** `y=x+F(x)` give Jacobian `I+∂F/∂x` → a literal ×1 path so the product can't collapse; **warmup/clipping** stop `r` transiently spiking. Concretely: sigmoid has `φ'≤0.25` → factor ≤0.25 → vanishing (why deep sigmoid nets won't train); ReLU active has `φ'=1`, with `‖W‖≈1` → factor ~1 → stable.

**Which part is the "Jacobian"?** A Jacobian is the matrix of partials of a vector function (`f:ℝⁿ→ℝᵐ` → `m×n` matrix `J_{ij}=∂f_i/∂x_j`). Each layer's local derivative is a Jacobian: **linear** `z=Wx+b` → `∂z/∂x = W` (so backward = "× Wᵀ"); **activation** `a=φ(z)` → `∂a/∂z = diag(φ'(z))` (so backward = "⊙ φ'"); **softmax** → `diag(ŷ) − ŷŷᵀ`. In matrix form `δ₁ = diag(φ'(z₁))·W₂ᵀ·(ŷ−y)` — each factor is a (transposed) layer Jacobian. The chain rule stacks them → the net's derivative is the **product of per-layer Jacobians**; `r ≈ ‖W‖·|φ'|` is the magnitude of one. (Backprop computes a **vector–Jacobian product** `vᵀJ`, never the full matrix.)

> **In Practice — frameworks fuse softmax+CE for stability.** Computing `softmax` then `log` separately can overflow (`e^{large}`) or underflow (`log(0) = −inf`). Libraries use a fused, log-sum-exp-stabilized `cross_entropy_with_logits`. **Symptom that you got this wrong:** loss is `inf`/`NaN` from step 0, or only when a logit gets large. **Look at:** whether you're double-applying softmax (e.g. softmax in the model *and* a loss that expects raw logits) — an extremely common real bug.

### Self-test §2
1. **Derive backprop for a 2-layer net.** → Reproduce §2.2–§2.3 from memory. The three rules: weight grad = `δ aᵀ`; backward through linear = `Wᵀδ`; backward through activation = `⊙ φ'`.
2. **What's `∂L/∂(logits)` for softmax-cross-entropy?** → `ŷ − y`. Be ready to sketch why (softmax Jacobian × CE grad telescopes).
3. **Explain vanishing/exploding gradients from the backprop equations.** → The layer-1 gradient is a product of many `Wᵀ` and `φ'` factors; consistently <1 → vanish, >1 → explode. Geometric in depth.
4. **Why is ReLU's backward pass essentially a mask?** → `φ'(z)=1` for z>0 else 0, so it passes the incoming gradient unchanged or zeros it — no shrinking on the active path.

---

## §3 — Initialization & gradient flow

### 3.1 The goal of initialization

**What is fan_in?** `fan_in` = the number of inputs feeding into a neuron (the input dimension; how many weighted connections get summed for one output). `fan_out` = number of outgoing connections (output dimension). Linear layer 512→256 (`W` is 256×512): `fan_in=512`, `fan_out=256`. Conv: `fan_in = in_channels × k_h × k_w`. It appears in init because a neuron sums `fan_in` products, so output variance grows linearly with it — more inputs → bigger sum → smaller weights (`∝ 1/fan_in`) to keep scale constant.

You want the **variance of activations (forward) and gradients (backward) to stay roughly constant across layers** at the start of training. If each layer multiplies the signal's variance by some factor ≠ 1, then over many layers the signal explodes or vanishes before you've taken a single useful step.

- **Xavier/Glorot init:** scale weights by `1/sqrt(fan_in)` (roughly) — derived assuming a linear/tanh regime, balances forward and backward variance.
- **He/Kaiming init:** scale by `sqrt(2/fan_in)` — the factor of 2 corrects for ReLU killing half the units (halving variance), so it's the right default for ReLU nets.

The point isn't memorizing constants; it's the **principle**: init is chosen so the *variance is preserved layer-to-layer*. Get it wrong and you start in a vanishing/exploding regime before normalization or residuals can save you.

**Derivation — why constant variance & where the 2 comes from.** A layer `z=Wx` sums over `fan_in` inputs; if each layer scales variance by a factor ≠1, over L layers it scales like `factor^L` (same compounding as §2.4) → factor<1 vanishes the signal, >1 explodes it; same for backward gradients. So preserve variance. With `z_i = Σ_j W_ij x_j` (W, x i.i.d. mean-0, independent): `Var(z) = fan_in · Var(W) · Var(x)`. Preserve → `Var(W) = 1/fan_in` → `std = 1/√fan_in` (**Xavier**, assumes linear/tanh-near-0). **ReLU** zeros half its inputs, so the second moment passed forward is halved: `E[a²] = E[max(0,z)²] = ½E[z²] = ½Var(z)`. Redo: `Var(z^l) = fan_in·Var(W)·½·Var(z^{l-1})`; preserve → `Var(W) = 2/fan_in` → `std = √(2/fan_in)` (**He**). The 2 exactly cancels ReLU's ½ — without it the signal shrinks ~½ per layer and vanishes over depth.

### 3.2 Why this still matters in the age of normalization

LayerNorm/BatchNorm reduce sensitivity to init, but at very large scale and at the *start* of training, init still governs whether early steps are stable. Modern LLMs often use small init scaled by depth (e.g. scaling residual-branch weights by `1/sqrt(N_layers)`) so that the residual stream variance doesn't grow with depth. **Tie this back to §2.4:** it's all about keeping that product-of-Jacobians near 1.

**Why 1/√N?** A residual block adds into a running stream: `x_l = x_{l-1} + F_l(x_{l-1})`, so the stream is a *sum* over blocks. Variances add → if each block contributes `≈σ²`, then `Var(x_N) ≈ Var(x_0) + N·σ²` — variance **grows linearly with depth** (std ~ √N), so late activations dwarf early ones (instability). Scale each branch by α: `x_l = x_{l-1} + α·F_l(...)` → `Var(x_N) ≈ Var(x_0) + N·α²σ²`. Pick `α = 1/√N` → `N·(1/N)·σ² = σ²`, constant regardless of depth. Why √N not 1/N: *variances* add, so a sum of N terms has variance ∝ N → divide std by √N. Same `1/√(count)` pattern as `1/√fan_in` (which normalizes the sum *within* a layer; `1/√N` normalizes the sum *across* depth). Used by GPT-2; DeepNorm/Fixup do depth-dependent residual scaling.

> **In Practice — symptom/cause for init.** *See* loss that is huge or NaN on step 0–10 and gradient norms that grow with layer depth → *suspect* bad init scale (too large) → *look at* the init std and any place you forgot to scale residual branches. Conversely, *see* loss totally flat and tiny gradient norms in early layers → *suspect* too-small init (vanishing) → *look at* per-layer gradient-norm logging.

### Self-test §3
1. **Why scale init by `1/sqrt(fan_in)`?** → To keep activation/gradient variance ≈ constant across layers, preventing exploding/vanishing signal at initialization.
2. **Why does He init use `sqrt(2/fan_in)` for ReLU?** → ReLU zeros ~half the inputs, halving variance; the factor of 2 compensates.
3. **If normalization exists, why care about init at all?** → Init governs the first steps before stats stabilize, and at depth/scale controls residual-stream growth; bad init still causes early NaNs or stalls.

---

## §4 — Normalization & residual connections

**Watch first:** the relevant parts of Karpathy's *"Let's build GPT"* and any BatchNorm explainer; but the concepts below are the interview core.

### 4.1 Normalization: what it does and the variants

Normalization layers **re-center and re-scale activations** so each layer sees a well-conditioned input distribution, which stabilizes and speeds up optimization. They then apply a learned scale `γ` and shift `β` so the network can undo the normalization if it wants.

| Variant | Normalizes over… | Depends on batch? | Where used |
|---|---|---|---|
| **BatchNorm (BN)** | each feature, across the **batch** | **Yes** | CNNs / vision |
| **LayerNorm (LN)** | all features, **within one example** | No | Transformers (BERT, GPT) |
| **RMSNorm** | like LN but **no mean-subtraction**, divide by RMS only | No | LLaMA, many modern LLMs |

### 4.2 Why transformers use LayerNorm/RMSNorm, not BatchNorm (very common question)

**What exactly are LN / RMSNorm?** **LayerNorm**, for one token's features `x ∈ ℝ^d`, normalizes *across the d features of that single example*: `μ = mean_i(x_i)`, `σ² = var_i(x_i)`, `x̂_i = (x_i − μ)/√(σ²+ε)`, `y_i = γ_i x̂_i + β_i` (learned per-feature γ, β). **RMSNorm** drops mean-centering — just `y_i = γ_i · x_i / RMS(x)` where `RMS(x) = √(mean_i(x_i²)+ε)` (cheaper, ~equal quality). **BatchNorm** instead normalizes *each feature across the batch*: `μ_i = mean_b(x_{b,i})` over the B batch examples.

**Why BN stats are noisy:** they're *sample estimates* of feature mean/variance from only B examples; standard error of the mean ∝ 1/√B (variance noisier), so with small B they jump batch-to-batch — each example's normalization depends on which random examples shared its batch (coupling + noise). Worse for transformers: small effective batch (long sequences), ragged variable-length batches, and B=1 decoding (batch stats undefined). LN/RMSNorm use the d features of one example (d large, fixed) → batch-independent, deterministic, identical train/inference (BN uses batch stats at train but running EMA at test → different function in each mode → bugs).

**Why does RMSNorm match LN — theory or empirical?** Both. LN does re-center (−μ) + re-scale (/σ); RMSNorm keeps only re-scaling. Hypothesis (Zhang & Sennrich 2019): LN's benefit comes mainly from **re-scaling invariance, not re-centering** — normalization helps by controlling activation/gradient *magnitude* (smoother landscape, stable per-layer LR), which is the /RMS part. Mean subtraction is near-redundant: the learned affine + next linear layer absorb a constant offset, and in high-dim the mean is small vs scale. Both LN & RMSNorm are scale-invariant (`f(αx)=f(x)`) → self-stabilizing gradient; RMSNorm keeps this, only drops shift-invariance. Verdict: well-motivated hypothesis + strong empirical parity (faster, used by LLaMA/T5/Gemma). Not a theorem, more than "just empirical."

Several reasons, all worth saying:
1. **Batch dependence is a liability for sequences.** BN's statistics are computed across the batch, so they're noisy with small batches and ill-defined for variable-length sequences / autoregressive decoding where you process one token at a time. LN normalizes *within a single example*, so it's independent of batch size and sequence position.
2. **Train/inference mismatch.** BN uses batch stats at train time but running averages at inference — a source of subtle bugs. LN behaves identically in both.
3. **RMSNorm** drops the mean-centering (keeps only the RMS rescaling), which is cheaper and empirically about as good — at LLM scale, shaving that compute matters.

### 4.3 Residual connections — the single most important trick for depth

A residual (skip) block computes `y = x + F(x)` instead of `y = F(x)`. Why this unlocked very deep networks:

- **Gradient highway.** Differentiate `y = x + F(x)`: `∂y/∂x = I + ∂F/∂x`. The identity term means the gradient has a path that's multiplied by **1**, not by a small weight-matrix factor. Re-read §2.4: the failure was a product of many <1 factors; the `+ I` guarantees at least one term in that product stays order-1, so gradients can flow to early layers even in very deep nets.
- **Easier optimization target.** Each block only has to learn a *residual* (a small correction to identity) rather than a full transformation. Learning "do nothing + small fix" is easier than learning the identity from scratch.
- This is *why* networks went from ~20 layers to 100s+ (ResNets) and why every transformer block is wrapped in residuals.

### 4.4 Pre-norm vs post-norm (a frontier-relevant detail)

Where you put the LayerNorm relative to the residual matters:
- **Post-norm** (original transformer): `x → Sublayer → Add → LayerNorm`. Tends to need careful warmup; can be unstable at depth.
- **Pre-norm** (modern default): `x → LayerNorm → Sublayer → Add`. The residual stream stays "clean" (un-normalized) end-to-end, giving a cleaner gradient highway and much more stable training at depth — which is why almost all large LLMs use pre-norm.

**Equations** (sublayer `S` = attention or FFN): post-norm `x_{l+1} = LN(x_l + S(x_l))`; pre-norm `x_{l+1} = x_l + S(LN(x_l))`.

```python
def block_postnorm(x):
    x = ln1(x + attention(x))   # LN AFTER add — on the trunk
    x = ln2(x + ffn(x)); return x
def block_prenorm(x):
    x = x + attention(ln1(x))   # LN inside the branch; skip untouched
    x = x + ffn(ln2(x)); return x
# pre-norm: one final ln_final(x) before the output head
```

**Why pre-norm is cleaner/stabler.** Residual in practice: ResNet `y = x + F(x)` (F = conv-bn-relu-conv-bn); transformer wraps attention & FFN in residuals. Unroll pre-norm: `x_L = x_0 + Σ_l S_l(LN(x_l))` — a pure additive identity path with **nothing on it**; LN sits on the *branch*, so the gradient keeps a clean ×1 highway at every depth. Post-norm `x_{l+1}=LN(x_l + S(x_l))` puts LN on the **trunk**, so the backward gradient crosses L LayerNorm Jacobians (not identity → rescale/project) → clean path broken → gradients grow/shrink with depth → needs warmup, unstable deep (Xiong et al. 2020). Trade-off: pre-norm's clean trunk → residual stream variance grows with depth (§3.2, fixed by 1/√N + final LN). LLMs pick pre-norm because trainability > bounded activations.

> **In Practice — normalization symptoms.** *See* training that is stable with batch size 256 but diverges or behaves erratically at batch size 8 → *suspect* BatchNorm's noisy small-batch statistics → *look at* whether you should switch to LN/GroupNorm. *See* a deep post-norm transformer that won't train without a long warmup and is touchy → *suspect* post-norm instability → *consider* pre-norm. *See* eval much worse than train *only* for BN models → *suspect* the train/running-stat mismatch (e.g. you left the model in train mode at eval) → *look at* `model.eval()` / batchnorm momentum.

### Self-test §4
1. **BatchNorm vs LayerNorm — and why transformers prefer LN?** → BN normalizes per-feature across the batch (batch-dependent, train/inf mismatch, bad for variable-length/decoding); LN normalizes within an example (batch-independent, identical at train/inf). Transformers process variable-length sequences and decode one token at a time → LN/RMSNorm.
2. **Why do residual connections help training?** → `∂y/∂x = I + ∂F/∂x`: the identity gives a gradient path multiplied by 1, preventing vanishing through depth; and each block only learns a correction to identity, an easier target.
3. **Pre-norm vs post-norm — which and why?** → Pre-norm keeps the residual stream un-normalized (cleaner gradient highway), trains more stably at depth; the modern default for large models.
4. **What does RMSNorm drop vs LayerNorm?** → Mean-subtraction; it only rescales by RMS. Cheaper, ~equal quality.

---

## §5 — Optimization in practice

**Watch first:** 3Blue1Brown gradient descent (Ch. 2, if not already); the rest is conceptual.

### 5.1 The optimizer ladder (know the *why* of each step up)

- **(Batch) Gradient Descent:** use the full dataset for each step. Accurate gradient, but one step per epoch — far too slow.
- **SGD / mini-batch SGD:** estimate the gradient from a small batch. Noisy but cheap and frequent; the noise even helps escape sharp minima. The workhorse.
- **Momentum:** accumulate a velocity `v = βv + g`, step with `v`. Damps oscillation across ravines, accelerates along consistent directions. Think "heavy ball rolling downhill."
- **RMSProp:** divide each parameter's step by a running RMS of its recent gradients → **per-parameter adaptive learning rate**. Helps when gradients have very different scales across parameters.
- **Adam:** momentum (1st moment) **+** RMSProp (2nd moment), with bias correction. Robust default; works well with little tuning.
- **AdamW:** Adam with **decoupled weight decay** — the single most-used optimizer for transformers. See §5.2.

**The math.** `g_t = ∇L`. **Momentum:** `v_t = β·v_{t-1} + g_t`, `θ_t = θ_{t-1} − η·v_t` (β≈0.9; accumulate consistent directions, damp oscillation). **RMSProp:** `s_t = ρ·s_{t-1} + (1−ρ)g_t²`, `θ_t = θ_{t-1} − η·g_t/(√s_t+ε)` (per-param adaptive LR). **Adam:** `m_t = β₁m_{t-1}+(1−β₁)g_t`, `v_t = β₂v_{t-1}+(1−β₂)g_t²`, bias-correct `m̂=m/(1−β₁ᵗ)`, `v̂=v/(1−β₂ᵗ)` (because m,v start at 0), step `θ_t = θ_{t-1} − η·m̂/(√v̂+ε)`. **AdamW:** `θ_t = θ_{t-1} − η·m̂/(√v̂+ε) − η·λ·θ_{t-1}` (decay on weights directly).

**In production transformers.** AdamW is the universal default: β₁=0.9, **β₂=0.95** (lowered from 0.999 — faster-adapting 2nd moment, robust to gradient spikes), ε=1e-8. LR: linear **warmup** → **cosine decay** to ~10% peak (peak ~1–6e-4). **Weight decay ~0.1**, decoupled, applied **only to matmul weights** (exclude biases, LN/RMSNorm gains, embeddings). Grad clip global norm 1.0. **Memory:** Adam stores m AND v per param → optimizer state = 2× params; with mixed precision ~**16 bytes/param** (2 bf16 weight + 2 grad + 4 fp32 m + 4 fp32 v + 4 fp32 master) → 7B model ≈ 112 GB of states → **why FSDP/ZeRO shard optimizer state**. bf16 compute + fp32 master/moments. Lower-memory alts: Adafactor (factorized 2nd moment; T5/PaLM), Lion (sign-based, momentum-only), Shampoo (preconditioned), Muon. AdamW still dominates.

### 5.2 Why AdamW decouples weight decay (a favorite question)

"L2 regularization" and "weight decay" are the *same thing* for plain SGD, but **not** for Adam. In vanilla Adam, if you implement weight decay by adding `λw` to the gradient, that decay term gets divided by the per-parameter adaptive denominator (the `√v` term) — so parameters with large gradient history get *less* effective decay. That couples regularization strength to gradient magnitude, which is not what you want.

**AdamW fixes this** by applying the decay *directly to the weights*, separately from the adaptive gradient update:
```
w ← w − lr · (adam_update)  −  lr · λ · w     # decay applied directly, not through √v
```
Result: weight decay acts as intended (uniform pull toward 0), decoupled from the adaptive scaling. This reliably improves generalization for transformers — which is why it's the default.

### 5.3 Learning-rate schedules & warmup

- **Warmup:** start the LR near 0 and ramp up over the first few hundred/thousand steps. **Why:** at initialization the model's estimates (and Adam's 2nd-moment estimate `v`) are unreliable; large early steps in this regime cause instability/divergence. Warmup lets the statistics settle before you take big steps. This is *especially* important for transformers (large models, adaptive optimizer, sensitive early dynamics).
- **Cosine decay (after warmup):** smoothly anneal the LR down following a cosine curve to a small final value. Large LR early explores; small LR late refines. The standard schedule for LLM pretraining.
- **Step decay / linear decay:** simpler alternatives; cosine is the common modern default.

### 5.4 Gradient clipping

Clip the global gradient norm to a threshold (e.g. rescale so `||g|| ≤ 1.0`) before the optimizer step. **Why:** occasionally a bad batch or a sharp region produces a huge gradient that would throw the weights far off and spike the loss (recall the exploding-gradient product from §2.4). Clipping caps the step size in those moments — cheap insurance that prevents one bad batch from destroying a run. Nearly all large-model training uses it.

### 5.5 Batch size, gradient accumulation, mixed precision, checkpointing

- **Batch size effects:** larger batches → less gradient noise (more stable, often allows higher LR via roughly square-root or linear scaling rules) but diminishing returns and worse exploration past a point; smaller batches → noisier but sometimes generalize better. It's a compute/stability/generalization tradeoff.
- **Gradient accumulation:** to *simulate* a big batch you can't fit in memory, run several micro-batches, **sum their gradients**, and step once. Mathematically (nearly) identical to one large batch. Key gotcha: average correctly and don't forget to zero grads only after the step.
- **Mixed precision (FP16/BF16):** store/compute in 16-bit for speed and memory, keep a master copy / certain reductions in FP32. **BF16** has the same exponent range as FP32 (just fewer mantissa bits) so it rarely overflows — preferred on modern hardware. **FP16** has a narrow range and often needs **loss scaling** (multiply the loss by a large constant before backward, unscale before the step) to keep small gradients from underflowing to 0.
- **Activation (gradient) checkpointing:** trade compute for memory — don't store all intermediate activations for the backward pass; recompute them during backward. Lets you train bigger models / longer sequences at the cost of ~30% extra compute. (This is a *memory* technique, not a numerical-stability one — don't confuse it with model checkpoints.)

> **In Practice — mixed precision symptoms.** *See* loss going NaN partway through specifically after you enabled FP16 → *suspect* overflow/underflow in 16-bit → *look at* whether loss scaling is on and whether you should switch to BF16. *See* gradients that are exactly 0 for small-magnitude params in FP16 → *suspect* underflow → *increase loss scale or use BF16*. *See* training fine in FP32 but subtly worse final quality in low precision → *suspect* precision loss in sensitive ops (softmax, layernorm, loss) → *keep those in FP32*.

### Self-test §5
1. **Adam vs SGD — when/why?** → Adam adapts per-parameter LR (momentum + RMS), robust with little tuning, great for sparse/ill-scaled gradients and transformers; SGD(+momentum) can generalize better in some vision settings and is simpler. Default to AdamW for transformers.
2. **Why does AdamW decouple weight decay?** → In Adam, L2-as-gradient gets divided by the adaptive `√v` denominator, making decay depend on gradient magnitude. Applying decay directly to weights restores uniform regularization and improves generalization.
3. **Why is warmup needed (esp. for transformers)?** → Early on, model and Adam's 2nd-moment estimates are unreliable; big steps then cause divergence. Ramping LR lets statistics settle. Transformers are especially sensitive.
4. **Why clip gradients?** → To cap the damage from occasional huge gradients (bad batch / sharp region) that would otherwise spike the loss or NaN the run.
5. **How does gradient accumulation simulate a larger batch?** → Sum gradients over several micro-batches, then take one optimizer step — equivalent to one large batch, trading time for memory.
6. **FP16 vs BF16?** → BF16 keeps FP32's exponent range (rarely overflows, usually no loss scaling); FP16 has narrow range and typically needs loss scaling to avoid underflow.

---

## §6 — The debugging playbook ("see X → suspect Y → look at Z")

This is the section frontier interviewers love, because it tests whether you've actually *trained* things. The canonical prompt:

> **"Your loss was decreasing fine, then spikes hard at 30k steps. Walk me through what you check."**

There's no single answer — the skill is having an **organized differential diagnosis**. Below is a reusable table. In an interview, *say the categories out loud and reason about which is most likely given the details they give you* (e.g. "spike at a specific step that's reproducible on resume → I'd suspect a specific bad data shard or a checkpoint/resume mismatch, not random instability").

### 6.1 Master symptom → cause → look-at table

| Symptom | Likely causes (Y) | Where to look (Z) |
|---|---|---|
| **Loss = NaN/inf from step 0** | Double softmax; wrong loss API (logits vs probs); bad init (too large); LR absurdly high; data has NaNs/inf | Loss function wiring; init std; print logit/grad norms on step 1; scan a batch for NaNs |
| **Loss NaN after enabling FP16** | Overflow/underflow in 16-bit; no loss scaling | Turn on loss scaling or switch to BF16; keep softmax/LN/loss in FP32 |
| **Sudden loss spike mid-training** | Bad/corrupt data batch or shard; LR too high for current region; gradient explosion; optimizer-state corruption; mixed-precision overflow | Log gradient norm per step (did it spike just before?); is the spike reproducible at the same step on resume → data shard; check if clipping is on |
| **Loss spikes that *recover* on their own** | Transient sharp region / occasional bad batch | Add/lower gradient clipping; usually benign if rare |
| **Loss spike right after resuming from checkpoint** | Optimizer state (Adam moments) not restored; LR schedule reset; data loader not resumed to same position; RNG/precision mismatch | Confirm optimizer + scheduler + dataloader state all saved/restored; check schedule step counter |
| **Loss flat / not decreasing at all** | LR too low or 0; gradients vanishing; frozen params / `requires_grad=False`; wrong loss; labels shuffled/misaligned | Print grad norms (≈0 → vanishing or frozen); verify a tiny model can overfit a tiny batch (sanity check) |
| **Train loss great, eval loss bad** (gap grows) | Overfitting; data leakage in train; train/eval preprocessing mismatch; BatchNorm left in train mode at eval | Regularization; check eval pipeline matches; `model.eval()`; look for leakage/contamination |
| **Train and eval both plateau high** | Underfitting; model too small / under-trained; LR too low; bad data | Scale model/compute; raise LR/lengthen schedule; inspect data quality |
| **Loss decreases but downstream eval gets *worse*** | Optimizing the proxy not the goal (e.g. perplexity ↓ but reasoning ↓); distribution shift; over-optimization/reward hacking in post-training; eval contamination masking real regressions | Decouple metrics; check eval set for contamination; look at *what* the model now gets wrong qualitatively |
| **Loss oscillates wildly** | LR too high; batch too small (noisy); no warmup | Lower LR; add warmup; increase batch / accumulation |
| **A layer's activations all → 0** | Dead ReLUs; too-high LR pushed biases negative | Activation histograms per layer; lower LR; switch to GELU/leaky |
| **Gradient norm grows with layer depth** | Bad init scale; missing residual scaling; exploding through depth | Per-layer grad-norm logging; init std; residual-branch scaling |
| **Throughput suddenly drops** | Stragglers in distributed training; data-loading bottleneck; checkpoint I/O | GPU util vs data-loader wait; per-rank timing; check for one slow rank |

### 6.2 The general method (say this framing in interviews)

1. **Localize in time:** is it gradual or a sudden spike? At a *specific reproducible step* (→ data/checkpoint) or random (→ numerical/LR)?
2. **Look at gradient norms first.** They're the cheapest, most diagnostic signal. A spike in grad norm just before the loss spike → exploding gradients / bad batch. Grad norm ≈ 0 → vanishing / frozen params.
3. **Check the data at that point.** Frontier training is data-bound; a corrupt shard, a tokenization bug, or wrong loss-masking (e.g. computing loss on padding/prompt tokens) is a top suspect for localized spikes.
4. **Check the "plumbing":** double softmax, logits-vs-probs, loss masking, checkpoint/resume of *optimizer + scheduler + dataloader* state, mixed-precision config.
5. **Sanity baseline:** can the model **overfit a single batch** to ~0 loss? If not, something is fundamentally broken (wiring, labels, LR), independent of scale. This is the fastest way to separate "bug" from "needs more compute."

> **The meta-point interviewers want:** you treat training as an instrumented experiment. You log grad norms, activation stats, and per-shard data provenance *before* you need them, and you reason about likelihood rather than randomly trying fixes.

### Self-test §6
1. **"Loss spikes at 30k steps — what do you check?"** → Localize (reproducible step? → data shard / resume bug; random? → LR/numerical). Check grad-norm trace just before the spike, inspect the data batch at that step, verify clipping and mixed-precision config, confirm it's not a resume/optimizer-state issue. Give the *categories*, then prioritize by the clues offered.
2. **"How do you debug NaNs?"** → Step 0 vs mid-training distinction; check double-softmax / loss API / init scale first; if FP16-triggered, loss scaling or BF16; scan data for NaN/inf; print where the first NaN appears (forward vs backward).
3. **"Train loss good, eval bad — causes?"** → Overfitting, data leakage, train/eval preprocessing mismatch, BN in train mode at eval, or (subtle) a proxy/goal mismatch where the loss improved but the capability didn't.
4. **"How do you tell a bug from 'needs more compute'?"** → Try to overfit one batch to ~0. Success → wiring is fine, it's a capacity/data/compute question; failure → it's a bug (labels, LR, masking, init).

---

## §7 — Consolidated #16 self-test (Deep Learning block)

Do these **out loud, timed (~2 min each)**, then check against the section noted. This is your dress rehearsal.

1. Derive backprop for a two-layer network. → §2
2. Why do residual connections help? → §4.3
3. BatchNorm vs LayerNorm (and why transformers use LN). → §4.2
4. Adam vs SGD. → §5.1
5. Why learning-rate warmup? → §5.3
6. Why gradient clipping? → §5.4
7. What causes exploding gradients? → §2.4 / §5.4
8. What causes loss spikes? → §6
9. How do you debug NaNs? → §6.1
10. Why does mixed precision help (and when does it hurt)? → §5.5
11. (Bonus) Why do neural nets need nonlinearities? → §1.2
12. (Bonus) Why decoupled weight decay in AdamW? → §5.2

### What "great" sounds like today
- You **derive** (backprop, the `ŷ−y` gradient, `I + ∂F/∂x`) instead of reciting names.
- You connect tricks to **one unifying cause** — the product-of-Jacobians (§2.4) explains init, normalization, residuals, warmup, and clipping all at once. Saying that out loud signals real understanding.
- You debug like an **experimentalist**: localize → grad norms → data → plumbing → overfit-one-batch.

---

## Quick-reference cheat sheet (review tonight)

- **Nonlinearity** = the reason depth helps (else net collapses to one affine map).
- **Backprop rules:** weight grad `= δ aᵀ`; back through linear `= Wᵀδ`; back through activation `= ⊙ φ'`.
- **Softmax-CE logit gradient = `ŷ − y`.**
- **Vanishing/exploding = product of L Jacobian factors** drifting <1 or >1. Everything below fixes this:
  - **Init** keeps per-layer variance ≈ 1 (`1/√fan_in`; He = `√(2/fan_in)` for ReLU).
  - **Normalization** conditions activations (LN/RMSNorm in transformers; not BN — batch dependence + train/inf mismatch).
  - **Residuals** give a `×1` gradient highway (`∂y/∂x = I + ∂F/∂x`); **pre-norm** keeps the highway clean.
  - **Warmup** avoids big early steps while Adam's stats are unreliable; **cosine decay** anneals.
  - **Clipping** caps damage from occasional huge gradients.
- **AdamW** = Adam + decoupled weight decay (decay applied to weights, not through `√v`).
- **BF16 > FP16** for range (FP16 needs loss scaling). **Grad accumulation** simulates big batches. **Activation checkpointing** trades compute for memory.
- **Debugging reflex:** localize in time → look at grad norms → inspect the data → check plumbing (double softmax, loss masking, resume state) → overfit one batch.

---

## Debug drill — "the deep model that won't train"

**Mock (AI-coding round):** A decoder-only transformer trains fine at **12 layers** but the loss **spikes to NaN within ~300 steps at 48 layers** (same data/optimizer, lr=3e-4, 2k warmup). A teammate's near-identical model trains fine at 48. The difference is in the block:

```python
class TransformerBlock(nn.Module):
    """Pre-norm decoder block."""
    def forward(self, x, mask):
        # pre-norm: normalize, then sublayer, then residual add
        x = self.norm1(x)
        x = x + self.attn(x, mask)
        x = self.norm2(x)
        x = x + self.mlp(x)
        return x
```
Questions: (1) Is this actually pre-norm? (2) Why NaN only at 48 layers? (3) One-line fix? (4) Why don't clip=1.0 + warmup save it?

**Answer:** The bug is `x = self.norm1(x)` **overwriting** x, so the residual add uses the *normalized* x as the residual — the residual stream is normalized in place every sublayer. That's **not** pre-norm; with an LN on the trunk every layer it's effectively **post-norm**, destroying the clean ×1 highway. Fix (normalize only the branch input):
```python
def forward(self, x, mask):
    x = x + self.attn(self.norm1(x), mask)
    x = x + self.mlp(self.norm2(x))
    return x
```
Why depth-dependent: post-norm instability scales with depth (gradient crosses L LayerNorm Jacobians ≠ identity) → diverges at 48, tolerated at 12. Clipping/warmup don't fix a structural gradient pathology — it's architectural. (Bonus: production also scales residual proj by ~1/√(2·n_layers), omitted here.)
