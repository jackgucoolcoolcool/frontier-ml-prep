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

### 2.2 The one identity that makes it clean: softmax + cross-entropy

A beautiful, must-know result: **the gradient of softmax-cross-entropy w.r.t. the logits is just `ŷ − y`.**

```
∂L/∂z₂ = ŷ − y          # (predicted probabilities minus one-hot truth)
```

Why this matters: it's clean, numerically stable (which is why frameworks fuse `softmax` + `cross_entropy` into one op — see the In Practice box), and it gives the whole backward pass a simple starting point. Call this `δ₂ = ŷ − y` (the "error at the output").

*Sketch of why:* `∂L/∂z_i = Σ_k (∂L/∂ŷ_k)(∂ŷ_k/∂z_i)`. The cross-entropy term gives `∂L/∂ŷ_k = −y_k/ŷ_k`, and the softmax Jacobian is `∂ŷ_k/∂z_i = ŷ_k(δ_{ki} − ŷ_i)`. Multiply and sum, use `Σ_k y_k = 1`, and everything telescopes to `ŷ_i − y_i`.

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

That's it. Forward = matmul then activation; backward = transpose-matmul then multiply-by-derivative. Every deep-learning framework is automating exactly this.

### 2.4 Why "vanishing/exploding gradients" fall right out of this

Look at `δ₁ = (W₂ᵀ δ₂) ⊙ φ'(z₁)`. For an L-layer net, the gradient at layer 1 is a **product of L−1 weight matrices and L−1 activation-derivative factors**. Two failure modes:

- If those factors are consistently **< 1** (e.g. saturated sigmoids with φ'≈small, or weight matrices with spectral norm < 1), the product shrinks geometrically → **vanishing gradient**, early layers barely learn.
- If they're consistently **> 1** (large weights), the product blows up → **exploding gradient**, you get NaNs/loss spikes.

This single observation motivates **almost every architectural trick in §3–§4**: careful initialization (keep factors ~1), normalization (control the scale of activations), and residual connections (provide a "×1" gradient path). Keep this product-of-Jacobians picture in your head — it's the unifying lens.

> **In Practice — frameworks fuse softmax+CE for stability.** Computing `softmax` then `log` separately can overflow (`e^{large}`) or underflow (`log(0) = −inf`). Libraries use a fused, log-sum-exp-stabilized `cross_entropy_with_logits`. **Symptom that you got this wrong:** loss is `inf`/`NaN` from step 0, or only when a logit gets large. **Look at:** whether you're double-applying softmax (e.g. softmax in the model *and* a loss that expects raw logits) — an extremely common real bug.

### Self-test §2
1. **Derive backprop for a 2-layer net.** → Reproduce §2.2–§2.3 from memory. The three rules: weight grad = `δ aᵀ`; backward through linear = `Wᵀδ`; backward through activation = `⊙ φ'`.
2. **What's `∂L/∂(logits)` for softmax-cross-entropy?** → `ŷ − y`. Be ready to sketch why (softmax Jacobian × CE grad telescopes).
3. **Explain vanishing/exploding gradients from the backprop equations.** → The layer-1 gradient is a product of many `Wᵀ` and `φ'` factors; consistently <1 → vanish, >1 → explode. Geometric in depth.
4. **Why is ReLU's backward pass essentially a mask?** → `φ'(z)=1` for z>0 else 0, so it passes the incoming gradient unchanged or zeros it — no shrinking on the active path.

---

## §3 — Initialization & gradient flow

### 3.1 The goal of initialization

You want the **variance of activations (forward) and gradients (backward) to stay roughly constant across layers** at the start of training. If each layer multiplies the signal's variance by some factor ≠ 1, then over many layers the signal explodes or vanishes before you've taken a single useful step.

- **Xavier/Glorot init:** scale weights by `1/sqrt(fan_in)` (roughly) — derived assuming a linear/tanh regime, balances forward and backward variance.
- **He/Kaiming init:** scale by `sqrt(2/fan_in)` — the factor of 2 corrects for ReLU killing half the units (halving variance), so it's the right default for ReLU nets.

The point isn't memorizing constants; it's the **principle**: init is chosen so the *variance is preserved layer-to-layer*. Get it wrong and you start in a vanishing/exploding regime before normalization or residuals can save you.

### 3.2 Why this still matters in the age of normalization

LayerNorm/BatchNorm reduce sensitivity to init, but at very large scale and at the *start* of training, init still governs whether early steps are stable. Modern LLMs often use small init scaled by depth (e.g. scaling residual-branch weights by `1/sqrt(N_layers)`) so that the residual stream variance doesn't grow with depth. **Tie this back to §2.4:** it's all about keeping that product-of-Jacobians near 1.

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
