# LLM Architecture Frontier — Deep Dive

**Why this doc:** Your research round is with an LLM-architecture expert whose recent work spans **MoE, long-context/memory layers, efficiency & test-time compute, model merging, and unified multimodal**. This goes deeper than the Day-2 transformer basics into the *frontier architecture* topics that decide an architecture-heavy research conversation.

**The bar:** for each topic, know the **mechanism**, the **tradeoff**, the **failure mode**, and **one experiment you'd run**. Architecture interviews reward "why this design, what breaks, what would you try" — not memorized definitions.

> Builds on Day 2 (transformers) and the Post-Training/RL doc. Equations in code blocks.

---

## §1 — Mixture-of-Experts (MoE)

The single most important frontier-architecture topic. **The idea:** replace a dense FFN with **many expert FFNs** and a **router** that sends each token to only a few experts. You get more **parameters** (capacity) without proportionally more **FLOPs** (compute), because each token only activates a small subset.

### 1.1 Mechanism
```
For each token x:
  router logits  g = W_r · x                      # (n_experts,)
  pick top-k experts (k usually 1 or 2)
  weights = softmax(top-k logits)
  output = Σ_{i in top-k} weights_i · Expert_i(x)  # only k FFNs run
```
- **Sparse activation:** total params scale with `n_experts`, but per-token compute scales with `k`. E.g. 64 experts, top-2 → ~32× params at ~2× the FFN FLOPs.
- **Decouples capacity from compute** — the core reason MoE is central to scaling frontier models cheaply.

### 1.2 Load balancing (the central problem)
Left alone, the router **collapses** — it sends most tokens to a few favored experts, leaving others untrained ("dead experts"). Fixes:
- **Auxiliary load-balancing loss:** penalize imbalance so token assignment spreads across experts. Classic form:
```
L_aux = α · n_experts · Σ_i (f_i · P_i)
  f_i = fraction of tokens routed to expert i
  P_i = mean router prob mass on expert i
```
This is minimized when load is uniform.
- **Capacity factor:** cap how many tokens each expert takes per batch; overflow tokens are **dropped** (skip the FFN via the residual). Tradeoff: higher capacity = less dropping but more compute/memory.
- **Noisy / parameter-free routing:** add noise for exploration, or selection schemes that balance without a learned gate (your interviewer's *GatePro* line — parameter-free expert selection optimization). Be ready to discuss alternatives to the standard learned top-k gate.

### 1.3 Design axes & variants
- **Fine-grained experts:** more, smaller experts → better specialization and combinatorial flexibility (DeepSeek-MoE style).
- **Shared experts:** a few experts every token uses (capture common knowledge) + routed experts for specialization.
- **Top-1 (Switch) vs Top-2:** top-1 is cheapest; top-2 gives the router a comparison signal and usually better quality.
- **Routing granularity:** per-token (standard) vs per-sequence; token-choice (token picks experts) vs expert-choice (expert picks tokens — naturally balanced).

### 1.4 Tradeoffs & failure modes (say these)
- **Pro:** huge capacity per FLOP; strong quality-per-training-compute.
- **Con — inference:** all experts must be **in memory** even though each token uses few → high memory footprint; routing causes load imbalance across devices (**expert parallelism** + all-to-all communication overhead).
- **Con — training instability:** router is discrete/noisy; needs the aux loss tuned (too strong hurts quality, too weak → collapse). **Z-loss** often added for router-logit stability.
- **Failure modes:** expert collapse, token dropping at low capacity, training/inference load mismatch.

### 1.5 Experiments you'd propose
- Sweep `n_experts` × `k` × capacity factor at fixed *active* params, measuring quality-per-training-FLOP and inference memory.
- Compare token-choice vs expert-choice routing on load balance and quality.
- Ablate shared-expert count.

### Self-test §1
1. **Why MoE?** → Decouples parameters (capacity) from per-token FLOPs via sparse top-k activation — more capacity at ~constant compute.
2. **Main training problem and fix?** → Router collapse / load imbalance; fixed with an auxiliary load-balancing loss (+ capacity factor, z-loss, expert-choice routing).
3. **Why is MoE inference-expensive despite low FLOPs?** → All experts must reside in memory; routing creates load imbalance and all-to-all communication across devices.
4. **Top-1 vs top-2 vs fine-grained/shared experts?** → top-1 cheapest; top-2 gives comparison signal/quality; fine-grained = specialization; shared = common knowledge.

---

## §2 — Long context & memory

### 2.1 The two costs of long context (recap + go deeper)
- **Attention is O(T²)** in compute/memory (the score matrix).
- **KV cache grows O(T)** and dominates inference memory.
Long-context work attacks both.

### 2.2 Attention-level approaches
- **GQA / MQA:** share K/V across heads → shrink KV cache (Day 2 §3). The standard production lever.
- **Sliding-window / local attention:** each token attends to a window of w recent tokens → O(T·w). Often interleaved with a few global layers (e.g. Mistral, Longformer).
- **Sparse attention:** attend to a structured subset (strided/block patterns).
- **Linear attention / state-space (Mamba):** reformulate attention as a recurrence with O(T) compute and O(1) state — trades exact all-pairs mixing for efficiency; strong for very long sequences.
- **FlashAttention:** exact, IO-aware — reduces *memory traffic* (not the quadratic). (Day 5.)

### 2.3 Memory layers (your interviewer's UltraMem line)
**Memory layers / product-key memory:** a giant trainable **key-value memory** table queried by attention-like lookup. The model retrieves from a huge memory with sub-linear cost (product keys factorize the lookup), adding **parameters/knowledge capacity without proportional FLOPs** — conceptually MoE-adjacent (sparse access to a big parameter store). UltraMem-style work scales these to 100B+ params with strong long-context behavior. **Be ready to compare** memory layers vs MoE: both add sparse-access capacity; memory layers store *facts* in a KV table, MoE routes to *FFN experts*.

### 2.4 Position & context extension
- **RoPE scaling** (position interpolation, NTK-aware, YaRN): rescale rotary frequencies so a model trained at 4k runs at 32k–128k+. (Day 2 §1.3.)
- **Lost-in-the-middle:** models under-use mid-context info → motivates better long-context training + retrieval.
- **Long-context vs RAG:** stuff everything in context (simple, expensive, lost-in-middle) vs retrieve on demand (cheaper, needs a retriever). Often combined.

### Self-test §2
1. **Ways to make attention cheaper for long context?** → GQA/MQA (KV cache), sliding-window/local, sparse, linear/state-space (Mamba), FlashAttention (IO, not quadratic).
2. **What is a memory layer and how does it relate to MoE?** → A large trainable KV memory queried by factorized (product-key) lookup; adds capacity without proportional FLOPs — like MoE but stores facts in a KV table vs routing to FFN experts.
3. **How do you extend a model's context length?** → RoPE rescaling (position interpolation / NTK / YaRN) + long-context fine-tuning; watch lost-in-the-middle.

---

## §3 — Efficiency & test-time compute

### 3.1 Looped / recurrent-depth transformers (your interviewer's Parallel Loop Transformer)
**Idea:** instead of N distinct layers, **reuse** a block by looping it multiple times (weight sharing across depth / recurrence in depth). Benefits: **parameter efficiency** (fewer unique weights) and **adaptive compute** (loop more for harder inputs). "Parallel loop" variants restructure the recurrence to parallelize the test-time compute scaling. **Tradeoff:** weight sharing can cap representational diversity; gains depend on the task.

### 3.2 Adaptive computation
Spend variable compute per token/example: early-exit (stop at a shallow layer when confident), Mixture-of-Depths (route only some tokens through a layer), ACT-style halting. **Theme:** not every token needs full depth.

### 3.3 Test-time compute scaling (the big recent shift)
**Spend more compute at inference to get better answers** — the "thinking" paradigm:
- **Chain-of-thought / longer reasoning traces:** more tokens of reasoning → better answers (compute as a knob at test time).
- **Best-of-n / self-consistency:** sample many, vote/pick best.
- **Search / verifier-guided** (tree search, process-reward-guided): explore multiple paths, score with a verifier.
- **Why it matters:** a *smaller* model that "thinks longer" can beat a bigger model answering instantly — shifting the scaling frontier from train-time to **test-time** compute. Pairs with verifiable-reward RL (Post-Training doc §6).

### 3.4 Speculative decoding (inference latency)
Small draft model proposes several tokens; the big model verifies them in **one** parallel forward pass → multiple tokens per expensive step when the draft agrees. Exact (same distribution). (Day 2 §4.2.)

### Self-test §3
1. **What's a looped/recurrent-depth transformer and its tradeoff?** → Reuse a block across depth (weight sharing/recurrence) for parameter efficiency + adaptive compute; risk: limited representational diversity.
2. **What is test-time compute scaling?** → Spend more inference compute (longer CoT, best-of-n, search/verifier) to improve answers; a small "thinking" model can beat a big instant one — shifts the frontier to inference.
3. **How does speculative decoding speed up decoding?** → Draft model proposes tokens, big model verifies in parallel; multiple tokens per step, exact output.

---

## §4 — Model merging

### 4.1 What & why
Combine **multiple trained models (or checkpoints) into one** by operating on weights — no extra training (or little). Why it works: independently trained models often lie in a connected low-loss region, so averaging can land in a good basin and **average out noise** while combining capabilities.

### 4.2 Methods
- **Model soups:** average the weights of multiple fine-tunes (same init) → better, more robust than any single one.
- **Weight averaging across training (EMA / checkpoint averaging):** average checkpoints along a run for a flatter, better-generalizing solution.
- **Task arithmetic:** add/subtract **task vectors** (Δ = finetuned − base) to add or remove capabilities.
- **TIES / DARE:** resolve sign conflicts / sparsify deltas before merging to reduce interference.
- **Merging in pretraining** (your interviewer's line): merge models/branches *during* pretraining to combine data/skills or stabilize — an active frontier direction.

### 4.3 Tradeoffs
Cheap (no/low training), combines capabilities, improves robustness — **but** only works when models are compatible (shared init / nearby in weight space); naive averaging of unrelated models fails (interference). Sign conflicts and scale mismatches are the main failure modes.

### Self-test §4
1. **Why does weight averaging (model soups) work?** → Same-init fine-tunes lie in a connected low-loss region; averaging lands in a good, flatter basin and averages out noise.
2. **What is a task vector?** → Δ = finetuned − base weights; add/subtract to compose or remove capabilities (task arithmetic).
3. **When does merging fail?** → Incompatible models (different init / far apart), sign conflicts, scale mismatch → interference (mitigated by TIES/DARE).

---

## §5 — Unified multimodal architectures

### 5.1 The unification trend
Move from separate vision-encoder + LLM (Day 4) toward **one autoregressive model that both understands and generates** images (and text) in a single sequence — your interviewer's recent direction.

### 5.2 The tokenization question (continuous vs discrete) — key debate
To put images into a transformer you need image "tokens." Two camps:
- **Discrete tokens (VQ-VAE / VQ-GAN):** quantize image patches to a codebook of discrete tokens → model images exactly like text (autoregressive next-token over image tokens). Clean unification; but **quantization loses information** and codebook training is finicky.
- **Continuous tokens:** keep image representations continuous (no quantization) and model them with diffusion or continuous-output heads. **Higher fidelity** (no quantization loss); but breaks the clean discrete-next-token recipe — needs a continuous loss (e.g. diffusion head). Your interviewer's "unified autoregressive visual generation and understanding with **continuous tokens**" sits here.
- **The tension:** discrete = simple unification, lossy; continuous = high fidelity, harder to fit the autoregressive LLM framing. Be able to argue both sides.

### 5.3 Understanding + generation in one model
A single model that can both caption (understand) and generate images shares representations and can improve each via the other. Challenges: balancing the two objectives, tokenizer choice (above), and avoiding one task degrading the other.

### Self-test §5
1. **Continuous vs discrete visual tokens — tradeoff?** → Discrete (VQ-VAE): clean text-like autoregression but quantization loss; continuous: high fidelity but needs a continuous/diffusion loss, breaking simple next-token. 
2. **Why unify understanding + generation?** → Shared representations, mutual improvement, one model; challenge is balancing objectives + tokenizer choice.

---

## §6 — Scaling & width (taste)

- **Scaling laws / Chinchilla** (Day 2 §5): balance params & data; training-optimal ≠ deployment-optimal.
- **Width vs depth:** how to spend a parameter budget — wider (more per-layer capacity, parallelizes well) vs deeper (more sequential composition). Your interviewer's *Virtual Width Networks* explores decoupling effective width from compute.
- **Normalization & training dynamics** (Day 1 §4): BN/LN/GN/RMSNorm + his **Weight Standardization** (normalize weights for micro-batch training) — a rapport touchpoint.
- **The architecture-taste meta-question:** given X more compute, do you go bigger-dense, MoE, longer-context, more test-time compute, or better data? Answer with tradeoffs + a cheap proxy experiment first (Day 5 research taste).

---

## §7 — Consolidated self-test

1. Why does MoE decouple capacity from compute, and what's the main training problem? → §1
2. Compare memory layers and MoE. → §2.3
3. Name 4 ways to make long-context attention cheaper. → §2.2
4. What is test-time compute scaling and why can a small model beat a big one? → §3.3
5. Why does model-soup weight averaging work, and when does merging fail? → §4
6. Continuous vs discrete visual tokens — argue both sides. → §5.2
7. Looped/recurrent-depth transformers — benefit and tradeoff? → §3.1
8. Given 10× compute, MoE vs dense vs long-context vs test-time compute — how decide? → §6 (tradeoffs + cheap proxy first)

### What "great" sounds like for this round
- You discuss each architecture as a **capacity/compute/memory tradeoff with a failure mode**, not a definition.
- You connect topics: MoE ↔ memory layers (sparse capacity), test-time compute ↔ verifiable-reward RL, model merging ↔ flat minima.
- Every "what would you try" is a **specific, compute-aware, cheap-proxy-first** experiment (Day 5 pattern).

---

## Cheat sheet

- **MoE:** sparse top-k experts → params≫FLOPs; router **collapse** → aux load-balancing loss (+ capacity factor, z-loss, expert-choice). Inference: all experts in memory + all-to-all. Fine-grained/shared experts.
- **Long context:** O(T²) attn + O(T) KV cache. Fixes: GQA/MQA, sliding/sparse, linear/state-space (Mamba), FlashAttention (IO). **Memory layers** = factorized KV store, sparse capacity (MoE cousin). Extend via RoPE scaling; watch lost-in-the-middle.
- **Test-time compute:** looped/recurrent depth (param-efficient, adaptive), adaptive computation (early-exit, MoD), **think longer** (CoT, best-of-n, search+verifier) — small thinking model > big instant one. Speculative decoding for latency.
- **Model merging:** soups (avg same-init finetunes), checkpoint/EMA averaging, **task vectors** (Δ=ft−base), TIES/DARE; works near shared init, fails on incompatible models.
- **Unified multimodal:** one AR model understands+generates; **discrete (VQ, lossy) vs continuous (high-fidelity, needs diffusion/continuous loss)** tokens.
- **Scaling/width:** Chinchilla; width vs depth; Weight Standardization (micro-batch norm). Given more compute → weigh dense/MoE/context/test-time + cheap proxy first.
