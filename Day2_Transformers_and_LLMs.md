# Day 2 — Transformers & LLMs

**Target:** Research Scientist loop at a frontier lab (Meta TBD / MSL style).
**Mode:** Concepts & derivations first, light code, explanation + self-test Q&A.
**Time budget:** ~8 hours.
**The bar for today:** Derive self-attention from scratch and explain *every* design choice (√d_k, causal masking, multi-head, KV cache, pre-norm). Reason about pretraining and post-training like a researcher — scaling laws, data mixtures, and *when* to reach for SFT vs RLHF vs DPO.

> **How to use this doc.** Same format as Day 1: watch-first video, explanation + intuition, an **In Practice** box, then **Self-test** Q&A. The two highest-leverage parts today are **§2 (attention derivation)** and **§5 (post-training landscape)** — protect those.

---

## Suggested 8-hour schedule

| Block | Time | Topic | Section |
|-------|------|-------|---------|
| 1 | 0:00–1:00 | Tokenization, embeddings, positional encodings (RoPE) | §1 |
| 2 | 1:00–2:30 | Self-attention, derived from scratch | §2 |
| — | 2:30–2:45 | Break | |
| 3 | 2:45–4:00 | Multi-head, the full block, pre/post-norm | §3 |
| 4 | 4:00–5:00 | Inference: KV cache, prefill vs decode, long context | §4 |
| — | 5:00–5:15 | Break | |
| 5 | 5:15–6:30 | Pretraining: next-token = MLE, perplexity, scaling laws | §5 |
| 6 | 6:30–8:00 | Post-training: SFT → RLHF → DPO + self-test | §6, §7 |

---

## §1 — Tokenization, embeddings, positional encodings

**Watch first:**
- 3Blue1Brown, *"Transformers, the tech behind LLMs"* (Ch.5) — https://www.3blue1brown.com/lessons/gpt
- Karpathy, *"Let's build the GPT Tokenizer"* — https://karpathy.ai/zero-to-hero.html

### 1.1 Tokenization

LLMs don't see characters or words — they see **tokens**, sub-word units produced by an algorithm like **Byte-Pair Encoding (BPE)**. BPE starts from bytes/characters and greedily merges the most frequent adjacent pairs until it hits a target vocabulary size (e.g. 50k–256k). Why sub-word:
- **Open vocabulary:** any string is representable (no out-of-vocab), because you can always fall back to bytes.
- **Efficiency:** common words become single tokens; rare words split into pieces. Balances sequence length against vocab size.

**Things to be able to say:**
- Tokenization choices have real consequences: a tokenizer poor at numbers/whitespace/code hurts math and coding performance. Many "weird" model failures (arithmetic, counting letters in a word) trace back to tokenization, not reasoning.
- Larger vocab → shorter sequences (cheaper attention, which is quadratic in length) but a larger embedding/output matrix. It's a tradeoff.

### 1.2 Embeddings

Each token id indexes into an **embedding matrix** `E ∈ R^{vocab × d_model}`, producing a `d_model`-dim vector. These are *learned*. The output projection (logits) often **ties weights** with `E` (weight tying) — saves parameters and tends to help, since input and output token spaces are related.

### 1.3 Positional encodings — and why RoPE won

Self-attention is **permutation-invariant**: it has no inherent notion of order (it's a weighted sum over a set). So you must *inject position*.

| Scheme | Idea | Trade-off |
|---|---|---|
| **Sinusoidal (absolute)** | Add fixed sin/cos vectors of different frequencies to embeddings | Simple; extrapolates poorly to longer contexts |
| **Learned absolute** | A learned vector per position | Can't go beyond trained max length |
| **RoPE (Rotary)** | *Rotate* Q and K vectors by an angle proportional to position | Encodes **relative** position; better long-context behavior; the modern default (LLaMA, most LLMs) |
| **ALiBi** | Add a distance-based linear bias to attention scores | Cheap, extrapolates well |

**Why RoPE is worth understanding:** it rotates the query and key vectors so that the *dot product* between positions i and j depends only on their **relative offset** (i − j), not absolute positions. Relative position is what actually matters for language, and it extrapolates more gracefully to longer sequences than absolute encodings. Context-length extension tricks (e.g. position interpolation, NTK-aware scaling) are mostly about *rescaling RoPE frequencies* so a model trained at 4k can run at 32k+.

> **In Practice — tokenization symptoms.** *See* a model that's great at prose but bad at arithmetic or "how many r's in strawberry" → *suspect* tokenization (numbers/characters merged into multi-char tokens) → *look at* how the tokenizer splits digits and whitespace, not the model's "reasoning." *See* a model degrade sharply past its trained context length → *suspect* positional-encoding extrapolation failure → *look at* RoPE scaling / position interpolation.

### Self-test §1
1. **Why sub-word tokenization (BPE) over words or characters?** → Open vocab (no OOV, byte fallback), and a tunable length/vocab tradeoff: common words = 1 token, rare = pieces. Characters give long sequences; words give OOV + huge vocab.
2. **Why do transformers need positional encodings?** → Attention is permutation-invariant (a weighted sum over a set), so order must be injected explicitly.
3. **What's the advantage of RoPE?** → It encodes *relative* position (the QK dot product depends on i−j), matches what language needs, and extrapolates to longer contexts better than absolute encodings.

---

## §2 — Self-attention, from scratch

**Watch first:** 3Blue1Brown, *"Attention in transformers, step-by-step"* (Ch.6) — https://www.3blue1brown.com/lessons/attention

This is *the* derivation to have cold.

### 2.1 The intuition

Attention lets each token **gather information from other tokens**, weighted by relevance. Each token emits three vectors:
- **Query (Q):** "what am I looking for?"
- **Key (K):** "what do I offer?"
- **Value (V):** "what I'll actually pass along if attended to."

A token's output is a **weighted average of all Values**, where the weights come from how well its Query matches each Key.

### 2.2 The equation

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
```
- `Q = X W_Q`, `K = X W_K`, `V = X W_V` — linear projections of the input `X ∈ R^{seq × d_model}`.
- `Q Kᵀ ∈ R^{seq × seq}` is the matrix of all pairwise **attention scores** (token i's query · token j's key).
- `softmax` over each row turns scores into a probability distribution (the attention weights).
- Multiply by `V` → each token's output is the weighted sum of Values.

### 2.3 Why divide by √d_k (must-know)

The dot product of two `d_k`-dimensional vectors with unit-variance components has **variance ≈ d_k** — it grows with dimension. If you feed large-magnitude scores into softmax, softmax **saturates**: it becomes nearly one-hot, gradients through it vanish, and training destabilizes. Dividing by `√d_k` rescales the scores back to unit variance, keeping softmax in a well-behaved (non-saturated) regime. **One-line answer:** "to keep the dot-product variance ~1 so softmax doesn't saturate and kill gradients."

### 2.4 Causal masking

For autoregressive (decoder-only) LMs, a token must not attend to *future* tokens — otherwise it sees the answer it's trying to predict. **Causal masking** sets the upper-triangular entries of `Q Kᵀ` to `−∞` *before* softmax, so those positions get weight 0. This enforces left-to-right information flow and is what lets us train on all positions in parallel with next-token prediction.

### 2.5 Complexity — the central scaling pain

`Q Kᵀ` is `seq × seq`, so attention is **O(seq² · d)** in compute and **O(seq²)** in memory for the score matrix. This is *the* reason long context is expensive, and why so much research (FlashAttention, sparse/linear attention, sliding windows) targets it. Internalize: **doubling context ~4×'s attention cost.**

> **In Practice — attention symptoms.** *See* training NaN/instability that worsens with model width → *suspect* missing or wrong `√d_k` scaling → *look at* the attention score computation. *See* OOM when you increase sequence length a little → *suspect* the O(seq²) score matrix → *look at* FlashAttention (which avoids materializing the full matrix) or shorter context.

### Self-test §2
1. **Derive self-attention.** → `softmax(QKᵀ/√d_k)V`, with Q/K/V as linear projections; explain Q=query, K=key, V=value, and the weighted-average interpretation.
2. **Why divide by √d_k?** → Dot-product variance grows ~d_k; without rescaling, softmax saturates → vanishing gradients/instability. √d_k restores unit variance.
3. **What is causal masking and why?** → Set future positions to −∞ before softmax so a token can't attend ahead; enforces autoregressive flow and enables parallel next-token training.
4. **What's attention's complexity and why does it matter?** → O(seq²·d); long context is quadratically expensive — the core scaling bottleneck.

---

## §3 — Multi-head attention & the full transformer block

### 3.1 Multi-head attention (MHA)

Instead of one attention, run `h` attention "heads" in parallel, each with its own smaller Q/K/V projections (dimension `d_k = d_model/h`), then concatenate and project. **Why multiple heads:** each head can specialize in a different relationship (syntax, coreference, positional patterns), and the model attends to information from **different representation subspaces** simultaneously. It's cheap (total compute ≈ single full-width attention) but much more expressive.

**Worth knowing (inference-efficiency variants):**
- **MQA (Multi-Query Attention):** all heads share *one* K and V → drastically smaller KV cache, faster decoding, small quality cost.
- **GQA (Grouped-Query Attention):** a middle ground — groups of heads share K/V. The modern default (LLaMA-2/3) because it nearly matches MHA quality with much smaller KV cache.

### 3.2 The full block (decoder-only, pre-norm)

A transformer block is two sub-layers, each wrapped in a residual connection with pre-norm:
```
x = x + Attention(LayerNorm(x))        # mix information across tokens
x = x + FFN(LayerNorm(x))              # process each token independently
```
- **Attention** moves information *between* positions (the only place tokens "talk").
- **FFN** (a 2-layer MLP, usually 4× wider, with GELU/SwiGLU) processes each position *independently* — this is where most parameters and "knowledge" live.
- **Residual stream:** think of `x` as a shared bus that each sub-layer reads from and writes additive updates to. (Recall Day 1 §4.3 — the `+` is the gradient highway.)

### 3.3 Architecture families

| Family | Attention | Use | Examples |
|---|---|---|---|
| **Encoder-only** | Bidirectional (no mask) | Understanding/embeddings/classification | BERT |
| **Decoder-only** | Causal | Generation; the dominant LLM design | GPT, LLaMA |
| **Encoder-decoder** | Encoder bidirectional + decoder cross-attends | Seq2seq (translation, summarization) | T5, original Transformer |

**Why decoder-only dominates LLMs:** a single simple objective (next-token prediction) scales beautifully, trains on any text, and unifies all tasks as text generation. Simplicity + scalability beat architectural specialization.

### Self-test §3
1. **Why multiple attention heads?** → Each head attends to different representation subspaces / relationships (syntax, coreference, position); more expressive at ~same cost as one full-width head.
2. **What do attention vs FFN each do in a block?** → Attention mixes information across positions; FFN processes each position independently (where most params/knowledge sit).
3. **Encoder-only vs decoder-only vs encoder-decoder — when each?** → Bidirectional understanding (BERT); causal generation (GPT — dominant); seq2seq with separate input/output (T5).
4. **What are MQA/GQA and why?** → Share K/V across heads (all / in groups) to shrink the KV cache and speed decoding; GQA is the modern quality/efficiency sweet spot.

---

## §4 — Inference: KV cache, prefill vs decode, long context

### 4.1 The KV cache

When generating autoregressively, at each step you produce one new token and append it. Naively you'd recompute attention over the whole prefix every step — O(n²) repeated work. Instead, you **cache the K and V vectors of all previous tokens** (they don't change) and only compute Q/K/V for the *new* token. This turns per-step work from "re-attend over everything from scratch" into "compute one new token's Q, attend against cached K/V." **The KV cache is the single most important inference optimization.**

- **Cost:** the cache grows linearly with sequence length and is often the **memory bottleneck** at inference. Size ≈ `2 × n_layers × seq × d_model × bytes` (the 2 = K and V). This is exactly why MQA/GQA (§3.1) exist — they shrink it.

### 4.2 Prefill vs decode — two different bottlenecks

LLM inference has two phases with **opposite performance characteristics** — a favorite interview distinction:
- **Prefill** (process the prompt): all prompt tokens go through in parallel → big matmuls → **compute-bound**, high GPU utilization.
- **Decode** (generate tokens one at a time): one token per step, must read the entire model + KV cache from memory each step → **memory-bandwidth-bound**, low arithmetic intensity, GPU often underutilized.

**Implications you should state:**
- **Batching** helps decode a lot (amortizes the weight-loading over many sequences) → **continuous batching** is a key serving technique.
- **Speculative decoding** attacks decode latency: a small "draft" model proposes several tokens, the big model verifies them in one parallel forward pass — you get multiple tokens per expensive step when the draft is right.
- Long prompts are dominated by prefill cost; long generations by decode cost.

### 4.3 Why long context is expensive (tie it together)

Two compounding costs: **attention is O(seq²)** (§2.5), and the **KV cache grows linearly** and dominates memory at decode. Long-context methods attack both: FlashAttention (memory-efficient exact attention), sliding-window/sparse attention (cut the quadratic), GQA/MQA + KV-cache quantization (shrink the cache), and RoPE scaling (extend trained length).

> **In Practice — inference symptoms.** *See* generation latency dominated by per-token time, GPU underutilized → *suspect* memory-bandwidth-bound decode → *look at* batching / speculative decoding / smaller KV cache (GQA). *See* OOM at long context during generation, not training → *suspect* KV cache growth → *look at* cache quantization, GQA, or shorter context. *See* time-to-first-token high on long prompts → *that's prefill* (compute-bound) → *look at* prompt length / chunked prefill.

### Self-test §4
1. **What is the KV cache and why does it help?** → Cache past tokens' K/V (they're fixed) so each decode step only computes the new token's Q and attends to cached K/V — avoids recomputing attention over the whole prefix. Biggest inference win.
2. **Prefill vs decode — what's the difference?** → Prefill processes the prompt in parallel (compute-bound); decode generates one token at a time (memory-bandwidth-bound). Different bottlenecks → different optimizations.
3. **Why is decoding slower per-token than prefill?** → Decode does one token per step but must reload all weights + KV cache from memory each step → low arithmetic intensity, bandwidth-bound.
4. **Why is long context expensive?** → O(seq²) attention + linearly-growing KV cache (memory bottleneck). Methods target both.

---

## §5 — Pretraining: objective, perplexity, scaling laws

### 5.1 Next-token prediction = maximum likelihood

LLMs are trained to predict the next token given the prefix, minimizing **cross-entropy** loss. This is exactly **maximum likelihood estimation**: maximizing the probability the model assigns to the actual training sequences. (Recall Day 1 §2.2 — softmax-CE, gradient `ŷ − y`.) **Teacher forcing**: during training you feed the *true* previous tokens (not the model's own predictions), so all positions train in parallel.

### 5.2 Perplexity

**Perplexity = exp(cross-entropy loss)** — intuitively, the model's average "branching factor," i.e. how many tokens it's effectively choosing among. Lower is better. Key nuances to state:
- Perplexity is only comparable across models with the **same tokenizer/vocab** (it's per-token).
- **Lower perplexity does not guarantee better downstream behavior.** A model can have great perplexity on web text yet poor reasoning or chat quality — perplexity rewards predicting the *distribution of text*, which is not the same as being helpful/correct. (This sets up Day 3's eval material and is a classic question.)

### 5.3 Scaling laws & Chinchilla (frontier-critical)

**Scaling laws:** loss falls predictably as a **power law** in model size (N), data (D), and compute (C). This predictability is *why* labs can de-risk huge runs — you extrapolate from small experiments.

**Chinchilla's key result:** for a fixed compute budget, there's a **compute-optimal balance** between model size and training tokens — roughly scale **parameters and data equally** (≈ 20 tokens per parameter). Earlier models (GPT-3) were **too big and undertrained** for their compute. Implications you should be able to discuss:
- Given more compute, you don't just make the model bigger — you scale data proportionally.
- **Inference cost** changes the calculus: a *smaller* model trained on *more* data (beyond Chinchilla-optimal, "overtraining") is often better in practice because it's cheaper to serve — LLaMA-style models are deliberately overtrained for this reason. (A great "research taste" point: compute-optimal for *training* ≠ optimal for *deployment*.)

> **In Practice — pretraining symptoms.** *See* perplexity improving but reasoning/coding evals flat or worse → *suspect* you're optimizing the proxy not the goal, or contamination inflated prior evals → *look at* decoupled task evals + contamination checks (Day 3). *See* loss following the predicted scaling curve then bending → *suspect* data repetition/exhaustion or a data-quality cliff → *look at* epoch count and data mixture.

### Self-test §5
1. **Why is next-token prediction maximum likelihood?** → Minimizing token cross-entropy = maximizing the probability assigned to the true sequences = MLE.
2. **What is perplexity and its limitation?** → exp(cross-entropy); average branching factor. Only comparable with same tokenizer, and lower perplexity ≠ better downstream (it rewards modeling text distribution, not helpfulness/correctness).
3. **What did Chinchilla show?** → Compute-optimal training balances model size and data roughly equally (~20 tokens/param); prior big models were undertrained. Given more compute, scale data too.
4. **Why might you train *past* compute-optimal (overtrain a smaller model)?** → To cut inference cost — a smaller, more-trained model is cheaper to serve at similar quality. Training-optimal ≠ deployment-optimal.

---

## §6 — Post-training: SFT → RLHF → DPO

This is the most important section for current frontier work. The arc: a pretrained model predicts *plausible* text; post-training makes it **helpful, harmless, and instruction-following.**

### 6.1 Supervised fine-tuning (SFT) / instruction tuning

Fine-tune the pretrained model on **(instruction, good response)** pairs — high-quality demonstrations. This teaches the model the *format* and *behavior* of being a helpful assistant (following instructions, answering rather than continuing text). It's the foundation every later step builds on. Limitation: demonstrations only show *one* good answer; they don't teach the model to *rank* better vs worse responses, and collecting high-quality demos is expensive.

### 6.2 Why RL at all? (the framing to nail)

We want to optimize for **human preferences** ("which response is better?") — but "better" is:
- **Not differentiable** (you can't backprop through a human judgment), and
- Easier to *compare* than to *demonstrate* (people can reliably say A > B even when they couldn't write the ideal answer).

So we **learn a reward model** from preference comparisons, then **optimize the policy (the LLM) to produce high-reward responses.** That optimization is a reinforcement-learning problem: the reward is a scalar signal over generated sequences.

### 6.3 RLHF (the classic pipeline)

Three stages:
1. **SFT** — get a decent instruction-follower.
2. **Reward model (RM):** collect human **pairwise preferences** (A vs B for the same prompt), train a model to score responses so preferred ones score higher (a ranking/Bradley-Terry loss).
3. **Policy optimization (PPO):** fine-tune the LLM to maximize RM reward, with a **KL-divergence penalty** that keeps it close to the SFT model.

**Why the KL penalty is essential:** without it, the policy will **reward-hack** — drift into weird, repetitive, or degenerate text that games the imperfect RM but isn't actually good. The KL term anchors the policy near the trusted SFT distribution. This "optimize reward but don't stray too far" is the heart of RLHF.

**Downsides of PPO-RLHF:** complex and unstable; you maintain *four* models (policy, reference, reward, value/critic); expensive; sensitive to hyperparameters.

### 6.4 DPO (Direct Preference Optimization)

**Key insight:** you can skip the explicit reward model and the RL loop. DPO uses a clever reparameterization showing the optimal RLHF policy has a **closed-form relationship to the reward**, so you can optimize **directly on the preference pairs** with a simple classification-style loss — making the preferred response more likely and the dispreferred less likely, relative to the reference model.

**DPO vs RLHF — the comparison to have ready:**

| | PPO-RLHF | DPO |
|---|---|---|
| Explicit reward model? | Yes | No (implicit) |
| RL loop / on-policy sampling? | Yes | No (offline, on a fixed preference dataset) |
| Stability / simplicity | Harder, more moving parts | Much simpler, more stable |
| Compute / engineering | Heavy (4 models) | Light (2 models) |
| Flexibility | Can keep improving with fresh on-policy data; reward model reusable | Tied to the preference dataset's coverage; offline |
| Where it shines | Frontier labs wanting maximal control + online improvement | Most teams wanting strong results cheaply |

**Honest nuance (good to say):** DPO is simpler and often matches PPO, but PPO/online RL can pull ahead when you can generate *fresh on-policy* data and iterate, because the model learns from its *own* current mistakes rather than a static dataset.

### 6.5 The broader landscape

- **RLAIF / Constitutional AI:** replace (some) human preference labels with **AI-generated** preferences guided by a set of principles ("constitution") — scales past expensive human labeling and gives more consistent, steerable feedback.
- **Rejection sampling / best-of-n:** sample n responses, keep the best (by an RM), fine-tune on those. A simple, strong baseline — often surprisingly competitive and used to bootstrap better SFT data.
- **Verifiable-reward RL (reasoning):** for math/code, the reward can be an **automatic checker** (does the test pass? is the answer correct?) rather than a learned RM. This is behind recent reasoning-model gains — clean, ungameable reward → strong RL signal. Process supervision (reward each *step*) vs outcome supervision (reward only the final answer) is the key axis here.

### 6.6 Failure modes to name

- **Reward hacking / over-optimization:** policy exploits RM flaws; mitigated by KL penalty and capping optimization.
- **Alignment tax:** safety/preference tuning can slightly degrade raw capability — a real tradeoff.
- **Over-refusal / under-refusal:** too much safety tuning → refuses benign requests; too little → unsafe. Both are eval'd explicitly.
- **Sycophancy:** RLHF can teach the model to tell users what they want to hear (preferred by raters) rather than what's true.

> **In Practice — post-training symptoms.** *See* RLHF output getting repetitive/weird while RM reward keeps climbing → *suspect* reward hacking → *look at* the KL penalty (too low?) and RM quality. *See* the model refusing harmless requests after safety tuning → *suspect* over-refusal → *look at* the refusal/over-refusal eval and safety-data balance. *See* chat quality improve but benchmark scores drop → *suspect* alignment tax / distribution shift from RLHF → *decouple* and track both.

### Self-test §6
1. **SFT vs RLHF — what does each add?** → SFT teaches assistant behavior/format from demonstrations (one good answer); RLHF optimizes for *preferences* via a reward model, teaching the model to rank better vs worse, beyond what demos can.
2. **Why use RL for LLM alignment at all?** → Human preference is non-differentiable and easier to compare than demonstrate; learn a reward model from comparisons, then optimize the policy against it.
3. **Why the KL penalty in RLHF?** → To prevent reward hacking — keep the policy near the trusted SFT distribution while improving reward.
4. **RLHF vs DPO?** → DPO drops the explicit RM and RL loop, optimizing preferences directly (simpler, stable, cheaper, offline); PPO-RLHF is heavier but allows on-policy iteration and a reusable reward model. (Use the table.)
5. **What is reward hacking / alignment tax?** → Policy exploits RM flaws (hacking); safety tuning slightly reducing capability (tax).
6. **When can you use a non-learned reward?** → Verifiable domains (math/code) where a checker gives a clean automatic reward — strong, ungameable RL signal.

---

## §7 — Consolidated #16 self-test (Transformers/LLMs block)

Do these out loud, timed.

1. Derive self-attention. → §2
2. Why divide by √d_k? → §2.3
3. What is causal masking? → §2.4
4. What is the KV cache? → §4.1
5. Why is decoding slower than prefill? → §4.2
6. How do you extend context length? → §4.3 / §1.3 (RoPE scaling)
7. What is perplexity? → §5.2
8. Why can lower perplexity fail to improve chat quality? → §5.2
9. What is instruction tuning? → §6.1
10. RLHF vs DPO? → §6.4
11. (Frontier) How would you choose a pretraining data mixture? → preview Day 3; mention ablations + scaling
12. (Frontier) What would you do with 10× more compute — scale model, data, context, or post-training? → §5.3: scale data with model (Chinchilla), then weigh inference cost; post-training often high-ROI

### What "great" sounds like today
- You **derive** attention and can justify *every* constant and mask.
- You connect **inference cost to architecture** (KV cache → GQA; O(seq²) → FlashAttention).
- You discuss post-training as a **landscape of tradeoffs** (SFT/RLHF/DPO/RLAIF/verifiable-reward), not a single recipe — and you bring up KL, reward hacking, and alignment tax unprompted.

---

## Quick-reference cheat sheet

- **Pipeline:** tokens → embeddings (+positions via **RoPE**, which encodes *relative* position) → N×[attention + FFN, each residual + pre-norm] → logits.
- **Attention = `softmax(QKᵀ/√d_k)V`.** √d_k keeps dot-product variance ~1 so softmax doesn't saturate. Causal mask = −∞ on future positions. Cost **O(seq²·d)**.
- **Heads** attend to different subspaces; **GQA/MQA** share K/V to shrink the **KV cache**.
- **Block:** attention mixes across tokens; FFN processes each token (most params). Residual stream = additive bus + gradient highway.
- **Decoder-only dominates** (one scalable objective unifies all tasks).
- **Inference:** KV cache avoids recompute; **prefill = compute-bound, decode = memory-bandwidth-bound**; batching + speculative decoding help decode.
- **Next-token = MLE; perplexity = exp(CE)** (same-tokenizer only; ≠ downstream quality).
- **Chinchilla:** balance params & data (~20 tok/param); overtrain smaller models to cut inference cost.
- **Post-training:** SFT (demos) → reward model (preferences) → PPO-RLHF (with **KL penalty** vs reward hacking) → **DPO** (skip RM+RL, optimize preferences directly: simpler/cheaper/offline). RLAIF scales labels; verifiable rewards for math/code. Watch for reward hacking, alignment tax, over-refusal, sycophancy.
