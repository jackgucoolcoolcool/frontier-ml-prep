# Concept tags + targeted remediation content for the gap report.
#
# Every hidden test maps to one concept tag (see tag_for). When a test fails,
# the attempt log records the tag; /gaps aggregates the history and attaches
# the remediation content below for exactly the concepts the candidate missed.

PAGES = "https://jackgucoolcoolcool.github.io/frontier-ml-prep/"

# primary concept per stage (stability/shape failures override by test name)
STAGE_TAGS = {
    "w1s1": "attention-mechanics",
    "w1s2": "causal-masking",
    "w1s3": "attention-mechanics",
    "w1s4": "kv-cache",
    "w2s1": "normalization",
    "w2s2": "normalization",
    "w2s3": "backprop-gradients",
    "w2s4": "backprop-gradients",
    "f1s1": "sampling-mechanics",
    "f1s2": "sampling-mechanics",
    "f1s3": "decoding-controls",
    "f1s4": "decoding-controls",
    "f2s1": "preference-losses",
    "f2s2": "preference-losses",
    "f2s3": "rl-objectives",
    "f2s4": "rl-objectives",
    "a1s1": "algorithms",
    "a1s2": "algorithms",
    "m1s1": "classic-ml",
    "m1s2": "classic-ml",
    "m1s3": "classic-ml",
}


def tag_for(stage_id, test_name):
    n = test_name.lower()
    if "stable" in n or "nan" in n or "overflow" in n:
        return "numerical-stability"
    if "shape" in n:
        return "shape-discipline"
    return STAGE_TAGS.get(stage_id, "general")


REMEDIATION = {

"numerical-stability": {
    "title": "Numerical stability (log-sum-exp thinking)",
    "symptom": "NaN/inf when logits or reward margins get large — softmax, cross-entropy, sigmoid losses.",
    "explain": (
        "float64 overflows at exp(~709); fp16/bf16 far earlier. Any expression with a bare "
        "`exp` is a bug waiting for scale. The universal fix is to shift into log space and "
        "subtract the max BEFORE exponentiating — softmax is shift-invariant, so subtracting "
        "the row max changes nothing mathematically and bounds every exponent at 0. The same "
        "move powers stable cross-entropy (log-softmax, never log(softmax(x)) in two steps) and "
        "stable sigmoid losses (-log sigmoid(x) = softplus(-x) = logaddexp(0, -x)). Interviewers "
        "at frontier labs test this deliberately: a correct-but-unstable implementation reads "
        "as \"has never trained a real model\"."
    ),
    "pattern": (
        "# softmax: subtract max\n"
        "z = x - x.max(axis=-1, keepdims=True)\n"
        "p = np.exp(z); p /= p.sum(axis=-1, keepdims=True)\n\n"
        "# log-softmax / CE: stay in log space\n"
        "logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))\n\n"
        "# -log sigmoid(x)  (Bradley-Terry, DPO):\n"
        "loss = np.logaddexp(0.0, -x)      # = softplus(-x), exact for any x"
    ),
    "drills": [
        ["Why is softmax(x) == softmax(x - max(x)) exactly?",
         "exp(x_i - m) = exp(x_i)/exp(m); the constant exp(m) appears in numerator and denominator and cancels. Shift invariance is a property of the normalization."],
        ["Your DPO loss returns inf at implicit-reward margin -10000. What line is broken and what replaces it?",
         "Something like -np.log(sigmoid(h)): sigmoid underflows to 0, log(0) = -inf. Replace with np.logaddexp(0, -h), which is ~ -h + exp(h) accurate in both tails."],
    ],
    "docs": [["Day 1 — DL Fundamentals (softmax & CE)", PAGES + "Day1_Deep_Learning_Fundamentals.html"],
             ["ML Coding From Scratch", PAGES + "Coding_Implementations_From_Scratch.html"]],
},

"shape-discipline": {
    "title": "Shape discipline (keepdims, broadcasting, batch axes)",
    "symptom": "Wrong output shape, silent broadcasting bugs, code that works for 2-D but breaks at (B, T, d).",
    "explain": (
        "Most live-coding failures aren't conceptual — they're shape bugs the candidate can't see "
        "because they never wrote the shapes down. Two habits fix nearly all of them: (1) annotate "
        "every tensor line with a shape comment as you type it; (2) every reduction gets "
        "keepdims=True unless you have a reason — dropping a dim then broadcasting against it is "
        "the classic silent bug (it often RUNS, with wrong numbers, which is worse than crashing). "
        "Also: normalize over axis=-1, not axis=1, so code survives a leading batch dim."
    ),
    "pattern": (
        "Q = X @ Wq                      # (T, dk)\n"
        "S = Q @ K.T / np.sqrt(dk)       # (T, T)\n"
        "m = x.mean(axis=-1, keepdims=True)   # (..., 1) -- broadcasts back safely\n"
        "# quick self-test before submitting:\n"
        "assert out.shape == (T, dv), out.shape"
    ),
    "drills": [
        ["x is (B,T,d). What does (x - x.mean(axis=-1)) do, and why is it a disaster?",
         "x.mean(axis=-1) is (B,T); subtracting from (B,T,d) fails to align (broadcasts (B,T) as (1,B,T) → error) or, in the 2-D case (T,d) minus (T,), silently broadcasts across the WRONG axis when T==d. keepdims=True makes it (B,T,1) and always correct."],
        ["In MHA, why reshape(T, H, dh) then transpose(1, 0, 2), rather than reshape(H, T, dh) directly?",
         "The memory layout is per-token contiguous features; reshape(H,T,dh) would scramble tokens across heads. reshape splits the LAST axis in place; the transpose then moves heads to the front without touching data order per head."],
    ],
    "docs": [["ML Coding From Scratch", PAGES + "Coding_Implementations_From_Scratch.html"]],
},

"attention-mechanics": {
    "title": "Attention & multi-head mechanics",
    "symptom": "Wrong scaling constant, softmax over the wrong axis, head-split reshapes that scramble data.",
    "explain": (
        "The canonical chain is Q=XWq, K=XWk, V=XWv → S = QKᵀ/√d_k → A = softmax(S, axis=-1) → AV. "
        "The three classic errors: scaling by √d_model instead of √d_k (in MHA it must be the HEAD "
        "dim); softmax over axis 0 instead of -1 (each QUERY's row must sum to 1 — it distributes "
        "attention over keys); and in multi-head, splitting heads with a wrong reshape/transpose "
        "pair. Multi-head is exactly: reshape (T,d)→(T,H,dh) → transpose →(H,T,dh) → batched "
        "single-head attention (NumPy @ broadcasts over the leading axis) → inverse transpose → "
        "reshape → output projection Wo."
    ),
    "pattern": (
        "def mha(X, Wq, Wk, Wv, Wo, H):\n"
        "    T, d = X.shape; dh = d // H\n"
        "    split = lambda W: (X @ W).reshape(T, H, dh).transpose(1, 0, 2)\n"
        "    Q, K, V = split(Wq), split(Wk), split(Wv)\n"
        "    A = softmax(Q @ K.transpose(0, 2, 1) / np.sqrt(dh), axis=-1)\n"
        "    return (A @ V).transpose(1, 0, 2).reshape(T, d) @ Wo"
    ),
    "drills": [
        ["Attention output rows are identical for every position. Name the two most likely bugs.",
         "(1) softmax over the wrong axis (columns sum to 1, every query gets the same mixture); (2) scores saturated (forgot √d_k with large d), making softmax ~uniform or one-hot everywhere."],
        ["Complexity of self-attention in time and memory for sequence length T?",
         "Time O(T²·d) for QKᵀ and AV; memory O(T²) for the score matrix (the thing FlashAttention removes by tiling + online softmax, computing exact attention without materializing S)."],
    ],
    "docs": [["Day 2 — Transformers & LLMs", PAGES + "Day2_Transformers_and_LLMs.html"],
             ["ML Coding From Scratch", PAGES + "Coding_Implementations_From_Scratch.html"]],
},

"causal-masking": {
    "title": "Causal masking",
    "symptom": "Future tokens leak into past positions; masking applied after softmax; broken non-causal path.",
    "explain": (
        "The mask must hit the SCORES, before softmax: set disallowed entries to -inf so exp gives "
        "exactly 0 and the softmax renormalizes over the allowed prefix. Zeroing attention WEIGHTS "
        "after softmax breaks normalization (rows no longer sum to 1) and still leaks gradient. "
        "np.tril on a bool ones-matrix gives the allowed set; np.where(mask, S, -np.inf) applies it "
        "and broadcasts cleanly over a leading head axis. The property to verify out loud: changing "
        "token t+1 must not change any output at ≤ t — that's the test an interviewer will run."
    ),
    "pattern": (
        "mask = np.tril(np.ones((T, T), dtype=bool))   # True = may attend\n"
        "S = np.where(mask, S, -np.inf)                # BEFORE softmax\n"
        "A = softmax(S, axis=-1)                        # rows renormalize over prefix"
    ),
    "drills": [
        ["Why does -inf masking survive the stable-softmax max-subtraction?",
         "Each row's max is finite (the diagonal is always allowed), so -inf - max = -inf and exp(-inf) = 0 exactly. Row 0 attends only to itself with weight 1."],
        ["Your causal LM gets suspiciously low training loss. Why is a leaking mask the first suspect?",
         "With future leakage the model can copy the target token from the input — loss collapses toward 0 without learning. Any 'too good' LM loss means check masking and data (shift-by-one) first."],
    ],
    "docs": [["Day 2 — Transformers & LLMs", PAGES + "Day2_Transformers_and_LLMs.html"]],
},

"kv-cache": {
    "title": "KV-cache incremental decoding",
    "symptom": "Recomputing old K/V each step, cache shape drift, incremental output ≠ full causal attention.",
    "explain": (
        "At decode step t you need ONE new query row q_t, but attention over ALL keys/values so far. "
        "K and V for past tokens never change (causal), so cache them and append one row per step "
        "— O(T·d) per step instead of O(T²·d). No mask is needed: the cache contains only the past, "
        "so attending over the whole cache IS causal attention. The correctness invariant to state "
        "and test: feeding tokens one at a time must exactly reproduce row t of full causal "
        "attention at every step. And know the systems consequence: decode is memory-bandwidth "
        "bound on re-reading the cache — that's why GQA/MQA, sliding windows, paged/quantized "
        "caches exist."
    ),
    "pattern": (
        "def decode_step(x_t, Wq, Wk, Wv, cache):\n"
        "    q = (x_t @ Wq)[None, :]\n"
        "    k, v = (x_t @ Wk)[None, :], (x_t @ Wv)[None, :]\n"
        "    K = k if cache['K'] is None else np.vstack([cache['K'], k])\n"
        "    V = v if cache['V'] is None else np.vstack([cache['V'], v])\n"
        "    out = (softmax(q @ K.T / np.sqrt(K.shape[1]), axis=-1) @ V)[0]\n"
        "    return out, {'K': K, 'V': V}"
    ),
    "drills": [
        ["Why is there no causal mask in the decode step?",
         "The cache only contains positions ≤ t; there is no future to mask. Full-sequence attention needs the mask because K holds all positions at once."],
        ["Per-token KV-cache memory for 32 layers, 8 KV heads × 128 dim, fp16?",
         "32 × 2(K,V) × 8 × 128 × 2 bytes = 128 KB/token → ~13 GB for a 100k-token trajectory. This bounds batch size and step latency for agent rollouts."],
    ],
    "docs": [["LLM Architecture Frontier", PAGES + "LLM_Architecture_Frontier.html"],
             ["Day 2 — Transformers & LLMs", PAGES + "Day2_Transformers_and_LLMs.html"]],
},

"normalization": {
    "title": "LayerNorm / RMSNorm",
    "symptom": "Normalizing over the wrong axis, eps outside the sqrt, mean-subtracting in RMSNorm, biased-vs-unbiased variance confusion.",
    "explain": (
        "Both norms operate over the FEATURE axis (last), per token — never over batch or time. "
        "LayerNorm: (x − μ)/√(σ² + eps)·g + b with biased (population) variance; eps sits inside "
        "the sqrt, added to variance. RMSNorm drops the mean and the bias entirely: x/√(mean(x²)+eps)·g "
        "— cheaper (one fewer reduction) and empirically sufficient for transformers, which is why "
        "Llama-class models use it. Pre-norm (norm inside the residual branch) is the modern default "
        "because it preserves a clean identity path — that's the standard follow-up question."
    ),
    "pattern": (
        "def rmsnorm(x, g, eps=1e-6):\n"
        "    return x / np.sqrt((x**2).mean(axis=-1, keepdims=True) + eps) * g\n\n"
        "def layernorm(x, g, b, eps=1e-5):\n"
        "    mu = x.mean(axis=-1, keepdims=True)\n"
        "    var = x.var(axis=-1, keepdims=True)      # biased\n"
        "    return (x - mu) / np.sqrt(var + eps) * g + b"
    ),
    "drills": [
        ["Why does RMSNorm on input x+100 (constant shift) give a different result than LayerNorm?",
         "LayerNorm removes the mean, so it's shift-invariant; RMSNorm divides by RMS of the raw values, so a constant offset changes the output. That's exactly the re-centering RMSNorm bets you don't need."],
        ["What breaks if eps goes outside the sqrt: √var + eps?",
         "For tiny variance the output scale becomes 1/eps-ish instead of ~1/√eps semantics people tune for — different stability behavior and silently non-standard; tests against reference implementations fail."],
    ],
    "docs": [["Day 1 — DL Fundamentals", PAGES + "Day1_Deep_Learning_Fundamentals.html"],
             ["LLM Architecture Frontier", PAGES + "LLM_Architecture_Frontier.html"]],
},

"backprop-gradients": {
    "title": "Backprop by hand (softmax-CE, ReLU, linear layers)",
    "symptom": "Gradient doesn't match finite differences: missing 1/B, wrong ReLU gate, transposed matmuls.",
    "explain": (
        "Three facts cover the whole MLP backward. (1) Softmax + CE fuse into dlogits = (p − onehot)/B "
        "— derive it once (∂/∂z_k of −z_y + logΣe^z = p_k − 𝟙[k=y]) and never re-derive under "
        "pressure; the /B appears because the loss is a MEAN. (2) Linear layer y = xW + b: "
        "dW = xᵀ·dy, db = dy.sum(0), dx = dy·Wᵀ — get the transposes right by matching shapes, "
        "not by memory: dW must be (in, out). (3) ReLU is a gate: dh_pre = dh * (h_pre > 0), using "
        "the PRE-activation. And the professional habit this simulator grades: check any hand "
        "gradient against central finite differences on a tiny input before trusting it."
    ),
    "pattern": (
        "dlogits = (p - onehot) / B          # (B,C)  the one identity to memorize\n"
        "dW2 = h.T @ dlogits                 # (H,C)  <- shapes dictate the transpose\n"
        "db2 = dlogits.sum(axis=0)\n"
        "dh  = dlogits @ W2.T                # (B,H)\n"
        "dh_pre = dh * (h_pre > 0)           # ReLU gate on PRE-activation\n"
        "# verify: (f(w+h) - f(w-h)) / 2h  vs analytic, tol 1e-4"
    ),
    "drills": [
        ["Your dW1 matches finite differences to 1e-2 but not 1e-4. Likely cause?",
         "A finite-diff step crossing a ReLU kink, or one-sided differences (O(h) error) instead of central (O(h²)). If ALL grads are off by the same factor, you forgot /B."],
        ["Why does db = dy.sum(axis=0) rather than mean?",
         "b is broadcast over the batch in the forward, so its gradient accumulates over the batch. The 1/B already lives in dlogits from the mean loss — dividing again double-counts."],
    ],
    "docs": [["Day 1 — DL Fundamentals (backprop)", PAGES + "Day1_Deep_Learning_Fundamentals.html"]],
},

"sampling-mechanics": {
    "title": "Temperature / top-k / top-p mechanics",
    "symptom": "Temperature multiplied instead of divided, top-p cutoff off by one, forgetting to renormalize, T=0 divide-by-zero.",
    "explain": (
        "The pipeline is: logits/T → softmax → top-k filter → top-p filter → renormalize after EVERY "
        "filter. Temperature DIVIDES the logits (T<1 sharpens, T>1 flattens); handle T=0 as an "
        "explicit greedy branch before dividing. Top-p: sort descending, keep the SMALLEST prefix "
        "whose cumsum ≥ p (never zero tokens — the top token always survives). Filter on original "
        "vocab indices via argsort — sorting the probs themselves loses token identity. Every filter "
        "changes total mass, so renormalize or your samples are silently biased."
    ),
    "pattern": (
        "p = softmax(logits / T)\n"
        "order = np.argsort(-p)              # original indices, desc\n"
        "cs = np.cumsum(p[order])\n"
        "cnt = np.searchsorted(cs, top_p) + 1   # smallest prefix >= p\n"
        "keep = order[:cnt]\n"
        "m = np.zeros_like(p); m[keep] = p[keep]; p = m / m.sum()"
    ),
    "drills": [
        ["p = [0.5, 0.3, 0.2], top_p = 0.8. Which tokens survive and with what probabilities?",
         "cumsum = [0.5, 0.8, 1.0]; smallest prefix ≥ 0.8 is the first two. Renormalized: [0.625, 0.375, 0]."],
        ["Why apply temperature before top-p rather than after?",
         "Temperature reshapes the distribution and therefore WHICH tokens fall in the nucleus: T>1 fattens the tail so more tokens enter. Applied after (to filtered probs) it can't change membership, only re-tilt survivors — a different, usually unintended sampler."],
    ],
    "docs": [["Fundamentals — Sampling & RL", PAGES + "Fundamentals_Sampling_and_RL.html"],
             ["Day 6 — Post-Training: Sampling & RL", PAGES + "Day6_PostTraining_Sampling_and_RL.html"]],
},

"decoding-controls": {
    "title": "Decoding controls (repetition penalty, best-of-n)",
    "symptom": "Penalty asymmetry wrong (boosting negative logits), mutating inputs, tie-breaking and rng discipline in best-of-n.",
    "explain": (
        "Repetition penalty (CTRL): for already-generated tokens, divide POSITIVE logits by the "
        "penalty, multiply NEGATIVE ones — both push toward less likely. A single rule (always "
        "divide) makes unlikely repeated tokens MORE likely, which is the trap. Never mutate the "
        "caller's logits array. Best-of-n: sample n, score with the reward model, return the "
        "argmax with strict > so ties go earliest; the interview points here are determinism "
        "(consume the rng in a fixed order) and knowing best-of-n ≈ policy improvement with "
        "KL ≈ log n − (n−1)/n, degrading into reward hacking as n grows."
    ),
    "pattern": (
        "out = logits.copy()                 # never mutate input\n"
        "for t in set(generated):\n"
        "    out[t] = out[t] / penalty if out[t] > 0 else out[t] * penalty\n\n"
        "best_t, best_r = None, -float('inf')\n"
        "for _ in range(n):\n"
        "    t = generate(rng); r = reward(t)\n"
        "    if r > best_r: best_t, best_r = t, r    # strict >: earliest wins"
    ),
    "drills": [
        ["Penalty 1.5 on logits [2.0, -1.0] (both already generated). Result, and why the asymmetry?",
         "[1.333, -1.5]. Dividing -1.0 by 1.5 would give -0.667 — MORE likely. Both ops must move logits toward less-likely, hence divide-if-positive / multiply-if-negative."],
        ["Best-of-64 outputs score higher on the RM but read worse to humans. What's happening?",
         "Reward over-optimization (Goodhart): with more samples you increasingly select RM errors rather than genuine quality. Fixes: smaller n, better/ensembled RM, KL-penalized selection, or verifiable rewards."],
    ],
    "docs": [["Fundamentals — Sampling & RL", PAGES + "Fundamentals_Sampling_and_RL.html"],
             ["Day 6 — Post-Training: Sampling & RL", PAGES + "Day6_PostTraining_Sampling_and_RL.html"]],
},

"preference-losses": {
    "title": "Preference losses (Bradley–Terry RM, DPO)",
    "symptom": "Wrong pairing/grouping of policy vs reference terms, β applied in the wrong place, unstable -log σ.",
    "explain": (
        "Bradley–Terry: L = −log σ(r_chosen − r_rejected) — only reward DIFFERENCES are identified, "
        "absolute scale is free. DPO is the same sigmoid loss on IMPLICIT rewards r = β·log(π/π_ref): "
        "L = −log σ(β[(logπ_c − logπref_c) − (logπ_r − logπref_r)]). The grouping trap: it's "
        "(policy − reference) per response FIRST, then chosen − rejected; mis-pairing silently "
        "flips or corrupts the objective while still training 'fine'. β is INSIDE the sigmoid and "
        "sets the KL-anchor strength to the reference. Always compute −log σ(x) as logaddexp(0, −x). "
        "Log the margin (mean of the β-scaled term) — it's the health metric of a DPO run."
    ),
    "pattern": (
        "h = beta * ((logp_c - ref_c) - (logp_r - ref_r))   # implicit reward margin\n"
        "loss = np.logaddexp(0.0, -h).mean()                 # -log sigmoid(h), stable\n"
        "# sanity: h=0 -> loss = log 2 = 0.693"
    ),
    "drills": [
        ["In a healthy DPO run the margin grows, yet log-prob of the CHOSEN responses falls. Contradiction?",
         "No — DPO optimizes the GAP. Both chosen and rejected likelihoods can fall with the gap widening (mass moves off-distribution). It's a known dynamic; monitor chosen log-prob separately if you care about it, or use β/regularization to limit drift."],
        ["Why must the reference model stay frozen in DPO?",
         "The implicit reward is β·log(π/π_ref); if π_ref moves, the reward definition moves with it and the closed-form correspondence to the KL-constrained RLHF objective breaks — you're chasing a moving target."],
    ],
    "docs": [["Post-Training & RL Deep Dive", PAGES + "Post_Training_and_RL_Deep_Dive.html"],
             ["Day 6 — Post-Training: Sampling & RL", PAGES + "Day6_PostTraining_Sampling_and_RL.html"]],
},

"rl-objectives": {
    "title": "RL objectives (PPO clip, GRPO advantages)",
    "symptom": "Pessimism wrong for negative advantages, sign errors (maximize vs minimize), GRPO std/eps placement, zero-variance groups.",
    "explain": (
        "PPO: with ρ = exp(logπ_new − logπ_old), the surrogate is min(ρA, clip(ρ, 1±ε)A) and the "
        "LOSS is its negative mean. The min is the whole idea — a pessimistic lower bound. Don't "
        "branch on sign(A): elementwise min handles both cases, and for A<0 it correctly keeps the "
        "UNCLIPPED (more negative) term when ρ runs away, so you're never protected from getting "
        "worse. GRPO: advantages = (r − group_mean)/(group_std + eps) per prompt-group, replacing "
        "the critic with a Monte-Carlo baseline; eps goes on the STD so an all-equal group yields "
        "exactly 0 advantage (no signal) instead of NaN. Know the std-normalization difficulty bias "
        "(near-solved prompts get amplified) — it's a live research point (Dr. GRPO)."
    ),
    "pattern": (
        "ratio = np.exp(logp_new - logp_old)\n"
        "obj = np.minimum(ratio * adv, np.clip(ratio, 1-eps, 1+eps) * adv)\n"
        "loss = -obj.mean()\n\n"
        "adv = (R - R.mean(1, keepdims=True)) / (R.std(1, keepdims=True) + 1e-6)"
    ),
    "drills": [
        ["ρ = 1.5, A = −1, ε = 0.2. What does the objective take and why not the clipped −1.2?",
         "min(−1.5, −1.2) = −1.5, the unclipped term. The bound is pessimistic: when the update made things worse, clipping must not cap the penalty — gradients keep pushing back."],
        ["A GRPO group has rewards [1,1,1,1]. What advantage should each sample get, and what goes wrong with eps on the variance instead of the std?",
         "Zero — a uniform group carries no learning signal. With (r−μ)/√(σ²+eps), σ²=0 gives division by √eps ≈ 1e-3, amplifying float noise into huge fake advantages; eps on the std gives exactly 0/(0+eps) = 0."],
    ],
    "docs": [["Post-Training & RL Deep Dive", PAGES + "Post_Training_and_RL_Deep_Dive.html"],
             ["Day 6 — Post-Training: Sampling & RL", PAGES + "Day6_PostTraining_Sampling_and_RL.html"]],
},

"algorithms": {
    "title": "Core algorithmic patterns",
    "symptom": "Sliding-window pointer moving backward, merge logic missing nested/touching intervals, unhandled empty inputs.",
    "explain": (
        "The two patterns here generalize: (1) interval merging = sort by start, then one linear "
        "pass extending with max(end) — the max is what survives nested intervals; 'touching' "
        "semantics (≤ vs <) is a clarifying question to ask OUT LOUD. (2) Sliding window with a "
        "last-seen map: the left pointer only ever moves FORWARD — left = max(left, seen[c]+1); "
        "without the max, a stale index from before the window drags left backward and re-admits "
        "duplicates ('abba'). State complexity unprompted: it's expected at this level."
    ),
    "pattern": (
        "intervals.sort(key=lambda iv: iv[0])\n"
        "if s <= out[-1][1]: out[-1][1] = max(out[-1][1], e)   # touch + nested\n\n"
        "if c in seen: left = max(left, seen[c] + 1)           # never backward\n"
        "seen[c] = right; best = max(best, right - left + 1)"
    ),
    "drills": [
        ["Trace 'abba': what happens without the max() when the second 'a' arrives?",
         "seen['a']=0, left is already 2 (after the 'bb' collision). left = 0+1 = 1 moves BACKWARD, window 'bba' contains duplicate b, length overcounts. max(2, 1) keeps left = 2, answer 2."],
        ["Merge [[1,10],[2,3],[5,6]] — what does naive end-overwriting produce and why is it wrong?",
         "After [1,10], seeing [2,3] overwrites end to 3 → [1,3], then [5,6] wrongly separates. max(10,3) keeps [1,10]. Nested intervals are the test case that catches it."],
    ],
    "docs": [],
},

"classic-ml": {
    "title": "Classic ML from scratch (distances, k-means, kNN)",
    "symptom": "Pairwise-distance loops instead of one matmul, k-means that NaNs on an empty cluster, mean() without axis, non-deterministic tie-breaking.",
    "explain": (
        "The classic-ML screen is won on four reflexes. (1) All pairwise distances as ONE "
        "expression: ||a-b||² = ||a||² - 2a·b + ||b||² — the cross term is a single matmul; "
        "clip tiny negatives (np.maximum(D, 0)) because the expansion cancels catastrophically "
        "on near-duplicate points and a downstream sqrt would NaN. (2) Every reduction carries "
        "an explicit axis: X[mask].mean() with no axis collapses to a SCALAR that silently "
        "broadcasts into the centroid row. (3) Name the degenerate case before the interviewer "
        "does: empty cluster (keep the old centroid — mean of an empty slice is NaN), tied vote "
        "(state a deterministic rule; bincount+argmax gives smallest-label-wins for free), zero "
        "vector (eps on the denominator). (4) Know WHY k-means terminates: both steps "
        "monotonically decrease within-cluster SSE and there are finitely many partitions — a "
        "local optimum only, so init (k-means++, restarts) decides quality."
    ),
    "pattern": (
        "# all pairwise squared distances, no loops:\n"
        "D = (A*A).sum(1, keepdims=True) - 2*A@B.T + (B*B).sum(1)[None, :]\n"
        "D = np.maximum(D, 0)                    # cancellation guard\n\n"
        "# k-means update, both traps handled:\n"
        "mask = labels == j\n"
        "if mask.any(): C[j] = X[mask].mean(axis=0)   # axis=0! empty keeps old\n\n"
        "# kNN vote, deterministic tie-break:\n"
        "idx = np.argpartition(row, k-1)[:k]     # O(n), not argsort's O(n log n)\n"
        "pred = np.bincount(y[idx]).argmax()     # first max = smallest tied label"
    ),
    "drills": [
        ["Your k-means run returns a centroid row of all NaN. What happened, and what are the standard fixes?",
         "A cluster went empty; np.mean of an empty slice is NaN and it never recovers. Fix: keep the previous centroid (or re-seed at the farthest point). Prevent: k-means++ init spreads centroids so emptying is rare."],
        ["Why can the vectorized ||a||² - 2a·b + ||b||² give a negative squared distance, when the direct (a-b)² sum can't?",
         "For near-identical points far from the origin the three terms are huge and nearly cancel; float rounding of each term leaves a tiny negative residue. The direct form subtracts FIRST (small numbers, no cancellation) but needs O(nmd) memory when broadcast. Clip with np.maximum(D, 0)."],
    ],
    "docs": [["RL Coding Pack — §15 classic ML warm-ups", PAGES + "RL_Coding_Interview_Pack.html"],
             ["Day 7 — 100 drills", PAGES + "Day7_100_Drills.html"]],
},

"crash-discipline": {
    "title": "Reading tracebacks fast (crash discipline)",
    "symptom": "Submissions that crash before tests run: syntax errors, NameErrors, wrong signatures.",
    "explain": (
        "In a live interview a crash costs double: the time AND the impression. Rules: read the "
        "traceback bottom-up (last line = what, first frames = where); after any signature change, "
        "re-run once before submitting; keep a 3-line smoke test at the bottom of the buffer "
        "(tiny input, print shapes) and run it habitually. Most crashes here are re-definition "
        "drift — a later stage renamed an argument the earlier helper still uses."
    ),
    "pattern": (
        "# keep a live smoke test at the bottom of your buffer:\n"
        "if __name__ == '__main__':\n"
        "    X = np.random.randn(3, 4)\n"
        "    print(attention(X, W, W, W).shape)   # run after EVERY edit"
    ),
    "drills": [
        ["What's the fastest read of 'TypeError: attention() takes 4 positional arguments but 5 were given'?",
         "The caller (the hidden test) uses the NEW extended signature; you edited the body but not the def line — add the keyword argument with a default so both old and new call sites work."],
    ],
    "docs": [],
},

}
