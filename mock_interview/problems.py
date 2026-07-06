# Problem bank for the mock coding interview simulator.
# Interviewers anonymized: W = rigor/efficiency (perception/3D background),
# F = post-training/RL/agents. A = generic algorithmic warm-up.
#
# Each session = one layered problem, revealed stage by stage like a real
# CoderPad interview. Each stage carries hidden tests (run server-side),
# two escalating hints, a verbal probe question, and a model solution.

SESSIONS = [

# =====================================================================
# W1 — Attention ladder (the single most likely ask)
# =====================================================================
{
  "id": "w1",
  "interviewer": "W",
  "persona": "Interviewer W — rigorous, efficiency-minded (perception/3D systems background). Wants vectorized code, correct shapes, complexity called out.",
  "title": "Attention, from scratch",
  "minutes": 45,
  "intro": "Hi, thanks for joining. I've read your background so let's get straight to coding — plain NumPy today, no frameworks. We'll start simple and keep extending it, so keep your code easy to change. Feel free to ask clarifying questions at any point.",
  "closing": "That's all I had — nice working with you. Any questions for me about the team?",
  "stages": [
    {
      "id": "w1s1",
      "title": "Scaled dot-product attention",
      "prompt": "First task: implement single-head scaled dot-product **self-attention**.\n\nSignature:\n```\ndef attention(X, Wq, Wk, Wv):\n```\n`X` is `(T, d_model)`, `Wq`/`Wk` are `(d_model, d_k)`, `Wv` is `(d_model, d_v)`. Return the `(T, d_v)` output.\n\nWrite it as you'd want it reviewed — and talk me through the shapes as you go.",
      "starter": "import numpy as np\n\ndef attention(X, Wq, Wk, Wv):\n    # X: (T, d_model)\n    pass\n",
      "hints": [
        "Break it down: Q = X@Wq, K = X@Wk, V = X@Wv. Then scores = Q @ K.T / sqrt(d_k) — shape (T, T). Softmax over which axis? Then weights @ V.",
        "Your softmax needs to be numerically stable: subtract the row-wise max before exponentiating. np.exp of a score around 700+ overflows to inf and you'll get NaNs."
      ],
      "probe": {
        "q": "Why do we divide by √d_k? What goes wrong without it?",
        "a": "With unit-variance q, k the dot product has variance d_k, so scores grow like √d_k. Large-magnitude scores push softmax into saturation — near one-hot attention — so gradients through softmax vanish and training destabilizes. Dividing by √d_k keeps score variance ~1 regardless of head size."
      },
      "tests": """
def _sm(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def _ref_attn(X, Wq, Wk, Wv):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    return _sm(Q @ K.T / np.sqrt(Wk.shape[1])) @ V

def _t_shape():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 8)); Wq = rng.standard_normal((8, 4))
    Wk = rng.standard_normal((8, 4)); Wv = rng.standard_normal((8, 6))
    out = attention(X, Wq, Wk, Wv)
    assert out is not None, "function returned None"
    assert out.shape == (5, 6), f"expected shape (5, 6) = (T, d_v), got {out.shape}"
_check("output shape is (T, d_v)", _t_shape)

def _t_ref():
    rng = np.random.default_rng(1)
    for _ in range(3):
        X = rng.standard_normal((7, 8)); Wq = rng.standard_normal((8, 4))
        Wk = rng.standard_normal((8, 4)); Wv = rng.standard_normal((8, 4))
        got = attention(X, Wq, Wk, Wv)
        exp = _ref_attn(X, Wq, Wk, Wv)
        assert np.allclose(got, exp, atol=1e-5), "values don't match reference (check scaling by sqrt(d_k) and softmax axis)"
_check("matches reference on random inputs", _t_ref)

def _t_stable():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((4, 8)) * 60.0
    Wq = rng.standard_normal((8, 4)); Wk = rng.standard_normal((8, 4)); Wv = rng.standard_normal((8, 4))
    out = attention(X, Wq, Wk, Wv)
    assert np.isfinite(out).all(), "NaN/inf with large-magnitude inputs -- is your softmax numerically stable?"
_check("numerically stable on large logits", _t_stable)
""",
      "solution": "def softmax(z, axis=-1):\n    z = z - z.max(axis=axis, keepdims=True)   # stability: exp of shifted logits\n    e = np.exp(z)\n    return e / e.sum(axis=axis, keepdims=True)\n\ndef attention(X, Wq, Wk, Wv):\n    Q, K, V = X @ Wq, X @ Wk, X @ Wv          # (T,dk) (T,dk) (T,dv)\n    scores = Q @ K.T / np.sqrt(Wk.shape[1])    # (T,T)\n    A = softmax(scores, axis=-1)               # rows sum to 1\n    return A @ V                               # (T,dv)"
    },
    {
      "id": "w1s2",
      "title": "Causal masking",
      "prompt": "Good. Now make it usable for a decoder: add causal masking.\n\nExtend the signature to:\n```\ndef attention(X, Wq, Wk, Wv, causal=False):\n```\nWhen `causal=True`, position `t` must attend only to positions `≤ t`. The `causal=False` path must keep working.",
      "starter": None,
      "hints": [
        "Build a (T, T) lower-triangular boolean mask (np.tril). Set the disallowed scores to -inf (or -1e9) *before* the softmax — not after.",
        "np.where(np.tril(np.ones((T, T), dtype=bool)), scores, -np.inf) — then your existing stable softmax handles the rest; exp(-inf) = 0."
      ],
      "probe": {
        "q": "Why mask with -inf before softmax rather than zeroing attention weights after?",
        "a": "Zeroing after softmax breaks normalization — rows no longer sum to 1, so outputs are scaled wrong and gradients flow to masked positions. Setting scores to -inf makes exp() exactly 0 and the softmax renormalizes over only the allowed positions."
      },
      "tests": """
def _sm(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def _ref_attn_c(X, Wq, Wk, Wv, causal=False):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    S = Q @ K.T / np.sqrt(Wk.shape[1])
    if causal:
        T = S.shape[0]
        S = np.where(np.tril(np.ones((T, T), dtype=bool)), S, -np.inf)
    return _sm(S) @ V

def _t_causal_ref():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((6, 8)); Wq = rng.standard_normal((8, 4))
    Wk = rng.standard_normal((8, 4)); Wv = rng.standard_normal((8, 4))
    got = attention(X, Wq, Wk, Wv, causal=True)
    exp = _ref_attn_c(X, Wq, Wk, Wv, causal=True)
    assert np.allclose(got, exp, atol=1e-5), "causal output doesn't match reference"
_check("causal=True matches reference", _t_causal_ref)

def _t_causality():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((6, 8)); Wq = rng.standard_normal((8, 4))
    Wk = rng.standard_normal((8, 4)); Wv = rng.standard_normal((8, 4))
    out1 = attention(X, Wq, Wk, Wv, causal=True)
    X2 = X.copy(); X2[4:] = rng.standard_normal((2, 8)) * 3.0
    out2 = attention(X2, Wq, Wk, Wv, causal=True)
    assert np.allclose(out1[:4], out2[:4], atol=1e-5), "changing FUTURE tokens changed PAST outputs -- the mask is leaking"
_check("no information leaks from the future", _t_causality)

def _t_noncausal_still_works():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((5, 8)); Wq = rng.standard_normal((8, 4))
    Wk = rng.standard_normal((8, 4)); Wv = rng.standard_normal((8, 4))
    got = attention(X, Wq, Wk, Wv, causal=False)
    exp = _ref_attn_c(X, Wq, Wk, Wv, causal=False)
    assert np.allclose(got, exp, atol=1e-5), "causal=False path broke"
_check("causal=False path still correct", _t_noncausal_still_works)
""",
      "solution": "def attention(X, Wq, Wk, Wv, causal=False):\n    Q, K, V = X @ Wq, X @ Wk, X @ Wv\n    scores = Q @ K.T / np.sqrt(Wk.shape[1])          # (T,T)\n    if causal:\n        T = scores.shape[0]\n        mask = np.tril(np.ones((T, T), dtype=bool))   # True = allowed\n        scores = np.where(mask, scores, -np.inf)      # before softmax\n    A = softmax(scores, axis=-1)\n    return A @ V"
    },
    {
      "id": "w1s3",
      "title": "Multi-head attention",
      "prompt": "Now multi-head. New function:\n```\ndef mha(X, Wq, Wk, Wv, Wo, n_heads, causal=False):\n```\nAll of `Wq, Wk, Wv, Wo` are `(d_model, d_model)`; `d_model` is divisible by `n_heads`. Split the projected `Q, K, V` into `n_heads` contiguous chunks along the feature dimension (i.e. `reshape(T, H, d_h)`), run attention per head with scaling by √d_h, concatenate, then apply the output projection `Wo`.\n\nVectorize across heads — no Python loop over heads, please.",
      "starter": None,
      "hints": [
        "reshape (T, d) -> (T, H, d_h) -> transpose to (H, T, d_h). Then batched matmul: Q @ K.transpose(0, 2, 1) is (H, T, T) — NumPy's @ broadcasts over the leading head axis.",
        "Scale by sqrt(d_h), not sqrt(d_model). After A @ V you have (H, T, d_h): transpose back to (T, H, d_h), reshape to (T, d), then @ Wo."
      ],
      "probe": {
        "q": "Same total parameters and FLOPs — why do multiple heads help over one big head?",
        "a": "Each head applies softmax over its own scores, so different heads can attend to different positions/relations simultaneously (syntax vs positional vs copying). A single head must commit to one mixture per query — one softmax is a convex combination with one attention pattern. Heads also lower d_h per pattern, acting like a low-rank factorization of a richer attention operator."
      },
      "tests": """
def _sm(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def _ref_mha(X, Wq, Wk, Wv, Wo, n_heads, causal=False):
    T, d = X.shape; dh = d // n_heads
    def split(M): return (X @ M).reshape(T, n_heads, dh).transpose(1, 0, 2)
    Q, K, V = split(Wq), split(Wk), split(Wv)
    S = Q @ K.transpose(0, 2, 1) / np.sqrt(dh)
    if causal:
        S = np.where(np.tril(np.ones((T, T), dtype=bool)), S, -np.inf)
    out = (_sm(S) @ V).transpose(1, 0, 2).reshape(T, d)
    return out @ Wo

def _t_mha():
    rng = np.random.default_rng(6)
    T, d, H = 5, 8, 2
    X = rng.standard_normal((T, d))
    Wq, Wk, Wv, Wo = (rng.standard_normal((d, d)) for _ in range(4))
    got = mha(X, Wq, Wk, Wv, Wo, H)
    assert got.shape == (T, d), f"expected shape {(T, d)}, got {got.shape}"
    exp = _ref_mha(X, Wq, Wk, Wv, Wo, H)
    assert np.allclose(got, exp, atol=1e-5), "doesn't match reference (check the head split: reshape(T,H,dh) then transpose, and scale by sqrt(d_h))"
_check("multi-head output matches reference", _t_mha)

def _t_mha_causal():
    rng = np.random.default_rng(7)
    T, d, H = 6, 12, 3
    X = rng.standard_normal((T, d))
    Wq, Wk, Wv, Wo = (rng.standard_normal((d, d)) for _ in range(4))
    got = mha(X, Wq, Wk, Wv, Wo, H, causal=True)
    exp = _ref_mha(X, Wq, Wk, Wv, Wo, H, causal=True)
    assert np.allclose(got, exp, atol=1e-5), "causal multi-head doesn't match reference"
_check("causal multi-head matches reference", _t_mha_causal)

def _t_heads_differ():
    rng = np.random.default_rng(8)
    T, d = 5, 8
    X = rng.standard_normal((T, d))
    Wq, Wk, Wv, Wo = (rng.standard_normal((d, d)) for _ in range(4))
    a = mha(X, Wq, Wk, Wv, Wo, 1)
    b = mha(X, Wq, Wk, Wv, Wo, 4)
    assert not np.allclose(a, b, atol=1e-5), "1 head and 4 heads gave identical outputs -- are you actually splitting heads?"
_check("head count actually changes the computation", _t_heads_differ)
""",
      "solution": "def mha(X, Wq, Wk, Wv, Wo, n_heads, causal=False):\n    T, d = X.shape\n    dh = d // n_heads\n    def split(W):                                  # (T,d) -> (H,T,dh)\n        return (X @ W).reshape(T, n_heads, dh).transpose(1, 0, 2)\n    Q, K, V = split(Wq), split(Wk), split(Wv)\n    S = Q @ K.transpose(0, 2, 1) / np.sqrt(dh)     # (H,T,T), scale by head dim\n    if causal:\n        mask = np.tril(np.ones((T, T), dtype=bool))\n        S = np.where(mask, S, -np.inf)             # broadcasts over heads\n    out = softmax(S, axis=-1) @ V                  # (H,T,dh)\n    out = out.transpose(1, 0, 2).reshape(T, d)     # concat heads\n    return out @ Wo"
    },
    {
      "id": "w1s4",
      "title": "KV-cache decoding",
      "prompt": "Last one, and this is the one I care about for inference: incremental decoding with a **KV cache**. Back to single-head for simplicity.\n\n```\ndef decode_step(x_t, Wq, Wk, Wv, cache):\n```\n`x_t` is one token's representation, shape `(d_model,)`. `cache` is a dict `{\"K\": ..., \"V\": ...}` holding all previous keys/values (both `None` on the first call). Compute this step's attention output over the full history *without* recomputing old K/V, and return `(out, cache)` where `out` has shape `(d_v,)` and `cache` is updated.\n\nFeeding tokens one at a time must reproduce exactly what full causal attention would produce at each position.",
      "starter": None,
      "hints": [
        "This step only needs ONE query: q = x_t @ Wq, shape (1, d_k). Compute this step's k, v, append them to the cache (np.vstack), then attend: softmax(q @ K_all.T / sqrt(d_k)) @ V_all.",
        "No mask needed — the cache only contains the past, so attending over everything in it IS causal attention. Handle the first step: if cache['K'] is None, the new k, v become the cache."
      ],
      "probe": {
        "q": "For a 32-layer model with d_model 4096, GQA with 8 KV heads of dim 128, fp16, what's the KV-cache size per token, and why does this dominate long-rollout agent inference?",
        "a": "Per token: 32 layers × 2 (K and V) × 8 heads × 128 dim × 2 bytes = 128 KB/token — a 100k-token agent trajectory is ~13 GB per sequence. Decode is memory-bandwidth-bound: every step re-reads the whole cache, so cache size sets both max batch size and step latency. That's exactly why GQA/MQA, sliding windows, and cache quantization exist."
      },
      "tests": """
def _sm(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def _ref_full_causal(X, Wq, Wk, Wv):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    S = Q @ K.T / np.sqrt(Wk.shape[1])
    T = S.shape[0]
    S = np.where(np.tril(np.ones((T, T), dtype=bool)), S, -np.inf)
    return _sm(S) @ V

def _t_incremental():
    rng = np.random.default_rng(9)
    T, d, dk = 6, 8, 4
    X = rng.standard_normal((T, d))
    Wq = rng.standard_normal((d, dk)); Wk = rng.standard_normal((d, dk)); Wv = rng.standard_normal((d, dk))
    full = _ref_full_causal(X, Wq, Wk, Wv)
    cache = {"K": None, "V": None}
    for t in range(T):
        out, cache = decode_step(X[t], Wq, Wk, Wv, cache)
        out = np.asarray(out).reshape(-1)
        assert np.allclose(out, full[t], atol=1e-5), f"step {t}: incremental output doesn't match row {t} of full causal attention"
_check("incremental decode == full causal attention", _t_incremental)

def _t_cache_grows():
    rng = np.random.default_rng(10)
    d, dk = 8, 4
    Wq = rng.standard_normal((d, dk)); Wk = rng.standard_normal((d, dk)); Wv = rng.standard_normal((d, dk))
    cache = {"K": None, "V": None}
    for t in range(3):
        _, cache = decode_step(rng.standard_normal(d), Wq, Wk, Wv, cache)
    assert cache["K"] is not None and cache["V"] is not None, "cache not populated"
    assert cache["K"].shape == (3, dk), f"after 3 steps cache K should be (3, {dk}), got {cache['K'].shape}"
_check("cache accumulates one K,V row per step", _t_cache_grows)
""",
      "solution": "def decode_step(x_t, Wq, Wk, Wv, cache):\n    q = (x_t @ Wq)[None, :]                        # (1, dk) -- only new query\n    k = (x_t @ Wk)[None, :]\n    v = (x_t @ Wv)[None, :]\n    K = k if cache[\"K\"] is None else np.vstack([cache[\"K\"], k])\n    V = v if cache[\"V\"] is None else np.vstack([cache[\"V\"], v])\n    # cache holds only the past -> attending over all of it IS causal\n    A = softmax(q @ K.T / np.sqrt(K.shape[1]), axis=-1)   # (1, t+1)\n    out = (A @ V)[0]                               # (dv,)\n    return out, {\"K\": K, \"V\": V}"
    }
  ]
},

# =====================================================================
# W2 — Norms & backprop ladder
# =====================================================================
{
  "id": "w2",
  "interviewer": "W",
  "persona": "Interviewer W — rigorous, efficiency-minded. This session probes whether you can differentiate what you build.",
  "title": "Norms & backprop by hand",
  "minutes": 45,
  "intro": "Hi again. Today I want to see the mechanics under the hood — we'll implement normalization layers and then differentiate a small network by hand. NumPy only. Correctness first, then we'll talk about efficiency.",
  "closing": "Good session — that finite-difference habit will serve you well. Any questions for me?",
  "stages": [
    {
      "id": "w2s1",
      "title": "RMSNorm",
      "prompt": "Warm-up: implement **RMSNorm** (what Llama/most modern LLMs use).\n\n```\ndef rmsnorm(x, g, eps=1e-6):\n```\n`x` can be any shape ending in the feature dim `(..., d)`; `g` is a learned gain of shape `(d,)`. Normalize over the **last** axis only. No mean subtraction — that's the point of RMSNorm.",
      "starter": "import numpy as np\n\ndef rmsnorm(x, g, eps=1e-6):\n    pass\n",
      "hints": [
        "rms = sqrt(mean(x**2, axis=-1, keepdims=True) + eps). Return x / rms * g.",
        "keepdims=True is the whole game — without it the division broadcasts wrong for 2D/3D inputs."
      ],
      "probe": {
        "q": "Why did RMSNorm replace LayerNorm in modern LLMs?",
        "a": "It drops the mean-subtraction and the bias — fewer ops and one less reduction over the feature dim (~10–30% cheaper norm), and empirically the re-centering isn't needed for transformer quality: the re-scaling invariance is what stabilizes training. Simpler kernel, same or better loss."
      },
      "tests": """
def _ref_rms(x, g, eps=1e-6):
    return x / np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps) * g

def _t_1d():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(8); g = rng.standard_normal(8)
    assert np.allclose(rmsnorm(x, g), _ref_rms(x, g), atol=1e-6), "wrong on a 1-D vector"
_check("correct on a 1-D vector", _t_1d)

def _t_3d():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 3, 8)) + 5.0   # nonzero mean on purpose
    g = rng.standard_normal(8)
    got = rmsnorm(x, g)
    assert got.shape == x.shape, f"shape changed: {got.shape}"
    assert np.allclose(got, _ref_rms(x, g), atol=1e-6), "wrong on (B,T,d) input -- normalize over the LAST axis only, and don't subtract the mean"
_check("correct on (B,T,d) with nonzero mean", _t_3d)
""",
      "solution": "def rmsnorm(x, g, eps=1e-6):\n    rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)\n    return x / rms * g"
    },
    {
      "id": "w2s2",
      "title": "LayerNorm",
      "prompt": "Now classic **LayerNorm** for comparison:\n```\ndef layernorm(x, g, b, eps=1e-5):\n```\nSame shape rules; `g` and `b` are `(d,)`. Subtract the mean, divide by the standard deviation (biased/population variance), then scale and shift.",
      "starter": None,
      "hints": [
        "mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True) (NumPy's default is the biased estimator, which is what you want). Return (x - mu) / sqrt(var + eps) * g + b.",
        "eps goes INSIDE the sqrt, added to the variance — not outside."
      ],
      "probe": {
        "q": "Pre-norm vs post-norm — why did everyone move to pre-norm?",
        "a": "Post-norm (original Transformer) puts LN after the residual add, so the residual path itself is normalized — gradients must pass through every LN, and deep stacks need careful warmup or they diverge. Pre-norm keeps a clean identity residual path (x + f(LN(x))), giving well-behaved gradients at any depth — trains stably without warmup tricks. Cost: slightly weaker final performance per param, which is why some recent models revisit hybrid schemes."
      },
      "tests": """
def _ref_ln(x, g, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b

def _t_ln():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 6)) * 3 + 2
    g = rng.standard_normal(6); b = rng.standard_normal(6)
    assert np.allclose(layernorm(x, g, b), _ref_ln(x, g, b), atol=1e-6), "doesn't match reference (biased variance, eps inside sqrt)"
_check("matches reference LayerNorm", _t_ln)

def _t_ln_stats():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((5, 16)) * 7 - 4
    g = np.ones(16); b = np.zeros(16)
    out = layernorm(x, g, b)
    assert np.allclose(out.mean(axis=-1), 0, atol=1e-6), "with g=1,b=0 each row should have mean 0"
    assert np.allclose(out.std(axis=-1), 1, atol=1e-3), "with g=1,b=0 each row should have std ~1"
_check("normalized rows have mean 0, std 1", _t_ln_stats)
""",
      "solution": "def layernorm(x, g, b, eps=1e-5):\n    mu = x.mean(axis=-1, keepdims=True)\n    var = x.var(axis=-1, keepdims=True)      # biased (population) variance\n    return (x - mu) / np.sqrt(var + eps) * g + b"
    },
    {
      "id": "w2s3",
      "title": "Softmax cross-entropy + its gradient",
      "prompt": "Now the part I really want to see. Implement softmax cross-entropy **and its gradient with respect to the logits**, by hand:\n```\ndef softmax_xent(logits, y):\n    # logits: (B, C) float, y: (B,) int class labels\n    # returns (loss, dlogits)\n```\n`loss` is the mean over the batch; `dlogits` is `(B, C)`, the gradient of that mean loss. Must be numerically stable for large logits.\n\nBefore you code: what's the closed form of the gradient?",
      "starter": None,
      "hints": [
        "The famous result: dlogits = (softmax(logits) - onehot(y)) / B. Derive it from loss = -log p_y with p = softmax: dp_y/dlogit_j passes through softmax's Jacobian and everything collapses.",
        "For stable loss use log-softmax: z = logits - max; logp = z - log(sum(exp(z))). Loss = -logp[range(B), y].mean(). Never compute log(softmax) as two separate steps."
      ],
      "probe": {
        "q": "Sketch the derivation of dL/dlogits = p − y (one-hot).",
        "a": "L = −z_y + log∑_j e^{z_j}. Then ∂L/∂z_k = −𝟙[k=y] + e^{z_k}/∑_j e^{z_j} = p_k − 𝟙[k=y]. The softmax Jacobian and the log cancel into this clean form — which is why frameworks fuse softmax+CE into one op (better numerics AND a trivial backward)."
      },
      "tests": """
def _ref_loss(logits, y):
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return -logp[np.arange(len(y)), y].mean()

def _t_loss():
    rng = np.random.default_rng(4)
    logits = rng.standard_normal((6, 5)) * 2
    y = rng.integers(0, 5, 6)
    loss, _ = softmax_xent(logits, y)
    assert np.isclose(loss, _ref_loss(logits, y), atol=1e-6), "loss value wrong"
_check("loss matches reference", _t_loss)

def _t_loss_stable():
    logits = np.array([[1000.0, 0.0, -1000.0], [500.0, 499.0, -500.0]])
    y = np.array([0, 1])
    loss, dl = softmax_xent(logits, y)
    assert np.isfinite(loss), "loss overflows on large logits -- use the log-sum-exp trick"
    assert np.isfinite(dl).all(), "gradient has NaN/inf on large logits"
_check("stable for logits ~1000", _t_loss_stable)

def _t_grad():
    rng = np.random.default_rng(5)
    logits = rng.standard_normal((4, 5)); y = rng.integers(0, 5, 4)
    _, dl = softmax_xent(logits, y)
    assert dl.shape == logits.shape, f"dlogits shape {dl.shape} != {logits.shape}"
    num = np.zeros_like(logits); h = 1e-5
    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            lp = logits.copy(); lp[i, j] += h
            lm = logits.copy(); lm[i, j] -= h
            num[i, j] = (_ref_loss(lp, y) - _ref_loss(lm, y)) / (2 * h)
    assert np.allclose(dl, num, atol=1e-4), "analytic gradient doesn't match finite differences (did you divide by B?)"
_check("gradient matches finite differences", _t_grad)
""",
      "solution": "def softmax_xent(logits, y):\n    B = logits.shape[0]\n    z = logits - logits.max(axis=1, keepdims=True)       # stability\n    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))\n    loss = -logp[np.arange(B), y].mean()\n    p = np.exp(logp)\n    dlogits = p.copy()\n    dlogits[np.arange(B), y] -= 1.0                       # p - onehot\n    dlogits /= B                                          # grad of the MEAN\n    return loss, dlogits"
    },
    {
      "id": "w2s4",
      "title": "2-layer MLP backward pass",
      "prompt": "Final stage: full backward pass for a 2-layer MLP classifier, by hand. The forward is fixed:\n```\nh_pre  = X @ W1 + b1        # (B, H)\nh      = np.maximum(h_pre, 0)  # ReLU\nlogits = h @ W2 + b2        # (B, C)\nloss   = softmax-CE mean over batch (as before)\n```\nImplement:\n```\ndef mlp_grads(X, y, W1, b1, W2, b2):\n    # returns {\"dW1\": ..., \"db1\": ..., \"dW2\": ..., \"db2\": ...}\n```\nI'll check every gradient against finite differences, so shapes AND values.",
      "starter": None,
      "hints": [
        "Chain rule backwards from dlogits = (p - onehot)/B: dW2 = h.T @ dlogits, db2 = dlogits.sum(0), dh = dlogits @ W2.T.",
        "Through ReLU: dh_pre = dh * (h_pre > 0). Then dW1 = X.T @ dh_pre, db1 = dh_pre.sum(0). Keep a shape table as you go: dlogits (B,C), dh (B,H), dW1 (D,H)."
      ],
      "probe": {
        "q": "Your loss won't go down after you wired this into SGD. Give your first three debugging steps, in order.",
        "a": "1) Finite-difference check each gradient on a tiny batch (exactly what these tests do) — isolates a wrong formula. 2) Overfit a single batch of ~10 examples to ~zero loss — if it can't, the model/loss wiring is broken, not the data. 3) Check the loss at init ≈ log(C) — catches bad init, wrong label indexing, or a softmax over the wrong axis. Then: learning rate sweep, gradient norms per layer for vanishing/exploding."
      },
      "tests": """
def _fwd_loss(X, y, W1, b1, W2, b2):
    h = np.maximum(X @ W1 + b1, 0)
    logits = h @ W2 + b2
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return -logp[np.arange(len(y)), y].mean()

def _t_backward():
    rng = np.random.default_rng(6)
    B, D, H, C = 4, 3, 5, 4
    X = rng.standard_normal((B, D))
    y = rng.integers(0, C, B)
    W1 = rng.standard_normal((D, H)) * 0.5; b1 = rng.standard_normal(H) * 0.1
    W2 = rng.standard_normal((H, C)) * 0.5; b2 = rng.standard_normal(C) * 0.1
    grads = mlp_grads(X, y, W1, b1, W2, b2)
    params = {"dW1": W1, "db1": b1, "dW2": W2, "db2": b2}
    h = 1e-5
    for name, P in params.items():
        assert name in grads, f"missing key {name!r} in returned dict"
        G = np.asarray(grads[name])
        assert G.shape == P.shape, f"{name} shape {G.shape} != param shape {P.shape}"
        num = np.zeros_like(P)
        it = np.nditer(P, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            old = P[idx]
            P[idx] = old + h; lp = _fwd_loss(X, y, W1, b1, W2, b2)
            P[idx] = old - h; lm = _fwd_loss(X, y, W1, b1, W2, b2)
            P[idx] = old
            num[idx] = (lp - lm) / (2 * h)
            it.iternext()
        assert np.allclose(G, num, atol=1e-4), f"{name} doesn't match finite differences"
_check("all four gradients match finite differences", _t_backward)
""",
      "solution": "def mlp_grads(X, y, W1, b1, W2, b2):\n    B = X.shape[0]\n    # forward (keep intermediates)\n    h_pre = X @ W1 + b1\n    h = np.maximum(h_pre, 0)\n    logits = h @ W2 + b2\n    z = logits - logits.max(axis=1, keepdims=True)\n    p = np.exp(z); p /= p.sum(axis=1, keepdims=True)\n    # backward\n    dlogits = p.copy()\n    dlogits[np.arange(B), y] -= 1.0\n    dlogits /= B                          # (B,C)\n    dW2 = h.T @ dlogits                   # (H,C)\n    db2 = dlogits.sum(axis=0)             # (C,)\n    dh = dlogits @ W2.T                   # (B,H)\n    dh_pre = dh * (h_pre > 0)             # ReLU gate\n    dW1 = X.T @ dh_pre                    # (D,H)\n    db1 = dh_pre.sum(axis=0)              # (H,)\n    return {\"dW1\": dW1, \"db1\": db1, \"dW2\": dW2, \"db2\": db2}"
    }
  ]
},

# =====================================================================
# F1 — Sampling & decoding ladder
# =====================================================================
{
  "id": "f1",
  "interviewer": "F",
  "persona": "Interviewer F — post-training/RL/agents. Cares about the WHY behind each knob and its effect on model behavior.",
  "title": "Sampling & decoding",
  "minutes": 45,
  "intro": "Hey, good to meet you. I work on post-training, so today is about the sampling stack — the thing between logits and tokens. We'll build it up piece by piece. As you code, tell me how each knob changes model behavior; that matters to me as much as the code.",
  "closing": "Nice — you clearly know your way around a sampler. Any questions about what we're working on?",
  "stages": [
    {
      "id": "f1s1",
      "title": "Temperature sampling",
      "prompt": "Start with the distribution itself:\n```\ndef sample_probs(logits, temperature=1.0):\n```\n`logits` is a 1-D float array over the vocab. Return the **probability distribution** you'd sample from (we test the distribution, not a random draw). Rules:\n- `temperature=0` means greedy: a one-hot on the argmax.\n- otherwise softmax of `logits / temperature`, numerically stable for large logits.",
      "starter": "import numpy as np\n\ndef sample_probs(logits, temperature=1.0):\n    pass\n",
      "hints": [
        "Handle temperature == 0 first (np.zeros + p[argmax] = 1), THEN divide — otherwise you divide by zero.",
        "Stable softmax: z = logits/T; z -= z.max(); p = exp(z); p /= p.sum()."
      ],
      "probe": {
        "q": "What does temperature actually do to the distribution, and when would you run T > 1 vs T → 0 in a post-training pipeline?",
        "a": "T rescales logit gaps: T<1 sharpens toward the mode, T>1 flattens toward uniform; T→0 is argmax. In RLHF/GRPO rollouts you sample T≈1 (or higher) for diverse exploration — diversity is what gives the advantage signal; for evals/reward-scoring or agentic tool calls you drop toward T=0 for determinism and precision. Best-of-n also wants T high enough that the n samples actually differ."
      },
      "tests": """
def _sm(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def _t_t1():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal(10) * 2
    assert np.allclose(sample_probs(logits, 1.0), _sm(logits), atol=1e-6), "temperature=1 should be plain softmax"
_check("T=1 is plain softmax", _t_t1)

def _t_thalf():
    rng = np.random.default_rng(1)
    logits = rng.standard_normal(8)
    assert np.allclose(sample_probs(logits, 0.5), _sm(logits / 0.5), atol=1e-6), "temperature should DIVIDE the logits"
_check("T=0.5 sharpens correctly", _t_thalf)

def _t_t0():
    logits = np.array([0.1, 2.5, -1.0, 2.4])
    p = sample_probs(logits, 0.0)
    exp = np.zeros(4); exp[1] = 1.0
    assert np.allclose(p, exp), "temperature=0 should be a one-hot on the argmax"
_check("T=0 is greedy one-hot", _t_t0)

def _t_stable():
    logits = np.array([1000.0, 999.0, 0.0])
    p = sample_probs(logits, 1.0)
    assert np.isfinite(p).all(), "overflow on large logits -- subtract the max"
    assert np.isclose(p.sum(), 1.0, atol=1e-6), "probabilities must sum to 1"
_check("stable on logits ~1000", _t_stable)
""",
      "solution": "def sample_probs(logits, temperature=1.0):\n    if temperature == 0:\n        p = np.zeros_like(logits, dtype=float)\n        p[np.argmax(logits)] = 1.0\n        return p\n    z = logits / temperature\n    z = z - z.max()                # stability\n    p = np.exp(z)\n    return p / p.sum()"
    },
    {
      "id": "f1s2",
      "title": "Top-k and top-p (nucleus)",
      "prompt": "Now the filters. Extend to:\n```\ndef sample_probs(logits, temperature=1.0, top_k=None, top_p=None):\n```\nOrder of operations: temperature softmax → top-k → top-p. Definitions:\n- **top-k**: keep only the `k` highest-probability tokens, renormalize.\n- **top-p**: sort descending, keep the **smallest** prefix whose cumulative probability is `≥ p` (always at least 1 token), zero the rest, renormalize.\n\nBoth can be active at once.",
      "starter": None,
      "hints": [
        "Top-k: idx = np.argsort(-p)[:k]; build a new zero array, copy p[idx] in, renormalize. Don't sort p itself — you need original vocab positions.",
        "Top-p: order = np.argsort(-p); cs = np.cumsum(p[order]); the cutoff count is np.searchsorted(cs, top_p) + 1 (first index where cumsum >= p). Keep order[:count], zero the rest, renormalize."
      ],
      "probe": {
        "q": "Why did nucleus (top-p) largely replace top-k as the default?",
        "a": "A fixed k ignores the shape of the distribution: when the model is confident, k=50 admits garbage from the flat tail; when it's genuinely uncertain, k=50 may cut valid options. Top-p adapts — it keeps whatever number of tokens covers p probability mass, so it truncates the unreliable tail (where the model's probability estimates are worst) while preserving calibrated uncertainty. Min-p pushes the same idea further using a threshold relative to the top token."
      },
      "tests": """
def _sm(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def _ref_sp(logits, temperature=1.0, top_k=None, top_p=None):
    if temperature == 0:
        p = np.zeros_like(logits, dtype=float); p[np.argmax(logits)] = 1.0; return p
    p = _sm(logits / temperature)
    if top_k is not None:
        idx = np.argsort(-p)[:top_k]
        m = np.zeros_like(p); m[idx] = p[idx]; p = m / m.sum()
    if top_p is not None:
        order = np.argsort(-p)
        cs = np.cumsum(p[order])
        cnt = int(np.searchsorted(cs, top_p)) + 1
        keep = order[:cnt]
        m = np.zeros_like(p); m[keep] = p[keep]; p = m / m.sum()
    return p

_LOGITS = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0])

def _t_topk():
    got = sample_probs(_LOGITS, 1.0, top_k=2)
    exp = _ref_sp(_LOGITS, 1.0, top_k=2)
    assert np.isclose(got.sum(), 1.0, atol=1e-6), "must renormalize after filtering"
    assert (got[2:] == 0).all(), "only the top 2 tokens should have nonzero probability"
    assert np.allclose(got, exp, atol=1e-6), "top-k values wrong"
_check("top-k keeps k best and renormalizes", _t_topk)

def _t_topp():
    got = sample_probs(_LOGITS, 1.0, top_p=0.8)
    exp = _ref_sp(_LOGITS, 1.0, top_p=0.8)
    assert np.allclose(got, exp, atol=1e-6), "top-p cutoff wrong: keep the SMALLEST prefix with cumsum >= p"
_check("top-p nucleus cutoff correct", _t_topp)

def _t_topp_one_token():
    logits = np.array([10.0, 0.0, 0.0, 0.0])
    got = sample_probs(logits, 1.0, top_p=0.1)
    assert np.isclose(got[0], 1.0, atol=1e-6), "even with tiny p, at least one token must survive"
_check("top-p always keeps >= 1 token", _t_topp_one_token)

def _t_combined():
    got = sample_probs(_LOGITS, 0.7, top_k=4, top_p=0.9)
    exp = _ref_sp(_LOGITS, 0.7, top_k=4, top_p=0.9)
    assert np.allclose(got, exp, atol=1e-6), "temperature -> top-k -> top-p combination wrong"
_check("temperature + top-k + top-p compose", _t_combined)
""",
      "solution": "def sample_probs(logits, temperature=1.0, top_k=None, top_p=None):\n    if temperature == 0:\n        p = np.zeros_like(logits, dtype=float)\n        p[np.argmax(logits)] = 1.0\n        return p\n    z = logits / temperature\n    z = z - z.max()\n    p = np.exp(z); p = p / p.sum()\n    if top_k is not None:\n        idx = np.argsort(-p)[:top_k]          # original vocab positions\n        m = np.zeros_like(p); m[idx] = p[idx]\n        p = m / m.sum()\n    if top_p is not None:\n        order = np.argsort(-p)\n        cs = np.cumsum(p[order])\n        cnt = int(np.searchsorted(cs, top_p)) + 1   # smallest prefix >= p\n        keep = order[:cnt]\n        m = np.zeros_like(p); m[keep] = p[keep]\n        p = m / m.sum()\n    return p"
    },
    {
      "id": "f1s3",
      "title": "Repetition penalty",
      "prompt": "Models loop. Add the CTRL-style **repetition penalty**, applied to logits *before* sampling:\n```\ndef apply_repetition_penalty(logits, generated_ids, penalty):\n```\nFor every token id that already appears in `generated_ids`: if its logit is positive, divide by `penalty`; if negative, multiply by `penalty` (`penalty > 1` discourages repeats in both cases). Return new logits — do **not** mutate the input. Tokens repeated multiple times are penalized once.",
      "starter": None,
      "hints": [
        "Iterate over set(generated_ids) (or np.unique). Copy the logits first — logits.copy().",
        "The asymmetry is the trap: dividing a NEGATIVE logit by penalty>1 makes it LESS negative, i.e. MORE likely. That's why negative logits multiply instead."
      ],
      "probe": {
        "q": "Why does the penalty divide positive logits but multiply negative ones? And what's one failure mode of repetition penalties?",
        "a": "Both operations push the logit toward more-negative preference: dividing shrinks positive logits toward 0; multiplying makes negative logits more negative. A single rule (always divide) would BOOST already-unlikely repeated tokens. Failure mode: it penalizes legitimately repeated tokens too — code identifiers, structured output, the word 'the' — which is why frequency/presence penalties or DRY-style n-gram penalties are often preferred."
      },
      "tests": """
def _t_basic():
    logits = np.array([2.0, -1.0, 0.5, 3.0])
    out = apply_repetition_penalty(logits, [0, 2, 2], 1.5)
    exp = np.array([2.0 / 1.5, -1.0, 0.5 / 1.5, 3.0])
    assert np.allclose(out, exp, atol=1e-8), "positive seen logits should be divided by the penalty; unseen logits untouched"
_check("positive logits divided by penalty", _t_basic)

def _t_negative():
    logits = np.array([-2.0, 1.0, -0.5])
    out = apply_repetition_penalty(logits, [0, 2], 2.0)
    exp = np.array([-4.0, 1.0, -1.0])
    assert np.allclose(out, exp, atol=1e-8), "negative seen logits should be MULTIPLIED by the penalty (more negative = less likely)"
_check("negative logits multiplied by penalty", _t_negative)

def _t_no_mutation():
    logits = np.array([2.0, -1.0, 0.5])
    orig = logits.copy()
    apply_repetition_penalty(logits, [0, 1], 1.5)
    assert np.array_equal(logits, orig), "input logits were mutated -- return a copy"
_check("input logits not mutated", _t_no_mutation)
""",
      "solution": "def apply_repetition_penalty(logits, generated_ids, penalty):\n    out = np.asarray(logits, dtype=float).copy()\n    for t in set(generated_ids):\n        if out[t] > 0:\n            out[t] /= penalty      # shrink toward 0\n        else:\n            out[t] *= penalty      # push further negative\n    return out"
    },
    {
      "id": "f1s4",
      "title": "Best-of-n with a reward model",
      "prompt": "Last piece — the simplest inference-time alignment method, **best-of-n**:\n```\ndef best_of_n(generate, reward, n, rng):\n```\n`generate(rng)` returns one sampled trajectory (a list of token ids); `reward(tokens)` scores it (float, deterministic). Draw exactly `n` samples by calling `generate(rng)` `n` times in order (nothing else touches `rng`), score each, and return `(best_tokens, best_reward)`. Ties go to the **earliest** sample.",
      "starter": None,
      "hints": [
        "Straight loop: call generate(rng) n times, keep (tokens, r) with the running max. Use strict > for the update so ties keep the earliest.",
        "Don't collect-then-argmax with >= — the tie rule is 'earliest wins', so only replace the champion when the new reward is strictly greater."
      ],
      "probe": {
        "q": "How does best-of-n relate to RLHF, and what happens as n grows large?",
        "a": "Best-of-n IS a form of policy improvement against the reward model without touching weights — its KL from the base policy is roughly log(n) − (n−1)/n, so it's a gentle, analyzable optimizer. As n grows you increasingly hit reward-model errors: Goodhart/over-optimization — samples that exploit RM blind spots outscore genuinely good ones. It's also the workhorse for distillation (rejection sampling / RAFT: SFT on best-of-n outputs) and for test-time compute with verifiers."
      },
      "tests": """
def _mk_generate():
    def generate(rng):
        L = int(rng.integers(3, 8))
        return [int(t) for t in rng.integers(0, 50, L)]
    return generate

def _reward_sum(toks):
    return -abs(sum(toks) - 100)

def _t_bon():
    gen = _mk_generate()
    got_t, got_r = best_of_n(gen, _reward_sum, 6, np.random.default_rng(11))
    # reference run with identical rng sequence
    rng = np.random.default_rng(11)
    best_t, best_r = None, -1e18
    for _ in range(6):
        t = gen(rng); r = _reward_sum(t)
        if r > best_r:
            best_t, best_r = t, r
    assert list(got_t) == best_t, "returned trajectory isn't the highest-reward one (or rng was consumed out of order -- call generate(rng) exactly n times)"
    assert np.isclose(got_r, best_r), f"returned reward {got_r} != expected {best_r}"
_check("returns the argmax-reward sample", _t_bon)

def _t_ties_earliest():
    calls = []
    def gen(rng):
        t = [len(calls)]; calls.append(1); return t
    got_t, got_r = best_of_n(gen, lambda toks: 7.0, 5, np.random.default_rng(0))
    assert list(got_t) == [0], "on ties, the EARLIEST sample must win (use strict > when updating the champion)"
    assert np.isclose(got_r, 7.0)
_check("ties resolved to earliest sample", _t_ties_earliest)
""",
      "solution": "def best_of_n(generate, reward, n, rng):\n    best_t, best_r = None, -float(\"inf\")\n    for _ in range(n):\n        t = generate(rng)\n        r = reward(t)\n        if r > best_r:          # strict >: earliest wins ties\n            best_t, best_r = t, r\n    return best_t, best_r"
    }
  ]
},

# =====================================================================
# F2 — Post-training losses ladder
# =====================================================================
{
  "id": "f2",
  "interviewer": "F",
  "persona": "Interviewer F — post-training/RL/agents. This session walks the RLHF stack: reward model → DPO → PPO → GRPO.",
  "title": "Post-training losses (RM → DPO → PPO → GRPO)",
  "minutes": 45,
  "intro": "Welcome back. Today we implement the actual objectives of the alignment stack, in NumPy — values only, no autograd. I care a lot about numerical stability here; these losses blow up in fun ways at scale. Narrate the math as you write it.",
  "closing": "That was the whole stack — solid. Anything you want to ask me?",
  "stages": [
    {
      "id": "f2s1",
      "title": "Reward model: Bradley–Terry loss",
      "prompt": "We train a reward model on preference pairs with the **Bradley–Terry** objective:\n```\ndef rm_loss(r_chosen, r_rejected):\n```\nBoth inputs are `(B,)` arrays of scalar rewards. Return the mean of `-log σ(r_chosen - r_rejected)`.\n\nOne requirement: it must be exact for reward gaps like ±1000 — think about what `exp` does there.",
      "starter": "import numpy as np\n\ndef rm_loss(r_chosen, r_rejected):\n    pass\n",
      "hints": [
        "-log σ(x) = log(1 + e^{-x}) = softplus(-x). The naive form overflows when x is very negative.",
        "np.logaddexp(0, -x) computes log(1 + e^{-x}) stably for any x. That one-liner is the whole stage."
      ],
      "probe": {
        "q": "Why train the RM on pairwise comparisons instead of absolute scores, and what does the Bradley–Terry model assume?",
        "a": "Humans are inconsistent at absolute scoring (calibration drifts across annotators and time) but reliable at ranking two candidates side by side. Bradley–Terry assumes each item has a latent scalar quality and P(a ≻ b) = σ(r_a − r_b) — which means only reward DIFFERENCES are identified; the absolute scale is free, so RM scores need normalization before use in PPO (hence reward whitening / KL anchoring)."
      },
      "tests": """
def _t_value():
    r_c = np.array([1.0, 2.0, 0.0])
    r_r = np.array([0.0, 2.0, 1.0])
    got = rm_loss(r_c, r_r)
    exp = np.mean(np.logaddexp(0.0, -(r_c - r_r)))
    assert np.isclose(got, exp, atol=1e-8), f"loss {got} != expected {exp} (mean of -log sigmoid(r_c - r_r))"
_check("loss value correct (incl. tie -> log 2)", _t_value)

def _t_stable_pos():
    got = rm_loss(np.array([1000.0]), np.array([0.0]))
    assert np.isfinite(got) and got < 1e-6, "huge positive margin should give ~0 loss, finite"
_check("stable at margin +1000", _t_stable_pos)

def _t_stable_neg():
    got = rm_loss(np.array([0.0]), np.array([1000.0]))
    assert np.isfinite(got), "loss overflowed at margin -1000 -- use logaddexp / softplus"
    assert np.isclose(got, 1000.0, rtol=1e-6), "at margin -1000 the loss should be ~1000 (softplus is ~linear there)"
_check("stable and ~linear at margin -1000", _t_stable_neg)
""",
      "solution": "def rm_loss(r_chosen, r_rejected):\n    margin = r_chosen - r_rejected\n    # -log sigmoid(m) = log(1 + e^{-m}) = logaddexp(0, -m)  [stable]\n    return np.mean(np.logaddexp(0.0, -margin))"
    },
    {
      "id": "f2s2",
      "title": "DPO loss",
      "prompt": "Now skip the RM entirely — **DPO**:\n```\ndef dpo_loss(logp_c, logp_r, ref_c, ref_r, beta=0.1):\n    # returns (loss, margin)\n```\nInputs are `(B,)` summed log-probs of chosen/rejected responses under the **policy** (`logp_*`) and the frozen **reference** (`ref_*`).\n\n- `loss` = mean of `-log σ(β · [(logp_c - ref_c) - (logp_r - ref_r)])`\n- `margin` = mean of that β-scaled term inside the sigmoid (the implicit reward margin — what you'd log in training).\n\nSame stability requirement as before.",
      "starter": None,
      "hints": [
        "Build h = beta * ((logp_c - ref_c) - (logp_r - ref_r)); loss = np.logaddexp(0, -h).mean(); margin = h.mean().",
        "Watch the grouping: it's (policy - reference) per response, THEN chosen-minus-rejected. Getting the pairing wrong silently flips the sign of learning."
      ],
      "probe": {
        "q": "What role does the reference model play in DPO, and what happens as β → 0?",
        "a": "The (logp − ref) terms are the implicit reward r(x,y) = β log π/π_ref: the reference anchors the objective so DPO maximizes preference fit under a KL constraint to π_ref — exactly the RLHF objective's closed-form solution. β sets the constraint strength: small β = weak KL anchor, the policy drifts far from the reference chasing preferences (degeneration, likelihood of chosen can even FALL while the margin grows); large β barely moves. β→0 in the math recovers an unconstrained preference optimizer."
      },
      "tests": """
def _ref_dpo(pc, pr, rc, rr, beta):
    h = beta * ((pc - rc) - (pr - rr))
    return np.logaddexp(0.0, -h).mean(), h.mean()

def _t_dpo():
    pc = np.array([-10.0, -5.0, -8.0]); pr = np.array([-12.0, -4.0, -8.0])
    rc = np.array([-11.0, -5.5, -7.0]); rr = np.array([-11.0, -5.0, -9.0])
    got_l, got_m = dpo_loss(pc, pr, rc, rr, beta=0.1)
    exp_l, exp_m = _ref_dpo(pc, pr, rc, rr, 0.1)
    assert np.isclose(got_l, exp_l, atol=1e-8), f"loss {got_l} != {exp_l} (check the (policy-ref) grouping)"
    assert np.isclose(got_m, exp_m, atol=1e-8), f"margin {got_m} != {exp_m}"
_check("loss and margin match reference", _t_dpo)

def _t_beta():
    pc = np.array([-5.0]); pr = np.array([-9.0]); rc = np.array([-6.0]); rr = np.array([-6.0])
    l_small, _ = dpo_loss(pc, pr, rc, rr, beta=0.01)
    l_big, _ = dpo_loss(pc, pr, rc, rr, beta=1.0)
    exp_small, _ = _ref_dpo(pc, pr, rc, rr, 0.01)
    exp_big, _ = _ref_dpo(pc, pr, rc, rr, 1.0)
    assert np.isclose(l_small, exp_small, atol=1e-8) and np.isclose(l_big, exp_big, atol=1e-8), "beta isn't applied inside the sigmoid correctly"
_check("beta scales the margin inside sigmoid", _t_beta)

def _t_dpo_stable():
    big = np.array([1e4]); zero = np.array([0.0])
    l, m = dpo_loss(zero, big, zero, zero, beta=1.0)
    assert np.isfinite(l), "overflow for extreme log-prob gaps -- logaddexp again"
_check("stable for extreme margins", _t_dpo_stable)
""",
      "solution": "def dpo_loss(logp_c, logp_r, ref_c, ref_r, beta=0.1):\n    # implicit rewards: beta * log(pi/pi_ref) per response\n    h = beta * ((logp_c - ref_c) - (logp_r - ref_r))\n    loss = np.logaddexp(0.0, -h).mean()   # -log sigmoid(h), stable\n    return loss, h.mean()"
    },
    {
      "id": "f2s3",
      "title": "PPO clipped objective",
      "prompt": "Back to on-policy RL — the **PPO clipped surrogate**:\n```\ndef ppo_loss(logp_new, logp_old, adv, clip_eps=0.2):\n```\nAll `(B,)` per-token arrays; `adv` are advantage estimates. With ratio `ρ = exp(logp_new - logp_old)`:\n\nloss = `-mean( min(ρ · A, clip(ρ, 1-ε, 1+ε) · A) )`\n\nGet the pessimism right for **negative** advantages too — that's where most hand-rolled PPOs are wrong.",
      "starter": None,
      "hints": [
        "ratio = np.exp(logp_new - logp_old); clipped = np.clip(ratio, 1-eps, 1+eps). Objective per element = np.minimum(ratio*adv, clipped*adv). Loss = -mean.",
        "Don't branch on the sign of A — np.minimum of the two products handles both cases. For A<0 the min picks the MORE NEGATIVE (unclipped) term when the ratio runs away, which is exactly the pessimistic bound."
      ],
      "probe": {
        "q": "What failure is the clip actually preventing, and why is taking the min (not just clipping the ratio) essential?",
        "a": "Off-policy drift within an epoch of minibatch updates: the surrogate ρ·A is only a valid local approximation near ρ=1; unconstrained, a few high-advantage tokens drive huge policy jumps and collapse. Clipping alone isn't enough because clip(ρ)·A would also LIMIT THE PENALTY when the update makes things worse (A<0, ρ large). The min keeps the objective a pessimistic LOWER bound: gradients stop when the move would overshoot in your favor, but never stop when you're getting worse."
      },
      "tests": """
def _t_hand():
    # single element, ratio=1.5, A=1, eps=0.2 -> objective min(1.5, 1.2) = 1.2, loss = -1.2
    l = ppo_loss(np.array([np.log(1.5)]), np.array([0.0]), np.array([1.0]))
    assert np.isclose(l, -1.2, atol=1e-6), f"hand-check failed: expected -1.2, got {l}"
_check("hand-checkable case: ratio 1.5, A=+1", _t_hand)

def _t_four_quadrants():
    logp_old = np.zeros(4)
    logp_new = np.log(np.array([1.5, 0.5, 1.5, 0.5]))
    adv = np.array([1.0, 1.0, -1.0, -1.0])
    # objectives: min(1.5,1.2)=1.2 | min(.5,.8)=.5 | min(-1.5,-1.2)=-1.5 | min(-.5,-.8)=-.8
    exp = -np.mean([1.2, 0.5, -1.5, -0.8])   # = 0.15
    got = ppo_loss(logp_new, logp_old, adv)
    assert np.isclose(got, exp, atol=1e-6), f"expected {exp}, got {got} -- check the negative-advantage cases (pessimism = the more negative term)"
_check("all four clip/sign quadrants correct", _t_four_quadrants)

def _t_no_clip_region():
    rng = np.random.default_rng(3)
    logp_old = rng.standard_normal(5) * 0.01
    logp_new = logp_old + rng.standard_normal(5) * 0.01   # ratios ~1, inside band
    adv = rng.standard_normal(5)
    ratio = np.exp(logp_new - logp_old)
    exp = -np.mean(ratio * adv)
    got = ppo_loss(logp_new, logp_old, adv)
    assert np.isclose(got, exp, atol=1e-8), "inside the trust region the loss should equal the unclipped surrogate"
_check("reduces to plain surrogate inside the band", _t_no_clip_region)
""",
      "solution": "def ppo_loss(logp_new, logp_old, adv, clip_eps=0.2):\n    ratio = np.exp(logp_new - logp_old)\n    clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)\n    # pessimistic bound: element-wise min handles BOTH signs of A\n    objective = np.minimum(ratio * adv, clipped * adv)\n    return -objective.mean()"
    },
    {
      "id": "f2s4",
      "title": "GRPO group-relative advantages",
      "prompt": "Finally, the trick behind DeepSeek-R1-style training — **GRPO** drops the value network and normalizes within a group of samples from the same prompt:\n```\ndef grpo_advantages(rewards, eps=1e-6):\n```\n`rewards` is `(P, N)` — `P` prompts, `N` sampled responses each. Return per-sample advantages: `(r - group_mean) / (group_std + eps)`, computed per row (population std). A zero-variance group (all N rewards equal) must return zeros, not NaN.",
      "starter": None,
      "hints": [
        "mean = rewards.mean(axis=1, keepdims=True); std = rewards.std(axis=1, keepdims=True). Return (rewards - mean) / (std + eps).",
        "keepdims=True keeps broadcasting per-row. The eps on std (not variance) is what makes the all-equal row give 0/(0+eps) = 0."
      ],
      "probe": {
        "q": "Why can GRPO drop the value network, and what subtle bias does the std normalization introduce?",
        "a": "The value net in PPO is just a variance-reduction baseline. With N samples per prompt, the group mean IS a Monte-Carlo baseline for that prompt — no learned critic needed, which saves a model's worth of memory and avoids critic-lag pathologies; the tradeoff is N× rollout cost per prompt. The std division introduces a difficulty bias: near-solved or near-impossible prompts (tiny reward variance) get their tiny advantage gaps amplified to ±1, overweighting uninformative prompts — which is why Dr. GRPO and others drop the std term."
      },
      "tests": """
def _ref_grpo(R, eps=1e-6):
    m = R.mean(axis=1, keepdims=True)
    s = R.std(axis=1, keepdims=True)
    return (R - m) / (s + eps)

def _t_grpo():
    R = np.array([[1.0, 0.0, 1.0, 0.0],
                  [0.3, 0.9, 0.6, 0.0]])
    got = grpo_advantages(R)
    exp = _ref_grpo(R)
    assert got.shape == R.shape, f"shape {got.shape} != {R.shape}"
    assert np.allclose(got, exp, atol=1e-6), "advantages don't match (per-ROW mean/std, population std, eps added to std)"
_check("group-relative advantages correct", _t_grpo)

def _t_zero_row_mean():
    R = np.array([[1.0, 2.0, 3.0]])
    got = grpo_advantages(R)
    assert np.isclose(got.mean(), 0.0, atol=1e-6), "each group's advantages must average to ~0"
_check("advantages are centered per group", _t_zero_row_mean)

def _t_degenerate():
    R = np.array([[0.5, 0.5, 0.5], [1.0, 0.0, 0.5]])
    got = grpo_advantages(R)
    assert np.isfinite(got).all(), "NaN on a zero-variance group -- eps goes on the std"
    assert np.allclose(got[0], 0.0, atol=1e-6), "an all-equal group carries no signal: advantages should be exactly 0"
_check("zero-variance group -> zeros, no NaN", _t_degenerate)
""",
      "solution": "def grpo_advantages(rewards, eps=1e-6):\n    mean = rewards.mean(axis=1, keepdims=True)   # per-prompt baseline\n    std = rewards.std(axis=1, keepdims=True)     # population std\n    return (rewards - mean) / (std + eps)        # eps on std: 0-var group -> 0"
    }
  ]
},

# =====================================================================
# A1 — Algorithmic warm-up (either round may open with one of these)
# =====================================================================
{
  "id": "a1",
  "interviewer": "A",
  "persona": "Generic algorithmic screen — clean Python, edge cases, complexity. Either coding round may open with one of these.",
  "title": "Algorithmic warm-up",
  "minutes": 30,
  "intro": "Quick warm-up before the ML material — two classic problems. Clean Python, state your complexity, handle the edge cases without being told.",
  "closing": "Good. That's the bar for the warm-up — fast, clean, no prompting on edge cases.",
  "stages": [
    {
      "id": "a1s1",
      "title": "Merge intervals",
      "prompt": "Given a list of intervals `[start, end]` (possibly unsorted, possibly touching), merge all overlapping or touching intervals:\n```\ndef merge_intervals(intervals):\n```\n`[[1,3],[2,6],[8,10]]` → `[[1,6],[8,10]]`. Touching counts: `[[1,4],[4,5]]` → `[[1,5]]`. Return a list of lists, sorted by start.",
      "starter": "def merge_intervals(intervals):\n    pass\n",
      "hints": [
        "Sort by start first. Then one pass: if the current interval's start <= the last merged interval's end, extend; else append.",
        "Extending must use max(last_end, cur_end) — the nested-interval case [[1,10],[2,3]] breaks naive overwriting."
      ],
      "probe": {
        "q": "Complexity? And could you do better if the input were already sorted?",
        "a": "O(n log n) from the sort, O(n) merge pass, O(n) output space. Already sorted → the sort drops and it's O(n) time, O(1) extra space if you may merge in place. Worth saying unprompted."
      },
      "tests": """
def _norm(out):
    return [list(x) for x in out]

def _t_basic():
    assert _norm(merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]])) == [[1, 6], [8, 10], [15, 18]]
_check("classic overlapping case", _t_basic)

def _t_touching():
    assert _norm(merge_intervals([[1, 4], [4, 5]])) == [[1, 5]], "touching intervals must merge"
_check("touching intervals merge", _t_touching)

def _t_unsorted_nested():
    assert _norm(merge_intervals([[5, 6], [1, 10], [2, 3]])) == [[1, 10]], "handle unsorted input and fully-nested intervals (use max of ends)"
_check("unsorted + nested intervals", _t_unsorted_nested)

def _t_empty():
    assert _norm(merge_intervals([])) == [], "empty input -> empty output"
_check("empty input", _t_empty)
""",
      "solution": "def merge_intervals(intervals):\n    if not intervals:\n        return []\n    intervals = sorted(intervals, key=lambda iv: iv[0])\n    out = [list(intervals[0])]\n    for s, e in intervals[1:]:\n        if s <= out[-1][1]:                    # overlap or touch\n            out[-1][1] = max(out[-1][1], e)    # max: nested case\n        else:\n            out.append([s, e])\n    return out"
    },
    {
      "id": "a1s2",
      "title": "Longest substring without repeats",
      "prompt": "Second one: length of the longest substring without repeating characters.\n```\ndef longest_unique(s):\n```\n`\"abcabcbb\"` → 3 (`\"abc\"`), `\"pwwkew\"` → 3 (`\"wke\"`). Aim for one pass.",
      "starter": None,
      "hints": [
        "Sliding window with a dict char -> last index. Advance right; when s[right] was seen inside the current window, jump left past it.",
        "The trap is \"abba\": when you see the second 'a', the stored index of 'a' is 0 — OUTSIDE the current window (left is already 2). Only move left forward: left = max(left, seen[c] + 1)."
      ],
      "probe": {
        "q": "Complexity, and why is the max() in the left-pointer update necessary?",
        "a": "O(n) time — each pointer only moves forward — and O(min(n, alphabet)) space. Without max(), a stale index from before the current window drags LEFT BACKWARD (\"abba\": seeing the 2nd 'a' would reset left from 2 to 1), re-admitting a duplicate and overcounting."
      },
      "tests": """
def _t_cases():
    cases = {"abcabcbb": 3, "bbbb": 1, "pwwkew": 3, "": 0, "abcdef": 6}
    for s, exp in cases.items():
        got = longest_unique(s)
        assert got == exp, f"longest_unique({s!r}) = {got}, expected {exp}"
_check("standard cases", _t_cases)

def _t_abba():
    got = longest_unique("abba")
    assert got == 2, f"longest_unique('abba') = {got}, expected 2 -- the left pointer must never move backward (left = max(left, seen[c]+1))"
_check("the 'abba' stale-index trap", _t_abba)
""",
      "solution": "def longest_unique(s):\n    seen = {}\n    left = 0\n    best = 0\n    for right, c in enumerate(s):\n        if c in seen:\n            left = max(left, seen[c] + 1)   # never move left backward\n        seen[c] = right\n        best = max(best, right - left + 1)\n    return best"
    }
  ]
}
]
