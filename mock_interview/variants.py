# Variations of each stage's underlying concept, rotated per attempt so
# repeat runs test the same idea from a different angle.
#
# CODING_VARIANTS[stage_id] = list of alternates. Each has label, prompt,
# hints, tests, solution (and optional starter — used for "fix this buggy
# code" variants). Variant 0 is always the base stage in problems.py;
# rotation index = attempts % (1 + len(variants)).
#
# VERBAL_VARIANTS[stage_id] = alternate framings of the same question for the
# resume deep-dives. The "strong answer" bar is shared with the base.

CODING_VARIANTS = {

# ---------------- W1: attention ----------------
"w1s1": [
  {
    "label": "batched",
    "prompt": "Implement single-head scaled dot-product self-attention, **batched**:\n```\ndef attention(X, Wq, Wk, Wv):\n```\n`X` is `(B, T, d_model)`; weights as before. Return `(B, T, d_v)`. No Python loop over the batch — broadcast it.",
    "starter": "import numpy as np\n\ndef attention(X, Wq, Wk, Wv):\n    # X: (B, T, d_model)\n    pass\n",
    "hints": [
      "X @ Wq broadcasts to (B, T, d_k). For scores you need Q @ K^T per batch: Q @ K.transpose(0, 2, 1) -> (B, T, T).",
      "Softmax over the LAST axis with keepdims=True, subtracting the max for stability — then weights @ V is (B, T, d_v)."
    ],
    "tests": """
def _sm(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def _ref1(X, Wq, Wk, Wv):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    return _sm(Q @ K.T / np.sqrt(Wk.shape[1])) @ V

def _t_shape():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2, 5, 8)); Wq = rng.standard_normal((8, 4))
    Wk = rng.standard_normal((8, 4)); Wv = rng.standard_normal((8, 6))
    out = attention(X, Wq, Wk, Wv)
    assert out is not None and out.shape == (2, 5, 6), f"expected (2, 5, 6) = (B, T, d_v), got {None if out is None else out.shape}"
_check("output shape is (B, T, d_v)", _t_shape)

def _t_ref():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((3, 6, 8)); Wq = rng.standard_normal((8, 4))
    Wk = rng.standard_normal((8, 4)); Wv = rng.standard_normal((8, 4))
    got = attention(X, Wq, Wk, Wv)
    for b in range(3):
        assert np.allclose(got[b], _ref1(X[b], Wq, Wk, Wv), atol=1e-5), f"batch {b} doesn't match the per-example reference (transpose the right axes: K.transpose(0,2,1))"
_check("each batch element matches single-example attention", _t_ref)

def _t_stable():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((2, 4, 8)) * 60.0
    Wq = rng.standard_normal((8, 4)); Wk = rng.standard_normal((8, 4)); Wv = rng.standard_normal((8, 4))
    assert np.isfinite(attention(X, Wq, Wk, Wv)).all(), "NaN/inf on large inputs -- stable softmax"
_check("numerically stable on large logits", _t_stable)
""",
    "solution": "def softmax(z, axis=-1):\n    z = z - z.max(axis=axis, keepdims=True)\n    e = np.exp(z)\n    return e / e.sum(axis=axis, keepdims=True)\n\ndef attention(X, Wq, Wk, Wv):\n    Q, K, V = X @ Wq, X @ Wk, X @ Wv           # (B,T,dk) each\n    S = Q @ K.transpose(0, 2, 1) / np.sqrt(Wk.shape[1])   # (B,T,T)\n    return softmax(S, axis=-1) @ V              # (B,T,dv)"
  }
],

"w1s2": [
  {
    "label": "arbitrary mask",
    "prompt": "Generalize the mask. Instead of a `causal` flag, accept **any** attention mask:\n```\ndef attention(X, Wq, Wk, Wv, mask=None):\n```\n`mask` is a `(T, T)` boolean array, `True` = query row may attend to that key column; `None` = full attention. Causal is then just `mask=np.tril(...)` — but yours must handle e.g. padding masks too. Assume every row has at least one allowed key.",
    "starter": None,
    "hints": [
      "Same trick as causal: np.where(mask, scores, -np.inf) BEFORE the softmax.",
      "Don't special-case causal — if your causal version worked, this is the same np.where with the caller's mask."
    ],
    "tests": """
def _sm(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def _ref_m(X, Wq, Wk, Wv, mask=None):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    S = Q @ K.T / np.sqrt(Wk.shape[1])
    if mask is not None:
        S = np.where(mask, S, -np.inf)
    return _sm(S) @ V

def _t_none():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((5, 8)); Wq = rng.standard_normal((8, 4))
    Wk = rng.standard_normal((8, 4)); Wv = rng.standard_normal((8, 4))
    assert np.allclose(attention(X, Wq, Wk, Wv), _ref_m(X, Wq, Wk, Wv), atol=1e-5), "mask=None path wrong"
_check("mask=None gives full attention", _t_none)

def _t_causal_mask():
    rng = np.random.default_rng(4)
    T = 6
    X = rng.standard_normal((T, 8)); Wq = rng.standard_normal((8, 4))
    Wk = rng.standard_normal((8, 4)); Wv = rng.standard_normal((8, 4))
    m = np.tril(np.ones((T, T), dtype=bool))
    assert np.allclose(attention(X, Wq, Wk, Wv, mask=m), _ref_m(X, Wq, Wk, Wv, m), atol=1e-5), "causal-shaped mask wrong"
_check("tril mask reproduces causal attention", _t_causal_mask)

def _t_padding_mask():
    rng = np.random.default_rng(5)
    T = 6
    X = rng.standard_normal((T, 8)); Wq = rng.standard_normal((8, 4))
    Wk = rng.standard_normal((8, 4)); Wv = rng.standard_normal((8, 4))
    m = np.ones((T, T), dtype=bool); m[:, T-2:] = False   # last 2 keys are padding
    got = attention(X, Wq, Wk, Wv, mask=m)
    assert np.allclose(got, _ref_m(X, Wq, Wk, Wv, m), atol=1e-5), "padding-style mask wrong (mask must hit SCORES before softmax)"
    X2 = X.copy(); X2[T-2:] = rng.standard_normal((2, 8)) * 5
    got2 = attention(X2, Wq, Wk, Wv, mask=m)
    assert not np.allclose(got, got2, atol=1e-5) or np.allclose((X2[:T-2]), (X[:T-2])), "sanity"
    # masked-out keys must not influence output rows at all when only keys change
    X3 = X.copy(); X3[T-2:] = 0
    Q_same = np.allclose(attention(X3, Wq, Wk, Wv, mask=m)[:T-2].shape, got[:T-2].shape)
    assert Q_same
_check("padding mask blocks masked keys", _t_padding_mask)
""",
    "solution": "def attention(X, Wq, Wk, Wv, mask=None):\n    Q, K, V = X @ Wq, X @ Wk, X @ Wv\n    S = Q @ K.T / np.sqrt(Wk.shape[1])\n    if mask is not None:\n        S = np.where(mask, S, -np.inf)   # before softmax; rows renormalize\n    return softmax(S, axis=-1) @ V"
  }
],

"w1s3": [
  {
    "label": "cross-attention",
    "prompt": "Multi-head **cross**-attention — the VLM-fusion workhorse:\n```\ndef mha_cross(Xq, Xkv, Wq, Wk, Wv, Wo, n_heads):\n```\nQueries come from `Xq` `(T_q, d)`, keys/values from `Xkv` `(T_kv, d)` — think text tokens attending to image tokens. Same head-split convention as before (reshape to `(T, H, d_h)`, transpose), scale by `√d_h`, concat, project with `Wo`. No loop over heads.",
    "starter": None,
    "hints": [
      "Only difference from self-attention: Q projects from Xq, K and V project from Xkv. Scores are (H, T_q, T_kv) — rectangular is fine.",
      "Split each with its OWN sequence length: Q reshape (T_q, H, dh); K, V reshape (T_kv, H, dh). No mask — cross-attention is usually full."
    ],
    "tests": """
def _sm(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def _ref_cross(Xq, Xkv, Wq, Wk, Wv, Wo, H):
    Tq, d = Xq.shape; Tkv = Xkv.shape[0]; dh = d // H
    Q = (Xq @ Wq).reshape(Tq, H, dh).transpose(1, 0, 2)
    K = (Xkv @ Wk).reshape(Tkv, H, dh).transpose(1, 0, 2)
    V = (Xkv @ Wv).reshape(Tkv, H, dh).transpose(1, 0, 2)
    A = _sm(Q @ K.transpose(0, 2, 1) / np.sqrt(dh))
    return (A @ V).transpose(1, 0, 2).reshape(Tq, d) @ Wo

def _t_cross():
    rng = np.random.default_rng(6)
    Tq, Tkv, d, H = 4, 7, 8, 2
    Xq = rng.standard_normal((Tq, d)); Xkv = rng.standard_normal((Tkv, d))
    Wq, Wk, Wv, Wo = (rng.standard_normal((d, d)) for _ in range(4))
    got = mha_cross(Xq, Xkv, Wq, Wk, Wv, Wo, H)
    assert got.shape == (Tq, d), f"expected {(Tq, d)}, got {got.shape} -- output length follows the QUERIES"
    assert np.allclose(got, _ref_cross(Xq, Xkv, Wq, Wk, Wv, Wo, H), atol=1e-5), "doesn't match reference (K,V come from Xkv; scores are (H, T_q, T_kv))"
_check("cross-attention matches reference", _t_cross)

def _t_kv_influence():
    rng = np.random.default_rng(7)
    Xq = rng.standard_normal((3, 8)); Xkv = rng.standard_normal((5, 8))
    Wq, Wk, Wv, Wo = (rng.standard_normal((8, 8)) for _ in range(4))
    a = mha_cross(Xq, Xkv, Wq, Wk, Wv, Wo, 2)
    b = mha_cross(Xq, rng.standard_normal((5, 8)), Wq, Wk, Wv, Wo, 2)
    assert not np.allclose(a, b, atol=1e-5), "changing Xkv didn't change the output -- are K,V really projected from Xkv?"
_check("K,V actually come from the second sequence", _t_kv_influence)
""",
    "solution": "def mha_cross(Xq, Xkv, Wq, Wk, Wv, Wo, n_heads):\n    Tq, d = Xq.shape; Tkv = Xkv.shape[0]\n    dh = d // n_heads\n    Q = (Xq @ Wq).reshape(Tq, n_heads, dh).transpose(1, 0, 2)\n    K = (Xkv @ Wk).reshape(Tkv, n_heads, dh).transpose(1, 0, 2)\n    V = (Xkv @ Wv).reshape(Tkv, n_heads, dh).transpose(1, 0, 2)\n    S = Q @ K.transpose(0, 2, 1) / np.sqrt(dh)   # (H, Tq, Tkv)\n    out = (softmax(S, axis=-1) @ V).transpose(1, 0, 2).reshape(Tq, d)\n    return out @ Wo"
  }
],

"w1s4": [
  {
    "label": "sliding window",
    "prompt": "Production twist: a **sliding-window** KV cache (Mistral-style). Single head:\n```\ndef decode_step(x_t, Wq, Wk, Wv, cache, window):\n```\nSame cache dict as before, but keep **at most the last `window`** K/V rows (including this step's). The output attends over exactly what's in the cache. Feeding tokens one at a time must equal full attention restricted to the last `window` tokens at each step.",
    "starter": None,
    "hints": [
      "Append this step's k, v first, THEN truncate: K = K[-window:], V = V[-window:].",
      "No mask needed — the cache IS the receptive field. q attends over all cached rows."
    ],
    "tests": """
def _sm(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def _ref_win(X_hist, Wq, Wk, Wv, window):
    # attention output for the LAST token of X_hist over the last `window` tokens
    Xw = X_hist[-window:]
    q = (X_hist[-1] @ Wq)[None, :]
    K = Xw @ Wk; V = Xw @ Wv
    return (_sm(q @ K.T / np.sqrt(K.shape[1])) @ V)[0]

def _t_window():
    rng = np.random.default_rng(8)
    T, d, dk, W = 8, 8, 4, 3
    X = rng.standard_normal((T, d))
    Wq = rng.standard_normal((d, dk)); Wk = rng.standard_normal((d, dk)); Wv = rng.standard_normal((d, dk))
    cache = {"K": None, "V": None}
    for t in range(T):
        out, cache = decode_step(X[t], Wq, Wk, Wv, cache, W)
        exp = _ref_win(X[:t + 1], Wq, Wk, Wv, W)
        assert np.allclose(np.asarray(out).reshape(-1), exp, atol=1e-5), f"step {t}: output should attend over the last {W} tokens only"
_check("incremental sliding-window decode is exact", _t_window)

def _t_capped():
    rng = np.random.default_rng(9)
    d, dk, W = 8, 4, 3
    Wq = rng.standard_normal((d, dk)); Wk = rng.standard_normal((d, dk)); Wv = rng.standard_normal((d, dk))
    cache = {"K": None, "V": None}
    for t in range(7):
        _, cache = decode_step(rng.standard_normal(d), Wq, Wk, Wv, cache, W)
    assert cache["K"].shape == (W, dk), f"cache must be capped at window={W} rows, got {cache['K'].shape} -- memory is the whole point"
_check("cache never exceeds the window", _t_capped)
""",
    "solution": "def decode_step(x_t, Wq, Wk, Wv, cache, window):\n    q = (x_t @ Wq)[None, :]\n    k = (x_t @ Wk)[None, :]\n    v = (x_t @ Wv)[None, :]\n    K = k if cache[\"K\"] is None else np.vstack([cache[\"K\"], k])\n    V = v if cache[\"V\"] is None else np.vstack([cache[\"V\"], v])\n    K, V = K[-window:], V[-window:]          # cap AFTER appending\n    A = softmax(q @ K.T / np.sqrt(K.shape[1]), axis=-1)\n    return (A @ V)[0], {\"K\": K, \"V\": V}"
  }
],

# ---------------- W2: norms & backprop ----------------
"w2s1": [
  {
    "label": "debug",
    "prompt": "Different format: a teammate's RMSNorm keeps failing CI. **Fix it in place and tell me each bug out loud** — don't rewrite from scratch:\n```\ndef rmsnorm(x, g, eps=1e-6):\n    mu = x.mean(axis=-1, keepdims=True)\n    rms = np.sqrt(((x - mu) ** 2).mean(axis=-1) + eps)\n    return x / rms * g\n```",
    "starter": "import numpy as np\n\ndef rmsnorm(x, g, eps=1e-6):\n    mu = x.mean(axis=-1, keepdims=True)\n    rms = np.sqrt(((x - mu) ** 2).mean(axis=-1) + eps)\n    return x / rms * g\n",
    "hints": [
      "Bug 1 is conceptual: RMSNorm does NOT subtract the mean — that's LayerNorm. This code computes std, not RMS.",
      "Bug 2 is a shape bug: the inner .mean(axis=-1) is missing keepdims=True, so the division broadcasts wrong for (B, T, d) inputs."
    ],
    "tests": """
def _ref_rms(x, g, eps=1e-6):
    return x / np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps) * g

def _t_1d():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(8) + 3.0; g = rng.standard_normal(8)
    assert np.allclose(rmsnorm(x, g), _ref_rms(x, g), atol=1e-6), "still normalizing (x - mean) -- RMSNorm has no mean subtraction"
_check("no mean subtraction (bug 1 fixed)", _t_1d)

def _t_3d():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 3, 8)) + 5.0
    g = rng.standard_normal(8)
    got = rmsnorm(x, g)
    assert got.shape == x.shape, f"broadcasting broke on (B,T,d): {got.shape} -- keepdims (bug 2)"
    assert np.allclose(got, _ref_rms(x, g), atol=1e-6), "values wrong on (B,T,d)"
_check("keepdims fixed (bug 2)", _t_3d)
""",
    "solution": "def rmsnorm(x, g, eps=1e-6):\n    # bug 1: no mean subtraction -- RMS, not std\n    # bug 2: keepdims=True so the division broadcasts over the last axis\n    rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)\n    return x / rms * g"
  }
],

"w2s2": [
  {
    "label": "debug",
    "prompt": "Same drill, LayerNorm this time. Two bugs — find both, fix in place, name them out loud:\n```\ndef layernorm(x, g, b, eps=1e-5):\n    mu = x.mean(axis=-1, keepdims=True)\n    var = x.var(axis=-1, keepdims=True, ddof=1)\n    return (x - mu) / (np.sqrt(var) + eps) * g + b\n```",
    "starter": "import numpy as np\n\ndef layernorm(x, g, b, eps=1e-5):\n    mu = x.mean(axis=-1, keepdims=True)\n    var = x.var(axis=-1, keepdims=True, ddof=1)\n    return (x - mu) / (np.sqrt(var) + eps) * g + b\n",
    "hints": [
      "Bug 1: ddof=1 is the UNBIASED (sample) variance — norm layers use the biased/population variance (NumPy's default, ddof=0).",
      "Bug 2: eps must go INSIDE the sqrt, added to the variance: sqrt(var + eps). Outside, tiny-variance rows blow up differently and nothing matches reference implementations."
    ],
    "tests": """
def _ref_ln(x, g, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b

def _t_ln():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 6)) * 3 + 2
    g = rng.standard_normal(6); b = rng.standard_normal(6)
    assert np.allclose(layernorm(x, g, b), _ref_ln(x, g, b), atol=1e-6), "still off -- check ddof (population variance) and eps placement (inside sqrt)"
_check("matches reference (both bugs fixed)", _t_ln)

def _t_tiny_var():
    x = np.ones((2, 4)) * 7.0
    x[0, 0] += 1e-8
    g = np.ones(4); b = np.zeros(4)
    out = layernorm(x, g, b)
    assert np.isfinite(out).all() and np.abs(out).max() < 10, "near-constant rows misbehave -- eps inside the sqrt bounds the output"
_check("stable on near-constant rows", _t_tiny_var)
""",
    "solution": "def layernorm(x, g, b, eps=1e-5):\n    mu = x.mean(axis=-1, keepdims=True)\n    var = x.var(axis=-1, keepdims=True)          # bug 1: ddof=0 (population)\n    return (x - mu) / np.sqrt(var + eps) * g + b # bug 2: eps inside sqrt"
  }
],

"w2s3": [
  {
    "label": "binary logistic",
    "prompt": "Same concept, binary flavor — **logistic loss with logits**, stable, plus its gradient:\n```\ndef logistic_xent(logits, y):\n    # logits: (B,) float, y: (B,) in {0, 1}\n    # returns (loss, dlogits)  -- loss is the mean over the batch\n```\nloss per example = `-[y·log σ(z) + (1−y)·log(1−σ(z))]`. Must be exact for `z = ±1000`. What's the closed form of the gradient?",
    "starter": None,
    "hints": [
      "Algebra first: the loss simplifies to softplus(z) - y·z = logaddexp(0, z) - y*z. That form never overflows.",
      "The gradient collapses just like softmax-CE: dlogits = (sigmoid(z) - y) / B."
    ],
    "tests": """
def _ref_bl(z, y):
    return float(np.mean(np.logaddexp(0.0, z) - y * z))

def _t_val():
    rng = np.random.default_rng(3)
    z = rng.standard_normal(6) * 2; y = rng.integers(0, 2, 6).astype(float)
    loss, _ = logistic_xent(z, y)
    assert np.isclose(loss, _ref_bl(z, y), atol=1e-8), "loss value wrong (use logaddexp(0, z) - y*z)"
_check("loss matches reference", _t_val)

def _t_stable():
    z = np.array([1000.0, -1000.0]); y = np.array([1.0, 0.0])
    loss, dl = logistic_xent(z, y)
    assert np.isfinite(loss) and loss < 1e-6, "confident-and-correct should give ~0 loss, finite"
    assert np.isfinite(dl).all(), "gradient overflows at z=±1000"
_check("exact at z = ±1000", _t_stable)

def _t_grad():
    rng = np.random.default_rng(4)
    z = rng.standard_normal(5); y = rng.integers(0, 2, 5).astype(float)
    _, dl = logistic_xent(z, y)
    h = 1e-5
    num = np.zeros_like(z)
    for i in range(len(z)):
        zp = z.copy(); zp[i] += h
        zm = z.copy(); zm[i] -= h
        num[i] = (_ref_bl(zp, y) - _ref_bl(zm, y)) / (2 * h)
    assert np.allclose(dl, num, atol=1e-4), "gradient doesn't match finite differences ((sigmoid(z) - y)/B)"
_check("gradient matches finite differences", _t_grad)
""",
    "solution": "def logistic_xent(logits, y):\n    B = len(logits)\n    loss = np.mean(np.logaddexp(0.0, logits) - y * logits)   # softplus(z) - y z\n    sig = np.where(logits >= 0,\n                   1.0 / (1.0 + np.exp(-logits)),\n                   np.exp(logits) / (1.0 + np.exp(logits)))  # stable sigmoid\n    dlogits = (sig - y) / B\n    return loss, dlogits"
  }
],

"w2s4": [
  {
    "label": "with L2",
    "prompt": "Same MLP backward, but the loss now has **L2 regularization** (weights only, not biases):\n```\nloss = softmax-CE mean + lam * (sum(W1²) + sum(W2²))\n```\nImplement:\n```\ndef mlp_grads_l2(X, y, W1, b1, W2, b2, lam):\n    # returns {\"dW1\": ..., \"db1\": ..., \"dW2\": ..., \"db2\": ...}\n```\nFinite differences will check every entry, so the reg term's gradient has to be right too.",
    "starter": None,
    "hints": [
      "The backward is the plain MLP backward plus one extra term on each weight: d(lam·sum(W²))/dW = 2·lam·W. Biases get nothing.",
      "Common slips: forgetting the 2, or accidentally regularizing b1/b2."
    ],
    "tests": """
def _fwd_l2(X, y, W1, b1, W2, b2, lam):
    h = np.maximum(X @ W1 + b1, 0)
    logits = h @ W2 + b2
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    ce = -logp[np.arange(len(y)), y].mean()
    return ce + lam * (np.sum(W1 ** 2) + np.sum(W2 ** 2))

def _t_l2():
    rng = np.random.default_rng(5)
    B, D, H, C = 4, 3, 5, 4
    lam = 0.05
    X = rng.standard_normal((B, D)); y = rng.integers(0, C, B)
    W1 = rng.standard_normal((D, H)) * 0.5; b1 = rng.standard_normal(H) * 0.1
    W2 = rng.standard_normal((H, C)) * 0.5; b2 = rng.standard_normal(C) * 0.1
    grads = mlp_grads_l2(X, y, W1, b1, W2, b2, lam)
    params = {"dW1": W1, "db1": b1, "dW2": W2, "db2": b2}
    h = 1e-5
    for name, P in params.items():
        G = np.asarray(grads[name])
        num = np.zeros_like(P)
        it = np.nditer(P, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            old = P[idx]
            P[idx] = old + h; lp = _fwd_l2(X, y, W1, b1, W2, b2, lam)
            P[idx] = old - h; lm = _fwd_l2(X, y, W1, b1, W2, b2, lam)
            P[idx] = old
            num[idx] = (lp - lm) / (2 * h)
            it.iternext()
        assert np.allclose(G, num, atol=1e-4), f"{name} doesn't match finite differences (weights get +2*lam*W; biases get NO reg term)"
_check("all gradients incl. L2 term match finite differences", _t_l2)
""",
    "solution": "def mlp_grads_l2(X, y, W1, b1, W2, b2, lam):\n    B = X.shape[0]\n    h_pre = X @ W1 + b1\n    h = np.maximum(h_pre, 0)\n    logits = h @ W2 + b2\n    z = logits - logits.max(axis=1, keepdims=True)\n    p = np.exp(z); p /= p.sum(axis=1, keepdims=True)\n    d = p.copy(); d[np.arange(B), y] -= 1.0; d /= B\n    dW2 = h.T @ d + 2 * lam * W2      # reg on weights only\n    db2 = d.sum(axis=0)\n    dh = d @ W2.T\n    dh_pre = dh * (h_pre > 0)\n    dW1 = X.T @ dh_pre + 2 * lam * W1\n    db1 = dh_pre.sum(axis=0)\n    return {\"dW1\": dW1, \"db1\": db1, \"dW2\": dW2, \"db2\": db2}"
  }
],

# ---------------- F1: sampling ----------------
"f1s1": [
  {
    "label": "log-space",
    "prompt": "Same concept in log space — return **log-probabilities** (what you'd actually keep for PPO ratios):\n```\ndef sample_logprobs(logits, temperature=1.0):\n```\nRules: `temperature=0` → greedy one-hot in log space (`0.0` for the argmax, `-inf` elsewhere); otherwise the stable log-softmax of `logits / temperature`. `exp` of your output must sum to 1.",
    "starter": None,
    "hints": [
      "Log-softmax, never log(softmax(x)): z = logits/T; z -= z.max(); logp = z - log(sum(exp(z))).",
      "For T=0: np.full_like(..., -np.inf) then set the argmax entry to 0.0 (log 1)."
    ],
    "tests": """
def _ref_lsm(z):
    z = z - z.max()
    return z - np.log(np.exp(z).sum())

def _t_t1():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal(10) * 2
    got = sample_logprobs(logits, 1.0)
    assert np.allclose(got, _ref_lsm(logits), atol=1e-6), "T=1 should be plain log-softmax"
    assert np.isclose(np.exp(got).sum(), 1.0, atol=1e-6), "exp(logp) must sum to 1"
_check("T=1 is log-softmax, normalized", _t_t1)

def _t_stable():
    logits = np.array([1000.0, 999.0, 0.0])
    got = sample_logprobs(logits, 1.0)
    assert np.isfinite(got[:2]).all(), "log-probs of the top tokens must be finite at logits ~1000"
    assert got[0] > got[1] > got[2], "ordering must be preserved"
_check("stable at logits ~1000 (log-space trick)", _t_stable)

def _t_t0():
    logits = np.array([0.1, 2.5, -1.0])
    got = sample_logprobs(logits, 0.0)
    assert np.isclose(got[1], 0.0, atol=1e-9), "argmax gets log(1) = 0"
    assert np.isneginf(got[0]) and np.isneginf(got[2]), "everything else gets -inf"
_check("T=0 is greedy one-hot in log space", _t_t0)
""",
    "solution": "def sample_logprobs(logits, temperature=1.0):\n    if temperature == 0:\n        out = np.full(len(logits), -np.inf)\n        out[np.argmax(logits)] = 0.0\n        return out\n    z = logits / temperature\n    z = z - z.max()\n    return z - np.log(np.exp(z).sum())   # log-softmax, one pass, stable"
  }
],

"f1s2": [
  {
    "label": "min-p",
    "prompt": "The newer truncation rule — **min-p sampling**:\n```\ndef sample_probs_minp(logits, temperature=1.0, min_p=None):\n```\nAfter the temperature softmax, keep only tokens with `p_i >= min_p * max(p)`, zero the rest, renormalize. The threshold adapts to the model's confidence — that's the whole point. `min_p=None` = no filtering; the argmax always survives.",
    "starter": None,
    "hints": [
      "threshold = min_p * p.max(); keep = p >= threshold; p = p*keep; p /= p.sum().",
      "No sorting needed — that's what makes min-p cheaper than top-p. The comparison is against the MAX, not a cumulative sum."
    ],
    "tests": """
def _sm(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def _ref_minp(logits, temperature=1.0, min_p=None):
    p = _sm(logits / temperature)
    if min_p is not None:
        keep = p >= min_p * p.max()
        p = np.where(keep, p, 0.0)
        p = p / p.sum()
    return p

_L = np.array([2.0, 1.0, 0.0, -2.0])

def _t_plain():
    assert np.allclose(sample_probs_minp(_L, 1.0), _sm(_L), atol=1e-6), "min_p=None should be plain temperature softmax"
_check("no filter when min_p is None", _t_plain)

def _t_minp():
    got = sample_probs_minp(_L, 1.0, min_p=0.3)
    exp = _ref_minp(_L, 1.0, 0.3)
    assert np.isclose(got.sum(), 1.0, atol=1e-6), "renormalize after filtering"
    assert np.allclose(got, exp, atol=1e-6), "min-p rule wrong: threshold is min_p * max(p), keep p >= threshold"
_check("min-p threshold correct", _t_minp)

def _t_argmax_survives():
    got = sample_probs_minp(_L, 1.0, min_p=1.0)
    assert np.isclose(got[0], 1.0, atol=1e-6), "min_p=1.0 keeps exactly the argmax"
_check("argmax always survives", _t_argmax_survives)

def _t_with_temp():
    got = sample_probs_minp(_L, 0.5, min_p=0.2)
    exp = _ref_minp(_L, 0.5, 0.2)
    assert np.allclose(got, exp, atol=1e-6), "temperature must be applied BEFORE the min-p filter"
_check("composes with temperature", _t_with_temp)
""",
    "solution": "def sample_probs_minp(logits, temperature=1.0, min_p=None):\n    z = logits / temperature\n    z = z - z.max()\n    p = np.exp(z); p = p / p.sum()\n    if min_p is not None:\n        keep = p >= min_p * p.max()   # confidence-adaptive threshold\n        p = np.where(keep, p, 0.0)\n        p = p / p.sum()\n    return p"
  }
],

"f1s3": [
  {
    "label": "freq/presence",
    "prompt": "The other anti-repetition scheme (OpenAI-style) — **frequency and presence penalties**:\n```\ndef apply_freq_presence(logits, generated_ids, presence=0.0, frequency=0.0):\n```\nFor each token id: subtract `presence` once if it has appeared at all, plus `frequency * count` for how many times it appeared. Additive on logits (unlike the multiplicative CTRL penalty — no positive/negative asymmetry to worry about). Don't mutate the input.",
    "starter": None,
    "hints": [
      "Count occurrences (collections.Counter or a dict). new[t] = logits[t] - presence*(count>0) - frequency*count.",
      "Copy first. Tokens never generated are untouched."
    ],
    "tests": """
def _t_basic():
    logits = np.array([1.0, 2.0, 3.0])
    out = apply_freq_presence(logits, [0, 0, 2], presence=0.5, frequency=0.25)
    exp = np.array([1.0 - 0.5 - 0.5, 2.0, 3.0 - 0.5 - 0.25])
    assert np.allclose(out, exp, atol=1e-8), f"expected {exp}, got {out} (presence once per SEEN token, frequency times COUNT)"
_check("presence + frequency applied correctly", _t_basic)

def _t_zero():
    logits = np.array([1.0, -2.0])
    out = apply_freq_presence(logits, [0, 1, 0], presence=0.0, frequency=0.0)
    assert np.allclose(out, logits), "zero penalties should be a no-op"
_check("zero penalties are a no-op", _t_zero)

def _t_no_mutation():
    logits = np.array([1.0, 2.0])
    orig = logits.copy()
    apply_freq_presence(logits, [0], presence=1.0, frequency=1.0)
    assert np.array_equal(logits, orig), "input mutated -- return a copy"
_check("input not mutated", _t_no_mutation)
""",
    "solution": "def apply_freq_presence(logits, generated_ids, presence=0.0, frequency=0.0):\n    out = np.asarray(logits, dtype=float).copy()\n    counts = {}\n    for t in generated_ids:\n        counts[t] = counts.get(t, 0) + 1\n    for t, c in counts.items():\n        out[t] -= presence + frequency * c   # additive: no sign asymmetry\n    return out"
  }
],

"f1s4": [
  {
    "label": "self-consistency",
    "prompt": "The other inference-time selection scheme — **self-consistency (majority voting)**:\n```\ndef majority_vote(answers):\n```\n`answers` is a list of final answers (hashable — ints, strings) from n samples. Return the most frequent one; ties go to the answer that **first reached** that count... simpler rule, same spirit: ties go to the one seen **earliest** in the list. No reward model needed — agreement is the signal.",
    "starter": None,
    "hints": [
      "Count with a dict; track first-seen order. max over (count, -first_seen_index).",
      "Python's sort is stable: iterate in first-seen order and only replace the champion on a STRICTLY greater count."
    ],
    "tests": """
def _t_basic():
    assert majority_vote([3, 5, 3, 2, 3]) == 3
_check("plain majority", _t_basic)

def _t_tie():
    assert majority_vote([1, 2, 1, 2]) == 1, "on ties, earliest-seen answer wins"
    assert majority_vote([7, 4, 4, 7]) == 7, "earliest-seen, not largest or latest"
_check("ties go to earliest-seen", _t_tie)

def _t_strings():
    assert majority_vote(["42", "41", "42"]) == "42"
_check("works on string answers", _t_strings)

def _t_single():
    assert majority_vote([9]) == 9
_check("single sample", _t_single)
""",
    "solution": "def majority_vote(answers):\n    counts = {}\n    for a in answers:                 # first-seen order preserved by dict\n        counts[a] = counts.get(a, 0) + 1\n    best, best_c = None, 0\n    for a, c in counts.items():\n        if c > best_c:                # strict >: earliest-seen wins ties\n            best, best_c = a, c\n    return best"
  }
],

# ---------------- F2: post-training losses ----------------
"f2s1": [
  {
    "label": "with metrics",
    "prompt": "Same Bradley–Terry objective, but return what you'd actually log during RM training:\n```\ndef rm_stats(r_chosen, r_rejected):\n    # returns (loss, accuracy, mean_margin)\n```\n`loss` = mean `-log σ(r_c - r_r)` (stable at ±1000, as always); `accuracy` = fraction with `r_c > r_r` strictly; `mean_margin` = mean of `r_c - r_r`.",
    "starter": None,
    "hints": [
      "Loss is the same logaddexp(0, -(r_c - r_r)) as before. accuracy = (margin > 0).mean().",
      "Ties (r_c == r_r) count as WRONG for accuracy — strict >."
    ],
    "tests": """
def _t_stats():
    r_c = np.array([1.0, 2.0, 0.0, 3.0])
    r_r = np.array([0.0, 2.0, 1.0, 1.0])
    loss, acc, margin = rm_stats(r_c, r_r)
    exp_loss = np.mean(np.logaddexp(0.0, -(r_c - r_r)))
    assert np.isclose(loss, exp_loss, atol=1e-8), "loss wrong"
    assert np.isclose(acc, 0.5), f"accuracy should be 0.5 (tie counts as wrong), got {acc}"
    assert np.isclose(margin, np.mean(r_c - r_r), atol=1e-8), "margin wrong"
_check("loss, accuracy (strict), margin", _t_stats)

def _t_stable():
    loss, acc, margin = rm_stats(np.array([0.0]), np.array([1000.0]))
    assert np.isfinite(loss) and np.isclose(loss, 1000.0, rtol=1e-6), "stability: use logaddexp"
    assert acc == 0.0 and np.isclose(margin, -1000.0)
_check("stable at margin -1000", _t_stable)
""",
    "solution": "def rm_stats(r_chosen, r_rejected):\n    margin = r_chosen - r_rejected\n    loss = np.mean(np.logaddexp(0.0, -margin))   # -log sigmoid, stable\n    acc = float(np.mean(margin > 0))              # strict: ties are wrong\n    return loss, acc, float(margin.mean())"
  }
],

"f2s2": [
  {
    "label": "IPO",
    "prompt": "DPO's cousin that doesn't saturate — **IPO**:\n```\ndef ipo_loss(logp_c, logp_r, ref_c, ref_r, beta=0.1):\n```\nWith `h = (logp_c − ref_c) − (logp_r − ref_r)` (note: **no** β inside), the IPO loss is the mean of `(h − 1/(2β))²`. It regresses the log-ratio margin to a fixed target instead of pushing it to infinity — that's the fix for DPO's overfitting on easy pairs.",
    "starter": None,
    "hints": [
      "h = (pc - rc) - (pr - rr); loss = np.mean((h - 1.0/(2*beta))**2). That's the whole thing.",
      "The trap is putting beta inside h like DPO does — in IPO beta only sets the TARGET margin 1/(2β)."
    ],
    "tests": """
def _t_hand():
    # h = 1, beta = 0.1 -> target 5 -> (1-5)^2 = 16
    l = ipo_loss(np.array([-4.0]), np.array([-5.0]), np.array([-5.0]), np.array([-5.0]), beta=0.1)
    assert np.isclose(l, 16.0, atol=1e-8), f"hand-check failed: expected 16.0, got {l}"
_check("hand-checkable case (h=1, target=5)", _t_hand)

def _t_ref():
    rng = np.random.default_rng(1)
    pc, pr, rc, rr = (rng.standard_normal(5) * 3 for _ in range(4))
    h = (pc - rc) - (pr - rr)
    exp = np.mean((h - 1.0 / (2 * 0.3)) ** 2)
    assert np.isclose(ipo_loss(pc, pr, rc, rr, beta=0.3), exp, atol=1e-8), "beta belongs in the TARGET 1/(2*beta), not inside h"
_check("matches reference on random inputs", _t_ref)

def _t_perfect():
    # margin exactly at target -> zero loss
    beta = 0.25; target = 1.0 / (2 * beta)
    l = ipo_loss(np.array([target]), np.array([0.0]), np.array([0.0]), np.array([0.0]), beta=beta)
    assert np.isclose(l, 0.0, atol=1e-10), "margin exactly at 1/(2*beta) should give zero loss -- IPO regresses to a target, it doesn't maximize"
_check("zero loss at the target margin", _t_perfect)
""",
    "solution": "def ipo_loss(logp_c, logp_r, ref_c, ref_r, beta=0.1):\n    h = (logp_c - ref_c) - (logp_r - ref_r)      # no beta here\n    target = 1.0 / (2 * beta)                     # beta sets the target margin\n    return np.mean((h - target) ** 2)             # squared: no saturation, no runaway"
  }
],

"f2s3": [
  {
    "label": "KL shaping",
    "prompt": "The other place the ratio shows up — **KL-shaped rewards** for RLHF:\n```\ndef shape_rewards(rewards, logp, ref_logp, beta):\n    # returns (shaped_rewards, mean_kl)\n```\nPer-token: `shaped = reward − β·(logp − ref_logp)`. Also return the mean of `(logp − ref_logp)` — the standard (possibly negative per-token) KL estimate you'd log. Don't mutate inputs.",
    "starter": None,
    "hints": [
      "shaped = rewards - beta * (logp - ref_logp); kl = (logp - ref_logp).mean(). Two lines.",
      "Sign check: if the policy assigns HIGHER logprob than the reference (drifting), the penalty must REDUCE the reward."
    ],
    "tests": """
def _t_shape():
    r = np.array([1.0, 0.0, 2.0])
    lp = np.array([-1.0, -2.0, -0.5])
    ref = np.array([-1.5, -2.0, -1.5])
    shaped, kl = shape_rewards(r, lp, ref, beta=0.1)
    exp = r - 0.1 * (lp - ref)
    assert np.allclose(shaped, exp, atol=1e-8), "shaped = r - beta*(logp - ref_logp)"
    assert np.isclose(kl, (lp - ref).mean(), atol=1e-8), "mean KL estimate wrong"
_check("shaping and KL estimate correct", _t_shape)

def _t_sign():
    shaped, _ = shape_rewards(np.array([1.0]), np.array([-0.5]), np.array([-2.0]), beta=0.5)
    assert shaped[0] < 1.0, "policy above reference (drift) must be PENALIZED -- check your sign"
_check("drift is penalized, not rewarded", _t_sign)

def _t_no_mutation():
    r = np.array([1.0, 2.0]); orig = r.copy()
    shape_rewards(r, np.array([-1.0, -1.0]), np.array([-1.0, -1.0]), beta=1.0)
    assert np.array_equal(r, orig), "input rewards mutated"
_check("inputs not mutated", _t_no_mutation)
""",
    "solution": "def shape_rewards(rewards, logp, ref_logp, beta):\n    drift = logp - ref_logp                 # >0 where policy left the reference\n    shaped = rewards - beta * drift          # new array; inputs untouched\n    return shaped, float(drift.mean())"
  }
],

"f2s4": [
  {
    "label": "RLOO",
    "prompt": "GRPO's cleaner sibling — **leave-one-out (RLOO) advantages**:\n```\ndef rloo_advantages(rewards):\n```\n`rewards` is `(P, N)` with `N ≥ 2`. Each sample's baseline is the mean of the **other** `N−1` samples in its group: `adv[i,j] = r[i,j] − mean(r[i, k≠j])`. No std division. Vectorize it — no loop over samples.",
    "starter": None,
    "hints": [
      "mean of others = (row_sum - r_ij) / (N - 1). So adv = r - (row_sum - r) / (N - 1), with row_sum = rewards.sum(axis=1, keepdims=True).",
      "Sanity: an all-equal group must give exactly zeros (each sample equals the others' mean)."
    ],
    "tests": """
def _t_hand():
    R = np.array([[1.0, 2.0, 3.0]])
    got = rloo_advantages(R)
    exp = np.array([[1.0 - 2.5, 2.0 - 2.0, 3.0 - 1.5]])
    assert np.allclose(got, exp, atol=1e-8), f"expected {exp}, got {got} (baseline = mean of the OTHER samples)"
_check("hand-checkable 1x3 case", _t_hand)

def _t_shape_vec():
    rng = np.random.default_rng(2)
    R = rng.standard_normal((3, 5))
    got = rloo_advantages(R)
    assert got.shape == R.shape
    for i in range(3):
        for j in range(5):
            exp = R[i, j] - (R[i].sum() - R[i, j]) / 4
            assert np.isclose(got[i, j], exp, atol=1e-8), "leave-one-out baseline wrong"
_check("matches element-wise definition", _t_shape_vec)

def _t_degenerate():
    R = np.array([[0.7, 0.7, 0.7, 0.7]])
    got = rloo_advantages(R)
    assert np.allclose(got, 0.0, atol=1e-10), "all-equal group must give exactly zero advantages"
_check("all-equal group -> zeros", _t_degenerate)
""",
    "solution": "def rloo_advantages(rewards):\n    N = rewards.shape[1]\n    row_sum = rewards.sum(axis=1, keepdims=True)\n    baseline = (row_sum - rewards) / (N - 1)   # mean of the others, unbiased\n    return rewards - baseline"
  }
],

# ---------------- A1: algorithms ----------------
"a1s1": [
  {
    "label": "insert interval",
    "prompt": "Variation on the theme: the intervals are **already sorted and disjoint** — insert one new interval and merge:\n```\ndef insert_interval(intervals, new):\n```\n`insert_interval([[1,3],[6,9]], [2,5])` → `[[1,5],[6,9]]`. Touching merges. Aim for one pass, O(n).",
    "starter": None,
    "hints": [
      "Three phases: copy intervals ending before new starts; absorb every overlapping interval into new (min start, max end); copy the rest.",
      "'Overlaps' includes touching: iv[0] <= new[1] and iv[1] >= new[0]."
    ],
    "tests": """
def _norm(out):
    return [list(x) for x in out]

def _t_basic():
    assert _norm(insert_interval([[1, 3], [6, 9]], [2, 5])) == [[1, 5], [6, 9]]
_check("classic insert case", _t_basic)

def _t_multi():
    assert _norm(insert_interval([[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]], [4, 8])) == [[1, 2], [3, 10], [12, 16]], "must absorb every overlapped interval"
_check("absorbs multiple intervals", _t_multi)

def _t_edges():
    assert _norm(insert_interval([], [5, 7])) == [[5, 7]]
    assert _norm(insert_interval([[1, 5]], [6, 8])) == [[1, 5], [6, 8]]
    assert _norm(insert_interval([[1, 2]], [2, 3])) == [[1, 3]], "touching merges"
_check("empty list / disjoint / touching", _t_edges)
""",
    "solution": "def insert_interval(intervals, new):\n    out = []\n    i, n = 0, len(intervals)\n    s, e = new\n    while i < n and intervals[i][1] < s:      # strictly before\n        out.append(list(intervals[i])); i += 1\n    while i < n and intervals[i][0] <= e:     # overlap or touch\n        s = min(s, intervals[i][0])\n        e = max(e, intervals[i][1])\n        i += 1\n    out.append([s, e])\n    while i < n:\n        out.append(list(intervals[i])); i += 1\n    return out"
  }
],

"a1s2": [
  {
    "label": "k distinct",
    "prompt": "Same sliding-window muscle, different constraint: longest substring with **at most k distinct characters**:\n```\ndef longest_k_distinct(s, k):\n```\n`(\"eceba\", 2)` → 3 (`\"ece\"`), `(\"aa\", 1)` → 2, `k=0` → 0. One pass.",
    "starter": None,
    "hints": [
      "Window with a char->count dict. Grow right; while len(counts) > k, shrink from the left (decrement, delete at zero).",
      "The count-delete on zero is what keeps len(counts) meaningful — forgetting it inflates the distinct count forever."
    ],
    "tests": """
def _t_cases():
    cases = [("eceba", 2, 3), ("aa", 1, 2), ("a", 0, 0), ("araaci", 2, 4), ("abcabcbb", 3, 8), ("", 2, 0)]
    for s, k, exp in cases:
        got = longest_k_distinct(s, k)
        assert got == exp, f"longest_k_distinct({s!r}, {k}) = {got}, expected {exp}"
_check("standard cases incl. k=0 and empty", _t_cases)

def _t_shrink():
    assert longest_k_distinct("abaccc", 2) == 4, "window must shrink correctly when a third char enters ('accc')"
_check("window shrink logic", _t_shrink)
""",
    "solution": "def longest_k_distinct(s, k):\n    if k == 0:\n        return 0\n    counts = {}\n    left = best = 0\n    for right, c in enumerate(s):\n        counts[c] = counts.get(c, 0) + 1\n        while len(counts) > k:\n            lc = s[left]\n            counts[lc] -= 1\n            if counts[lc] == 0:\n                del counts[lc]        # keep len(counts) == true distinct count\n            left += 1\n        best = max(best, right - left + 1)\n    return best"
  }
],

# ---------------- M1: classic ML (k-means + kNN) ----------------
"m1s1": [
  {
    "label": "cosine similarity",
    "prompt": "Same shape of problem, different metric this time: a **cosine similarity** matrix.\n```\ndef cosine_sim(A, B):\n```\n`A` is `(n, d)`, `B` is `(m, d)`; return the `(n, m)` matrix with entry `(i, j)` = `A[i]·B[j] / (||A[i]|| ||B[j]||)`. No Python loops.\n\nOne degenerate case: an all-zero row must produce similarity 0 (not NaN) — guard the norms with a small eps (1e-12 on the denominator).",
    "starter": None,
    "hints": [
      "Normalize the rows first: An = A / (norm + eps) with np.linalg.norm(A, axis=1, keepdims=True). Then the entire matrix is one matmul: An @ Bn.T.",
      "Put the eps on the DENOMINATOR: A / (norm + 1e-12). A zero row then normalizes to a zero row, and every similarity involving it is exactly 0 — no NaN, no special-casing."
    ],
    "tests": """
def _ref_cos(A, B):
    n, m = A.shape[0], B.shape[0]
    S = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            na, nb = np.linalg.norm(A[i]), np.linalg.norm(B[j])
            if na > 0 and nb > 0:
                S[i, j] = A[i] @ B[j] / (na * nb)
    return S

def _t_ref():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 4)); B = rng.standard_normal((5, 4))
    got = cosine_sim(A, B)
    assert got is not None, "function returned None"
    assert got.shape == (6, 5), f"expected shape (6, 5), got {got.shape}"
    assert np.allclose(got, _ref_cos(A, B), atol=1e-6), "values don't match a loop reference (normalize rows, then one matmul)"
_check("matches loop reference", _t_ref)

def _t_scale_invariant():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((4, 3)); B = rng.standard_normal((4, 3))
    assert np.allclose(cosine_sim(A * 7.0, B * 0.01), cosine_sim(A, B), atol=1e-6), "cosine similarity must be scale-invariant"
    assert np.allclose(np.diag(cosine_sim(A, A)), 1.0, atol=1e-6), "self-similarity of a nonzero vector must be 1"
_check("scale-invariance and unit self-similarity", _t_scale_invariant)

def _t_zero_row():
    A = np.array([[0.0, 0.0], [1.0, 0.0]])
    B = np.array([[3.0, 4.0]])
    S = cosine_sim(A, B)
    assert np.isfinite(S).all(), "zero vector produced NaN/inf -- guard the norm with eps in the denominator"
    assert abs(S[0, 0]) < 1e-6, "an all-zero row must give similarity 0"
_check("zero-vector row stays finite and 0", _t_zero_row)
""",
    "solution": "def cosine_sim(A, B):\n    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)  # zero row -> zero row\n    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)\n    return An @ Bn.T"
  }
],

"m1s2": [
  {
    "label": "debug: broken k-means",
    "prompt": "Debugging flavor this time. This k-means came out of a real screen and it returns garbage. There are **two distinct bugs** — find both, fix them in place, and tell me what symptom each one causes. Same contract as before: `(centroids, labels)`, and an empty cluster keeps its previous centroid.",
    "starter": "import numpy as np\n\ndef kmeans(X, k, init, n_iters=100):\n    C = init.astype(float).copy()\n    labels = None\n    for _ in range(n_iters):\n        D = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1)\n        new = D.argmin(axis=1)\n        if labels is not None and (new == labels).all():\n            break\n        labels = new\n        for j in range(k):\n            C[j] = X[labels == j].mean()\n    return C, labels\n",
    "hints": [
      "Run it on three obvious blobs and print C after one update — every centroid turns into a constant row. What does X[mask].mean() with NO axis argument compute, and what happens when you assign that scalar into C[j]?",
      "The second bug only fires when a cluster loses all its points: the mean of an empty slice is NaN, and the poisoned centroid never recovers. Guard the update with `if mask.any():`."
    ],
    "tests": """
def _blobs(seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [10.0, 10.0], [-10.0, 10.0]])
    X = np.vstack([c + rng.standard_normal((20, 2)) * 0.5 for c in centers])
    true = np.repeat(np.arange(3), 20)
    return X, centers, true

def _t_blobs():
    X, centers, true = _blobs()
    C, labels = kmeans(X, 3, init=centers + 0.7)
    labels = np.asarray(labels)
    for j in range(3):
        exp = X[true == j].mean(axis=0)
        assert np.allclose(C[j], exp, atol=1e-6), f"centroid {j} is wrong -- what axis does .mean() reduce over by default?"
        assert (labels[true == j] == j).all(), "all points of one blob must share one label"
_check("recovers three separated blobs (bug 1)", _t_blobs)

def _t_empty_cluster():
    X, centers, true = _blobs(2)
    init = np.vstack([centers + 0.7, [[1e6, 1e6]]])   # 4th centroid never wins a point
    C, labels = kmeans(X, 4, init=init)
    assert np.isfinite(C).all(), "centroids contain NaN -- the empty-cluster update is still broken"
    assert np.allclose(C[3], [1e6, 1e6]), "an empty cluster must keep its previous centroid"
_check("empty cluster survives (bug 2)", _t_empty_cluster)

def _t_fixed_point():
    X, centers, true = _blobs(1)
    C1, l1 = kmeans(X, 3, init=centers + 0.7)
    C2, l2 = kmeans(X, 3, init=C1)
    assert np.allclose(C1, C2, atol=1e-8), "converged centroids must be a fixed point"
_check("converged result is a fixed point", _t_fixed_point)
""",
    "solution": "def kmeans(X, k, init, n_iters=100):\n    C = init.astype(float).copy()\n    labels = None\n    for _ in range(n_iters):\n        D = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1)\n        new = D.argmin(axis=1)\n        if labels is not None and (new == labels).all():\n            break\n        labels = new\n        for j in range(k):\n            mask = labels == j\n            if mask.any():                       # bug 2: empty slice -> NaN centroid\n                C[j] = X[mask].mean(axis=0)      # bug 1: mean() with no axis is a SCALAR\n    return C, labels"
  }
],

"m1s3": [
  {
    "label": "kNN regression",
    "prompt": "Regression flavor this time — **inverse-distance-weighted kNN**:\n```\ndef knn_regress(X_train, y_train, X_test, k):\n```\n`y_train` is `(n_train,)` floats. For each test point take the `k` nearest training points by Euclidean distance `d_i` and return the weighted average of their targets with weights `w_i = 1 / (d_i + 1e-8)` — note: distance, **not** squared distance. Return a `(n_test,)` float array.\n\nSanity property: a query that coincides with a training point should return (essentially) that point's target.",
    "starter": None,
    "hints": [
      "Weights need TRUE distances: np.sqrt of your squared-distance matrix. Per row: idx = np.argsort(row)[:k], w = 1/(row[idx] + 1e-8), prediction = (w * y_train[idx]).sum() / w.sum().",
      "The exact-match case works by itself: d = 0 gives weight 1e8, which swamps the other neighbors — that's exactly why the eps goes on the DISTANCE, not anywhere else."
    ],
    "tests": """
def _ref_knnr(Xtr, ytr, Xte, k):
    out = []
    for x in Xte:
        d = np.sqrt(((Xtr - x) ** 2).sum(axis=1))
        idx = np.argsort(d)[:k]
        w = 1.0 / (d[idx] + 1e-8)
        out.append((w * ytr[idx]).sum() / w.sum())
    return np.array(out)

def _t_ref():
    rng = np.random.default_rng(0)
    Xtr = rng.standard_normal((30, 3)); ytr = rng.standard_normal(30)
    Xte = rng.standard_normal((8, 3))
    for k in (1, 3, 5):
        got = np.asarray(knn_regress(Xtr, ytr, Xte, k))
        exp = _ref_knnr(Xtr, ytr, Xte, k)
        assert np.allclose(got, exp, atol=1e-6), f"mismatch vs reference at k={k} -- weights use DISTANCE + 1e-8, not squared distance"
_check("matches reference for k in 1/3/5", _t_ref)

def _t_exact_match():
    Xtr = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    ytr = np.array([5.0, -2.0, 3.0])
    got = np.asarray(knn_regress(Xtr, ytr, Xtr[:1], 3))
    assert abs(got[0] - 5.0) < 1e-4, "a query equal to a training point must return ~that point's target (its weight 1/eps dominates)"
_check("exact-match query returns its own target", _t_exact_match)
""",
    "solution": "def knn_regress(X_train, y_train, X_test, k):\n    D = np.sqrt(pairwise_sq_dists(X_test, X_train))  # true distances for the weights\n    preds = np.empty(len(X_test))\n    for i, row in enumerate(D):\n        idx = np.argpartition(row, k - 1)[:k]        # O(n) selection, order irrelevant\n        w = 1.0 / (row[idx] + 1e-8)                  # exact match -> weight 1e8, dominates\n        preds[i] = (w * y_train[idx]).sum() / w.sum()\n    return preds"
  }
],
}


# ---------------------------------------------------------------------
# Verbal variants: alternate framings of the same resume question.
# The strong-answer bar is shared with the base question.
# ---------------------------------------------------------------------

VERBAL_VARIANTS = {

# r1 — multimodal pretraining
"r1q1": [{"prompt": "Different opening this time: imagine I'm a very senior researcher but I've never worked on multimodal. In two minutes, teach me what makes multimodal pretraining different from text pretraining — using YOUR work and one decision you personally made as the running example."}],
"r1q2": [{"prompt": "Let me come at the input question from the failure side: a competitor ships a model that beats yours badly on dense document understanding but loses on natural images. Reverse-engineer their vision-input design choices versus yours — encoder, resolution, token budget — and tell me which of your choices you'd now revisit."}],
"r1q3": [{"prompt": "Mixture question, adversarial edition: I claim data mixtures are over-engineered — 'just train on everything, dedup it, and let scale sort it out.' Attack or defend that position using what you actually observed when re-weighting multimodal mixtures."}],
"r1q4": [{"prompt": "Instead of your worst run, tell me about your best mid-run SAVE: a case where monitoring caught something early enough to fix cheaply. What signal fired, what would have happened without it, and what does that say about which dashboards actually matter?"}],
"r1q5": [{"prompt": "Flip the resource question: your vision pretraining compute is CUT by 10× for the next cycle. What do you protect, what do you drop, and what does the ranking reveal about where you believe the value really comes from?"}],

# r2 — scaling laws
"r2q1": [{"prompt": "Teaching framing: a strong new PhD hire has read Kaplan and Chinchilla and thinks they understand scaling laws. Give them the 'here's what the papers don't tell you' briefing for doing this on a real frontier multimodal run — the three biggest gaps between the papers and practice."}],
"r2q2": [{"prompt": "Concrete scenario: your text-only scaling fit is beautiful; adding the multimodal mixture makes the same methodology produce garbage — the fit residuals blow up. Walk me through your debugging: what are the candidate causes and in what order do you test them?"}],
"r2q3": [{"prompt": "The ladder, but under fire: a VP wants a risky architecture change in the next hero run. Your ladder says the change is neutral-to-negative but the error bars are wide. You have three weeks and finite compute. What exactly do you run, and what do you tell the VP at each decision point?"}],
"r2q4": [{"prompt": "Proxy-metric framing: convince me that loss on a held-out slice can stand in for a capability the board actually cares about — and then tell me the case from your own work where a proxy MISLED you and how you caught it."}],
"r2q5": [{"prompt": "Role-reversal: YOU are the reviewer. A team submits a hero-run proposal justified by scaling curves. List the five things you check in their methodology before signing off, in the order you'd check them — and which failure you see most often."}],

# r3 — science of data
"r3q1": [{"prompt": "Dedup framing flip: argue the OPPOSITE side first — give me the strongest case that aggressive dedup is harmful (what signal lives in repetition?), then tell me where you actually draw the line and why."}],
"r3q2": [{"prompt": "Filtering war story: tell me about the worst thing a quality filter ever silently did to a model you worked on — or if you caught it pre-launch, how. What did that change about your filter-review process?"}],
"r3q3": [{"prompt": "Budget framing: you get one engineer-year to invest in data infrastructure — embedding/clustering maps, attribution tooling, or filter evaluation harnesses. Which one, and what's the decision it improves within six months?"}],
"r3q4": [{"prompt": "Attribution, product edition: legal asks 'did source X materially contribute to capability Y' — for a renewal negotiation. How close can you actually get to answering that, with what method, and what error bars do you put on the answer?"}],
"r3q5": [{"prompt": "Flywheel skeptic: 'synthetic data is just model distillation into yourself; it can only smooth, never add information.' Steelman that, then tell me exactly where the information DOES enter in the pipelines you've seen work — and how you'd detect the day the flywheel starts eating its own tail."}],

# r4 — visual & physical understanding
"r4q1": [{"prompt": "Same end-to-end drill but I pick: **derendering**. Definition, why it's hard, the data recipe, the eval you trust, and the current frontier — and if that's not the capability you know deepest, tell me so and negotiate me to the one you do."}],
"r4q2": [{"prompt": "Product framing: the Maps team says your model can't reliably answer 'which building is left of the fountain' and they're blocked on launch. You have one quarter. What's your triage — data, resolution, visual CoT, or eval fixes first — and what do you promise them?"}],
"r4q3": [{"prompt": "Eval design under adversarial pressure: assume every benchmark you publish gets optimized against within months — by your own org. Design the eval process (not just the eval) that keeps measuring real capability under that pressure."}],
"r4q4": [{"prompt": "Interference, but as a postmortem: a release regressed video understanding after a big document-data push, and it shipped anyway because nobody caught it. Write the postmortem out loud: root cause, why the process missed it, and the two cheapest changes that prevent recurrence."}],
"r4q5": [{"prompt": "Contrarian roadmap edition: name a visual capability that's currently HYPED that you'd deprioritize, and the underrated one you'd fund with the freed resources. Make the case as if to a roadmap review."}],

# r5 — understanding <-> generation
"r5q1": [{"prompt": "Mechanism-first framing: rank the three MMU4MMGen mechanisms — recaptioning/filtering the generator's data, VLM-as-reward-model, and inference-time critique loops — by return on investment in your experience. Defend the ranking with what each one measurably moved."}],
"r5q2": [{"prompt": "Skeptic framing: 'generation helping understanding is a story researchers tell to justify unified models — contrastive pretraining plus captions wins every controlled comparison.' Where is that claim right, where is it wrong, and what's the cleanest experiment you know that separates the two?"}],
"r5q3": [{"prompt": "Design review framing: you're reviewing two competing proposals for the next model family — one fully unified understand+generate model, one specialist pair with a structured interface. What are the three questions you ask each team, and what answers would make you pick which?"}],
"r5q4": [{"prompt": "The organizational question, failure edition: describe how a joint venture like MMU4MMGen DIES — the standard failure modes of cross-org research collaborations — and which mechanisms you put in place specifically because you'd seen those failures before."}],
"r5q5": [{"prompt": "Investment framing: you're advising someone allocating research headcount for 2027 across understanding, generation, and the interface between them. Give the allocation and the portfolio logic — including what evidence would trigger a rebalance."}],

# r6 — post-training & agentic
"r6q1": [{"prompt": "Comparative framing: a text post-training lead is about to take over VLM post-training for the first time. Give them the 'three things that will surprise you' briefing — where their text instincts will be wrong — and the one habit that transfers perfectly."}],
"r6q2": [{"prompt": "Hallucination, incentive edition: I claim visual hallucination is fundamentally an INCENTIVES bug — raters reward confident detail — and no amount of perception improvement fixes it. Where is that right, where does it break, and what did you change in the reward pipeline versus the perception stack?"}],
"r6q3": [{"prompt": "UI control, data-strategy edition: human demonstration traces are expensive and go stale as UIs change. Design the data strategy that keeps an operator agent current for a year — the human/synthetic/rollout mix, the refresh triggers, and the eval that tells you the data has gone stale."}],
"r6q4": [{"prompt": "Verticals, prioritization edition: Health, Geo, and Drive all escalate on the same week wanting model improvements for THEIR capability. You can properly serve one this cycle. Walk me through the prioritization call and how you keep the other two from forking the model."}],
"r6q5": [{"prompt": "Prediction framing: post-training for Gemini 5 — what does it look like? Extrapolate the Gemini 1→3 trendline you lived through: what grows, what disappears, and what's the contrarian bet you'd staff that most post-training leads wouldn't?"}],
}
