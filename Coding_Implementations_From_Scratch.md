# ML Coding From Scratch — Implementation Reference

**Why this doc:** The coding rounds test whether you can implement core ML primitives **from a blank file**, correctly and efficiently. This is a reference of the highest-probability asks, with clean PyTorch you should be able to reproduce from memory. Given your interviewers' focus, the **attention family** and **post-training losses (DPO, reward model, PPO)** are the most likely.

**How to use it:** don't just read — *close the doc and re-type each one timed.* For each, I note what an interviewer probes. Code targets clarity + correctness first, then efficiency (vectorized, no Python loops over tokens).

> Assumes `import torch`, `import torch.nn as nn`, `import torch.nn.functional as F`. Shapes use `B`=batch, `T`=sequence length, `D`=model dim, `H`=heads.

---

## 1. Stable softmax + cross-entropy

```python
def softmax(z, dim=-1):
    z = z - z.max(dim=dim, keepdim=True).values     # subtract max for numerical stability
    e = torch.exp(z)
    return e / e.sum(dim=dim, keepdim=True)

def cross_entropy(logits, targets):
    # logits: (B, C), targets: (B,) int class indices
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)  # log-softmax, stable
    return -logp[torch.arange(len(targets)), targets].mean()
```
**Probes:** Why subtract the max? (overflow). Why `logsumexp` instead of `log(sum(exp))`? (stability). The softmax-CE gradient is `ŷ − y` — be ready to say it.

---

## 2. Scaled dot-product attention (the #1 ask)

```python
def attention(Q, K, V, mask=None):
    # Q,K,V: (B, T, d_k). Returns (B, T, d_v)
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / d_k**0.5          # (B, T, T)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    return weights @ V
```
**Probes:** Why `/√d_k`? (keeps dot-product variance ~1 so softmax doesn't saturate). Why `-inf` before softmax? (→ 0 weight). Get the `transpose(-2,-1)` and shapes right.

---

## 3. Causal mask

```python
def causal_mask(T):
    # lower-triangular: position t attends to <= t
    return torch.tril(torch.ones(T, T)).bool()           # (T, T), True = keep
```
Used as `attention(Q, K, V, mask=causal_mask(T))`. **Probe:** why causal masking enables parallel next-token training (each position predicts the next without seeing it).

---

## 4. Multi-head attention (write this fluently)

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d_k = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                  # each (B, H, T, d_k)
        scores = q @ k.transpose(-2, -1) / self.d_k**0.5  # (B, H, T, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1) @ v              # (B, H, T, d_k)
        attn = attn.transpose(1, 2).reshape(B, T, D)      # concat heads
        return self.out(attn)
```
**Probes:** the reshape/permute to split heads (most common bug spot — narrate shapes), why heads (different representation subspaces), total compute ≈ single full-width attention.

---

## 5. Cross-attention (likely for Fei — VLM / encoder-decoder)

Same math, but **Q comes from one stream, K/V from another** (e.g. text queries attending to image features).

```python
class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h, self.d_k = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x_q, x_kv):
        # x_q: (B, Tq, D) e.g. text;  x_kv: (B, Tk, D) e.g. image tokens
        B, Tq, D = x_q.shape
        Tk = x_kv.size(1)
        q = self.q(x_q).reshape(B, Tq, self.h, self.d_k).transpose(1, 2)
        kv = self.kv(x_kv).reshape(B, Tk, 2, self.h, self.d_k).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]                               # (B, H, Tk, d_k)
        scores = q @ k.transpose(-2, -1) / self.d_k**0.5  # (B, H, Tq, Tk)
        attn = F.softmax(scores, dim=-1) @ v
        attn = attn.transpose(1, 2).reshape(B, Tq, D)
        return self.out(attn)
```
**Probes:** the difference from self-attention (Q vs K/V sources, asymmetric Tq≠Tk), and how this is the fusion mechanism in Flamingo-style VLMs.

---

## 6. LayerNorm & RMSNorm

```python
class LayerNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d)); self.b = nn.Parameter(torch.zeros(d))
        self.eps = eps
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.g * (x - mu) / torch.sqrt(var + self.eps) + self.b

class RMSNorm(nn.Module):                                  # no mean-centering; cheaper
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.g * x / rms
```
**Probes:** LN normalizes over the **feature dim within each token** (not the batch) — why transformers use it over BN. RMSNorm drops the mean subtraction. (Tie-in: Siyuan's Weight Standardization normalizes the *weights* instead.)

---

## 7. Transformer block (pre-norm, decoder-style)

```python
class Block(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4):
        super().__init__()
        self.ln1, self.ln2 = LayerNorm(d_model), LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model), nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model))
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)   # pre-norm: norm INSIDE the residual
        x = x + self.mlp(self.ln2(x))
        return x
```
**Probes:** pre-norm vs post-norm (norm location relative to residual; pre-norm trains more stably), attention mixes across tokens / MLP processes each token, the residual `x +` as the gradient highway.

---

## 8. KV-cache generation loop

```python
@torch.no_grad()
def generate(model, idx, max_new, kv_cache=None):
    for _ in range(max_new):
        logits = model(idx[:, -1:], kv_cache)   # only feed the LAST token; cache holds the rest
        probs = F.softmax(logits[:, -1], dim=-1)
        nxt = torch.multinomial(probs, 1)
        idx = torch.cat([idx, nxt], dim=1)
    return idx
```
**Probes:** why cache K/V (they don't change for past tokens → avoid recompute), why decode is memory-bandwidth-bound, cache size grows linearly with T (→ GQA/MQA shrink it).

---

## 9. Top-k / top-p (nucleus) sampling

```python
def sample(logits, temperature=1.0, top_k=None, top_p=None):
    logits = logits / temperature
    if top_k is not None:
        kth = torch.topk(logits, top_k).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float('-inf'))
    if top_p is not None:
        s, idx = torch.sort(logits, descending=True)
        cum = torch.softmax(s, dim=-1).cumsum(-1)
        s = s.masked_fill(cum - torch.softmax(s, -1) > top_p, float('-inf'))
        logits = torch.full_like(logits, float('-inf')).scatter(-1, idx, s)
    return torch.multinomial(F.softmax(logits, dim=-1), 1)
```
**Probes:** temperature (sharpens/flattens the distribution), top-k (fixed count) vs top-p (smallest set whose cumulative prob ≥ p — adaptive).

---

## 10. DPO loss (high-probability post-training ask)

```python
def dpo_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l, beta=0.1):
    # each arg: (B,) = sum of token log-probs of the response under that model
    pi_logratio  = pi_logps_w  - pi_logps_l           # policy's preferred-vs-dispreferred margin
    ref_logratio = ref_logps_w - ref_logps_l          # reference's margin
    logits = beta * (pi_logratio - ref_logratio)      # implicit reward difference
    return -F.logsigmoid(logits).mean()
```
**Probes:** where the `β log(π/π_ref)` implicit reward comes from (DPO derivation), why it's relative to a frozen reference (controls deviation, like the KL term), and how you'd get the per-response log-probs (sum token log-probs over the response, masking the prompt).

Helper they may ask for — sequence log-prob with prompt masking:
```python
def sequence_logp(logits, labels, prompt_len):
    # logits: (B, T, V); labels: (B, T); only score response tokens
    logp = F.log_softmax(logits[:, :-1], dim=-1)
    tok = logp.gather(-1, labels[:, 1:].unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    mask = torch.arange(tok.size(1)) >= (prompt_len - 1)            # ignore prompt
    return (tok * mask).sum(-1)
```

---

## 11. Reward model head + Bradley-Terry loss

```python
class RewardModel(nn.Module):
    def __init__(self, backbone, d_model):
        super().__init__()
        self.backbone = backbone                 # an LLM producing (B, T, D)
        self.head = nn.Linear(d_model, 1)        # scalar reward
    def forward(self, x):
        h = self.backbone(x)                     # (B, T, D)
        return self.head(h[:, -1]).squeeze(-1)   # reward at last token -> (B,)

def reward_loss(r_w, r_l):
    # r_w, r_l: (B,) rewards for preferred / dispreferred responses
    return -F.logsigmoid(r_w - r_l).mean()       # Bradley-Terry
```
**Probes:** why score the **last token** (it has seen the whole response), why only score *differences* matter (absolute scale is arbitrary), and how this RM feeds PPO.

---

## 12. PPO clipped objective (core)

```python
def ppo_loss(logp, old_logp, advantages, eps=0.2):
    ratio = torch.exp(logp - old_logp)                 # π_new / π_old
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
    return -torch.min(unclipped, clipped).mean()       # maximize -> negate
```
**Probes:** why the ratio (importance sampling lets you reuse the batch for several steps), why clip (trust region — bound the step), why `min` (pessimistic bound removes incentive to exceed the clip). Full PPO adds value loss + entropy.

---

## 13. Best-of-n / rejection sampling

```python
@torch.no_grad()
def best_of_n(model, reward_model, prompt, n=8):
    cands = [model.generate(prompt) for _ in range(n)]   # n samples
    scores = torch.stack([reward_model(c) for c in cands])
    return cands[scores.argmax()]
```
**Probes:** simple but strong baseline; bounded by the base model's best sample; used to bootstrap better SFT data (rejection sampling = SFT on the best-of-n).

---

## 14. (Bonus) Backprop for a 2-layer MLP in NumPy

```python
# forward:  z1 = W1@x + b1 ; a1 = relu(z1) ; z2 = W2@a1 + b2 ; p = softmax(z2)
# loss = cross_entropy(p, y)
d2  = p - y_onehot              # dL/dz2  (the ŷ - y identity)
dW2 = d2 @ a1.T                 # outer product
db2 = d2
da1 = W2.T @ d2                 # push back through linear
d1  = da1 * (z1 > 0)            # back through ReLU = mask
dW1 = d1 @ x.T
db1 = d1
```
**Probes:** the three rules — weight grad `= δ aᵀ`, back through linear `= Wᵀδ`, back through activation `= ⊙ φ'`. (Full detail in Day 1 §2.)

---

## Drill checklist (re-type each from blank, timed)

- [ ] stable softmax + cross-entropy
- [ ] scaled dot-product attention + causal mask
- [ ] multi-head attention (shapes from memory)
- [ ] cross-attention
- [ ] LayerNorm / RMSNorm
- [ ] transformer block (pre-norm)
- [ ] KV-cache generate loop
- [ ] top-k / top-p sampling
- [ ] **DPO loss** + sequence log-prob with prompt masking
- [ ] **reward model head + Bradley-Terry loss**
- [ ] **PPO clipped objective**
- [ ] best-of-n
- [ ] 2-layer MLP backprop (NumPy)

**Interview style reminders:** narrate shapes as you go; test on a tiny example; handle batch dim and edge cases (T=1, empty); call out complexity (attention O(T²·d)); mention numerical stability (logsumexp, subtract max). For Weiyue: emphasize correctness + efficiency. For Fei: connect to reasoning/agents/multimodal.
