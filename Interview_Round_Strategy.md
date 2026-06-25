# Frontier-Lab Loop — Round Strategy (F / W / S)

**Why this doc:** A strategy for a 3-round frontier-lab Research Scientist loop, tuned to each interviewer's research focus. Interviewers anonymized as **F**, **W**, **S**.

- **F — AI coding round.** Focus area: **post-training / RL / agents** (multimodal VLMs, long-horizon reasoning, embodied/agentic systems background).
- **W — AI coding round.** Focus area: **post-training / RL / agents** (rigorous, production-grade engineering background in perception/3D vision).
- **S — Research round.** Focus area: **pretraining & model architecture** (MoE, long-context/memory, efficiency & test-time compute, model merging, unified multimodal; deep architecture + normalization roots).

> The two coding interviewers do post-training/RL/agents → expect **post-training/RL/agent-flavored ML coding**, not generic LeetCode. The research interviewer is an architecture expert → expect deep architecture tradeoff + experimental-design discussion.

---

## 4-day plan

| Day | Focus | Serves | Material |
|-----|-------|--------|----------|
| 1 | Architecture + coding foundations | all | Day 1 §4, Day 2 §1–3 |
| 2 | **Post-training / RL / Agents (deep)** | F + W | Post-Training/RL doc, Day 4 §3, Coding doc |
| 3 | Pretraining, architecture frontier + research taste | S | LLM Architecture Frontier doc, Day 2 §5, Day 5 |
| 4 | Coding mocks + mock drill + behavioral | all | Coding doc, Day 4 §5, Day 5 §6 |

**Day 1** — read transformer architecture deep (attention, MHA, cross-attention, normalization, RoPE, pre/post-norm, MoE intro); then CODE blind + timed: softmax+CE, 2-layer backprop, self-attention → MHA → cross-attention, LayerNorm/RMSNorm, transformer block, KV cache, top-k/top-p.

**Day 2 (top priority)** — RL mechanics (policy gradient, PPO objective/advantage/clipping/KL, reward models) + SFT→RLHF→DPO→RLAIF→rejection→verifiable; agents (tool use, ReAct, long-horizon, eval); CODE: DPO loss, reward head + Bradley-Terry, PPO clipped objective, rollout / best-of-n.

**Day 3 (research round)** — pretraining (MLE, data mixtures, tokenization), scaling laws/Chinchilla; architecture frontier (MoE, long-context & memory layers, efficiency & test-time compute, model merging, unified multimodal); research taste + 5-point paper frame + frontier/Llama papers.

**Day 4** — timed coding mocks (coding-round sim), full #16 mock drill, research-proposal drill (research-round sim), 5 behavioral stories, weak-spot review.

**De-prioritize** (4 days): classical ML, audio/video multimodal, deep distributed-systems internals.

---

## Rounds 1 & 2 — AI coding (F, W)

**What "AI coding" means at a foundation-model lab:** implement ML components from scratch in PyTorch/NumPy + solid algorithmic ability + debugging. Both interviewers do post-training/RL/agents, so expect that flavor.

### Top priority — implement from memory, clean and fast
- **Self-attention → multi-head → causal mask → cross-attention.** Most likely single ask; vectorized, correct shapes, ~10 min.
- **Post-training losses:** **DPO loss**, **reward-model head + Bradley-Terry loss**, **PPO clipped objective**, a **sampling / best-of-n rollout** loop. (See the Coding From Scratch doc.)
- **Backprop for a 2-layer MLP** (softmax-CE gradient = ŷ − y).
- LayerNorm/RMSNorm, residual block, full transformer block, KV-cache generate loop, top-k/top-p.

### General DS&A (don't neglect)
Arrays/strings, hashmaps, two-pointers, sliding window, BFS/DFS, heaps, intervals, DP basics. At least one round may include a "normal" algorithmic problem.

### How to win these rounds (style)
- **Narrate shapes and invariants** as you code; test on a small example; handle edge cases (T=1, batch dim).
- For **F:** connect "why" answers to *behavior* — reasoning, multimodality, agentic/long-horizon use (e.g. why KV cache matters for an agent doing long rollouts; cross-attention as VLM fusion).
- For **W:** emphasize **correctness + efficiency**; vectorize, call out complexity, mention how you'd unit-test / what breaks at scale.

---

## Round 3 — Research (S)

An LLM-architecture expert. Expect: paper discussion, architectural tradeoffs, experimental design, "what would you try / why do you believe this result."

### Priority topics (LLM architecture frontier)
- **MoE:** top-k routing, load-balancing aux loss, router/expert collapse, capacity factor, fine-grained + shared experts, params-vs-FLOPs, inference challenges, parameter-free expert selection.
- **Long-context & memory:** memory layers / product-key memory, KV-cache scaling, attention variants (GQA, sliding window, linear/state-space), position extrapolation (RoPE scaling), retrieval vs long-context.
- **Efficiency & test-time compute:** looped/recurrent-depth transformers, adaptive computation, test-time "thinking" compute, speculative decoding.
- **Model merging:** model soups, task vectors, TIES/DARE, merging during pretraining.
- **Unified multimodal:** autoregressive image generation + understanding, continuous vs discrete visual tokens, image tokenizers.
- **Scaling & width:** scaling laws, width/depth tradeoffs, normalization & training dynamics (incl. weight normalization for micro-batch).

### How to win this round
- Discuss every architecture as a **capacity / compute / memory tradeoff with a failure mode**, not a definition.
- Open paper discussion with the **5-point frame:** Problem · Core idea · Why it worked · Limitation · *one follow-up experiment.*
- Bring a **specific, falsifiable, compute-aware** proposal to every open question (cheap proxy first, scale winners).
- Have a crisp take on the **vision → multimodal → foundation-model convergence** ready.

---

## Cross-cutting priorities (ranked)

1. **Code attention + transformer block + DPO/reward/PPO losses + backprop from a blank file, timed.** (Rounds 1–2 core.)
2. **Post-Training & RL mechanics** — policy gradient → PPO → DPO derivations, reward modeling, verifiable rewards. (Rounds 1–2.)
3. **LLM Architecture Frontier** — MoE, long-context/memory, test-time compute, merging, unified multimodal. (Round 3.)
4. **Research taste + 5-point paper frame** — for the research round.
5. **General DS&A** — hedge for the coding rounds.
6. **2–3 frontier papers + Llama papers** in the 5-point frame — general bar.

## Linked material
- Post-Training & RL Mechanics deep dive
- ML Coding From Scratch reference
- LLM Architecture Frontier deep dive
- Days 1–5 + Activation notes + Q&A log
