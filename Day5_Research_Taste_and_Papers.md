# Day 5 — Research Taste + Paper Deep-Dives

**Target:** Research Scientist loop at a frontier lab (Meta TBD / MSL style).
**Mode:** Concepts + spoken practice; this is the differentiator day.
**Time budget:** ~8 hours.
**The bar for today:** Walk in able to (1) discuss 2–3 papers with *real* depth using a repeatable frame, (2) propose **sharp, falsifiable, compute-aware** experiments, and (3) reason about *what's worth doing* — the skill that separates "studied ML" from "thinks like a researcher."

> Research taste is what RS interviews actually select for. The technical days give you the substrate; today is about *judgment* on top of it.

---

## Suggested 8-hour schedule

| Block | Time | Topic | Section |
|-------|------|-------|---------|
| 1 | 0:00–1:30 | Research taste fundamentals + the weak→strong pattern | §1 |
| 2 | 1:30–2:15 | The 5-point paper deep-dive frame | §2 |
| — | 2:15–2:30 | Break | |
| 3 | 2:30–3:45 | Anchor paper 1: Chinchilla (scaling) | §3 |
| 4 | 3:45–5:15 | Anchor topic 2: RL/DPO landscape | §4 |
| — | 5:15–5:30 | Break | |
| 5 | 5:30–6:45 | Anchor paper 3: FlashAttention (systems) | §5 |
| 6 | 6:45–8:00 | Research-proposal practice (out loud) | §6 |

---

## §1 — Research taste fundamentals

### 1.1 What "research taste" actually means

It's the set of judgments that determine whether your time/compute produces real progress:
- **Problem selection:** picking high-leverage problems (big if true, tractable, on the critical path) over merely interesting ones.
- **Hypothesis formation:** stating a *specific, falsifiable* prediction before running anything.
- **Clean experimental design:** controls, baselines, ablations, isolating one variable (Day 3 §3.3).
- **Skepticism:** distrusting noisy/too-good results; asking "what would make this not true?"
- **Debugging negative results:** distinguishing "the idea is wrong" from "the implementation/eval is broken."
- **Compute-awareness:** knowing what's worth scaling and what to test cheaply first.
- **Knowing when to stop:** killing a direction that isn't working.

### 1.2 The weak → strong answer pattern (internalize this)

Frontier interviewers can tell taste from *how you answer open questions.* The difference:

**Weak (vague, not falsifiable, not compute-aware):**
> "I'd train a better model with more and cleaner data."

**Strong (specific, falsifiable, controlled, compute-aware):**
> "I'd test whether targeted high-quality synthetic math traces improve pass@k on *held-out, post-cutoff* contest problems, after controlling for contamination. I'd run small-scale **data-mixture ablations** first — same model size and token budget, varying only the math-trace fraction — and scale only the mixtures that improve reasoning **without** regressing general chat quality, refusal rate, or hallucination."

What makes the strong answer strong — check your own answers against these:
1. **Specific intervention** (targeted synthetic math traces, not "better data").
2. **Falsifiable metric** (pass@k on held-out problems) with a **clear success/failure line**.
3. **Controls** (contamination, compute-matched, hold other capabilities fixed).
4. **Compute-awareness** (small-scale first, scale only winners).
5. **Awareness of side effects / tradeoffs** (don't regress chat/safety — the alignment-tax instinct).

### 1.3 The taste question bank (practice all out loud)

- "What paper changed your thinking recently?" → have 2–3 ready (§3–§5).
- "What result do you *not* believe?" → pick something plausibly over-claimed; explain the likely confound (contamination, cherry-picked eval, no compute-matched baseline).
- "What would you work on with **1,000 GPUs**?" vs "with only **8 GPUs**?" → big-compute: a real scaling/data/post-training bet; small-compute: clever proxy experiments, ablations, evals, mechanistic studies.
- "How would you improve LLM reasoning?" → verifiable-reward RL, process supervision, high-quality traces, better eval to avoid contamination — be specific.
- "How would you make agents more reliable?" → error-recovery training, environment-feedback RL, better long-horizon eval, prompt-injection robustness.
- "What's an underrated bottleneck in frontier AI?" → e.g. evaluation/contamination, data quality, inference cost, long-horizon reliability — pick one and defend it.
- "What experiment would you run first?" → always the **cheapest decisive** one.

> **The meta-skill:** for *every* open question, produce a *specific, falsifiable, compute-aware* answer with a named first experiment and an awareness of what could confound it.

### Self-test §1
1. **Turn a weak answer into a strong one.** → Add: specific intervention, falsifiable metric + success line, controls (contamination/compute-matched), small-scale-first, side-effect awareness.
2. **"What result do you not believe, and why?"** → Name one; cite the missing control (compute-matched baseline, contamination check, CIs, cherry-picked eval).
3. **"1,000 GPUs vs 8 GPUs?"** → Big: a scaling/data/post-training bet with a clear hypothesis. Small: proxy/ablation/eval/mechanistic work that informs the big bet cheaply.

---

## §2 — The 5-point paper deep-dive frame

Use this for **any** paper — so you can speak to one an interviewer raises, not only your prepped three.

For each paper, be able to state:
1. **Problem** — what limitation/question motivated it? (Why did this need to exist?)
2. **Core idea** — the key insight, in one or two sentences.
3. **Why it worked** — the mechanism; *why* the idea addresses the problem.
4. **Main limitation** — what it doesn't solve, its cost, or where it breaks.
5. **One follow-up experiment you'd run** — shows you think like a contributor, not a reader.

Practice compressing each anchor paper into these five points until you can do it from memory in ~90 seconds. The follow-up (point 5) is where you demonstrate taste — make it *specific and falsifiable.*

---

## §3 — Anchor 1: Chinchilla (compute-optimal scaling)

*Hoffmann et al., 2022 — "Training Compute-Optimal Large Language Models."*

**1. Problem.** Given a fixed compute budget, how should you split it between **model size (N)** and **training tokens (D)**? Models like GPT-3 were huge but (it turned out) undertrained — was that optimal?

**2. Core idea.** Empirically fit how loss depends on N and D at fixed compute; find the compute-optimal allocation. Result: **N and D should scale roughly equally** — about **20 training tokens per parameter**. Most large models of the era were **too big and trained on too little data.**

**3. Why it worked.** Loss follows predictable **power laws** in N and D; by training many models across the N/D grid and fitting the surface, you can *solve* for the optimum at a given compute. Chinchilla (70B, 1.4T tokens) beat Gopher (280B) using the **same compute**, validating the prediction — a smaller, better-trained model won.

**4. Main limitation.** It optimizes **training** compute only. It ignores **inference cost** — if you serve the model a lot, a *smaller* model trained on *more* tokens than Chinchilla-optimal ("overtraining") is better overall because it's cheaper to run. (LLaMA models deliberately overtrain for this reason.) Also assumes a fixed data distribution and enough non-repeated data — data quality and repetition complicate the clean power laws.

**5. Follow-up you'd run.** Re-derive the optimal N/D frontier under a **combined training+inference compute objective** for a target query volume — quantifying how far past Chinchilla-optimal you should overtrain as a function of expected inference load. (Specific, falsifiable, decision-relevant.)

**Why this is a great anchor:** it's the densest "research taste" paper — it's about *allocating* resources, power-law extrapolation, and the deep point that **compute-optimal-for-training ≠ optimal-for-deployment.** Say that line in an interview.

### Self-test §3
1. **What did Chinchilla show and why does it matter?** → Compute-optimal training balances N and D (~20 tok/param); prior models were undertrained. It lets you de-risk big runs by extrapolating power laws.
2. **What's Chinchilla's blind spot?** → Inference cost — overtraining a smaller model is better when serving heavily. Training-optimal ≠ deployment-optimal.

---

## §4 — Anchor 2: The RL / post-training landscape (anchored on DPO)

Rather than one paper, own the **comparative map** (deep version of Day 2 §6). The interviewer wants to see you reason about *tradeoffs in practice*.

### 4.1 Why RL for LLMs at all
We want to optimize **human preference**, which is **non-differentiable** and easier to *compare* than to *demonstrate*. So: learn from comparisons, optimize the policy toward preferred behavior. KL-regularize to a reference model so you don't drift into degenerate, reward-hacking text.

### 4.2 The methods and their practical pros/cons

| Method | How it works | Pros | Cons / when it fails |
|---|---|---|---|
| **SFT** | Fine-tune on (prompt, good answer) demos | Simple, stable, essential foundation | Only shows one good answer; can't rank; demos expensive; no negative signal |
| **PPO-RLHF** | Reward model from preferences + on-policy RL with KL penalty | Powerful; **on-policy** (learns from its own current outputs); reward model reusable; can keep improving | Complex/unstable; 4 models (policy, ref, RM, critic); expensive; reward hacking; hyperparameter-sensitive |
| **DPO** | Reparameterize so optimal policy relates to reward in closed form → optimize preferences directly, **offline** | Simple, stable, cheap (no RM, no RL loop); strong results | **Offline** — bound to the fixed preference dataset's coverage; can't learn from fresh on-policy mistakes; less control |
| **RLAIF / Constitutional** | AI-generated preferences guided by principles | Scales past human labeling; consistent, steerable | Inherits the judge model's biases; quality depends on the constitution |
| **Rejection sampling / best-of-n** | Sample n, keep best by an RM, fine-tune on those | Dead simple; strong baseline; bootstraps better SFT data | Bounded by the base model's best samples; compute at sampling time |
| **Verifiable-reward RL** | Reward = automatic checker (tests pass / answer correct) | Clean, **ungameable** reward → strong signal; behind reasoning gains | Only works where you can verify (math, code); reward sparsity |

### 4.3 The judgments to articulate
- **SFT vs RLHF:** demos teach behavior/format; preference optimization teaches *ranking* better-vs-worse, which demos can't.
- **DPO vs PPO:** DPO is simpler/cheaper/more stable and often matches PPO; **PPO/online RL pulls ahead when you can iterate with fresh on-policy data**, because the model learns from its *own current* failures rather than a static dataset.
- **Process vs outcome supervision:** rewarding each reasoning *step* (process) gives denser signal and better reasoning than rewarding only the final answer (outcome), at higher labeling cost.
- **The universal risks:** reward hacking (→ KL penalty), alignment tax (capability dip), over/under-refusal, sycophancy.

### Self-test §4
1. **Walk the post-training landscape with pros/cons.** → Use the table: SFT (foundation, no ranking) → PPO-RLHF (powerful, on-policy, heavy/unstable) → DPO (simple/offline, coverage-bound) → RLAIF (scales labels) → rejection sampling (strong baseline) → verifiable-reward (clean signal, math/code only).
2. **When does PPO beat DPO in practice?** → When you can generate fresh on-policy data and iterate — the model learns from its own current mistakes vs DPO's static dataset.
3. **Process vs outcome supervision?** → Process rewards each step (denser, better reasoning, costlier); outcome rewards only the final answer.

---

## §5 — Anchor 3: FlashAttention (IO-aware exact attention)

*Dao et al., 2022 — "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness."*

**1. Problem.** Standard attention materializes the full **N×N** score matrix in slow GPU **high-bandwidth memory (HBM)**, making it **memory-bandwidth-bound** and O(N²) in memory — the bottleneck for long sequences (Day 2 §2.5, §4.3). The compute isn't the limit; the **memory traffic** is.

**2. Core idea.** Compute attention **without ever materializing the full score matrix**. Tile Q/K/V into blocks that fit in fast on-chip **SRAM**, compute attention block-by-block, and use the **online softmax** trick to combine blocks correctly. It's **exact** (not an approximation) — same result, far less HBM traffic.

**3. Why it worked.** It's an **IO-aware** algorithm: GPU performance here is gated by reads/writes between HBM and SRAM, not FLOPs. By keeping intermediate work in SRAM and avoiding the N×N write/read to HBM, it slashes memory traffic → big wall-clock speedups and **O(N) memory** instead of O(N²). This is the canonical "know your hardware" result — same math, restructured around the memory hierarchy.

**4. Main limitation.** It's still **O(N²) compute** (it attacks memory, not the fundamental quadratic) — truly long context still needs sparse/linear attention or other tricks. It's also a hardware-specific, hand-tuned kernel (GPU-dependent); newer versions (FlashAttention-2/3) refine the parallelization for newer GPUs.

**5. Follow-up you'd run.** Combine FlashAttention's IO-aware tiling with a **sparse/sliding-window** attention pattern and measure the wall-clock vs quality frontier at long context — testing whether you can get sub-quadratic compute *and* IO-efficiency without quality loss on long-context retrieval tasks.

**Why this anchor shows range:** it connects the **attention math (Day 2)** to **GPU memory hierarchy (systems)** — exactly the "reason from math to hardware" signal an RS+systems-literate candidate wants. Saying "the bottleneck was HBM traffic, not FLOPs — they restructured the computation around SRAM" is a strong line.

### Self-test §5
1. **What problem does FlashAttention solve and how?** → Standard attention is memory-bandwidth-bound from materializing the N×N matrix in HBM; FlashAttention tiles Q/K/V into SRAM and uses online softmax to compute attention exactly without ever writing the full matrix → O(N) memory, big speedup.
2. **Is it an approximation?** → No — exact attention; it changes the IO pattern, not the math.
3. **What's its limitation?** → Still O(N²) compute; attacks memory, not the quadratic. Long context still needs sparse/linear methods.

---

## §6 — Research-proposal practice (do this out loud)

For each prompt below: give a **2–3 minute spoken proposal** with **(a)** a specific hypothesis, **(b)** a falsifiable metric + success line, **(c)** controls (contamination, compute-matched, side-effects), **(d)** the *cheapest decisive first experiment*, and **(e)** what would make you kill it. Then self-grade against the §1.2 strong-answer checklist.

1. **"How would you improve LLM reasoning?"**
2. **"How would you make agents more reliable on long-horizon tasks?"**
3. **"You have 10× more compute — what do you do?"** (Anchor on Chinchilla: scale data with model; but weigh inference cost and post-training ROI — don't just say "bigger model.")
4. **"What's an underrated bottleneck in frontier AI, and how would you attack it?"**
5. **"Design an experiment to test whether synthetic data helps reasoning without hurting general ability."** (This is the §1.2 worked example — deliver it cleanly.)

**Self-grade rubric (per answer):**
- [ ] Specific intervention (not "more/better X")
- [ ] Falsifiable metric with a success/failure line
- [ ] Controls: contamination, compute-matched, hold other capabilities fixed
- [ ] Compute-aware: cheap proxy / small-scale first, scale winners
- [ ] Names the first experiment and a kill condition
- [ ] Aware of side effects (alignment tax, regressions, safety)

### What "great" sounds like today
- You discuss each anchor paper in the **5-point frame** from memory, ending with a *specific* follow-up.
- Every open-ended answer is **specific, falsifiable, controlled, and compute-aware**, with a named first experiment.
- You volunteer the deep lines: *"training-optimal ≠ deployment-optimal" (Chinchilla); "the bottleneck was HBM traffic, not FLOPs" (FlashAttention); "PPO wins when you can iterate on-policy" (post-training).*

---

## Quick-reference cheat sheet

- **Research taste = problem selection + falsifiable hypotheses + clean controls + skepticism + compute-awareness + knowing when to stop.**
- **Weak→strong:** specific intervention · falsifiable metric + success line · controls (contamination, compute-matched) · small-scale-first · side-effect awareness · named first experiment + kill condition.
- **Paper frame (any paper):** Problem · Core idea · Why it worked · Limitation · Your follow-up.
- **Chinchilla:** balance N & D (~20 tok/param); power-law extrapolation de-risks big runs; **training-optimal ≠ deployment-optimal** (overtrain to cut inference cost).
- **Post-training landscape:** SFT (foundation) · PPO-RLHF (powerful, on-policy, heavy) · DPO (simple, offline, coverage-bound) · RLAIF (scales labels) · rejection sampling (strong baseline) · verifiable-reward (clean signal, math/code). PPO beats DPO when you can iterate on-policy. KL penalty fights reward hacking.
- **FlashAttention:** exact attention, IO-aware tiling into SRAM + online softmax → O(N) memory, big speedup; **memory traffic was the bottleneck, not FLOPs**; still O(N²) compute.
- **For every open question:** specific, falsifiable, compute-aware, cheapest-decisive-first-experiment.
