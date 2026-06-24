# Post-Training & RL for LLMs — Mechanics Deep Dive

**Why this doc:** Your coding/research conversations will lean heavily on **post-training, RL, and agents**. This goes a level deeper than the Day 2 overview — into the *mechanics* (policy gradient → PPO → DPO derivations, reward modeling, GRPO) that an RL-focused interviewer expects you to actually understand, not just name.

**The bar:** be able to derive the DPO loss, write the PPO clipped objective and say why each piece exists, explain the reward-model loss, and reason about reward hacking / KL / verifiable rewards. Pair this with the coding doc (`Coding_Implementations_From_Scratch`).

> Notation: policy $π_θ$ (the LLM), reference/old policy $π_{ref}$, prompt $x$, response $y$, reward $r$. Equations are in code blocks for readability.

---

## §1 — The RL framing of LLM post-training

An autoregressive LLM **is a policy**. Map it to RL:

| RL concept | LLM equivalent |
|---|---|
| State $s_t$ | prompt + tokens generated so far |
| Action $a_t$ | the next token |
| Policy $π_θ(a_t\mid s_t)$ | the LLM's next-token distribution |
| Trajectory | a full generated response $y$ |
| Reward $R$ | a scalar score for the whole response (usually only at the end) |

We want to **maximize expected reward** of generations:
```
J(θ) = E_{x~D, y~π_θ(·|x)} [ R(x, y) ]
```
The catch: $R$ comes from human preference or a verifier — it's **non-differentiable** w.r.t. $θ$, and you can't backprop through sampling. That's why we need **policy-gradient** methods.

---

## §2 — Policy gradient (REINFORCE)

### The log-derivative trick (derive this)
We can't differentiate through the sampling, but we can use:
```
∇θ E_{y~π_θ}[R(y)] = E_{y~π_θ}[ R(y) · ∇θ log π_θ(y) ]
```
**Why:** $∇θ E[R] = ∇θ Σ_y π_θ(y) R(y) = Σ_y R(y) ∇θ π_θ(y)$. Use the identity $∇θ π_θ = π_θ ∇θ \log π_θ$ (since $∇\log f = ∇f / f$), giving $Σ_y π_θ(y) R(y) ∇θ \log π_θ(y) = E[R · ∇θ \log π_θ]$. ∎

For a token sequence, $\log π_θ(y) = Σ_t \log π_θ(a_t\mid s_t)$, so:
```
∇θ J = E[ R(x,y) · Σ_t ∇θ log π_θ(a_t | s_t) ]
```
**Intuition:** scale the gradient of each token's log-prob by the reward. High-reward sequences → push their tokens up; low-reward → push down. This is REINFORCE.

### The variance problem → baselines & advantage
REINFORCE is **high variance** (reward magnitudes are noisy and absolute). Subtract a **baseline** $b$ that doesn't depend on the action — it reduces variance without biasing the gradient:
```
∇θ J = E[ (R - b) · ∇θ log π_θ(y) ]
```
Best baseline ≈ the expected reward, i.e. a **value function** $V(s)$. Define the **advantage**:
```
A(s,a) = Q(s,a) - V(s)    ≈  "how much better was this action than average"
```
Using advantage instead of raw reward is the core variance-reduction idea behind actor-critic / PPO. **Be ready to say:** "we subtract a baseline (the value function) so the gradient uses *advantage* — how much better than expected — which cuts variance."

---

## §3 — PPO (Proximal Policy Optimization)

PPO is the RL algorithm in classic RLHF. The problem it solves: vanilla policy gradient takes one noisy step per sample and can **destructively over-update**. PPO lets you take **multiple optimization steps on the same batch** while preventing the policy from moving too far — a cheap "trust region."

### The clipped surrogate objective (know this cold)
Define the **importance ratio** between new and old policy:
```
r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
```
PPO maximizes:
```
L_CLIP(θ) = E_t[ min( r_t(θ) · A_t ,  clip(r_t(θ), 1-ε, 1+ε) · A_t ) ]
```
**Why each piece:**
- `r_t · A_t` is the normal policy-gradient surrogate (importance-weighted advantage).
- `clip(r_t, 1-ε, 1+ε)` (ε≈0.2) caps how far the ratio can move → the policy can't change too much in one update.
- `min(...)` takes the **more pessimistic** of clipped vs unclipped → removes the incentive to push the ratio beyond the clip region. (For positive advantage it caps the upside; for negative advantage it caps the downside.)

This is a **first-order approximation of a trust region** (TRPO's KL constraint) that's far simpler to implement.

### The full PPO loss
```
L = L_CLIP  -  c1 · L_value  +  c2 · H[π_θ]
```
- **Value loss** `L_value = (V_φ(s) - R_target)²` — trains the critic that produces the baseline. (`-c1` because we minimize it.)
- **Entropy bonus** `H[π_θ]` — encourages exploration / prevents premature collapse.
- **GAE** (Generalized Advantage Estimation) is the usual way to compute $A_t$ from rewards + value estimates, trading bias vs variance via λ.

### Why PPO-RLHF is heavy
You maintain **four models**: policy, reference (frozen, for KL), reward model, and value/critic. It's compute-hungry, unstable, and hyperparameter-sensitive — which is the whole motivation for DPO (§5).

---

## §4 — Reward modeling (the RM)

RLHF optimizes a **learned** reward model trained from human **pairwise preferences** (easier to compare than to score absolutely).

### Bradley-Terry preference model (derive the loss)
Given prompt $x$ with a preferred response $y_w$ and dispreferred $y_l$, model the probability that $y_w$ is preferred as:
```
P(y_w ≻ y_l) = σ( r(x, y_w) - r(x, y_l) )
```
where $σ$ is the sigmoid and $r$ is the reward model (an LLM with a scalar head). Train by maximum likelihood → **minimize**:
```
L_RM = - E_{(x, y_w, y_l)}[ log σ( r(x, y_w) - r(x, y_l) ) ]
```
**Read it:** push the preferred response's score *above* the dispreferred one's; the sigmoid+log makes it a smooth classification loss on the score *difference*. (Note: only score *differences* matter — the RM's absolute scale is arbitrary.)

### Using the RM in PPO with a KL penalty
The reward fed to PPO is the RM score **minus a KL penalty** to the reference policy:
```
R(x, y) = r_RM(x, y)  -  β · KL( π_θ(·|x) || π_ref(·|x) )
```
**Why the KL term (critical):** the RM is imperfect. Without anchoring, PPO will **reward-hack** — drift into degenerate text (repetition, weird tokens, length exploitation) that scores high on the flawed RM but isn't actually good. The KL keeps the policy near the trusted SFT/reference distribution. β trades reward vs faithfulness.

---

## §5 — DPO (Direct Preference Optimization) — derive it

DPO's insight: you can skip the RM **and** the RL loop, optimizing preferences directly with a simple loss. The derivation is worth knowing — it's a favorite.

### Step 1 — the optimal KL-regularized policy
The RLHF objective `max_π E[r] - β·KL(π||π_ref)` has a known closed-form optimum:
```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp( r(x,y) / β )
```
(the reward-weighted reference distribution; $Z(x)$ is a normalizing partition function).

### Step 2 — invert to express reward in terms of the policy
Solve that for $r$:
```
r(x,y) = β · log( π*(y|x) / π_ref(y|x) )  +  β · log Z(x)
```
The key trick: the intractable $\log Z(x)$ depends **only on $x$**, not on $y$ — so it **cancels** in any preference *difference* $r(x,y_w) - r(x,y_l)$.

### Step 3 — plug into Bradley-Terry
Substitute this implied reward into the RM preference loss. The DPO loss becomes:
```
L_DPO = - E[ log σ( β·log(π_θ(y_w|x)/π_ref(y_w|x)) - β·log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]
```
**Read it:** it's the *same* Bradley-Terry loss, but the "reward" is now `β·log(π_θ/π_ref)` — the policy *is* its own implicit reward model. You directly increase the log-prob margin of preferred over dispreferred responses (relative to the reference), no RM and no sampling/RL loop.

### DPO vs PPO (the practical comparison)
- **DPO:** offline (fixed preference dataset), 2 models (policy + frozen ref), stable, cheap, simple. Bound to the dataset's coverage; can't learn from its *own current* mistakes.
- **PPO:** online/on-policy (samples fresh, learns from current failures), reusable RM, more control — but 4 models, unstable, expensive.
- **One-liner:** "DPO reparameterizes RLHF so the policy is its own reward model — same Bradley-Terry objective, no RL loop. PPO wins when you can iterate on fresh on-policy data."

---

## §6 — The modern landscape (name these)

- **RLAIF / Constitutional AI:** replace human preference labels with **AI-generated** ones guided by principles. Scales labeling; inherits the judge's biases.
- **Rejection sampling / best-of-n:** sample n responses, keep the best by an RM, SFT on those. Simple, strong baseline; bootstraps better SFT data; bounded by the base model's best samples.
- **GRPO (Group Relative Policy Optimization):** PPO variant used in recent reasoning models (e.g. DeepSeek). **Drops the value/critic model** — instead samples a *group* of responses per prompt and uses the **group's mean reward as the baseline**, normalizing advantages within the group. Cheaper (no critic) and well-suited to verifiable rewards. Worth knowing as the current reasoning-RL workhorse.
- **Verifiable-reward RL (RLVR):** for math/code, reward = an **automatic checker** (tests pass / answer correct). Clean, ungameable signal → strong RL; behind recent reasoning gains. No RM needed.
- **Process vs outcome supervision:** reward each reasoning *step* (process, denser, better reasoning, costlier to label) vs only the final answer (outcome).

---

## §7 — Failure modes & knobs (bring these up unprompted)

- **Reward hacking / over-optimization:** policy exploits RM flaws. Mitigate: KL penalty, RM ensembles, early stopping, capping optimization.
- **Alignment tax:** post-training can slightly reduce raw capability — a real tradeoff.
- **Over/under-refusal:** too much safety tuning → refuses benign requests; too little → unsafe. Eval'd explicitly.
- **Sycophancy:** RLHF can teach the model to tell raters what they want to hear (rated higher) rather than what's true.
- **Length bias:** RMs often prefer longer answers; controlled for explicitly.

---

## §8 — RL + Agents (the Fei/Weiyue bridge)

Agents add a **multi-step environment** — the reward comes from *task success*, often after many tool calls.
- **Environment feedback as reward:** tool results, test pass/fail, task completion → a (often sparse, often verifiable) reward. This is RL with a real environment, not just a preference RM.
- **Long-horizon credit assignment:** which of the 20 steps caused success/failure? Hard, sparse-reward problem. Process supervision and step-level rewards help.
- **On-policy iteration matters more for agents:** the agent must learn from its *own* rollouts (its mistakes are distribution-specific), favoring online RL over purely offline methods.
- **Eval ties in:** agent success rate, steps, cost, unsafe actions, prompt-injection robustness (see Day 4 §3).

---

## §9 — Self-test

1. **Derive the policy-gradient estimator.** → log-derivative trick: `∇θ E[R] = E[R · ∇θ log π_θ]`; for sequences sum token log-probs.
2. **Why subtract a baseline / use advantage?** → variance reduction without bias; advantage = how much better than expected (Q − V).
3. **Write the PPO clipped objective and explain the clip + min.** → `min(r·A, clip(r,1±ε)·A)`; clip bounds the policy step (trust region), min takes the pessimistic bound so there's no incentive to exceed the clip.
4. **What four models does PPO-RLHF need?** → policy, reference (KL), reward model, value/critic.
5. **Write the reward-model loss.** → `-log σ(r(x,y_w) − r(x,y_l))` (Bradley-Terry); only score differences matter.
6. **Why the KL penalty in RLHF?** → RM is imperfect; KL anchors the policy to the reference to prevent reward hacking.
7. **Derive the DPO loss.** → optimal KL-reg policy `π* ∝ π_ref·exp(r/β)` → invert to `r = β log(π/π_ref) + β log Z`; logZ cancels in differences → plug into Bradley-Terry → `-log σ(β log(π_θ(y_w)/π_ref(y_w)) − β log(π_θ(y_l)/π_ref(y_l)))`.
8. **DPO vs PPO — when each?** → DPO offline/simple/stable, dataset-bound; PPO online/on-policy, heavier, better when you can iterate.
9. **What is GRPO and why is it used for reasoning?** → PPO without a critic; group-sampled mean reward as baseline, normalized advantages; cheap, pairs well with verifiable rewards.
10. **When can you skip the reward model entirely?** → verifiable domains (math/code) with an automatic checker (RLVR); or DPO (implicit reward).

---

## Cheat sheet

- LLM = policy; maximize `E[R(x,y)]`; R non-differentiable → policy gradient.
- **REINFORCE:** `∇θ J = E[R · ∇θ log π_θ(y)]`. High variance → subtract baseline → **advantage** `A = Q − V`.
- **PPO:** `L_CLIP = E[min(r·A, clip(r,1−ε,1+ε)·A)]`; + value loss + entropy; 4 models; clip = cheap trust region.
- **Reward model:** Bradley-Terry `-log σ(r(y_w) − r(y_l))`; PPO reward = `r_RM − β·KL(π||π_ref)` (KL stops reward hacking).
- **DPO:** policy is its own implicit reward; `-log σ(β log(π_θ(y_w)/π_ref(y_w)) − β log(π_θ(y_l)/π_ref(y_l)))`. Offline, 2 models, stable.
- **GRPO:** PPO minus critic; group-mean baseline. **RLVR:** checker reward for math/code.
- **Risks:** reward hacking (→KL), alignment tax, over/under-refusal, sycophancy, length bias.
- **Agents:** reward = task success (sparse/verifiable); long-horizon credit assignment; on-policy matters.
