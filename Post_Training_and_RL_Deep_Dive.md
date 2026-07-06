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

### Why the trick? Both sides are integrals — only one is an *expectation*
Before the trick: $∇θ J = ∫ ∇θ P(τ|θ)·R(τ)\,dτ$. After: $∫ P(τ|θ)·∇θ\log P(τ|θ)·R(τ)\,dτ$. **Same integral, same value** — but only the second has the shape *density × function*, i.e. an expectation $E_{τ\sim π_θ}[·]$. That's the whole point:

- $∇θ P(τ|θ)$ is **not a probability distribution** (can be negative, doesn't sum to 1) — there is nothing to sample from, so the first form can only be evaluated by enumerating the exponential trajectory space. Intractable.
- The trick moves $P(τ|θ)$ from being *differentiated* to being the *density out front* → Monte Carlo: sample rollouts from the policy, average $∇θ\log π·R$. Unbiased for any N (LLN).
- Bonus: $\log$ turns the trajectory product $p(s_0)\prod_t π_θ(a_t|s_t)p(s_{t+1}|s_t,a_t)$ into a **sum**, and the environment dynamics terms have no θ → they **vanish**. This cancellation is *why REINFORCE is model-free*.

**One-liner:** "the log-derivative trick converts an un-samplable integral into a samplable expectation — and cancels the dynamics, making the method model-free."

**Naming:** $J(θ) = E_{τ\sim π_θ}[R(τ)]$ is the **expected return / performance objective** (the letter J is old control-theory convention for a cost functional). $∇θ J$ is the **policy gradient**; the identity is the **Policy Gradient Theorem**; REINFORCE is its *vanilla* estimator. REINFORCE ⊂ policy-gradient methods: all members (A2C, TRPO, PPO, GRPO…) estimate the **same** $∇θ J$ and differ only in the weight $Ψ$ multiplying $∇θ\log π$: $Ψ = R(τ)$ (REINFORCE) → $R−b(s)$ (baseline) → $A(s,a)$ (actor-critic) → clipped-ratio·$A$ (PPO).

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

### Why the baseline is unbiased (derive this — 3 lines)
Rests on one fact: **the expected score is zero.** For fixed $s$:
```
E_{a~π}[∇θ log π(a|s)] = Σ_a π·(∇θπ/π) = ∇θ Σ_a π(a|s) = ∇θ 1 = 0
```
So the extra term $E[∇θ\log π(a|s)·b(s)] = E_s[\,b(s)·0\,] = 0$ — **any** function of the state alone adds exactly zero to the expected gradient. Conditions & consequences:
- $b$ must not take the action **as an input** ($b(s_t)$, not $b(s_t,a_t)$) — else it can't be pulled out of $E_a$ and you get bias.
- "Action-independent" is about **input signature, not training provenance**: a value head sharing the policy's transformer trunk, trained on action-generated data, is still legal — at use time $V_φ(s_t)$ is computed from the prefix only, before $a_t$ is sampled, so it's constant w.r.t. $E_{a_t}$.
- **Accuracy of $b$ affects only variance, never bias.** $b(s)=42$ is unbiased; $b≈V(s)$ is the variance-minimizing choice.
- Why it helps: recenters returns around zero — the signal becomes "better or worse than average *for this state*" instead of huge all-positive numbers.

### Reward-to-go = Q, baseline = V, difference = advantage
Combine the two variance reductions and the value-function names fall out:
- **Reward-to-go** $\hat R_t = Σ_{t'≥t} γ^{t'−t} r_{t'}$ is a single-sample unbiased MC estimate of $Q^π(s_t,a_t)$ — conditioned on state **and the action taken**. The gradient weight *must* carry the action's info, so the signal is Q-like.
- **Baseline** must be action-independent → the most informative legal choice is the action-average of Q, which **is** $V^π(s) = E_{a\sim π}[Q^π(s,a)]$.
- $\hat A_t = \hat R_t − V_φ(s_t) ≈ Q − V = A$. The pairing is forced, not a convention.

### Why learn V (not A or Q directly)? — the three-part answer
1. **Only V has an unbiased regression target.** Training pairs are $(s_t,\; \hat R_t)$ — the return actually observed from $s_t$ in that rollout. Each target is a sample of $Q(s_t,a_t)$ for the sampled action, but across visits actions are drawn from π, so targets scatter around $E_a[Q] = V(s)$ — and **MSE regression converges to the mean of its targets**. The action-randomness becomes label noise; $V_φ$ learns the action-average without ever branching. There is **no observable sample of A**: the advantage is a difference of expectations, nothing you collect *is* one.
2. **A learned action-dependent weight breaks unbiasedness.** Plug a trained $A_ψ(s,a)$ into the gradient and its approximation errors correlate with actions → bias (the illegal $b(s,a)$ case). In $\hat R_t − V_φ(s_t)$, the action-dependence comes entirely from the *sampled* return (unbiased) and the *learned* part is action-independent (safe).
3. **Identifiability + cost.** $Q = V + A$ is only identified under the constraint $E_{a\sim π}[A(s,·)] = 0$ — enforcing it *is* computing V (dueling DQN has to bolt on exactly this mean-subtraction). And a direct A/Q target would require rolling out **every action from the same state**: $O(|A|)$ full rollouts per state (50k+ tokens for an LLM), needing state resets — exponential over the horizon. The standard estimator replaces "branch over all actions at this state" with "regress to the mean over many states" — the value net **amortizes the exponential branching**.

**GRPO is the brute-force version made affordable:** for LLMs the state (prompt) is resettable for free and the whole response is one action, so sample a group of G responses and use $\hat A_i = (r_i − \text{mean}_G)/\text{std}_G$ — the group mean is an *empirical* V(s) computed by actual branching (G ≈ 8–64, not $|A|^T$). When branching is cheap and one-step, the critic becomes optional; RLOO is the leave-one-out variant of the same idea.

### Training the value function
Supervised MSE against a return target, jointly with the policy: $L(φ) = E_t[(V_φ(s_t) − \hat R_t^{target})^2]$. The target choice is the bias/variance dial:

| Target | Formula | Bias | Variance |
| --- | --- | --- | --- |
| Monte Carlo | $Σ_{t'≥t} γ^{t'−t} r_{t'}$ | none | high |
| TD(0) bootstrap | $r_t + γV_φ(s_{t+1})$ | some (self-referential) | low |
| GAE / TD(λ) | λ-blend of n-step returns | tunable | tunable |

In PPO/RLHF: **GAE** computes $\hat A_t$ and the value target $\hat R_t = \hat A_t + V_φ(s_t)$ in one sweep. Practical notes: targets are computed once per batch and treated as constants (stop-gradient — bootstrap targets otherwise chase their own tail); the critic is usually a **linear value head on the shared trunk** (total loss $L_{policy} − c_1 L_{value} + c_2 H[π]$; watch for the two objectives fighting over features); PPO often **clips the value update** too.

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

### The landscape map (two axes)
Plot every method on: **x = on-policy-ness** (fresh rollouts → replay → fixed offline dataset) and **y = machinery** (reward model + critic → direct/critic-free). Three clusters:

| Cluster | Methods | Character |
| --- | --- | --- |
| On-policy policy gradient | PPO/RLHF, TRPO, A2C (with critic); GRPO, RLOO, REINFORCE (critic-free) | sample-hungry, directly optimizes reward; samples go stale when θ moves |
| Off-policy actor-critic | SAC, TD3/DDPG, DQN | replay buffer reuses old data — possible *because* a bootstrapped Q-critic can evaluate stale transitions (off-policy ⟹ you need a critic) |
| Offline preference (*PO) | DPO, IPO, KTO, SimPO, ORPO | zero rollouts, no RM, no critic — closed-form supervised loss on preference pairs |

**The correlation to remember:** top-left → bottom-right = shedding machinery *and* on-policy sampling together. RLHF-PPO (RM + critic + rollouts) → GRPO (drop the critic, keep rollouts) → DPO/*PO (drop the RM *and* the rollouts). DPO and GRPO are diagonal opposites — both "simpler than PPO", in opposite directions; that's the current frontier debate.

***PO family differences (one line each):** IPO — regularizes DPO against overfitting deterministic preferences; KTO — unpaired good/bad labels (prospect-theory utility), no pairs needed; ORPO — drops the reference model, folds preference odds into the SFT loss (single stage); SimPO — also reference-free, length-normalized average log-prob as the implicit reward. Trend within the family: shed even more models (the same gradient that separates GRPO from PPO).

### The orthogonal axis: model-based / world-model methods
Everything above is **model-free** — recall the dynamics $p(s'|s,a)$ *cancelled out* of $∇θ\log P(τ|θ)$; that cancellation is the definition. World-model methods (Dreamer, PlaNet, TD-MPC, JEPA-style planners) learn $\hat p(s'|s,a)$ and roll out **in imagination**. Planning there is often **zeroth-order reward-weighted sampling** (CEM / MPPI): sample action sequences, reweight by $\exp(R/η)$, refit, repeat — a fixed-point iteration that concentrates on high-return trajectories.
- **Deep connection (RL-as-inference):** policy gradient and reward-weighted refitting optimize the *same* $J = E[R]$ — first-order (differentiate) vs zeroth-order (reweight samples); the reward-weighted update is the EM view whose gradient recovers the PG form.
- **Hybrids:** Dreamer/TD-MPC = learn a world model, run actor-critic PG *inside* imagined rollouts (model gives cheap data, PG gives an amortized fast policy). Pure CEM planning = no policy at all, optimize actions online at inference.

---

## §7 — Failure modes & knobs (bring these up unprompted)

- **Reward hacking / over-optimization:** policy exploits RM flaws. Mitigate: KL penalty, RM ensembles, early stopping, capping optimization.
- **Alignment tax:** post-training can slightly reduce raw capability — a real tradeoff.
- **Over/under-refusal:** too much safety tuning → refuses benign requests; too little → unsafe. Eval'd explicitly.
- **Sycophancy:** RLHF can teach the model to tell raters what they want to hear (rated higher) rather than what's true.
- **Length bias:** RMs often prefer longer answers; controlled for explicitly.

---

## §8 — RL + Agents (the the interviewer/the interviewer bridge)

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
