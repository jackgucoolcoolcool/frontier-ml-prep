# Scaled RL Data for Frontier LLMs — Current Practice → Self-Improving Designs

> **Key idea:** the bottleneck for frontier post-training has moved from "tokens" to **tasks × environments × reward signals**. Pretraining consumed the internet; RL consumes *verifiable experience*, and nobody has an internet of that. Whoever builds the machine that manufactures reliable RL experience at scale — without humans in the inner loop — wins the next scaling regime. This doc: what the field does today (mid-2026), why it saturates, and concrete designs for self-improving data engines in three domains: **code, knowledge work, long-context work**.

**Framing for interviews:** RL is a *compute-to-data converter*. Pretraining converts data → capability; RL converts compute + a reward signal → data (rollouts) → capability. So "scaling RL data" really decomposes into three multiplicable resources: (1) **task supply** (unique, in-distribution-of-use, right difficulty), (2) **reward fidelity** (how often the signal points the right way, and how hard it is to hack), (3) **rollout throughput** (env speed, async infra, staleness tolerance). Any design should say which of the three it scales and what it pays.

---

## 1. Current practice (as of mid-2026)

### 1.1 Code — the most mature domain

- **RLVR with execution oracles.** Unit tests, compilation, runtime checks as reward. This is the canonical case because the **generator–verifier gap** is maximal: writing the patch is hard, running the tests is cheap and objective.
- **Task mining from git history.** Issue↔PR pairs (SWE-bench recipe) scaled up: take a repo, check out the pre-fix commit, hide the fix, use the repo's own tests (plus fail-to-pass filtering) as the oracle. Labs run this over millions of repos with heavy filtering for test quality, flakiness, and environment reproducibility.
- **Agentic RL in real harnesses.** Training happens *inside* the deployment scaffold (terminal, editor tools, browser) — multi-turn, multi-task, async frameworks like [AgentRL](https://arxiv.org/pdf/2510.04206); co-evolving the harness itself ([EvoTrainer](https://arxiv.org/pdf/2606.03108)). Hermetic containerized sandboxes; thousands of concurrent environments; the env-step, not the GPU, is often the wall-clock bottleneck.
- **Synthetic task generation.** Procedural bug injection (mutate a passing repo, ask the model to fix it), dependency-bump tasks, spec-to-code with generated test suites. Adaptive-difficulty procedural generators ([RLVE](https://arxiv.org/pdf/2511.07317)) keep pass-rates in the learnable band.

### 1.2 Knowledge work — reward is the bottleneck, not tasks

- **Rubric-based rewards.** Per-prompt rubrics (checklists of what a good answer must contain / avoid), initially expert-written, now mostly model-drafted and human-audited. LLM-judge grades against the rubric; reward = weighted rubric satisfaction. This is the workhorse for non-verifiable domains (analysis, writing, advice, research).
- **Deep-research agents.** Multi-step search+synthesis tasks with rubric or reference-answer grading ([LiteResearcher](https://arxiv.org/html/2604.17931v2)-style pipelines); reward often decomposes into *retrieval hit* (verifiable: did you find the source?) + *synthesis quality* (rubric).
- **Preference RL persists for style/safety** but is now a thin layer on top of rubric/verifiable RL, not the main capability driver.
- **Weakness:** judges saturate and get hacked; rubric coverage is spiky; task distribution drifts from what users actually ask.

### 1.3 Long-context work — the least developed

- **Synthetic multi-hop retrieval** over real corpora (haystack tasks are dead as training signal — models saturate them; they survive only as smoke tests).
- **Long-horizon agent trajectories** are the real long-context data: a 100-turn SWE session *is* a long-context task. Context management itself (compaction, memory files, what to re-read) is starting to be rewarded rather than hard-coded — see my notes in **Agentic_RL_Compaction_and_Staleness**.
- **Reward problem:** outcome reward over a 500k-token trajectory gives ~1 bit for hours of compute — the credit-assignment problem is the domain-defining difficulty.

### 1.4 Cross-cutting infra reality

- **Async, off-policy-tolerant training** (importance-weighted corrections, staleness caps) because environments are slow and heterogeneous.
- **Difficulty curation is automatic:** GRPO-style group statistics double as a free difficulty probe — if all G rollouts pass or all fail, advantage is zero and the sample is wasted; pipelines resample/filter to keep pass-rate near the max-gradient band (~20–80%).
- **Task synthesis from unverifiable text:** e.g. [Golden Goose](https://arxiv.org/pdf/2601.22975) — convert arbitrary internet text into verifiable tasks (mask a derivable fact, ask for it, check the mask). Early but points at the key trick of §3.

---

## 2. Why current practice saturates — the failure catalogue

| Failure | Mechanism | Symptom |
|---|---|---|
| **Task exhaustion** | mined tasks (git issues, exam questions) are finite; procedural generators are narrow | pass@1 climbs, real-world transfer flattens |
| **Verifier saturation** | policy exceeds judge; judge noise becomes the ceiling | reward ↑, human evals flat or ↓ |
| **Reward hacking** | any gap between proxy and intent gets optimized into | test-deletion, rubric keyword stuffing, sycophantic judged answers |
| **Distribution narrowing** | RL sharpens the policy onto reward modes; entropy collapse | diversity ↓, pass@k for large k ↓ even as pass@1 ↑ |
| **Self-training drift** | model-generated tasks + model-graded rewards → closed loop amplifies model's own blind spots | generational "model collapse" in task space |
| **Credit assignment at horizon** | outcome-only reward over 10⁵–10⁶ tokens | slow, high-variance learning on exactly the tasks that matter most |

> **In practice:** every scaled RL pipeline is a race between capability gain and hack discovery. The operational metric that matters is not reward but **audited-human-agreement per unit of reward gained**, tracked continuously.

---

## 3. The organizing principle: manufacture verifiability

Almost every promising direction is an instance of one trick: **find a transformation that makes the reward asymmetrically cheap relative to the task.** Four generic asymmetries:

1. **Execution asymmetry** (code): running is cheaper than writing.
2. **Hiding asymmetry** (knowledge/long-context): *remove* something derivable from a trusted artifact; recovering it is hard, checking recovery is `==`. Backtranslation-of-verifiability.
3. **Generation–verification asymmetry** (proofs, structured claims): checking a proof/citation/derivation is easier than producing it — reward = verifier pass.
4. **Consistency asymmetry** (self-play): two independent paths must agree (majority vote, cross-examination, forward-vs-backward solving); agreement is checkable without ground truth.

Everything in §4 composes these with a **task generator** and an **auto-curriculum**.

---

## 4. Future designs — self-improving data engines

### 4.1 The three-player self-play kernel (the core proposal)

Roles (can be one model, role-prompted, or separate checkpoints):

- **Proposer** generates tasks *with a verification artifact* (test suite, hidden answer, rubric, proof obligation).
- **Solver** (the policy we care about) attempts the task.
- **Verifier** applies the artifact; where the artifact is a rubric, the Verifier is a judge model.

Reward design is the whole game:

- **Solver:** standard RLVR on Verifier output.
- **Proposer:** rewarded for tasks that are (a) *valid* (artifact passes sanity/spot-check), (b) *at the frontier* — solver group pass-rate near 50% (the GRPO group statistic is a free, differentiable-through-sampling difficulty signal), (c) *novel* (embedding-distance penalty vs. task bank). This is the auto-curriculum: difficulty tracks the policy by construction ([R-Zero / SeRL](https://arxiv.org/abs/2505.20347) lineage, [π-Play](https://arxiv.org/pdf/2604.14054)).
- **Verifier:** the weak link. Improve it with **asymmetric supervision**: train it on (i) hacks discovered by a red-team policy explicitly rewarded for *fooling* the Verifier while failing human audit (adversarial mining of reward hacks as data), (ii) sparse expert audits used as meta-labels.

**Why this can self-improve without drifting:** the loop injects grounding at two points — the verification *artifact* is checked mechanically where possible, and audits anchor the Verifier. The design principle: **self-play scales the task axis; grounding must scale the reward axis** — never let both come from the unaided model.

### 4.2 Code: "world of broken repos"

- **Injector/Fixer self-play.** Injector edits a green repo to make tests fail *subtly* (rewarded when the Fixer's pass-rate is mid-band and the diff is small/semantically sneaky — not `rm -rf src`); Fixer is the policy. Oracle = the repo's own tests + differential execution vs. the pre-injection binary. Unlimited tasks from a finite repo corpus; difficulty auto-tracks.
- **Spec-first generation.** Proposer writes a spec + property-based tests (Hypothesis-style) + a reference solution; validity check = reference passes tests, tests kill mutants of the reference. Solver sees only the spec. This filters out under-specified tasks *mechanically* — mutation-kill-rate is a computable test-quality score.
- **Compositional environments.** Treat verified envs as composable bricks (chain: fix the build → then the test → then perf-regression; cf. [recursive env composition](https://arxiv.org/pdf/2606.12373)) — horizon and difficulty scale combinatorially from a fixed brick set.
- **Deployment flywheel.** Real agent sessions (with consent) → hindsight-relabel into tasks: the session's *final* state defines the goal; replay from initial state as a fresh env. Failures become tasks with a known-hard region; successes become distillation targets. This is the one task source whose distribution *provably matches use*.

### 4.3 Knowledge work: rubric ecosystems + hide-and-recover

- **Backtranslated verifiable knowledge tasks.** From trusted corpora (papers, filings, medical guidelines, internal docs): delete a derivable conclusion/number/step; task = recover it with sources; reward = exact/entailment match on the hidden span + citation-span checker (the citation must actually contain the claim — a mechanical string/entailment check). Generalizes [Golden Goose](https://arxiv.org/pdf/2601.22975); scales with the *pretraining corpus*, which is the only thing we have internet-scale amounts of.
- **Co-evolving rubrics.** Rubrics are a *population*, not a fixed spec: mutate/recombine rubrics; select on agreement-with-audit and hack-resistance ([self-evolving rubrics](https://arxiv.org/pdf/2602.10885), [EvoLM](https://arxiv.org/pdf/2605.03871)). Policy trains against the current rubric ensemble (ensembling resists single-rubric keyword hacks); rubric fitness is measured against the sparse human-audit stream. Two-timescale co-evolution: fast policy, slow rubric.
- **Cross-examination rewards (consistency asymmetry).** For claims with no oracle: a second instance tries to *refute* the answer with independent search; reward for the answerer scales with surviving adversarial scrutiny; the refuter is rewarded for successful refutations that a judge+sources confirm. Debate-as-reward, but scoped to *factual* disputes where sources adjudicate — avoids the "persuasive ≠ true" failure of open-ended debate.

### 4.4 Long-context: make context management the rewarded skill

- **Provenance-verified multi-hop synthesis.** Proposer walks a real corpus building a k-hop chain (entity → doc → fact → doc...), records the trace; task = answer with citations; verifier checks each hop's span. The *trace* makes an unverifiable-looking synthesis task mechanically checkable. Curriculum = hops × distractor volume × context length.
- **Compaction-aware RL.** Long-horizon agent task where context overflows *by design*; the policy's memory/compaction actions are part of the trajectory. Dense shaping: at random checkpoints, ask probe questions answerable only from information the policy should have retained; probe accuracy is a cheap intermediate reward for "did you keep the right things." (Direct extension of the compaction/staleness notes.)
- **Hindsight horizon-splitting for credit assignment.** Slice long trajectories at compaction/summary boundaries; treat the summary as the state; train value estimates at boundary granularity (options/hierarchical view). Turns the 1-bit-per-500k-token problem into per-segment learning signal without a learned dense PRM.

### 4.5 The outer loop: iterated amplification-and-distillation

AlphaZero's recipe, transplanted: **spend compute to make a better teacher, distill, repeat.**

1. **Amplify:** best current policy + search (best-of-n with verifier, tool use, multi-agent cross-examination, long deliberation) produces solutions *better than the policy's own 1-sample distribution*.
2. **Verify/filter:** keep only artifact-verified or ensemble-agreed trajectories.
3. **Distill:** SFT/RL the base policy toward amplified outputs (RL, not just SFT, to preserve exploration — distill the *achievable reward*, not the token sequence).
4. **Re-generate tasks** at the new frontier (Proposer re-tunes difficulty) and loop.

The **generation-over-generation dashboard** (what I'd actually watch): (a) generator–verifier gap trend; (b) audited human agreement with Verifier at fixed sample budget; (c) task-embedding diversity vs. gen-0; (d) pass@k at *large* k (entropy-collapse canary); (e) transfer to a frozen, never-trained-on holdout of real user tasks. Self-improvement is real iff (e) moves; everything else can be gamed by the loop itself.

---

## 5. Research questions I'd propose (interview-ready)

1. **Scaling laws for RL data:** capability as a function of (unique tasks) × (rollouts per task) × (reward fidelity) — is there a data-vs-compute exchange rate like Chinchilla's? Hypothesis: unique-task count dominates once rollouts/task exceed a small number; reward fidelity enters as an effective-data multiplier.
2. **Where does self-play stop?** Characterize the drift rate of closed-loop task generation vs. the audit budget needed to pin it. Prediction: audit demand grows sub-linearly if verification artifacts are mechanical, linearly+ if judge-based.
3. **Hack-mining as an alignment asset:** train the red-team policy *deliberately*, publish the hack corpus internally, and measure how verifier robustness scales with hacks discovered — turning reward hacking from a tax into a data source.
4. **Difficulty targeting:** is 50% pass-rate actually optimal, or does the max-learning-signal band shift with group size G and KL budget? (Connects to the GRPO variance analysis in Post_Training_and_RL_Deep_Dive.)

## 6. Soundbites

- "Pretraining ate the internet; RL has to *cook its own food* — the research problem is the kitchen, not the stove."
- "Every scalable reward is a manufactured asymmetry: execution, hiding, verification, or consistency."
- "Self-play scales tasks; grounding must scale rewards. A loop that self-generates both is a model-collapse machine."
- "The only unfakeable metric of a self-improvement loop is transfer to frozen, real, held-out user tasks."

---

## Sources

- [RLVE: adaptive verifiable environments](https://arxiv.org/pdf/2511.07317) · [AgentRL](https://arxiv.org/pdf/2510.04206) · [EvoTrainer](https://arxiv.org/pdf/2606.03108) · [Verifiable envs as LEGO bricks](https://arxiv.org/pdf/2606.12373) · [Golden Goose](https://arxiv.org/pdf/2601.22975) · [LiteResearcher](https://arxiv.org/html/2604.17931v2) · [Agentic RL survey](https://arxiv.org/pdf/2509.02547) · [SeRL](https://arxiv.org/abs/2505.20347) · [π-Play](https://arxiv.org/pdf/2604.14054) · [Self-evolving rubrics](https://arxiv.org/pdf/2602.10885) · [EvoLM: co-evolved rubrics](https://arxiv.org/pdf/2605.03871) · [ARE: scaling agent environments](https://arxiv.org/pdf/2509.17158)

*Written 2026-07-07. Companion docs: Post_Training_and_RL_Deep_Dive, Agentic_RL_Compaction_and_Staleness, Self_Improving_VLMs.*
