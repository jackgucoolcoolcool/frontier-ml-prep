# F3 — The full RL loop, staged like a real CoderPad round.
# 3-hour deep session: rollout -> reward-to-go -> REINFORCE gradient -> GAE
# -> PPO clipped objective -> GRPO step for LLMs. Pure NumPy throughout
# (the grading harness imports numpy only).
#
# Reading companion: RL_Coding_Interview_Pack.html (sections 2-7).

RL_FULL_LOOP_SESSION = {
  "id": "f3",
  "interviewer": "F",
  "persona": "Interviewer F — post-training/RL/agents. Wants the loop structure stated up front (rollout → score → loss → update → discard), detach discipline, and the bias/variance story told unprompted.",
  "title": "The full RL loop (REINFORCE → GAE → PPO → GRPO)",
  "minutes": 180,
  "intro": "Today is one long build: we'll construct a working policy-gradient trainer from nothing, then upgrade it piece by piece until it's PPO, then adapt it to the LLM setting (GRPO). Pure NumPy. At every stage, tell me what's a constant and what gradients flow through — that discipline is most of what I'm grading. Suggested reading before/while you code: RL_Coding_Interview_Pack.html.",
  "closing": "That's the whole arc — you just built REINFORCE, PPO, and GRPO from scratch. If you can reproduce today's six functions cold, there is no RL coding round that can surprise you.",
  "stages": [

    # ------------------------------------------------------------------
    {
      "id": "f3s1",
      "title": "Rollout: sampling actions from a softmax policy",
      "prompt": "We start with the environment below (a 5-state chain: walk right to state 4 for +1, small per-step penalty, 20-step limit) and a **tabular softmax policy**: parameters `W` of shape `(5, 2)`, where `W[s]` are the logits for the 2 actions in state `s`.\n\nImplement:\n```\ndef action_probs(W, s):        # -> (2,) softmax over W[s]\ndef sample_action(W, s, rng):  # -> int action drawn from action_probs\ndef collect_episode(env, W, rng):  # -> (states, actions, rewards) lists\n```\n`collect_episode` runs ONE full episode with the current policy: reset, then step until `done`, recording the state you acted in, the action, and the reward received.\n\nUse `rng` (a `np.random.Generator`) for all randomness. Talk me through why we *sample* rather than take the argmax.",
      "starter": "import numpy as np\n\nclass ChainEnv:\n    \"\"\"5-state chain. Actions: 0=left, 1=right. Start at 0.\n    Reward: +1.0 on reaching state 4 (terminal), else -0.01 per step.\n    Episode also terminates after 20 steps.\"\"\"\n    def reset(self):\n        self.s, self.t = 0, 0\n        return self.s\n    def step(self, a):\n        self.s = max(0, self.s - 1) if a == 0 else min(4, self.s + 1)\n        self.t += 1\n        done = (self.s == 4) or (self.t >= 20)\n        r = 1.0 if self.s == 4 else -0.01\n        return self.s, r, done\n\ndef action_probs(W, s):\n    pass\n\ndef sample_action(W, s, rng):\n    pass\n\ndef collect_episode(env, W, rng):\n    # returns (states, actions, rewards) for one episode\n    pass\n",
      "hints": [
        "action_probs: stable softmax of the 2-vector W[s] (subtract max before exp). sample_action: rng.choice(2, p=probs) is fine. collect_episode: s = env.reset(); loop: record s, sample a, step, record a and r, stop on done.",
        "Common structural bug: recording the state AFTER stepping. You must record the state you were in when you CHOSE the action — the (s, a, r) triple is (state acted in, action taken, reward received for that transition)."
      ],
      "probe": {
        "q": "Why sample from the policy during training instead of taking the argmax action?",
        "a": "Two reasons. (1) Exploration: argmax is deterministic, so the agent only ever sees one trajectory per policy and can never discover better actions — the gradient starves. (2) The policy-gradient theorem is an expectation over the policy's own distribution: E_{a~π}[∇log π · Ψ]. The estimator is only unbiased if actions are drawn from π. Argmax is for evaluation, never for on-policy training."
      },
      "tests": """
class _Chain:
    def reset(self):
        self.s, self.t = 0, 0
        return self.s
    def step(self, a):
        self.s = max(0, self.s - 1) if a == 0 else min(4, self.s + 1)
        self.t += 1
        done = (self.s == 4) or (self.t >= 20)
        r = 1.0 if self.s == 4 else -0.01
        return self.s, r, done

def _t_probs():
    W = np.array([[0.0, 0.0]] * 5)
    p = action_probs(W, 0)
    assert p is not None, "action_probs returned None"
    p = np.asarray(p, dtype=float)
    assert p.shape == (2,), f"expected shape (2,), got {p.shape}"
    assert np.allclose(p, [0.5, 0.5]), f"uniform logits must give [0.5, 0.5], got {p}"
    W2 = np.zeros((5, 2)); W2[3] = [1.0, 3.0]
    p2 = np.asarray(action_probs(W2, 3), dtype=float)
    e = np.exp([1.0, 3.0]); e = e / e.sum()
    assert np.allclose(p2, e, atol=1e-6), "softmax values wrong"
_check("action_probs is a correct softmax over W[s]", _t_probs)

def _t_stable():
    W = np.zeros((5, 2)); W[0] = [800.0, 802.0]
    p = np.asarray(action_probs(W, 0), dtype=float)
    assert np.isfinite(p).all(), "overflow on large logits -- subtract the max before exp"
_check("softmax numerically stable on large logits", _t_stable)

def _t_sampling_dist():
    rng = np.random.default_rng(0)
    W = np.zeros((5, 2)); W[2] = [0.0, np.log(3.0)]   # probs [0.25, 0.75]
    n = 4000
    acts = [sample_action(W, 2, rng) for _ in range(n)]
    freq = np.mean([a == 1 for a in acts])
    assert abs(freq - 0.75) < 0.04, f"empirical P(a=1)={freq:.3f}, expected ~0.75 -- are you sampling from the softmax (not argmax)?"
_check("sample_action matches the policy distribution (not argmax)", _t_sampling_dist)

def _t_episode():
    rng = np.random.default_rng(1)
    W = np.zeros((5, 2)); W[:, 1] = 5.0               # strongly prefer 'right'
    env = _Chain()
    S, A, R = collect_episode(env, W, rng)
    assert len(S) == len(A) == len(R), f"lengths differ: {len(S)}, {len(A)}, {len(R)}"
    assert S[0] == 0, f"first recorded state must be the reset state 0, got {S[0]} -- record the state you ACT in, before stepping"
    assert len(S) >= 4, "a right-preferring policy needs >= 4 steps to reach state 4"
    assert abs(R[-1] - 1.0) < 1e-9, f"last reward should be +1.0 (reached goal), got {R[-1]}"
    assert all(abs(r + 0.01) < 1e-9 for r in R[:-1]), "intermediate rewards should be -0.01"
_check("collect_episode records (state-acted-in, action, reward) correctly", _t_episode)
""",
      "solution": "def action_probs(W, s):\n    z = W[s] - W[s].max()          # stability\n    e = np.exp(z)\n    return e / e.sum()\n\ndef sample_action(W, s, rng):\n    return int(rng.choice(2, p=action_probs(W, s)))\n\ndef collect_episode(env, W, rng):\n    states, actions, rewards = [], [], []\n    s = env.reset()\n    done = False\n    while not done:\n        a = sample_action(W, s, rng)\n        states.append(s)            # the state we acted in\n        actions.append(a)\n        s, r, done = env.step(a)\n        rewards.append(r)\n    return states, actions, rewards"
    },

    # ------------------------------------------------------------------
    {
      "id": "f3s2",
      "title": "Score: reward-to-go + whitening",
      "prompt": "Now the SCORE phase. Implement:\n```\ndef reward_to_go(rewards, gamma):   # list -> np.array of G_t\ndef whiten(x, eps=1e-8):            # (x - mean) / (std + eps)\n```\n`G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...` — the discounted sum from t onward.\n\nTwo things I want stated while you write: why reward-to-go instead of the full episode return at every step, and why whitening doesn't bias the gradient.",
      "starter": None,
      "hints": [
        "Compute it BACKWARDS in one pass: G = 0; for r in reversed(rewards): G = r + gamma*G; prepend (or append then reverse). Forward computation is O(T^2) and a red flag.",
        "Whitening: the mean subtraction is a baseline (any constant is action-independent -> zero bias, from E[∇log π] = 0); the std division just rescales the step size."
      ],
      "probe": {
        "q": "Why weight each action by the reward-to-go G_t rather than the total episode return R(τ)?",
        "a": "Causality: the action at time t cannot influence rewards earned before t, so those terms are pure noise in the weight — they average to a constant that the baseline argument shows contributes zero to the expected gradient, but they add variance. Reward-to-go strips them: strictly lower variance, still unbiased. G_t is also a single-sample estimate of Q(s_t, a_t), which is what the weight 'should' be."
      },
      "tests": """
def _t_rtg():
    got = np.asarray(reward_to_go([1.0, 2.0, 3.0], 0.5))
    exp = np.array([1 + 0.5*2 + 0.25*3, 2 + 0.5*3, 3.0])
    assert got.shape == (3,), f"expected shape (3,), got {got.shape}"
    assert np.allclose(got, exp), f"got {got}, expected {exp} -- compute backwards: G = r + gamma*G"
_check("reward_to_go values", _t_rtg)

def _t_rtg_gamma1():
    got = np.asarray(reward_to_go([-0.01]*3 + [1.0], 1.0))
    exp = np.array([0.97, 0.98, 0.99, 1.0])
    assert np.allclose(got, exp), f"gamma=1 case wrong: got {got}"
_check("reward_to_go with gamma=1", _t_rtg_gamma1)

def _t_whiten():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    w = whiten(x)
    assert abs(w.mean()) < 1e-7, "whitened mean must be ~0"
    assert abs(w.std() - 1.0) < 1e-3, "whitened std must be ~1"
    c = whiten(np.array([5.0, 5.0, 5.0]))
    assert np.isfinite(c).all(), "constant input must not divide by zero -- add eps to the std"
_check("whiten: zero mean, unit std, eps-guarded", _t_whiten)
""",
      "solution": "def reward_to_go(rewards, gamma):\n    G, out = 0.0, []\n    for r in reversed(rewards):\n        G = r + gamma * G\n        out.append(G)\n    return np.array(out[::-1])\n\ndef whiten(x, eps=1e-8):\n    x = np.asarray(x, dtype=float)\n    return (x - x.mean()) / (x.std() + eps)"
    },

    # ------------------------------------------------------------------
    {
      "id": "f3s3",
      "title": "Update: the REINFORCE gradient, by hand",
      "prompt": "No autodiff today — you'll compute the policy gradient analytically. For the tabular softmax policy, ∇_{W[s]} log π(a|s) has a famous closed form: **(onehot(a) − π(·|s))**, and it's zero for every row of W other than s.\n\nImplement:\n```\ndef reinforce_grad(W, states, actions, weights):\n```\nReturn `dW` of shape `(5, 2)` — the **ascent** direction `Σ_t weights[t] · ∇_W log π(a_t|s_t)`. (`weights` = whitened reward-to-go from the previous stage.)\n\nThen wire the full loop (you have all the pieces) and satisfy yourself it learns: `W += lr * dW` should drive the policy toward 'always right'. The hidden tests check the gradient itself.",
      "starter": None,
      "hints": [
        "For each timestep t: p = action_probs(W, states[t]); g = -p; g[actions[t]] += 1  (that's onehot minus probs); dW[states[t]] += weights[t] * g. All other rows untouched.",
        "Derive it if asked: log π(a|s) = W[s,a] - logsumexp(W[s]). d/dW[s,j] = 1[j=a] - softmax(W[s])[j]. It's the same 'probs minus onehot' as the cross-entropy gradient, sign-flipped because here we differentiate log-prob, not negative log-prob."
      ],
      "probe": {
        "q": "Your loss in autodiff frameworks is -(logp * weights).sum() with weights detached. What breaks if weights are NOT detached?",
        "a": "The weight (advantage/return) is a constant in the policy-gradient theorem — the estimator is E[Ψ·∇log π] with Ψ held fixed. If gradients flow into Ψ (e.g. through the critic that produced it, or through the return's dependence on the policy), you're differentiating a different objective: extra terms appear, the estimator is no longer the policy gradient, and value errors leak directly into the policy update. Detaching enforces 'Ψ is data, not function'."
      },
      "tests": """
def _softmax_ref(v):
    z = v - v.max(); e = np.exp(z); return e / e.sum()

def _ref_grad(W, S, A, wts):
    dW = np.zeros_like(W, dtype=float)
    for s, a, w in zip(S, A, wts):
        g = -_softmax_ref(W[s]); g[a] += 1.0
        dW[s] += w * g
    return dW

def _t_grad_exact():
    rng = np.random.default_rng(3)
    W = rng.standard_normal((5, 2))
    S = [0, 1, 2, 1, 3]; A = [1, 1, 0, 1, 1]; wts = [0.5, -0.2, 1.0, 0.3, 2.0]
    got = reinforce_grad(W, S, A, wts)
    exp = _ref_grad(W, S, A, wts)
    assert got is not None and np.asarray(got).shape == (5, 2), "dW must be (5, 2)"
    assert np.allclose(got, exp, atol=1e-6), "gradient wrong -- per step: dW[s] += w * (onehot(a) - softmax(W[s]))"
_check("matches the analytic score-function gradient", _t_grad_exact)

def _t_untouched_rows():
    W = np.zeros((5, 2))
    got = np.asarray(reinforce_grad(W, [2], [1], [1.0]))
    assert np.allclose(got[[0, 1, 3, 4]], 0.0), "rows for unvisited states must stay zero"
    assert np.allclose(got[2], [-0.5, 0.5]), f"visited row wrong: got {got[2]}, expected [-0.5, 0.5]"
_check("only visited state rows receive gradient", _t_untouched_rows)

def _t_ascent_direction():
    # positive weight must increase the prob of the taken action
    W = np.zeros((5, 2))
    dW = np.asarray(reinforce_grad(W, [1], [1], [2.0]))
    W2 = W + 0.1 * dW
    z = W2[1] - W2[1].max(); e = np.exp(z); p1 = (e / e.sum())[1]
    assert p1 > 0.5, "ascent step with positive weight must raise P(taken action) -- check the sign"
_check("positive-weight step raises the taken action's probability", _t_ascent_direction)
""",
      "solution": "def reinforce_grad(W, states, actions, weights):\n    dW = np.zeros_like(W, dtype=float)\n    for s, a, w in zip(states, actions, weights):\n        g = -action_probs(W, s)     # -π(·|s)\n        g[a] += 1.0                 # + onehot(a)\n        dW[s] += w * g              # score-function estimator, ascent direction\n    return dW\n\n# the full loop, for the record:\n# for it in range(300):\n#     S, A, R = collect_episode(env, W, rng)\n#     wts = whiten(reward_to_go(R, 0.99))\n#     W += 0.1 * reinforce_grad(W, S, A, wts)"
    },

    # ------------------------------------------------------------------
    {
      "id": "f3s4",
      "title": "GAE: the bias-variance dial",
      "prompt": "Reward-to-go is unbiased but noisy. Bring in a value function. Implement Generalized Advantage Estimation over a **flat rollout buffer that may contain several episodes** (this is the production form):\n```\ndef gae(rewards, values, dones, last_v, gamma=0.99, lam=0.95):\n    # rewards, values, dones: np.arrays of shape (T,)\n    # dones[t] = 1.0 if the episode ended AT step t\n    # last_v: V(s_T) — bootstrap for the state after the final step\n    # returns (advantages, value_targets), each (T,)\n```\nδ_t = r_t + γ·V(s_{t+1})·(1−done_t) − V(s_t), advantage accumulates backwards as A_t = δ_t + γλ(1−done_t)·A_{t+1}, and value targets are A + V.\n\nThe `(1−done)` masks are the whole game here — tell me what each one prevents.",
      "starter": None,
      "hints": [
        "Backward loop over t: v_next = last_v if t == T-1 else values[t+1]; nt = 1 - dones[t]; delta = rewards[t] + gamma*v_next*nt - values[t]; acc = delta + gamma*lam*nt*acc; adv[t] = acc.",
        "Both masks matter: the one in delta stops BOOTSTRAPPING across an episode boundary (the next state is a new episode's start — its value is irrelevant); the one in the accumulator stops the ADVANTAGE chain from leaking backward across the boundary."
      ],
      "probe": {
        "q": "What do λ=0 and λ=1 reduce to, and what's the trade?",
        "a": "λ=0: A_t = δ_t = r_t + γV(s_{t+1}) − V(s_t) — one-step TD advantage: lowest variance, but biased wherever V is wrong (it bootstraps). λ=1: A_t = Σ γ^k r_{t+k} − V(s_t) — Monte-Carlo advantage: unbiased regardless of V (V only acts as a baseline), but high variance. λ interpolates the geometric mixture of n-step estimators; 0.95 is the standard compromise. Same bias-variance ladder as MC vs TD, packaged in one knob."
      },
      "tests": """
def _ref_gae(r, v, d, last_v, gamma, lam):
    T = len(r); adv = np.zeros(T); acc = 0.0
    for t in reversed(range(T)):
        v_next = last_v if t == T - 1 else v[t + 1]
        nt = 1.0 - d[t]
        delta = r[t] + gamma * v_next * nt - v[t]
        acc = delta + gamma * lam * nt * acc
        adv[t] = acc
    return adv, adv + v

def _t_gae_basic():
    r = np.array([1.0, 0.0, 1.0, 0.0]); v = np.array([0.5, 0.4, 0.3, 0.2])
    d = np.zeros(4); last_v = 0.1
    a, ret = gae(r, v, d, last_v, 0.9, 0.8)
    ea, eret = _ref_gae(r, v, d, last_v, 0.9, 0.8)
    assert np.allclose(a, ea, atol=1e-6), f"advantages wrong: got {a}, expected {ea}"
    assert np.allclose(ret, eret, atol=1e-6), "value targets must be advantages + values"
_check("GAE values on a no-terminal buffer", _t_gae_basic)

def _t_gae_terminal():
    r = np.array([0.0, 1.0, 0.0, 1.0]); v = np.array([0.9, 0.8, 0.7, 0.6])
    d = np.array([0.0, 1.0, 0.0, 1.0])     # two episodes in one buffer
    a, _ = gae(r, v, d, 5.0, 0.99, 0.95)   # big last_v: must be masked by final done
    ea, _ = _ref_gae(r, v, d, 5.0, 0.99, 0.95)
    assert np.allclose(a, ea, atol=1e-6), "terminal masking wrong -- (1-done) must gate BOTH the bootstrap in delta AND the backward accumulator"
    # episode-boundary independence: step 1's advantage must not see episode 2
    a2, _ = gae(r[:2], v[:2], d[:2], 0.0, 0.99, 0.95)
    assert np.allclose(a[:2], a2, atol=1e-6), "advantages leak across the episode boundary -- check the accumulator mask"
_check("terminal (1-done) masking cuts both chains", _t_gae_terminal)

def _t_lambda_limits():
    r = np.array([1.0, 2.0, 3.0]); v = np.array([0.3, 0.2, 0.1]); d = np.zeros(3)
    a0, _ = gae(r, v, d, 0.05, 0.9, 0.0)
    exp0 = np.array([r[0] + 0.9*v[1] - v[0], r[1] + 0.9*v[2] - v[1], r[2] + 0.9*0.05 - v[2]])
    assert np.allclose(a0, exp0, atol=1e-6), "lambda=0 must reduce to one-step TD residuals"
    a1, _ = gae(r, v, d, 0.05, 0.9, 1.0)
    g2 = r[2] + 0.9*0.05; g1 = r[1] + 0.9*g2; g0 = r[0] + 0.9*g1
    assert np.allclose(a1, np.array([g0, g1, g2]) - v, atol=1e-6), "lambda=1 must reduce to discounted-return minus value"
_check("lambda=0 -> TD residual; lambda=1 -> MC advantage", _t_lambda_limits)
""",
      "solution": "def gae(rewards, values, dones, last_v, gamma=0.99, lam=0.95):\n    T = len(rewards)\n    adv = np.zeros(T)\n    acc = 0.0\n    for t in reversed(range(T)):\n        v_next = last_v if t == T - 1 else values[t + 1]\n        nonterminal = 1.0 - dones[t]\n        delta = rewards[t] + gamma * v_next * nonterminal - values[t]\n        acc = delta + gamma * lam * nonterminal * acc\n        adv[t] = acc\n    return adv, adv + values"
    },

    # ------------------------------------------------------------------
    {
      "id": "f3s5",
      "title": "PPO: the clipped surrogate",
      "prompt": "Now the PPO objective itself. Implement:\n```\ndef ppo_policy_loss(logp_new, logp_old, adv, eps=0.2):\n    # all inputs (N,) arrays; returns a scalar LOSS (to minimize)\n```\nRules of the game: the ratio is computed in **log space**, `logp_old` and `adv` are constants (they came from the rollout), and the clip must be the **pessimistic** pairing.\n\nAlso implement the diagnostic every PPO dashboard logs:\n```\ndef clip_fraction(logp_new, logp_old, eps=0.2):  # fraction of |ratio-1| > eps\n```\nWalk me through what the min() does for positive vs negative advantage — that explanation is the actual deliverable of this stage.",
      "starter": None,
      "hints": [
        "ratio = np.exp(logp_new - logp_old). s1 = ratio*adv; s2 = np.clip(ratio, 1-eps, 1+eps)*adv; loss = -np.mean(np.minimum(s1, s2)). The MIN of the two surrogates, then negate for a loss.",
        "Why min is pessimistic: for adv>0 it caps the payoff of pushing ratio above 1+eps (no incentive to overshoot); for adv<0 the clipped branch caps how much you can gain by crushing the ratio below 1-eps. Either way the objective stops rewarding movement outside the trust region."
      ],
      "probe": {
        "q": "At the first gradient step after a rollout, what is the ratio, and what does the loss reduce to?",
        "a": "θ = θ_old, so every ratio is exactly 1: both branches equal adv, the min is inactive, and the loss is -mean(adv) — whose gradient is exactly the vanilla policy gradient. The clip only engages on epochs 2..K as θ drifts from θ_old; that's the sense in which PPO is 'first-order faithful' to the true gradient and only constrains the reuse of stale data."
      },
      "tests": """
def _t_ratio_one():
    lp = np.log(np.array([0.3, 0.5, 0.2]))
    adv = np.array([1.0, -2.0, 0.5])
    loss = ppo_policy_loss(lp, lp.copy(), adv, 0.2)
    assert np.isclose(loss, -adv.mean(), atol=1e-7), f"at ratio=1 loss must be -mean(adv)={-adv.mean():.4f}, got {loss:.4f}"
_check("ratio=1 reduces to -mean(adv)", _t_ratio_one)

def _t_clip_positive_adv():
    # ratio = 2.0 with adv > 0: clipped branch (1+eps)*adv must win the min
    logp_old = np.array([np.log(0.2)]); logp_new = np.array([np.log(0.4)])
    adv = np.array([1.0])
    loss = ppo_policy_loss(logp_new, logp_old, adv, 0.2)
    assert np.isclose(loss, -1.2, atol=1e-6), f"expected -(1+eps)*adv = -1.2, got {loss:.4f} -- is the min/clip applied?"
_check("positive adv, ratio>1+eps -> clipped (pessimistic) branch", _t_clip_positive_adv)

def _t_clip_negative_adv():
    # ratio = 2.0 with adv < 0: UNCLIPPED branch is more pessimistic -> min keeps it
    logp_old = np.array([np.log(0.2)]); logp_new = np.array([np.log(0.4)])
    adv = np.array([-1.0])
    loss = ppo_policy_loss(logp_new, logp_old, adv, 0.2)
    assert np.isclose(loss, 2.0, atol=1e-6), f"expected -min(2*-1, 1.2*-1) = 2.0, got {loss:.4f} -- min, not max: pessimism means the WORSE surrogate is kept"
_check("negative adv, ratio>1: unclipped branch kept (that's the pessimism)", _t_clip_negative_adv)

def _t_ratio_below():
    # ratio = 0.5 with adv < 0: clipped branch (1-eps)*adv wins the min
    logp_old = np.array([np.log(0.4)]); logp_new = np.array([np.log(0.2)])
    adv = np.array([-2.0])
    loss = ppo_policy_loss(logp_new, logp_old, adv, 0.2)
    assert np.isclose(loss, 1.6, atol=1e-6), f"expected -(1-eps)*adv = 1.6, got {loss:.4f} -- no reward for crushing the ratio below 1-eps"
_check("negative adv, ratio<1-eps -> clipped branch caps the gain", _t_ratio_below)

def _t_clipfrac():
    logp_old = np.log(np.array([0.5, 0.5, 0.5, 0.5]))
    logp_new = np.log(np.array([0.5, 0.65, 0.5, 0.3]))   # ratios 1.0, 1.3, 1.0, 0.6
    cf = clip_fraction(logp_new, logp_old, 0.2)
    assert np.isclose(cf, 0.5, atol=1e-6), f"expected 0.5 (2 of 4 outside [0.8, 1.2]), got {cf}"
_check("clip_fraction diagnostic", _t_clipfrac)
""",
      "solution": "def ppo_policy_loss(logp_new, logp_old, adv, eps=0.2):\n    ratio = np.exp(logp_new - logp_old)          # log-space: never divide probs\n    s1 = ratio * adv\n    s2 = np.clip(ratio, 1 - eps, 1 + eps) * adv\n    return -np.mean(np.minimum(s1, s2))          # pessimistic pair, negated\n\ndef clip_fraction(logp_new, logp_old, eps=0.2):\n    ratio = np.exp(logp_new - logp_old)\n    return float(np.mean(np.abs(ratio - 1.0) > eps))"
    },

    # ------------------------------------------------------------------
    {
      "id": "f3s6",
      "title": "GRPO: the LLM step (groups, masks, KL)",
      "prompt": "Last stage — adapt everything to the LLM setting. Two functions:\n```\ndef group_advantages(rewards, eps=1e-4):\n    # rewards: (B, G) — B prompts, G sampled completions each\n    # returns (B, G): per-GROUP normalized (r - group_mean) / (group_std + eps)\n\ndef grpo_loss(logp, logp_old, logp_ref, adv_seq, mask, eps=0.2, beta=0.04):\n    # logp, logp_old, logp_ref: (N, L) per-token log-probs (N = B*G sequences)\n    # adv_seq: (N,) one advantage per sequence — broadcast to its tokens\n    # mask:    (N, L) 1.0 on completion tokens, 0.0 on prompt/padding\n    # per-token: -min(ratio*A, clip(ratio)*A) + beta * kl,  kl = exp(lr) - lr - 1\n    #            with lr = logp_ref - logp   (the k3 estimator)\n    # return the mean over REAL tokens only: sum(per_tok * mask) / sum(mask)\n```\nThe two lines that decide this stage: the (B, G) normalization axis, and the masked mean. Both have tests aimed straight at them.",
      "starter": None,
      "hints": [
        "group_advantages: mean/std with axis=1, keepdims=True — statistics are PER PROMPT. Normalizing over the flat batch mixes prompts of different difficulty and corrupts the advantage.",
        "grpo_loss: ratio = np.exp(logp - logp_old); broadcast adv with adv_seq[:, None]; per_tok = -np.minimum(ratio*A, np.clip(ratio, 1-eps, 1+eps)*A) + beta*(np.exp(lr) - lr - 1). Then (per_tok * mask).sum() / mask.sum() — dividing by mask.sum() (not N*L) is the point."
      ],
      "probe": {
        "q": "Why is there no value network anywhere in GRPO, and what did we give up?",
        "a": "The group mean IS the baseline — an empirical V(prompt) computed by actually branching: G completions from the same state. Affordable because an LLM prompt resets for free and the whole completion is one action; this is exactly the exponential branching a critic exists to amortize in classical RL. What's lost: per-token credit assignment — every token in a completion gets the same advantage, so 'which step of the reasoning earned the reward' arrives only statistically across many groups. Also: all-same-reward groups (std≈0) contribute ~zero gradient — GRPO is blind on too-easy and too-hard prompts."
      },
      "tests": """
def _t_group_adv():
    r = np.array([[1.0, 0.0, 1.0, 0.0],
                  [10.0, 10.0, 12.0, 8.0]])
    a = group_advantages(r)
    assert a.shape == (2, 4), f"shape must be (B, G), got {a.shape}"
    assert np.allclose(a.mean(axis=1), 0.0, atol=1e-6), "each GROUP must have mean 0 -- normalize with axis=1"
    exp0 = (r[0] - 0.5) / (r[0].std() + 1e-4)
    assert np.allclose(a[0], exp0, atol=1e-3), "row-0 values wrong"
_check("advantages normalized per group (axis=1)", _t_group_adv)

def _t_group_not_global():
    # same rewards, different rows: global normalization would give different answers
    r = np.array([[0.0, 1.0], [100.0, 101.0]])
    a = group_advantages(r)
    assert np.allclose(a[0], a[1], atol=1e-3), "rows with identical within-group structure must get identical advantages -- you normalized over the whole batch, not per group"
_check("no cross-prompt leakage (the classic GRPO bug)", _t_group_not_global)

def _t_degenerate_group():
    a = group_advantages(np.array([[1.0, 1.0, 1.0]]))
    assert np.isfinite(a).all(), "all-same-reward group must not blow up -- eps in the denominator"
    assert np.allclose(a, 0.0, atol=1e-6), "all-same-reward group must give ~zero advantage"
_check("degenerate (all-correct / all-wrong) group is safe", _t_degenerate_group)

def _t_grpo_loss_value():
    logp     = np.log(np.array([[0.5, 0.4], [0.2, 0.6]]))
    logp_old = logp.copy()                       # ratio = 1 everywhere
    logp_ref = logp.copy()                       # kl = 0 everywhere
    adv  = np.array([2.0, -1.0])
    mask = np.ones((2, 2))
    loss = grpo_loss(logp, logp_old, logp_ref, adv, mask)
    assert np.isclose(loss, -0.5, atol=1e-6), f"ratio=1, kl=0 -> loss = -mean(broadcast adv) = -0.5, got {loss}"
_check("ratio=1, kl=0 sanity value", _t_grpo_loss_value)

def _t_mask_denominator():
    logp     = np.log(np.full((1, 4), 0.5))
    logp_old = logp.copy(); logp_ref = logp.copy()
    adv  = np.array([1.0])
    mask = np.array([[0.0, 0.0, 1.0, 1.0]])      # 2 real tokens of 4
    loss = grpo_loss(logp, logp_old, logp_ref, adv, mask)
    assert np.isclose(loss, -1.0, atol=1e-6), f"masked mean must divide by mask.sum()=2, not 4: expected -1.0, got {loss} -- prompt/pad positions must not dilute the loss"
_check("masked mean divides by real-token count", _t_mask_denominator)

def _t_kl_term():
    logp     = np.log(np.full((1, 1), 0.5))
    logp_old = logp.copy()
    logp_ref = np.log(np.full((1, 1), 0.25))     # lr = log(0.5) -> k3 = 0.5 - log(0.5) - 1
    adv  = np.array([0.0])                        # isolate the KL term
    mask = np.ones((1, 1))
    loss = grpo_loss(logp, logp_old, logp_ref, adv, mask, beta=1.0)
    exp = 0.5 - np.log(0.5) - 1.0
    assert np.isclose(loss, exp, atol=1e-6), f"k3 KL wrong: expected exp(lr)-lr-1 = {exp:.4f}, got {loss:.4f} with lr = logp_ref - logp"
    assert loss > 0, "k3 estimator is non-negative by construction"
_check("k3 KL estimator (exp(lr) - lr - 1, lr = logp_ref - logp)", _t_kl_term)
""",
      "solution": "def group_advantages(rewards, eps=1e-4):\n    m = rewards.mean(axis=1, keepdims=True)     # per-prompt statistics\n    s = rewards.std(axis=1, keepdims=True)\n    return (rewards - m) / (s + eps)\n\ndef grpo_loss(logp, logp_old, logp_ref, adv_seq, mask, eps=0.2, beta=0.04):\n    ratio = np.exp(logp - logp_old)              # (N, L)\n    A = adv_seq[:, None]                         # broadcast: one A per sequence\n    s1 = ratio * A\n    s2 = np.clip(ratio, 1 - eps, 1 + eps) * A\n    lr = logp_ref - logp                         # k3 KL estimator\n    kl = np.exp(lr) - lr - 1.0\n    per_tok = -np.minimum(s1, s2) + beta * kl\n    return float((per_tok * mask).sum() / mask.sum())   # mean over REAL tokens"
    },
  ]
}
