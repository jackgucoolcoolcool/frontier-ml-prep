# Mock Coding Interview — CoderPad Simulator

A local, offline-gradable simulation of a frontier-lab ML coding interview
(45 min, layered extensions, live-coding style — no AI assist, no autocomplete).

## Run

Nothing to run — the server is **always on** at http://localhost:5050, managed
by a launchd agent (`~/Library/LaunchAgents/com.yiminggu.mock-interview.plist`)
that starts it at login and restarts it if it crashes. To uninstall:
`launchctl bootout gui/$(id -u)/com.yiminggu.mock-interview && rm ~/Library/LaunchAgents/com.yiminggu.mock-interview.plist`.

Manual fallback: `python3 app.py` (needs `pip3 install flask numpy`).

**How the pieces connect:** macOS blocks non-terminal launchers from reading
`~/Documents`, so the served copy lives at
`~/Library/Application Support/mock-interview/`. A Claude Code hook
(`.claude/settings.json`) runs `./sync_preview.sh` automatically whenever
`app.py` / `problems.py` / `remediation.py` / `index.html` /
`gap_report.html` are edited — it copies the files over and restarts the
server. Debriefs auto-save on interview end into its `sessions/` dir;
`mock_interview/sessions` is a symlink to it.

## What it does

- **5 sessions**, each one layered problem revealed stage by stage, flavored to
  the two coding interviewers (anonymized **W** = rigor/efficiency,
  **F** = post-training/RL/agents), plus an algorithmic warm-up:
  - **W1** Attention from scratch → causal mask → multi-head → KV-cache decode
  - **W2** RMSNorm → LayerNorm → softmax-CE + gradient → 2-layer MLP backprop
    (graded against finite differences)
  - **F1** Temperature → top-k/top-p → repetition penalty → best-of-n
  - **F2** Bradley–Terry RM loss → DPO → PPO clip → GRPO advantages
  - **A1** Merge intervals, longest unique substring
- A scripted **interviewer**: intro, stage prompts, time warnings, escalating
  hints when submits fail, and a verbal **probe question** after each stage
  (answer it out loud — that's half the real score).
- **Hidden tests** run server-side in a real Python subprocess (15 s timeout),
  including the classic traps: numerical stability at logits ~1000, causal-mask
  leaks, the `abba` sliding-window trap, PPO pessimism with negative advantages.
- **Debrief**: per-stage results/times/hints, self-score rubric, model
  solutions + strong probe answers, notes — saved to `sessions/` (gitignored)
  and localStorage.
- **Reading Room** (`/study`, `study.html`): pre-mock reading for **all five**
  coding tracks, in the same stage ladder as the interviews. Per stage:
  concept → math (MathJax) → the exact code the grader accepts → the traps
  the hidden tests spring → the probe Q&A, with visualizations (attention
  heatmaps, head-split & pre/post-norm diagrams, KV-cache cost curves,
  CE loss/gradient, temperature sweep, top-k/top-p/min-p cutoffs, best-of-n
  Goodhart curve, Bradley–Terry loss/gradient, DPO-saturation-vs-IPO,
  the canonical PPO clip figure, GRPO-vs-RLOO difficulty bias, interval-merge
  and "abba" sliding-window traces). Ch. 4 includes the full DPO derivation
  from the KL-constrained RLHF objective and the PPO pessimism quadrant
  table; variant material (IPO, KL shaping, RLOO, insert-interval,
  k-distinct) is covered inline. **Ch. 6 (research-round architecture)**:
  the full decoder block with parameter accounting (N ≈ 12Ld²), SwiGLU,
  RoPE (relative-position derivation, base-vs-distance decay chart,
  long-context scaling), GQA/MQA, MoE (top-k routing, load-balancing aux
  loss, router collapse, params-vs-FLOPs chart), and the vision path
  (ViT patchify, CLIP/SigLIP losses, VLM connector design-space table).
- **Resume deep-dive mocks** (purple cards): 6 verbal sessions covering the
  Google/Gemini resume topics — multimodal pretraining, MM scaling laws,
  Science of Data, visual & physical understanding, MMU↔MMGen, and MM
  post-training/agentic verticals (`resume_sessions.py`). Each question:
  answer OUT LOUD 3–5 min → interviewer follow-ups press you → reveal what a
  strong answer covers → self-score 1–5. Scores are logged per topic
  (latest score per question wins) and feed the gap report's
  "Resume deep-dive readiness" section with re-drill links.
- **Focus banner + drill links**: the picker shows your weakest concept/topic;
  `/?drill=<stage_id>` jumps straight into any single stage or question.
- **Variant rotation** (`variants.py`): every coding stage has alternate
  flavors of the same concept (batched attention, arbitrary masks,
  cross-attention, sliding-window KV cache, fix-the-buggy-code norms,
  binary logistic, L2 backprop, log-space sampling, min-p, freq/presence
  penalties, self-consistency, RM metrics, IPO, KL shaping, RLOO, insert
  interval, k-distinct window), and every resume question has an alternate
  framing (skeptic/teaching/product/postmortem angles). The variant served
  rotates with your attempt count for that stage — a re-drill never repeats
  the exact task, so you prove the concept, not the memorized answer.
  `verify.py` checks all base + variant solutions (98 checks).
- **Error logging + Gap Report** (`/gap_report`): every Submit is logged to
  `sessions/attempts.jsonl` with each failed test classified into a concept
  tag (numerical stability, causal masking, backprop gradients, …). The Gap
  Report ranks your weakest concepts and assembles targeted remediation for
  exactly those: your actual failure messages, an explanation, the pattern to
  memorize, out-loud drills, and links into the Day docs. Tag content lives in
  `remediation.py`.

## Interview discipline

Treat every run as real: talk out loud the whole time, ask a clarifying
question before coding, narrate shapes, test small examples with **Run**
(Cmd/Ctrl+Enter) before **Submit stage**.

**History manager** (`/history`): every coding submission, verbal score, and
debrief listed individually with checkboxes — delete exactly the records you
want (they move to `sessions/archive/deleted_*.jsonl`, never destroyed, and
stop counting toward the gap report and variant rotation immediately).
Deletion is guarded by index+timestamp so concurrent submissions are safe.

**Reset history**: the 🗑 button on the gap report archives all logged
history (submissions, verbal scores, debriefs) to `sessions/archive/<ts>/` —
nothing is deleted — and clears the picker's localStorage stats. Gap report
and variant rotation start fresh; restore by moving files back out of the
archive folder.

`verify.py` re-checks that every model solution passes its own hidden tests.
