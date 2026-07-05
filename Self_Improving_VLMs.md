# Self-Improving VLMs — The Full Map, the Data-Side Zoom, and the MM Web-Scraper Project

> **Why this doc:** "How do frontier multimodal models improve themselves?" is a top-tier research-taste conversation, and "walk me through a data-engine project" is a top-tier project-deep-dive conversation. Part I is the open brainstorm (vision / pretraining / post-training × agent / data / model). Part II zooms into data-side self-improvement. Part III is a fully-worked mock project — *Self-Improving Multimodal Web Scraper* — written as if executed inside a Gemini-scale org, with the real problems such a project actually hits.
>
> **Honesty note:** Part III is a *hypothetical composite* for mock-interview practice and design-exercise answers ("design a data engine for Gemini"). Use it to practice narration, trade-off defense, and pushback handling — not as literal work history.

---

## Part I — The full brainstorm: how modern VLMs self-improve

### §1.0 The one mental model: every self-improvement scheme is Generate → Verify → Update

Every scheme in this zoo is one loop with three slots:

$$\text{Generate}(\pi_\theta) \;\to\; \text{Verify}(V) \;\to\; \text{Update}(\theta' \leftarrow \theta)$$

- **Generate:** the model (or an agent built from it) produces *something* — captions, rationales, trajectories, crawl decisions, preference judgments, synthetic images.
- **Verify:** some signal decides what survives — a programmatic checker, cross-modal consistency, a judge model, environment reward, a human audit, or downstream-loss improvement.
- **Update:** the surviving artifact changes the model — as pretraining data, SFT data, preference pairs, RL reward, distillation targets, or a changed *data-acquisition policy*.

**The whole design space is: what do you generate, who verifies it, and where does the update land?** The single most important axis is **verifier quality**: self-improvement is exactly as trustworthy as its weakest verifier (this is the generator–verifier gap: verification is usually easier than generation, and the loop mines that gap). Everything else is plumbing.

The loop can close at three timescales:
1. **Inference-time** (no weight update): self-consistency, self-critique + revise, best-of-n with a verifier, tool calls. "Self-improvement" per query.
2. **Post-training-time** (days): rejection-sampling FT, iterative DPO, RLVR, RLAIF.
3. **Pretraining-time** (months): the model curates/relabels/synthesizes the corpus for the *next* model generation. This is where the biggest compounding lives, and where Part III operates.

### §1.1 Vision stage (encoder & perception)

- **Self-supervised objectives are already self-improvement in miniature.** DINO/DINOv2/v3: an EMA teacher (the model's own slow average) generates targets for the student — Generate=teacher features, Verify=architectural (centering/sharpening prevents collapse), Update=distillation. JEPA: predict your own latent of masked/future content. MAE: reconstruct your own masked input. No labels ever enter.
- **Model-in-the-loop annotation engines.** SAM's data engine is the canonical citation: assisted-manual → semi-auto → fully-auto annotation, model retrained each round, 1.1B masks. Same pattern for grounding boxes, OCR pseudo-labels, depth/pose pseudo-labels from specialist teachers.
- **Recaptioning flywheels.** Alt-text is garbage; so use the current VLM to rewrite captions, train the next contrastive/generative model on them (DALL·E 3's recaptioner, VeCLIP, CapsFusion, Nemotron-style pipelines). Key learned lesson: **fuse** synthetic + raw captions (CapsFusion) — pure synthetic captions lose the proper-noun/world-knowledge in alt-text and collapse caption style.
- **Synthetic imagery with programmatic ground truth.** Rendered charts/tables/documents/UIs/math figures give *infinite perfectly-labeled* perception data (the label is the rendering program's source). Text-to-image models fill tail concepts; cycle-consistency (caption the generated image, compare to prompt) filters junk.
- **Encoder consolidation by self-/agglomerative distillation.** Distill CLIP+DINO+SAM into one backbone (AM-RADIO), or distill your own bigger encoder down (DINOv3's pipeline). The model family improves itself by digesting its own zoo.

### §1.2 Pretraining stage

- **Model-based filtering.** Train a filter network to select data for the next model (Data Filtering Networks / DataComp): CLIP-score gates, quality classifiers, aesthetic scores. DFN's counterintuitive result: the best *filter* model is trained on a small verified-clean set, not the biggest model — filtering ability ≠ general capability.
- **Learnability / influence scoring.** Score examples by excess loss vs a reference model (RHO-loss: pick points the current model gets wrong but a clean reference gets right — filters both the known and the unlearnable), or by estimated influence on target-task loss (MATES, datamodels/TRAK approximations).
- **Mixture & curriculum optimization.** DoReMi (proxy model + group-DRO reweights domains), RegMix (fit a regression from small-run mixtures → extrapolate), scaling-law-based mixture extrapolation (fit per-domain loss curves at small scale, optimize the mix at target scale — the Apple/NMM-style multimodal scaling-law lens applies directly: the optimal image:text:interleaved ratio is itself a scaling-law object). Curriculum: quality-annealing — save the highest-grade curated data for the final LR-decay phase, where it moves the needle most.
- **Generational self-distillation.** Checkpoint N filters, relabels, and synthesizes the corpus for checkpoint N+1; weak-to-strong: a weaker supervisor's labels can still elicit stronger-model capability, which is why the loop doesn't obviously cap at teacher level.
- **Synthetic interleaved documents** — model-authored "textbook" pages around real images (Phi-style, multimodalized), OCR-enriched documents, chart-with-derivation pages.

> **Whiteboard equations.** RHO-loss selection: $x^\star = \arg\max_{x\in B}[\ell_\theta(x) - \ell_{\text{ref}}(x)]$ (high current loss = not yet learned; low reference loss = learnable — a reference model turns raw loss into learnability). DoReMi: $\alpha^\star = \arg\max_\alpha \min_\theta \sum_d \alpha_d[\ell_{\theta,d} - \ell_{\text{ref},d}]$. **Mixture as a scaling-law object:** fit $L_d(N, D_d) = E_d + A_d N^{-\alpha_d} + B_d D_d^{-\beta_d}$ per domain, allocate tokens so $\partial L/\partial D_d = \lambda\ \forall d$ — since $\beta_d, E_d$ differ per domain, **the optimal mixture is scale-dependent** (this is why per-category value curves cross; see P1 and Fig. 6 in the HTML).

### §1.3 Post-training stage

- **Rejection-sampling / self-training family (the workhorse):** STaR, ReST, ReST^EM: sample k solutions, keep verified-correct ones (+ *rationalization*: if none correct, show the answer and ask for the reasoning), fine-tune, iterate. Simple, stable, embarrassingly parallel.
- **RL with verifiable rewards (RLVR)** — the strongest current loop because the verifier is programmatic: visual math with checkable answers, chart/table QA against source data, OCR exact-match, counting vs detector, **screenshot-to-code with render-and-compare** (generate code, render it, pixel/DOM-diff against target — a gorgeous purely-visual verifier), GUI-agent tasks with environment-state success checks.
- **RLAIF / self-rewarding:** the model judges its own (or peers') outputs → preference pairs → iterative DPO (Self-Rewarding LMs); constitutional-style critique-and-revise generates its own SFT data. Multimodal versions (RLAIF-V) split claims into atoms and verify each atom separately.
- **Hallucination-targeted loops:** generate (hallucinated, corrected) caption pairs via perturbation or detector feedback → DPO (POVID, HA-DPO); reward = fraction of caption atoms confirmed by detectors/OCR.
- **Self-consistency as pseudo-labeling:** majority-vote answer across samples becomes the label when no verifier exists (works because verification-by-agreement is cheaper than correctness).
- **Agentic post-training:** run web/GUI/tool agents in real or simulated environments, keep successful trajectories → SFT → then RL on environment reward. The environment is a free verifier.

> **Whiteboard equations.** ReST^EM is EM on the verifier: $\theta_{t+1} = \arg\max_\theta \mathbb{E}_{y\sim\pi_{\theta_t}}[\mathbb{1}[V(x,y){=}1]\log\pi_\theta(y|x)]$. DPO: $-\log\sigma(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)})$. RLVR: $\max_\theta \mathbb{E}[R_{\text{verify}}] - \beta\,\mathrm{KL}(\pi_\theta\Vert\pi_{\text{ref}})$. **GRPO** (how RLVR actually runs): group-relative advantage $A_i = (r_i - \text{mean}(r_{1:G}))/\text{std}(r_{1:G})$, PPO-style clipped loss, no value network. Doer notes: all-pass/all-fail groups give zero gradient — task-difficulty curation (pass-rate ≈ 50%) is itself a data problem; std-normalization makes reward scale irrelevant but reward *blind spots* fatal.

### §1.4 The three levers, cross-cutting

**Agent lever** (the model *acting* to improve itself):
- **Collector:** crawls/browses to acquire data (→ Part III).
- **Annotator:** captions, boxes, transcribes, rates.
- **Verifier/judge:** scores outputs, checks claims, red-teams.
- **Curriculum designer:** proposes its own tasks at the frontier of its ability (Voyager, self-instruct task proposal, Absolute-Zero-style propose-and-solve with the solver's learnability as the proposer's reward).
- **Tool-user:** calls OCR/detectors/search/code — tools inject external ground truth into an otherwise closed loop.

**Data lever:** filter, relabel, synthesize, actively acquire, mix/schedule, dedup, recycle deployment data. (All of Part II.)

**Model lever:**
- EMA/self-distillation teachers; distill big→small and best-of-n→policy (amortize search into weights).
- Weight averaging (soups, EMA of RL checkpoints), merging specialists.
- MoE upcycling (grow a dense checkpoint into experts).
- Weak-to-strong generalization across model generations.
- **Deployment flywheel:** thumbs, regenerations, edits, retention → preference data → next post-train. The quietest but largest real-world loop.

### §1.5 Why the loop fails (know these cold)

| Failure | Mechanism | Mitigation |
|---|---|---|
| **Model collapse** | recursive training on own outputs → tails vanish, variance shrinks (Shumailov et al.) | keep a **real-data anchor ratio**; accumulate rather than replace; track tail/diversity metrics |
| **Self-preference bias** | judge prefers its own style → scores drift from quality | judge ensembles, human-audited golden sets, style-controlled eval |
| **Reward hacking** | generator exploits verifier blind spots | verifier hierarchies (programmatic > consistency > judge), refresh verifiers, KL limits |
| **Diversity narrowing** | selection pressure homogenizes data/behavior | entropy/coverage monitors, exploration budgets, cluster caps |
| **Contamination** | self-collected data swallows the evals | decontamination service, canaries, fresh held-out evals |
| **Capability ceiling** | no new information enters a closed loop | tools, environments, fresh web data, humans = external information sources; the loop *redistributes* competence unless a verifier injects truth |

### §1.6 Deep insights — the theory under the plumbing

1. **Selection vs. gradient — the two arms are not equal.** Filtering, best-of-n, rejection sampling are *selection operators*: they reweight the support of the current distribution but cannot create new modes. Selection is also KL-bounded: best-of-n drifts at most $\log n - \frac{n-1}{n}$ nats from the base policy (~2.5 nats at n=32; doubling n buys only ~0.69 more). The escape hatch is **iterate-and-distill** (ReST^EM, iterative DPO): each round distills the selected distribution *into the weights, re-centering the reference*, then selects again — the KL budget compounds across rounds, and gradient updates generalize selected behaviors into compositions selection alone could never reach. This is the deep reason "amplify → distill → repeat" is the master recipe, and why one-shot best-of-1024 is not.
2. **Entropy accounting.** Every loop converts diversity into capability. Entropy income: fresh web data, tools, environments, humans. Entropy expense: every selection step. Income < expense → collapse (model collapse is just the bankruptcy case). The anchor ratio works because it lower-bounds income. Doer corollary: track corpus/policy entropy like a KPI, not as a post-mortem.
3. **Verifier decay is dynamical, not an event.** Under optimization pressure, every *learned* verifier slides down the hierarchy toward "heuristic" — Goodhart is a rate, not a threshold. Consequence: **refresh cadence beats initial quality**; a mediocre judge refreshed every cycle outlives a brilliant frozen one. (P2/P9/P15 are the same theorem observed three times.)
4. **Elicitation vs. creation.** Post-training self-improvement mostly *surfaces* what pretraining latently contains; the pass@k vs pass@1 gap is the elicitable headroom. Doer rule: **measure pass@1024 before green-lighting an RL loop** — if pass@k is flat, no reward will help; go fix the pretraining data instead. Corollary: data-side loops raise the ceiling, post-training loops raise the floor toward it.
5. **Captions are the label space.** For VLMs, recaptioning isn't "cleaning" — it decides *which visual features become linearly accessible*: dense spatial captions teach layout, entity-rich alt-text teaches world-knowledge grounding, OCR-injection teaches reading. "What you caption is what you get": the mixture over caption *styles* is a first-class hyperparameter, not a footnote.
6. **Data outlives architecture.** A corpus improvement pays into every future model generation and survives architecture migrations; a post-training trick pays once. That asymmetry — not fashion — is why data engines get the senior headcount, and why the slowest loop (pretraining-timescale) has the highest NPV.
7. **The binding rate is eval latency.** The cycle time of Generate → Verify → Update is dominated by "how long until you can *trust* a measurement," not by generation. Halving eval latency doubles the improvement rate of the whole engine — which is why probe suites and scaling ladders (P1) are the crown jewels, and why the org that measures fastest wins.

---

## Part II — Zoom: data-side self-improvement

**Definition:** the model improves the *corpus* (and the corpus-building policy) rather than directly improving weights. Six verbs, ordered by how much the model creates vs. selects:

### §2.1 FILTER — score and select
- Signals: model-judge quality scores distilled into cheap scorers (DFN pattern: big model labels 1M pages → train a 100M scorer → run on 10B pages), CLIP alignment, OCR density, learnability (reference-model excess loss), toxicity/NSFW/PII gates.
- Design insight: **a filter is a model with a ceiling** — it embodies one generation's taste. Refresh it each model generation or it becomes the bottleneck (and its false-negative *systematics*, not its error rate, are what hurt: it silently deletes whole capability categories, e.g. medical images flagged as gore).

### §2.2 RELABEL — enrich what you keep
- Dense recaptioning (spatial relations, attributes, text-in-image), OCR injection, grounding boxes, table→markdown, chart→underlying-data.
- **Caption fusion** beats caption replacement: synthetic captions are visually thorough but style-collapsed and entity-poor; alt-text is noisy but carries proper nouns and world knowledge. Fuse them (CapsFusion), don't overwrite.
- Verifier stack for labels: programmatic (OCR/detector cross-check) > cycle-consistency (does the caption retrieve the image?) > judge score.

### §2.3 SYNTHESIZE — create what the web doesn't have
- QA/instruction pairs from images; CoT rationales via STaR (keep only rationales reaching the verified answer).
- **Rendered-with-source data**: charts, documents, UIs, geometry figures generated by programs — the program *is* the ground truth, so supervision is infinite and perfectly verifiable. The highest-leverage synthetic category for VLMs.
- Text-to-image for tail concepts + cycle-consistency filtering; hard-negative construction (minimal caption edits that flip correctness) for contrastive/DPO data.

### §2.4 ACQUIRE — actively collect (the scraper verb)
- Eval-diagnostics → targeted acquisition: model weak on infographic reasoning → crawl infographic-rich domains. Closes the loop from *measured weakness* to *data intake*.
- Uncertainty/disagreement-driven: pages where ensemble members disagree are informative.
- This is a **bandit/RL problem over the web**: action = what to crawl/render, reward = realized data value. Part III lives here.

### §2.5 MIX & SCHEDULE
- DoReMi/RegMix/scaling-law extrapolation for domain weights; multimodal ratios (image:text:interleaved) as scaling-law objects.
- Curriculum: short→dense captions; quality annealing (best data last, during LR decay).

### §2.6 RECYCLE deployment exhaust
- Preference pairs from thumbs/regenerations/edits; hard-example mining from failures; implicit success signals (user stopped rephrasing).

### §2.7 Cross-cutting machinery
- **Dedup** (MinHash + SemDeDup embedding clustering) — also a *quality* intervention (dup density anti-correlates with quality) and a *memorization* intervention.
- **Decontamination** against every eval you'll ever report.
- **The anchor ratio:** never let model-touched data exceed a fixed fraction; real data is the entropy source.
- **A data point's value** ≈ quality × learnability × marginal-diversity × verifiability. Selection on any one alone fails: quality-only → homogeneous; learnability-only → noise-seeking; diversity-only → junk.
- **The meta-loop:** eval diagnostics → acquisition/synthesis targets → curation → proxy-pretrain validation → mixture update. Data-side self-improvement = making this loop *fast* and *trustworthy*.

**Three more whiteboard tools:**
1. **Targeted acquisition = importance resampling (DSIR):** $w(x) \propto \hat p_{\text{target}}(x)/\hat p_{\text{raw}}(x)$ with densities estimated from cheap features (hashed n-grams / embedding bins); eval-gap crawling is DSIR with $\hat p_{\text{target}}$ fit on exemplars of the weak capability.
2. **Cycle-consistency label filter:** within a batch of $B$ images, keep synthetic caption $c_i$ iff $\arg\max_j \text{sim}(c_i, x_j) = i$ — catches captions describing *an image like this* rather than *this image*; raise $B$ to raise strictness.
3. **Diversity accounting:** cluster the intake embedding space, track $H = -\sum_k p_k \log p_k$, report **effective clusters $e^H$** (perplexity of the cluster distribution). The P4 alarm metric; entropy in nats, naturally. (Visualized as Fig. 5 in the HTML.)

---

## Part III — The project: Self-Improving Multimodal Web Scraper (Gemini-setting mock narrative)

*(Composite/hypothetical; for interview practice and "design this system" answers.)*

### §3.1 Elevator pitch (STAR, 45 seconds)

**Situation.** Gemini pretraining wanted more *high-value* multimodal web data — interleaved image-text documents, charts, real documents, UI screenshots. The classic pipeline (bulk crawl → static heuristic filters) wastes most crawl/render/storage cost on boilerplate, and its filters encode one frozen notion of quality.

**Task.** Build a closed-loop data engine: a VLM-guided scraper that decides *where to crawl*, *what to extract and render*, *how to label it*, and — the self-improving part — updates its own acquisition policy and quality scorer from measured downstream data value.

**Action.** Five subsystems (below); three policy-improvement iterations shipped.

**Result** (illustrative numbers for practice): ~4× useful-tokens-per-crawl-dollar vs the heuristic baseline; at matched token budget, proxy-scale pretrains showed +6 ChartQA, +4 DocVQA, +3 InfographicVQA, neutral on text-only evals; the distilled quality scorer replaced two legacy heuristic filters.

### §3.2 System design

1. **Acquisition policy (the "where").** A prioritized frontier over URLs/domains. Features: domain-level yield history, URL patterns, link-graph authority, cheap pre-render fingerprints (content-type guess from HTML head). Starts as heuristics; becomes a learned value model trained on *realized* data value of past crawls; explored as a contextual bandit with an explicit exploration budget.
2. **Rendering & extraction (the "what").** Tiered: cheap static HTML parse for everything; full headless render (screenshot + DOM + reading-order alignment) only when the policy predicts value above a threshold. Constructs interleaved documents (text blocks + images in reading order); specialist extractors for charts, tables, PDFs, UIs.
3. **Labeling arm (the "enrich").** Flash-tier VLM recaptions images with fused captions (alt-text + synthetic); OCR cross-checks scene text; grounding boxes for a subset; QA-pair synthesis for post-training reuse of the same pages.
4. **Value scoring (the "keep").** A distilled small scorer (DFN pattern: big-model judgments on ~1M pages → 300M-param scorer → runs on everything) plus a learnability score (reference-model excess loss) plus a marginal-diversity score against a corpus embedding index. Kept as *separate* signals, combined at selection time — never collapsed into one opaque scalar (you need to debug them independently).
5. **Feedback loop (the "improve").** Weekly proxy pretrains (~1B params, fixed seeds/data order) on candidate batches → per-capability eval suite → (a) mixture weights updated, (b) scorer retrained with fresh judgments, (c) acquisition policy updated with realized value, (d) eval-gap report drives *targeted* crawl campaigns (weak at infographics → infographic-heavy domains up-weighted).

**The self-improvement claim, precisely:** the loop improves *the data engine* (policy + scorer + labeler), which improves the corpus, which improves the model, which improves the judgments powering the engine. Model-side training itself stays standard.

### §3.3 Real problems faced (and what actually worked)

**P1 — Credit assignment: you can't wait for a frontier pretrain to learn if data helped.**
The core measurement problem: true data value is defined by a training run you can't afford to run per batch. Tried in order: (a) CLIP-score filtering — plateaued fast, biased toward stock-photo aesthetics, actively anti-selected documents/charts; (b) learnability-only (reference-model excess loss) — better, but noise-seeking: garbled OCR pages score as "hard but learnable"; (c) **landed on a scaling ladder**: proxy pretrains at 3 sizes with per-capability eval suites, checking that a candidate batch's *ranking* is stable across scales before trusting it. Found rank correlation proxy↔larger-scale ≈ 0.7 overall but *much worse for OCR-heavy data*, whose value only shows at scale — so OCR data got its own dedicated probe suite instead of the generic one. **Lesson:** data value is scale-dependent; validate the *transfer of rankings*, not a single score. (This is a scaling-law problem — fit per-category value curves, extrapolate, and distrust categories whose curves cross.)

**P2 — The distilled scorer learned style, not quality.**
After distilling big-model judgments, the scorer's top decile filled with SEO listicles that *mimic* the surface structure of good content (headings, images, FAQs). Diagnosis: the judge itself had self-preference/style bias, and distillation amplified it. Fixes: a human-audited **golden set** (few thousand pages, refreshed monthly) as the scorer's actual acceptance test; adversarial probe sets (SEO farms, machine-translated content, template mills); ensembling the judge with judge-independent signals (OCR density, link-graph authority, template fingerprints). **Lesson:** a distilled scorer inherits its teacher's biases *compressed and amplified*; you need ground truth that never touched the teacher.

**P3 — Recaption hallucination poisoning.**
Audit found the recaptioner asserting wrong object counts, misread scene text, and invented attributes on ~a few percent of images — at pretraining scale that's a curriculum in *confident hallucination*. Fixes: verifier stack — OCR exact-match for any quoted scene text, open-vocab detector confirmation for counted/named objects, cycle-consistency (does the caption retrieve its image from within a batch?); uncertainty-aware phrasing templates for unverifiable claims; **kept ~40% raw alt-text via caption fusion** as an entropy/entity anchor. Post-fix, proxy models improved on hallucination evals (POPE/CHAIR-style) instead of regressing. **Lesson:** synthetic label errors are not random noise — they're systematically fluent, so the model learns them *preferentially*.

**P4 — Feedback-loop distribution collapse.**
By policy iteration 3, the crawler had concentrated on a few scorer-favorite content types; embedding-cluster entropy of weekly intake dropped ~20%, and long-tail domains vanished. Nothing "failed" — every individual page was good; the *portfolio* degraded. Fixes: fixed 15% exploration budget (stratified random + frontier domains); per-cluster intake caps; a submodular-style marginal-diversity bonus in the acquisition objective; coverage dashboards with alarms as first-class launch criteria. **Lesson:** optimizing item-level score is the wrong objective; data value is a *set function*. This is the data-engine version of RLHF entropy collapse — same math, same fix shape (explicit entropy/KL term).

**P5 — Templated near-duplicates.**
E-commerce and news templates dominated raw intake; embedding dedup either missed them (different products, same template) or, when tightened, deleted legitimately-similar content — *charts of different data look alike in embedding space*. Fix: two-stage dedup — DOM-structure MinHash for template detection with per-domain caps, then embedding dedup only *within* a content type with type-specific thresholds (documents tight, charts loose). **Lesson:** one global similarity threshold is a category error in multimodal corpora.

**P6 — Rendering fidelity and cost.**
Lazy-loaded images missing from screenshots, cookie banners occupying 30% of pixels, JS-heavy pages timing out the render farm, anti-bot walls. Fixes: scroll-and-settle policies with network-idle detection; a banner/overlay classifier + DOM-node removal before screenshot; tiered render budgets (static parse → light render → full render) gated by predicted value; strict robots.txt compliance and per-domain politeness as non-negotiable constraints (also: blocked-crawl signals fed back as features, not fought). **Lesson:** at scale, the boring rendering layer determines data quality as much as any ML component; garbage screenshots poison everything downstream and are *invisible* in aggregate metrics until you look at samples. Institutionalized weekly "look at 100 random samples" reviews.

**P7 — Eval contamination, adversarially concentrated.**
The scraper *naturally gravitates* toward benchmark-like content — benchmark figures circulate on the web, and bench-like pages score high on every value signal (that's why they were chosen as benchmarks). Decontamination (n-gram + perceptual-hash + embedding match against all internal/external eval suites, plus canary strings) caught a small overall rate that was **heavily concentrated in exactly the highest-value categories**. Fix: decontamination as a mandatory pipeline stage with quarantine logs and eval-suite versioning; report contamination rates alongside every claimed win. **Lesson:** a value-seeking acquisition policy is a contamination-seeking policy; the better your engine, the worse this gets.

**P8 — Safety/licensing filters vs. recall.**
CSAM/NSFW/PII filtering runs at maximal recall, non-negotiably — which collateral-damaged medical imagery, art nudes, and document photos containing incidental faces. License-signal extraction (CC markers, paywalls, takedown lists) added another recall hit. Fixes: tiered filtering with a human-review lane for the gray zone; provenance metadata retained end-to-end so takedowns/audits are executable later; explicit accounting of what capability categories the safety stack suppresses (so the gap is *known*, not mysterious). **Lesson:** safety filters are part of the data distribution; you must measure what they remove, because "mysteriously bad at X" is often "the filter ate X."

**P9 — The acquisition policy reward-hacked the scorer.**
The bandit found a scorer blind spot: equation-dense pages scored very high (the scorer associated math density with quality) but many were OCR-garbled or scraped-and-mangled math forums — the policy flooded intake with them. Classic Goodhart, exactly analogous to RLHF reward hacking. Fixes: **asynchronous adversarial cadence** — the scorer is refreshed with fresh big-model judgments *after* every policy update, so the policy optimizes a moving (repaired) target; a KL-style constraint limiting how far each iteration's intake distribution can move from the previous one; human spot-checks of every new policy's top-decile picks *before* ingestion. **Lesson:** any learned value model under optimization pressure will be hacked; schedule verifier refresh against optimizer updates, and rate-limit distribution shift.

**P10 — Attribution: which change caused the win?**
A mixture update, a scorer refresh, and a new chart extractor shipped in the same cycle; ChartQA went +5; leadership asked which change to double down on. Untangling post-hoc was nearly impossible. Fixes: one-change-per-proxy-run discipline (proxy runs are cheap — use them as controlled experiments, seeds and data order fixed); interleaved A/B pretrains at matched token counts for any headline claim; cheap influence approximations (TRAK/datamodel-style) for post-hoc triage only, never for claims. **Lesson:** a data engine without experimental discipline produces improvements you can't reproduce — treat data changes with the same rigor as architecture changes.

### §3.4 Results framing and next steps

- Frame wins as **data-efficiency claims at matched compute**: "+6 ChartQA at equal token budget" beats "we collected 10× more data."
- Next steps (shows momentum): unify with post-training — the same pages yield verifiable QA pairs (charts with source data = free RLVR tasks); promote the scorer into annealing-phase curation where quality matters most; extend to video (the render farm generalizes); feed deployment eval-gaps directly into crawl campaigns, closing the largest loop.

### §3.5 Interview delivery notes

- **30-second version:** P1 + P4 + headline result. **2-minute:** add P2/P3/P9. **Deep dive:** P1, P4, P9 are the research-taste problems (measurement, set-valued objectives, Goodhart); P5–P7 show engineering scar tissue; P8/P10 show judgment and rigor.
- The three deep problems are all one theme: **the loop optimizes proxies, and every proxy breaks under optimization pressure** — the job is building the verifier hierarchy and refresh schedule that keeps proxies honest.
- Literature anchors to drop naturally: SAM data engine, DFN/DataComp, RHO-loss, DoReMi/RegMix, CapsFusion, SemDeDup, STaR/ReST, self-rewarding LMs, model-collapse (Shumailov), weak-to-strong.
- Expected pushback + answers: *"Doesn't this collapse?"* → anchor ratio + exploration budget + entropy monitoring (P4). *"How do you know the data helped?"* → scaling ladder + rank-transfer validation + A/B discipline (P1, P10). *"Why not just filter harder?"* → filters have ceilings and systematic false-negatives (§2.1, P8).

### §3.6 Crawler self-improvement — the render-vs-extraction critic

*(Deep-dive follow-up to §3.3: how the crawler itself improves, beyond bandit acquisition.)*

**The key idea: every page ships its own ground truth.** The rendered screenshot is what a human sees; the extracted interleaved document is what the model will eat. Any divergence between the two is a pipeline bug — and detecting it requires **no labels**, because the page supplies both views. This is render-and-compare turned inward: instead of verifying *model outputs* against a rendering, you verify *the pipeline's output* against the rendering. It's the same Generate → Verify → Update loop applied to infrastructure, with a level-2 (cross-modal consistency) verifier.

**The critic.** A VLM critic takes (a) full-page screenshot tiles, (b) the extracted interleaved doc, (c) optionally the DOM, and emits a *structured* discrepancy report — taxonomy code, severity, and localization (screenshot bounding box ↔ doc span). The taxonomy:

| Code | Failure | Typical cause |
|---|---|---|
| **MISSING** | visible in screenshot, absent from doc | lazy-load images, JS-rendered text, canvas-drawn charts |
| **PHANTOM** | in doc, invisible on page | nav/boilerplate leakage, cookie banners, hidden SEO text |
| **ORDER** | wrong reading order | multi-column layouts, caption attached to the wrong image |
| **FIDELITY** | present but corrupted | mangled tables, OCR errors, wrong chart→data extraction |
| **COVERAGE** | page truncated | infinite scroll, pagination, tabs/accordions |
| **SEMANTIC** | image↔text misaligned | alt-text describes a different image, fused caption drifted |

**Stage attribution — which component is guilty?** Three cheap diffs localize the fault before any fix is attempted: DOM-declared resources vs what actually painted (→ **rendering** bug); DOM text vs doc text (→ **extraction** bug); screenshot vs synthetic captions (→ **labeling** bug). Without this, every discrepancy report starts a cross-team argument; with it, reports route themselves.

**Loop A — improve the crawler (the component):**
1. **Template-level rule patching.** Cluster discrepancy reports by DOM-template hash; a code-gen model writes or patches per-template extraction adapters; a golden regression set (screenshot, verified-extraction pairs) gates every patch. High-frequency templates get fixed first — blast-radius-ordered triage.
2. **Screenshot-grounded extractor.** The critic doesn't just flag — it *proposes the corrected doc from the screenshot*. Distill those corrections into the extraction model. End state: extraction stops being DOM heuristics audited by a VLM and becomes **a VLM that reads the page image with the DOM as a hint** — trained almost entirely by cycle-consistency with its own pipeline.
3. **Render-policy bandit.** Discrepancies attributed to rendering (missing lazy-loads, unsettled layouts) become the reward signal for the render controller: scroll/settle policies, wait budgets, interaction probes — optimized per critic-score-per-render-dollar.
4. **Acquisition feature.** Per-domain *extraction fixability* feeds back into the crawl policy: don't keep buying pages you can't correctly extract.

**Loop B — improve the data (the product):**
1. **Quarantine & repair.** Severity-weighted: re-render with a better policy, re-extract with the patched adapter, or accept the critic's corrected doc directly (with the P15 caveat below).
2. **Blast-radius backfill.** Every doc carries lineage metadata (extractor version, template hash, render config). When a systematic bug surfaces — say, captions attached to the wrong image across one template family — you query the corpus for affected docs and reprocess retroactively. **Lineage is what makes pipeline bugs reversible;** without it every bug is permanent corpus damage.
3. **The taxonomy becomes the pipeline's eval.** Confirmed failures graduate into an adversarial golden suite every extractor release must pass; the discrepancy-rate histogram per template/domain/version is the pipeline's dashboard.

**The new problems this creates (P11–P15):**
- **P11 — Shared blind spot.** If the *render* is broken (lazy-load never fired), screenshot and doc agree — and both are wrong. The critic sees consistency, not truth. Fix: multi-render disagreement probes (two renders under different policies; divergence flags a rendering issue) and DOM-declared-resource checks (an `<img>` with a src that never painted is a smoking gun no screenshot comparison can see).
- **P12 — Critic cost at corpus scale.** Frontier-critic every page and the economics die. Same answer as before: a distillation cascade — the big critic runs on stratified samples and trains a cheap discrepancy *predictor* that runs on everything; sampling is then biased toward predicted-error. (The T3→T0 cascade of §3.2 gains a parallel critic column.)
- **P13 — Critic false positives on interactive content.** The critic flags "missing" content that is *reachable but not visible* — behind tabs, accordions, hover states. You need the visible/reachable distinction in the taxonomy, and you must measure critic **precision on human-audited goldens before wiring it to automation**: a noisy critic auto-patching extraction rules is a self-inflicted DDoS on your own pipeline.
- **P14 — Ordering is sometimes genuinely ambiguous.** Card grids and dashboards have no canonical reading order. Don't force one: mark ambiguity explicitly and train downstream with order-tolerant objectives (or layout tokens) rather than punishing the extractor for an unanswerable question.
- **P15 — Goodhart, round three.** An extractor distilled from critic corrections inherits the critic's blind spots — the same shape as P2 (scorer style bias) and P9 (policy hacking the scorer). Same fix shape too: refresh the critic from the newest frontier model each generation, and keep a human-audited golden set that never touched any critic as the acceptance test.

**Why this framing wins in an interview:** it upgrades the scraper story from "a bandit that picks URLs" to a **self-verifying system in which the page itself is the supervisor** — and it closes a second, deeper flywheel: better VLM → sharper critic → better extractor → cleaner interleaved data → better VLM. The data engine bootstraps its own perception.

### §3.7 Field notes — details only doers know

*The texture that separates "I read about data engines" from "I ran one." Grouped by where the scar tissue is.*

**Measurement discipline**
- **Paired seeds or it didn't happen.** At 1B proxy scale, most public VLM benchmarks carry 1–3 points of seed noise. A "+2 win" that evaporated under paired-seed reruns once killed a quarter's roadmap. Every proxy A/B: same seed, same data order, only the candidate batch swapped. Error bars on everything, or the meeting is fiction.
- **Early-signal probes beat benchmarks at proxy scale.** A 2k-item exact-match OCR probe, an entity-cloze set, and per-domain perplexity slices detect data effects that ChartQA (noisy, 2.5k items) cannot see at 1B. Maintain an explicit probe→benchmark mapping, validated on the scaling ladder — the probes are the instrument panel; benchmarks are the quarterly report.
- **The insertion protocol changes the answer.** The same candidate batch measured *mixed into* pretraining vs *inserted in the annealing slot* gives different — sometimes opposite-sign — value estimates. Standardize one protocol (we fixed a 20%-of-anneal insertion slot) or no two measurements are comparable.

**The ledger**
- **The mixture you specify is not the mixture you train on.** Dedup once deleted 80% of one domain, silently re-weighting the whole mixture downstream of the config file. Only post-filter token accounting — the **data ledger** — is truth. Recompute mixture weights after *every* pipeline change; diff the ledger, not the config.
- **Watch the score histogram, not just the evals.** A thresholded scorer puts the most ambiguous pages exactly at the cut, so small scorer drift moves millions of pages across it. A silent scorer retrain once dropped intake volume 30% for a week before anyone noticed — the downstream evals wouldn't have shown it for a month. Alarm on intake volume per content type and score-distribution drift (KS test against last week's histogram).

**The render farm**
- **The p99 kills you, not the median.** Median render 1.2s; p99 45s — JS-heavy pages, crypto-miner scripts pegging CPU, one SVG map page with 300k DOM nodes stalling extraction for minutes. Survival kit: recycle browser processes every N pages (headless Chrome leaks), hard DOM-node caps, a kill-list for known CPU-trap pages, per-page compute budgets enforced by timeout tiers.
- **Keep canary pages.** A per-domain set of known-stable pages, re-rendered daily, is the only reliable way to distinguish "the site redesigned" from "our IP range got blocked." Without canaries, both look like extraction-failure spikes and you debug the wrong layer for a day.
- **The web is nonstationary — as a monitoring requirement, not a philosophy.** A major CMS platform update once shifted the template distribution of a double-digit percentage of the crawl overnight; extraction failure spiked while every service was "healthy." Template-distribution drift alarms belong next to loss curves.

**Data content traps**
- **Alt-text is secretly load-bearing for entities.** A 100%-synthetic-caption slice improved VQA but cratered proper-noun retrieval (landmarks, products, people) — invisible in aggregate metrics, caught only by the entity-cloze probe. The ~40% raw-caption fusion ratio wasn't taste; it came from a 0/20/40/60/80 sweep against that probe.
- **The axis-label trap.** Chart→data extraction fails *silently* on log axes and truncated axes — the extracted numbers are fluent and wrong, teaching the model wrong magnitudes at scale. Rule: only keep chart QA pairs whose numbers cross-check against the page text or an underlying data table. Fluent-and-wrong is worse than absent (same theorem as P3).
- **Paraphrase contamination rides the scrape date.** One "improvement" traced to crawl recency: newer pages *discussed* a benchmark (paraphrased items, worked solutions in blog posts). N-gram decontam missed it entirely; embedding decontam caught half. Date-stratified ablations are part of the decontamination kit, not an optional extra.

**Humans and constraints**
- **Reviewer-minutes are a budget line.** Better safety filters surface *more* borderline cases, so the gray-zone human-review queue grows superlinearly as filters improve. Plan reviewer-minutes per million pages as a first-class resource or the pipeline stalls on its human lane while every machine dashboard shows green.
- **Politeness-aware optimization.** Crawl-rate limits mean the bandit's best arms are drip-fed — a high-value domain at 1 req/s is worth less than its score implies. If the acquisition optimizer doesn't see politeness constraints inside the value estimate, it spends its budget standing in queues.

---

*Cross-links: Day3 (data & eval), Day6/Post_Training_and_RL_Deep_Dive (RLVR, reward hacking — P9 is the same math), VLM_VLA_Unified_Omni (encoder families the data feeds), Proposal_JEPA_Latent_VLM (latent-space objectives as the model-lever complement to this data lever).*
