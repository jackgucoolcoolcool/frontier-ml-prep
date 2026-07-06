# Resume deep-dive verbal mock sessions — Google/Gemini topics only.
#
# Format: each stage is a QUESTION, not a coding task. You answer OUT LOUD
# (3-5 minutes), the interviewer presses with follow-ups, then you reveal
# what a strong answer covers and self-score 1-5. Scores are logged per
# topic and surface in the gap report.
#
# The "strong" guides describe the STRUCTURE and content areas a strong
# answer hits, anchored to public knowledge — fill them with YOUR real
# specifics, spoken at a confidentiality-safe abstraction level (public
# tech-report level: approaches and trade-offs, not unpublished numbers,
# codenames, or roadmap).

ANSWER_FRAME = ("Frame every answer: claim → mechanism → trade-off you weighed → "
                "a concrete anchor (scale, metric, or result you can share) → what you'd do next.")

VERBAL_SESSIONS = [

{
  "id": "r1",
  "mode": "verbal",
  "interviewer": "R",
  "topic": "mm-pretraining",
  "title": "Resume: Multimodal pretraining",
  "minutes": 40,
  "persona": "Research deep-dive — your core Gemini pretraining story. Expect an architecture-minded interviewer (the S round) to press on design decisions.",
  "intro": "Thanks for making time. I've read your resume — you've been on multimodal pretraining since Gemini 1.0. I want to go deep on that today. Answer out loud, 3-5 minutes per question; I'll push back like a real interviewer. " + ANSWER_FRAME + " Keep specifics at tech-report abstraction — practicing WHERE that line is, is part of the drill.",
  "closing": "That's a strong core story when the decisions are yours and the numbers are concrete. Tighten wherever you scored under 4.",
  "stages": [
    {
      "id": "r1q1",
      "title": "Opening narrative (2-minute version)",
      "prompt": "Walk me through your multimodal pretraining work: the overall system at a high level, then zoom into the part that was **yours** — a decision you drove, and how it turned out.",
      "followups": [
        "Of everything you just described, which decision was yours alone versus the team's consensus? What did people disagree about?",
        "If that decision had gone the other way, what would the model look like today?"
      ],
      "strong": "A 2-minute pyramid: one sentence of context (native multimodal model, vision+text+more trained jointly), one sentence on your scope (visual understanding pretraining across all Gemini versions), then ONE concrete decision story with stakes — e.g., a data-mixture call, an encoder/tokenization choice, or a resolution/tiling strategy — with the alternative considered, why you chose, and a measurable outcome. Interviewers grade ownership language: 'I decided/measured/was wrong about', not 'we did'. End with one sentence of what you'd change now. The classic failure is touring the org chart for 5 minutes without ever landing on a decision."
    },
    {
      "id": "r1q2",
      "title": "Vision input design: encoder, tokens, resolution",
      "prompt": "Let's go one level down. How should a frontier VLM ingest images: separate pretrained vision encoder vs native patches into the transformer? How do you handle resolution and token budget? Argue the trade-offs as if we were designing from scratch today.",
      "followups": [
        "When does a contrastively-pretrained encoder (CLIP/SigLIP-style) actually beat training vision from scratch inside the LLM — and when does it become the bottleneck?",
        "OCR on a dense document: walk me through exactly where a fixed 224px encoder fails and what tiling (or native-resolution) buys you, in tokens and FLOPs."
      ],
      "strong": "Covers: (1) the design space — frozen/tuned CLIP-style encoder + adapter (Flamingo, BLIP-2, PaLI) vs patches-direct (Fuyu-style) vs hybrid; (2) why contrastive init wins at small scale (semantic prior for free) but can ceiling: trained on web alt-text semantics, weak at dense text/layout; (3) resolution economics — tokens per image scale ~quadratically with side length; tiling (sub-images + thumbnail) vs NaViT-style variable resolution; token budgets (~hundreds per image) vs information density for OCR; (4) interference: image tokens consume sequence budget and attention; (5) a stated position with a scale caveat ('at frontier compute I'd bet on X because...'). Bonus: video (temporal sampling) and the audio tower as analogous decisions."
    },
    {
      "id": "r1q3",
      "title": "Data mixture and curriculum",
      "prompt": "You created the multimodal data sourcing strategy — webpages, PDFs, books, agentic traces, acquired sources. How do you actually set the mixture ratios across modalities and sources, and how do you decide when to re-weight?",
      "followups": [
        "Concretely: you double the image-text weight and text-only benchmarks drop half a point. What do you do — and who do you have to convince?",
        "How do you know when a source is worth acquiring versus synthesizing versus scraping more of what you have?"
      ],
      "strong": "Covers: (1) mixtures set empirically — small-scale mixture ablations + scaling-law extrapolation, not intuition; (2) the guardrail metric suite: text-only regressions are the political and technical constraint on adding MM data (catastrophic-forgetting risk); (3) quality-vs-quantity: dedup and filtering often beat more tokens (data-constrained scaling results — repeating good data ~4 epochs is near-free, beyond that decaying returns); (4) source valuation: marginal-value-per-token experiments — hold mixture fixed, swap in candidate source, measure target capabilities; (5) curriculum: staging resolution/task complexity, annealing high-quality sources late; (6) a real re-weight story with the trade-off you accepted. The follow-up wants a decision process (bisect the regression, quantify the trade, escalate with data) not a shrug."
    },
    {
      "id": "r1q4",
      "title": "War story: a run going wrong",
      "prompt": "Tell me about a time a large multimodal pretraining run misbehaved — loss spike, modality imbalance, a capability mysteriously tanking mid-run. How did you find it and what was the fix?",
      "followups": [
        "What monitoring did you have BEFORE the incident, and what did you add after?"
      ],
      "strong": "A real debugging narrative with instruments: per-modality/per-source loss decomposition, gradient norms per tower, checkpoint rollback + data bisection, skipped-batch/spike-skip machinery, numerics (bf16 overflow in attention logits, softmax saturation), data poisoning from a bad shard or a decoding bug in one source pipeline. Strong answers name the false hypothesis they chased first — that's what makes it credible — and end with the systemic fix (a canary eval, a data validator, an alert), not just the one-off patch. If your real story is confidential, practice telling it at the level of mechanism without the identifying details."
    },
    {
      "id": "r1q5",
      "title": "Research taste: 10× vision compute",
      "prompt": "Last one: if I gave you 10× the compute budget for visual understanding pretraining — no new headcount, no new data deals — what's the first experiment you run, and what result would make you scale it to the hero run?",
      "followups": [
        "Why is that the highest-leverage experiment rather than just scaling what already works?"
      ],
      "strong": "One crisp hypothesis (not a portfolio): stated as 'I believe X because of evidence Y; the experiment isolates it by Z; the go/no-go metric is W at N scale.' Good candidates you can argue: native-resolution vision at scale, much longer visual context (video/document-length), generation-as-auxiliary-objective for understanding, or data flywheel (model-curated data) at 10× curation compute. The follow-up tests whether you understand opportunity cost — the answer should acknowledge what the 10× is NOT spent on and why the information value of the experiment beats certain-but-incremental scaling."
    }
  ]
},

{
  "id": "r2",
  "mode": "verbal",
  "interviewer": "R",
  "topic": "mm-scaling",
  "title": "Resume: Multimodal scaling laws",
  "minutes": 40,
  "persona": "Research deep-dive — scaling laws for hero compute, laddering, single-change ablations. Your highest-leverage topic for an architecture-minded interviewer.",
  "intro": "Your resume says you co-developed scaling laws for optimal hero compute, laddering, and multimodal single-change ablation. That's exactly my area of skepticism, so I'll push hard here. Out loud, 3-5 minutes each. " + ANSWER_FRAME,
  "closing": "Scaling-law answers are strongest when you volunteer the failure modes before I ask. Re-run anything under 4.",
  "stages": [
    {
      "id": "r2q1",
      "title": "Methodology from scratch",
      "prompt": "Walk me through how you'd build a scaling law to size a frontier multimodal training run: what you sweep, what you fit, and how you get from the fit to an actual (N, D, mixture) decision for hero compute.",
      "followups": [
        "IsoFLOP profiles or a parametric L(N, D) fit — which did you use and why? How many runs, at what scales, did the fit actually take?",
        "How does the vision tower enter the compute accounting? Is an image token a token?"
      ],
      "strong": "Covers: (1) the two classic methods — IsoFLOP curves (fix C, sweep N, find the valley) vs parametric Chinchilla-style L = E + A/N^α + B/D^β fit — and their failure modes (IsoFLOP needs many runs per compute slice; parametric fits are sensitive to the low-compute points and the fitting procedure — the Chinchilla-vs-Kaplan discrepancy came partly from LR-schedule/fitting artifacts); (2) MM specifics: compute accounting with heterogeneous towers, image tokens ≠ text tokens in information content, mixture as an extra axis you either fix or fit per-modality curves; (3) from fit to decision: predicted optimal N,D at hero FLOPs + inference-cost correction (overtraining smaller models on purpose, Llama-style); (4) error bars — extrapolating 2-3 orders of magnitude means you backtest: fit small, predict medium, verify, THEN trust."
    },
    {
      "id": "r2q2",
      "title": "What's different about multimodal",
      "prompt": "How do multimodal scaling laws differ from text-only ones? Where does the clean Chinchilla picture break when images and video enter?",
      "followups": [
        "Is there transfer? If I add image data, does text loss follow a predictable curve — or is the interaction unmodelable?",
        "What do you do when the capability you care about (say OCR or spatial reasoning) doesn't move smoothly with loss?"
      ],
      "strong": "Covers: (1) mixture as a first-class dimension — per-modality losses with cross-terms; naive single-loss laws hide modality trade-offs; (2) data-constrained regimes hit earlier for some sources (high-quality paired data is finite; epoching enters, à la data-constrained scaling laws); (3) transfer/interference: some MM data helps text (documents, code screenshots) while some displaces it — you measure marginal curves, not assume; (4) eval noise: downstream MM benchmarks are noisier and more saturating than perplexity, so you need smooth proxy losses per capability (e.g., loss on an OCR-heavy held-out slice as OCR proxy) and you validate proxy→capability correlation; (5) 'emergence' framed properly: often a metric artifact (discontinuous accuracy on top of smooth log-likelihood — the Schaeffer critique), which is exactly why proxies work."
    },
    {
      "id": "r2q3",
      "title": "Laddering and single-change ablation",
      "prompt": "Explain your laddering methodology: how a change gets validated from small scale up to a hero-run decision. Design the ladder for one concrete example — say, swapping the image tokenizer or changing the vision-token budget per image.",
      "followups": [
        "A change wins at your two smallest ladder rungs and loses at the third. Ship it to hero or kill it? Walk me through the actual decision.",
        "How do you keep single-change discipline when twenty teams all want their change in the next run?"
      ],
      "strong": "Covers: (1) ladder design: 3-5 rungs spanning ~2-3 orders of compute, each rung trained to compute-optimal (not fixed steps — undertrained rungs reverse rankings), same data/eval protocol; (2) you extrapolate the TREND (does the delta grow, shrink, cross zero with scale), not the point estimate — with a variance estimate from seed reruns at small rungs; (3) the crossing case: fit delta-vs-scale, weigh predicted hero delta against risk; often the honest answer is 'run one more rung' and the real constraint is calendar time; (4) interaction risk: single-change ablation is the whole point — combined changes get a combined rung before hero; (5) process reality: a change-review ritual, pre-registered metrics, and someone empowered to say no. This is where staff-level judgment shows: compute, calendar, and politics all priced in."
    },
    {
      "id": "r2q4",
      "title": "Choosing the loss that the law predicts",
      "prompt": "Scaling laws predict a loss. Loss on what, exactly? How do you choose the validation distribution so that the law's prediction actually tracks what leadership cares about?",
      "followups": [
        "Your law says the hero run will hit its loss target, and it does — but the flagship capability misses. What went wrong in your setup?"
      ],
      "strong": "Covers: (1) held-out slice matched to the TRAINING mixture predicts training health but not product value — you also need capability-weighted eval slices (OCR-heavy, spatial, chart/document, video) chosen to mirror the roadmap; (2) contamination discipline on those slices (dedup val from train, else the law flatters itself); (3) the miss scenario: loss-capability mapping shifted — either the capability needed a data ingredient that scaled sublinearly in the mixture, or it's bottlenecked by something loss doesn't see (tokenizer/resolution ceiling, eval prompt format); the fix is per-capability proxy losses validated against the capability across past ladder runs; (4) honesty about what laws CAN'T predict: post-training deltas, RL gains, tail behaviors."
    },
    {
      "id": "r2q5",
      "title": "The skeptic",
      "prompt": "Here's my honest position: scaling 'laws' are curve-fits on 6 points with the x-axis on a log scale — of course they look straight. Why should I authorize nine figures of compute on your extrapolation?",
      "followups": [
        "Tell me about a time your extrapolation was wrong. What was the post-mortem?"
      ],
      "strong": "The best answer AGREES first: they are empirical regularities, not physics; Kaplan→Chinchilla proved the fits can be systematically off (LR schedule, fitting range). Then the defense: (1) backtesting — hold out your largest completed run, fit on the rest, show prediction error; (2) they're decision tools, not oracles — the alternative isn't certainty, it's sizing nine figures by vibes; the law needs to beat the null of 'copy last run's ratios', which it demonstrably does; (3) risk management — laddered de-risking en route, canary evals early in the hero run with pre-agreed abort/adjust criteria. A real 'we were wrong' story (data-constrained source hit an epoch wall, mixture interaction, val contamination) with the process fix lands this at staff level. Refusing to concede any weakness fails the question."
    }
  ]
},

{
  "id": "r3",
  "mode": "verbal",
  "interviewer": "R",
  "topic": "data-science",
  "title": "Resume: Science of Data",
  "minutes": 40,
  "persona": "Research deep-dive — dedup, filtering, embedding/clustering, training-data attribution, self-improving data.",
  "intro": "Your resume has a 'Science of Data' line: embedding, clustering, dedup, filtering, attribution, self-improving data. Most candidates hand-wave data work — I want the mechanisms. Out loud, 3-5 minutes each. " + ANSWER_FRAME,
  "closing": "Data answers convince when they include a measured effect size. Note where you had none.",
  "stages": [
    {
      "id": "r3q1",
      "title": "Dedup at web scale",
      "prompt": "How do you deduplicate a web-scale multimodal corpus? Exact, near, and cross-modal — what's the machinery, and what's the measured payoff?",
      "followups": [
        "Image dedup specifically: what counts as a duplicate — same file, same pixels, same content re-encoded, same photo cropped? Where do you draw it and why?",
        "How much can you REPEAT good data before it hurts? How do you know?"
      ],
      "strong": "Covers: (1) text: exact hash → MinHash/LSH shingling for near-dup docs; (2) images: perceptual hashing for re-encodes/crops, embedding-similarity thresholds for semantic near-dups — with the honest note that the threshold is a policy decision (same-scene photos: dup or diversity?); (3) cross-modal: same image + different alt-text, boilerplate captions repeated millions of times; (4) payoffs you can cite: dedup reduces memorization/regurgitation risk and improves quality-per-token (C4/CCNet lineage, Falcon/RefinedWeb ablations); val-set dedup is also an eval-integrity requirement; (5) repeats: data-constrained scaling results — up to ~4 epochs of good data ≈ fresh, decaying value beyond; quality-weighted repeat beats one epoch of junk. A concrete before/after from your pipeline seals it."
    },
    {
      "id": "r3q2",
      "title": "Quality filtering without killing the tail",
      "prompt": "Describe your quality-filtering stack — heuristics, classifiers, model-based scoring. And the hard part: how do you verify the filter isn't quietly deleting tail capabilities you'll need next year?",
      "followups": [
        "Your new filter improves headline benchmarks but a small OCR eval on receipts regresses. What's the process from that signal?"
      ],
      "strong": "Covers: (1) the stack in layers: cheap heuristics (length, lang-ID, boilerplate) → trained quality classifiers (risk: they encode 'looks like Wikipedia' bias) → LLM-as-scorer on samples (expensive, calibration drift); for images: resolution/watermark/aesthetics vs document-utility tension — aesthetic filters are actively harmful for OCR/chart data; (2) tail protection: capability-sliced evals BEFORE/after every filter change, stratified sampling audits of what got removed, per-cluster removal-rate dashboards (this is where your embedding/clustering infra earns its keep); (3) the regression process: characterize removed set in the receipt cluster, loosen or carve out that stratum, add the eval to the pre-merge gate; (4) philosophy: filters are trained on proxies of today's taste — version them, keep the removed data, make removal reversible."
    },
    {
      "id": "r3q3",
      "title": "Embedding, clustering, and what it changed",
      "prompt": "You built MM data embedding/clustering/visualization infrastructure. What decisions did it actually change? Give me one concrete story where looking at the map changed the mixture or the sourcing plan.",
      "followups": [
        "How do you validate that clusters are meaningful rather than embedding artifacts?"
      ],
      "strong": "Covers: (1) the machinery briefly (embed with a strong MM encoder, cluster hierarchically, project for visualization, topic-label clusters with an LLM); (2) the point is the DECISIONS: coverage-gap discovery ('we're rich in memes, poor in engineering diagrams'), source overlap quantification before an acquisition ('60% of the candidate corpus sits in clusters we already saturate'), targeted sourcing/synthesis for thin clusters that map to roadmap capabilities; (3) validation: intruder tests, cluster purity against labeled slices, and the operational test — does upweighting cluster X move the correlated eval? (4) one story told with numbers-shaped anchors ('a double-digit percent of source Y was redundant; we renegotiated/skipped'). Infrastructure without a decision story reads as a tooling hobby — always land the 'so what'."
    },
    {
      "id": "r3q4",
      "title": "Training-data attribution",
      "prompt": "Training-data attribution at frontier scale: what's actually tractable, and what did you use it for?",
      "followups": [
        "Influence functions don't scale naively to frontier models. What approximations make attribution usable, and what do you give up?"
      ],
      "strong": "Covers: (1) the tractable toolbox: gradient-similarity methods (TracIn-style checkpoint-sampled inner products), datamodel/regression approaches on data subsets, and pragmatic proxies — n-gram/embedding retrieval against the corpus to find likely sources of a behavior; (2) what's given up: convexity assumptions are false, single-example influence is noisy — batch/cluster-level attribution is the honest granularity; (3) use cases: debugging weird behaviors back to a data shard, valuing sources for renewal/acquisition decisions, memorization/copyright triage, and steering the self-improving-data loop (find what data made the model better on capability X, get more of it); (4) a workflow reality: attribution generates hypotheses; the confirmatory tool is still a counterfactual retrain or an ablated mixture rung on the ladder."
    },
    {
      "id": "r3q5",
      "title": "The data flywheel and model collapse",
      "prompt": "Self-improving data: models filtering, captioning, and generating their own training data. How do you build that flywheel so it compounds instead of collapsing?",
      "followups": [
        "Where is synthetic data clearly working today, and where is it clearly dangerous?"
      ],
      "strong": "Covers: (1) the loop's safe modes: model-as-curator (rating/filtering real data — lowest collapse risk), model-as-annotator (recaptioning images with dense, accurate captions — proven big wins for VLM training), model-as-generator (highest risk, needs grounding); (2) collapse mechanics: distribution narrowing, error amplification, tails vanishing when generations feed back unfiltered (the model-collapse literature) — mitigations: always anchor on fresh human data, verify synthetic with independent checkers (verifiable domains: code execution, math checking, render-and-compare for derendering), provenance-tag everything, cap synthetic fractions per mixture and ablate them on the ladder; (3) works today: recaptioning, verifiable reasoning traces, targeted long-tail synthesis (rare scripts, charts); dangerous: open-ended prose/images recycled at scale; (4) measurement: track diversity metrics and tail-capability evals, not just headline scores."
    }
  ]
},

{
  "id": "r4",
  "mode": "verbal",
  "interviewer": "R",
  "topic": "vision-capabilities",
  "title": "Resume: Visual & physical understanding",
  "minutes": 40,
  "persona": "Research deep-dive — derendering, OCR, 2D spatial, 3D physical intelligence, and the evals that measure them.",
  "intro": "You own the visual and physical understanding roadmap — derendering, OCR, 2D spatial, 3D physical intelligence. Let's test how deep the ownership goes. Out loud, 3-5 minutes each. " + ANSWER_FRAME,
  "closing": "Capability answers need: definition, data recipe, eval, frontier. Re-drill any question missing one of the four.",
  "stages": [
    {
      "id": "r4q1",
      "title": "Deep-dive one capability end to end",
      "prompt": "Pick ONE — derendering, OCR, or spatial understanding — and take me end to end: precise definition, why it's hard, the data recipe, how you eval it, and where the frontier is right now.",
      "followups": [
        "For derendering specifically: what's the output representation — SVG, code, DOM, layout JSON? How does that choice change the data you can synthesize and the eval you can trust?"
      ],
      "strong": "The gold-standard structure. E.g. derendering: image → faithful structured representation (markup/code/vector) — inverse graphics for documents/UIs/charts; hard because it's dense, exact, and compositional (one wrong coordinate ≠ slightly wrong, it's broken); data recipe is the render-invert trick — synthesize structure, render it, train the inverse — giving unlimited perfectly-labeled pairs, plus real-world pairs (webpage screenshots ↔ DOM) with noise; eval: round-trip fidelity (re-render and compare pixels/perceptually) beats string match on the markup — string metrics punish equivalent representations; frontier: long dense outputs, faithfulness under clutter, generalizing past the synthetic renderer's style distribution. Whatever capability you pick: definition → hardness → data → eval → frontier, with your actual contribution flagged."
    },
    {
      "id": "r4q2",
      "title": "Why VLMs are bad at space, and what works",
      "prompt": "Frontier VLMs still fumble counting, left-of/right-of relations, and metric geometry. Why, mechanistically? And which of the fixes have you found actually move the needle — dense-prediction auxiliary tasks, coordinate tokens, synthetic geometry data, visual chain-of-thought?",
      "followups": [
        "How do you evaluate 3D *physical* intelligence — stability, contact, dynamics — without a robot in the loop?"
      ],
      "strong": "Covers: (1) mechanisms of failure: contrastive/caption pretraining rewards bag-of-concepts semantics, not geometry; aggressive downsampling destroys metric detail; language priors override visual evidence (the model answers from plausibility); attention pooling loses positional precision; (2) what moves it: dense supervision (detection/segmentation/depth as auxiliary or as data — grounding tokens with coordinates), high-resolution/tiled inputs, synthetic data with exact geometric labels at scale, and visual CoT (point-then-answer, draw-then-reason) which converts implicit geometry into explicit tokens; (3) 3D-without-robots eval: simulation (physics engines render scenarios with ground-truth outcomes — stability, collision, occlusion), video forward-prediction QA ('what happens next'), multi-view consistency checks, and human-labeled real photo sets for sim-to-real anchoring; the honest caveat: sim evals overstate transfer, so you triangulate."
    },
    {
      "id": "r4q3",
      "title": "Eval design that survives contact with progress",
      "prompt": "You have to measure these capabilities for every model release. How do you design evals that don't saturate in six months, resist train-set contamination, and still correlate with what users feel?",
      "followups": [
        "Auto-raters vs human raters for visual tasks — where does LLM-as-judge break?"
      ],
      "strong": "Covers: (1) anti-saturation: difficulty ladders with headroom by construction (parametric generators that can always emit harder instances — more objects, finer fonts, deeper nesting), report full difficulty curves not single scores; (2) contamination: private held-out splits, freshly generated instances per release, canary strings, dedup of eval against training corpus as a release gate; (3) validity: correlate offline evals against human preference / task success on real product traffic; kill evals that stop correlating; (4) judge limits: LLM-judges inherit the same visual blind spots being measured — fine for text-side rubric checking, unreliable for 'is the bounding box right' or dense OCR fidelity; use programmatic checkers where output is structured (render-and-compare, exact-match fields) and calibrated human panels for the rest; (5) the meta-point: evals are a product with a roadmap and an owner, not a static file."
    },
    {
      "id": "r4q4",
      "title": "Capability interference",
      "prompt": "You crank up dense OCR/document data and your natural-image and video scores dip. Is that real interference or a mixture artifact? How do you diagnose it and what are the levers?",
      "followups": [
        "When do you accept a targeted post-training fix versus insisting it be solved in the pretraining mixture?"
      ],
      "strong": "Covers: (1) diagnosis first: is it displacement (fixed token budget — OCR share grew, natural-image share shrank) or true interference (same share, worse transfer)? Rerun with budget held constant per slice; check per-capability loss curves, not just end evals; (2) levers: total budget growth, resolution routing (dense data at high res, natural images cheaper), staging (capability-heavy data late in training vs annealed), architectural relief (capacity via MoE experts effectively specializing); (3) the post-training question is roadmap judgment: post-training fixes are fast, cheap, and brittle (regression risk on the next base model, limited headroom); pretraining fixes are slow but compound — rule of thumb: perception primitives belong in pretraining, task formats and style in post-training; (4) admit the honest reality: some trade-offs remain trade-offs, and someone (you) has to own the priority call."
    },
    {
      "id": "r4q5",
      "title": "Roadmap taste: the underrated capability",
      "prompt": "Eighteen-month horizon: what's the most underrated visual capability — the one that unlocks the most downstream value relative to attention it gets — and what's your concrete plan for it?",
      "followups": [
        "Who disagrees with you on this, and what's their best argument?"
      ],
      "strong": "Graded on argument quality, not the pick. Structure: name the capability (defensible picks: derendering/screen-to-structure as the substrate for computer-use agents; long-horizon video understanding as the substrate for physical/world reasoning; precise spatial grounding as the substrate for robotics and AR) → why underrated (misaligned incentives: it's not a headline benchmark, it's an enabler — its value shows up in OTHER teams' metrics) → the plan: data (synthesis pipeline), eval (build the missing benchmark first — evals summon effort), model lever, one product proof-point → and the steelmanned counter-argument answered honestly. Naming who disagrees and conceding their best point is the staff-level move; a plan without an eval is the classic miss."
    }
  ]
},

{
  "id": "r5",
  "mode": "verbal",
  "interviewer": "R",
  "topic": "mmu-mmgen",
  "title": "Resume: Understanding ↔ Generation",
  "minutes": 40,
  "persona": "Research deep-dive — MMU4MMGen and MMGen4MMU joint ventures: self-critique, visual thinking, unified vs. separate models.",
  "intro": "The understanding-generation flywheel — MMU4MMGen and MMGen4MMU — is the most research-y line on your resume, so expect the most pushback here. Out loud, 3-5 minutes each. " + ANSWER_FRAME,
  "closing": "These answers win on mechanisms and measurables, lose on vision-talk. Check your scores honestly.",
  "stages": [
    {
      "id": "r5q1",
      "title": "Understanding improving generation",
      "prompt": "MMU4MMGen: how does an understanding model concretely improve an image/video generation model? Walk me through the mechanisms — self-critique, grounded prompting, reward modeling — and where each actually bites.",
      "followups": [
        "Design the critic-in-the-loop for me: what does the critic score, when does it run (training vs inference), and how do you stop the generator from learning to fool it?"
      ],
      "strong": "Covers the mechanism menu with where-it-bites for each: (1) data-side — VLM recaptioning of training images (dense, accurate captions demonstrably improve text-image alignment; the DALL-E 3 recipe made this public); VLM filtering of low-quality pairs; (2) training-side — VLM as reward model for RL/preference-tuning of the generator (prompt faithfulness, count/attribute/relation correctness — exactly what aesthetic scores miss); (3) inference-side — critique-and-iterate loops: generate, VLM checks against prompt, targeted regenerate/edit; grounded generation via understanding-derived structure (layout/boxes/sketch/code as an intermediate the generator conditions on — the 'grounded coding-sketching' idea); (4) anti-reward-hacking: ensemble/rotate critics, keep held-out human eval as the arbiter, adversarial audits of high-reward samples, cap KL/update size — the RM-overoptimization playbook transplanted. Anchor with a measurable: alignment/faithfulness evals (spatial-relation and counting suites), not vibes."
    },
    {
      "id": "r5q2",
      "title": "Generation improving understanding",
      "prompt": "Now the reverse. 'Visual thinking' and dense prediction from generative models improving understanding — make it concrete. When does the ability to generate actually help a model answer a question it would otherwise miss?",
      "followups": [
        "Isn't this just chain-of-thought with pixels — an inference-time crutch? What's the evidence it teaches the model anything durable?"
      ],
      "strong": "Covers: (1) the mechanisms: generation as world model (predict-what-happens-next supervision teaches dynamics/3D structure that captioning never does — video prediction as physics pretraining); analysis-by-synthesis (imagine candidate scenes, compare against the image — useful for occlusion/counterfactual questions); visual scratchpads (model draws/edits an intermediate — marks the objects it's counting, sketches the geometry — externalizing spatial state like CoT externalizes logic); dense prediction (depth/segmentation heads) as geometry-forcing auxiliary losses; (2) the follow-up's honest answer: inference-time visual CoT IS a crutch in the same way text CoT is — and that's fine (test-time compute); the durable claim needs distillation evidence (train on your own visual-thinking traces, measure no-scratchpad improvement) or representation probes showing better geometry after generative pretraining; concede where evidence is thin — this is an active bet, not settled science; (3) a crisp example from your work at safe abstraction."
    },
    {
      "id": "r5q3",
      "title": "Unified vs. separate models",
      "prompt": "The architecture question underneath the whole venture: one unified model that both understands and generates, or specialist models with clean interfaces? Argue it properly — tokenizers, objectives, interference, serving.",
      "followups": [
        "If unified: AR over discrete visual codes, or a diffusion head bolted onto the LLM? Defend one.",
        "What does the interference between generation and understanding objectives actually look like in training?"
      ],
      "strong": "Covers: (1) unified pros: shared world knowledge, in-context multimodal chains (understand→reason→generate in one pass), one serving stack, the flywheel internalized; cons: objective interference, generation quality historically lags specialists, tokenizer compromise; (2) separate pros: each SOTA, independent iteration cadence and teams; cons: interface bottlenecks (text prompts lose information — hence structured intermediates), duplicated world models; (3) the technical crux: visual representation for generation — discrete codes (VQ) make AR unification clean but cap fidelity and burn sequence length; continuous latents + diffusion head keep quality but split the objective (public reference points: Chameleon/Emu-style unified AR vs LLM+diffusion hybrids; Gemini's image-out direction); (4) interference specifics: generation wants pixel-complete detail, understanding wants abstraction — shows up as capacity competition and conflicting tokenizer preferences; MoE/routing as partial relief; (5) a position: plausibly 'unified is the destination, staged hybrids are the path' — argued, not hedged."
    },
    {
      "id": "r5q4",
      "title": "Running a cross-org joint venture",
      "prompt": "You co-lead this across two orgs with different roadmaps and incentives. Technically and organizationally, how do you make a joint venture like this actually produce — shared evals? interface contracts? checkpoint exchange? Who owns what?",
      "followups": [
        "Tell me about a real conflict — priorities, credit, or compute — and how it resolved."
      ],
      "strong": "A staff-level systems answer: (1) shared artifacts first — a joint eval suite both sides commit to (the eval IS the treaty), interface contracts (what the understanding model provides: reward API, captioner checkpoint, critique schema — versioned like any dependency), a cadence of checkpoint/data exchanges with clear licensing of credit; (2) explicit ownership: each mechanism has ONE owning org with the other as customer — joint ownership of everything is ownership of nothing; (3) incentive design: launches co-attributed, headline metrics chosen so both orgs' leadership sees their goals move; (4) the conflict story: compute allocation or 'whose model gets the win' is the realistic one — resolved by escalating with a shared doc of options priced in compute/calendar/metric terms, not by hallway consensus. Interviewers at frontier labs weigh this heavily for senior roles: the technical plan is table stakes; making two orgs' gradients point the same direction is the skill."
    },
    {
      "id": "r5q5",
      "title": "The long bet",
      "prompt": "Ten-year view: does the understanding-generation distinction survive? What's your long bet, and what evidence in the next two years would change your mind?",
      "followups": [
        "What would you tell a new grad choosing between the two sides today?"
      ],
      "strong": "Graded on epistemics: a clear position (defensible: the distinction dissolves — prediction/generation becomes the pretraining objective for perception, as in world-model bets; or: economic specialization persists even if architectures unify, the way encoders and decoders both exist in every other stack) + named falsifiers ('if unified models close the generation-quality gap at comparable compute by year X, I'm wrong about specialization' / 'if generative pretraining keeps failing to beat contrastive+captioning on understanding evals at scale, I'm wrong about the flywheel'). The new-grad answer reveals values: pick the side with the tighter feedback loop and the scarcer skills, and note the arbitrage — people fluent in BOTH sides own the interface, which is where you've positioned yourself. Conviction plus falsifiability is the whole game here."
    }
  ]
},

{
  "id": "r6",
  "mode": "verbal",
  "interviewer": "R",
  "topic": "mm-posttraining",
  "title": "Resume: MM post-training & agentic verticals",
  "minutes": 40,
  "persona": "Research deep-dive — multimodal post-training (Gemini 1 & 3) and agentic vertical integration: UI control, Health, Geo, MM Document.",
  "intro": "Last track: you did multimodal post-training on Gemini 1 and 3, and you oversee the multimodal side of the agentic verticals — UI control, health, Geo, documents. The two coding interviewers both come from post-training, so this material may leak into every round. Out loud, 3-5 minutes each. " + ANSWER_FRAME,
  "closing": "You'll get post-training questions from every interviewer on the loop. Anything under 4 here gets priority.",
  "stages": [
    {
      "id": "r6q1",
      "title": "What's different about VLM post-training",
      "prompt": "You've done post-training on two Gemini generations. What's genuinely different about post-training a vision-language model versus a text model — data, rewards, failure modes?",
      "followups": [
        "Where do reward models fail specifically on visual tasks, and what did you do about it?"
      ],
      "strong": "Covers: (1) data: visual instruction data is scarcer and costlier — annotators must look at images (slower, noisier); coverage must span capability × image-distribution, not just task types; synthetic pipelines (model-generated Q/A over real images, verified) carry more of the load; (2) rewards: preference labels on visual answers are less reliable — raters miss fine-grained visual errors, so RMs inherit blindness to exactly the hallucinations you must kill; fixes: grounded/verifiable rewards where possible (structured outputs checked programmatically — OCR fields, boxes, render-compare), specialist verifier models for counting/spatial claims, rubric-based rating with forced image-checking steps; (3) failure modes: post-training amplifies language priors (the model gets more fluent at confidently ignoring the image), style gains masking perception regressions — so perception evals must gate post-training releases too; (4) the Gemini 1 → 3 evolution shape (say at public level): far more RL, verifiable and rubric rewards, agentic trajectories entering the mix."
    },
    {
      "id": "r6q2",
      "title": "Visual hallucination",
      "prompt": "Visual hallucination: the model fluently describes things that aren't in the image. Give me the causal chain as you see it, and rank the mitigations by what actually worked.",
      "followups": [
        "How do you measure hallucination well enough to know a mitigation worked — without the eval itself being gameable by abstention?"
      ],
      "strong": "Covers: (1) causal chain: pretraining captions are noisy and co-occurrence-driven (models learn 'kitchens have toasters'); visual features are lossy (resolution, pooling) so the language prior fills gaps; SFT on fluent-but-unverified answers rewards confident completion; RLHF raters prefer detailed answers, penalizing hedging — a systematic pressure TOWARD hallucination; (2) mitigations ranked: better perception (resolution/dense supervision) attacks the root; grounded rewards / verifier-checked claims during RL; contrastive hard negatives (nearly-right captions labeled wrong); calibrated abstention training ('I can't tell from the image') with rater guidelines that stop punishing it; inference-time self-checking loops as the cheap last layer; (3) measurement: object-level checks against ground truth (POPE-style probing, verified dense-caption evals), plus a precision-recall framing — an abstain-always model scores perfect precision, so you must report coverage too; adversarial 'leading question' splits catch sycophantic hallucination."
    },
    {
      "id": "r6q3",
      "title": "UI control: agents that see screens",
      "prompt": "UI control — an agent that perceives a screen and acts. Your resume says human agentic traces are part of your data strategy. Walk me through the stack: perception, action space, data, and how you evaluate an agent that clicks.",
      "followups": [
        "Why is UI control so much harder than VQA on screenshots? Where exactly does the difficulty concentrate?"
      ],
      "strong": "Covers: (1) perception: screens are dense, tiny-text, high-resolution documents — precise grounding (element localization) matters more than gist; derendering/screen-to-structure is the enabling capability; (2) action space: pixel coordinates vs element-referenced actions (a11y-tree/DOM ids) — the coordinate route generalizes to any pixels but is brittle; structured routes are robust but depend on instrumentation; (3) data: human demonstration traces (screen, action, intent triplets) are gold but expensive and stale as UIs change; augment with synthetic environments and model rollouts filtered by success; (4) eval: end-to-end task success in live/simulated environments (public anchors: WebArena/OSWorld-style), with step-level metrics for diagnosis; flakiness demands many seeds and hermetic envs; (5) the hardness answer: it's sequential — grounding errors compound over 20 steps (0.95^20 ≈ 0.36), state changes after every action, feedback is delayed and sparse, and the cost of a wrong click can be irreversible — VQA is one-shot perception, UI control is closed-loop control on top of perception."
    },
    {
      "id": "r6q4",
      "title": "Verticals without wrecking the generalist",
      "prompt": "Health, Geo, Drive documents — each vertical wants the model better at THEIR thing. Finetune? Domain data in pretraining? RAG and tools? Walk me through how you decide, and how you stop vertical work from regressing the generalist model.",
      "followups": [
        "Medical is the scary one: what changes in your bar for a health-adjacent visual capability?"
      ],
      "strong": "Covers: (1) the decision ladder: prompt/tooling first (cheapest, reversible: RAG over domain corpora, tool calls for authoritative lookups), then post-training adapters/targeted SFT (fast, bounded blast radius), then pretraining-mixture changes (slow, compounding — reserved for perception primitives the vertical exposes, like medical-image reading or map/satellite conventions); the deciding question: is the gap knowledge (retrievable) or capability (must be learned)?; (2) generalist protection: frozen regression suites as merge gates, capability budgets per release, one shared base model with vertical deltas rather than forks (fork = permanent tax); (3) health specifics: the bar flips from helpfulness to calibrated reliability — abstention as a first-class behavior, expert-labeled evals instead of crowd raters, distribution-shift audits (hospital A vs B imaging), regulatory/safety review in the loop, and honesty that some capabilities ship narrow-and-gated rather than general; (4) your actual role: the cross-vertical advisor call — spotting when three verticals ask for the same underlying primitive and pulling it into the core roadmap."
    },
    {
      "id": "r6q5",
      "title": "Post-training strategy: Gemini 1 vs Gemini 3",
      "prompt": "You post-trained Gemini 1 and Gemini 3, generations apart. At whatever abstraction you can share: what changed in your approach, what did the field learn in between, and what did YOU personally update on?",
      "followups": [
        "What's one thing you believed about post-training in the Gemini 1 era that you now think was wrong?"
      ],
      "strong": "Covers the field's public arc, mapped to personal lessons: (1) era shift: from SFT + pairwise-preference RLHF as the workhorse → RL at much larger scale with verifiable rewards (code exec, math checking, structured visual checks), rubric-based and AI-assisted feedback (RLAIF/Constitutional-style) reducing rater bottlenecks, reasoning induced through RL (o1/R1-era lesson: post-training can CREATE capability, not just style — the biggest doctrinal update); (2) multimodal specifics: from bolting visual SFT onto a text recipe → treating perception-grounded rewards and agentic trajectories as first-class; evals maturing from static benchmarks to live task success; (3) the personal-update question is the real one — strong candidates offer something concrete and slightly costly, e.g. 'I treated post-training as polish and staffed it accordingly; the reasoning-RL results proved that wrong and I rebalanced my roadmap/time split', or an update about data: preference data quality dominating algorithm choice (DPO-vs-PPO mattering less than what's in the pairs). Refusing to name a changed mind fails the question."
    }
  ]
}
]
