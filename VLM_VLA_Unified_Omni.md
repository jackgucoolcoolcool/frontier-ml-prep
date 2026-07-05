# VLM / VLA → Unified Omni Models — The Full Map

**Why this doc:** the single most likely "research taste" conversation in 2026: *how do the big visual-representation families (CLIP, DINO, JEPA) relate to unified multimodal understanding (MMU) + generation (MMGen) models, and what does a "truly omni" model look like?* This doc builds the whole map: representation learning → understanding → generation → unification → omni → action (VLA).

**The bar:** for every model family know the **objective (equation)**, **what its representation is good/bad at**, **the failure mode**, and **where it plugs into a unified model**. The punchline you should be able to argue either side of: *understanding and generation are converging onto shared semantic representation spaces, and the same generative math (AR + diffusion/flow) is becoming the universal decoder for pixels, audio, and actions.*

> Builds on Day 4 (multimodal basics) and LLM_Architecture_Frontier (MoE, unified-multimodal §). Equations in MathJax.

---

## §0 — The one-screen mental model

Three research programs, one convergence:

| Program | Question | Objective family | Flagships | What the representation is |
|---|---|---|---|---|
| **Contrastive VL** | "Which caption goes with which image?" | contrastive / alignment | CLIP, SigLIP, SigLIP 2 | global, **semantic, language-aligned** |
| **Self-supervised vision** | "Can pixels alone teach vision?" | self-distillation + masked prediction | DINO, iBOT, DINOv2, DINOv3, MAE | dense, **spatial/geometric**, language-free |
| **Predictive world models** | "Can you predict what happens next — in latent space?" | latent prediction (JEPA) | I-JEPA, V-JEPA, V-JEPA 2(-AC) | **predictive/dynamical**, abstraction over pixels |

And two consumer stacks that used to be separate and are now merging:

- **MMU (understanding):** vision encoder → connector → LLM. Eats CLIP/SigLIP (and increasingly DINO) features. Output: text.
- **MMGen (generation):** text/LLM conditioning → AR or diffusion/flow decoder → pixels/audio. Historically built on **reconstruction latents** (VAE/VQ), now moving to **semantic latents** (REPA, RAE) — i.e. *the same encoders MMU uses*.

**Unified model** = one set of weights doing both. **Omni** = unified across text+image+video+audio(+speech out), interleaved and streaming. **VLA** = the same recipe with actions as an output modality; **world models (JEPA)** are the environment-side generative model. That's the whole correlation in one paragraph — the rest of the doc earns it.

---

## §1 — CLIP & the contrastive line (language-aligned semantics)

### 1.1 CLIP (2021) — mechanism

Dual encoders: image encoder $f$, text encoder $g$, projected to a shared space, trained on 400M web pairs (WIT) with the symmetric InfoNCE loss over an $N\times N$ batch similarity matrix:

$$\mathcal{L}_{\text{CLIP}} = -\frac{1}{2N}\sum_{i=1}^{N}\Big[\log\frac{e^{\langle u_i, v_i\rangle/\tau}}{\sum_j e^{\langle u_i, v_j\rangle/\tau}} + \log\frac{e^{\langle u_i, v_i\rangle/\tau}}{\sum_j e^{\langle u_j, v_i\rangle/\tau}}\Big]$$

with $u_i = f(x_i)/\lVert f(x_i)\rVert$, $v_i = g(t_i)/\lVert g(t_i)\rVert$, learned temperature $\tau$ (init 0.07). Every other in-batch pair is a negative → **needs huge batches** (32k) for hard negatives. Zero-shot classification = retrieval against prompt templates ("a photo of a {label}").

### 1.2 SigLIP (2023) / SigLIP 2 (2025)

Replace softmax-over-batch with an **independent sigmoid per pair** — no global normalization:

$$\mathcal{L}_{\text{SigLIP}} = -\frac{1}{N}\sum_{i,j}\log\sigma\big(z_{ij}(t\,\langle u_i,v_j\rangle + b)\big),\qquad z_{ij}=+1 \text{ iff } i{=}j \text{ else } -1$$

- Decouples loss from batch size (no denominator over the batch) → memory-friendly, scales better, better at small/medium batch.
- **SigLIP 2** adds captioning-based pretraining, self-distillation and masked-prediction terms (i.e. imports DINO-style tricks), and **NaFlex** variable native resolution → much better **dense** features. Note the direction: the contrastive line is absorbing the SSL line's dense objectives.

### 1.3 What CLIP-space is good and bad at (interview gold)

- **Good:** open-vocabulary semantics, zero-shot transfer, retrieval; the default "eyes" of VLMs; the default *text conditioner* of diffusion models (SD uses CLIP text embeddings). CLIP has clean **scaling laws** in data/compute (Cherti et al.).
- **Bad — say these:** behaves like a **bag of words** (fails attribute binding / relations — "horse eating grass" vs "grass eating horse", ARO benchmark); weak on **counting, spatial layout, fine detail**; **CLIP-blind pairs** (MMVP, "Eyes Wide Shut"): image pairs nearly identical in CLIP space but visually different — VLMs built on CLIP inherit these blind spots, while **DINOv2 separates the same pairs**. Root cause: a global contrastive objective only needs features that discriminate captions, so it can discard spatial/compositional detail.

> Key idea: *the training objective determines what information survives in the representation.* Contrastive-to-text keeps what language mentions; it throws away much of what generation and robotics need. This single sentence powers half this doc.

---

## §2 — DINO line (language-free, dense, geometric)

### 2.1 DINO (2021) — self-distillation without labels

Student $\theta_s$ matches an EMA teacher $\theta_t$ across augmented views (multi-crop: 2 global + several local crops; local→global forces part-to-whole):

$$\mathcal{L}_{\text{DINO}} = \sum_{\text{views}} H\big(P_t(x^{(1)}),\, P_s(x^{(2)})\big),\qquad P_t = \mathrm{softmax}\big((h_t(x)-c)/\tau_t\big)$$

Collapse is avoided by two opposing forces on the teacher: **centering** (subtract running mean $c$ — prevents one-dim dominance) and **sharpening** (low teacher temperature $\tau_t \approx 0.04 < \tau_s$ — prevents uniform outputs). Emergent: ViT attention maps segment objects with zero labels.

### 2.2 iBOT → DINOv2 (2023) → DINOv3 (2025)

- **iBOT:** masked image modeling *in feature space* (predict teacher's patch tokens for masked patches) + DINO's CLS loss → dense + global.
- **DINOv2:** the scaling recipe — **curated data** (LVD-142M built by dedup + retrieval-based curation against quality seeds, not raw web), DINO+iBOT losses, **KoLeo** regularizer (spreads features apart), ViT-g (1.1B), then **distillation** down to B/L sizes. Frozen features are SOTA-level on dense tasks (segmentation, **depth**) where CLIP is weak.
- **DINOv3:** 7B ViT on ~1.7B curated images; key new trick: **Gram anchoring** — during long training, dense patch features degrade, so penalize the patch-feature Gram matrix for drifting from an earlier "gram teacher" checkpoint, preserving dense quality at scale. A frozen DINOv3 backbone beats specialized SOTA on dense benchmarks — the "one frozen vision backbone for everything" thesis.
- Contrast **MAE**: reconstruct masked *pixels* (75% masking, cheap). Great finetuning init, but weaker frozen/linear-probe semantics — pixel targets pull capacity toward low-level statistics. (This pixel-vs-latent-target tension returns in §3 and §5.)

### 2.3 Scaling-law angle (lean in)

- **Web-SSL (2025):** train DINO-style SSL on 2B web images (language-free) — at sufficient model+data scale it **matches or beats CLIP as a VLM encoder even on OCR/chart VQA**. Argues language supervision is a data-curation shortcut, not a necessity; scale + data quality dominate.
- Apple's **AIM / AIMv2**: autoregressive image modeling (predict next patch; AIMv2 adds multimodal AR over image+text) shows clean, LLM-like **scaling laws for vision encoders** — a third pretraining family (generative-AR) with the smoothest scaling story. Fits the multimodal-scaling-laws interest: Apple's native-multimodal scaling-law work (early-fusion ≥ late-fusion at matched compute; MoE helps multimodal more than dense) is the systems-level version of the same question.

### 2.4 Where DINO plugs into everything else

- **VLMs:** adding DINOv2 alongside CLIP/SigLIP (Cambrian-1's spatial vision aggregator, Eagle's mixed encoders) fixes CLIP-blind failures, grounding, depth-ish queries.
- **Robotics/VLA:** OpenVLA's encoder is literally **DINOv2 + SigLIP fused** — spatial + semantic.
- **Generation:** REPA aligns diffusion features *to DINOv2*; RAE makes DINOv2 features *the latent space itself* (§5.4). DINO quietly became infrastructure for all three stacks.

---

## §3 — JEPA line (predict in representation space → world models)

### 3.1 The argument

LeCun's claim: generative models that predict **pixels** waste capacity modeling unpredictable, irrelevant detail (exact leaf positions, sensor noise). A **Joint-Embedding Predictive Architecture** predicts the *representation* of the missing/future content:

$$\mathcal{L}_{\text{JEPA}} = \big\lVert \phi\big(\underbrace{s_\theta(x_{\text{ctx}})}_{\text{context enc}},\, m\big) - \underbrace{\bar{s}_{\theta'}(x_{\text{tgt}})}_{\text{EMA target enc}}\big\rVert_2^2$$

— predictor $\phi$ conditioned on position/mask tokens $m$; target encoder is EMA to prevent collapse (same trick as DINO/BYOL). The encoder is *free to discard* unpredictable detail — abstraction is built into the objective. Failure mode to name: **representation collapse** (everything maps to a constant) — held off by EMA asymmetry + architecture, not by a reconstruction term.

### 3.2 The family

- **I-JEPA (2023):** images; predict representations of target blocks from a context block. No handcrafted augmentation stack, no pixel loss; strong semantics at good compute.
- **V-JEPA (2024):** video; masked spatio-temporal tube prediction in latent space; strong motion/appearance features, label-efficient.
- **V-JEPA 2 (2025):** scale to ~1M hours of internet video, ViT-g ≈1B. Then three payoffs: (i) SOTA-ish motion understanding probes; (ii) align with an LLM → competitive **video QA** (JEPA encoder as VLM eyes); (iii) **V-JEPA 2-AC**: bolt a ~300M **action-conditioned predictor** on top, finetune on only ~62h of unlabeled robot video (DROID) → **zero-shot planning** on a real arm in new labs: plan by model-predictive control, choosing actions minimizing energy $E = \lVert \hat{z}_{t+k}(a_{1:k}) - z_{\text{goal}}\rVert$ in representation space, goal given as an image.

### 3.3 How JEPA plans — LeCun's world-model blueprint → V-JEPA 2-AC's MPC loop

**The key correction to make in an interview: JEPA by itself is not a planner.** The JEPA *objective* gives you two artifacts — an encoder $\bar s(\cdot)$ and a latent predictor — and nothing else. Planning requires wrapping those in LeCun's world-model blueprint (*A Path Towards Autonomous Machine Intelligence*, 2022), which has four modules:

| Blueprint module | Role | V-JEPA 2-AC instantiation |
|---|---|---|
| **Perception** | encode observation $x_t \to z_t$ | frozen V-JEPA 2 ViT-g encoder |
| **World model** | action-conditioned latent dynamics $\hat z_{t+1} = P_\phi(z_t, a_t)$ | ~300M block-causal transformer, trained on ~62h unlabeled DROID video (frames + recorded actions/proprio, no rewards, no task labels) |
| **Cost / energy module** | score imagined futures | hand-coded: $E = \lVert \hat z_{t+H} - z_{\text{goal}} \rVert_1$, goal specified as an **image** |
| **Actor / optimizer** | find actions minimizing energy | **CEM inside receding-horizon MPC** (no learned policy at all) |

**Step 1 — make the predictor action-conditioned.** V-JEPA 2's pretrained predictor knows "what tends to happen next" but takes no actions. V-JEPA 2-AC freezes the encoder and trains a new predictor with teacher forcing + a multi-step rollout term so it stays calibrated when fed its own outputs:

$$\mathcal{L}(\phi) = \sum_t \big\lVert P_\phi(z_t, a_t) - z_{t+1} \big\rVert_1 \;+\; \sum_k \big\lVert \underbrace{P_\phi(\hat z_{t+k-1}, a_{t+k-1})}_{\text{fed its own prediction}} - z_{t+k} \big\rVert_1$$

**Step 2 — plan by energy minimization (LeCun's "Mode-2" deliberative loop).** No reward function, no RL, no demonstrations of the test task:

1. Encode current frame $z_t = \bar s(x_t)$ and goal image $z_g = \bar s(x_{\text{goal}})$.
2. Sample $K$ action sequences $a_{t:t+H} \sim \mathcal{N}(\mu, \sigma)$.
3. Roll each out through the world model: $\hat z_{t+1} = P_\phi(z_t, a_t),\ \hat z_{t+2} = P_\phi(\hat z_{t+1}, a_{t+1}), \dots$
4. Score by energy $E(a_{t:t+H}) = \lVert \hat z_{t+H} - z_g \rVert_1$.
5. **CEM:** refit $(\mu, \sigma)$ on the top-$k$ elites; repeat 2–4 a few iterations.
6. **MPC:** execute only the first action, observe the new frame, **replan** from step 1.

$$a^*_{t:t+H} = \arg\min_{a_{t:t+H}} \; \Big\lVert P_\phi\big(\cdots P_\phi(P_\phi(z_t, a_t), a_{t+1}) \cdots\big) - \bar s(x_{\text{goal}}) \Big\rVert_1$$

The "policy" is **inference-time optimization against a world model** — that's exactly why it transfers zero-shot to new labs/scenes: nothing task-specific was ever trained, only dynamics.

**Why plan in latent rather than pixel space (the LeCun argument):** (i) the encoder already discarded the unpredictable bits, so a *deterministic* predictor doesn't pay the pixel-L2 blur tax; (ii) latent distance ≈ semantic progress, so "distance to goal embedding" is a meaningful energy, whereas pixel distance is dominated by lighting/texture nuisance; (iii) rollouts are cheap — no rendering. Same argument, different encoder: **DINO-WM** (world model on frozen DINOv2 patch features, zero-shot latent MPC).

**What V-JEPA 2-AC still lacks vs the full blueprint (great "limitations" material):**
- **No latent variable for uncertainty:** the blueprint's predictor takes $z$ latents to represent *multimodal futures*; V-JEPA 2-AC is a point-estimate — the latent version of blur under stochastic dynamics.
- **No learned cost/critic:** L1-to-goal-embedding is hand-picked; the blueprint wants learnable (and safety) costs.
- **No hierarchy:** H-JEPA plans subgoals in coarser latents over long horizons, low level fills in actions; V-JEPA 2-AC plans flat over ~short horizons because autoregressive latent rollouts **drift/compound error** (hence replanning every step).
- **No Mode-1 distillation:** amortizing the CEM planner into a fast reactive policy is exactly how you'd turn a world model into a VLA-speed controller. (§8: "VLA proposes, world model verifies" is the expected hybrid.)
- Reported practical failure modes: sensitivity to camera pose (energy landscape shifts with viewpoint), goal-as-image being clumsier than language goals, CEM cost growing with action dimensionality.

### 3.4 Case study — building on LeWorldModel (LeWM): can you bolt an LLM on it for captioning?

**LeWM (March 2026, LeCun's AMI-era line, follows LeJEPA):** the first JEPA that trains stably **end-to-end from pixels** with exactly two losses — next-embedding prediction (MSE) + **SIGReg** (LeJEPA's regularizer pushing latents toward an isotropic Gaussian, making collapse *provably* impossible; ~6 hyperparameters → 1). ~15M params, **one 192-d token per frame** (~200× fewer tokens than DINO-WM), action-conditioned predictor, CEM planning ~48× faster than foundation-model world models. Trained **per-environment** (Push-T, Reacher, OGBench-Cube; underperforms on Two-Room — low-dim tasks strain the Gaussian regularizer). The counter-thesis to scale-everything: tiny, provably stable, single-GPU.

**Answer: yes — there's direct precedent and a clean recipe; the honest caveat is what the latent can support.**

- **Precedent:** V-JEPA 2 aligned its encoder with an LLM for video QA (§3.2) — "world-model encoder as VLM eyes" is proven. **The scale mismatch:** V-JEPA 2 is ~1B trained on ~1M hrs of internet video; LeWM is 15M trained per-environment, so its latent keeps only the control-relevant *predictable bits* of one environment. Expect **state-level captions** ("the T-block is left of target, moving right"), not open vocabulary — and no cross-environment transfer. Right-sized for a mechanism study, not a path to a general captioner.
- **Recipe (LLaVA pattern, one GPU):** freeze the LeWM encoder; train a projector $P:\mathbb{R}^{192}\!\to$ LLM embedding space; standard caption NLL

$$\mathcal{L}_{\text{cap}} = -\sum_t \log p_{\text{LLM}}\big(w_t \mid w_{<t},\, P(z_{1:T})\big)$$

  A 0.5–3B LLM with LoRA suffices. One token per frame → a whole trajectory is only $T$ tokens: unusually cheap **temporal** captioning.
- **Free labels:** simulators give ground-truth state → auto-generate captions, program-checkable (which doubles as a verifiable reward if you later RL it).
- **Sanity check before touching the LLM:** linear-probe the 192-d latent for every fact you want captioned (the paper already probes physical properties). If a linear probe can't read it, no LLM will caption it.
- **SIGReg bonus:** the latent is regularized toward an isotropic Gaussian — a well-conditioned source distribution for the projector, arguably friendlier to alignment than raw JEPA latents.

**Four experiments, ascending research value:**

1. **Observed-trajectory captioning** with program-checked accuracy — the baseline (plumbing).
2. **Narrated imagination:** roll the predictor forward under a candidate plan and caption the *imagined* latent future ("push left → the block leaves the table"). Horizon-$k$ caption accuracy = a language-space metric of world-model quality, and an interpretability probe of what the latent encodes.
3. **Language-conditioned goals:** invert the interface — text → latent goal via a learned inverse projector; run CEM against it instead of a goal image (directly addresses §3.3's "goal-as-image is clumsy" limitation).
4. **Objective-interference test (lead with this one):** add captioning as an *auxiliary loss during* LeWM training — does alignment (nameable bits) help or fight prediction (predictable bits)? A direct, single-GPU test of §9.1's entropy-budget thesis. Known failure mode to watch: unfreezing the encoder for captions distorts the dynamics-optimized latent → planning success drops. Measure caption accuracy *and* planning success together.

### 3.5 Why JEPA matters for the unified story

JEPA is **MMGen in latent space**: it's a generative model of *future representations* rather than future pixels. That makes it (a) the "world model" complement to VLA policies (§8), and (b) the philosophical opposite of Emu3-style "predict every pixel token" unification. The synthesis (§9): RAE shows you can have it both ways — do generation *in a semantic latent space* and keep a decoder for when you actually need pixels.

---

## §4 — VLMs / MMU: the understanding stack

### 4.1 The standard recipe

`vision encoder (frozen-ish SigLIP/CLIP) → connector → LLM`, trained with (1) alignment pretraining on captions/interleaved data, (2) visual instruction tuning, (3) preference/RL polish. Connector design axis:

| Connector | Example | Tokens into LLM | Tradeoff |
|---|---|---|---|
| Linear / MLP projector | LLaVA, most modern VLMs | all patch tokens | simple, lossless, token-hungry |
| Q-Former (learned queries, BERT-style) | BLIP-2 | ~32 | compresses hard; loses dense detail; fell out of favor |
| Perceiver resampler + gated cross-attn into frozen LLM | Flamingo | fixed | preserves LLM; `tanh`-gated cross-attn init at 0 for stability |

Modern additions: **native/dynamic resolution** (Qwen2-VL's naive dynamic res; tiling in LLaVA-NeXT/InternVL), **M-RoPE** (decompose RoPE into temporal/height/width for images+video), token compression for video, and **mixed encoders** (Cambrian-1: CLIP+SigLIP+DINOv2+ConvNeXt fused via a spatial aggregator). Frontier open MMU: Qwen2.5-VL, InternVL3; frontier closed: GPT-4o/o-series vision, Gemini 2.5 family.

### 4.2 What limits MMU (say these)

- **The encoder ceiling:** VLM perception errors trace back to CLIP-space blind spots (MMVP). Fixes: better/mixed encoders (DINO in), higher resolution, or *generative* supervision of the encoder.
- **Connector information bottleneck** vs **LLM context cost** — the eternal token-budget tradeoff (video makes it brutal).
- **Hallucination:** language prior overrides weak visual evidence (POPE-style objects). Mitigations: grounding data, contrastive decoding vs image-free logits, RL with visual-fact rewards.

---

## §5 — MMGen: diffusion, flow matching, and the semantic-latent turn (RAE)

### 5.1 Diffusion in five lines

Forward: $q(x_t|x_0)=\mathcal{N}(\sqrt{\bar\alpha_t}\,x_0,\,(1-\bar\alpha_t)I)$. Train a denoiser by the simple loss

$$\mathcal{L}_{\text{DDPM}} = \mathbb{E}_{x_0,\epsilon,t}\,\big\lVert \epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\; t)\big\rVert^2$$

which is denoising **score matching** ($\epsilon_\theta \propto -\sqrt{1-\bar\alpha_t}\,\nabla_x \log p_t(x)$). Sampling = integrate the reverse SDE/ODE. **Classifier-free guidance**: train with condition dropout, sample with $\tilde\epsilon = \epsilon_\theta(x_t,\varnothing) + s\,[\epsilon_\theta(x_t,c) - \epsilon_\theta(x_t,\varnothing)]$ — the quality/diversity dial (too high s → saturation, mode-drop).

- **Latent diffusion (LDM/SD):** run diffusion in a VAE's 8×-downsampled latent — compute win that made image gen practical.
- **DiT (2023):** replace U-Net with a transformer over latent patches + **adaLN-zero** conditioning; FID scales smoothly with GFLOPs → the scaling-law argument that won; now the default (SD3, FLUX, Sora-class video).
- **Flow matching / rectified flow:** define straight paths $x_t = (1-t)x_0 + t\,x_1$ ($x_1\sim\mathcal{N}$) and regress the velocity

$$\mathcal{L}_{\text{FM}} = \mathbb{E}\,\lVert v_\theta(x_t, t) - (x_1 - x_0)\rVert^2$$

— simpler, straighter ODE paths → fewer sampling steps; the SD3/FLUX objective. **Remember for §8: π0 uses exactly this loss to generate robot actions.** Few-step: consistency/distillation, MeanFlow.

### 5.2 The tokenizer/latent problem

AR image gen needs discrete tokens (**VQ-VAE → VQGAN**: codebook + perceptual + adversarial losses); diffusion needs a continuous latent (**KL-VAE**). Both are trained **for reconstruction**, so their latents are semantically poor — linear-probe ImageNet accuracy of SD-VAE latents is ~8%. The generative model must then *rediscover semantics from scratch inside a pixel-oriented space* — a huge hidden tax on training compute. Discrete VQ adds quantization loss + codebook-collapse pathologies, and AR-over-VQ models inherit them.

### 5.3 AR image generation (the other decoder family)

- **LlamaGen/Parti:** plain next-token over VQ codes.
- **VAR (2024):** next-**scale** prediction (coarse→fine token maps) — restores AR scaling behavior for images, NeurIPS'24 best paper.
- **MAR (2024):** masked AR with a small **per-token diffusion head** → continuous tokens, *no VQ at all* — an early sign that "AR backbone + tiny diffusion decoder per step" composes well (echoed later by omni speech talkers, §7).

### 5.4 REPA → RAE: generation adopts understanding's representations

- **REPA (2024):** add an auxiliary loss aligning a DiT's intermediate features to **frozen DINOv2** features (cosine similarity through a projection): $\mathcal{L} = \mathcal{L}_{\text{diff}} + \lambda\,\mathcal{L}_{\text{align}}$. Result: ~**17.5× faster convergence**. Reading: most of diffusion training was spent *learning semantics the SSL world already had*.
- **RAE — Representation Autoencoders (2025):** go further — **replace the VAE**. Encoder = **frozen DINOv2/SigLIP/MAE**; train only a ViT decoder (L1 + LPIPS + adversarial) back to pixels. Reconstruction matches/beats SD-VAE, and the latent space linear-probes at ~84% (vs ~8%). Do DiT diffusion *directly in this high-dimensional semantic latent*: needs width ≥ latent dim (fix: a shallow-wide DiT head, and noise-augmented decoding for robustness), then trains dramatically faster and hits SOTA ImageNet FIDs (~1.1–1.5 at 256/512). **RAEv2 (2026)** simplifies/improves the recipe; **MeanFlow-RAE (CVPR'26)** gets few-step generation in RAE latent space.
- **Why this is the correlation linchpin:** the *same frozen encoder* can now serve MMU (as VLM eyes) *and* define MMGen's latent space. Understanding and generation stop being different representations with a converter between them, and become **encode/reason vs. denoise/decode in one semantic space**. It's also a practical answer to JEPA-vs-generation: predict/diffuse in representation space (JEPA's point), keep a decoder for pixels (generation's point).

---

## §6 — Unified MMU + MMGen: the design space

One transformer that both *reads* and *writes* images (± video/audio). Four architecture patterns — know the table cold:

| Pattern | Idea | Examples | Pros | Cons |
|---|---|---|---|---|
| **A. Single AR, discrete everything** | one vocab: text + VQ image (+video) tokens, pure next-token | Chameleon, Emu3, Emu3.5 | conceptually clean; one loss; scales like an LLM | VQ latents semantically weak → understanding lags; training instability (Chameleon needed QK-norm etc.); slow raster decode |
| **B. Decoupled encoders, one AR core** | understanding path (SigLIP) ≠ generation path (VQ), same transformer | Janus, Janus-Pro | resolves the und/gen representation conflict cheaply; strong both ways at small scale | two vision systems to maintain; gen still VQ-AR quality-capped |
| **C. AR + diffusion in one transformer** | next-token loss on text, diffusion/flow loss on image latents, same weights (or split experts) | Transfusion, Show-o, **BAGEL** (MoT: und+gen experts sharing attention over one sequence; SigLIP eyes + FLUX-VAE gen latents) | best gen quality per FLOP; native interleaving/editing; BAGEL shows **emergent editing/manipulation** from interleaved video+web data | two losses/schedulers to balance; bidirectional attn inside images complicates KV caching |
| **D. LLM + external diffusion decoder via query tokens** | frozen(ish) MLLM emits learned queries/continuous embeddings → conditions a separate diffusion model | MetaQuery, **MetaMorph** (predict SigLIP embeddings, diffusion-decode); productionized flavor: GPT-4o image gen, Gemini "Nano Banana" image editing | minimal risk to MMU ability; modular; fast to ship | not end-to-end; bottleneck at the interface; editing fidelity depends on decoder conditioning richness |

Key facts to deploy:

- **The core tension:** understanding wants **abstraction** (discard pixels), generation wants **completeness** (keep pixels). Janus answers "decouple", BAGEL answers "split experts, share attention", RAE-style answers "make one latent space that's both semantic and decodable" — expect pattern C/D models to adopt RAE-like latents next; empirically, unified-tokenizer papers keep finding the **visual tokenizer is the binding constraint** of pattern A.
- **Does unification help?** Evidence yes: MetaMorph shows und↔gen co-training is mutually beneficial; BAGEL shows emergent editing with scale/interleaved data; generation gives understanding a self-check ("draw it, then look at it" — generative chain-of-thought, and ROVER-style benchmarks test exactly this reciprocity). Evidence "not yet": most unified models still trail specialist gen (FLUX) *or* specialist MMU (Qwen2.5-VL) at matched size — the free lunch shows up mainly in **editing and interleaved tasks** that inherently need both.
- **GPT-4o image gen (2025)** made "the LLM itself generates the image (AR, with some decoder)" mainstream — instruction-following and world-knowledge in generation jumped vs. pure text-conditioned diffusion; **Gemini 2.5 Flash Image** did the same for consistent editing. The commercial frontier converged on unified-ish patterns C/D before open source matched it.

---

## §7 — "Truly omni": add audio/speech/video-in-and-out, streaming

What "omni" must mean (checklist to argue): **any-to-any** modalities (text/image/video/audio in; at least text+speech+image out), **interleaved** generation, **streaming/real-time full-duplex**, **one model** (not a pipeline of ASR→LLM→TTS), and **cross-modal transfer** (modality A data improves modality B).

- **Audio enters exactly like vision did:** neural codecs (SoundStream/EnCodec/Mimi) with **residual vector quantization (RVQ)** = the VQGAN of audio; speech becomes token streams the LLM can read/write. Multi-codebook AR + tiny decoders mirrors MAR's "AR + small generative head" trick.
- **Qwen2.5-Omni / Qwen3-Omni — Thinker-Talker:** the Thinker (LLM, MoE in Qwen3-Omni) does all understanding/reasoning and emits text + hidden states; the **Talker** autoregressively generates multi-codebook speech tokens off those states, streamed through a lightweight codec decoder (~200ms latency); **TMRoPE** interleaves time-aligned audio+video positions. This is pattern D applied to speech: reasoning core + modality decoder, end-to-end trained.
- **Moshi:** full-**duplex** (listen while speaking, dual audio streams) with an "inner monologue" text stream to stabilize reasoning — the systems frontier of real-time.
- **GPT-4o / Gemini:** trained natively multimodal end-to-end from the start (Gemini famously so); GPT-4o demonstrated end-to-end speech (emotion, interruptions) — the existence proof that one network can hold all modalities.
- Honest state of "truly omni", mid-2026: text+image+video+audio **in** is solved-ish; text+speech **out** is production-grade; **image out** is unified in the frontier labs (4o, Gemini) and rising in open source (BAGEL, Janus, Emu3.5); **video out** unified with understanding is still mostly separate (Sora/Veo-class models are their own stacks); and true **cross-modal reciprocal reasoning** (ROVER) is where models still visibly fail.

---

## §8 — VLA: actions as just another modality

### 8.1 The lineage

- **RT-2 (2023):** take a VLM (PaLI-X/PaLM-E), write actions as **text tokens** (per-dim binning), co-finetune on web + robot data → web knowledge transfers to manipulation ("pick up the extinct animal" → grabs the dinosaur toy).
- **OpenVLA (2024):** open 7B; **DINOv2+SigLIP fused encoder** + Llama-2; trained on ~1M Open-X-Embodiment episodes. (Note the encoder choice — §2.4's thesis in production.)
- **Tokenization upgrade — FAST:** naive per-timestep binning fails for high-frequency dexterous control (autocorrelated actions → trivial next-token prediction); FAST compresses action chunks with a **DCT + BPE** scheme → AR VLAs train ~5× faster and match diffusion heads.
- **π0 (2024) / π0.5 (2025):** VLM backbone (PaliGemma ~3B) + a ~300M **action expert** that generates continuous **action chunks by flow matching** (50 Hz) — literally $\mathcal{L}_{\text{FM}}$ from §5.1 applied to joint trajectories. π0.5 adds hierarchical inference (predict subtask in language, then actions) + broad co-training → open-world homes.
- **GR00T N1 (2025):** humanoid foundation model, explicit **dual-system**: System-2 VLM (slow, reasons) + System-1 **diffusion transformer** action head (fast), co-trained on real + synthetic "dream" data generated by video world models.

### 8.2 Design axes & the world-model connection

- **Action decoder:** discrete AR tokens (RT-2/FAST) vs diffusion/flow expert (π0, GR00T). Same tradeoff as image gen AR-vs-diffusion: discrete composes with the LLM; continuous heads match action smoothness/multimodality. The field's answer is converging on "AR reasoning + flow/diffusion decoding" — pattern C/D again.
- **Policy vs world model:** a VLA is a *policy* $p(a|o, \text{instruction})$; **V-JEPA 2-AC** instead learns *dynamics* $p(z_{t+1}|z_t, a)$ and plans by latent MPC — no instruction data needed, goals given as images. World models also serve as **data engines** (GR00T dreams, Genie-3-style interactive video worlds) and evaluators. Expect hybrids: VLA proposes, world model verifies.
- **Correlation to the unified story:** VLA = MMU (perceive + read instructions) + MMGen (generate a trajectory, just in actuator space instead of pixel space). A truly omni model that can already read video and generate sequences has, architecturally, everything a VLA needs except embodiment data.

---

## §9 — The correlation map (the answer to "how does it all connect")

**Axis 1 — what space do you predict in?** Pixels/discrete codes (MAE, VQ-AR, pixel diffusion) ↔ semantic latents (DINO/JEPA features, RAE latents). Ten years of evidence says: *learn representations with abstraction (contrastive/distillation/JEPA), generate in the most semantic space you can still decode from* (LDM → REPA → RAE is one monotone trend line).

**Axis 2 — what objective?** Contrastive (CLIP) / self-distillation+masked-latent (DINO, JEPA) / generative AR (AIM, Emu3) / generative diffusion-flow (DiT, FLUX). They're converging in *capability* (Web-SSL ≈ CLIP at scale; AIMv2 competitive with both) and in *representation geometry* — the **Platonic Representation Hypothesis** (Huh et al. 2024): as models scale, representations across objectives *and modalities* become increasingly aligned, converging toward a shared statistical model of reality. That's the theoretical backbone for expecting unification to work at all.

Concrete role of each family in the omni endgame:

| Ingredient | Role in a unified omni model |
|---|---|
| **CLIP/SigLIP** | language-aligned eyes for MMU; text conditioning for gen; the alignment "glue" between modalities |
| **DINOv2/v3** | dense spatial backbone: grounding/depth for MMU, robot perception for VLA, **latent space for MMGen via RAE** |
| **JEPA / V-JEPA 2** | the world-model organ: predict future latents; video understanding eyes; latent planning for VLA; the argument for latent-space (not pixel) prediction |
| **Diffusion / flow matching** | the universal decoder: images (FLUX), video (Sora-class), speech (codec decoders), **actions (π0)** — one math, four modalities |
| **AR transformer** | the universal reasoner/sequencer: text, interleaving, tool-use, and the conditioning spine of patterns A–D |
| **RAE/REPA** | the bridge: makes understanding's representations be generation's substrate — collapses the und/gen dichotomy |

### 9.1 The three-objective landscape — world model vs alignment vs generation (and the unification template)

The sharpest way to carve representation learning is three objectives: **(1) world-model/predictive** (JEPA family), **(2) cross-modal alignment** (CLIP family — note: not a "language model"; it *distills the abstraction humans already encoded in language*; the LM proper is a fourth object that becomes the reasoning core), **(3) generative/distribution-modeling** (diffusion/AR). That's the right coarse map, but each camp is a bundle of choices on three orthogonal axes:

| Axis | Options | CLIP | DINO | JEPA | Diffusion/AR |
|---|---|---|---|---|---|
| **Target space** | data (pixels) vs representation | representation | representation | representation | data |
| **Supervision source** | self vs cross-modal | cross-modal (text) | self | self | self (± text cond.) |
| **Uncertainty modeled** | contrastive / regression / full distribution | contrastive | distillation (invariance) | regression (point estimate) | full distribution |

All are one template — choose a (context, target) pair, a target transform $\tau$, a divergence $D$:

$$\mathcal{L} = \mathbb{E}_{(\text{context},\,\text{target})}\; D\big(f_\theta(\text{context}),\ \tau(\text{target})\big)$$

— $\tau$ = identity (pixels), EMA/frozen encoder (DINO/JEPA), or other-modality encoder (CLIP); $D$ = InfoNCE, L2, or NLL/score.

**The entropy budget (the memorable framing):** the three objectives differ in *which bits of the world they spend capacity on*. Generation models **all bits** (incl. unpredictable texture — densest signal, zero abstraction). JEPA models **only the predictable bits** (abstraction built in; but a deterministic predictor can't represent multimodal futures — the latent version of blur — and can't emit outputs). CLIP models **only the nameable bits** (semantic; blind to whatever captions omit).

**Four unification routes, all empirically underway:**

1. **Loss stacking on the encoder:** SigLIP 2 = CLIP + self-distillation + captioning; DINOv2 = invariance + masked latent prediction (iBOT); AIMv2 = generative AR over image+text. Crude, works.
2. **Representation bridging (the deep one):** run the generative objective *inside* the SSL/aligned latent — REPA, RAE (images); **DINO-WM** (world model on frozen DINOv2 patch features, zero-shot planning); V-JEPA 2-AC. Key identity to say out loud: **latent diffusion over SSL features ≈ JEPA + a distribution** — swap JEPA's point-estimate L2 predictor for a diffusion/flow head and you fix the multimodal-future problem while keeping abstraction; the pixel decoder becomes optional (display only).
3. **Architectural unification (omni):** an AR core absorbs alignment (it happens *inside* the LLM now), diffusion/flow decoders realize outputs (pixels, audio, actions — π0), a latent-predictor head is the world model. One trunk, three objectives as heads.
4. **Convergence at scale:** Platonic Representation Hypothesis — trained on enough of the same world, the three geometries align anyway; Web-SSL (language-free SSL matching CLIP on VQA at scale) is the empirical flag.

**"If at all" — argue both sides.** *Against* full unification: LeCun's claim that data-space generation is never needed for representation or planning — only for human-facing output; energy minimization over latents suffices, decoders are peripherals. Engineering echo: loss interference, tokenizer compromises, specialists winning at matched compute. *For:* planning under uncertainty needs **distributions over futures** (point-estimate JEPA isn't enough); generation is the densest supervision available; interleaved tasks need all three at once. **The 2026-evidence synthesis: unify the representation (one semantic latent space), stack objectives on the encoder, and model distributions as heads over that latent — full pixel-space unification is the one part you can skip.**

**The synthesis sentence** (memorize): *A truly omni model is an AR reasoning core operating over a shared semantic latent space — built by CLIP/DINO/JEPA-style representation learning — with diffusion/flow decoders hanging off it to realize any output modality: pixels, audio, or actions; JEPA-style latent prediction is the same machinery pointed at the future, which is what makes it a world model and, with an action interface, a robot.*

**Where it's still open (research-taste flags):** unified models trailing specialists at matched compute; visual tokenizer/latent choice for pattern A; video-out unification; balancing AR + diffusion losses in one network (interference, KV-cache complications); whether generation demonstrably improves understanding at scale (reciprocity benchmarks like ROVER say not yet); post-training (RLHF/RLVR) recipes for image/action outputs; and evaluation itself (GenEval/DPG for gen vs MMMU for und measure different organs of the same model).

---

## §10 — Self-test drills

- Why does CLIP need huge batches while SigLIP doesn't? → CLIP's InfoNCE normalizes over the whole batch (negatives come from the batch → more/harder negatives with size); SigLIP's per-pair sigmoid has no batch-wide denominator, decoupling loss quality from batch size.
- What two mechanisms stop DINO from collapsing? → Teacher centering (subtract EMA mean; kills single-mode dominance) and sharpening (low teacher temperature; kills the uniform solution) — they oppose each other, plus EMA teacher asymmetry.
- What is Gram anchoring and why did DINOv3 need it? → Long large-scale training degrades dense patch features; DINOv3 penalizes drift of the patch-feature Gram matrix from an earlier checkpoint's, preserving dense quality while global features keep improving.
- JEPA vs MAE in one line each? → MAE reconstructs masked pixels (low-level, great finetune init); JEPA predicts the latent representation of masked/future content with an EMA target encoder (abstraction built in, no pixel tax; collapse is the risk).
- Why are SD-VAE latents a problem for both diffusion and unified models? → Trained purely for reconstruction → semantically empty (~8% linear probe), so the generator wastes compute learning semantics inside them, and an AR/unified model reading VQ tokens gets weak understanding features.
- What did REPA show, and what did RAE change? → REPA: aligning DiT features to frozen DINOv2 gives ~17.5× faster convergence — diffusion was re-learning known semantics. RAE: replace the VAE entirely with a frozen SSL encoder + trained decoder; diffuse in that semantic latent (wide DiT head) → faster training, SOTA FID, and a latent space shared with understanding.
- Janus vs BAGEL vs Transfusion in one line each? → Janus: decouple und (SigLIP) and gen (VQ) encoders, one AR core. Transfusion: one transformer, next-token loss on text + diffusion loss on image latents. BAGEL: Mixture-of-Transformers — und and gen experts sharing self-attention over one interleaved sequence (SigLIP in, FLUX-VAE latents out).
- Why did pure discrete AR unification (Chameleon/Emu3) underperform on understanding? → Their vision is VQ reconstruction tokens — semantically weak latents (the tokenizer is the bottleneck) — plus early-fusion training instabilities; models with a semantic encoder path (Janus/BAGEL) understand better.
- How does π0 generate actions, and what's the connection to image generation? → A ~300M action expert generates continuous action chunks via flow matching (regress velocity along straight noise→action paths) conditioned on VLM features — the exact objective used by SD3/FLUX for pixels, applied to trajectories.
- What is V-JEPA 2-AC and how does it act without task training? → An action-conditioned latent predictor finetuned on ~62h unlabeled robot video atop V-JEPA 2; at test time it plans by MPC — search action sequences minimizing distance between predicted future latents and a goal image's latent — zero-shot on new scenes.
- What's the Thinker-Talker split and why? → Omni models (Qwen-Omni) separate the reasoning LLM (Thinker: text + hidden states) from a streaming speech-token generator (Talker) reading Thinker states — keeps reasoning quality intact while enabling low-latency full-duplex speech; pattern "AR core + modality decoder" again.
- Argue both sides: will unified und+gen beat specialist pipelines? → For: shared semantics (RAE/REPA evidence), mutual transfer (MetaMorph), emergent editing (BAGEL), world-knowledge generation (4o); interleaved/editing tasks *require* it. Against: at matched compute unified models still trail FLUX-class gen and best MMU; loss interference and tokenizer compromises are real; commercial "unified" systems may quietly be pattern-D pipelines. Then commit: the trend line (REPA→RAE, 4o, Nano Banana) favors unification wherever tasks interleave modalities.

---

*Created 2026-07-03. Companion visual: lineage map + unified design-space diagrams in the HTML version.*
