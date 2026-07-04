# LatentFusion: Multimodal Language Models that Predict in Frozen Self-Supervised Latent Spaces

**Research Proposal** · July 2026 · Status: draft v0.5

> **Figure 1 (see HTML version):** a language model reads interleaved text + CLIP-encoded visuals and, at every visual span, predicts the *frozen* DINOv2/V-JEPA latents of the *next* visual content — text keeps its ordinary cross-entropy; no pixels are ever generated; no collapse is possible because targets are frozen.

---

## Abstract

Current unified multimodal models couple language modeling with *pixel-level* generative objectives — discrete VQ tokens (Chameleon), or diffusion over image latents (Transfusion). Both force the model to spend capacity on high-frequency visual detail that is irrelevant for reasoning, and neither exploits the strong *predictive* representations learned by self-supervised vision models. We propose **LatentFusion**: a language model that **consumes** images through a standard text-aligned vision tower (CLIP/CoCa) but is **trained to predict** upcoming visual content in frozen self-supervised latent spaces — **DINOv2 for image spans** (dense semantics, geometry) and **video-JEPA for video spans** (dynamics, physics). Input space and target space are deliberately decoupled: the model *sees* in a text-aligned space and *predicts* in spaces that encode exactly what text alignment lacks. Targets are frozen and precomputed, so representation collapse is impossible by construction and the target encoders never run in the training loop. We study a spectrum of prediction objectives — dense regression, contrastive (InfoNCE), discretized-latent cross-entropy (including an extended-vocabulary variant that is *literally* standard LLM training), and a lightweight latent flow head — as progressively stronger treatments of the multimodal-future / regression-to-the-mean problem. We hypothesize this yields stronger spatial, temporal, and world-model capability per FLOP than pixel-generative training, at the cost of native image synthesis — recoverable, optionally, with a separately trained latent-to-pixel decoder.

## 1. Motivation

1. **Pixel objectives overpay for detail.** Diffusion and VQ losses require modeling texture and high-frequency structure. JEPA-family results (I-JEPA, V-JEPA, V-JEPA 2) show that predicting in a learned latent space yields better representations at a fraction of the compute — but these models have no language interface.
2. **Text-aligned encoders are blind to what SSL encoders see.** CLIP/CoCa features dominate VLM stacks but are weak carriers of dense spatial structure (DINOv2's strength: parts, layout, geometry, correspondence) and of motion/physical dynamics (V-JEPA's strength). A prediction objective in those spaces injects exactly the missing information — while the input tower keeps OCR/VQA/grounding strength intact.
3. **Frozen targets remove JEPA's hardest engineering problem.** Joint latent-prediction training needs EMA target encoders and stop-gradients to avoid collapse. Frozen, precomputed targets make training a stationary-target problem: no collapse is possible, and the loop stays as simple as standard VLM training.

**One-line pitch:** *a language model that sees in CLIP space and learns to predict in DINO/JEPA space acquires the dense-spatial and temporal world knowledge that text-aligned VLMs structurally lack — without pixel-level generative cost.*

## 2. Related work (positioning)

Three converging threads, none of which occupies our slot:

**(i) Feature prediction/reconstruction as a VLM auxiliary objective.** MetaMorph (AR regression of frozen SigLIP features improves understanding — same-space, same-frame, regression-only); ROSS (denoising reconstruction of the input image's latent tokens — same-frame); Emu/Emu2 (AR regression of frozen CLIP/EVA features in interleaved data); REPA (the mirror direction: aligning diffusion-transformer internals to DINOv2 targets speeds up generation — evidence the DINO target carries dense signal).

**(ii) World models in frozen SSL latent spaces — without language.** DINO-WM (dynamics + planning directly in frozen DINOv2 space); V-JEPA 2-AC (action-conditioned prediction in JEPA space, zero-shot robot planning); IWM (JEPA-style image world models).

**(iii) Semantic visual tokenizers.** SEED/SEED-LLaMA, LaViT, VILA-U (quantize semantic features into discrete tokens for plain LLM prediction); Janus (decouples understanding vs. generation encoders — kindred to our input/target decoupling).

| Work | Visual objective | Target space | Language-conditioned? | Future prediction? |
| --- | --- | --- | --- | --- |
| Chameleon / Transfusion | VQ-AR / diffusion | pixels (discrete/VAE) | ✅ | ✅ (but pixel-level) |
| Emu2, MetaMorph, ROSS | regression / denoising | frozen text-aligned feats | ✅ | ❌ (same-frame / same-space) |
| REPA | alignment | frozen DINOv2 | ❌ (DiT) | ❌ |
| DINO-WM, V-JEPA 2-AC, IWM | latent dynamics | frozen DINO / JEPA | ❌ | ✅ |
| SEED, LaViT, VILA-U, Janus | discrete semantic tokens | quantized CLIP-ish | ✅ | partially |
| **LatentFusion (ours)** | regression / NCE / discrete-CE / flow | **frozen DINOv2 + V-JEPA** | ✅ | ✅ |

**The open slot:** nobody predicts *future* content in a *complementary, dynamics-oriented* frozen space from a *language-conditioned* model with a distribution-aware loss. DINO-WM has no language; MetaMorph/ROSS have no future; Emu has no complementary space.

### 2.1 Design-space analysis: five axes

Every method above is a point in a five-dimensional design space. Making the axes explicit shows both where prior work clusters and why our coordinates are chosen rather than arbitrary:

| Axis | Range | Ours | Why |
| --- | --- | --- | --- |
| **A1 — Input space** | pixels · VQ codes · text-aligned feats | CLIP/CoCa | keep OCR/VQA; blind spots patched by A2, not A1 |
| **A2 — Target space** | pixels · VAE · text-aligned · dense-SSL · video-SSL | DINOv2 + V-JEPA | maximally complementary to A1 |
| **A3 — Temporal offset** | reconstruct (Δ=0) · predict (Δ>0) | Δ>0 (next span / t+k) | Δ=0 degenerates to feature translation |
| **A4 — Objective** | point (L2) → ranking (NCE) → categorical (CE) → continuous (flow) | ablation axis | distributionality only matters when Δ>0 makes futures multimodal |
| **A5 — Target trainability** | frozen · EMA · jointly trained | frozen | collapse impossible; stationary loss; targets precomputable |

Cross-correlations with prior work — each neighbor validates one axis-setting while leaving ours open:

- **REPA ⇒ A2:** DINOv2 targets carry useful dense signal even for a *generative* DiT — evidence the target space works, in a different host model.
- **ROSS & MetaMorph ⇒ A3 baseline:** even Δ=0 reconstruction/regression of visual features already improves VLM understanding; our claim is that Δ>0 adds world-modeling on top. (These two are also our built-in ablation arm: set Δ=0 and we *reproduce* them.)
- **DINO-WM & V-JEPA 2-AC ⇒ A2×A3:** frozen SSL spaces support Δ>0 dynamics well enough to *plan* — but with no language conditioning (A1 has no text).
- **MAR & RCG ⇒ A4:** continuous distributional heads work in low-dim latent spaces; Emu2's blurriness shows what happens when A4=point while futures are multimodal.
- **Chameleon/Transfusion ⇒ A3 with the wrong A2:** they do predict future visual content, but in pixel space — paying for texture. We keep their A3 and move A2.
- **SEED/LaViT ⇒ A4=categorical is trainable at LLM scale;** Janus ⇒ decoupling spaces per role (our A1≠A2) beats forcing one space to do everything.
- **A5 interaction:** joint/EMA targets (V-JEPA proper) could adapt to the task but reintroduce collapse machinery and non-stationary targets inside LLM training; frozen is the only setting where targets ship like labels. The cost — a non-adaptable space — is exactly what the week-1 probes and H2 measure.

The unexplored cell is precisely (A1=text-aligned, A2=dense/video-SSL, A3=Δ>0, A4=distribution-aware, A5=frozen).

### 2.2 What does each visual objective actually learn? (why the target space is the real decision)

Five training paradigms dominate visual representation learning. They differ not in *how much* they learn but in **what decides which information is kept** — and that selector is the deepest reason target-space choice (A2) matters:

| # | Paradigm | Exemplars | What selects the kept information | What is learned | What is systematically discarded |
| --- | --- | --- | --- | --- | --- |
| 1 | **Contrastive image–text** | CLIP, SigLIP | what captions *mention* | global, nameable semantics; open-vocab recognition | spatial layout, counting, relations, part structure — the "bag-of-concepts" failure (MMVP / *Eyes Wide Shut*) |
| 2 | **Co-training with an LM** (captioning / AR text) | CoCa, LLaVA-style SFT | what captions *sequentially verbalize* | more compositional/relational than (1) — word order forces binding; instruction-following | anything text rarely describes: texture statistics, precise geometry, dynamics |
| 3 | **Geometric / self-distillation SSL** | DINO, DINOv2, iBOT | augmentation-invariance on the image itself — no text ceiling | dense correspondences, object parts, depth/layout cues; emergent segmentation | names & cross-modal grounding; temporal dynamics |
| 4 | **Latent feature prediction** | I-JEPA, V-JEPA | **predictability** — keep what makes masked/future content predictable | dynamics, physical regularity, persistent structure; noise/texture dropped *because unpredictable* | high-frequency detail, exact appearance; text alignment |
| 5 | **Diffusion / pixel-generative** | Transfusion branch, MAR, Emu-gen | pixel likelihood — *everything*, weighted by pixel variance | full appearance distribution; usable features mid-network (cf. REPA needing DINO to speed it up) | nothing — which is the problem: capacity spent ∝ pixel entropy, not semantic value |

Readings of this table that shape the proposal:

- **The selector, not the architecture, is the objective's identity.** (1)/(2) are *text-bounded* (can't learn what language doesn't say), (3) is *augmentation-bounded* (invariances are hand-chosen), (4) is *predictability-bounded* (keeps exactly the world-model-relevant bits), (5) is *unbounded* (and therefore unfocused).
- **Empirical anchors:** MMVP/"Eyes Wide Shut" shows CLIP-blind pairs breaking VLMs — a (1)-failure; DINOv2's dense-probe dominance is a (3)-signature; V-JEPA 2-AC planning is a (4)-signature; REPA — a *generative* model learning faster when aligned to DINO — shows (5) does not subsume (3) even with more compute.
- **Why LatentFusion combines (2)+(3)+(4) and skips (5):** input tower & text CE give us (1)/(2)'s names and verbalization; DINO targets add (3)'s geometry; JEPA targets add (4)'s dynamics; each patched blind spot corresponds to a *different selector*. (5) is omitted from training (kept only as an offline decoder) because its selector — pixel entropy — is precisely the one we argue is wasteful.
- **This also predicts H2's outcome pattern:** gains should appear exactly on benchmarks probing the *selector gap* between input and target space (spatial for DINO, temporal for JEPA), and nowhere else. If gains appear elsewhere, the complementarity theory is wrong in an interesting way.

## 3. Method

### 3.1 Architecture: decoupled input and target spaces

- **Input tower:** a pretrained CLIP/CoCa vision encoder (initialized from an existing heavily-trained checkpoint; optionally fine-tuned), projected into the LLM. This preserves text-aligned strengths — OCR, documents, VQA, grounding.
- **Target spaces (frozen, precomputed offline):**
  - **Image spans → DINOv2** latents: dense semantics, object parts, geometry. Precedent: DINO-WM, REPA, RCG.
  - **Video spans → V-JEPA 2** latents: motion, physical dynamics. Precedent: V-JEPA 2-AC.
- **Separate lightweight prediction heads** per target space (shared trunk), each with its own per-dimension whitening — DINO and JEPA latents have different dimensionality and statistics; never mix them in one head.

Complementarity is the design principle: predicting SigLIP targets from a CLIP input is nearly redundant; predicting DINO/JEPA targets injects what the input space lacks.

### 3.2 What to predict: future, not same-frame

Targets must be the latents of **upcoming visual content** — the next image in the interleaved sequence, or frames \(t+k\) of a video given frames \(\le t\). Predicting the latents of the *same* frame the input tower is currently encoding degenerates into CLIP→DINO feature translation — a distillation signal, kept only as a small auxiliary term. The primary loss is temporal/anticipatory: *given what you see and read, predict what comes next.*

### 3.3 The loss, concretely, in a standard causal LM

Everything stays next-token prediction under the ordinary causal mask; only *what the head outputs* changes at positions whose next slot is visual. With hidden states \(h_i\), LM head \(W\), latent head \(g\) (2-layer MLP per target space):

\[
\mathcal{L} \;=\; \underbrace{\sum_{i:\,x_{i+1}\in\text{text}} -\log p_W(x_{i+1}\mid h_i)}_{\text{unchanged text CE}} \;+\; \lambda \underbrace{\sum_{i:\,x_{i+1}\in\text{visual}} \ell\big(g(h_i),\, \bar z_{i+1}\big)}_{\text{latent prediction}}, \qquad \bar z = \text{whiten}(z)
\]

```python
h = model.transformer(inputs_embeds=emb).last_hidden_state    # [B, L, D], causal mask as usual

# text loss — untouched
L_text = F.cross_entropy(lm_head(h[:, :-1]).transpose(1, 2), tokens[:, 1:],
                         reduction='none')[text_next_mask[:, 1:]].mean()

# latent loss — positions whose NEXT slot is a visual target
pred  = dino_head(h[:, :-1][visual_next_mask[:, 1:]])          # [M, d_dino]
tgt   = (z_dino - mu) / sigma                                  # precomputed, whitened, frozen
L_reg = F.mse_loss(pred, tgt)                                  # or InfoNCE / discrete-CE / flow

loss = L_text + lam * L_reg
```

Input embeddings for visual slots come from the CLIP/CoCa tower through a projector; target latents `z` ship from the dataloader like labels. Input and target streams never mix. For future prediction, the target at an image span is the *next* span's latents — same code, different target alignment.

### 3.4 Prediction objectives (ablation axis)

Ordered by strength against regression-to-the-mean:

**(a) Dense regression.** \(\mathcal{L}_{\text{reg}} = \tfrac{1}{N}\sum_i \|\hat z_i - z_i\|_2^2\). Simplest; dense gradient (MetaMorph shows this improves understanding); collapses multimodal futures to their mean. Optional VICReg-style variance regularizer.

**(b) Contrastive (InfoNCE).** \(\mathcal{L}_{\text{nce}} = -\log \frac{\exp(\mathrm{sim}(\hat z_i, z_i)/\tau)}{\sum_j \exp(\mathrm{sim}(\hat z_i, z_j)/\tau)}\). Only needs to *rank* the true future above in-batch negatives — no mean-collapse, negligible cost, most JEPA-native (energy-based) fix.

**(c) Discretized-latent cross-entropy.** Quantize the frozen target space once (k-means or FSQ), predict codes with softmax CE. A categorical natively represents multimodal futures — AR-over-tokens without VQ-on-pixels lossiness.

**(c′) Extended-vocabulary variant.** Add the visual codes to the LLM vocabulary and predict them through the *normal LM head* — the entire system is then literally standard LLM training: one CE loss, one softmax, perplexity and scaling curves for free. The most infrastructure-friendly variant and the natural first run. **Tokenizer caveat:** codes must quantize *semantic* spaces (DINO/JEPA latents, or SEED/LaViT-style tokenizers) — VQGAN/pixel-VQ codes would smuggle texture back in and revert to Chameleon-style pixel modeling.

**(d) Latent flow-matching head (MAR-style).** \(\mathcal{L}_{\text{flow}} = \mathbb{E}_{t,\epsilon}\|v_\theta(z_i^t, t, c_i) - (z_i - \epsilon)\|_2^2\) with LLM-emitted conditioning \(c_i\). Fully distributional, sampleable rollouts; iterative inference (cheap: low-dim); needs schedules/whitening. The upper-bound treatment.

| | (a) regression | (b) contrastive | (c/c′) discrete-CE | (d) latent flow |
| --- | --- | --- | --- | --- |
| Multimodal futures | mean-collapse | ranking, no collapse | full categorical | full continuous |
| Gradient to LLM | dense, strong | moderate | dense | indirect (cond. vector) |
| Inference | single pass | single pass | single pass | iterative sampling |
| Extra machinery | none | negatives, τ | codebook build | schedules, whitening, CFG |
| Sampleable rollouts | no (point) | no | yes | yes (richest) |
| Infra friendliness | high | high | **highest (c′)** | low |
| Precedent | Emu2, MetaMorph | CPC/CLIP-style | SEED/LaViT | MAR, RCG |

### 3.5 Text-side prediction: narrated rollouts (captioning arm)

The prediction can also run in the **reverse direction**: from predicted future latents back into language. Add a caption objective conditioned on the *predicted* latent \(\hat z\) (not the ground-truth image):

\[
\mathcal{L}_{\text{cap}} = -\sum_t \log p\big(w_t \mid w_{<t}, \hat z\big)
\]

where \(w\) is a caption/description of the future frame (from video ASR/alt-text, or synthetically captioned). This is the CoCa lesson — contrastive and captioning losses teach complementary things — imported into the predictive setting, and it buys three things:

1. **Grounding pressure.** The predicted latent must remain *decodable into language*, blocking degenerate solutions where \(\hat z\) drifts into regions of the frozen space that satisfy the latent loss but carry no usable semantics.
2. **A narrated world model.** Latent rollout → caption gives human-readable imagined futures ("the ball will bounce off the table") — free interpretability for H3's rollout evals, and a natural VLA interface (describe the predicted consequence of an action before executing it).
3. **A text-metric for latent quality.** Caption quality of \(\hat z\) vs. caption quality of the true \(z\) is a direct, scalable measure of how much semantic content prediction preserves — cheaper than training probe heads.

**Risks:** future-caption data is noisier (ASR misalignment); and the model may shortcut — inferring the caption from surrounding *text* context rather than from \(\hat z\). Mitigation: gradient-stop the text context into the caption head, or evaluate with context-ablated captioning.

### 3.6 Optional: latent-to-pixel decoder

A diffusion decoder trained separately to invert the target encoders (RCG-style), used for visualization and optional generation only; kept out of the main training loop by design.

## 4. Hypotheses & experiments

**H1 (understanding).** Adding latent prediction on frozen SSL targets improves an otherwise-identical VLM baseline. *Eval:* spatial suites (segmentation-flavored VQA, counting, spatial relations) for DINO targets; temporal/physical suites (Perception Test, TempCompass) for JEPA targets; MetaMorph-style no-prediction-loss ablation as baseline. Because the input tower is CLIP/CoCa, OCR/DocVQA should be *unchanged* — a regression test, not a trade-off.

**H2 (target space matters — headline ablation).** Hold everything fixed; swap targets among SigLIP / DINOv2 / V-JEPA. Prediction: SigLIP targets ≈ no gain (redundant with input); DINOv2 wins spatial; V-JEPA wins temporal/embodied; the per-benchmark-family map is a publishable result on its own.

**H3 (objective).** (b)/(c)/(d) ≥ (a) wherever futures are genuinely multimodal (video continuation, next-image-in-document); (a) remains competitive for understanding gains. *Eval:* latent rollout metrics (feature PSNR, retrieval@k of true future), planning success à la V-JEPA 2-AC / DINO-WM.

**H4 (efficiency).** At matched FLOPs, better spatial+temporal-understanding scaling than a Transfusion-style pixel-diffusion baseline (0.3B/1B/3B ladder; loss-vs-compute and benchmark-vs-compute curves; objective (c′) gives the cleanest scaling ordinate).

### 4.1 De-risking ladder — staged go/no-go gates

Each gate is an order of magnitude cheaper than the next; each has an explicit kill/pivot criterion so no compute is spent past a dead assumption.

| Gate | Cost | Experiment | Green light | Kill / pivot |
| --- | --- | --- | --- | --- |
| **D0 — Target-space probes** | hours, 1 GPU | linear-probe frozen DINOv2 + V-JEPA 2 features on *our* eval suites | probes beat CLIP features on spatial/temporal tasks | pivot target space (A2) before anything else |
| **D1 — Predictability check** | ~1 day | small transformer predicts next-frame JEPA/DINO latents from context latents only (no LLM, DINO-WM-style) | prediction ≫ copy-last-frame baseline in retrieval@k | futures unpredictable in this space ⇒ reconsider Δ or space |
| **D2 — Translation control** | ~1 day | CLIP→DINO Δ=0 regression probe | measures the "translation floor" all later runs must beat | — (this is a measuring stick, not a gate) |
| **D3 — Frozen-LLM MVE** | days, single node | frozen 7B LLM, train only projector + heads, objectives (a)+(c′) | latent loss learns above D2 floor; no text-CE regression | if signal only equals D2: Δ>0 targets aren't being used ⇒ fix data interleaving |
| **D4 — Unfrozen MVE + H1** | ~1 week | unfreeze LLM, full MVE vs. no-prediction-loss baseline | H1 moves ≥1 spatial/temporal benchmark outside noise; OCR unchanged | if OCR regresses: rebalance λ; if nothing moves: stop or reframe as world-model-only (H3 path) |
| **D5 — Objectives + H2/H3** | weeks | (b)/(d) heads, target-space swap, rollout evals | per-axis map matches predictions | publish the map either way — negative H2 is still a result |
| **D6 — Scaling ladder (H4)** | largest | 0.3B/1B/3B vs. pixel-diffusion baseline | favorable slope | claims restricted to measured regime |

**Minimal viable experiment (D3/D4):** existing CLIP/CoCa tower + open 7B LLM + frozen DINOv2 targets, objectives (a) and (c′), on interleaved image-text data. Single-node; targets precomputed. Baseline: same run, prediction loss off. Video/JEPA arm second.

## 5. Risks & mitigations

| Risk | Severity | Mitigation |
| --- | --- | --- |
| Same-frame prediction degenerates to feature translation | **High — main design risk** | temporal targets only (§3.2); translation kept as small aux term; monitor gain vs. a translation-only control |
| Prediction losses don't move downstream benchmarks | Medium | H1 ablation is the primary claim, not loss curves |
| Latent geometry unfriendly to flow head | Medium | per-dim whitening per target space (RCG/Emu2 recipe) |
| Codebook quality bottlenecks (c)/(c′) | Medium | FSQ over k-means; residual VQ; sweep codebook size |
| Loss balancing (\(\lambda_k\)) unstable across objectives | Low | small-scale grid; μP-style transfer upward |
| Caption arm shortcuts via text context instead of \(\hat z\) | Medium | gradient-stop text context into caption head; context-ablated caption eval |
| Two target spaces complicate the pipeline | Low | targets precomputed offline; separate heads, shared trunk |

*(Resolved by design vs. v0.1: OCR/fine-detail weakness — the input tower is now CLIP/CoCa, so frozen-SSL blind spots affect only the auxiliary targets, not what the model sees.)*

## 6. What this is not

- **Not an image generator.** Native synthesis is deliberately traded away; §3.6 recovers it approximately if needed.
- **Not joint JEPA training.** No EMA/stop-grad anywhere; stability is bought with frozen targets, at the cost of non-adaptable target spaces (probed in week 1, mapped in H2).

## 7. Timeline (single researcher + modest compute)

| Weeks | Milestone |
| --- | --- |
| 1–2 | Linear probes of DINOv2 + V-JEPA 2 features on target evals (go/no-go) |
| 3–6 | MVE: DINO targets, objectives (a)+(c′), H1 ablation + translation-only control |
| 7–10 | Objectives (b)/(d) on images; whitening + codebook studies |
| 11–14 | Video/V-JEPA arm; target-space swap (H2) |
| 15–18 | Scaling ladder (H4), write-up |

## References (non-exhaustive)

I-JEPA (Assran et al. 2023) · V-JEPA (Bardes et al. 2024) · V-JEPA 2 / 2-AC (Meta 2025) · DINOv2 (Oquab et al. 2023) · DINO-WM (Zhou et al. 2024) · IWM (Garrido et al. 2024) · Transfusion (Zhou et al. 2024) · Chameleon (2024) · Emu2 (Sun et al. 2024) · MetaMorph (Tong et al. 2024) · ROSS (Wang et al. 2024) · REPA (Yu et al. 2024) · SEED-LLaMA (Ge et al. 2023) · LaViT (Jin et al. 2024) · VILA-U (2024) · Janus (DeepSeek 2024) · MAR (Li et al. 2024) · RCG (Li et al. 2023) · VICReg (Bardes et al. 2022) · CLIP (Radford et al. 2021) · CoCa (Yu et al. 2022) · SigLIP (Zhai et al. 2023) · iBOT (Zhou et al. 2022) · MMVP / Eyes Wide Shut (Tong et al. 2024)
