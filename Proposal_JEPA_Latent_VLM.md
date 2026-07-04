# LatentFusion: Multimodal Language Models that Predict in a Frozen Video-JEPA Space

**Research Proposal** · July 2026 · Status: draft v0.1

---

## Abstract

Current unified multimodal models couple language modeling with *pixel-level* generative objectives — discrete VQ tokens (Chameleon), or diffusion over image latents (Transfusion). Both force the model to spend capacity on high-frequency visual detail that is irrelevant for reasoning, and neither exploits the strong *predictive* representations learned by self-supervised video models. We propose **LatentFusion**: a two-stage recipe in which (1) a JEPA-style encoder is pretrained on large-scale video and then **frozen**, and (2) a language model is trained on interleaved image/video–text data where image spans are represented — and *predicted* — as frozen JEPA latents. Prediction is trained with either (a) dense latent regression, (b) a lightweight diffusion/flow head over the latent (MAR-style), or (c) both. We hypothesize this yields a VLM with substantially stronger temporal, physical, and world-model capabilities per FLOP than pixel-generative training, at the cost of native image synthesis — which we recover, optionally, with a separately trained latent-to-pixel decoder. The frozen-target design eliminates representation collapse by construction and makes the recipe unusually simple and stable to scale.

## 1. Motivation

1. **Pixel objectives overpay for detail.** Diffusion and VQ losses require modeling texture and high-frequency structure. JEPA-family results (I-JEPA, V-JEPA, V-JEPA 2) show that predicting in a learned latent space yields better representations at a fraction of the compute — but these models have no language interface.
2. **Video-SSL features encode dynamics that text-aligned encoders miss.** CLIP/SigLIP-style encoders dominate VLM stacks but are trained on static image–text alignment; they are weak carriers of motion, physical intuition, and temporal structure. V-JEPA 2 demonstrates that video-JEPA latents support zero-shot planning (V-JEPA 2-AC). No current VLM *predicts* in such a space.
3. **The two-stage frozen design removes JEPA's hardest engineering problem.** Joint latent-prediction training needs EMA target encoders and stop-gradients to avoid collapse. Freezing the encoder after Stage 1 makes Stage 2 a stationary-target problem: no collapse is possible, and the LLM training loop stays as simple as standard VLM training.

**One-line pitch:** *a language model trained to autoregressively predict the future in a frozen video-JEPA space learns a textual world model — temporal and physical understanding that pixel-generative and contrastive-encoder VLMs structurally lack.*

## 2. Related work (positioning)

| Work | Visual objective | Target space | Gap we fill |
| --- | --- | --- | --- |
| Chameleon | AR over VQ tokens | discrete pixels | pixel-level, lossy quantization |
| Transfusion | diffusion head | VAE pixel latents | still pixel-generative |
| Emu / Emu2 | AR regression | frozen CLIP/EVA feats | regression-to-mean; image-contrastive space |
| MetaMorph | aux. regression | frozen SigLIP feats | static, text-aligned space; no distributional head |
| RCG | diffusion | frozen SSL (image) reps | unconditional image gen, no LLM |
| MAR | per-token diffusion head | VAE latents | pixels, not semantic latents |
| V-JEPA 2 / 2-AC | latent prediction | video-JEPA | no language interface |
| **LatentFusion (ours)** | regression and/or latent diffusion | **frozen video-JEPA** | language-conditioned prediction in a dynamics-rich space |

## 3. Method

### Stage 1 — Frozen video-JEPA encoder

Pretrain (or reuse: V-JEPA 2 ViT-g) a spatiotemporal encoder \(f\) with the standard JEPA objective — EMA target encoder, block masking, latent L2. After pretraining the online and target encoders coincide; **freeze \(f\) permanently**. Precompute per-clip latents \(z = f(x) \in \mathbb{R}^{N \times d}\) offline — a major systems win: Stage 2 never runs the vision tower.

### Stage 2 — Interleaved LLM training

Sequences of the form `[text][BOI] z_1 … z_N [EOI][text] …` over interleaved image/video–text corpora. A projector maps \(z\) into the LLM width on input. Losses:

- **Text positions:** standard next-token cross-entropy.
- **Image/video spans:** predict the frozen latents of the upcoming visual span from prior context, via one of three heads:

**(a) Dense regression head.**
\[
\mathcal{L}_{\text{reg}} = \tfrac{1}{N}\sum_i \big\| \hat z_i - z_i \big\|_2^2 \quad (\text{or } 1-\cos(\hat z_i, z_i))
\]
Simplest; strong dense gradient into the LLM (MetaMorph shows this *improves understanding*); suffers regression-to-the-mean under genuine uncertainty.

**(b) Latent diffusion / flow-matching head (MAR-style).** The LLM emits a conditioning vector \(c_i\); a small MLP head is trained with flow matching in latent space:
\[
\mathcal{L}_{\text{flow}} = \mathbb{E}_{t, \epsilon}\big\| v_\theta(z_i^t, t, c_i) - (z_i - \epsilon) \big\|_2^2
\]
Models the full conditional distribution → diverse sampled rollouts for planning; requires per-dimension latent normalization (cf. Emu2, RCG) and adds sampling cost.

**(c) Both:** regression as auxiliary dense signal + flow head for distributional prediction, weighted \(\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_r \mathcal{L}_{\text{reg}} + \lambda_f \mathcal{L}_{\text{flow}}\). The (a) vs (b) vs (c) comparison is a headline experiment.

### Regression vs. latent-diffusion head — expected trade-offs

| | Regression (a) | Latent diffusion/flow (b) |
| --- | --- | --- |
| Conditional distribution | mean only (blurry in feature space) | full distribution, sampleable |
| Gradient signal to LLM | dense, strong for understanding | indirect (via conditioning vector) |
| Inference | single pass | iterative sampling (cheap: low-dim) |
| Complexity | one L2 term | schedules, normalization, CFG |
| World-model rollouts | deterministic point rollouts | diverse imagined futures |
| Precedent | Emu2, MetaMorph | MAR, RCG |

### Optional Stage 3 — Latent-to-pixel decoder

A diffusion decoder trained separately to invert \(f\) (RCG-style), used only for visualization and optional generation. Kept out of the main training loop by design.

## 4. Hypotheses & experiments

**H1 (understanding).** Adding latent prediction on frozen JEPA targets improves a standard VLM baseline on temporal/physical benchmarks. *Eval:* Perception Test, TempCompass, physical-reasoning suites, vs. an identical model without the prediction loss (MetaMorph-style ablation).

**H2 (target space matters).** Video-JEPA targets beat SigLIP targets on temporal/embodied tasks and lose on OCR/document tasks. *Eval:* swap target space, hold everything else fixed; DocVQA/ChartQA vs. temporal suites. Mitigation arm: dual input streams (SigLIP + JEPA in, JEPA-only targets).

**H3 (distributional head).** Flow head ≥ regression for rollout/planning quality; regression ≥ flow head for understanding; combined (c) is Pareto-best. *Eval:* latent-space rollout metrics (feature PSNR, retrieval@k of true future frame), robot planning success à la V-JEPA 2-AC.

**H4 (efficiency).** At matched FLOPs, LatentFusion reaches better temporal-understanding scaling than a Transfusion-style pixel-diffusion baseline. *Eval:* small scaling ladder (e.g. 0.3B/1B/3B LLM), loss-vs-compute and benchmark-vs-compute curves.

**De-risking probe (week 1):** linear-probe frozen V-JEPA 2 features on the target downstream tasks *before* any Stage-2 training; if probes are weak on everything we care about, revise the target space first.

### Minimal viable experiment
Frozen V-JEPA 2 ViT-L + open 7B LLM (frozen-then-unfrozen) + regression head, on an interleaved video-text subset (e.g. HowTo100M/OBELICS-style mix). Single-node scale; latents precomputed. Baseline: same data, prediction loss off.

## 5. Risks & mitigations

| Risk | Severity | Mitigation |
| --- | --- | --- |
| JEPA space weak on OCR/fine detail (frozen ⇒ can't adapt) | **High — main scientific risk** | early linear probes; dual-encoder input; honest scoping to temporal/embodied claims |
| Regression-to-mean makes prediction loss uninformative | Medium | flow head (b); report per-dim variance of predictions |
| Latent geometry unfriendly to diffusion | Medium | per-dim whitening (RCG/Emu2 recipe) |
| "Loss went down" without downstream gains | Medium | H1 ablation is the primary claim, not the loss curve |
| Loss balancing (\(\lambda_r, \lambda_f\)) unstable | Low | grid at small scale; μP-style transfer upward |
| Tube vs. per-frame tokenization mismatch with text interleaving | Low | start per-frame; ablate tubes |

## 6. What this is not

- **Not an image generator.** Native synthesis is deliberately traded away; Stage 3 recovers it approximately if needed.
- **Not joint JEPA training.** No EMA/stop-grad in Stage 2; stability is bought with the frozen encoder, at the cost of a non-adaptable target space (tested in H2).

## 7. Timeline (single researcher + modest compute)

| Weeks | Milestone |
| --- | --- |
| 1–2 | Linear probes of V-JEPA 2 features on target evals (go/no-go) |
| 3–6 | MVE: regression head, H1 ablation |
| 7–10 | Flow head, H3; latent normalization study |
| 11–14 | Target-space swap (H2), dual-encoder arm |
| 15–18 | Scaling ladder (H4), write-up |

## References (non-exhaustive)

I-JEPA (Assran et al. 2023) · V-JEPA (Bardes et al. 2024) · V-JEPA 2 / 2-AC (Meta 2025) · Transfusion (Zhou et al. 2024) · Chameleon (2024) · Emu2 (Sun et al. 2024) · MetaMorph (Tong et al. 2024) · MAR (Li et al. 2024) · RCG (Li et al. 2023)
