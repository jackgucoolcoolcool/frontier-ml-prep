# Day 4 — Multimodal & Agentic AI + Full Mock Drill

**Target:** Research Scientist loop at a frontier lab (Meta TBD / MSL style).
**Mode:** Concepts first, light code, explanation + self-test Q&A.
**Time budget:** ~8 hours.
**The bar for today:** Be **conversant** (not expert) on how multimodal models and agents work and how you'd *evaluate* them — then spend the back half doing a **full out-loud mock drill** across the entire #16 question bank. Multimodal/agents is broad-but-shallow for most loops; the consolidation drill is where today's real value is.

---

## Suggested 8-hour schedule

| Block | Time | Topic | Section |
|-------|------|-------|---------|
| 1 | 0:00–1:30 | Vision-language models: CLIP, ViT, fusion | §1 |
| 2 | 1:30–2:30 | Audio & video basics | §2 |
| — | 2:30–2:45 | Break | |
| 3 | 2:45–4:15 | Agents: tool use, ReAct, memory, evaluation | §3 |
| 4 | 4:15–5:00 | RAG & retrieval (bonus, high-ROI) | §4 |
| — | 5:00–5:15 | Break | |
| 5 | 5:15–8:00 | **Full #16 mock drill** across all days | §5 |

---

## §1 — Vision-language models (VLMs)

**Watch first:**
- OpenCV Live #177, *"Vision Language Models: Understanding CLIP"* — https://www.youtube.com/watch?v=gBgEcn7AxZs
- 3Blue1Brown, *"Visualizing transformers and attention"* (TNG talk) — https://www.youtube.com/watch?v=KJtZARuO3JY

### 1.1 CLIP — contrastive image-text pretraining (the foundational idea)

CLIP trains an **image encoder** and a **text encoder** jointly so that matching image-text pairs land **close together** in a shared embedding space, and non-matching pairs land far apart. The mechanism:
- Take a batch of N (image, caption) pairs. Encode all images and all texts.
- Compute the N×N similarity matrix; the **contrastive loss** pulls the N correct (image, caption) pairs together and pushes the N²−N incorrect pairs apart (a softmax over similarities, both directions).

**Why this matters:** it learns a **joint vision-language space** from cheap web image-caption pairs (no manual labels), enabling **zero-shot classification** (compare an image's embedding to text embeddings like "a photo of a cat" vs "a photo of a dog") and serving as the visual backbone for many VLMs.

### 1.2 Vision Transformers (ViT) & patch embeddings

ViT applies the transformer to images by splitting an image into fixed **patches** (e.g. 16×16), linearly embedding each patch into a vector ("tokens"), adding positional encodings, and running standard transformer layers. **Key point:** an image becomes a *sequence of patch tokens*, so the same attention machinery from Day 2 applies. This is what made unified multimodal transformers possible.

### 1.3 Fusion strategies (how vision meets language in a VLM)

| Strategy | How | Trade-off |
|---|---|---|
| **Early fusion** | Concatenate image tokens with text tokens, one joint transformer | Deep interaction; more compute |
| **Late fusion** | Encode each modality separately, combine at the end (like CLIP) | Cheap, modular; shallow interaction |
| **Cross-attention** | Text attends to image features via cross-attention layers (Flamingo-style) | Strong interaction, efficient; the common VLM pattern |

The modern recipe for an LLM-based VLM: a **vision encoder** (often CLIP-style ViT) produces image tokens → a **projection/adapter** maps them into the LLM's token space → the **LLM** processes image + text tokens together. You're essentially teaching a pretrained LLM to "see" by feeding it visual tokens.

### 1.4 VLM hallucination (the signature failure)

VLMs often **describe things not in the image** (objects, text, attributes) because the language prior is strong and the visual grounding is weak — the model "guesses" plausible content. Especially bad for: counting, precise spatial relations, OCR-heavy images, and fine detail. Evaluating this needs grounded, adversarial benchmarks (does the described object actually exist?).

> **In Practice — VLM symptoms.** *See* the model confidently describe an object that isn't in the image → *suspect* language-prior dominance / weak visual grounding → *look at* grounding evals (e.g. POPE-style object-existence probes) and the vision-encoder resolution. *See* failures specifically on text-in-images → *suspect* OCR/resolution limits → *look at* image tokenization resolution.

### Self-test §1
1. **How does CLIP work?** → Jointly trains image + text encoders with a contrastive loss so matching pairs are close in a shared embedding space; enables zero-shot classification from cheap web pairs.
2. **How does a ViT process an image?** → Splits into patches → linearly embeds each as a token → adds positions → standard transformer. Image = sequence of patch tokens.
3. **Fusion strategies for VLMs?** → Early (concat tokens, one transformer), late (separate encoders, combine at end — CLIP), cross-attention (text attends to image — Flamingo). LLM-VLMs: vision encoder → projector → LLM.
4. **Why do VLMs hallucinate?** → Strong language prior overrides weak visual grounding; worst on counting, spatial, OCR, fine detail.

---

## §2 — Audio & video basics

You need enough to be *conversant*, not deep.

### 2.1 Audio

- **Spectrograms:** audio is often converted to a **mel-spectrogram** (a time × frequency image of the signal), then processed like an image/sequence — this is how models like Whisper handle speech.
- **Audio tokenization:** alternatively, discretize audio into tokens with a neural codec (e.g. residual vector quantization) so a transformer can model sound like text — enabling audio generation and unified audio-text models.
- **Temporal modeling:** audio is long and sequential; the challenge is handling long time spans efficiently.

### 2.2 Video

- **Frame sampling:** video is huge (many frames/sec), so models **sample frames** (e.g. 1 fps) rather than process every frame — a quality/cost tradeoff.
- **Long-context challenge:** even sampled, video produces *many* tokens (each frame = many patch tokens), hitting the O(seq²) attention wall hard. Long video is a frontier problem precisely because of token count.
- **Temporal reasoning:** understanding *order* and *change* over time (not just per-frame content) is the hard, under-solved part — many "video" benchmarks are actually solvable from a single frame, which is itself an eval-design pitfall (ties to Day 3).

> **In Practice — video symptoms.** *See* a "video" model scoring well but you suspect it ignores time → *suspect* the benchmark is single-frame-solvable → *look at* whether shuffling/removing frames changes the answer (a temporal-understanding probe). *See* OOM/cost blowup on long video → *suspect* token explosion from frames → *look at* frame-sampling rate and token-reduction.

### Self-test §2
1. **How do models handle audio?** → Convert to mel-spectrogram (time×freq "image") or tokenize via a neural codec, then transformer. Challenge: long temporal spans.
2. **Why is long video hard?** → Each frame = many tokens; sampling helps but token count still explodes against O(seq²) attention; temporal reasoning (order/change) is the unsolved part.

---

## §3 — Agents & tool use

> "Agent" = an LLM that **takes actions** in a loop (calling tools, observing results, deciding next steps) to accomplish a goal, rather than emitting a single response.

### 3.1 Tool / function calling

The model is given tool schemas (name, description, arguments) and can **emit a structured call** (e.g. JSON) that the system executes, returning the result into the context. This lets the LLM use calculators, search, code execution, APIs, databases — overcoming its fixed knowledge and poor arithmetic. The model decides *when* and *how* to call; the harness executes and feeds back the observation.

### 3.2 ReAct — the canonical agent loop

**ReAct = Reasoning + Acting**, interleaving **Thought → Action → Observation** repeatedly:
- **Thought:** the model reasons about what to do next.
- **Action:** it calls a tool.
- **Observation:** the tool result is added to context.
- Loop until the task is solved.

Why it works: interleaving reasoning with acting lets the model **plan, use external feedback to correct itself, and adjust** — far more robust than a single forward pass. Variants: **planning-first** agents (make a full plan, then execute) vs **reactive** agents (decide step-by-step). Real systems blend both.

### 3.3 Memory, retrieval, environment feedback

- **Memory:** context windows are finite, so agents need external memory (scratchpads, summaries, vector stores) to persist information across long tasks.
- **Retrieval:** pull relevant info on demand (see §4, RAG).
- **Environment feedback:** the agent learns from *results* (error messages, test failures, web pages) — this loop is what makes agents powerful and is increasingly an **RL signal** (reward = task success).

### 3.4 Evaluating agents (a favorite question)

> **"How would you evaluate a computer-use / coding agent?"** — use the Day 3 framework, specialized:

1. **Define tasks + success criteria** precisely (verifiable end states — file created, test passes, correct booking).
2. **Decompose** the eval: perception, planning, tool execution, and **error recovery** are separate skills — measure them, not just final success.
3. **Metrics beyond success rate:** step count / efficiency, latency, cost, and **unsafe actions** (did it delete files, leak data, make unauthorized purchases?).
4. **Adversarial + distribution-shift cases**, including **prompt injection** (malicious content in a web page / tool output hijacking the agent).
5. **Robustness:** does it recover from a failed action, or spiral?
6. **Human audit** for ambiguous tasks; **track regressions** across model versions.
7. **Sandboxing** the eval so a misbehaving agent can't cause real harm.

**Why agent eval is hard:** tasks are long-horizon and multi-step, so partial credit and error attribution are tricky; success is often binary and sparse; and environments are stochastic. State this — it shows you understand the difficulty.

### 3.5 Agent safety

- **Sandboxing:** restrict what actions are possible (no real money, no destructive ops without confirmation).
- **Prompt injection:** the top agent vulnerability — untrusted content in the context can hijack the agent's goals. Defenses: separating trusted instructions from untrusted data, permission boundaries, human confirmation for high-impact actions.
- **Safety boundaries / human-in-the-loop** for irreversible or outward-facing actions.

> **In Practice — agent symptoms.** *See* an agent that solves easy tasks but spirals on hard ones → *suspect* poor error recovery / no replanning → *look at* the recovery sub-skill and loop termination. *See* an agent follow instructions hidden in a web page → *suspect* prompt injection → *look at* trusted/untrusted context separation + permissioning. *See* high success but high cost → *suspect* inefficient looping → *look at* step-count metrics.

### Self-test §3
1. **What is tool/function calling?** → The model emits a structured call against given tool schemas; the system executes and returns the result into context — overcoming fixed knowledge / weak arithmetic.
2. **What is ReAct?** → Interleaved Thought→Action→Observation loop; reasoning + external feedback lets the agent plan and self-correct.
3. **How would you evaluate a computer-use agent?** → Define verifiable tasks/success; decompose perception/planning/execution/recovery; measure success rate, steps, latency, cost, unsafe actions; adversarial + prompt-injection cases; human audit; sandbox; track regressions.
4. **What is the top agent security risk?** → Prompt injection (untrusted content hijacking goals). Defend via trusted/untrusted separation, permissions, human confirmation.

---

## §4 — RAG & retrieval (bonus, high-ROI)

Retrieval-Augmented Generation: fetch relevant documents and put them in context so the model answers from **up-to-date, grounded** information instead of (or alongside) its parameters.

- **Dense retrieval** (embed query + docs, nearest-neighbor via ANN) vs **sparse** (BM25, keyword) vs **hybrid** (both). **Reranking** a shortlist with a stronger model improves precision. **Chunking** splits docs; **query rewriting** improves recall.
- **RAG vs fine-tuning (classic question):** RAG for **knowledge that changes / is large / needs citation** (facts, docs, freshness); fine-tuning for **behavior, format, style, or skills**. Knowledge → RAG; behavior → fine-tune. Often both.
- **Why more retrieved context can *hurt*:** distraction by irrelevant passages, "lost in the middle" (models under-use mid-context info), conflicting sources, and pushing useful content out of the window. More ≠ better.
- **Prompt injection via retrieved docs:** retrieved content is untrusted — it can carry malicious instructions. Same defense as agents.

### Self-test §4
1. **RAG vs fine-tuning — when each?** → RAG for changing/large/citable knowledge + freshness; fine-tuning for behavior/format/skills. Knowledge vs behavior. Often combined.
2. **Why can more retrieved context hurt?** → Irrelevant-passage distraction, lost-in-the-middle, conflicting sources, evicting useful tokens.
3. **How do you evaluate retrieval quality?** → Retrieval metrics (recall@k, precision, MRR/nDCG) separately from end-answer quality; and grounding/citation correctness.

---

## §5 — FULL MOCK DRILL (the main event)

Spend the back half here. **Rules:** say each answer **out loud**, **timed (~2 min)**, *then* check the section. Flag every stumble for a final review pass. This simulates the real loop.

### ML fundamentals (Day 1)
1. Explain the bias-variance tradeoff. *(prep beyond these docs)*
2. Why does regularization help?
3. Derive the logistic regression loss. *(prep)*
4. What is cross-entropy? → D1 §2
5. What is KL divergence? → D2 §6 (RLHF), D3
6. Explain calibration.
7. How do you handle class imbalance?
8. How do you detect data leakage? → D3 §1–2
9. Why might validation performance not predict production? → D1 §6 (dist shift, contamination)
10. How would you debug overfitting? → D1 §6

### Deep learning (Day 1)
11. Derive backprop for a two-layer net. → D1 §2
12. Why do residual connections help? → D1 §4.3
13. BatchNorm vs LayerNorm. → D1 §4.2
14. Adam vs SGD. → D1 §5.1
15. Why learning-rate warmup? → D1 §5.3
16. Why gradient clipping? → D1 §5.4
17. What causes exploding gradients? → D1 §2.4
18. What causes loss spikes? → D1 §6
19. How do you debug NaNs? → D1 §6
20. Why does mixed precision help? → D1 §5.5

### Transformers / LLMs (Day 2)
21. Derive self-attention. → D2 §2
22. Why divide by √d_k? → D2 §2.3
23. What is causal masking? → D2 §2.4
24. What is the KV cache? → D2 §4.1
25. Why is decoding slower than prefill? → D2 §4.2
26. How do you extend context length? → D2 §1.3/§4.3
27. What is perplexity? → D2 §5.2
28. Why can lower perplexity fail to improve chat quality? → D2 §5.2
29. What is instruction tuning? → D2 §6.1
30. RLHF vs DPO. → D2 §6.4

### Frontier model work (Days 2–3)
31. How would you choose a pretraining data mixture? → D3 §3.1
32. How would you detect contamination? → D3 §2.3
33. How would you evaluate reasoning? → D3 §6
34. How would you evaluate a coding model? → D3 §6
35. How would you evaluate an agent? → D4 §3.4
36. How do you design an ablation? → D3 §3.3
37. What would you do with more compute? → D2 §5.3 (Chinchilla: scale data; weigh inference; post-training ROI)
38. What would you do with limited compute? → small-scale ablations, proxy tasks, then scale winners
39. How do you decide whether to scale an experiment? → cheap proxy first; scale only if it improves on a real held-out metric compute-matched
40. How do you know a result is real? → D3 §3.3/§4.3 (compute-matched + CIs + contamination checks)

### Systems (conversant level)
41. Data parallelism vs tensor parallelism. → DP = replicate model, split data, all-reduce grads; TP = split each matmul across devices.
42. What is pipeline parallelism? → split *layers* across devices, micro-batch to keep them busy.
43. What is FSDP/ZeRO sharding? → shard params/grads/optimizer-state across devices to fit big models.
44. Why is training memory expensive? → params + grads + optimizer state (Adam = 2× params) + activations.
45. Estimate memory for a 70B model. → ~140GB params in FP16; +grads +Adam state ≈ several× → needs sharding across many GPUs.
46. What is activation checkpointing? → D1 §5.5 (recompute activations in backward to save memory).
47. How does quantization affect inference? → smaller/faster (int8/int4) at some accuracy cost; great for serving.
48. What is speculative decoding? → D2 §4.2 (draft model proposes, big model verifies in parallel).
49. How do you reduce serving latency? → batching, KV-cache/GQA, speculative decoding, quantization, shorter outputs.
50. How do you monitor model regressions? → D3 §5.4 (versioned regression suite + online metrics).

> **After the drill:** make a list of every question you stumbled on. Those are tomorrow-morning's review. Most candidates know the easy ones — the loop is decided by the ones you fumbled.

### What "great" sounds like today
- On multimodal/agents you're **fluent on mechanism and evaluation** without overclaiming depth.
- In the drill you **derive and reason**, connect across days (e.g. "this is the same Goodhart problem as evals," "KV cache is why GQA exists"), and stay **compute-aware**.

---

## Quick-reference cheat sheet

- **CLIP:** contrastive image+text encoders → shared embedding space → zero-shot. **ViT:** image = patch tokens → transformer.
- **VLM fusion:** early (concat) / late (CLIP) / cross-attention (Flamingo); LLM-VLM = vision encoder → projector → LLM. **Hallucination** = language prior > visual grounding.
- **Audio** = mel-spectrogram or codec tokens. **Video** = sampled frames → token explosion → long-context wall; temporal reasoning is the hard part.
- **Agents:** tool/function calling (structured calls) + **ReAct** (Thought→Action→Observation loop) + memory + environment feedback. Evaluate by verifiable tasks, decomposed sub-skills, success/steps/cost/unsafe-actions, prompt-injection adversarial, sandboxed, regression-tracked. **Prompt injection** is the top risk.
- **RAG:** dense/sparse/hybrid + rerank; **knowledge → RAG, behavior → fine-tune**; more context can hurt (lost-in-the-middle, distraction). Retrieved docs are untrusted.
- **Mock drill:** out loud, timed, flag stumbles — that list is your highest-value review.
