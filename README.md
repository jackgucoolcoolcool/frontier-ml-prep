# Frontier ML Interview Prep

Self-contained study site for frontier-lab research scientist interviews: a 7-day fundamentals program, research deep dives (architecture & multimodal, RL & post-training, data & scaling), a research proposal, coding practice, and interview strategy. Everything is plain HTML — no build step.

## How to use

**Browse online:** https://jackgucoolcoolcool.github.io/frontier-ml-prep/ — the styled index for the whole site.

**Or run locally:**

```bash
git clone https://github.com/jackgucoolcoolcool/frontier-ml-prep.git
cd frontier-ml-prep
python3 -m http.server 8000   # then open http://localhost:8000/home.html
```

(Opening `home.html` directly with `open home.html` also works; the pages have no external dependencies.)

The sections below mirror `home.html` so you can jump straight to a page on GitHub.

## The 7-day plan

| Day | Page |
|---|---|
| 1 | [Deep Learning Fundamentals](Day1_Deep_Learning_Fundamentals.html) |
| 2 | [Transformers & LLMs](Day2_Transformers_and_LLMs.html) |
| 3 | [Data & Evaluation](Day3_Data_and_Evaluation.html) |
| 4 | [Multimodal & Agents](Day4_Multimodal_and_Agents.html) |
| 5 | [Research Taste & Papers](Day5_Research_Taste_and_Papers.html) |
| 6 | [Post-Training, Sampling & RL](Day6_PostTraining_Sampling_and_RL.html) |
| 7 | [100 Drills](Day7_100_Drills.html) |

## Deep dives — architecture, vision & multimodal

- [LLM Architecture Frontier](LLM_Architecture_Frontier.html) — MoE, attention variants, long context
- [ViT Training Tutorial](ViT_Training_Tutorial.html) — how a ViT is trained, end to end
- [VLM / VLA → Unified Omni models](VLM_VLA_Unified_Omni.html)
- [MMU ⇄ MMGen Transfer](MMU_MMGen_Transfer_Deep_Dive.html) — understanding ↔ generation transfer
- [Self-Improving VLMs](Self_Improving_VLMs.html)

## Deep dives — RL & post-training

- [Post-Training & RL Mechanics](Post_Training_and_RL_Deep_Dive.html) — SFT → RM → RLHF/DPO/GRPO, the full pipeline
- [Agentic RL: Compaction & Staleness](Agentic_RL_Compaction_and_Staleness.html) — long-horizon agent training issues
- [Scaled RL Data Brainstorm](Scaled_RL_Data_Brainstorm.html) — where verifiable RL data at scale comes from

## Deep dives — science of data & scaling

- [Science of Multimodal Data — Master](Science_of_MM_Data_Master.html)
- [Scaling behaviors](Science_of_MM_Data_Scaling.html)
- [Project designs](Science_of_MM_Data_Projects.html)
- [Scaling-law methods](Science_of_MM_Data_ScalingLaw_Methods.html)

## Vision Banana — paper deep dive & job talk

- [Vision Banana — Deep Dive](Vision_Banana_Deep_Dive.html) — perception as RGB generation (15 Q&As + colormap demo)
- [Vision Banana — Coverage Map & Primer](Vision_Banana_Coverage_and_Primer.html)
- [Vision Banana — Job Talk](Vision_Banana_Job_Talk.html)

## Research proposal

- [LatentFusion — Predicting in Frozen SSL Latent Spaces](Proposal_JEPA_Latent_VLM.html)

## Coding practice

- [Mock Interview Simulator](mock_interview/) — timed problems, variants, history, gap report (`python app.py`)
- [ML Coding From Scratch](Coding_Implementations_From_Scratch.html) — attention, transformer block, DPO/RM/PPO losses, sampling
- [RL Coding Interview Pack](RL_Coding_Interview_Pack.html) — whiteboard math + full REINFORCE / PPO / GRPO / RLOO loops
- [Sampling & RL Fundamentals](Fundamentals_Sampling_and_RL.html) — decoding from scratch + classical RL
- [ML Debugging — 50 Interview Sessions](ML_Debugging_50_Sessions.html) and [v2 — 50 fresh sessions](ML_Debugging_50_Sessions_v2.html)
- [Sampling & RL Cheat Sheet (NumPy)](Coding_Prep_Sampling_RL.html)

## Strategy, notes & Q&A

- [Resume Interview Prep](Resume_Interview_Prep.html)
- [Interview Round Strategy](Interview_Round_Strategy.html)
- [Notes: Activation Functions](Notes_Activation_Functions.html)
- [Interview Q&A Log](Interview_QA_Log.html) — running log of every technical question encountered

Some pages have `.md` twins with the same basename — same content, readable directly on GitHub.
