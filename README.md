# Medical Doctor Agent — Verifier-Guided GRPO for Structured Clinical Reasoning

## Introduction

This repository turns compact LLMs into transparent, verifiable medical reasoners. We pair a tight output grammar (`<THINK>...</THINK>` → `<ANSWER>...</ANSWER>`) with a two-stage post-training pipeline:

1. **Supervised Fine-Tuning (SFT)** to learn the format and basic chain-of-thought
2. **Group-Relative Policy Optimization (GRPO)** with a multi-reward signal that jointly optimizes strict/partial format adherence and semantic answer correctness using an LLM verifier (RL with Verifiable Rewards, "RLVR")
<img width="1401" height="791" alt="Screenshot 2025-11-12 at 18 11 18" src="https://github.com/user-attachments/assets/62b04535-6a70-4f4f-83b8-c5141c91fdc1" />

The result is a doctor-style agent that explains its reasoning, emits machine-readable answers, and is harder to "reward-hack".

**Why this matters:** Traditional RLHF often relies on subjective preference models and single scalar rewards that can be gamed (e.g., longer responses). Here, we replace subjectivity with programmable, verifiable rewards and compare completions relative to peers (group-relative advantages), aligning learning with what clinicians and downstream systems actually need: clear structure + correct conclusions.

---

## Key Contributions

- **RLVR instead of RLHF/RLAIF:** Use an AI verifier to check objective alignment with ground truth (handles aliases/synonyms); no learned preference model required.
- **GRPO + Dr-GRPO:** Group-relative advantages for stability; length-bias removal so gains aren't tied to verbosity.
- **Multi-reward design:** (i) strict format, (ii) soft/partial format, (iii) verifier-based correctness; all programmable and cheap to compute.
- **Efficient adaptation:** LoRA (r=32, α=64) on attention & MLP projections, mixed precision, gradient checkpointing—single-GPU friendly.

---

## Datasets

### Training Data
- **SFT:** medical-o1 reasoning SFT — high-quality CoT traces to teach the `<THINK>`/`<ANSWER>` protocol.
- **RL:** medical-o1 verifiable problems — open-ended questions with verifiable answers for the LLM-verifier reward.

### Evaluation Benchmarks
- **MedQA-USMLE** (test, n=1,273): 4-option clinical reasoning questions.
- **MedMCQA** (validation, n=4,183): large-scale medical MCQ across 21 subjects.

---

## Methodology

### Two-Stage Training

1. **SFT (format learning):** We first align the model to emit structured, machine-readable outputs: reasoning inside `<THINK>...</THINK>`, decision in `<ANSWER>...</ANSWER>`. This dramatically stabilizes RL by making rewards reliable and extraction trivial.
<img width="386" height="675" alt="Screenshot 2025-11-12 at 18 13 04" src="https://github.com/user-attachments/assets/e2530553-65ad-4c9d-ab8d-e89c5a1336fd" />

2. **GRPO (reasoning optimization):** For each prompt, we sample a group of completions, compute composite rewards, normalize within-group, and update with a clipped PPO-style surrogate (no value network). We use Dr-GRPO to remove sequence-length bias.
<img width="860" height="694" alt="Screenshot 2025-11-12 at 18 13 22" src="https://github.com/user-attachments/assets/5fc502d3-7ddd-4ee6-a822-d6c47c6330ac" />

### Why GRPO over PPO

- **No critic/value network:** Lower memory, fewer moving parts, less variance on long CoT sequences.
- **Relative, not absolute:** Per-prompt race within a heat (mean/std normalization) stabilizes gradients across mixed-difficulty questions.
- **Reward-hacking resistant:** Comparing siblings shifts focus from "how high is the raw score" to "which completion is better."

### Why RLVR over RLHF/RLAIF

- **Objective, not subjective:** The verifier checks ground-truth consistency (with alias handling) instead of learning human preferences.
- **Cheaper & scalable:** Programmable rewards (regex + verifier logits)—no expensive human label loops.
- **Task-aligned:** Optimizes format + correctness, the two properties clinicians and pipelines need.

### LoRA (parameter-efficient fine-tuning)

- **Config:** r=32, α=64, dropout=0.1
- **Targets:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Rationale:** Maximize reasoning gains per GPU hour while preserving base model priors; enables rapid ablation across backbones.
<img width="744" height="601" alt="Screenshot 2025-11-12 at 18 14 34" src="https://github.com/user-attachments/assets/f9a5e042-7188-4217-ad28-79191bff6cb1" />

---

## What's New vs. Prior Work

- Multi-signal, verifiable rewards in place of a single noisy scalar; no learned reward model.
- Group-relative advantages that stabilize long-sequence RL and directly track "better answers".
- Length-agnostic optimization (Dr-GRPO) to prevent verbosity from masquerading as competence.
- End-to-end transparency: streaming `<THINK>` and a compact `<ANSWER>` with one-click reveal of RAG contexts for provenance.

---

## Results
<img width="409" height="306" alt="Screenshot 2025-11-12 at 18 15 44" src="https://github.com/user-attachments/assets/8970c3e0-d74d-456d-9394-a5754fa83c50" />

<img width="1233" height="544" alt="Screenshot 2025-11-12 at 18 16 08" src="https://github.com/user-attachments/assets/f26c668f-e4fc-4752-8317-99ed371a92a3" />

**Takeaways:** GRPO beats PPO on the same backbone (+3.93 pp MedQA, +6.03 pp MedMCQA). Base+GRPO approaches or exceeds Instruct+PPO (−0.70 pp on MedQA; +4.38 pp on MedMCQA), showing that group-relative, multi-reward RL can bootstrap reasoning even from a non-instruction-tuned model. The overall best is Instruct+GRPO.

### Before/After SFT+GRPO (Format vs Answer vs Both)

#### MedQA
- **Instruct:** Format 50.75% → 100.00% (+49.25 pp), Answer 29.46% → 49.41% (+19.95 pp), Both 29.46% → 49.41% (+19.95 pp)
- **Base:** Format 41.71% → 99.84% (+58.13 pp), Answer 18.77% → 44.78% (+26.01 pp), Both 18.77% → 44.78% (+26.01 pp)

#### MedMCQA
- **Instruct:** Format 75.35% → 100.00% (+24.65 pp), Answer 38.27% → 46.07% (+7.80 pp), Both 38.27% → 46.07% (+7.80 pp)
- **Base:** Format 36.89% → 99.93% (+63.04 pp), Answer 15.66% → 44.42% (+28.76 pp), Both 15.66% → 44.42% (+28.76 pp)

**What the model actually learned:** Structure saturates to ~100%—SFT nails the grammar and GRPO maintains it—while semantic correctness is where the big gains arrive (e.g., +28.76 pp on MedMCQA for Base). This pattern is exactly what we want: format is necessary but not sufficient; GRPO's verifier-guided reward shifts probability mass toward factually correct conclusions.

---

## System & GUI

We ship a Gradio ChatInterface that streams `<THINK>` (complex chain-of-thought) and then reveals a compact `<ANSWER>`. Two toggles provide transparency on demand:
<img width="1421" height="900" alt="Screenshot 2025-11-12 at 18 17 12" src="https://github.com/user-attachments/assets/b09dd791-53f9-4a58-8ce3-9f2a4a7cd646" />

- **RAG Contexts:** Retrieved snippets grounding the answer
<img width="1552" height="920" alt="Screenshot 2025-11-12 at 18 18 00" src="https://github.com/user-attachments/assets/0b70c55f-5c6c-49d8-befd-3e39e70c31fb" />
- **Thinking:** The full reasoning trace
<img width="1598" height="924" alt="Screenshot 2025-11-12 at 18 18 37" src="https://github.com/user-attachments/assets/3c3bd7da-fbf5-471a-b2ac-b9547f3153f8" />

---

## Limitations

- **Reward & hyper-parameter sensitivity:** Verifier thresholds/weights matter.
- **Semantic variance remains:** Structure saturates early; correctness gains can be bimodal on hard cases.
- **Compute:** RL adds overhead beyond SFT (LoRA alleviates but does not remove).
- **Training time:** Short runs limit full convergence and ablation coverage.

---

## Future Work

- **Backbones:** Extend to Gemma, Mistral, Llama families and larger Qwen/Llama scales.
- **Verifier:** Ensembles, calibration, and richer checks (e.g., contraindications, safety).
- **Longer RL runs:** Curriculum schedules; harder verifiable tasks; retrieval-aware rewards.

---

## Ethics Note

This project is a research prototype. It does not provide medical advice. Any clinical use requires rigorous validation, governance, and human oversight.

---

## Contact

Questions, issues, or collaboration ideas? Open a GitHub issue or reach out to our team.

---

## License

See [LICENSE](LICENSE) for details.
