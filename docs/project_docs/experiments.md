# MAS4CS Experiments

## Overview

Three experiments isolate the effect of architecture and model specialization on customer service performance, using the same evaluation metrics across all runs for direct comparison.

All experiment settings (models, splits, limits) are defined in `src/experiments/config.py`. 
Results are saved to `results/experiments/` as timestamped JSON files and a running `leaderboard.txt`.

---

## Experiment 1: Single-Agent Baseline

**Question:** Can a single LLM handle customer service complexity without architectural help?

**Setup:** One LLM receives the full conversation history, available services, and policy rules. Tool usage is simulated via two LLM calls per turn: Call 1 extracts domain, intent, and slots and the code runs a DB query using those slots. Call 2 receives the DB results and generates a grounded natural language response. This two-call design mirrors what a real tool-calling LLM would do and keeps Exp1 directly comparable to Exp2. No state management, no multi-agent orchestration.

**Models tested:**
- API-based: `gpt-4o-mini`, `claude-3-haiku-20240307`
- Open-source: Llama 3.2 (3B), Qwen 2.5 (3B, 7B, 14B), Gemma 2 (9B), Mistral Nemo (12B)

**Purpose:** Establishes the performance baseline of a single model as a reference point for Experiments 2 and 3.

---

## Experiment 2: MAS Graph

**Question:** Does a structured multi-agent architecture improve performance over the single-agent baseline, using the same underlying models?

**Setup:** Same models as Experiment 1, now distributed across agent roles (Triage, Policy, Action, Supervisor, Memory) in the LangGraph workflow. Tool usage is rule-based and deterministic as the Action agent's code routes to `find_entity` or `book_entity` directly based on intent, with no LLM tool decision. Two architecture types are tested:

- **Homogeneous**: same model across all agent roles, which isolates the effect of architecture alone
- **Heterogeneous**: different models per role (e.g., GPT for Triage + Action, Haiku for Supervisor) that tests whether cross-model supervision reduces failures

**Purpose:** Does the MAS architecture itself improve performance over a single-LLM baseline?

---

## Experiment 3: MAS Graph with Fine-Tuned Models

**Question:** Can role-specific fine-tuned open-source models match or exceed general-purpose commercial models?

**Setup:** LoRA fine-tuning of open-source models (3B–14B) using Unsloth on MultiWOZ 2.2, run on EuroHPC (Leonardo Booster partition). Each model is fine-tuned on role-specific subsets of the training data. Results are compared against the best Experiment 1 and 2 configurations.

**Purpose:** Tests whether "small but expert" models can outperform "large but general" models in a structured MAS at a fraction of the cost.

---

## Results

All results saved to `results/runs/run_*/` per run:
- `exp{N}_{model}_{timestamp}_dataset.json` — aggregated metrics across all dialogues
- `exp{N}_{model}_{timestamp}_dialogues.json` — per-dialogue metrics
- `exp{N}_{model}_{timestamp}_turns.json` — per-turn metrics
- `leaderboard.txt` — running comparison table across all experiments and configs

Additionally, results are saved per domain (hotel / restaurant separately) for detailed analysis.

---

## Infrastructure Notes

**Local machine:** 32GB RAM, NVIDIA GTX 1050 Ti (4GB VRAM). 
LLMs run on GPU and 4GB VRAM limits local inference to 3B models only, which proved too small for reliable structured output generation during testing.

**Running strategy:**
- API-based models (`gpt-4o-mini`, `claude-3-haiku`) run locally 
- Open-source models (7B+) and all LoRA fine-tuning run on EuroHPC (Leonardo Booster)

---
