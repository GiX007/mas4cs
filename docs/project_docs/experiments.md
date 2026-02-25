# MAS4CS Experiments

## Overview

Three experiments are designed to isolate the effect of architecture and model  specialization on customer service performance, using the same evaluation metrics across all runs for direct comparison.

All experiment settings (models, splits, limits) are defined in `src/experiments/config.py`.
Results are saved to `results/experiments/` as timestamped JSON files + a running `leaderboard.txt`.

---

## Experiment 1: Single-Agent Baseline

**Question:** Can a single LLM handle customer service complexity without architectural help?

**Setup:**
- One LLM receives the full conversation history, available services, and policy rules in a single mega-prompt
- No tools, no state management, no multi-agent orchestration
- Model must output structured JSON with: domain, intent, slots, action_type, policy_satisfied, response

**Models tested:**
- API-based: `gpt-4o-mini`, `claude-3-haiku-20240307`
- Open-source: Llama 3.2 (3B), Qwen 2.5 (3B, 7B, 14B), Gemma 2 (9B), Mistral Nemo (12B)

**Purpose:** Establishes the performance baseline of a single model as a reference point for Experiments 2 and 3.

---

## Experiment 2: MAS Graph

**Question:** Does a structured multi-agent architecture improve performance over the single-agent baseline?

**Setup:**
- Same models as Experiment 1, now split across agent roles (Triage, Policy, Action, Memory, Supervisor)
- Two architecture types tested:
  - **Homogeneous**: same model across all agent roles — isolates the effect of architecture
  - **Heterogeneous**: different models per role (small/fast for Triage, larger/stronger for Supervisor)

**Purpose:** Isolates whether the MAS architecture itself drives performance gains.

---

## Experiment 3: MAS Graph with Fine-Tuned Models

**Question:** Can role-specific fine-tuned open-source models match or exceed general-purpose commercial models?

**Setup:**
- LoRA fine-tuning of open-source models (3B–14B) using Unsloth on MultiWOZ 2.2
- Each model fine-tuned for its specific agent role ideally, but here general fine-tune on MultiWOZ 2.2 for all models
- Best configuration from Experiment 2 used as the architecture baseline
- Compare against baseline and a large giant model (e.g., gpt-4o-mini)

**Purpose:** Tests whether "small but expert" models can outperform "large but general" models in a structured MAS.

---

## Results

All results saved to `results/experiments/` per run:
- `exp{N}_{model}_{timestamp}_dataset.json` — aggregated metrics across all dialogues
- `exp{N}_{model}_{timestamp}_dialogues.json` — per-dialogue metrics
- `exp{N}_{model}_{timestamp}_turns.json` — per-turn metrics
- `leaderboard.txt` — running comparison table across all experiments

---

## Infrastructure Notes

Local machine: 32GB RAM, NVIDIA GTX 1050 Ti (4GB VRAM).

LLMs run on GPU, not RAM — 4GB VRAM limits local inference to 3B models. 
For example, during testing, 3B models struggled to follow structured JSON instructions reliably.

**Running strategy:**
- API-based models (`gpt-4o-mini`, `claude-3-haiku`) — run locally, no GPU required
- Open-source models (7B+) — run on EuropeanHPC

---

