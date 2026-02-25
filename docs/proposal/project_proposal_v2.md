# MAS4CS: A Multi-Agent System for Customer Service Automation

## Overview

This project investigates whether a structured, hierarchical multi-agent architecture can improve the reliability of automated customer service, without relying on larger or more powerful models.

The system is built on **MultiWOZ 2.2**, a standard multi-domain dialogue dataset (hotel, restaurant), and implemented using LangGraph. Five specialized agents handle distinct roles: 
intent detection, policy enforcement, response generation, memory management, and output validation.

See: [GitHub link](https://github.com/GiX007/mas4cs)

---

## Research Question

Does architectural structure alone improve customer service performance (e.g., task completion and policy compliance) over a single-LLM baseline, even when using the same underlying models?

Does architectural structure alone improve customer service performance (task completion and policy compliance) over a single-LLM baseline, even when using the same underlying models? 
And can further optimization through role-specific fine-tuning push performance beyond what general-purpose commercial models achieve?

---

## System Design

The workflow connects five agents in a deterministic graph:
```
User Message → Triage → Policy → Action → Memory → Supervisor
                                    ↑                    |
                                    |_____(retry)________|
```
A shared state object flows through all agents. The Supervisor validates outputs and can trigger a self-correction loop. 

### Tool Usage

In this implementation, agents are structured LLM nodes with distinct roles — not autonomous tool-using agents in the classical sense.  
No external APIs or databases are called. 
The goal is to isolate the effect of architectural structure before introducing real tool complexity. 
Real tool integration is left for future work.

---

## Experiments

Three experiments are designed to isolate the contribution of architecture and model specialization:

- **Experiment 1 — Single-Agent Baseline**
  - One LLM receives a conversation, and policies in a single prompt. No architecture, no state management
  - Tests the baseline of a single-model approach
  - Models: GPT-4o-mini, Claude-3-Haiku, open-source models (3B–7B-14B)

- **Experiment 2 — MAS Graph**
  - Same models, now distributed across agent roles in the LangGraph workflow
    - Homogeneous: same model across all roles 
    - Heterogeneous: different models per role 
  - Compared directly against Experiment 1 results

- **Experiment 3 — Fine-Tuned MAS**
  - Open-source models fine-tuned per agent role using LoRA (Unsloth)
  - Hypothesis: small models trained on a specific task ("experts") can match or exceed large general-purpose commercial models at a fraction of the cost
  - Compared against best results from Experiment 1 and 2

---

## Evaluation

11 metrics across three levels (turn, dialogue, dataset):
- Intent Accuracy, Domain Accuracy, Action-Type Accuracy
- Slot Accuracy, JGA, Hallucination Rate
- Policy Compliance, Task Completion, System Correctness
- Cost per dialogue (USD), Average latency per turn (seconds)
- LLM-judged Response Quality (Optional)

Metrics are computed at three levels: turn, dialogue, and dataset. 

*(A subset of these metrics will be selected as primary evaluation criteria based on experimental findings. Not all 11 are expected to be equally informative across all three experiments.)*

---

## Infrastructure & Limitations

- Local machine: 32GB RAM, NVIDIA GTX 1050 Ti (4GB VRAM). 4GB VRAM limits local inference to 3B models only 
  *(too small for reliable structured output generation, as confirmed during testing)*
- Running strategy:
  - API-based models (GPT-4o-mini, Claude-3-Haiku) — run locally
  - Open-source models (7B+) and fine-tuning — EuropeanHPC

---
