# MAS4CS Model Arsenal

This document lists the language models used in MAS4CS experiments.

---

## Scope

The goal is **model comparison under controlled architectures**, not proposing new models.  
All models are evaluated in:
- single-agent baselines
- structured multi-agent workflows (homogeneous and heterogeneous)
- optimized multi-agent workflows using fine-tuned models

---

## Open-Source Models 
**Platform:** HuggingFace + Unsloth  
**Usage:** local inference, LoRA fine-tuning, and maybe European HPC

### Model Families Covered
- **LLaMA**: 3.2 (3B)
- **Gemma 2**: 9B 
- **Mistral**: Nemo (12B)
- **Qwen 2.5**: 3B, 7B, 14B

### Quantization Policy
- Default: **bnb-4bit** for development and experiments  

---

## Commercial Models (APIs)

Used **only as reference baselines**, not for training.

- **gpt-4o-mini**
- **claude-3-haiku-20240307**

---

## Usage Strategy

### 1. Stack Configurations
- **Homogeneous**: Uses the same model family across all nodes to isolate the effect of the multi-agent architecture on performance
- **Heterogeneous**: Assigns different models per role to optimize performance:
  - small / fast (e.g., 3B) → Triage, Memory
  - larger / stronger (e.g., 14B) → Action, Supervisor

### 2. Fine-Tuning Approach
A role-specific fine-tuning is performed to transform general models into "specialized experts":
- Target: LoRA (Low-Rank Adaptation) fine-tuning of Open-Source models (3B–8B) using the Unsloth framework
- Dataset: Training on MultiWOZ 2.2 on specific domains
- Goal: To test if a "small but expert" model can match or exceed the performance of a "large but general" model in structured customer service tasks

---

## Notes

- **Unified Interface**: All models are wrapped behind a single API-agnostic layer
- **Zero-Logic Swapping**: Switching models requires no changes to the agent's core code
- **Output Normalization**: Strict JSON enforcement across all providers

---
