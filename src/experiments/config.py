"""
Experiment Configuration for MAS4CS.

Single source of truth for all experiment parameters.
"""

import os
from typing import Any


# Dataset
DATASET_PATH: str = os.path.join("dataset", "mw22_filtered.json")


# Shared Settings
MAX_DIALOGUES: int | None = None # None = full split
MAX_TOKENS: int = 512
TEMPERATURE: float = 0.0
SPLIT: str = "validation"
RESULTS_DIR: str = "results/experiments"


# Experiment 1: Single-Agent Baseline
EXP1_MODELS: list[str] = [

    # Open-Source small models (3B)
    # "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    # "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    #
    # # Open-Source medium models (7-9B)
    # "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    # "unsloth/gemma-2-9b-it-bnb-4bit",
    #
    # # Open-Source large models (>12B)
    # "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    # "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",

    # API-based models
    "gpt-4o-mini",
    "claude-3-haiku-20240307",

]


# Experiment 2: MAS Graph
EXP2_CONFIG: dict[str, Any] = {

    # Homogeneous architectures with small, medium, large Open-Source models and API-based models
    # "homogeneous_1": {
    #     "triage":     "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    #     "action":     "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    #     "supervisor": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    # },
    # "homogeneous_2": {
    #     "triage":     "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    #     "action":     "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    #     "supervisor": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    # },
    # "homogeneous_3": {
    #     "triage":     "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    #     "action":     "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    #     "supervisor": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    # },
    "homogeneous_gpt": {
        "triage":     "gpt-4o-mini",
        "action":     "gpt-4o-mini",
        "supervisor": "gpt-4o-mini",
    },
    "homogeneous_haiku": {
        "triage":     "claude-3-haiku-20240307",
        "action":     "claude-3-haiku-20240307",
        "supervisor": "claude-3-haiku-20240307",
    },

    # Heterogeneous architectures with small, medium, large Open-Source models and API-based models
    # "heterogeneous_1": {
    #     "triage":     "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    #     "action":     "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    #     "supervisor": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    # },
    # "heterogeneous_2": {
    #     "triage":     "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    #     "action":     "unsloth/gemma-2-9b-it-bnb-4bit",
    #     "supervisor": "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    # },
    # "heterogeneous_3": {
    #     "triage":     "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    #     "action":     "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    #     "supervisor": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    # },
    # "heterogeneous_4": {
    #     "triage":     "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    #     "action":     "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    #     "supervisor": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    # },
    "heterogeneous_5": {
        "triage":     "gpt-4o-mini",
        "action":     "claude-3-haiku-20240307",
        "supervisor": "gpt-4o-mini",
    },
    # "heterogeneous_6": {
    #     "triage":     "claude-3-haiku-20240307",
    #     "action":     "gpt-4o-mini",
    #     "supervisor": "claude-3-haiku-20240307",
    # },
    # "heterogeneous_7": {
    #     "triage":     "claude-3-haiku-20240307",
    #     "action":     "claude-3-haiku-20240307",
    #     "supervisor": "gpt-4o-mini",
    # },
    # "heterogeneous_8": {
    #     "triage":     "gpt-4o-mini",
    #     "action":     "gpt-4o-mini",
    #     "supervisor": "claude-3-haiku-20240307",
    # },
}

# Experiment 3: MAS Graph with Fine-Tuned Model
EXP3_MODELS: dict[str, Any] = {
    # "finetuned_heterogeneous_1": {
    #     "models": {
    #         "triage":     "finetuned/Qwen2.5-3B-triage-lora",
    #         "action":     "finetuned/Qwen2.5-7B-action-lora",
    #         "supervisor": "finetuned/Qwen2.5-14B-supervisor-lora",
    # },
    # "finetuned_heterogeneous_2": {
    #     "models": {
    #         "triage":     "finetuned/Qwen2.5-3B-triage-lora",
    #         "action":     "finetuned/Qwen2.5-7B-action-lora",
    #         "supervisor": "finetuned/Qwen2.5-14B-supervisor-lora",
    # },
    # "finetuned_heterogeneous_3": {
    #     "models": {
    #         "triage":     "finetuned/Qwen2.5-3B-triage-lora",
    #         "action":     "finetuned/Qwen2.5-7B-action-lora",
    #         "supervisor": "finetuned/Qwen2.5-14B-supervisor-lora",
    # },
    "finetuned_homogeneous_4": {
            "triage":     "gpt-4o-mini",
            "action":     "gpt-4o-mini",
            "supervisor": "gpt-4o-mini",
        },
}

