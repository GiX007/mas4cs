"""
Experiment Configuration for MAS4CS.

Single source of truth for all experiment parameters.
"""
from pathlib import Path
from typing import Any


# HuggingFace filtered dataset
DATASET_PATH: str = Path("dataset") / "mw22_filtered.json"

# Official MultiWOZ 2.2 from GitHub
MULTIWOZ_DIR = Path("data") / "multiwoz_github" / "data" / "MultiWOZ_2.2"
DB_DIR = Path("data") / "multiwoz_github" / "db"
SPLIT: str = "dev"  # "validation"

# Target domains for MAS4CS
TARGET_DOMAINS = {"hotel", "restaurant"}

# DB Query Settings
MAX_DB_RESULTS: int = 5  # Maximum number of matching entities returned by find_entity

# Shared Settings
MAX_DIALOGUES: int | None = None  # None = full split
MAX_TOKENS: int = 512
TEMPERATURE: float = 0.0

RUN_ID: str = "run_01_dev_full"
RESULTS_DIR: str = f"results/runs/{RUN_ID}/overall"
RESULTS_DIR_PER_DOMAIN: str = f"results/runs/{RUN_ID}/per_domain"


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
    # "homo_qwen2.5_3b": {
    #     "triage": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    #     "action": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    #     "supervisor": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    # },
    # "homo_qwen2.5_7b": {
    #     "triage": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    #     "action": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    #     "supervisor": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    # },
    # "homo_qwen2.5_14b": {
    #     "triage": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    #     "action": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    #     "supervisor": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    # },
    "homo_gpt": {
        "triage": "gpt-4o-mini",
        "action": "gpt-4o-mini",
        "supervisor": "gpt-4o-mini",
    },
    "homo_claude": {
        "triage": "claude-3-haiku-20240307",
        "action": "claude-3-haiku-20240307",
        "supervisor": "claude-3-haiku-20240307",
    },

    # Heterogeneous architectures with small, medium, large Open-Source models and API-based models
    # "hetero_qwen2.5": {
    #     "triage": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    #     "action": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    #     "supervisor": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    # },
    # "hetero_llama3.2_gemma2_nemo": {
    #     "triage": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    #     "action": "unsloth/gemma-2-9b-it-bnb-4bit",
    #     "supervisor": "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    # },
    # "hetero_gpt_claude_gpt": {
    #     "triage": "gpt-4o-mini",
    #     "action": "claude-3-haiku-20240307",
    #     "supervisor": "gpt-4o-mini",
    # },
    "hetero_claude_gpt_claude": {
        "triage": "claude-3-haiku-20240307",
        "action": "gpt-4o-mini",
        "supervisor": "claude-3-haiku-20240307",
    },
    "hetero_gpt_gpt_claude": {
        "triage": "gpt-4o-mini",
        "action": "gpt-4o-mini",
        "supervisor": "claude-3-haiku-20240307",
    },
    # "hetero_claude_claude_gpt": {
    #     "triage": "claude-3-haiku-20240307",
    #     "action": "claude-3-haiku-20240307",
    #     "supervisor": "gpt-4o-mini",
    # },
}

# Experiment 3: MAS Graph with Fine-Tuned Model
EXP3_MODELS: dict[str, Any] = {
    # "hetero_qwen2.5_ft": {
    #     "models": {
    #         "triage": "finetuned/Qwen2.5-3B-triage-lora",
    #         "action": "finetuned/Qwen2.5-7B-action-lora",
    #   `      "supervisor": "finetuned/Qwen2.5-14B-supervisor-lora",
    # },
    # "hetero_llama3.2_gemma2_nemo_ft": {
    #     "models": {
    #         "triage": "finetuned/Qwen2.5-3B-triage-lora",
    #         "action": "finetuned/Qwen2.5-7B-action-lora",
    #         "supervisor": "finetuned/Qwen2.5-14B-supervisor-lora",
    # },
    "homo_gpt_ft": {
            "triage": "gpt-4o-mini",
            "action": "gpt-4o-mini",
            "supervisor": "gpt-4o-mini",
        },
}

# Config names for error analysis
ANALYSIS_CONFIGS: dict[str, list[str]] = {
    "exp1": [m.split("/")[-1] for m in EXP1_MODELS],
    "exp2": list(EXP2_CONFIG.keys()),
    "exp3": list(EXP3_MODELS.keys()),
}
