"""
Experiment Runners for MAS4CS.

Provides top-level entry points for all experiments:
- Experiment 1: Single-Agent Baseline
- Experiment 2: MAS Graph
- Experiment 3: MAS Graph with Fine-Tuned Models
"""

from .config import (
    DATASET_PATH,
    SPLIT,
    MAX_DIALOGUES,
    MAX_TOKENS,
    TEMPERATURE,
    RESULTS_DIR,
    EXP1_MODELS,
    EXP2_CONFIG,
    EXP3_MODELS,
)
from .run_experiment_1 import run_experiment_1
from .run_experiment_2 import run_experiment_2
from .run_experiment_3 import run_experiment_3
from .helpers import (
    run_single_agent_turn,
    run_single_agent_dialogue,
    run_mas_dialogue,
    run_mas_config,
    save_experiment_results,
    print_and_save_comparison_table,
)

__all__ = [
    # Config
    "DATASET_PATH",
    "SPLIT",
    "MAX_DIALOGUES",
    "MAX_TOKENS",
    "TEMPERATURE",
    "RESULTS_DIR",
    "EXP1_MODELS",
    "EXP2_CONFIG",
    "EXP3_MODELS",

    # Shared helpers
    "run_single_agent_turn",
    "run_single_agent_dialogue",
    "run_mas_dialogue",
    "run_mas_config",
    "save_experiment_results",
    "print_and_save_comparison_table",

    # Experiments
    "run_experiment_1",
    "run_experiment_2",
    "run_experiment_3",
]

