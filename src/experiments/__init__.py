"""
Experiment Runners for MAS4CS.

Provides top-level entry points for all experiments:
- Experiment 1: Single-Agent Baseline
- Experiment 2: MAS Graph
- Experiment 3: MAS Graph with Fine-Tuned Models
"""
from .config import (
    DATASET_PATH, SPLIT, MULTIWOZ_DIR, DB_DIR, TARGET_DOMAINS, MAX_DB_RESULTS, MAX_DIALOGUES, MAX_TOKENS,
    TEMPERATURE, RESULTS_DIR, RESULTS_DIR_PER_DOMAIN, EXP1_MODELS, EXP2_CONFIG, EXP3_MODELS, ANALYSIS_CONFIGS
)
from .run_experiment_1_hf import run_experiment_1
from .run_experiment_2_hf import run_experiment_2
from .run_experiment_3_hf import run_experiment_3
from .run_experiment_1 import run_experiment_1
from .run_experiment_2 import run_experiment_2
from .run_experiment_3 import run_experiment_3
from .helpers_hf import (
    run_single_agent_turn, run_single_agent_dialogue, run_mas_dialogue,
    run_mas_config, save_experiment_results, print_and_save_comparison_table,
)
from .helpers import run_sa_turn, run_sa_dialogue, run_mas_dlg, run_mas_cfg, save_exp_results, print_save_table
from src.experiments.error_analysis import run_analysis
from src.experiments.debug_runs import debug_mas_graph, debug_single_agent

__all__ = [
    # Config
    "DATASET_PATH",
    "SPLIT",
    "MULTIWOZ_DIR",
    "DB_DIR",
    "TARGET_DOMAINS",
    "MAX_DB_RESULTS",
    "MAX_DIALOGUES",
    "MAX_TOKENS",
    "TEMPERATURE",
    "RESULTS_DIR",
    "RESULTS_DIR_PER_DOMAIN",
    "EXP1_MODELS",
    "EXP2_CONFIG",
    "EXP3_MODELS",
    "ANALYSIS_CONFIGS",

    # Shared helpers
    "run_single_agent_turn",
    "run_single_agent_dialogue",
    "run_mas_dialogue",
    "run_mas_config",
    "save_experiment_results",
    "print_and_save_comparison_table",

    "run_sa_turn",
    "run_sa_dialogue",
    "run_mas_dlg",
    "run_mas_cfg",
    "save_exp_results",
    "print_save_table",

    # Experiments
    "run_experiment_1_hf",
    "run_experiment_2_hf",
    "run_experiment_3_hf",
    "run_experiment_1",
    "run_experiment_2",
    "run_experiment_3",

    # Error Analysis
    "run_analysis",

    # Debug Runs
    "debug_mas_graph",
    "debug_single_agent",
]
