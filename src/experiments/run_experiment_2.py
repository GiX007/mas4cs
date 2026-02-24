"""
Experiment 2: MAS Graph.

Tests whether a structured multi-agent architecture improves performance over the single-agent baseline (Experiment 1).
Runs all configurations defined in EXP2_CONFIG.
"""

from src.models import clear_model_cache
from src.evaluation import gpt4_judge
from src.experiments.helpers import run_mas_config, print_and_save_comparison_table
from src.experiments import EXP2_CONFIG, RESULTS_DIR


def run_experiment_2() -> None:
    """
    Run Experiment 2 across all configurations in EXP2_CONFIG.

    Each configuration defines which model runs in each agent role.
    Results saved to RESULTS_DIR as timestamped JSON files.
    """
    all_results = {}

    for config_name, model_config in EXP2_CONFIG.items():
        result = run_mas_config(
            config_name=config_name,
            model_config=model_config,
            experiment_id="exp2",
            # judge_fn=gpt4_judge(),
        )
        all_results[config_name] = result
        clear_model_cache()

    print_and_save_comparison_table(
        all_results=all_results,
        experiment_id="exp2",
        experiment_title="MAS Graph",
        output_dir=RESULTS_DIR
    )

