"""
Experiment 2: MAS Graph.

Tests whether a structured multi-agent architecture improves performance over the single-agent baseline (Experiment 1).
Runs all configurations defined in EXP2_CONFIG.
"""
from src.models import clear_model_cache
from src.experiments.helpers import run_mas_cfg, print_save_table, print_save_table_per_domain
from src.experiments.config import EXP2_CONFIG, RESULTS_DIR


def run_experiment_2() -> None:
    """Run Experiment 2 across all configurations in EXP2_CONFIG."""
    all_results = {}

    for config_name, model_config in EXP2_CONFIG.items():
        result = run_mas_cfg(
            config_name=config_name,
            model_config=model_config,
            experiment_id="exp2",
        )
        all_results[config_name] = result
        clear_model_cache()

    print_save_table(
        all_results=all_results,
        experiment_id="exp2",
        experiment_title="MAS Graph",
        output_dir=RESULTS_DIR
    )

    print_save_table_per_domain(
        experiment_id="exp2",
        experiment_title="MAS Graph",
    )
