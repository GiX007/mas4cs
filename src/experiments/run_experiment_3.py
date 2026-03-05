"""
Experiment 3: MAS Graph with Fine-Tuned Models.

Tests whether fine-tuned models in a MAS architecture outperform models from Experiments 1 and 2.
Runs all configurations defined in EXP3_MODELS.
"""
from src.models import clear_model_cache
from src.experiments.helpers import run_mas_cfg, print_save_table, print_save_table_per_domain
from src.experiments.config import EXP3_MODELS, RESULTS_DIR


def run_experiment_3() -> None:
    """Run Experiment 3 across all fine-tuned model configurations."""
    all_results = {}

    for config_name, model_config in EXP3_MODELS.items():
        result = run_mas_cfg(
            config_name=config_name,
            model_config=model_config,
            experiment_id="exp3",
        )
        all_results[config_name] = result
        clear_model_cache()

    print_save_table(
        all_results=all_results,
        experiment_id="exp3",
        experiment_title="MAS Graph with Fine-Tuned Models",
        output_dir=RESULTS_DIR
    )

    print_save_table_per_domain(
        experiment_id="exp3",
        experiment_title="MAS Graph with Fine-Tuned Models",
    )
