"""
Experiment 3: MAS Graph with Fine-Tuned Models.

Tests whether fine-tuned models in a MAS architecture outperform previous models from Experiments 1 and 2.
Runs all configurations defined in EXP3_MODELS.
"""

from src.models import clear_model_cache
from src.evaluation import gpt4_judge
from src.experiments.helpers import run_mas_config, print_and_save_comparison_table
from src.experiments import EXP3_MODELS, RESULTS_DIR


def run_experiment_3() -> None:
    """
    Run Experiment 3 across all fine-tuned model configurations.

    Models and settings are defined in src/experiments/config.py.
    Results saved to RESULTS_DIR as timestamped JSON files.
    """
    if not EXP3_MODELS:
        print("  EXP3_MODELS is empty - skipping Experiment 3.")
        print("  Populate EXP3_MODELS in config.py with fine-tuned model paths.")
        return

    all_results = {}

    for config_name, model_config in EXP3_MODELS.items():
        result = run_mas_config(
            config_name=config_name,
            model_config=model_config,
            experiment_id="exp3",
            # judge_fn=gpt4_judge(),
        )
        all_results[config_name] = result
        clear_model_cache()

    print_and_save_comparison_table(
        all_results=all_results,
        experiment_id="exp3",
        experiment_title="MAS Graph with Fine-Tuned Models",
        output_dir=RESULTS_DIR
    )

