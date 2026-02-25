"""
Experiment 1: Single-Agent Baseline.

Tests how well a single LLM handles customer service without any multi-agent architecture.
Compares multiple models using the same evaluation metrics on the validation split.
"""

from typing import Any
from tqdm import tqdm

from src.data import load_split_data
from src.models import clear_model_cache
from src.evaluation import DatasetEvaluator, gpt4_judge
from src.experiments.helpers import run_single_agent_dialogue, save_experiment_results, print_and_save_comparison_table
from src.experiments import DATASET_PATH, EXP1_MODELS, MAX_DIALOGUES, MAX_TOKENS, TEMPERATURE, SPLIT, RESULTS_DIR


def run_experiment_1() -> None:
    """
    Run Experiment 1 across all configured models and save results.

    Models and settings are defined in src/experiments/config.py.
    Results are saved to RESULTS_DIR as timestamped JSON files.
    """
    all_results = {}

    for model_name in EXP1_MODELS:
        result = _run_single_model(model_name)
        all_results[model_name] = result
        clear_model_cache()  # free GPU memory before next model

    print_and_save_comparison_table(
        all_results=all_results,
        experiment_id="exp1",
        experiment_title="Single-Agent Baseline",
        output_dir=RESULTS_DIR
    )


def _run_single_model(model_name: str) -> dict[str, Any]:
    """
    Run Experiment 1 for one model on the configured dataset split.

    Args:
        model_name: Model identifier from UNSLOTH_MODELS or PAID_MODELS

    Returns:
        Dataset-level metrics dictionary
    """
    dialogues = load_split_data(DATASET_PATH, SPLIT)
    if MAX_DIALOGUES is not None:
        dialogues = dialogues[:MAX_DIALOGUES]

    total = len(dialogues)
    print("\n" + "-" * 60)
    print(f"Exp1 | {model_name} | {total} dialogues")

    dataset_evaluator = DatasetEvaluator()
    dialogue_results = []  # collect per-dialogue results
    failed_dialogues = 0

    for idx, dialogue in enumerate(tqdm(dialogues, desc=f"  {model_name}", unit="dlg", leave=True)):
        print(f"  [{idx + 1}/{total}] {dialogue['dialogue_id']}...", end=" ")

        result = run_single_agent_dialogue(
            dialogue=dialogue,
            model_name=model_name,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            # judge_fn=gpt4_judge()
        )

        if result is None:
            failed_dialogues += 1
            print("SKIPPED")
            continue

        dataset_evaluator.add_dialogue(result)
        dialogue_results.append(result)  # collect single dialogue result
        print(f"(turns={result['num_turns']})")

    final_metrics = dataset_evaluator.compute_dataset_metrics()
    final_metrics["model_name"] = model_name
    final_metrics["failed_dialogues"] = failed_dialogues
    final_metrics["split"] = SPLIT

    # Save all 3 files
    save_experiment_results(
        dataset_metrics=final_metrics,
        dialogue_results=dialogue_results,
        experiment_id="exp1",
        output_dir=RESULTS_DIR
    )

    return final_metrics

