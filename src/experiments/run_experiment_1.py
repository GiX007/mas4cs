"""
Experiment 1: Single-Agent Baseline.

Tests how well a single LLM handles customer service without any multi-agent architecture.
Compares multiple models using the same evaluation metrics on the validation split.
"""
from typing import Any
from tqdm import tqdm

from src.data import load_split
from src.models import clear_model_cache
from src.evaluation import DatasetEvaluator
from src.experiments.helpers import run_sa_dialogue, save_exp_results, print_save_table, save_exp_results_per_domain
from src.experiments.config import EXP1_MODELS, MAX_DIALOGUES, MAX_TOKENS, TEMPERATURE, SPLIT, RESULTS_DIR


def run_experiment_1() -> None:
    """Run Experiment 1 across all configured models and save results."""
    all_results = {}

    for model_name in EXP1_MODELS:
        result = _run_single_model(model_name)
        all_results[model_name] = result
        clear_model_cache()

    print_save_table(
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
    dialogues = load_split(SPLIT)
    if MAX_DIALOGUES is not None:
        dialogues = dialogues[:MAX_DIALOGUES]

    total = len(dialogues)
    print("\n" + "-" * 60)
    print(f"Exp1 | {model_name} | {total} dialogues")

    dataset_evaluator = DatasetEvaluator()
    dialogue_results = []
    failed_dialogues = 0

    for idx, dialogue in enumerate(tqdm(dialogues, desc=f"  {model_name}", unit="dlg", leave=True)):
        print(f"  [{idx + 1}/{total}] {dialogue['dialogue_id']}...", end=" ")

        result = run_sa_dialogue(
            dialogue=dialogue,
            model_name=model_name,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

        if result is None:
            failed_dialogues += 1
            print("SKIPPED")
            continue

        dataset_evaluator.add_dialogue(result)
        dialogue_results.append(result)
        print(f"(turns={result['num_turns']})")

    final_metrics = dataset_evaluator.compute_dataset_metrics()
    final_metrics["model_name"] = model_name
    final_metrics["failed_dialogues"] = failed_dialogues
    final_metrics["split"] = SPLIT

    save_exp_results(
        dataset_metrics=final_metrics,
        dialogue_results=dialogue_results,
        experiment_id="exp1",
        output_dir=RESULTS_DIR
    )

    save_exp_results_per_domain(
        dataset_metrics=final_metrics,
        dialogue_results=dialogue_results,
        experiment_id="exp1",
    )

    return final_metrics
