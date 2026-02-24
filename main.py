"""Main entry point for MAS4CS project."""

import time

from src.data import load_multiwoz, run_preprocessing_pipeline
from src.experiments import run_experiment_1, run_experiment_2, run_experiment_3
from src.utils import print_separator, capture_and_save


def run_experiments() -> None:
    """Run all experiments sequentially."""

    print("\n\n>>> Running Experiment 1: Single-Agent Baseline ...")
    run_experiment_1()

    print("\n\n>>> Running Experiment 2: MAS Graph ...")
    run_experiment_2()

    # print("\n\n>>> Running Experiment 3: Optimized MAS Graph with Fine-Tuned Models ...")
    # run_experiment_3()


def main() -> None:
    """Run the complete MAS4CS pipeline."""

    print_separator("MAS4CS - MULTI-AGENT SYSTEM FOR CUSTOMER SERVICE")

    # Step 1: Load and process dataset (or ensure it is ready on disk)
    print("\n\n>>> Loading and preparing the dataset ...")
    dataset = load_multiwoz()
    run_preprocessing_pipeline(dataset)

    # Step 2: Run all experiments
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    capture_and_save(func=run_experiments,
                     output_path=f"results/logs/run_{timestamp}.txt")

    print_separator("MAS4CS PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == '__main__':
    main()

