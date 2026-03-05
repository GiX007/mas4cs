"""Main entry point for MAS4CS project."""
import time
from pathlib import Path

from src.experiments import run_experiment_1, run_experiment_2, run_experiment_3, run_analysis, RESULTS_DIR
# from src.experiments import debug_single_agent, debug_mas_graph
from src.utils import print_separator, capture_and_save



def main() -> None:
    """Run the complete MAS4CS pipeline."""
    print_separator("MAS4CS - MULTI-AGENT SYSTEM FOR CUSTOMER SERVICE")
    start_time = time.time()

    # Load and process HuggingFace dataset (or ensure it is ready on disk)
    # To run this, use experiment_1_hf, experiment_2_hf, and experiment_3_hf
    # from src.data import load_multiwoz, run_preprocessing_pipeline
    # print("\n\n>>> Loading and preparing the dataset ...")
    # dataset = load_multiwoz()
    # run_preprocessing_pipeline(dataset)

    # Run experiment 1: Single-Agent Baseline
    print("\n\n>>> Running Experiment 1: Single-Agent Baseline ...")
    # debug_single_agent()  # Run and save debug version on 1 dialogue with prints
    run_experiment_1()

    # Run experiment 2: MAS Graph with different model configurations
    print("\n\n>>> Running Experiment 2: MAS Graph ...")
    # debug_mas_graph()  # Run and save debug version on 1 dialogue with prints
    run_experiment_2()

    # Run experiment 3: Optimized MAS Graph with fine-tuned models
    # print("\n\n>>> Running Experiment 3: Optimized MAS Graph with Fine-Tuned Models ...")
    # run_experiment_3()

    end_time = time.time()
    total_duration = end_time - start_time
    print(f"\n\n>>> Total execution time: {total_duration:.2f} seconds")

    print_separator("MAS4CS PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == '__main__':
    # main()

    # Run all experiments
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    capture_and_save(func=main, output_path=f"results/logs/run_{timestamp}.txt")

    # Step 5: Run error analysis
    error_analysis_output_path = Path(RESULTS_DIR) / f"error_analysis_{timestamp}.txt"
    capture_and_save(func=run_analysis, output_path=error_analysis_output_path)
