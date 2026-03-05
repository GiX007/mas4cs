"""One-dialogue debug runs for MAS Graph and Single Agent."""
from src.utils import capture_and_save


def debug_single_agent() -> None:
    """Run Exp1 Single Agent on 1 dialogue with all debug prints. Saves to logs."""
    from src.experiments import run_experiment_1
    capture_and_save(func=run_experiment_1, output_path="results/logs/debug_MUL1271_single_agent.txt")


def debug_mas_graph() -> None:
    """Run Exp2 MAS Graph on 1 dialogue with all debug prints. Saves to logs."""
    from src.experiments import run_experiment_2
    capture_and_save(func=run_experiment_2, output_path="results/logs/debug_MUL1271_mas_graph.txt")
