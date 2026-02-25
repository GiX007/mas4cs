"""
Error Analysis for MAS4CS Experiments.

Loads turn-level result files and identifies failure patterns.
To skip an experiment, comment out the relevant load call at the bottom.

Usage:
    python -m src.experiments.error_analysis
"""

import json
from pathlib import Path
from collections import defaultdict
from src.experiments import RESULTS_DIR
from src.utils import print_separator


def load_turns(filename_substring: str) -> list[dict]:
    """
    Load turns from the most recent _turns.json file matching the substring.

    Args:
        filename_substring: e.g. 'exp1_gpt-4o-mini' or 'exp2_heterogeneous_6'

    Returns:
        List of turn dicts with _model metadata attached
    """
    results_path = Path(RESULTS_DIR)
    matches = sorted(results_path.glob(f"*{filename_substring}*_turns.json"))

    if not matches:
        print(f"  [WARN] No file found for: {filename_substring}")
        return []

    filepath = matches[-1]
    data = json.loads(filepath.read_text(encoding="utf-8"))
    turns = data.get("turns", [])

    for turn in turns:
        turn["_model"] = data.get("model_name", "unknown")
        turn["_experiment"] = data.get("experiment", "unknown")

    print(f"  Loaded {len(turns):} turns from {filepath.name}")
    return turns


def run_analysis() -> None:
    """
    Full error analysis: load turns, print summary, print failure examples.
    """
    print_separator("ERROR ANALYSIS FOR MAS4CS EXPERIMENTS")
    print()
    all_turns = []

    # Experiment 1
    all_turns += load_turns("exp1_gpt-4o-mini")
    all_turns += load_turns("exp1_claude-3-haiku")

    # Experiment 2
    all_turns += load_turns("exp2_homogeneous_gpt")
    all_turns += load_turns("exp2_homogeneous_haiku")
    all_turns += load_turns("exp2_heterogeneous_6")

    if not all_turns:
        print("No turns loaded. Check RESULTS_DIR and filenames.")
        return

    # Per-config summary
    by_config: dict[str, list[dict]] = defaultdict(list)
    for turn in all_turns:
        by_config[turn["_model"]].append(turn)

    print_separator("PER-CONFIG TURN-LEVEL SUMMARY")
    print(f"{'Config':<30} {'Turns':>5} {'Intent%':>8} {'Slot%':>7} {'Hall%':>7} {'SysOK%':>8}")
    print(f"{'-'*60}")

    for config, turns in sorted(by_config.items()):
        n = len(turns)
        intent_ok = sum(1 for t in turns if t.get("intent_correct", False)) / n
        slot_ok = sum(t.get("slot_accuracy", 0) for t in turns) / n
        hall = sum(t.get("hallucination_rate", 0) for t in turns) / n
        sys_ok = sum(1 for t in turns if t.get("system_correct", False)) / n
        print(f"{config:<30} {n:>5} {intent_ok:>8.1%} {slot_ok:>7.1%} {hall:>7.1%} {sys_ok:>8.1%}")

    # Failure types and conditions
    failure_types = {
        "intent_error":     lambda t: not t.get("intent_correct", True),
        "slot_error":       lambda t: t.get("slot_accuracy", 1.0) < 1.0,
        "hallucination":    lambda t: t.get("hallucination_rate", 0.0) > 0.0,
        "policy_violation": lambda t: not t.get("policy_compliant", True),
        "system_incorrect": lambda t: not t.get("system_correct", True),
    }

    n_per_config = 5  # examples per config per failure type

    for failure_type, condition in failure_types.items():
        failed = [t for t in all_turns if condition(t)]

        print(f"\n{'='*60}")
        print(f"FAILURE TYPE: {failure_type.upper()} ({len(failed)} total)")
        print(f"{'='*60}")

        if not failed:
            print("  No failures of this type.")
            continue

        # Group failures by config
        failed_by_config: dict[str, list[dict]] = defaultdict(list)
        for turn in failed:
            failed_by_config[turn.get("_model", "?")].append(turn)

        # Print n_per_config examples from each config
        for config, config_turns in sorted(failed_by_config.items()):
            print(f"\n  >>> Config: {config} ({len(config_turns)} failures)")
            for i, turn in enumerate(config_turns[:n_per_config]):
                print(f"\n  --- Example {i+1} ---")
                print(f"  Dialogue : {turn.get('dialogue_id', '?')} | Turn {turn.get('turn_id', '?')}")
                print(f"  User     : {turn.get('user_message', '')}")
                print(f"  Response : {turn.get('system_response', '')[:120]}")

                if failure_type == "intent_error":
                    print(f"  Predicted: {turn.get('predicted_intent', '')}")
                    print(f"  GT       : {turn.get('ground_truth_intent', '')}")
                elif failure_type == "slot_error":
                    print(f"  Predicted slots: {turn.get('predicted_slots', {})}")
                    print(f"  Slot accuracy  : {turn.get('slot_accuracy', 0):.2f}")
                elif failure_type == "hallucination":
                    print(f"  Hall rate: {turn.get('hallucination_rate', 0):.2f}")
                elif failure_type == "policy_violation":
                    print(f"  Reason   : {turn.get('policy_reason', '')}")
                elif failure_type == "system_incorrect":
                    print(f"  Reason   : {turn.get('system_reason', '')}")

    print_separator("END OF ERROR ANALYSIS FOR MAS4CS EXPERIMENTS")

