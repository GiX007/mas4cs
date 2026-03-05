"""
Error Analysis for MAS4CS Experiments.

Loads turn-level result files and identifies failure patterns across all active experiment configurations defined in config.py.

Metrics in this report are turn-weighted averages (flat average across all turns), NOT dialogue-weighted averages as in the leaderboard/_dataset.json.

Example:
    Dialogue A: 2 turns, intent_correct = [True, True] → avg = 1.0
    Dialogue B: 8 turns, intent_correct = [False, ...] → avg = 0.0

    Turn-weighted (this file): (2 + 0) / 10 turns = 0.20
    Dialogue-weighted (leaderboard): (1.0 + 0.0) / 2 dialogues = 0.50

Longer dialogues have more influence on turn-weighted averages.
Use of this file for failure pattern inspection only.

Failure types: intent_error(predicted intent != ground truth intent), slot_error(slot_f1 < 1.0, missing or wrong slot values), hallucination(response mentions entities not in DB results),
               policy_violation(booking attempted with missing required slots), system_incorrect(response flagged as incorrect, when policy or hallucination)
"""
import json
from pathlib import Path
from tabulate import tabulate
from src.experiments import RESULTS_DIR,ANALYSIS_CONFIGS
from src.utils import print_separator


def load_turns(filename_substring: str) -> list[dict]:
    """
    Load turns from the most recent _turns.json file matching the substring.

    Args:
        filename_substring (e.g., 'exp1_gpt-4o-mini')

    Returns:
        List of turn dicts with _model metadata attached
    """
    results_path = Path(RESULTS_DIR)
    matches = sorted(results_path.glob(f"*{filename_substring}*_turns.json"))

    if not matches:
        print(f"No file found for: {filename_substring}")
        return []

    filepath = matches[-1]
    data = json.loads(filepath.read_text(encoding="utf-8"))
    turns = data.get("turns", [])

    for turn in turns:
        turn["_model"] = data.get("model_name", "unknown")
        turn["_experiment"] = data.get("experiment", "unknown")

    print(f"Loaded {len(turns):} turns from {filepath.name}")
    return turns


def run_analysis() -> None:
    """Full error analysis: load turns, print summary, print failure examples."""
    print_separator("ERROR ANALYSIS FOR MAS4CS EXPERIMENTS")
    print()

    all_turns = []
    for exp_id, config_names in ANALYSIS_CONFIGS.items():
        for config_name in config_names:
            all_turns += load_turns(f"{exp_id}_{config_name}")

    if not all_turns:
        print("No turns loaded. Check RESULTS_DIR and filenames.")
        return

    # Group turns by model/config
    by_config: dict[str, list[dict]] = {}
    for turn in all_turns:
        key = turn["_model"]
        if key not in by_config:
            by_config[key] = []
        by_config[key].append(turn)

    # Per-config summary (turn-weighted averages)
    print_separator("PER-CONFIG TURN-LEVEL SUMMARY")
    rows = []
    for config, turns in sorted(by_config.items()):
        n = len(turns)
        exp_id = turns[0].get("_experiment", "?")
        intent_ok = sum(1 for t in turns if t.get("intent_correct", False)) / n
        act_f1 = sum(t.get("action_type_f1", 0) for t in turns) / n
        slot_f1 = sum(t.get("slot_f1", 0) for t in turns) / n
        hall = sum(t.get("hallucination_rate", 0) for t in turns) / n
        polviol = sum(1 for t in turns if not t.get("policy_compliant", True)) / n
        sys_ok = sum(1 for t in turns if t.get("system_correct", False)) / n
        rows.append([
            f"{exp_id}/{config}", n,
            f"{intent_ok:.1%}", f"{act_f1:.1%}", f"{slot_f1:.1%}",
            f"{hall:.1%}", f"{polviol:.1%}", f"{sys_ok:.1%}",
        ])

    headers = ["Config", "Turns", "Intent%", "ActF1%", "SlotF1%", "Hall%", "PolViol%", "SysCorr%"]
    print(tabulate(rows, headers=headers, tablefmt="github"))

    # Failure types
    failure_types = {
        "intent_error": lambda t: not t.get("intent_correct", True),
        "slot_error": lambda t: t.get("slot_f1", 1.0) < 1.0,
        "hallucination": lambda t: t.get("hallucination_rate", 0.0) > 0.0,
        "policy_violation": lambda t: not t.get("policy_compliant", True),
        "system_incorrect": lambda t: not t.get("system_correct", True),
    }

    n_per_config = 5

    for failure_type, condition in failure_types.items():
        failed = [t for t in all_turns if condition(t)]

        print(f"\n{'-'*60}")
        print(f"FAILURE TYPE: {failure_type.upper()} ({len(failed)} total)")
        print(f"{'-'*60}")

        if not failed:
            print("No failures of this type.")
            continue

        # Group by config
        failed_by_config: dict[str, list[dict]] = {}
        for turn in failed:
            key = turn.get("_model", "?")
            if key not in failed_by_config:
                failed_by_config[key] = []
            failed_by_config[key].append(turn)

        for config, config_turns in sorted(failed_by_config.items()):
            print(f"\n>>> Config: {config} ({len(config_turns)} failures)")
            for i, turn in enumerate(config_turns[:n_per_config]):
                print(f"\nExample {i+1}")
                print(f"Dialogue: {turn.get('dialogue_id', '?')} | Turn {turn.get('turn_id', '?')}")
                print(f"User: {turn.get('user_message', '')}")
                print(f"Response: {turn.get('system_response', '')[:120]}")

                if failure_type == "intent_error":
                    print(f"Predicted: {turn.get('predicted_intent', '')}")
                    print(f"GT: {turn.get('ground_truth_intent', '')}")
                elif failure_type == "slot_error":
                    print(f"Predicted slots: {turn.get('predicted_slots', {})}")
                    print(f"Slot F1: {turn.get('slot_f1', 0):.2f}")
                elif failure_type == "hallucination":
                    print(f"Hall rate: {turn.get('hallucination_rate', 0):.2f}")
                elif failure_type == "policy_violation":
                    print(f"Reason: {turn.get('policy_reason', '')}")
                elif failure_type == "system_incorrect":
                    print(f"Reason: {turn.get('system_reason', '')}")

    print_separator("END OF ERROR ANALYSIS FOR MAS4CS EXPERIMENTS")
