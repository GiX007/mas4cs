"""
Experiment Helper Functions for MAS4CS.

Reusable functions for single-agent turn/dialogue execution
and result persistence. Used by all experiment runners.
"""

import os
from pathlib import Path
import json
import time
from typing import Any

from torch.jit.annotations import get_param_names

from src.data import load_split_data, BOOKING_REQUIRED_SLOTS, extract_ground_truth_intent, extract_slots_from_frames, normalize_slot_value
from src.models import call_model
from src.evaluation import DialogueEvaluator, DatasetEvaluator, gpt4_judge
from src.utils import DEFAULT_MEGA_PROMPT, format_dialogue_history, format_policy_rules, parse_model_json_response
from src.experiments import DATASET_PATH, RESULTS_DIR


def run_single_agent_turn(user_message: str, services: list[str], dialogue_history: list[dict[str, Any]], model_name: str, max_tokens: int = 512, temperature: float = 0.0) -> dict[str, Any] | None:
    """
    Run one dialogue turn through the single-agent baseline.

    Builds mega-prompt, calls model, parses JSON response.
    No tools, no state, no multi-agent orchestration.

    Args:
        user_message: Current user utterance
        services: Available domains for this dialogue (e.g. ['hotel', 'restaurant'])
        dialogue_history: Previous turns from dataset (speaker + utterance dicts)
        model_name: Model identifier from UNSLOTH_MODELS or PAID_MODELS
        max_tokens: Maximum tokens to generate (Default: 512)
        temperature: Sampling temperature (Default: 0.0 for deterministic output)

    Returns:
        Dictionary with keys: domain, intent, slots, action_type,
        policy_satisfied, response, input_tokens, output_tokens, cost, response_time.
        None if model returns invalid JSON.
    """
    # Format history and policy for prompt (text format)
    history_text = format_dialogue_history(dialogue_history)
    policy_text = format_policy_rules(BOOKING_REQUIRED_SLOTS)
    services_str = ", ".join(services)

    # Fill mega-prompt template (Pass info into {} of the prompt)
    prompt = DEFAULT_MEGA_PROMPT.format(
        services=services_str,
        history_text=history_text,
        user_message=user_message,
        policy_text=policy_text
    )

    # Call model
    model_response = call_model(
        model_name=model_name,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature
    )

    # Parse JSON response
    try:
        parsed = parse_model_json_response(model_response.text)
        # print(parsed)
    except json.JSONDecodeError:
        print(f"JSON parse failed. Raw response: {model_response.text[:100]}")
        return None

    # Attach model metadata to result
    parsed["input_tokens"] = model_response.input_tokens
    parsed["output_tokens"] = model_response.output_tokens
    parsed["cost"] = model_response.cost
    parsed["response_time"] = model_response.response_time

    return parsed


def run_single_agent_dialogue(dialogue: dict[str, Any], model_name: str, max_tokens: int = 512, temperature: float = 0.0, judge_fn=None) -> dict[str, Any] | None:
    """
    Run one full dialogue through the single-agent baseline.

    Loops through all turns, evaluates each USER turn, and returns aggregated dialogue-level metrics.

    Args:
        dialogue: Single dialogue dict from processed dataset
        model_name: LLM Model identifier from UNSLOTH_MODELS or PAID_MODELS
        max_tokens: Maximum tokens to generate (Default: 512)
        temperature: Sampling temperature (Default: 0.0)
        judge_fn: Optional function to call judge LLM for turn-level correctness evaluation (Default: None)

    Returns:
        Dialogue evaluation result dict from DialogueEvaluator, or None if no turns were successfully evaluated
    """
    evaluator = DialogueEvaluator(policy_requirements=BOOKING_REQUIRED_SLOTS, judge_llm_fn=judge_fn)

    services = dialogue["services"]
    turns = dialogue["turns"]
    history = []

    accumulated_slots: dict[str, dict[str, str]] = {}  # Accumulated slots across turns

    for turn in turns:

        # Only evaluate USER turns - SYSTEM turns update history only
        if turn["speaker"] != "USER":
            history.append(turn)
            continue

        # Run single agent turn
        result = run_single_agent_turn(
            user_message=turn["utterance"],
            services=services,
            dialogue_history=history,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature
        )

        # Skip turn if JSON parse failed
        if result is None:
            history.append(turn)
            continue

        # 1. Intent: Extract ground truth intent
        ground_truth_intent = extract_ground_truth_intent(turn, services)

        # Skip turns with no ground truth intent (e.g. closing turns for 'goodbye')
        if not ground_truth_intent:
            continue
        # print(f"\n  GT intent: {ground_truth_intent} | Predicted intent: {final_state['active_intent']} | Predicted domain: {final_state['current_domain']}")

        # 2. Action taken: Use intent as action_taken for booking turns to match BOOKING_REQUIRED_SLOTS keys
        action_taken = result["intent"] if result["intent"].startswith("book_") else result["action_type"]
        # print(f"\n  Action taken: {action_taken}")

        # 3. Slots: Extract accumulated ground truth slots
        ground_truth_slots = extract_slots_from_frames(turn["frames"])

        # Accumulate predicted slots across turns
        if result["domain"] not in accumulated_slots:  # ensure key exists (if not present, create empty)
            accumulated_slots[result["domain"]] = {}
        # Update slot by slot WITH normalization
        for slot, value in result["slots"].items():
            accumulated_slots[result["domain"]][slot] = normalize_slot_value(value)
        # print(f"\n  GT slots: {ground_truth_slots} | predicted slots: {accumulated_slots}")

        # print(f"\n  Turn {turn['turn_id']}: intent={result['intent']} | action={action_taken} | predicted={accumulated_slots} | gt={ground_truth_slots} | utterance: {turn['utterance']}")

        # 4. Dialogue Acts: Extract ground truth dialogue acts (list[str])
        ground_truth_act_types = turn["dialogue_acts"]


        # Feed predictions into evaluator
        evaluator.evaluate_turn(
            turn_id=turn["turn_id"],
            predicted_slots=accumulated_slots,  # accumulated predicted slots
            ground_truth_slots=ground_truth_slots,  # accumulated GT slots
            predicted_intent=result["intent"],
            ground_truth_intent=ground_truth_intent,
            predicted_act_type=[result["action_type"]],
            ground_truth_act_type=ground_truth_act_types,
            predicted_domain=result["domain"],
            action_taken=action_taken,
            user_message=turn["utterance"],
            system_response=result["response"]
        )

        history.append(turn)

    # Return None if no turns were evaluated
    if not evaluator.turn_metrics:
        return None

    # Build ground truth goal (used for task success evaluation)
    requires_booking = any(
        domain_data.get("active_intent", "").startswith("book_")
        for turn in turns
        if turn["speaker"] == "USER"
        for domain_data in turn.get("frames", {}).values()
    )

    booking_turn = next(
        (t for t in turns
         if t["speaker"] == "USER"
         and any(d.get("active_intent", "").startswith("book_")
                 for d in t.get("frames", {}).values())),
        None
    )
    # print(f"\n  booking turn: {booking_turn['turn_id'] if booking_turn else 'None'} | requires_booking={requires_booking} | domains={services} | speaker={booking_turn['speaker']} | utterance={booking_turn['utterance']}")

    ground_truth_goal = {
        "domains": services,  # e.g. ["hotel", "restaurant"]
        "requires_booking": requires_booking
    }

    result = evaluator.evaluate_dialogue(ground_truth_goal)

    # Attach dialogue metadata for saving
    if result is not None:
        result["dialogue_id"] = dialogue["dialogue_id"]
        result["services"] = dialogue["services"]

    return result


def run_mas_dialogue(dialogue: dict[str, Any], model_config: dict[str, str], workflow: Any, judge_fn=None) -> dict[str, Any] | None:
    """
    Run one full dialogue through the MAS workflow.
    Shared by Experiments 2 and 3.

    Args:
        dialogue: Single dialogue dict from processed dataset
        model_config: Dict mapping agent role -> model name
        workflow: Compiled LangGraph workflow
        judge_fn: Optional function to call judge LLM for turn-level correctness evaluation (Default: None)

    Returns:
        Dialogue evaluation result dict, or None if no turns evaluated
    """
    from src.core import initialize_state

    evaluator = DialogueEvaluator(policy_requirements=BOOKING_REQUIRED_SLOTS, judge_llm_fn=judge_fn)

    services = dialogue["services"]
    turns = dialogue["turns"]

    # CONSIDER THIS: slots_values resets every turn via initialize_state().
    # Option A: reset each turn (current) - each turn is independent
    # Option B: accumulate across turns - pass slots_values from previous state
    # This decision affects JGA and slot accuracy calculations significantly.
    accumulated_slots: dict[str, dict[str, str]] = {}

    for turn in turns:

        if turn["speaker"] != "USER":
            continue

        state = initialize_state(
            dialogue_id=dialogue["dialogue_id"],
            turn_id=turn["turn_id"],
            services=services,
            user_utterance=turn["utterance"]
        )

        state["model_config"] = model_config
        state["slots_values"] = accumulated_slots.copy()  # carry over accumulated slots

        try:
            final_state = workflow.invoke(state)
        except Exception as e:
            print(f"\n  Workflow error turn {turn['turn_id']}: {e}")
            continue

        accumulated_slots = final_state["slots_values"].copy()  # update for next turn

        # 1. Extract ground truth intent
        ground_truth_intent = extract_ground_truth_intent(turn, services)

        # Skip turns with no ground truth intent (e.g. closing turns for 'goodbye')
        if not ground_truth_intent:
            continue
        # print(f"\n  GT intent: {ground_truth_intent} | Predicted intent: {final_state['active_intent']} | Predicted domain: {final_state['current_domain']}")

        # 2. Use intent as action_taken for booking turns to match BOOKING_REQUIRED_SLOTS keys
        action_taken = final_state["active_intent"] if final_state["active_intent"].startswith("book_") else final_state["action_taken"]
        # print(f"\n  Action taken: {action_taken}")

        # 3. Extract accumulated ground truth slots
        ground_truth_slots = extract_slots_from_frames(turn["frames"])
        # print(f"\n  GT slots: {ground_truth_slots} | predicted slots: {final_state['slots_values']}")
        # print(f"\n  Turn {turn['turn_id']}: intent={final_state['active_intent']} | action={action_taken} | predicted={final_state['slots_values']} | gt={ground_truth_slots} | utterance: {turn['utterance']}")


        evaluator.evaluate_turn(
            turn_id=turn["turn_id"],
            predicted_slots=final_state["slots_values"],  # accumulated predicted
            ground_truth_slots=ground_truth_slots,  # accumulated GT
            predicted_intent=final_state["active_intent"] or "",
            ground_truth_intent=ground_truth_intent,
            predicted_act_type=final_state["dialogue_acts"],
            ground_truth_act_type=turn["dialogue_acts"],
            predicted_domain=final_state["current_domain"] or "",
            action_taken=action_taken,
            user_message=turn["utterance"],
            system_response=final_state["agent_response"] or ""
        )

    if not evaluator.turn_metrics:
        return None

    # Build ground truth goal from services and turns (used for task success evaluation)
    requires_booking = any(
        domain_data.get("active_intent", "").startswith("book_")
        for turn in turns
        if turn["speaker"] == "USER"
        for domain_data in turn.get("frames", {}).values()
    )

    ground_truth_goal = {
        "domains": services,  # e.g. ["hotel", "restaurant"]
        "requires_booking": requires_booking
    }

    result = evaluator.evaluate_dialogue(ground_truth_goal)

    if result is not None:
        result["dialogue_id"] = dialogue["dialogue_id"]
        result["services"] = services

    return result


def run_mas_config(config_name: str, model_config: dict[str, str], experiment_id: str, base_models: dict[str, str] | None = None, judge_fn=None) -> dict[str, Any]:
    """
    Run one MAS configuration on the dataset split.
    Shared by Experiments 2 and 3.

    Args:
        config_name: Configuration name (e.g. 'homogeneous_1')
        model_config: Dict mapping agent role -> model name
        experiment_id: e.g. 'exp2' or 'exp3'
        base_models: Base models used for fine-tuning (exp3 only, default None)
        judge_fn: Optional function to call judge LLM for turn-level correctness evaluation (Default: None)

    Returns:
        Dataset-level metrics dictionary
    """
    from src.core import create_workflow
    from src.experiments.config import MAX_DIALOGUES, SPLIT, RESULTS_DIR

    dialogues = load_split_data(DATASET_PATH, SPLIT)
    if MAX_DIALOGUES is not None:
        dialogues = dialogues[:MAX_DIALOGUES]

    total = len(dialogues)
    print("\n" + "-" * 60)
    print(f"{experiment_id.capitalize()} | {config_name} | {total} dialogues")
    print(f"  triage     : {model_config.get('triage', 'N/A')}")
    print(f"  action     : {model_config.get('action', 'N/A')}")
    print(f"  supervisor : {model_config.get('supervisor', 'N/A')}")

    workflow = create_workflow(enable_retry=True)

    dataset_evaluator = DatasetEvaluator()
    dialogue_results = []
    failed_dialogues = 0

    for idx, dialogue in enumerate(dialogues):
        print(f"  [{idx + 1}/{total}] {dialogue['dialogue_id']}...", end=" ")

        result = run_mas_dialogue(
            dialogue=dialogue,
            model_config=model_config,
            workflow=workflow,
            judge_fn=judge_fn

        )

        if result is None:
            failed_dialogues += 1
            print("SKIPPED")
            continue

        dataset_evaluator.add_dialogue(result)
        dialogue_results.append(result)
        print(f"(turns={result['num_turns']})")

    final_metrics = dataset_evaluator.compute_dataset_metrics()
    final_metrics["model_name"] = config_name
    final_metrics["configuration"] = config_name
    final_metrics["models"] = model_config
    final_metrics["base_models"] = base_models
    final_metrics["failed_dialogues"] = failed_dialogues
    final_metrics["split"] = SPLIT

    save_experiment_results(
        dataset_metrics=final_metrics,
        dialogue_results=dialogue_results,
        experiment_id=experiment_id,
        output_dir=RESULTS_DIR
    )

    return final_metrics


def save_experiment_results(dataset_metrics: dict[str, Any], dialogue_results: list[dict[str, Any]], experiment_id: str, output_dir: str = RESULTS_DIR) -> None:
    """
    Save experiment results to 3 JSON files:
    - _dataset.json: aggregated metrics across all dialogues
    - _dialogues.json: metrics per dialogue
    - _turns.json: metrics per turn across all dialogues

    Args:
        dataset_metrics: Aggregated metrics from DatasetEvaluator
        dialogue_results: List of per-dialogue results from DialogueEvaluator
        experiment_id: Experiment prefix (e.g. 'exp1', 'exp2')
        output_dir: Directory to save results (Default: 'results/experiments')
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build shared header (unified across all experiments)
    model_safe = dataset_metrics.get("model_name", "unknown").replace("/", "-")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    timestamp_readable = time.strftime("%Y-%m-%d %H:%M:%S")
    base_filename = f"{experiment_id}_{model_safe}_{timestamp}"

    header = {
        "experiment": experiment_id,
        "evaluation_level": None,  # set per file
        "split": dataset_metrics.get("split", "validation"),
        "timestamp": timestamp_readable,
        "model_name": dataset_metrics.get("model_name", None),
        "configuration": dataset_metrics.get("configuration", None),
        "models": dataset_metrics.get("models", None),
        "base_models": dataset_metrics.get("base_models", None),
    }

    # FILE 1: Dataset-level
    dataset_file = {
        **header,
        "evaluation_level": "dataset",
        "num_dialogues": dataset_metrics.get("num_dialogues", 0),
        "failed_dialogues": dataset_metrics.get("failed_dialogues", 0),
        "task_success_rate": dataset_metrics.get("task_success_rate", 0.0),
        "avg_intent_accuracy": dataset_metrics.get("avg_intent_accuracy", 0.0),
        "avg_domain_accuracy": dataset_metrics.get("avg_domain_accuracy", 0.0),
        "avg_action_type_accuracy": dataset_metrics.get("avg_action_type_accuracy", 0.0),
        "avg_slot_accuracy": dataset_metrics.get("avg_slot_accuracy", 0.0),
        "avg_jga": dataset_metrics.get("avg_jga", 0.0),
        "avg_hallucination_rate": dataset_metrics.get("avg_hallucination_rate", 0.0),
        "avg_system_correctness": dataset_metrics.get("avg_system_correctness", 0.0),
        "avg_memory_transfer_accuracy": dataset_metrics.get("avg_memory_transfer_accuracy", 0.0),
        "policy_violation_rate": dataset_metrics.get("policy_violation_rate", 0.0),
        "total_policy_violations": dataset_metrics.get("total_policy_violations", 0),
    }

    # FILE 2: Dialogue-level
    dialogues_list = []
    for d in dialogue_results:
        dialogues_list.append({
            "dialogue_id": d.get("dialogue_id", "unknown"),
            "services": d.get("services", []),
            "num_turns": d.get("num_turns", 0),
            "task_success": d.get("task_success", False),
            "task_reason": d.get("task_reason", ""),
            "avg_intent_accuracy": d.get("avg_intent_accuracy", 0.0),
            "avg_domain_accuracy": d.get("avg_domain_accuracy", 0.0),
            "avg_action_type_accuracy": d.get("avg_action_type_accuracy", 0.0),
            "avg_slot_accuracy": d.get("avg_slot_accuracy", 0.0),
            "avg_jga": d.get("avg_jga", 0.0),
            "avg_hallucination_rate": d.get("avg_hallucination_rate", 0.0),
            "avg_system_correctness": d.get("avg_system_correctness", 0.0),
            "memory_transfer_accuracy": d.get("memory_transfer_accuracy", 0.0),
            "policy_violations": d.get("policy_violations", 0),
        })

    dialogues_file = {
        **header,
        "evaluation_level": "dialogue",
        "dialogues": dialogues_list,
    }

    # FILE 3: Turn-level
    turns_list = []
    for d in dialogue_results:
        for t in d.get("turn_metrics", []):
            turns_list.append({
                "dialogue_id": d.get("dialogue_id", "unknown"),
                "turn_id": t.get("turn_id"),
                "domain": t.get("domain"),
                "user_message": t.get("user_message", ""),
                "system_response": t.get("system_response", ""),
                "predicted_intent": t.get("predicted_intent", ""),
                "ground_truth_intent": t.get("ground_truth_intent", ""),
                "intent_correct": t.get("intent_correct", False),
                "domain_accuracy": t.get("domain_accuracy", 0.0),
                "intent_accuracy": t.get("intent_accuracy", 0.0),
                "action_type": t.get("action", ""),
                "action_type_accuracy": t.get("action_type_accuracy", 0.0),
                "predicted_slots": t.get("predicted_slots", {}),
                "slot_accuracy": t.get("slot_accuracy", 0.0),
                "jga": t.get("jga", 0.0),
                "hallucination_rate": t.get("hallucination_rate", 0.0),
                "policy_compliant": t.get("policy_compliant", True),
                "policy_reason": t.get("policy_reason", ""),
                "system_correct": t.get("system_correct", False),
                "system_reason": t.get("system_reason", ""),
                "cost": t.get("cost", 0.0),
                "response_time": t.get("response_time", 0.0),
            })

    turns_file = {
        **header,
        "evaluation_level": "turn",
        "turns": turns_list,
    }

    # Save all 3 files
    for suffix, content in [
        ("dataset", dataset_file),
        ("dialogues", dialogues_file),
        ("turns", turns_file),
    ]:
        filepath = Path(output_dir) / f"{base_filename}_{suffix}.json"
        filepath.write_text(json.dumps(content, indent=2), encoding="utf-8")

    # print(f"Results saved: {output_dir}/{base_filename}_[dataset|dialogues|turns].json")


def print_and_save_comparison_table(all_results: dict[str, Any], experiment_id: str, experiment_title: str, output_dir: str = RESULTS_DIR) -> None:
    """
    Print comparison table to terminal and append to leaderboard.txt.
    Shared across all experiments - same metrics, same format.

    Args:
        all_results: Dict mapping model_name/config_name -> metrics dict
        experiment_id: e.g. 'exp1', 'exp2'
        experiment_title: e.g. 'Single-Agent Baseline'
        output_dir: Directory to save leaderboard (Default: 'results/experiments')
    """
    from tabulate import tabulate

    rows = []
    for name, result in all_results.items():
        short_name = name.split("/")[-1]  # shorten unsloth model paths
        rows.append([
            short_name,
            f"{result['avg_domain_accuracy']:.1%}",
            f"{result['avg_intent_accuracy']:.1%}",
            f"{result['avg_action_type_accuracy']:.1%}",
            f"{result['avg_slot_accuracy']:.1%}",
            f"{result['avg_jga']:.1%}",
            f"{result['avg_hallucination_rate']:.1%}",
            f"{result['avg_memory_transfer_accuracy']:.1%}",
            f"{result['policy_violation_rate']:.1%}",
            f"{result['avg_system_correctness']:.1%}",
            f"{result['task_success_rate']:.1%}",
            f"{result['avg_judge_score']:.2f}" if result.get('avg_judge_score') else "N/A",
        ])

    # "Model/MAS" works for both single model (exp1) and architecture (exp2/3)
    headers = ["Model/MAS", "Domain%", "Intent%", "ActType%", "Slot%", "JGA%", "Hall%", "Memory%", "Policy_Viol%", "SysCorrect%", "Task%", "Judge"]
    table = tabulate(rows, headers=headers, tablefmt="github")

    # Build text block
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    block = (
        f"\n{'-'*60}\n"
        f"Experiment {experiment_id[-1]}: {experiment_title}\n"
        f"Updated: {timestamp}\n"
        f"{'-'*60}\n\n"
        f"{table}\n"
    )

    # Print to terminal
    print(block)
    print(f"Full results saved to: {output_dir}/")
    print("-" * 60)

    # Append to leaderboard.txt
    os.makedirs(output_dir, exist_ok=True)
    leaderboard_path = os.path.join(output_dir, "leaderboard.txt")
    with open(leaderboard_path, "a", encoding="utf-8") as f:
        f.write(block)

