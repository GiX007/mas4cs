"""
Experiment Helper Functions for MAS4CS.

Reusable functions for single-agent turn/dialogue execution and result persistence. Used by all experiment runners.

Updated for official MultiWOZ 2.2 GitHub format.
"""
import os
from pathlib import Path
import json
import time
from tqdm import tqdm
from typing import Any

from src.data import (
    load_split, BOOKING_REQUIRED_SLOTS, extract_gt_intent, extract_gt_slots, extract_booking, extract_dialogue_acts,
    normalize_slot_value, VALID_ACTION_TYPES, INFORMABLE_SLOTS, BOOKING_SLOTS
)
from src.models import call_model
from src.evaluation import DialogueEvaluator, DatasetEvaluator, compute_tomiinek_metrics # , gpt4_judge,
from src.utils import DEFAULT_MEGA_PROMPT, format_dialogue_history, format_policy_rules, parse_model_json_response
from src.experiments.config import MAX_DIALOGUES, SPLIT, RESULTS_DIR, RESULTS_DIR_PER_DOMAIN, TARGET_DOMAINS


def run_sa_turn(user_message: str, services: list[str], dialogue_history: list[dict[str, Any]], model_name: str, accumulated_slots: dict[str, dict[str, str]], max_tokens: int = 512, temperature: float = 0.0) -> dict[str, Any] | None:
    """
    Run one dialogue turn through the single-agent baseline with rule-based tool use.

    Calls find_entity or book_entity based on predicted intent and slots.
    DB results are passed into the prompt so LLM response is grounded.

    Args:
        user_message: Current user utterance
        services: Available domains for this dialogue
        dialogue_history: Previous turns from dataset
        model_name: Model identifier
        accumulated_slots: Slots accumulated across previous turns
        max_tokens: Maximum tokens to generate (Default: 512)
        temperature: Sampling temperature (Default: 0.0)

    Returns:
        Parsed JSON dict with added keys: db_results, informed_entity,
        booked_entity, cost, response_time. None if JSON parse fails.
    """
    from src.core.tools import find_entity, book_entity

    history_text = format_dialogue_history(dialogue_history)
    policy_text = format_policy_rules(BOOKING_REQUIRED_SLOTS)
    services_str = ", ".join(services)

    # 1. First call: get intent + slots from LLM (Pass empty entity/ref — we don't know intent yet)
    prompt_first = DEFAULT_MEGA_PROMPT.format(
        services=services_str,
        history_text=history_text,
        user_message=user_message,
        policy_text=policy_text,
        entity="none",
        ref="none",
        valid_acts=VALID_ACTION_TYPES,
    )
    # print(f"\nPrompt SA-1: {prompt_first}")

    first_response = call_model(
        model_name=model_name,
        prompt=prompt_first,
        max_tokens=max_tokens,
        temperature=temperature
    )
    # print(f"\nLLM SA-1 response:\n {first_response.text}")

    try:
        parsed = parse_model_json_response(first_response.text)
    except json.JSONDecodeError:
        print(f"JSON parse failed. Raw: {first_response.text[:100]}")
        return None

    # 2. Rule-based tool call
    domain = parsed.get("domain", "")
    intent = parsed.get("intent", "")
    new_slots = parsed.get("slots", {})

    # Filter slots by intent to prevent LLM from pre-extracting booking slots during find_ turns (LLMs complete booking slots even when user has not mentioned them yet)
    if intent.startswith("find_"):
        new_slots = {k: v for k, v in new_slots.items() if k in INFORMABLE_SLOTS}
    elif intent.startswith("book_"):
        new_slots = {k: v for k, v in new_slots.items() if k in INFORMABLE_SLOTS | BOOKING_SLOTS}
    # print(f"\nAfter filtering slots: intent={intent} | new_slots after filter={new_slots} | accumulated={accumulated_slots}")

    # Merge new slots into accumulated (same pattern as run_sa_dialogue)
    if domain not in accumulated_slots:
        accumulated_slots[domain] = {}
    for slot, value in new_slots.items():
        normalized = normalize_slot_value(value)
        # Skip dontcare/none values
        if normalized in ("dontcare", "none", "note mentioned"):
            continue
        accumulated_slots[domain][slot] = normalized

    current_slots = accumulated_slots.get(domain, {})

    # Belief state = everything the user has told us so far across the whole dialogue, not just this turn
    belief_state = {f"{domain}-{k}": v for k, v in current_slots.items()} if current_slots else {}

    db_results = []
    booked_entity = None
    informed_entity = None

    if intent.startswith("book_") and parsed.get("policy_satisfied", False):
        booked_entity = book_entity(domain, belief_state)
        if booked_entity["success"]:
            informed_entity = booked_entity["entity"]
            db_results = [booked_entity["entity"]]

    elif intent.startswith("find_") and current_slots:
        db_results = find_entity(domain, belief_state)
        if db_results:
            informed_entity = db_results[0]
            # Save recommended entity name to accumulated slots, so next turn's booking can use it without asking the user again
            recommended_name = db_results[0].get("name", "")
            if recommended_name:
                accumulated_slots[domain]["name"] = recommended_name.lower()

    # 3. Second call: generate grounded response
    entity_str = (
        ", ".join([f"{k}={v}" for k, v in informed_entity.items()])
        if informed_entity else "none"
    )
    ref_str = booked_entity["ref"] if booked_entity and booked_entity["success"] else "none"

    prompt_second = DEFAULT_MEGA_PROMPT.format(
        services=services_str,
        history_text=history_text,
        user_message=user_message,
        policy_text=policy_text,
        entity=entity_str,
        ref=ref_str,
        valid_acts=VALID_ACTION_TYPES,
    )
    # print(f"\nPrompt SA-2: {prompt_second}")

    second_response = call_model(
        model_name=model_name,
        prompt=prompt_second,
        max_tokens=max_tokens,
        temperature=temperature
    )
    # print(f"\nLLM SA-2 response:\n {second_response.text}")

    try:
        parsed_final = parse_model_json_response(second_response.text)
    except json.JSONDecodeError:
        parsed_final = parsed  # Fallback: keep first parse

    # 4. Attach metadata
    parsed_final["db_results"] = db_results
    parsed_final["informed_entity"] = informed_entity
    parsed_final["booked_entity"] = booked_entity
    parsed_final["input_tokens"] = first_response.input_tokens + second_response.input_tokens
    parsed_final["output_tokens"] = first_response.output_tokens + second_response.output_tokens
    parsed_final["cost"] = first_response.cost + second_response.cost
    parsed_final["response_time"] = first_response.response_time + second_response.response_time

    return parsed_final


def run_sa_dialogue(dialogue: dict[str, Any], model_name: str, max_tokens: int = 512, temperature: float = 0.0, judge_fn=None) -> dict[str, Any] | None:
    """
    Run one full dialogue through the single-agent baseline.

    Loops through all turns, evaluates each USER turn, and returns aggregated dialogue-level metrics.

    Args:
        dialogue: Single dialogue dict from official GitHub MultiWOZ 2.2
        model_name: LLM Model identifier from UNSLOTH_MODELS or PAID_MODELS
        max_tokens: Maximum tokens to generate (Default: 512)
        temperature: Sampling temperature (Default: 0.0)
        judge_fn: Optional judge LLM function for turn-level evaluation (Default: None)

    Returns:
        Dialogue evaluation result dict, or None if no turns evaluated
    """
    evaluator = DialogueEvaluator(policy_requirements=BOOKING_REQUIRED_SLOTS, judge_llm_fn=judge_fn)

    services = dialogue["services"]
    turns = dialogue["turns"]
    history = []
    accumulated_slots: dict[str, dict[str, str]] = {}

    for i, turn in enumerate(turns):

        if turn["speaker"] != "USER":
            continue

        result = run_sa_turn(
            user_message=turn["utterance"],
            services=services,
            dialogue_history=history,
            model_name=model_name,
            accumulated_slots=accumulated_slots,
            max_tokens=max_tokens,
            temperature=temperature
        )

        if result is None:
            history.append(turn)
            continue

        # 1. Ground truth intent
        ground_truth_intent = extract_gt_intent(turn, services)
        if not ground_truth_intent:
            continue

        # 2. Action taken
        action_taken = result["intent"]

        # 3. Ground truth slots
        ground_truth_slots = extract_gt_slots(turn["frames"])

        # 4. Ground truth dialogue acts (act type) -> we look ahead to the next SYSTEM turn
        next_system = next((t for t in turns[i + 1:] if t["speaker"] == "SYSTEM"), None)
        ground_truth_acts = extract_dialogue_acts(next_system, services) if next_system else []

        # Turn-level metrics before evaluation
        # print("\nTurn-level info before turn-evaluation:")
        # print(f"turn={turn['turn_id']} | Predicted domain: {result['domain']} | GT: {ground_truth_intent.split('_')[-1]}")
        # print(f"turn={turn['turn_id']} | Predicted intent: {result['intent']} | GT: {ground_truth_intent}")
        # print(f"turn={turn['turn_id']} | Accumulated predicted slots: {accumulated_slots} | GT: {ground_truth_slots}")
        # print(f"turn={turn['turn_id']} | Predicted acts: {result.get('action_type', [])} | GT: {ground_truth_acts} | action_taken (internal info): {action_taken}")
        # print(f"turn={turn['turn_id']} | Valid_entities: {[e['name'] for e in result.get('db_results', []) if 'name' in e]}")
        # print(f"turn={turn['turn_id']} | Response: {result.get('response', '')}")

        evaluator.evaluate_turn(
            turn_id=turn["turn_id"],
            predicted_slots=accumulated_slots,
            ground_truth_slots=ground_truth_slots,
            predicted_intent=result["intent"],
            ground_truth_intent=ground_truth_intent,
            predicted_act_type=result.get("action_type", []) if isinstance(result.get("action_type"), list) else [result.get("action_type", "")],
            ground_truth_act_type=ground_truth_acts,
            predicted_domain=result["domain"],
            action_taken=action_taken,
            user_message=turn["utterance"],
            system_response=result["response"],
            valid_entities=[e["name"] for e in result.get("db_results", []) if "name" in e],
            response_time=result.get("response_time", 0.0),
            cost=result.get("cost", 0.0)
        )

        history.append(turn)

    if not evaluator.turn_metrics:
        return None

    requires_booking = extract_booking(turns, services)

    result = evaluator.evaluate_dialogue(services=services, requires_booking=requires_booking)
    # print(f"\nDialogue done: {dialogue['dialogue_id']} | turns={result['num_turns']} | violations={result['policy_violations']} | hall={result['avg_hallucination_rate']:.2f}")

    if result is not None:
        result["dialogue_id"] = dialogue["dialogue_id"]
        result["services"] = services

    return result


def run_mas_dlg(dialogue: dict[str, Any], model_config: dict[str, str], workflow: Any, judge_fn=None) -> dict[str, Any] | None:
    """
    Run one full dialogue through the MAS workflow. Used by Experiments 2 and 3.

    Args:
        dialogue: Single dialogue dict from official GitHub MultiWOZ 2.2
        model_config: Dict mapping agent role -> model name
        workflow: Compiled LangGraph workflow
        judge_fn: Optional judge LLM function for turn-level evaluation (Default: None)

    Returns:
        Dialogue evaluation result dict, or None if no turns evaluated
    """
    from src.core import initialize_state

    evaluator = DialogueEvaluator(policy_requirements=BOOKING_REQUIRED_SLOTS, judge_llm_fn=judge_fn)

    services = dialogue["services"]
    turns = dialogue["turns"]
    accumulated_slots: dict[str, dict[str, str]] = {}
    accumulated_history: list[dict[str, str]] = []
    accumulated_booking: dict | None = None

    for i, turn in enumerate(turns):

        if turn["speaker"] != "USER":
            continue

        state = initialize_state(
            dialogue_id=dialogue["dialogue_id"],
            turn_id=turn["turn_id"],
            services=services,
            user_utterance=turn["utterance"]
        )

        state["model_config"] = model_config
        state["slots_values"] = accumulated_slots.copy()
        state["conversation_history"] = accumulated_history.copy()
        state["booked_entity"] = accumulated_booking.copy() if accumulated_booking else None

        try:
            final_state = workflow.invoke(state)
        except Exception as e:
            print(f"\nWorkflow error turn {turn['turn_id']}: {e}")
            continue

        accumulated_slots = final_state["slots_values"].copy()
        accumulated_history = final_state["conversation_history"].copy()
        accumulated_booking = final_state.get("booked_entity")

        # 1. Ground truth intent
        ground_truth_intent = extract_gt_intent(turn, services)
        if not ground_truth_intent:
            continue

        # 2. Action taken
        action_taken = final_state["active_intent"]

        # 3. Ground truth slots
        ground_truth_slots = extract_gt_slots(turn["frames"])

        # 4. Ground truth dialogue acts (act type) -> we look ahead to the next SYSTEM turn
        next_system = next((t for t in turns[i + 1:] if t["speaker"] == "SYSTEM"), None)
        ground_truth_acts = extract_dialogue_acts(next_system, services) if next_system else []

        # Turn-level metrics before evaluation
        # print("Turn-level info before turn-evaluation:")
        # print(f"turn={turn['turn_id']} | domain: {final_state['current_domain']} | GT: {ground_truth_intent.split('_')[-1]}")
        # print(f"turn={turn['turn_id']} | intent: {final_state['active_intent']} | GT: {ground_truth_intent}")
        # print(f"turn={turn['turn_id']} | action_taken: {action_taken}")
        # print(f"turn={turn['turn_id']} | slots: {final_state['slots_values']} | GT: {ground_truth_slots}")
        # print(f"turn={turn['turn_id']} | acts: {final_state['dialogue_acts']} | GT: {ground_truth_acts}")
        # print(f"turn={turn['turn_id']} | valid_entities: {final_state.get('valid_entities', [])}")
        # print(f"turn={turn['turn_id']} | utterance: {turn['utterance']}")
        # print(f"turn={turn['turn_id']} | response: {str(final_state['agent_response'])}")
        # booked = final_state.get("booked_entity")
        # if booked and booked.get("success"):
        #     print(f"turn={turn['turn_id']} | booked_entity: {booked['entity']['name']} | ref: {booked['ref']}")

        evaluator.evaluate_turn(
            turn_id=turn["turn_id"],
            predicted_slots=final_state["slots_values"],
            ground_truth_slots=ground_truth_slots,
            predicted_intent=final_state["active_intent"] or "",
            ground_truth_intent=ground_truth_intent,
            predicted_act_type=final_state["dialogue_acts"],
            ground_truth_act_type=ground_truth_acts,
            predicted_domain=final_state["current_domain"] or "",
            action_taken=action_taken,
            user_message=turn["utterance"],
            system_response=final_state["agent_response"] or "",
            valid_entities=final_state.get("valid_entities", []),
            response_time=final_state["turn_response_time"],
            cost=final_state["turn_cost"]
        )

    if not evaluator.turn_metrics:
        return None

    requires_booking = extract_booking(turns, services)

    result = evaluator.evaluate_dialogue(services=services, requires_booking=requires_booking)
    # print(f"\nDialogue done: {dialogue['dialogue_id']} | turns={result['num_turns']} | violations={result['policy_violations']} | hall={result['avg_hallucination_rate']:.2f}")

    if result is not None:
        result["dialogue_id"] = dialogue["dialogue_id"]
        result["services"] = services

    return result


def run_mas_cfg(config_name: str, model_config: dict[str, str], experiment_id: str, base_models: dict[str, str] | None = None, judge_fn=None) -> dict[str, Any]:
    """
    Run one MAS configuration on the dataset split. Used by Experiments 2 and 3.

    Args:
        config_name: Configuration name (e.g., 'homogeneous_gpt')
        model_config: Dict mapping agent role -> model name
        experiment_id: e.g. 'exp2' or 'exp3'
        base_models: Base models used for fine-tuning (exp3 only, Default: None)
        judge_fn: Optional judge LLM function (Default: None)

    Returns:
        Dataset-level metrics dictionary
    """
    from src.core import create_workflow

    dialogues = load_split(SPLIT)
    if MAX_DIALOGUES is not None:
        dialogues = dialogues[:MAX_DIALOGUES]

    total = len(dialogues)
    print("\n" + "-" * 60)
    print(f"{experiment_id.capitalize()} | {config_name} | {total} dialogues")
    print(f"  triage : {model_config.get('triage', 'N/A')}")
    print(f"  action : {model_config.get('action', 'N/A')}")
    print(f"  supervisor : {model_config.get('supervisor', 'N/A')}")

    workflow = create_workflow(enable_retry=True)
    dataset_evaluator = DatasetEvaluator()
    dialogue_results = []
    failed_dialogues = 0

    for idx, dialogue in enumerate(tqdm(dialogues, desc=f"  {config_name}", unit="dlg", leave=True)):
        print(f"  [{idx + 1}/{total}] {dialogue['dialogue_id']}...", end=" ")

        result = run_mas_dlg(
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

    if not dialogue_results:
        print(f"All dialogues failed for config '{config_name}'. Skipping save.")
        return {}

    final_metrics = dataset_evaluator.compute_dataset_metrics()
    final_metrics["model_name"] = config_name
    final_metrics["configuration"] = config_name
    final_metrics["models"] = model_config
    final_metrics["base_models"] = base_models
    final_metrics["failed_dialogues"] = failed_dialogues
    final_metrics["split"] = SPLIT

    save_exp_results(
        dataset_metrics=final_metrics,
        dialogue_results=dialogue_results,
        experiment_id=experiment_id,
        output_dir=RESULTS_DIR
    )

    save_exp_results_per_domain(
        dataset_metrics=final_metrics,
        dialogue_results=dialogue_results,
        experiment_id=experiment_id,
    )

    return final_metrics


def save_exp_results(dataset_metrics: dict[str, Any], dialogue_results: list[dict[str, Any]], experiment_id: str, output_dir: str = RESULTS_DIR) -> None:
    """
    Save experiment results to 3 JSON files:
        - _dataset.json: aggregated metrics across all dialogues
        - _dialogues.json: per-dialogue metrics
        - _turns.json: per-turn metrics across all dialogues

    Args:
        dataset_metrics: Aggregated metrics from DatasetEvaluator
        dialogue_results: List of per-dialogue results from DialogueEvaluator
        experiment_id: Experiment prefix (e.g. 'exp1', 'exp2')
        output_dir: Directory to save results (Default: RESULTS_DIR)
    """
    os.makedirs(output_dir, exist_ok=True)

    model_safe = dataset_metrics.get("model_name", "unknown").replace("/", "-")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    timestamp_readable = time.strftime("%Y-%m-%d %H:%M:%S")
    base_filename = f"{experiment_id}_{model_safe}_{timestamp}"

    header = {
        "experiment": experiment_id,
        "evaluation_level": None,
        "split": dataset_metrics.get("split", "dev"),
        "timestamp": timestamp_readable,
        "model_name": dataset_metrics.get("model_name", None),
        "configuration": dataset_metrics.get("configuration", None),
        "models": dataset_metrics.get("models", None),
        "base_models": dataset_metrics.get("base_models", None),
    }

    # Compute official Tomiinek metrics on the full dialogue set
    tomiinek = compute_tomiinek_metrics(dialogue_results)
    dataset_metrics.update(tomiinek)

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
        "avg_action_type_f1": dataset_metrics.get("avg_action_type_f1", 0.0),
        "avg_slot_accuracy": dataset_metrics.get("avg_slot_accuracy", 0.0),
        "avg_slot_f1": dataset_metrics.get("avg_slot_f1", 0.0),
        "avg_jga": dataset_metrics.get("avg_jga", 0.0),
        "avg_hallucination_rate": dataset_metrics.get("avg_hallucination_rate", 0.0),
        "avg_system_correctness": dataset_metrics.get("avg_system_correctness", 0.0),
        "policy_violation_rate": dataset_metrics.get("policy_violation_rate", 0.0),
        "total_policy_violations": dataset_metrics.get("total_policy_violations", 0),
        "total_cost": dataset_metrics.get("total_cost", 0.0),
        "avg_latency_per_turn": dataset_metrics.get("avg_latency_per_turn", 0.0),
        "inform_rate": dataset_metrics.get("inform_rate", 0.0),
        "success_rate": dataset_metrics.get("success_rate", 0.0),
        "bleu": dataset_metrics.get("bleu", 0.0),
        "combined": dataset_metrics.get("combined", 0.0),
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
            "avg_action_type_f1": d.get("avg_action_type_f1", 0.0),
            "avg_slot_accuracy": d.get("avg_slot_accuracy", 0.0),
            "avg_slot_f1": d.get("avg_slot_f1", 0.0),
            "avg_jga": d.get("avg_jga", 0.0),
            "avg_hallucination_rate": d.get("avg_hallucination_rate", 0.0),
            "avg_system_correctness": d.get("avg_system_correctness", 0.0),
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
                "action_type_f1": t.get("action_type_f1", 0.0),
                "predicted_slots": t.get("predicted_slots", {}),
                "slot_accuracy": t.get("slot_accuracy", 0.0),
                "slot_f1": t.get("slot_f1", 0.0),
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

    for suffix, content in [
        ("dataset", dataset_file),
        ("dialogues", dialogues_file),
        ("turns", turns_file),
    ]:
        filepath = Path(output_dir) / f"{base_filename}_{suffix}.json"
        filepath.write_text(json.dumps(content, indent=2), encoding="utf-8")


def print_save_table(all_results: dict[str, Any], experiment_id: str, experiment_title: str, output_dir: str = RESULTS_DIR) -> None:
    """
    Print comparison table to terminal and append to leaderboard.txt.

    Args:
        all_results: Dict mapping model_name/config_name -> metrics dict
        experiment_id: e.g. 'exp1', 'exp2'
        experiment_title: e.g. 'Single-Agent Baseline'
        output_dir: Directory to save leaderboard (Default: RESULTS_DIR)
    """
    from tabulate import tabulate

    rows = []
    for name, result in all_results.items():
        if not result:
            print(f"  Skipping empty result for config '{name}'")
            continue
        short_name = name.split("/")[-1]
        rows.append([
            short_name,
            f"{result['avg_domain_accuracy']*100:.2f}",
            f"{result['avg_intent_accuracy']*100:.2f}",
            f"{result['avg_action_type_accuracy']*100:.2f}",
            f"{result['avg_action_type_f1']*100:.2f}",
            f"{result['avg_jga']*100:.2f}",
            f"{result['avg_slot_accuracy']*100:.2f}",  # recall
            f"{result['avg_slot_f1']*100:.2f}",  # F1
            f"{result['avg_hallucination_rate']*100:.2f}",
            f"{result['policy_violation_rate']*100:.2f}",
            f"{result['avg_system_correctness']*100:.2f}",
            f"{result['task_success_rate']*100:.2f}",
            f"{result['inform_rate']:.2f}" if result.get('inform_rate') is not None else 'N/A',
            f"{result['success_rate']:.2f}" if result.get('success_rate') is not None else 'N/A',
            # f"{result['bleu']:.2f}" if result.get('bleu') is not None else 'N/A',
            # f"{result['combined']:.2f}" if result.get('combined') is not None else 'N/A',
            # f"{result['avg_judge_score']:.2f}" if result.get('avg_judge_score') else "N/A",
            f"{result.get('total_cost', 0.0):.4f}",
            f"{result.get('avg_latency_per_turn', 0.0):.2f}",
        ])

    headers = [
        "Model/MAS", "Domain%", "Intent%", "ActType-R%", "ActType-F1%", "JGA%", "Slot-R%", "Slot-F1%", "Hall%", "PolViol%", "SysCorr%", "Book%",
        "Inform%", "Success%", # "BLEU", "Combined", # "Judge",
        "Cost($)", "Latency(s)"
    ]
    table = tabulate(rows, headers=headers, tablefmt="github")

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    block = (
        f"\n{'-'*60}\n"
        f"Experiment {experiment_id[-1]}: {experiment_title}\n"
        f"Updated: {timestamp}\n"
        f"{'-'*60}\n\n"
        f"{table}\n"
    )

    print(block)
    print(f"Full results saved to: {output_dir}/")
    print("-" * 60)

    os.makedirs(output_dir, exist_ok=True)
    leaderboard_path = os.path.join(output_dir, "leaderboard.txt")
    with open(leaderboard_path, "a", encoding="utf-8") as f:
        f.write(block)


def filter_dialogue_results_by_domain(dialogue_results: list[dict[str, Any]], domain: str) -> list[dict[str, Any]]:
    """
    Filter dialogue results to only include turns matching the target domain.

    Dialogues with no turns in the target domain are excluded entirely.
    Dialogue-level averages are recomputed from the filtered turns.

    Args:
        dialogue_results: Full list of dialogue result dicts from DialogueEvaluator.
        domain: Target domain to filter by (e.g. 'hotel', 'restaurant').

    Returns:
        List of dialogue dicts with turn_metrics filtered to target domain only.
        Dialogue-level metric averages are recomputed from filtered turns.
    """
    filtered = []

    for dialogue in dialogue_results:
        domain_turns = [t for t in dialogue.get("turn_metrics", []) if t.get("domain") == domain]

        if not domain_turns:
            continue

        num_turns = len(domain_turns)

        # Recompute dialogue-level averages from domain turns only
        filtered_dialogue = {
            **dialogue,
            "turn_metrics": domain_turns,
            "num_turns": num_turns,
            "task_success": None,  # Not meaningful per domain
            "task_reason": f"Per-domain view ({domain}) — task_success not applicable",
            "avg_intent_accuracy": sum(t["intent_accuracy"] for t in domain_turns) / num_turns,
            "avg_domain_accuracy": sum(t["domain_accuracy"] for t in domain_turns) / num_turns,
            "avg_action_type_accuracy": sum(t["action_type_accuracy"] for t in domain_turns) / num_turns,
            "avg_action_type_f1": sum(t["action_type_f1"] for t in domain_turns) / num_turns,
            "avg_slot_accuracy": sum(t["slot_accuracy"] for t in domain_turns) / num_turns,
            "avg_slot_precision": sum(t["slot_precision"] for t in domain_turns) / num_turns,
            "avg_slot_f1": sum(t["slot_f1"] for t in domain_turns) / num_turns,
            "avg_jga": sum(t["jga"] for t in domain_turns) / num_turns,
            "avg_hallucination_rate": sum(t["hallucination_rate"] for t in domain_turns) / num_turns,
            "avg_system_correctness": sum(t["system_correct"] for t in domain_turns) / num_turns,
            "policy_violations": sum(1 for t in domain_turns if not t["policy_compliant"]),
        }

        filtered.append(filtered_dialogue)

    return filtered


def save_exp_results_per_domain(dataset_metrics: dict[str, Any], dialogue_results: list[dict[str, Any]], experiment_id: str, output_dir: str = RESULTS_DIR_PER_DOMAIN) -> None:
    """
    Save experiment results split by domain to results/per_domain/{domain}/.

    For each domain in TARGET_DOMAINS, filters dialogue results to domain-only turns, recomputes dataset-level metrics, and saves the same three files as save_exp_results.
    task_success (~Booking success) is excluded at domain level as it requires full dialogue context.

    Args:
        dataset_metrics: Aggregated metrics from DatasetEvaluator (for header metadata).
        dialogue_results: List of per-dialogue results from DialogueEvaluator.
        experiment_id: Experiment prefix (e.g. 'exp1', 'exp2').
        output_dir: Base directory for per-domain results (Default: RESULTS_DIR_PER_DOMAIN).
    """
    for domain in TARGET_DOMAINS:
        domain_dialogues = filter_dialogue_results_by_domain(dialogue_results, domain)

        if not domain_dialogues:
            print(f"  No dialogues found for domain '{domain}', skipping.")
            continue

        # Recompute dataset-level metrics from domain-filtered dialogues
        domain_evaluator = DatasetEvaluator()
        for d in domain_dialogues:
            domain_evaluator.add_dialogue(d)
        domain_metrics = domain_evaluator.compute_dataset_metrics()

        # Compute Tomiinek metrics for this domain subset
        tomiinek = compute_tomiinek_metrics(domain_dialogues)
        domain_metrics.update(tomiinek)

        # Carry over metadata from original run
        domain_metrics["model_name"] = dataset_metrics.get("model_name")
        domain_metrics["configuration"] = dataset_metrics.get("configuration")
        domain_metrics["models"] = dataset_metrics.get("models")
        domain_metrics["base_models"] = dataset_metrics.get("base_models")
        domain_metrics["failed_dialogues"] = dataset_metrics.get("failed_dialogues")
        domain_metrics["split"] = dataset_metrics.get("split")

        domain_output_dir = Path(output_dir) / domain
        save_exp_results(
            dataset_metrics=domain_metrics,
            dialogue_results=domain_dialogues,
            experiment_id=experiment_id,
            output_dir=str(domain_output_dir),
        )
        # print(f"  Per-domain results saved: {domain} ({len(domain_dialogues)} dialogues)")


def print_save_table_per_domain(experiment_id: str, experiment_title: str, output_dir: str = RESULTS_DIR_PER_DOMAIN) -> None:
    """
    Read per-domain _dataset.json files and print/save per-domain leaderboard tables.

    Reads already-saved results from results/per_domain/{domain}/ and builds
    one table per domain with identical headers to print_save_table.
    Appends all tables to results/per_domain/leaderboard.txt.

    Args:
        experiment_id: e.g. 'exp1', 'exp2', 'exp3'
        experiment_title: e.g. 'Single-Agent Baseline'
        output_dir: Base directory for per-domain results (Default: RESULTS_DIR_PER_DOMAIN)
    """
    from tabulate import tabulate

    headers = [
        "Model/MAS", "Domain%", "Intent%", "ActType-R%", "ActType-F1%", "JGA%", "Slot-R%", "Slot-F1%",
        "Hall%", "PolViol%", "SysCorr%", "Book%",
        "Inform", "Success", # "BLEU", "Combined", # "Judge",
        "Cost($)", "Latency(s)"
    ]

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    full_block = ""

    for domain in sorted(TARGET_DOMAINS):
        domain_dir = Path(output_dir) / domain

        if not domain_dir.exists():
            print(f"No per-domain results found for '{domain}', skipping.")
            continue

        # Read all _dataset.json files for this experiment and domain
        dataset_files = sorted(domain_dir.glob(f"{experiment_id}_*_dataset.json"))

        if not dataset_files:
            print(f"No dataset files found for experiment '{experiment_id}' in '{domain_dir}'.")
            continue

        rows = []
        for filepath in dataset_files:
            result = json.loads(filepath.read_text(encoding="utf-8"))
            short_name = (result.get("model_name") or "unknown").split("/")[-1]

            rows.append([
                short_name,
                f"{result['avg_domain_accuracy']*100:.2f}",
                f"{result['avg_intent_accuracy']*100:.2f}",
                f"{result['avg_action_type_accuracy']*100:.2f}",
                f"{result['avg_action_type_f1']*100:.2f}",
                f"{result['avg_jga']*100:.2f}",
                f"{result['avg_slot_accuracy']*100:.2f}",  # recall
                f"{result['avg_slot_f1']*100:.2f}",  # F1
                f"{result['avg_hallucination_rate']*100:.2f}",
                f"{result['policy_violation_rate']*100:.2f}",
                f"{result['avg_system_correctness']*100:.2f}",
                f"{result.get('task_success_rate', 0.0)*100:.2f}",
                f"{result['inform_rate']:.2f}" if result.get('inform_rate') is not None else 'N/A',
                f"{result['success_rate']:.2f}" if result.get('success_rate') is not None else 'N/A',
                # f"{result['bleu']:.2f}" if result.get('bleu') is not None else 'N/A',
                # f"{result['combined']:.2f}" if result.get('combined') is not None else 'N/A',
                # f"{result['avg_judge_score']:.2f}" if result.get('avg_judge_score') else "N/A",
                f"${result.get('total_cost', 0.0):.4f}",
                f"{result.get('avg_latency_per_turn', 0.0):.2f}s",
            ])

        table = tabulate(rows, headers=headers, tablefmt="github")

        block = (
            f"\n{'-' * 60}\n"
            f"Experiment {experiment_id[-1]}: {experiment_title} | Domain: {domain.upper()}\n"
            f"Updated: {timestamp}\n"
            f"{'-' * 60}\n\n"
            f"{table}\n"
        )

        print(block)
        full_block += block

    # Append all domain tables to single leaderboard file
    os.makedirs(output_dir, exist_ok=True)
    leaderboard_path = Path(output_dir) / "leaderboard.txt"
    with open(leaderboard_path, "a", encoding="utf-8") as f:
        f.write(full_block)

    print(f"Per-domain results saved to: {leaderboard_path}")
    print("-" * 60)
