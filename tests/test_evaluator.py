"""
Integration tests for DialogueEvaluator and DatasetEvaluator.

Tests turn-level evaluation, dialogue-level aggregation, and dataset-level metrics.
"""
from src.evaluation import DialogueEvaluator, DatasetEvaluator
from src.data import BOOKING_REQUIRED_SLOTS
from src.utils import print_separator, capture_and_save


def test_dialogue_evaluator_single_turn() -> None:
    """Test DialogueEvaluator with a single turn."""
    print_separator("TEST DIALOGUE EVALUATOR: SINGLE TURN")

    evaluator = DialogueEvaluator(BOOKING_REQUIRED_SLOTS)

    print("\nEvaluating single turn: User searches for restaurant")

    turn_inputs = {
        "turn_id": 1,
        "predicted_slots": {"restaurant": {"area": "centre", "food": "indian"}},
        "ground_truth_slots": {"restaurant": {"area": "centre", "food": "indian"}},
        "predicted_intent": "find_restaurant",
        "ground_truth_intent": "find_restaurant",
        "predicted_act_type": ["Restaurant-Inform"],
        "ground_truth_act_type": ["Restaurant-Inform"],
        "predicted_domain": "restaurant",
        "action_taken": "find_restaurant"
    }

    print(f"\nTurn {turn_inputs['turn_id']} INPUTS\n")
    print(f"  Predicted Slots: {turn_inputs['predicted_slots']}")
    print(f"  Ground Truth Slots: {turn_inputs['ground_truth_slots']}")
    print(f"  Predicted Intent: {turn_inputs['predicted_intent']}")
    print(f"  Ground Truth Intent: {turn_inputs['ground_truth_intent']}")
    print(f"  Predicted Action Type: {turn_inputs['predicted_act_type']}")
    print(f"  Ground Truth Action Type: {turn_inputs['ground_truth_act_type']}")
    print(f"  Predicted Domain: {turn_inputs['predicted_domain']}")
    print(f"  Action Taken: {turn_inputs['action_taken']}")

    turn_result = evaluator.evaluate_turn(**turn_inputs)

    print(f"\nTurn {turn_result['turn_id']} RESULTS\n")
    print(f"  Domain: {turn_result['domain']}")
    print(f"  Predicted Slots: {turn_result['predicted_slots']}")

    print(f"\n  Intent & Routing Metrics:")
    print(f"    Intent Accuracy: {turn_result['intent_accuracy']:.2f} ({'OK' if turn_result['intent_correct'] else 'FAIL'})")
    print(f"    Action Type Accuracy: {turn_result['action_type_accuracy']:.2f} ({'OK' if turn_result['action_type_correct'] else 'FAIL'})")
    print(f"    Action Type F1: {turn_result['action_type_f1']:.2f}")
    print(f"    Domain Accuracy: {turn_result['domain_accuracy']:.2f} ({'OK' if turn_result['domain_correct'] else 'FAIL'})")

    print(f"\n  Slot Tracking Metrics:")
    print(f"    JGA: {turn_result['jga']:.2f}")
    print(f"    JGA Breakdown: {turn_result['jga_breakdown']}")
    print(f"    Slot Recall: {turn_result['slot_accuracy']:.2f} ({turn_result['slot_correct']}/{turn_result['slot_total']} correct)")
    print(f"    Slot F1: {turn_result['slot_f1']:.2f}")
    print(f"    Hallucination Rate: {turn_result['hallucination_rate']:.2f} ({turn_result['hallucination_count']}/{turn_result['entities_mentioned']} hallucinated)")

    print(f"\n  Policy & System:")
    print(f"    Policy Compliant: {turn_result['policy_compliant']}")
    print(f"    Policy Reason: {turn_result['policy_reason']}")
    print(f"    System Correct: {turn_result['system_correct']}")
    print(f"    System Reason: {turn_result['system_reason']}")
    print(f"    Action Taken: {turn_result['action']}")

    if 'judge_score' in turn_result:
        print(f"\n  LLM Judge:")
        print(f"    Score: {turn_result['judge_score']}/5")
        print(f"    Feedback: {turn_result.get('judge_feedback', {})}")

    dialogue_result = evaluator.evaluate_dialogue(services=["restaurant"], requires_booking=False)

    print("\n" + "-" * 60)
    print(f"DIALOGUE SUMMARY")
    print("-" * 60 + "\n")

    print(f"  Task Completion:")
    print(f"    Task Success: {dialogue_result['task_success']}")
    print(f"    Task Reason: {dialogue_result['task_reason']}")
    print(f"    Num Turns: {dialogue_result['num_turns']}")

    print(f"\n  Average Intent & Routing Metrics:")
    print(f"    Avg Intent Accuracy: {dialogue_result['avg_intent_accuracy']:.2%}")
    print(f"    Avg Action Type Accuracy: {dialogue_result['avg_action_type_accuracy']:.2%}")
    print(f"    Avg Action Type F1: {dialogue_result['avg_action_type_f1']:.2%}")
    print(f"    Avg Domain Accuracy: {dialogue_result['avg_domain_accuracy']:.2%}")

    print(f"\n  Average Slot Tracking Metrics:")
    print(f"    Avg JGA: {dialogue_result['avg_jga']:.2%}")
    print(f"    Avg Slot Recall: {dialogue_result['avg_slot_accuracy']:.2%}")
    print(f"    Avg Slot F1: {dialogue_result['avg_slot_f1']:.2%}")
    print(f"    Avg Hallucination Rate: {dialogue_result['avg_hallucination_rate']:.2%}")

    print(f"\n  Policy & System:")
    print(f"    Policy Violations: {dialogue_result['policy_violations']}")
    print(f"    Avg System Correctness: {dialogue_result['avg_system_correctness']:.2%}")

    if dialogue_result['avg_judge_score'] is not None:
        print(f"\n  LLM Judge:")
        print(f"    Avg Judge Score: {dialogue_result['avg_judge_score']:.2f}/5")


def test_dialogue_evaluator_multi_turn() -> None:
    """Test DialogueEvaluator with multiple turns (hotel booking scenario)."""
    print_separator("TEST DIALOGUE EVALUATOR: MULTI-TURN BOOKING")

    evaluator = DialogueEvaluator(BOOKING_REQUIRED_SLOTS)

    print("\nSimulating and Evaluating 3-turn hotel booking dialogue\n")

    # TURN 1
    print("-" * 60)
    print("Turn 1: User provides area and price")
    print("-" * 60 + "\n")

    turn1_inputs = {
        "turn_id": 1,
        "predicted_slots": {"hotel": {"area": "centre", "pricerange": "cheap"}},
        "ground_truth_slots": {"hotel": {"area": "centre", "pricerange": "cheap"}},
        "predicted_intent": "find_hotel",
        "ground_truth_intent": "find_hotel",
        "predicted_act_type": ["Hotel-Inform"],
        "ground_truth_act_type": ["Hotel-Inform"],
        "predicted_domain": "hotel",
        "action_taken": "find_hotel"
    }

    print(f"  Predicted Slots: {turn1_inputs['predicted_slots']}")
    print(f"  Intent: {turn1_inputs['predicted_intent']} | GT: {turn1_inputs['ground_truth_intent']}")
    print(f"  Action: {turn1_inputs['action_taken']}")

    turn1 = evaluator.evaluate_turn(**turn1_inputs)

    print(f"\n  RESULTS")
    print(f"  Intent: {turn1['intent_accuracy']:.2f} | Domain: {turn1['domain_accuracy']:.2f} | JGA: {turn1['jga']:.2f}")
    print(f"  Slot-R: {turn1['slot_accuracy']:.2f} | Slot-F1: {turn1['slot_f1']:.2f} | Hall: {turn1['hallucination_rate']:.2f}")
    print(f"  Policy: {turn1['policy_compliant']} | SysCorrect: {turn1['system_correct']}")

    # TURN 2
    print("\n" + "-" * 60)
    print("Turn 2: User adds hotel name")
    print("-" * 60 + "\n")

    turn2_inputs = {
        "turn_id": 2,
        "predicted_slots": {"hotel": {"area": "centre", "pricerange": "cheap", "name": "acorn guest house"}},
        "ground_truth_slots": {"hotel": {"area": "centre", "pricerange": "cheap", "name": "acorn guest house"}},
        "predicted_intent": "find_hotel",
        "ground_truth_intent": "find_hotel",
        "predicted_act_type": ["Hotel-Inform"],
        "ground_truth_act_type": ["Hotel-Inform"],
        "predicted_domain": "hotel",
        "action_taken": "find_hotel"
    }

    print(f"  Predicted Slots: {turn2_inputs['predicted_slots']}")
    print(f"  Intent: {turn2_inputs['predicted_intent']} | GT: {turn2_inputs['ground_truth_intent']}")

    turn2 = evaluator.evaluate_turn(**turn2_inputs)

    print(f"\n  RESULTS")
    print(f"  Intent: {turn2['intent_accuracy']:.2f} | Domain: {turn2['domain_accuracy']:.2f} | JGA: {turn2['jga']:.2f}")
    print(f"  Slot-R: {turn2['slot_accuracy']:.2f} | Slot-F1: {turn2['slot_f1']:.2f} | Hall: {turn2['hallucination_rate']:.2f}")
    print(f"  Policy: {turn2['policy_compliant']} | SysCorrect: {turn2['system_correct']}")

    # TURN 3
    print("\n" + "-" * 60)
    print("Turn 3: User provides all booking slots")
    print("-" * 60 + "\n")

    turn3_inputs = {
        "turn_id": 3,
        "predicted_slots": {
            "hotel": {
                "area": "centre", "pricerange": "cheap", "name": "acorn guest house",
                "bookday": "monday", "bookpeople": "2", "bookstay": "3"
            }
        },
        "ground_truth_slots": {
            "hotel": {
                "area": "centre", "pricerange": "cheap", "name": "acorn guest house",
                "bookday": "monday", "bookpeople": "2", "bookstay": "3"
            }
        },
        "predicted_intent": "book_hotel",
        "ground_truth_intent": "book_hotel",
        "predicted_act_type": ["Hotel-Request"],
        "ground_truth_act_type": ["Hotel-Request"],
        "predicted_domain": "hotel",
        "action_taken": "book_hotel"
    }

    print(f"  Predicted Slots: {turn3_inputs['predicted_slots']}")
    print(f"  Intent: {turn3_inputs['predicted_intent']} | GT: {turn3_inputs['ground_truth_intent']}")

    turn3 = evaluator.evaluate_turn(**turn3_inputs)

    print(f"\n  RESULTS")
    print(f"  Intent: {turn3['intent_accuracy']:.2f} | Domain: {turn3['domain_accuracy']:.2f} | JGA: {turn3['jga']:.2f}")
    print(f"  Slot-R: {turn3['slot_accuracy']:.2f} | Slot-F1: {turn3['slot_f1']:.2f} | Hall: {turn3['hallucination_rate']:.2f}")
    print(f"  Policy: {turn3['policy_compliant']} | Reason: {turn3['policy_reason']}")
    print(f"  SysCorrect: {turn3['system_correct']} | Reason: {turn3['system_reason']}")

    # DIALOGUE SUMMARY
    dialogue_result = evaluator.evaluate_dialogue(services=["hotel"], requires_booking=True)

    print("\n" + "-" * 60)
    print(f"DIALOGUE SUMMARY")
    print("-" * 60 + "\n")

    print(f"  Task Success: {dialogue_result['task_success']} | Reason: {dialogue_result['task_reason']}")
    print(f"  Num Turns: {dialogue_result['num_turns']}")

    print(f"\n  Avg Intent: {dialogue_result['avg_intent_accuracy']:.2%} | Avg Domain: {dialogue_result['avg_domain_accuracy']:.2%}")
    print(f"  Avg ActType: {dialogue_result['avg_action_type_accuracy']:.2%} | Avg ActType-F1: {dialogue_result['avg_action_type_f1']:.2%}")
    print(f"  Avg JGA: {dialogue_result['avg_jga']:.2%} | Avg Slot-R: {dialogue_result['avg_slot_accuracy']:.2%} | Avg Slot-F1: {dialogue_result['avg_slot_f1']:.2%}")
    print(f"  Avg Hall: {dialogue_result['avg_hallucination_rate']:.2%} | Policy Violations: {dialogue_result['policy_violations']}")
    print(f"  Avg SysCorrect: {dialogue_result['avg_system_correctness']:.2%}")

    if dialogue_result['avg_judge_score'] is not None:
        print(f"  Avg Judge Score: {dialogue_result['avg_judge_score']:.2f}/5")


def test_dialogue_evaluator_policy_violation() -> None:
    """Test DialogueEvaluator with policy violation — booking attempted with missing slots."""
    print_separator("TEST DIALOGUE EVALUATOR: POLICY VIOLATION")

    evaluator = DialogueEvaluator(BOOKING_REQUIRED_SLOTS)

    print("\nSimulating booking attempt with missing required slots\n")

    # TURN 1
    print("-" * 60)
    print("Turn 1: Partial hotel info (name + area only)")
    print("-" * 60 + "\n")

    turn1_inputs = {
        "turn_id": 1,
        "predicted_slots": {"hotel": {"name": "acorn guest house", "area": "centre"}},
        "ground_truth_slots": {"hotel": {"name": "acorn guest house", "area": "centre"}},
        "predicted_intent": "find_hotel",
        "ground_truth_intent": "find_hotel",
        "predicted_act_type": ["Hotel-Inform"],
        "ground_truth_act_type": ["Hotel-Inform"],
        "predicted_domain": "hotel",
        "action_taken": "find_hotel"
    }

    turn1 = evaluator.evaluate_turn(**turn1_inputs)

    print(f"  Intent: {turn1['intent_accuracy']:.2f} | JGA: {turn1['jga']:.2f} | Policy: {turn1['policy_compliant']}")

    # TURN 2
    print("\n" + "-" * 60)
    print("Turn 2: System attempts book_hotel WITHOUT required slots (VIOLATION)")
    print("-" * 60 + "\n")

    turn2_inputs = {
        "turn_id": 2,
        "predicted_slots": {"hotel": {"name": "acorn guest house", "area": "centre"}},
        "ground_truth_slots": {"hotel": {"name": "acorn guest house", "area": "centre"}},
        "predicted_intent": "book_hotel",
        "ground_truth_intent": "book_hotel",
        "predicted_act_type": ["Hotel-Book"],
        "ground_truth_act_type": ["Hotel-Request"],
        "predicted_domain": "hotel",
        "action_taken": "book_hotel"  # Should have requested missing slots first
    }

    turn2 = evaluator.evaluate_turn(**turn2_inputs)

    print(f"  Intent: {turn2['intent_accuracy']:.2f} | JGA: {turn2['jga']:.2f}")
    print(f"  Policy Compliant: {turn2['policy_compliant']} ({'OK' if turn2['policy_compliant'] else 'FAIL — VIOLATION'})")
    print(f"  Policy Reason: {turn2['policy_reason']}")
    print(f"  SysCorrect: {turn2['system_correct']} | Reason: {turn2['system_reason']}")

    # DIALOGUE SUMMARY
    dialogue_result = evaluator.evaluate_dialogue(services=["hotel"], requires_booking=True)

    print("\n" + "-" * 60)
    print(f"DIALOGUE SUMMARY")
    print("-" * 60 + "\n")

    print(f"  Task Success: {dialogue_result['task_success']} | Reason: {dialogue_result['task_reason']}")
    print(f"  Policy Violations: {dialogue_result['policy_violations']} ({'FAIL — VIOLATIONS DETECTED' if dialogue_result['policy_violations'] > 0 else 'OK'})")
    print(f"  Avg SysCorrect: {dialogue_result['avg_system_correctness']:.2%}")


def test_dataset_evaluator() -> None:
    """Test DatasetEvaluator with multiple dialogues."""
    print_separator("TEST DATASET EVALUATOR: MULTIPLE DIALOGUES")

    dataset_eval = DatasetEvaluator()

    print("\nSimulating 3 dialogues with different outcomes\n")

    # DIALOGUE 1: Perfect booking success
    print("-" * 60)
    print("Dialogue 1: Perfect booking success")
    print("-" * 60 + "\n")

    dialogue1 = {
        "task_success": True,
        "task_reason": "Booking completed successfully",
        "num_turns": 2,
        "avg_intent_accuracy": 1.0,
        "avg_action_type_accuracy": 1.0,
        "avg_action_type_f1": 1.0,
        "avg_domain_accuracy": 1.0,
        "avg_jga": 1.0,
        "avg_slot_accuracy": 1.0,
        "avg_slot_precision": 1.0,
        "avg_slot_f1": 1.0,
        "avg_hallucination_rate": 0.0,
        "avg_system_correctness": 1.0,
        "policy_violations": 0,
        "avg_judge_score": None,
        "turn_metrics": []
    }

    print(f"  Task Success: {dialogue1['task_success']} | Violations: {dialogue1['policy_violations']}")
    print(f"  Intent: {dialogue1['avg_intent_accuracy']:.2%} | JGA: {dialogue1['avg_jga']:.2%} | Hall: {dialogue1['avg_hallucination_rate']:.2%}")

    dataset_eval.add_dialogue(dialogue1)

    # DIALOGUE 2: Policy violation — booking failed
    print("\n" + "-" * 60)
    print("Dialogue 2: Policy violation — booking failed")
    print("-" * 60 + "\n")

    dialogue2 = {
        "task_success": False,
        "task_reason": "Policy violations detected (1 violations)",
        "num_turns": 3,
        "avg_intent_accuracy": 0.67,
        "avg_action_type_accuracy": 0.67,
        "avg_action_type_f1": 0.72,
        "avg_domain_accuracy": 1.0,
        "avg_jga": 0.67,
        "avg_slot_accuracy": 0.80,
        "avg_slot_precision": 0.90,
        "avg_slot_f1": 0.85,
        "avg_hallucination_rate": 0.10,
        "avg_system_correctness": 0.5,
        "policy_violations": 1,
        "avg_judge_score": None,
        "turn_metrics": []
    }

    print(f"  Task Success: {dialogue2['task_success']} | Violations: {dialogue2['policy_violations']}")
    print(f"  Intent: {dialogue2['avg_intent_accuracy']:.2%} | JGA: {dialogue2['avg_jga']:.2%} | Hall: {dialogue2['avg_hallucination_rate']:.2%}")

    dataset_eval.add_dialogue(dialogue2)

    # DIALOGUE 3: Info-only — booking not required (returns None, excluded from Book%)
    print("\n" + "-" * 60)
    print("Dialogue 3: Info-only dialogue (no booking required — excluded from Book%)")
    print("-" * 60 + "\n")

    dialogue3 = {
        "task_success": None,   # None = info-only, skipped in booking aggregation
        "task_reason": "No booking required — not applicable",
        "num_turns": 2,
        "avg_intent_accuracy": 1.0,
        "avg_action_type_accuracy": 0.50,
        "avg_action_type_f1": 0.60,
        "avg_domain_accuracy": 1.0,
        "avg_jga": 0.50,
        "avg_slot_accuracy": 0.60,
        "avg_slot_precision": 0.70,
        "avg_slot_f1": 0.65,
        "avg_hallucination_rate": 0.0,
        "avg_system_correctness": 1.0,
        "policy_violations": 0,
        "avg_judge_score": None,
        "turn_metrics": []
    }

    print(f"  Task Success: {dialogue3['task_success']} (None = excluded from Book%)")
    print(f"  Intent: {dialogue3['avg_intent_accuracy']:.2%} | JGA: {dialogue3['avg_jga']:.2%} | Hall: {dialogue3['avg_hallucination_rate']:.2%}")

    dataset_eval.add_dialogue(dialogue3)

    # DATASET-LEVEL AGGREGATION
    dataset_metrics = dataset_eval.compute_dataset_metrics()

    print("\n" + "-" * 60)
    print("DATASET-LEVEL METRICS (MACRO)")
    print("-" * 60 + "\n")

    booking_count = sum(1 for d in dataset_eval.dialogue_results if d["task_success"] is not None)
    print(f"  Dialogues: {dataset_metrics['num_dialogues']} total | {booking_count} booking dialogues")
    print(f"  Book% (booking only): {dataset_metrics['task_success_rate']:.2%}")

    print(f"\n  Avg Intent: {dataset_metrics['avg_intent_accuracy']:.2%} | Avg Domain: {dataset_metrics['avg_domain_accuracy']:.2%}")
    print(f"  Avg ActType: {dataset_metrics['avg_action_type_accuracy']:.2%} | Avg ActType-F1: {dataset_metrics['avg_action_type_f1']:.2%}")
    print(f"  Avg JGA: {dataset_metrics['avg_jga']:.2%} | Avg Slot-R: {dataset_metrics['avg_slot_accuracy']:.2%} | Avg Slot-F1: {dataset_metrics['avg_slot_f1']:.2%}")
    print(f"  Avg Hall: {dataset_metrics['avg_hallucination_rate']:.2%}")
    print(f"  Policy Violation Rate: {dataset_metrics['policy_violation_rate']:.2%} | Total Violations: {dataset_metrics['total_policy_violations']}")
    print(f"  Avg SysCorrect: {dataset_metrics['avg_system_correctness']:.2%}")

    if dataset_metrics.get('avg_judge_score') is not None:
        print(f"  Avg Judge Score: {dataset_metrics['avg_judge_score']:.2f}/5")


def test_evaluator_reset() -> None:
    """Test that evaluator reset works correctly."""
    print_separator("TEST EVALUATOR RESET FUNCTIONALITY")

    evaluator = DialogueEvaluator(BOOKING_REQUIRED_SLOTS)

    # DIALOGUE 1: 2 turns
    print("\nDialogue 1: Evaluate 2 turns")

    for i in range(1, 3):
        evaluator.evaluate_turn(
            turn_id=i,
            predicted_slots={"restaurant": {"area": "centre"}},
            ground_truth_slots={"restaurant": {"area": "centre"}},
            predicted_intent="find_restaurant",
            ground_truth_intent="find_restaurant",
            predicted_act_type=["Restaurant-Inform"],
            ground_truth_act_type=["Restaurant-Inform"],
            predicted_domain="restaurant",
            action_taken="find_restaurant"
        )

    print(f"  Turns accumulated: {len(evaluator.turn_metrics)} (expected 2)")

    # RESET
    evaluator.reset()
    print(f"\nAfter reset:")
    print(f"  Turns: {len(evaluator.turn_metrics)} (expected 0)")
    print(f"  History: {len(evaluator.dialogue_history)} (expected 0)")
    print(f"  Reset OK: {len(evaluator.turn_metrics) == 0 and len(evaluator.dialogue_history) == 0}")

    # DIALOGUE 2: 1 turn after reset
    print("\nDialogue 2: 1 turn after reset")

    evaluator.evaluate_turn(
        turn_id=1,
        predicted_slots={"hotel": {"area": "north"}},
        ground_truth_slots={"hotel": {"area": "north"}},
        predicted_intent="find_hotel",
        ground_truth_intent="find_hotel",
        predicted_act_type=["Hotel-Inform"],
        ground_truth_act_type=["Hotel-Inform"],
        predicted_domain="hotel",
        action_taken="find_hotel"
    )

    print(f"  Turns: {len(evaluator.turn_metrics)} (expected 1)")
    reset_ok = len(evaluator.turn_metrics) == 1 and len(evaluator.dialogue_history) == 1
    print(f"  Reset functionality: {'OK' if reset_ok else 'FAIL'}")


def run_all_tests() -> None:
    """Run all DialogueEvaluator and DatasetEvaluator tests in sequence."""
    print_separator("TEST EVALUATOR")

    test_dialogue_evaluator_single_turn()
    test_dialogue_evaluator_multi_turn()
    test_dialogue_evaluator_policy_violation()
    test_dataset_evaluator()
    test_evaluator_reset()

    print_separator("ALL EVALUATOR TESTS COMPLETED")


if __name__ == "__main__":
    capture_and_save(
        func=run_all_tests,
        output_path="docs/evals_inspection/evaluator_test_results.txt"
    )
