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

    policy_requirements = BOOKING_REQUIRED_SLOTS
    evaluator = DialogueEvaluator(policy_requirements)

    print("\nEvaluating single turn: User searches for restaurant")

    # Create dummy inputs
    turn_inputs = {
        "turn_id": 1,
        "predicted_slots": {"restaurant": {"area": "centre", "food": "indian"}},
        "ground_truth_slots": {"restaurant": {"area": "centre", "food": "indian"}},
        "predicted_intent": "find_restaurant",
        "ground_truth_intent": "find_restaurant",
        "predicted_act_type": ["Restaurant-Inform"],
        "ground_truth_act_type": ["Restaurant-Inform"],
        "predicted_domain": "restaurant",
        "action_taken": "search_restaurant"
    }

    # Print inputs
    print(f"\nTurn {turn_inputs['turn_id']} INPUTS\n")
    print(f"  Predicted Slots: {turn_inputs['predicted_slots']}")
    print(f"  Ground Truth Slots: {turn_inputs['ground_truth_slots']}")
    print(f"  Predicted Intent: {turn_inputs['predicted_intent']}")
    print(f"  Ground Truth Intent: {turn_inputs['ground_truth_intent']}")
    print(f"  Predicted Action Type: {turn_inputs['predicted_act_type']}")
    print(f"  Ground Truth Action Type: {turn_inputs['ground_truth_act_type']}")
    print(f"  Predicted Domain: {turn_inputs['predicted_domain']}")
    print(f"  Action Taken: {turn_inputs['action_taken']}")

    # Call evaluator for single turn evaluation
    turn_result = evaluator.evaluate_turn(**turn_inputs)

    # Print turn-level results (ALL metrics)
    print(f"\nTurn {turn_result['turn_id']} RESULTS\n")
    print(f"  Domain: {turn_result['domain']}")
    print(f"  Predicted Slots: {turn_result['predicted_slots']}")

    print(f"\n  Intent & Routing Metrics:")
    print(
        f"    Intent Accuracy: {turn_result['intent_accuracy']:.2f} ({'OK' if turn_result['intent_correct'] else 'ERROR'})")
    print(
        f"    Action Type Accuracy: {turn_result['action_type_accuracy']:.2f} ({'OK' if turn_result['action_type_correct'] else 'ERROR'})")
    print(
        f"    Domain Accuracy: {turn_result['domain_accuracy']:.2f} ({'OK' if turn_result['domain_correct'] else 'ERROR'})")

    print(f"\n  Slot Tracking Metrics:")
    print(f"    JGA: {turn_result['jga']:.2f}")
    print(f"    JGA Breakdown: {turn_result['jga_breakdown']}")
    print(
        f"    Slot Accuracy: {turn_result['slot_accuracy']:.2f} ({turn_result['slot_correct']}/{turn_result['slot_total']} correct)")
    print(
        f"    Hallucination Rate: {turn_result['hallucination_rate']:.2f} ({turn_result['hallucination_count']}/{turn_result['prediction_count']} hallucinated)")

    print(f"\n  Policy & Action:")
    print(f"    Policy Compliant: {turn_result['policy_compliant']}")
    print(f"    Policy Reason: {turn_result['policy_reason']}")
    print(f"    Action Taken: {turn_result['action']}")

    # LLM Judge (if available)
    if 'judge_score' in turn_result:
        print(f"\n  LLM Judge:")
        print(f"    Score: {turn_result['judge_score']}/5")
        print(f"    Feedback: {turn_result.get('judge_feedback', {})}")

    # Evaluate dialogue
    ground_truth_goal = {
        "domains": ["restaurant"],
        "requires_booking": False
    }

    dialogue_result = evaluator.evaluate_dialogue(ground_truth_goal)

    # Print dialogue-level results (ALL metrics)
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
    print(f"    Avg Domain Accuracy: {dialogue_result['avg_domain_accuracy']:.2%}")

    print(f"\n  Average Slot Tracking Metrics:")
    print(f"    Avg JGA: {dialogue_result['avg_jga']:.2%}")
    print(f"    Avg Slot Accuracy: {dialogue_result['avg_slot_accuracy']:.2%}")
    print(f"    Avg Hallucination Rate: {dialogue_result['avg_hallucination_rate']:.2%}")

    print(f"\n  Memory Transfer:")
    print(f"    Memory Transfer Accuracy: {dialogue_result['memory_transfer_accuracy']:.2%}")
    print(f"    Memory Transfers: {dialogue_result['memory_correct']}/{dialogue_result['memory_total']}")
    print(f"    Memory Events: {len(dialogue_result['memory_events'])}")

    print(f"\n  Policy Compliance:")
    print(f"    Policy Violations: {dialogue_result['policy_violations']}")

    # LLM Judge (if available)
    if dialogue_result['avg_judge_score'] is not None:
        print(f"\n  LLM Judge:")
        print(f"    Avg Judge Score: {dialogue_result['avg_judge_score']:.2f}/5")


def test_dialogue_evaluator_multi_turn() -> None:
    """Test DialogueEvaluator with multiple turns (hotel booking scenario)."""

    print_separator("TEST DIALOGUE EVALUATOR: MULTI-TURN BOOKING")

    policy_requirements = BOOKING_REQUIRED_SLOTS
    evaluator = DialogueEvaluator(policy_requirements)

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
        "action_taken": "search_hotel"
    }

    # Print inputs
    print(f"  Turn {turn1_inputs['turn_id']} INPUTS\n")
    print(f"  Predicted Slots: {turn1_inputs['predicted_slots']}")
    print(f"  Ground Truth Slots: {turn1_inputs['ground_truth_slots']}")
    print(f"  Predicted Intent: {turn1_inputs['predicted_intent']}")
    print(f"  Ground Truth Intent: {turn1_inputs['ground_truth_intent']}")
    print(f"  Predicted Action Type: {turn1_inputs['predicted_act_type']}")
    print(f"  Ground Truth Action Type: {turn1_inputs['ground_truth_act_type']}")
    print(f"  Predicted Domain: {turn1_inputs['predicted_domain']}")
    print(f"  Action Taken: {turn1_inputs['action_taken']}")

    # Evaluate turn
    turn1 = evaluator.evaluate_turn(**turn1_inputs)

    # Print results
    print(f"\n  Turn {turn1['turn_id']} RESULTS\n")
    print(f"  Intent Accuracy: {turn1['intent_accuracy']:.2f} ({'OK' if turn1['intent_correct'] else 'FAIL'})")
    print(
        f"  Action Type Accuracy: {turn1['action_type_accuracy']:.2f} ({'OK' if turn1['action_type_correct'] else 'FAIL'})")
    print(f"  Domain Accuracy: {turn1['domain_accuracy']:.2f} ({'OK' if turn1['domain_correct'] else 'FAIL'})")
    print(f"  JGA: {turn1['jga']:.2f}")
    print(f"  Slot Accuracy: {turn1['slot_accuracy']:.2f} ({turn1['slot_correct']}/{turn1['slot_total']})")
    print(f"  Hallucination Rate: {turn1['hallucination_rate']:.2f}")
    print(f"  Policy Compliant: {turn1['policy_compliant']}")

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
        "action_taken": "inform_hotel"
    }

    print(f"  Turn {turn2_inputs['turn_id']} INPUTS\n")
    print(f"  Predicted Slots: {turn2_inputs['predicted_slots']}")
    print(f"  Ground Truth Slots: {turn2_inputs['ground_truth_slots']}")
    print(f"  Predicted Intent: {turn2_inputs['predicted_intent']}")
    print(f"  Ground Truth Intent: {turn2_inputs['ground_truth_intent']}")
    print(f"  Predicted Action Type: {turn2_inputs['predicted_act_type']}")
    print(f"  Ground Truth Action Type: {turn2_inputs['ground_truth_act_type']}")
    print(f"  Predicted Domain: {turn2_inputs['predicted_domain']}")
    print(f"  Action Taken: {turn2_inputs['action_taken']}")

    turn2 = evaluator.evaluate_turn(**turn2_inputs)

    print(f"\n  Turn {turn2['turn_id']} RESULTS\n")
    print(f"  Intent Accuracy: {turn2['intent_accuracy']:.2f} ({'OK' if turn2['intent_correct'] else 'FAIL'})")
    print(
        f"  Action Type Accuracy: {turn2['action_type_accuracy']:.2f} ({'OK' if turn2['action_type_correct'] else 'FAIL'})")
    print(f"  Domain Accuracy: {turn2['domain_accuracy']:.2f} ({'OK' if turn2['domain_correct'] else 'FAIL'})")
    print(f"  JGA: {turn2['jga']:.2f}")
    print(f"  Slot Accuracy: {turn2['slot_accuracy']:.2f} ({turn2['slot_correct']}/{turn2['slot_total']})")
    print(f"  Hallucination Rate: {turn2['hallucination_rate']:.2f}")
    print(f"  Policy Compliant: {turn2['policy_compliant']}")

    # TURN 3
    print("\n" + "-" * 60)
    print("Turn 3: User provides booking details (bookday, bookpeople, bookstay)")
    print("-" * 60 + "\n")

    turn3_inputs = {
        "turn_id": 3,
        "predicted_slots": {
            "hotel": {
                "area": "centre",
                "pricerange": "cheap",
                "name": "acorn guest house",
                "bookday": "monday",
                "bookpeople": "2",
                "bookstay": "3"
            }
        },
        "ground_truth_slots": {
            "hotel": {
                "area": "centre",
                "pricerange": "cheap",
                "name": "acorn guest house",
                "bookday": "monday",
                "bookpeople": "2",
                "bookstay": "3"
            }
        },
        "predicted_intent": "book_hotel",
        "ground_truth_intent": "book_hotel",
        "predicted_act_type": ["Hotel-Request"],
        "ground_truth_act_type": ["Hotel-Request"],
        "predicted_domain": "hotel",
        "action_taken": "book_hotel"
    }

    print(f"  Turn {turn3_inputs['turn_id']} INPUTS\n")
    print(f"  Predicted Slots: {turn3_inputs['predicted_slots']}")
    print(f"  Ground Truth Slots: {turn3_inputs['ground_truth_slots']}")
    print(f"  Predicted Intent: {turn3_inputs['predicted_intent']}")
    print(f"  Ground Truth Intent: {turn3_inputs['ground_truth_intent']}")
    print(f"  Predicted Action Type: {turn3_inputs['predicted_act_type']}")
    print(f"  Ground Truth Action Type: {turn3_inputs['ground_truth_act_type']}")
    print(f"  Predicted Domain: {turn3_inputs['predicted_domain']}")
    print(f"  Action Taken: {turn3_inputs['action_taken']}")

    turn3 = evaluator.evaluate_turn(**turn3_inputs)

    print(f"\n  Turn {turn3['turn_id']} RESULTS\n")
    print(f"  Intent Accuracy: {turn3['intent_accuracy']:.2f} ({'OK' if turn3['intent_correct'] else 'FAIL'})")
    print(
        f"  Action Type Accuracy: {turn3['action_type_accuracy']:.2f} ({'OK' if turn3['action_type_correct'] else 'FAIL'})")
    print(f"  Domain Accuracy: {turn3['domain_accuracy']:.2f} ({'OK' if turn3['domain_correct'] else 'FAIL'})")
    print(f"  JGA: {turn3['jga']:.2f}")
    print(f"  Slot Accuracy: {turn3['slot_accuracy']:.2f} ({turn3['slot_correct']}/{turn3['slot_total']})")
    print(f"  Hallucination Rate: {turn3['hallucination_rate']:.2f}")
    print(f"  Policy Compliant: {turn3['policy_compliant']}")
    print(f"  Policy Reason: {turn3['policy_reason']}")


    # DIALOGUE-LEVEL EVALUATION

    ground_truth_goal = {
        "domains": ["hotel"],
        "requires_booking": True
    }

    dialogue_result = evaluator.evaluate_dialogue(ground_truth_goal)

    print("\n" + "-" * 60)
    print(f"DIALOGUE SUMMARY")
    print("-" * 60 + "\n")

    print(f"  Ground truth goal: {ground_truth_goal}\n")
    print(f"  Task Completion:")
    print(f"    Task Success: {dialogue_result['task_success']}")
    print(f"    Task Reason: {dialogue_result['task_reason']}")
    print(f"    Num Turns: {dialogue_result['num_turns']}")

    print(f"\n  Average Intent & Routing Metrics:")
    print(f"    Avg Intent Accuracy: {dialogue_result['avg_intent_accuracy']:.2%}")
    print(f"    Avg Action Type Accuracy: {dialogue_result['avg_action_type_accuracy']:.2%}")
    print(f"    Avg Domain Accuracy: {dialogue_result['avg_domain_accuracy']:.2%}")

    print(f"\n  Average Slot Tracking Metrics:")
    print(f"    Avg JGA: {dialogue_result['avg_jga']:.2%}")
    print(f"    Avg Slot Accuracy: {dialogue_result['avg_slot_accuracy']:.2%}")
    print(f"    Avg Hallucination Rate: {dialogue_result['avg_hallucination_rate']:.2%}")

    print(f"\n  Memory Transfer:")
    print(f"    Memory Transfer Accuracy: {dialogue_result['memory_transfer_accuracy']:.2%}")
    print(f"    Memory Transfers: {dialogue_result['memory_correct']}/{dialogue_result['memory_total']}")

    print(f"\n  Policy Compliance:")
    print(f"    Policy Violations: {dialogue_result['policy_violations']}")

    if dialogue_result['avg_judge_score'] is not None:
        print(f"\n  LLM Judge:")
        print(f"    Avg Judge Score: {dialogue_result['avg_judge_score']:.2f}/5")


def test_dialogue_evaluator_memory_transfer() -> None:
    """Test DialogueEvaluator with cross-domain memory transfer."""

    print_separator("TEST DIALOGUE EVALUATOR: MEMORY TRANSFER")

    policy_requirements = BOOKING_REQUIRED_SLOTS
    evaluator = DialogueEvaluator(policy_requirements)

    print("\nSimulating restaurant → hotel domain switch with memory transfer\n")

    # TURN 1
    print("-" * 60)
    print("Turn 1: Find restaurant (area=centre)")
    print("-" * 60 + "\n")

    turn1_inputs = {
        "turn_id": 1,
        "predicted_slots": {"restaurant": {"area": "centre", "food": "indian"}},
        "ground_truth_slots": {"restaurant": {"area": "centre", "food": "indian"}},
        "predicted_intent": "find_restaurant",
        "ground_truth_intent": "find_restaurant",
        "predicted_act_type": ["Restaurant-Inform"],
        "ground_truth_act_type": ["Restaurant-Inform"],
        "predicted_domain": "restaurant",
        "action_taken": "search_restaurant"
    }

    print(f"  Turn {turn1_inputs['turn_id']} INPUTS\n")
    print(f"  Predicted Slots: {turn1_inputs['predicted_slots']}")
    print(f"  Ground Truth Slots: {turn1_inputs['ground_truth_slots']}")
    print(f"  Predicted Intent: {turn1_inputs['predicted_intent']}")
    print(f"  Ground Truth Intent: {turn1_inputs['ground_truth_intent']}")
    print(f"  Predicted Action Type: {turn1_inputs['predicted_act_type']}")
    print(f"  Ground Truth Action Type: {turn1_inputs['ground_truth_act_type']}")
    print(f"  Predicted Domain: {turn1_inputs['predicted_domain']}")
    print(f"  Action Taken: {turn1_inputs['action_taken']}")

    turn1 = evaluator.evaluate_turn(**turn1_inputs)

    print(f"\n  Turn {turn1['turn_id']} RESULTS\n")
    print(f"  Intent Accuracy: {turn1['intent_accuracy']:.2f} ({'OK' if turn1['intent_correct'] else 'FAIL'})")
    print(f"  Action Type Accuracy: {turn1['action_type_accuracy']:.2f} ({'OK' if turn1['action_type_correct'] else 'FAIL'})")
    print(f"  Domain Accuracy: {turn1['domain_accuracy']:.2f} ({'OK' if turn1['domain_correct'] else 'FAIL'})")
    print(f"  JGA: {turn1['jga']:.2f}")
    print(f"  Slot Accuracy: {turn1['slot_accuracy']:.2f} ({turn1['slot_correct']}/{turn1['slot_total']})")
    print(f"  Hallucination Rate: {turn1['hallucination_rate']:.2f}")
    print(f"  Policy Compliant: {turn1['policy_compliant']}")

    # TURN 2
    print("\n" + "-" * 60)
    print("Turn 2: Switch to hotel (area should transfer from restaurant)")
    print("-" * 60 + "\n")

    turn2_inputs = {
        "turn_id": 2,
        "predicted_slots": {
            "restaurant": {"area": "centre", "food": "indian"},
            "hotel": {"area": "centre", "pricerange": "cheap"}  # area carried over
        },
        "ground_truth_slots": {
            "restaurant": {"area": "centre", "food": "indian"},
            "hotel": {"area": "centre", "pricerange": "cheap"}
        },
        "predicted_intent": "find_hotel",
        "ground_truth_intent": "find_hotel",
        "predicted_act_type": ["Hotel-Inform"],
        "ground_truth_act_type": ["Hotel-Inform"],
        "predicted_domain": "hotel",
        "action_taken": "search_hotel"
    }

    print(f"  Turn {turn2_inputs['turn_id']} INPUTS\n")
    print(f"  Predicted Slots: {turn2_inputs['predicted_slots']}")
    print(f"  Ground Truth Slots: {turn2_inputs['ground_truth_slots']}")
    print(f"  Predicted Intent: {turn2_inputs['predicted_intent']}")
    print(f"  Ground Truth Intent: {turn2_inputs['ground_truth_intent']}")
    print(f"  Predicted Action Type: {turn2_inputs['predicted_act_type']}")
    print(f"  Ground Truth Action Type: {turn2_inputs['ground_truth_act_type']}")
    print(f"  Predicted Domain: {turn2_inputs['predicted_domain']}")
    print(f"  Action Taken: {turn2_inputs['action_taken']}")

    turn2 = evaluator.evaluate_turn(**turn2_inputs)

    print(f"\n  Turn {turn2['turn_id']} RESULTS\n")
    print(f"  Intent Accuracy: {turn2['intent_accuracy']:.2f} ({'OK' if turn2['intent_correct'] else 'FAIL'})")
    print(f"  Action Type Accuracy: {turn2['action_type_accuracy']:.2f} ({'OK' if turn2['action_type_correct'] else 'FAIL'})")
    print(f"  Domain Accuracy: {turn2['domain_accuracy']:.2f} ({'OK' if turn2['domain_correct'] else 'FAIL'})")
    print(f"  JGA: {turn2['jga']:.2f}")
    print(f"  Slot Accuracy: {turn2['slot_accuracy']:.2f} ({turn2['slot_correct']}/{turn2['slot_total']})")
    print(f"  Hallucination Rate: {turn2['hallucination_rate']:.2f}")
    print(f"  Policy Compliant: {turn2['policy_compliant']}")

    # DIALOGUE-LEVEL EVALUATION
    ground_truth_goal = {
        "domains": ["restaurant", "hotel"],
        "requires_booking": False
    }

    dialogue_result = evaluator.evaluate_dialogue(ground_truth_goal)

    print("\n" + "-" * 60)
    print(f"DIALOGUE SUMMARY")
    print("-" * 60 + "\n")

    print(f"  Ground truth goal: {ground_truth_goal}\n")
    print(f"  Task Completion:")
    print(f"    Task Success: {dialogue_result['task_success']}")
    print(f"    Task Reason: {dialogue_result['task_reason']}")
    print(f"    Num Turns: {dialogue_result['num_turns']}")

    print(f"\n  Average Intent & Routing Metrics:")
    print(f"    Avg Intent Accuracy: {dialogue_result['avg_intent_accuracy']:.2%}")
    print(f"    Avg Action Type Accuracy: {dialogue_result['avg_action_type_accuracy']:.2%}")
    print(f"    Avg Domain Accuracy: {dialogue_result['avg_domain_accuracy']:.2%}")

    print(f"\n  Average Slot Tracking Metrics:")
    print(f"    Avg JGA: {dialogue_result['avg_jga']:.2%}")
    print(f"    Avg Slot Accuracy: {dialogue_result['avg_slot_accuracy']:.2%}")
    print(f"    Avg Hallucination Rate: {dialogue_result['avg_hallucination_rate']:.2%}")

    print(f"\n  Memory Transfer:")
    print(f"    Memory Transfer Accuracy: {dialogue_result['memory_transfer_accuracy']:.2%}")
    print(f"    Memory Transfers: {dialogue_result['memory_correct']}/{dialogue_result['memory_total']}")

    print(f"\n  Policy Compliance:")
    print(f"    Policy Violations: {dialogue_result['policy_violations']}")

    if dialogue_result['avg_judge_score'] is not None:
        print(f"\n  LLM Judge:")
        print(f"    Avg Judge Score: {dialogue_result['avg_judge_score']:.2f}/5")

    print(f"\n  Memory Transfer Events:")
    for event in dialogue_result['memory_events']:
        status = "OK" if event['transferred'] else "FAIL"
        print(f"    [{status}] Turn {event['turn_id']}: {event['from_domain']} → {event['to_domain']}, "
              f"slot='{event['slot']}', value='{event['value']}'")


def test_dialogue_evaluator_policy_violation() -> None:
    """Test DialogueEvaluator with policy violation."""

    print_separator("TEST DIALOGUE EVALUATOR: POLICY VIOLATION")

    policy_requirements = BOOKING_REQUIRED_SLOTS
    evaluator = DialogueEvaluator(policy_requirements)

    print("\nSimulating booking attempt with missing required slots\n")

    # TURN 1
    print("-" * 60)
    print("Turn 1: Partial hotel info (missing bookday, bookpeople, bookstay)")
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
        "action_taken": "search_hotel"
    }

    print(f"  Turn {turn1_inputs['turn_id']} INPUTS\n")
    print(f"  Predicted Slots: {turn1_inputs['predicted_slots']}")
    print(f"  Ground Truth Slots: {turn1_inputs['ground_truth_slots']}")
    print(f"  Predicted Intent: {turn1_inputs['predicted_intent']}")
    print(f"  Ground Truth Intent: {turn1_inputs['ground_truth_intent']}")
    print(f"  Predicted Action Type: {turn1_inputs['predicted_act_type']}")
    print(f"  Ground Truth Action Type: {turn1_inputs['ground_truth_act_type']}")
    print(f"  Predicted Domain: {turn1_inputs['predicted_domain']}")
    print(f"  Action Taken: {turn1_inputs['action_taken']}")

    turn1 = evaluator.evaluate_turn(**turn1_inputs)

    print(f"\n  Turn {turn1['turn_id']} RESULTS\n")
    print(f"  Intent Accuracy: {turn1['intent_accuracy']:.2f} ({'OK' if turn1['intent_correct'] else 'FAIL'})")
    print(f"  Action Type Accuracy: {turn1['action_type_accuracy']:.2f} ({'OK' if turn1['action_type_correct'] else 'FAIL'})")
    print(f"  Domain Accuracy: {turn1['domain_accuracy']:.2f} ({'OK' if turn1['domain_correct'] else 'FAIL'})")
    print(f"  JGA: {turn1['jga']:.2f}")
    print(f"  Slot Accuracy: {turn1['slot_accuracy']:.2f} ({turn1['slot_correct']}/{turn1['slot_total']})")
    print(f"  Hallucination Rate: {turn1['hallucination_rate']:.2f}")
    print(f"  Policy Compliant: {turn1['policy_compliant']}")
    print(f"  Policy Reason: {turn1['policy_reason']}")

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
        "action_taken": "book_hotel"  # Should have been request_slots!
    }

    print(f"  Turn {turn2_inputs['turn_id']} INPUTS\n")
    print(f"  Predicted Slots: {turn2_inputs['predicted_slots']}")
    print(f"  Ground Truth Slots: {turn2_inputs['ground_truth_slots']}")
    print(f"  Predicted Intent: {turn2_inputs['predicted_intent']}")
    print(f"  Ground Truth Intent: {turn2_inputs['ground_truth_intent']}")
    print(f"  Predicted Action Type: {turn2_inputs['predicted_act_type']}")
    print(f"  Ground Truth Action Type: {turn2_inputs['ground_truth_act_type']}")
    print(f"  Predicted Domain: {turn2_inputs['predicted_domain']}")
    print(f"  Action Taken: {turn2_inputs['action_taken']}")

    turn2 = evaluator.evaluate_turn(**turn2_inputs)

    print(f"\n  Turn {turn2['turn_id']} RESULTS\n")
    print(f"  Intent Accuracy: {turn2['intent_accuracy']:.2f} ({'OK' if turn2['intent_correct'] else 'FAIL'})")
    print(f"  Action Type Accuracy: {turn2['action_type_accuracy']:.2f} ({'OK' if turn2['action_type_correct'] else 'FAIL'})")
    print(f"  Domain Accuracy: {turn2['domain_accuracy']:.2f} ({'OK' if turn2['domain_correct'] else 'FAIL'})")
    print(f"  JGA: {turn2['jga']:.2f}")
    print(f"  Slot Accuracy: {turn2['slot_accuracy']:.2f} ({turn2['slot_correct']}/{turn2['slot_total']})")
    print(f"  Hallucination Rate: {turn2['hallucination_rate']:.2f}")
    print(f"  Policy Compliant: {turn2['policy_compliant']} ({'OK' if turn2['policy_compliant'] else 'FAIL - VIOLATION'})")
    print(f"  Policy Reason: {turn2['policy_reason']}")

    # DIALOGUE-LEVEL EVALUATION
    ground_truth_goal = {
        "domains": ["hotel"],
        "requires_booking": True
    }

    dialogue_result = evaluator.evaluate_dialogue(ground_truth_goal)

    print("\n" + "-" * 60)
    print(f"DIALOGUE SUMMARY")
    print("-" * 60 + "\n")

    print(f"  Ground truth goal: {ground_truth_goal}\n")
    print(f"  Task Completion:")
    print(f"    Task Success: {dialogue_result['task_success']}")
    print(f"    Task Reason: {dialogue_result['task_reason']}")
    print(f"    Num Turns: {dialogue_result['num_turns']}")

    print(f"\n  Average Intent & Routing Metrics:")
    print(f"    Avg Intent Accuracy: {dialogue_result['avg_intent_accuracy']:.2%}")
    print(f"    Avg Action Type Accuracy: {dialogue_result['avg_action_type_accuracy']:.2%}")
    print(f"    Avg Domain Accuracy: {dialogue_result['avg_domain_accuracy']:.2%}")

    print(f"\n  Average Slot Tracking Metrics:")
    print(f"    Avg JGA: {dialogue_result['avg_jga']:.2%}")
    print(f"    Avg Slot Accuracy: {dialogue_result['avg_slot_accuracy']:.2%}")
    print(f"    Avg Hallucination Rate: {dialogue_result['avg_hallucination_rate']:.2%}")

    print(f"\n  Memory Transfer:")
    print(f"    Memory Transfer Accuracy: {dialogue_result['memory_transfer_accuracy']:.2%}")
    print(f"    Memory Transfers: {dialogue_result['memory_correct']}/{dialogue_result['memory_total']}")

    print(f"\n  Policy Compliance:")
    print(f"    Policy Violations: {dialogue_result['policy_violations']} ({'FAIL - VIOLATIONS DETECTED' if dialogue_result['policy_violations'] > 0 else 'OK'})")

    if dialogue_result['avg_judge_score'] is not None:
        print(f"\n  LLM Judge:")
        print(f"    Avg Judge Score: {dialogue_result['avg_judge_score']:.2f}/5")


def test_dataset_evaluator() -> None:
    """Test DatasetEvaluator with multiple dialogues."""

    print_separator("TEST DATASET EVALUATOR: MULTIPLE DIALOGUES")

    dataset_eval = DatasetEvaluator()

    print("\nSimulating 3 dialogues with different outcomes\n")

    # DIALOGUE 1: Perfect success
    print("-" * 60)
    print("Dialogue 1: Perfect success (all metrics 1.0)")
    print("-" * 60 + "\n")

    dialogue1 = {
        "task_success": True,
        "task_reason": "All goals achieved",
        "num_turns": 2,
        "avg_intent_accuracy": 1.0,
        "avg_action_type_accuracy": 1.0,
        "avg_domain_accuracy": 1.0,
        "avg_jga": 1.0,
        "avg_slot_accuracy": 1.0,
        "avg_hallucination_rate": 0.0,
        "avg_system_correctness": 0.8,
        "memory_transfer_accuracy": 1.0,
        "memory_correct": 1,
        "memory_total": 1,
        "memory_events": [],
        "policy_violations": 0,
        "avg_judge_score": None
    }

    print("  Dialogue 1 Metrics:")
    print(f"    Task Success: {dialogue1['task_success']}")
    print(f"    Task Reason: {dialogue1['task_reason']}")
    print(f"    Num Turns: {dialogue1['num_turns']}")
    print(f"    Avg Intent Accuracy: {dialogue1['avg_intent_accuracy']:.2%}")
    print(f"    Avg Action Type Accuracy: {dialogue1['avg_action_type_accuracy']:.2%}")
    print(f"    Avg Domain Accuracy: {dialogue1['avg_domain_accuracy']:.2%}")
    print(f"    Avg JGA: {dialogue1['avg_jga']:.2%}")
    print(f"    Avg Slot Accuracy: {dialogue1['avg_slot_accuracy']:.2%}")
    print(f"    Avg Hallucination Rate: {dialogue1['avg_hallucination_rate']:.2%}")
    print(f"    Avg System Correctness: {dialogue1['avg_system_correctness']:.2%}")
    print(f"    Memory Transfer Accuracy: {dialogue1['memory_transfer_accuracy']:.2%} ({dialogue1['memory_correct']}/{dialogue1['memory_total']})")
    print(f"    Policy Violations: {dialogue1['policy_violations']}")

    dataset_eval.add_dialogue(dialogue1)

    # DIALOGUE 2: Partial success
    print("\n" + "-" * 60)
    print("Dialogue 2: Partial success (some errors)")
    print("-" * 60 + "\n")

    dialogue2 = {
        "task_success": False,
        "task_reason": "Policy violations detected (1 violations)",
        "num_turns": 3,
        "avg_intent_accuracy": 0.67,
        "avg_action_type_accuracy": 1.0,
        "avg_domain_accuracy": 1.0,
        "avg_jga": 0.67,
        "avg_slot_accuracy": 0.80,
        "avg_hallucination_rate": 0.10,
        "avg_system_correctness": 0.5,
        "memory_transfer_accuracy": 0.5,
        "memory_correct": 1,
        "memory_total": 2,
        "memory_events": [],
        "policy_violations": 1,
        "avg_judge_score": None
    }

    print("  Dialogue 2 Metrics:")
    print(f"    Task Success: {dialogue2['task_success']}")
    print(f"    Task Reason: {dialogue2['task_reason']}")
    print(f"    Num Turns: {dialogue2['num_turns']}")
    print(f"    Avg Intent Accuracy: {dialogue2['avg_intent_accuracy']:.2%}")
    print(f"    Avg Action Type Accuracy: {dialogue2['avg_action_type_accuracy']:.2%}")
    print(f"    Avg Domain Accuracy: {dialogue2['avg_domain_accuracy']:.2%}")
    print(f"    Avg JGA: {dialogue2['avg_jga']:.2%}")
    print(f"    Avg Slot Accuracy: {dialogue2['avg_slot_accuracy']:.2%}")
    print(f"    Avg Hallucination Rate: {dialogue2['avg_hallucination_rate']:.2%}")
    print(f"    Avg System Correctness: {dialogue2['avg_system_correctness']:.2%}")
    print(f"    Memory Transfer Accuracy: {dialogue2['memory_transfer_accuracy']:.2%} ({dialogue2['memory_correct']}/{dialogue2['memory_total']})")
    print(f"    Policy Violations: {dialogue2['policy_violations']}")

    dataset_eval.add_dialogue(dialogue2)

    # DIALOGUE 3: Failure
    print("\n" + "-" * 60)
    print("Dialogue 3: Failure (task not completed)")
    print("-" * 60 + "\n")

    dialogue3 = {
        "task_success": False,
        "task_reason": "Policy violations detected (2 violations)",
        "num_turns": 2,
        "avg_intent_accuracy": 0.50,
        "avg_action_type_accuracy": 0.50,
        "avg_domain_accuracy": 1.0,
        "avg_jga": 0.50,
        "avg_slot_accuracy": 0.60,
        "avg_hallucination_rate": 0.20,
        "avg_system_correctness": 0.3,
        "memory_transfer_accuracy": 0.0,
        "memory_correct": 0,
        "memory_total": 0,
        "memory_events": [],
        "policy_violations": 2,
        "avg_judge_score": None
    }

    print("  Dialogue 3 Metrics:")
    print(f"    Task Success: {dialogue3['task_success']}")
    print(f"    Task Reason: {dialogue3['task_reason']}")
    print(f"    Num Turns: {dialogue3['num_turns']}")
    print(f"    Avg Intent Accuracy: {dialogue3['avg_intent_accuracy']:.2%}")
    print(f"    Avg Action Type Accuracy: {dialogue3['avg_action_type_accuracy']:.2%}")
    print(f"    Avg Domain Accuracy: {dialogue3['avg_domain_accuracy']:.2%}")
    print(f"    Avg JGA: {dialogue3['avg_jga']:.2%}")
    print(f"    Avg Slot Accuracy: {dialogue3['avg_slot_accuracy']:.2%}")
    print(f"    Avg Hallucination Rate: {dialogue3['avg_hallucination_rate']:.2%}")
    print(f"    Avg System Correctness: {dialogue3['avg_system_correctness']:.2%}")
    print(f"    Memory Transfer Accuracy: {dialogue3['memory_transfer_accuracy']:.2%} ({dialogue3['memory_correct']}/{dialogue3['memory_total']})")
    print(f"    Policy Violations: {dialogue3['policy_violations']}")

    dataset_eval.add_dialogue(dialogue3)

    # DATASET-LEVEL AGGREGATION
    dataset_metrics = dataset_eval.compute_dataset_metrics()

    print("\n" + "-" * 60)
    print("DATASET-LEVEL METRICS (MACRO)")
    print("-" * 60 + "\n")

    print(f"  Dataset Summary:")
    print(f"    Number of Dialogues: {dataset_metrics['num_dialogues']}")
    print(
        f"    Task Success Rate: {dataset_metrics['task_success_rate']:.2%} ({sum(d['task_success'] for d in dataset_eval.dialogue_results)}/{dataset_metrics['num_dialogues']} dialogues)")

    print(f"\n  Average Intent & Routing Metrics:")
    print(f"    Avg Intent Accuracy: {dataset_metrics['avg_intent_accuracy']:.2%}")
    print(f"    Avg Action Type Accuracy: {dataset_metrics['avg_action_type_accuracy']:.2%}")
    print(f"    Avg Domain Accuracy: {dataset_metrics['avg_domain_accuracy']:.2%}")

    print(f"\n  Average Slot Tracking Metrics:")
    print(f"    Avg JGA: {dataset_metrics['avg_jga']:.2%}")
    print(f"    Avg Slot Accuracy: {dataset_metrics['avg_slot_accuracy']:.2%}")
    print(f"    Avg Hallucination Rate: {dataset_metrics['avg_hallucination_rate']:.2%}")

    print(f"\n  Memory Transfer:")
    print(f"    Avg Memory Transfer Accuracy: {dataset_metrics['avg_memory_transfer_accuracy']:.2%}")

    print(f"\n  Policy Compliance:")
    print(f"    Policy Violation Rate: {dataset_metrics['policy_violation_rate']:.2%}")
    print(f"    Total Policy Violations: {dataset_metrics['total_policy_violations']}")

    if dataset_metrics.get('avg_judge_score') is not None:
        print(f"\n  LLM Judge:")
        print(f"    Avg Judge Score: {dataset_metrics['avg_judge_score']:.2f}/5")


def test_evaluator_reset() -> None:
    """Test that evaluator reset works correctly."""

    print_separator("TEST EVALUATOR RESET FUNCTIONALITY")

    policy_requirements = BOOKING_REQUIRED_SLOTS
    evaluator = DialogueEvaluator(policy_requirements)

    # DIALOGUE 1: Evaluate 2 turns
    print("\n" + "-" * 60)
    print("Dialogue 1: Evaluate 2 turns")
    print("-" * 60 + "\n")

    print("  Turn 1:")
    evaluator.evaluate_turn(
        turn_id=1,
        predicted_slots={"restaurant": {"area": "centre"}},
        ground_truth_slots={"restaurant": {"area": "centre"}},
        predicted_intent="find_restaurant",
        ground_truth_intent="find_restaurant",
        predicted_act_type=["Restaurant-Inform"],
        ground_truth_act_type=["Restaurant-Inform"],
        predicted_domain="restaurant",
        action_taken="search_restaurant"
    )
    print(f"    Turns accumulated: {len(evaluator.turn_metrics)}")
    print(f"    Dialogue history length: {len(evaluator.dialogue_history)}")

    print("\n  Turn 2:")
    evaluator.evaluate_turn(
        turn_id=2,
        predicted_slots={"restaurant": {"area": "centre", "food": "italian"}},
        ground_truth_slots={"restaurant": {"area": "centre", "food": "italian"}},
        predicted_intent="find_restaurant",
        ground_truth_intent="find_restaurant",
        predicted_act_type=["Restaurant-Inform"],
        ground_truth_act_type=["Restaurant-Inform"],
        predicted_domain="restaurant",
        action_taken="search_restaurant"
    )
    print(f"    Turns accumulated: {len(evaluator.turn_metrics)}")
    print(f"    Dialogue history length: {len(evaluator.dialogue_history)}")

    print(f"\n  Dialogue 1 Complete:")
    print(f"    Total turns: {len(evaluator.turn_metrics)}")
    print(f"    Total history entries: {len(evaluator.dialogue_history)}")

    # RESET
    print("\n" + "-" * 60)
    print("Resetting evaluator...")
    print("-" * 60 + "\n")

    evaluator.reset()

    print(f"  After Reset:")
    print(f"    Turns: {len(evaluator.turn_metrics)} (should be 0)")
    print(f"    Dialogue history: {len(evaluator.dialogue_history)} (should be 0)")
    print(f"    Reset successful: {'OK' if len(evaluator.turn_metrics) == 0 and len(evaluator.dialogue_history) == 0 else 'FAIL'}")

    # DIALOGUE 2: Evaluate 1 turn after reset
    print("\n" + "-" * 60)
    print("Dialogue 2: Evaluate 1 turn after reset")
    print("-" * 60 + "\n")

    print("  Turn 1:")
    evaluator.evaluate_turn(
        turn_id=1,
        predicted_slots={"hotel": {"area": "north"}},
        ground_truth_slots={"hotel": {"area": "north"}},
        predicted_intent="find_hotel",
        ground_truth_intent="find_hotel",
        predicted_act_type=["Hotel-Inform"],
        ground_truth_act_type=["Hotel-Inform"],
        predicted_domain="hotel",
        action_taken="search_hotel"
    )
    print(f"    Turns accumulated: {len(evaluator.turn_metrics)}")
    print(f"    Dialogue history length: {len(evaluator.dialogue_history)}")

    print(f"\n  Dialogue 2 State:")
    print(f"    Total turns: {len(evaluator.turn_metrics)} (should be 1)")
    print(f"    Total history entries: {len(evaluator.dialogue_history)} (should be 1)")

    # Verification
    print("\n" + "-" * 60)
    print("RESET VERIFICATION")
    print("-" * 60 + "\n")

    reset_working = (len(evaluator.turn_metrics) == 1 and len(evaluator.dialogue_history) == 1)

    if reset_working:
        print("  Status: OK")
        print("    Dialogue 1 had 2 turns")
        print("    Reset cleared all state")
        print("    Dialogue 2 started fresh with 1 turn")
        print("\n  Reset functionality working correctly!")
    else:
        print("  Status: FAIL")
        print(f"    Expected 1 turn, got {len(evaluator.turn_metrics)}")
        print(f"    Expected 1 history entry, got {len(evaluator.dialogue_history)}")
        print("\n  Reset functionality NOT working!")


def run_all_tests() -> None:
    """Run all DialogueEvaluator and DatasetEvaluator tests in sequence."""
    print_separator("TEST EVALUATOR")

    test_dialogue_evaluator_single_turn()
    test_dialogue_evaluator_multi_turn()
    test_dialogue_evaluator_memory_transfer()
    test_dialogue_evaluator_policy_violation()
    test_dataset_evaluator()
    test_evaluator_reset()

    print_separator("ALL EVALUATOR TESTS COMPLETED")


if __name__ == "__main__":

    capture_and_save(func=run_all_tests,
                     output_path="docs/evals_inspection/evaluator_test_results.txt"
    )

