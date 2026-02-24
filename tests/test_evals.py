"""
Comprehensive tests for all evaluation methods.

Tests all objective metrics (JGA, slot accuracy, hallucination rate, policy compliance,
task success, intent accuracy, action-type accuracy, domain accuracy, memory transfer)
and subjective metrics (LLM-as-judge prompt formatting and response parsing).
"""


from src.evaluation import (
    calculate_intent_accuracy, calculate_action_type_accuracy, calculate_jga, calculate_slot_accuracy, calculate_hallucination_rate, calculate_policy_compliance,
    calculate_task_success, calculate_system_correctness, calculate_domain_accuracy, calculate_memory_transfer_accuracy, create_judge_prompt, parse_judge_response
)
from src.utils import print_separator, capture_and_save


def test_domain_accuracy() -> None:
    """Test domain routing accuracy calculation."""

    # Test 1: Correct routing
    print_separator("TEST DOMAIN ACCURACY (1): CORRECT ROUTING")
    predicted = "restaurant"
    ground_truth_intent = "find_restaurant"
    print(f"\nPredicted domain: {predicted}")
    print(f"Ground truth intent: {ground_truth_intent}\n")

    acc, is_correct, gt_domain = calculate_domain_accuracy(predicted, ground_truth_intent)
    print(f"Accuracy: {acc}")  # Should be 1.0
    print(f"Is Correct: {is_correct}")  # Should be True
    print(f"Extracted ground truth domain: {gt_domain}")  # Should be restaurant

    # Test 2: Wrong domain
    print_separator("TEST DOMAIN ACCURACY (2): WRONG DOMAIN")
    predicted = "hotel"
    ground_truth_intent = "find_restaurant"
    print(f"\nPredicted domain: {predicted}")
    print(f"Ground truth intent: {ground_truth_intent}\n")

    acc, is_correct, gt_domain = calculate_domain_accuracy(predicted, ground_truth_intent)
    print(f"Accuracy: {acc}")  # Should be 0.0
    print(f"Is Correct: {is_correct}")  # Should be False
    print(f"Extracted ground truth domain: {gt_domain}")  # Should be restaurant

    # Test 3: Case insensitive match
    print_separator("TEST DOMAIN ACCURACY (3): CASE INSENSITIVE")
    predicted = "Restaurant"  # Capital R
    ground_truth_intent = "book_restaurant"
    print(f"\nPredicted domain: {predicted}")
    print(f"Ground truth intent: {ground_truth_intent}\n")

    acc, is_correct, gt_domain = calculate_domain_accuracy(predicted, ground_truth_intent)
    print(f"Accuracy: {acc}")  # Should be 1.0
    print(f"Is Correct: {is_correct}")  # Should be True
    print(f"Extracted ground truth domain: {gt_domain}")  # Should be restaurant

    # Test 4: Different intent, same domain
    print_separator("TEST DOMAIN ACCURACY (4): DIFFERENT INTENT, SAME DOMAIN")
    predicted = "Hotel"
    ground_truth_intent = "book_hotel"  # Different action than find_hotel
    print(f"\nPredicted domain: {predicted}")
    print(f"Ground truth intent: {ground_truth_intent}\n")

    acc, is_correct, gt_domain = calculate_domain_accuracy(predicted, ground_truth_intent)
    print(f"Accuracy: {acc}")  # Should be 1.0 (domain is correct)
    print(f"Is Correct: {is_correct}")  # Should be True
    print(f"Extracted ground truth domain: {gt_domain}")  # Should be hotel


def test_intent_accuracy() -> None:
    """Test intent accuracy calculation."""

    # Test 1: Perfect match
    print_separator("TEST INTENT ACCURACY (1): PERFECT MATCH")
    predicted = "find_restaurant"
    ground_truth = "find_restaurant"
    print(f"\nPredicted: {predicted}")
    print(f"Ground truth: {ground_truth}\n")

    acc, is_correct = calculate_intent_accuracy(predicted, ground_truth)
    print(f"Accuracy: {acc}")  # Should be 1.0
    print(f"Is Correct: {is_correct}")  # Should be True

    # Test 2: Wrong intent
    print_separator("TEST INTENT ACCURACY (2): WRONG INTENT")
    predicted = "find_hotel"
    ground_truth = "find_restaurant"
    print(f"\nPredicted: {predicted}")
    print(f"Ground truth: {ground_truth}\n")

    acc, is_correct = calculate_intent_accuracy(predicted, ground_truth)
    print(f"Accuracy: {acc}")  # Should be 0.0
    print(f"Is Correct: {is_correct}")  # Should be False

    # Test 3: Close but not exact
    print_separator("TEST INTENT ACCURACY (3): SIMILAR BUT WRONG")
    predicted = "book_restaurant"
    ground_truth = "find_restaurant"
    print(f"\nPredicted: {predicted}")
    print(f"Ground truth: {ground_truth}\n")

    acc, is_correct = calculate_intent_accuracy(predicted, ground_truth)
    print(f"Accuracy: {acc}")  # Should be 0.0
    print(f"Is Correct: {is_correct}")  # Should be False

    # Test 4: Multiple wrong intent (rare in multiwoz, but happens in real life)
    print_separator("TEST ACCURACY FOR MULTIPLE INTENTS (4): WRONG INTENT")
    predicted = "find_hotel", "find_restaurant"
    ground_truth = "find_restaurant", "book_restaurant"
    print(f"\nPredicted: {predicted}")
    print(f"Ground truth: {ground_truth}\n")
    acc, is_correct, recall, precision, correct, predicted_total, gt_total = calculate_intent_accuracy(predicted, ground_truth, return_detailed=True)

    print(f"Accuracy: {acc}")  # Should be 0.0
    print(f"Is Correct: {is_correct}")  # Should be False
    print(f"\nRecall: {recall:.2f}, Precision: {precision:.2f} ({correct}/{gt_total}, {correct}/{predicted_total})")


def test_action_type_accuracy() -> None:
    """Test action-type accuracy calculation with detailed metrics."""

    # Test 1: Perfect match (single act)
    print_separator("TEST ACTION-TYPE ACCURACY (1): SINGLE ACT MATCH")
    predicted = ["Restaurant-Inform"]
    ground_truth = ["Restaurant-Inform"]
    print(f"\nPredicted: {predicted}")
    print(f"Ground truth: {ground_truth}\n")

    acc, is_correct, recall, precision, correct, p_total, gt_total = calculate_action_type_accuracy(predicted, ground_truth, return_detailed=True)
    print(f"Accuracy: {acc}")  # Should be 1.0
    print(f"Is Correct: {is_correct}")  # Should be True
    print(f"Recall: {recall:.2f}, Precision: {precision:.2f} ({correct}/{gt_total}, {correct}/{p_total})")

    # Test 2: Wrong act type
    print_separator("TEST ACTION-TYPE ACCURACY (2): WRONG ACT")
    predicted = ["Restaurant-Request"]
    ground_truth = ["Restaurant-Inform"]
    print(f"\nPredicted: {predicted}")
    print(f"Ground truth: {ground_truth}\n")

    acc, is_correct, recall, precision, correct, p_total, gt_total = calculate_action_type_accuracy(predicted, ground_truth, return_detailed=True)
    print(f"Accuracy: {acc}")  # Should be 0.0
    print(f"Is Correct: {is_correct}")  # Should be False
    print(f"Recall: {recall:.2f}, Precision: {precision:.2f} ({correct}/{gt_total}, {correct}/{p_total})")

    # Test 3: Multi-act match (order doesn't matter)
    print_separator("TEST ACTION-TYPE ACCURACY (3): MULTI-ACT MATCH")
    predicted = ["Hotel-Inform", "Hotel-Request"]
    ground_truth = ["Hotel-Request", "Hotel-Inform"]  # Different order
    print(f"\nPredicted: {predicted}")
    print(f"Ground truth: {ground_truth}\n")

    acc, is_correct, recall, precision, correct, p_total, gt_total = calculate_action_type_accuracy(predicted, ground_truth, return_detailed=True)
    print(f"Accuracy: {acc}")  # Should be 1.0
    print(f"Is Correct: {is_correct}")  # Should be True
    print(f"Recall: {recall:.2f}, Precision: {precision:.2f} ({correct}/{gt_total}, {correct}/{p_total})")

    # Test 4: Multi-act mismatch
    print_separator("TEST ACTION-TYPE ACCURACY (4): MULTI-ACT MISMATCH")
    predicted = ["Hotel-Inform", "Hotel-Request"]
    ground_truth = ["Hotel-Request", "Restaurant-Inform"]
    print(f"\nPredicted: {predicted}")  # Should be 0.0
    print(f"Ground truth: {ground_truth}\n")  # Should be False

    acc, is_correct, recall, precision, correct, p_total, gt_total = calculate_action_type_accuracy(predicted, ground_truth, return_detailed=True)
    print(f"Accuracy: {acc}")
    print(f"Is Correct: {is_correct}")
    print(f"Recall: {recall:.2f}, Precision: {precision:.2f} ({correct}/{gt_total}, {correct}/{p_total})")
    # Expected: Recall=0.50 (1/2), Precision=0.50 (1/2) - got Hotel-Request right, both wrong on second act

    # Test 5: Missing act
    print_separator("TEST ACTION-TYPE ACCURACY (5): MISSING ACT")
    predicted = ["Restaurant-Inform"]
    ground_truth = ["Restaurant-Inform", "Restaurant-Request"]
    print(f"\nPredicted: {predicted}")
    print(f"Ground truth: {ground_truth}\n")

    acc, is_correct, recall, precision, correct, p_total, gt_total = calculate_action_type_accuracy(predicted, ground_truth, return_detailed=True)
    print(f"Accuracy: {acc}")  # Should be 0.0
    print(f"Is Correct: {is_correct}")  # Should be False
    print(f"Recall: {recall:.2f}, Precision: {precision:.2f} ({correct}/{gt_total}, {correct}/{p_total})")
    # Expected: Recall=0.50 (1/2), Precision=1.00 (1/1) - incomplete but no hallucinations

    # Test 6: Extra act (hallucinated)
    print_separator("TEST ACTION-TYPE ACCURACY (6): EXTRA ACT")
    predicted = ["Hotel-Inform", "Hotel-Request", "Hotel-Book"]
    ground_truth = ["Hotel-Inform", "Hotel-Request"]
    print(f"\nPredicted: {predicted}")
    print(f"Ground truth: {ground_truth}\n")

    acc, is_correct, recall, precision, correct, p_total, gt_total = calculate_action_type_accuracy(predicted, ground_truth, return_detailed=True)
    print(f"Accuracy: {acc}")  # Should be 0.0
    print(f"Is Correct: {is_correct}")  # Should be False
    print(f"Recall: {recall:.2f}, Precision: {precision:.2f} ({correct}/{gt_total}, {correct}/{p_total})")
    # Expected: Recall=1.00 (2/2), Precision=0.67 (2/3) - complete but with hallucination


def test_slot_accuracy() -> None:
    """Test slot-level accuracy calculation."""

    # Test 1: Perfect match (same as JGA test)
    print_separator("TEST SLOT ACCURACY (1): PERFECT MATCH")
    predicted = {"restaurant": {"area": "centre", "pricerange": "expensive"}}
    ground_truth = {"restaurant": {"area": "centre", "pricerange": "expensive"}}
    print(f"\nPredicted: {predicted}")
    print(f"Ground truth: {ground_truth}\n")

    acc, correct, total = calculate_slot_accuracy(predicted, ground_truth)
    print(f"Slot Accuracy: {acc:.2f} ({correct}/{total})")  # Should be 1.00 (2/2)

    # Test 2: Partial match (1 out of 2 correct)
    print_separator("TEST SLOT ACCURACY (2): PARTIAL MATCH")
    predicted = {"restaurant": {"area": "centre", "pricerange": "cheap"}}
    ground_truth = {"restaurant": {"area": "centre", "pricerange": "expensive"}}
    print(f"\nPredicted: {predicted}")
    print(f"Ground truth: {ground_truth}\n")

    acc, correct, total = calculate_slot_accuracy(predicted, ground_truth)
    print(f"Slot Accuracy: {acc:.2f} ({correct}/{total})")  # Should be 0.50 (1/2)

    # Test 3: Missing domain
    print_separator("TEST SLOT ACCURACY (3): MISSING DOMAIN")
    predicted = {"restaurant": {"area": "centre"}}
    ground_truth = {
        "restaurant": {"area": "centre"},
        "hotel": {"pricerange": "cheap"}
    }
    print(f"\nPredicted: {predicted}")
    print(f"Ground truth: {ground_truth}\n")

    acc, correct, total = calculate_slot_accuracy(predicted, ground_truth)
    print(f"Slot Accuracy: {acc:.2f} ({correct}/{total})")  # Should be 0.50 (1/2)

    # Test 4: Extra slots in prediction (should not penalize)
    print_separator("TEST SLOT ACCURACY (4): EXTRA PREDICTION SLOTS")
    predicted = {"restaurant": {"area": "centre", "pricerange": "cheap", "food": "italian"}}
    ground_truth = {"restaurant": {"area": "centre", "pricerange": "cheap"}}
    print(f"\nPredicted: {predicted}")
    print(f"Ground truth: {ground_truth}\n")

    acc, correct, total = calculate_slot_accuracy(predicted, ground_truth)
    print(f"Slot Accuracy: {acc:.2f} ({correct}/{total})")  # Should be 1.00 (2/2)


def test_jga() -> None:
    """Test jga calculation for simple cases."""

    predicted = {
        "hotel": {"area": "centre", "price": "cheap"},
        "restaurant": {"food": "indian"}
    }
    ground_truth = {
        "hotel": {"area": "centre", "price": "cheap"},
        "restaurant": {"food": "indian"}
    }

    # Test 1: Perfect match
    print_separator("TEST JGA (1): PERFECT MATCH")
    print(f"\nPredicted: {predicted}")
    print(f"Ground truth: {ground_truth}")
    jga, breakdown = calculate_jga(predicted, ground_truth)
    print(f"\nJGA: {jga}")  # Should print: JGA: 1.0
    print(f"Breakdown: {breakdown}")  # All domains True

    # Test 2: Partial match (hotel wrong)
    print_separator("TEST JGA (2): PARTIAL MATCH")
    predicted["hotel"]["area"] = "north"  # Changed value
    print(f"\nPredicted: {predicted}")
    print(f"Ground truth: {ground_truth}")

    jga, breakdown = calculate_jga(predicted, ground_truth)
    print(f"\nJGA: {jga}")  # Should print: JGA: 0.0
    print(f"Breakdown: {breakdown}")  # hotel: False, restaurant: True

    # Test 3: Missing domain
    print_separator("TEST JGA (3): MISSING DOMAIN")
    predicted_missing = {"hotel": {"area": "centre"}}
    truth_extra = {"hotel": {"area": "centre"}, "taxi": {"destination": "hotel"}}
    print(f"\nPredicted: {predicted_missing}")
    print(f"Ground truth: {truth_extra}")

    jga, breakdown = calculate_jga(predicted_missing, truth_extra)
    print(f"\nJGA: {jga}")  # Should print: JGA: 0.0 (taxi domain missing)
    print(f"Breakdown: {breakdown}")

    # Test 4: Extra domain in prediction (hallucinated domain)
    print_separator("TEST JGA (4): EXTRA PREDICTED DOMAIN")
    predicted_extra = {"hotel": {"area": "centre"}, "taxi": {"destination": "hotel"}}
    truth_simple = {"hotel": {"area": "centre"}}
    print(f"\nPredicted: {predicted_extra}")
    print(f"Ground truth: {truth_simple}")

    jga, breakdown = calculate_jga(predicted_extra, truth_simple)
    print(f"\nJGA: {jga}")  # Should be 0.0 - extra domain = mismatch
    print(f"Breakdown: {breakdown}")  # taxi: False (not in GT)


def test_hallucination_rate() -> None:
    """Test hallucination rate calculation."""

    # Test 1: No hallucinations
    print_separator("TEST HALLUCINATION (1): NO HALLUCINATIONS")
    predicted = {"restaurant": {"area": "centre", "pricerange": "cheap"}}
    ground_truth = {"restaurant": {"area": "centre", "pricerange": "cheap"}}
    print(f"\nPredicted: {predicted}")
    print(f"Ground truth: {ground_truth}\n")

    hall_rate, hall_count, predicted_count = calculate_hallucination_rate(predicted, ground_truth)
    print(f"Hallucination Rate: {hall_rate:.2f} ({hall_count}/{predicted_count})")  # Should be 0.00 (0/2)

    # Test 2: Hallucinated domain
    print_separator("TEST HALLUCINATION (2): HALLUCINATED DOMAIN (1)")
    predicted = {"hotel": {"pricerange": "cheap"}}
    ground_truth = {"restaurant": {"area": "centre"}}
    print(f"\nPredicted: {predicted}")
    print(f"Ground truth: {ground_truth}\n")

    hall_rate, hall_count, predicted_count = calculate_hallucination_rate(predicted, ground_truth)
    print(f"Hallucination Rate: {hall_rate:.2f} ({hall_count}/{predicted_count})")  # Should be 1.00 (1/1)

    # Test 3: Hallucinated domain (extra slot prediction)
    print_separator("TEST HALLUCINATION (3): HALLUCINATED DOMAIN (2)")
    predicted = {"restaurant": {"area": "centre"}, "hotel": {"pricerange": "cheap"}}  # User never mentioned hotel
    ground_truth = {"restaurant": {"area": "centre"}}
    print(f"\nPredicted: {predicted}")
    print(f"Ground truth: {ground_truth}\n")

    hall_rate, hall_count, predicted_count = calculate_hallucination_rate(predicted, ground_truth)
    print(f"Hallucination Rate: {hall_rate:.2f} ({hall_count}/{predicted_count})")  # Should be 0.50 (1/2)

    # Test 4: Partial hallucinated (wrong slot name)
    print_separator("TEST HALLUCINATION (4): HALLUCINATED SLOT NAME")
    predicted = {"restaurant": {"pricerange": "expensive"}}  # User never mentioned hotel
    ground_truth = {"restaurant": {"area": "centre"}}
    print(f"\nPredicted: {predicted}")
    print(f"Ground truth: {ground_truth}\n")

    hall_rate, hall_count, predicted_count = calculate_hallucination_rate(predicted, ground_truth)
    print(f"Hallucination Rate: {hall_rate:.2f} ({hall_count}/{predicted_count})")  # Should be 1.00 (1/1)

    # Test 5: Partial hallucination (wrong slot value)
    print_separator("TEST HALLUCINATION (5): WRONG SLOT VALUE")
    predicted = {"restaurant": {"area": "centre", "pricerange": "expensive"}}
    ground_truth = {"restaurant": {"area": "centre", "pricerange": "cheap"}}
    print(f"\nPredicted: {predicted}")
    print(f"Ground truth: {ground_truth}\n")

    hall_rate, hall_count, predicted_count = calculate_hallucination_rate(predicted, ground_truth)
    print(f"Hallucination Rate: {hall_rate:.2f} ({hall_count}/{predicted_count})")  # Should be 0.50 (1/2)

    # Test 6: All hallucinations
    print_separator("TEST HALLUCINATION (6): COMPLETE HALLUCINATION")
    predicted = {"taxi": {"destination": "airport", "departure": "hotel"}}
    ground_truth = {"restaurant": {"area": "centre"}}
    print(f"\nPredicted: {predicted}")
    print(f"Ground truth: {ground_truth}\n")

    hall_rate, hall_count, predicted_count = calculate_hallucination_rate(predicted, ground_truth)
    print(f"Hallucination Rate: {hall_rate:.2f} ({hall_count}/{predicted_count})")  # Should be 1.00 (2/2)


def test_memory_transfer_accuracy() -> None:
    """Test cross-domain memory transfer accuracy calculation."""

    # Test 1: Successful transfer (Scenario A success case)
    print_separator("TEST MEMORY TRANSFER (1): SUCCESSFUL TRANSFER")
    dialogue_history = [
        {
            "turn_id": 1,
            "domain": "restaurant",
            "predicted_slots": {
                "restaurant": {"area": "centre", "food": "indian"}
            }
        },
        {
            "turn_id": 2,
            "domain": "hotel",
            "predicted_slots": {
                "restaurant": {"area": "centre", "food": "indian"},
                "hotel": {"area": "centre", "pricerange": "cheap"}  # area transferred
            }
        }
    ]

    acc, correct, total, events = calculate_memory_transfer_accuracy(dialogue_history)
    print(f"\nDialogue history:")
    for turn in dialogue_history:
        print(turn)
    print(f"\nTransfer Accuracy: {acc:.2f} ({correct}/{total})")  # Should be 1.00 (1/1)
    print(f"Transfer Events: {events}")

    # Test 2: Failed transfer (Scenario A failure case)
    print_separator("TEST MEMORY TRANSFER (2): FAILED TRANSFER")
    dialogue_history = [
        {
            "turn_id": 1,
            "domain": "restaurant",
            "predicted_slots": {
                "restaurant": {"area": "centre", "pricerange": "expensive"}
            }
        },
        {
            "turn_id": 2,
            "domain": "hotel",
            "predicted_slots": {
                "restaurant": {"area": "centre", "pricerange": "expensive"},
                "hotel": {"pricerange": "cheap"}  # area NOT transferred (memory failure)
            }
        }
    ]

    acc, correct, total, events = calculate_memory_transfer_accuracy(dialogue_history)
    print(f"\nDialogue history:")
    for turn in dialogue_history:
        print(turn)
    print(f"\nTransfer Accuracy: {acc:.2f} ({correct}/{total})")  # Should be 0.00
    print(f"Transfer Events:")
    for event in events:
        print(f"  - {event}")

    # Test 3: Partial transfer (one slot yes, one slot no)
    print_separator("TEST MEMORY TRANSFER (3): PARTIAL TRANSFER")
    dialogue_history = [
        {
            "turn_id": 1,
            "domain": "restaurant",
            "predicted_slots": {
                "restaurant": {"area": "north", "pricerange": "moderate"}
            }
        },
        {
            "turn_id": 2,
            "domain": "hotel",
            "predicted_slots": {
                "restaurant": {"area": "north", "pricerange": "moderate"},
                "hotel": {"area": "north"}  # area yes, pricerange no
            }
        }
    ]

    acc, correct, total, events = calculate_memory_transfer_accuracy(dialogue_history)
    print(f"\nDialogue history:")
    for turn in dialogue_history:
        print(turn)
    print(f"\nTransfer Accuracy: {acc:.2f} ({correct}/{total})")  # Should be 0.50 (1/2)
    print(f"Transfer Events:")
    for event in events:
        print(f"  - Slot '{event['slot']}': {event['value']} - Transferred: {event['transferred']}")

    # Test 4: Multiple domain switches
    print_separator("TEST MEMORY TRANSFER (4): MULTIPLE SWITCHES")
    dialogue_history = [
        {
            "turn_id": 1,
            "domain": "restaurant",
            "predicted_slots": {
                "restaurant": {"area": "centre"}
            }
        },
        {
            "turn_id": 2,
            "domain": "hotel",
            "predicted_slots": {
                "restaurant": {"area": "centre"},
                "hotel": {"area": "centre"}  # First transfer: success
            }
        },
        {
            "turn_id": 3,
            "domain": "taxi",
            "predicted_slots": {
                "restaurant": {"area": "centre"},
                "hotel": {"area": "centre"},
                "taxi": {"departure": "hotel", "destination": "restaurant"}  # area not applicable to taxi
            }
        }
    ]

    acc, correct, total, events = calculate_memory_transfer_accuracy(dialogue_history)
    print(f"\nDialogue history:")
    for turn in dialogue_history:
        print(turn)
    print(f"\nTransfer Accuracy: {acc:.2f} ({correct}/{total})")
    print(f"Number of transfer events: {len(events)}")

    # Test 5: No domain switches
    print_separator("TEST MEMORY TRANSFER (5): NO DOMAIN SWITCHES")
    dialogue_history = [
        {
            "turn_id": 1,
            "domain": "restaurant",
            "predicted_slots": {
                "restaurant": {"area": "centre"}
            }
        },
        {
            "turn_id": 2,
            "domain": "restaurant",
            "predicted_slots": {
                "restaurant": {"area": "centre", "food": "italian"}
            }
        }
    ]

    acc, correct, total, events = calculate_memory_transfer_accuracy(dialogue_history)
    print(f"\nDialogue history:")
    for turn in dialogue_history:
        print(turn)
    print(f"\nTransfer Accuracy: {acc:.2f} ({correct}/{total})")  # Should be 0.00 (0/0)
    print(f"Number of transfer events: {len(events)}")  # Should be 0


def test_policy_compliance() -> None:
    """Test policy compliance calculation."""

    from src.data import BOOKING_REQUIRED_SLOTS
    policy_requirements = BOOKING_REQUIRED_SLOTS

    # Test 1: Compliant booking (all slots present)
    print_separator("TEST POLICY (1): COMPLIANT BOOKING")
    current_slots = {
        "hotel": {
            "name": "acorn guest house",
            "bookday": "monday",
            "bookpeople": "2",
            "bookstay": "3"
        }
    }
    action = "book_hotel"
    print(f"\nAction: {action}")
    print(f"Current slots: {current_slots}\n")

    is_compliant, reason = calculate_policy_compliance(action, policy_requirements, current_slots)
    print(f"Compliant: {is_compliant}")  # Should be True
    print(f"Reason: {reason}")

    # Test 2: Policy violation (missing bookstay)
    print_separator("TEST POLICY (2): MISSING REQUIRED SLOT")
    current_slots = {
        "hotel": {
            "name": "acorn guest house",
            "bookday": "monday",
            "bookpeople": "2"
            # Missing "bookstay"
        }
    }
    action = "book_hotel"
    print(f"\nAction: {action}")
    print(f"Current slots: {current_slots}\n")

    is_compliant, reason = calculate_policy_compliance(action, policy_requirements, current_slots)
    print(f"Compliant: {is_compliant}")  # Should be False
    print(f"Reason: {reason}")

    # Test 3: Compliant - action not in policy (system requests slots)
    print_separator("TEST POLICY (3): ACTION NOT IN POLICY")
    current_slots = {
        "hotel": {
            "name": "acorn guest house"
            # Missing bookday, bookpeople, bookstay
        }
    }
    action = "request_slots"
    print(f"\nAction: {action}")
    print(f"Current slots: {current_slots}\n")

    is_compliant, reason = calculate_policy_compliance(action, policy_requirements, current_slots)
    print(f"Compliant: {is_compliant}")  # Should be True
    print(f"Reason: {reason}")

    # Test 4: No policy constraints (search action)
    print_separator("TEST POLICY (4): NO CONSTRAINTS")
    current_slots = {"restaurant": {"food": "indian"}}
    action = "search_restaurant"
    print(f"\nAction: {action}")
    print(f"Current slots: {current_slots}\n")

    is_compliant, reason = calculate_policy_compliance(action, policy_requirements, current_slots)
    print(f"Compliant: {is_compliant}")  # Should be True
    print(f"Reason: {reason}")





def test_system_correctness() -> None:
    """Test system correctness calculation."""

    # Test 1: Correctly handles incomplete booking
    print_separator("TEST SYSTEM CORRECTNESS (1): INCOMPLETE BOOKING - REQUEST")
    predicted_action = "request"
    predicted_intent = "book_hotel"
    predicted_slots = {"hotel": {"pricerange": "expensive", "bookpeople": "2"}}
    hallucination_detected = False
    policy_compliant = True
    current_domain = "hotel"

    print(f"\nPredicted action: {predicted_action}")
    print(f"Predicted intent: {predicted_intent}")
    print(f"Predicted slots: {predicted_slots}")
    print(f"Hallucination detected: {hallucination_detected}")
    print(f"Policy compliant: {policy_compliant}")
    print(f"Current domain: {current_domain}\n")

    correct, reason = calculate_system_correctness(
        predicted_action,
        predicted_intent,
        predicted_slots,
        hallucination_detected,
        policy_compliant,
        current_domain
    )
    print(f"System Correct: {correct}")  # Should be True
    print(f"Reason: {reason}")

    # Test 2: Fails - has all slots but doesn't book
    print_separator("TEST SYSTEM CORRECTNESS (2): COMPLETE SLOTS - WRONG ACTION")
    predicted_action = "search"
    predicted_intent = "book_hotel"
    predicted_slots = {"hotel": {"name": "hilton", "bookday": "monday", "bookpeople": "2", "bookstay": "3"}}
    hallucination_detected = False
    policy_compliant = True
    current_domain = "hotel"

    print(f"\nPredicted action: {predicted_action}")
    print(f"Predicted intent: {predicted_intent}")
    print(f"Predicted slots: {predicted_slots}")
    print(f"Hallucination detected: {hallucination_detected}")
    print(f"Policy compliant: {policy_compliant}")
    print(f"Current domain: {current_domain}\n")

    correct, reason = calculate_system_correctness(
        predicted_action,
        predicted_intent,
        predicted_slots,
        hallucination_detected,
        policy_compliant,
        current_domain
    )
    print(f"System Correct: {correct}")  # Should be False
    print(f"Reason: {reason}")

    # Test 3: Correctly completes booking
    print_separator("TEST SYSTEM CORRECTNESS (3): COMPLETE BOOKING - SUCCESS")
    predicted_action = "book"
    predicted_intent = "book_hotel"
    predicted_slots = {"hotel": {"name": "hilton", "bookday": "monday", "bookpeople": "2", "bookstay": "3"}}
    hallucination_detected = False
    policy_compliant = True
    current_domain = "hotel"

    print(f"\nPredicted action: {predicted_action}")
    print(f"Predicted intent: {predicted_intent}")
    print(f"Predicted slots: {predicted_slots}")
    print(f"Hallucination detected: {hallucination_detected}")
    print(f"Policy compliant: {policy_compliant}")
    print(f"Current domain: {current_domain}\n")

    correct, reason = calculate_system_correctness(
        predicted_action,
        predicted_intent,
        predicted_slots,
        hallucination_detected,
        policy_compliant,
        current_domain
    )
    print(f"System Correct: {correct}")  # Should be True
    print(f"Reason: {reason}")

    # Test 4: Fails - hallucination detected
    print_separator("TEST SYSTEM CORRECTNESS (4): HALLUCINATION DETECTED")
    predicted_action = "inform"
    predicted_intent = "find_restaurant"
    predicted_slots = {"restaurant": {"area": "centre"}}
    hallucination_detected = True
    policy_compliant = True
    current_domain = "restaurant"

    print(f"\nPredicted action: {predicted_action}")
    print(f"Predicted intent: {predicted_intent}")
    print(f"Predicted slots: {predicted_slots}")
    print(f"Hallucination detected: {hallucination_detected}")
    print(f"Policy compliant: {policy_compliant}")
    print(f"Current domain: {current_domain}\n")

    correct, reason = calculate_system_correctness(
        predicted_action,
        predicted_intent,
        predicted_slots,
        hallucination_detected,
        policy_compliant,
        current_domain
    )
    print(f"System Correct: {correct}")  # Should be False
    print(f"Reason: {reason}")

    # Test 5: Correctly handles search intent
    print_separator("TEST SYSTEM CORRECTNESS (5): SEARCH INTENT - SUCCESS")
    predicted_action = "search"
    predicted_intent = "find_restaurant"
    predicted_slots = {"restaurant": {"area": "centre", "pricerange": "expensive"}}
    hallucination_detected = False
    policy_compliant = True
    current_domain = "restaurant"

    print(f"\nPredicted action: {predicted_action}")
    print(f"Predicted intent: {predicted_intent}")
    print(f"Predicted slots: {predicted_slots}")
    print(f"Hallucination detected: {hallucination_detected}")
    print(f"Policy compliant: {policy_compliant}")
    print(f"Current domain: {current_domain}\n")

    correct, reason = calculate_system_correctness(
        predicted_action,
        predicted_intent,
        predicted_slots,
        hallucination_detected,
        policy_compliant,
        current_domain
    )
    print(f"System Correct: {correct}")  # Should be True
    print(f"Reason: {reason}")

    # Test 6: Policy violation - but correctly requests missing slots
    print_separator("TEST SYSTEM CORRECTNESS (6): POLICY VIOLATION - CORRECT REQUEST")
    predicted_action = "request"
    predicted_intent = "book_hotel"
    predicted_slots = {"hotel": {"pricerange": "expensive"}}  # Missing required booking slots
    hallucination_detected = False
    policy_compliant = False  # Policy violated (missing required slots)
    current_domain = "hotel"

    print(f"\nPredicted action: {predicted_action}")
    print(f"Predicted intent: {predicted_intent}")
    print(f"Predicted slots: {predicted_slots}")
    print(f"Hallucination detected: {hallucination_detected}")
    print(f"Policy compliant: {policy_compliant}")
    print(f"Current domain: {current_domain}\n")

    correct, reason = calculate_system_correctness(
        predicted_action,
        predicted_intent,
        predicted_slots,
        hallucination_detected,
        policy_compliant,
        current_domain
    )
    print(f"System Correct: {correct}")  # Should be True (system correctly asks for missing info)
    print(f"Reason: {reason}")


def test_task_success() -> None:
    """Test task success rate calculation."""

    # Test 1: Successful booking task
    print_separator("TEST TASK SUCCESS (1): SUCCESSFUL BOOKING")
    turn_results = [
        {"domain": "hotel", "action": "search", "accumulated_slots": {"hotel": {"area": "centre"}}},
        {"domain": "hotel", "action": "book", "accumulated_slots": {"hotel": {"name": "hilton", "bookday": "monday", "bookpeople": "2", "bookstay": "3"}}}
    ]
    ground_truth_goal = {
        "domains": ["hotel"],
        "requires_booking": True
    }
    print(f"\nTurn results: {len(turn_results)} turns")
    print(f"Final slots: {turn_results[-1]['accumulated_slots']}")
    print(f"Goal: {ground_truth_goal}\n")

    success, reason = calculate_task_success(turn_results, ground_truth_goal)
    print(f"Success: {success}")  # Should be True
    print(f"Reason: {reason}")

    # Test 2: Failed - missing booking slots
    print_separator("TEST TASK SUCCESS (2): MISSING BOOKING SLOTS")
    turn_results = [
        {"domain": "hotel", "action": "search", "accumulated_slots": {"hotel": {"area": "centre"}}},
        {"domain": "hotel", "action": "book", "accumulated_slots": {"hotel": {"bookpeople": "2", "bookstay": "3"}}}  # Missing name and bookday
    ]
    ground_truth_goal = {
        "domains": ["hotel"],
        "requires_booking": True
    }
    print(f"\nTurn results: {len(turn_results)} turns")
    print(f"Final slots: {turn_results[-1]['accumulated_slots']}")
    print(f"Goal: {ground_truth_goal}\n")

    success, reason = calculate_task_success(turn_results, ground_truth_goal)
    print(f"Success: {success}")  # Should be False
    print(f"Reason: {reason}")

    # Test 3: Successful info-only task
    print_separator("TEST TASK SUCCESS (3): INFO-ONLY SUCCESS")
    turn_results = [
        {"domain": "restaurant", "action": "search", "accumulated_slots": {"restaurant": {"area": "centre"}}},
        {"domain": "restaurant", "action": "inform", "accumulated_slots": {"restaurant": {"area": "centre", "food": "italian"}}}
    ]
    ground_truth_goal = {
        "domains": ["restaurant"],
        "requires_booking": False
    }
    print(f"\nTurn results: {len(turn_results)} turns")
    print(f"Final slots: {turn_results[-1]['accumulated_slots']}")
    print(f"Goal: {ground_truth_goal}\n")

    success, reason = calculate_task_success(turn_results, ground_truth_goal)
    print(f"Success: {success}")  # Should be True
    print(f"Reason: {reason}")








def test_judge_prompt_creation() -> None:
    """Test judge prompt creation."""

    print_separator("TEST LLM JUDGE PROMPT FORMATTING")

    user_message = "I want a cheap hotel in the centre"
    system_response = "I found the Acorn Guest House in the centre with cheap prices. Would you like to book it?"
    ground_truth_slots = {
        "hotel": {"area": "centre", "pricerange": "cheap", "name": "acorn guest house"}
    }
    policy_rules = [
        "Must not book without: name, day, people, stay",
        "Must offer alternatives if no exact match"
    ]

    prompt = create_judge_prompt(user_message, system_response, ground_truth_slots, policy_rules)

    print("\nGenerated Prompt:")
    print(prompt)
    print("\nPrompt includes:")
    print(f"- User message: {'✓' if user_message in prompt else '✗'}")
    print(f"- System response: {'✓' if system_response in prompt else '✗'}")
    print(f"- Ground truth: {'✓' if 'acorn guest house' in prompt else '✗'}")
    print(f"- Policy rules: {'✓' if 'Must not book without' in prompt else '✗'}")
    print(f"- Rubric scale: {'✓' if '1-5' in prompt else '✗'}")


def test_judge_response_parsing() -> None:
    """Test parsing of judge responses."""

    # Test 1: Valid JSON response
    print_separator("TEST PARSING (1): VALID JSON")
    valid_response = """{
    "score": 4,
    "correctness": "Accurate hotel name and attributes",
    "completeness": "Addresses all user constraints",
    "clarity": "Clear and professional",
    "policy_adherence": "Correctly does not book without required info",
    "overall_reasoning": "Good response, offers booking without forcing it"
}"""

    result = parse_judge_response(valid_response)
    print(f"\nSimulated LLM Response:\n {valid_response}")
    print("\nAfter Parsing:")
    print(f"Score: {result.get('score')}")
    print(f"Correctness: {result.get('correctness')}")
    print(f"Error: {result.get('error', 'None')}")

    # Test 2: JSON in the Markdown code block
    print_separator("TEST PARSING (2): MARKDOWN CODE BLOCK")
    markdown_response = """Here's my evaluation:
```json
{
    "score": 3,
    "correctness": "Partially correct",
    "completeness": "Missing some details",
    "clarity": "Adequate",
    "policy_adherence": "Follows policies",
    "overall_reasoning": "Average response"
}
```

Hope this helps!"""

    result = parse_judge_response(markdown_response)
    print(f"\nSimulated LLM Response:\n {markdown_response}")
    print("\nAfter Parsing:")
    print(f"Score: {result.get('score')}")
    print(f"Overall: {result.get('overall_reasoning')}")
    print(f"Error: {result.get('error', 'None')}")

    # Test 3: Invalid score range
    print_separator("TEST PARSING (3): INVALID SCORE")
    invalid_score = """{
    "score": 7,
    "correctness": "Good",
    "completeness": "Good",
    "clarity": "Good",
    "policy_adherence": "Good",
    "overall_reasoning": "Good"
}"""

    result = parse_judge_response(invalid_score)
    print(f"\nSimulated LLM Response:\n {invalid_score}")
    print("\nAfter Parsing:")
    print(f"Score: {result.get('score')}")
    print(f"Error: {result.get('error', 'None')}")

    # Test 4: Missing score field
    print_separator("TEST PARSING (4): MISSING SCORE")
    missing_score = """{
    "correctness": "Good",
    "completeness": "Good"
}"""

    result = parse_judge_response(missing_score)
    print(f"\nSimulated LLM Response:\n {missing_score}")
    print("\nAfter Parsing:")
    print(f"Score: {result.get('score')}")
    print(f"Error: {result.get('error', 'None')}")

    # Test 5: Malformed JSON
    print_separator("TEST PARSING (5): MALFORMED JSON")
    malformed = "This is not valid JSON at all!"

    result = parse_judge_response(malformed)
    print(f"\nSimulated LLM Response:\n {malformed}")
    print("\nAfter Parsing:")
    print(f"Score: {result.get('score')}")
    print(f"Error: {result.get('error', 'None')}")


def run_all_tests() -> None:
    """Run all evaluation metric tests in sequence."""
    print_separator("EVALUATION METRICS - COMPREHENSIVE TEST SUITE")

    print_separator("PART A: OBJECTIVE METRICS")
    test_domain_accuracy()
    test_intent_accuracy()
    test_action_type_accuracy()
    test_slot_accuracy()
    test_jga()
    test_hallucination_rate()
    test_memory_transfer_accuracy()
    test_policy_compliance()
    test_system_correctness()
    test_task_success()

    print_separator("PART B: SUBJECTIVE METRICS")
    test_judge_prompt_creation()
    test_judge_response_parsing()

    print_separator("ALL TESTS COMPLETED SUCCESSFULLY")


if __name__ == "__main__":

    capture_and_save(func=run_all_tests,
                     output_path="docs/evals_inspection/objective_and_judge_metrics.txt"
    )

