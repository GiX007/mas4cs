"""
Comprehensive tests for all evaluation methods.

Tests all objective metrics (JGA, slot accuracy, hallucination rate, policy compliance,
task success, intent accuracy, action-type accuracy, domain accuracy, memory transfer)
and subjective metrics (LLM-as-judge prompt formatting and response parsing).
"""
from src.evaluation import (
    calculate_intent_accuracy, calculate_action_type_accuracy, calculate_jga, calculate_slot_accuracy, calculate_hallucination_rate, calculate_policy_compliance,
    calculate_booking_success, calculate_system_correctness, calculate_domain_accuracy, create_judge_prompt, parse_judge_response
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
    acc, is_correct, recall, precision, f1, correct, predicted_total, gt_total = calculate_intent_accuracy(predicted, ground_truth, return_detailed=True)

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

    acc, is_correct, recall, precision, f1, correct, p_total, gt_total = calculate_action_type_accuracy(predicted, ground_truth, return_detailed=True)
    print(f"Accuracy: {acc}")  # Should be 1.0
    print(f"Is Correct: {is_correct}")  # Should be True
    print(f"Recall: {recall:.2f}, Precision: {precision:.2f} ({correct}/{gt_total}, {correct}/{p_total})")

    # Test 2: Wrong act type
    print_separator("TEST ACTION-TYPE ACCURACY (2): WRONG ACT")
    predicted = ["Restaurant-Request"]
    ground_truth = ["Restaurant-Inform"]
    print(f"\nPredicted: {predicted}")
    print(f"Ground truth: {ground_truth}\n")

    acc, is_correct, recall, precision, f1, correct, p_total, gt_total = calculate_action_type_accuracy(predicted, ground_truth, return_detailed=True)
    print(f"Accuracy: {acc}")  # Should be 0.0
    print(f"Is Correct: {is_correct}")  # Should be False
    print(f"Recall: {recall:.2f}, Precision: {precision:.2f} ({correct}/{gt_total}, {correct}/{p_total})")

    # Test 3: Multi-act match (order doesn't matter)
    print_separator("TEST ACTION-TYPE ACCURACY (3): MULTI-ACT MATCH")
    predicted = ["Hotel-Inform", "Hotel-Request"]
    ground_truth = ["Hotel-Request", "Hotel-Inform"]  # Different order
    print(f"\nPredicted: {predicted}")
    print(f"Ground truth: {ground_truth}\n")

    acc, is_correct, recall, precision, f1, correct, p_total, gt_total = calculate_action_type_accuracy(predicted, ground_truth, return_detailed=True)
    print(f"Accuracy: {acc}")  # Should be 1.0
    print(f"Is Correct: {is_correct}")  # Should be True
    print(f"Recall: {recall:.2f}, Precision: {precision:.2f} ({correct}/{gt_total}, {correct}/{p_total})")

    # Test 4: Multi-act mismatch
    print_separator("TEST ACTION-TYPE ACCURACY (4): MULTI-ACT MISMATCH")
    predicted = ["Hotel-Inform", "Hotel-Request"]
    ground_truth = ["Hotel-Request", "Restaurant-Inform"]
    print(f"\nPredicted: {predicted}")  # Should be 0.0
    print(f"Ground truth: {ground_truth}\n")  # Should be False

    acc, is_correct, recall, precision, f1, correct, p_total, gt_total = calculate_action_type_accuracy(predicted, ground_truth, return_detailed=True)
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

    acc, is_correct, recall, precision, f1, correct, p_total, gt_total = calculate_action_type_accuracy(predicted, ground_truth, return_detailed=True)
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

    acc, is_correct, recall, precision, f1, correct, p_total, gt_total = calculate_action_type_accuracy(predicted, ground_truth, return_detailed=True)
    print(f"Accuracy: {acc}")  # Should be 0.0
    print(f"Is Correct: {is_correct}")  # Should be False
    print(f"Recall: {recall:.2f}, Precision: {precision:.2f} ({correct}/{gt_total}, {correct}/{p_total})")
    # Expected: Recall=1.00 (2/2), Precision=0.67 (2/3) - complete but with hallucination


def test_slot_accuracy() -> None:
    """Test slot-level recall, precision, and F1 calculation."""
    # Test 1: Perfect match
    print_separator("TEST SLOT ACCURACY (1): PERFECT MATCH")
    predicted = {"restaurant": {"area": "centre", "pricerange": "expensive"}}
    ground_truth = {"restaurant": {"area": "centre", "pricerange": "expensive"}}
    print(f"\nPredicted: {predicted}")
    print(f"Ground truth: {ground_truth}\n")

    recall, precision, f1, num_correct, num_predicted, num_gt = calculate_slot_accuracy(predicted, ground_truth)
    print(f"Recall: {recall:.2f} ({num_correct}/{num_gt})")  # 1.00 (2/2)
    print(f"Precision: {precision:.2f} ({num_correct}/{num_predicted})")  # 1.00 (2/2)
    print(f"F1: {f1:.2f}")  # 1.00

    # Test 2: Partial match (1 out of 2 correct)
    print_separator("TEST SLOT ACCURACY (2): PARTIAL MATCH")
    predicted = {"restaurant": {"area": "centre", "pricerange": "cheap"}}
    ground_truth = {"restaurant": {"area": "centre", "pricerange": "expensive"}}
    print(f"\nPredicted: {predicted}")
    print(f"Ground truth: {ground_truth}\n")

    recall, precision, f1, num_correct, num_predicted, num_gt = calculate_slot_accuracy(predicted, ground_truth)
    print(f"Recall: {recall:.2f} ({num_correct}/{num_gt})")  # 0.50 (1/2)
    print(f"Precision: {precision:.2f} ({num_correct}/{num_predicted})")  # 0.50 (1/2)
    print(f"F1: {f1:.2f}")  # 0.50

    # Test 3: Missing domain
    print_separator("TEST SLOT ACCURACY (3): MISSING DOMAIN")
    predicted = {"restaurant": {"area": "centre"}}
    ground_truth = {
        "restaurant": {"area": "centre"},
        "hotel": {"pricerange": "cheap"}
    }
    print(f"\nPredicted: {predicted}")
    print(f"Ground truth: {ground_truth}\n")

    recall, precision, f1, num_correct, num_predicted, num_gt = calculate_slot_accuracy(predicted, ground_truth)
    print(f"Recall: {recall:.2f} ({num_correct}/{num_gt})")  # 0.50 (1/2)
    print(f"Precision: {precision:.2f} ({num_correct}/{num_predicted})")  # 1.00 (1/1)
    print(f"F1: {f1:.2f}")  # 0.67

    # Test 4: Extra slots in prediction (precision penalized, recall unaffected)
    print_separator("TEST SLOT ACCURACY (4): EXTRA PREDICTION SLOTS")
    predicted = {"restaurant": {"area": "centre", "pricerange": "cheap", "food": "italian"}}
    ground_truth = {"restaurant": {"area": "centre", "pricerange": "cheap"}}
    print(f"\nPredicted: {predicted}")
    print(f"Ground truth: {ground_truth}\n")

    recall, precision, f1, num_correct, num_predicted, num_gt = calculate_slot_accuracy(predicted, ground_truth)
    print(f"Recall: {recall:.2f} ({num_correct}/{num_gt})")  # 1.00 (2/2)
    print(f"Precision: {precision:.2f} ({num_correct}/{num_predicted})")  # 0.67 (2/3)
    print(f"F1: {f1:.2f}")  # 0.80


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
    """Test entity hallucination rate calculation."""
    # Test 1: No DB query made this turn → skip check
    print_separator("TEST HALLUCINATION (1): NO DB QUERY → SKIP")
    system_response = "I can help you find a restaurant. What area do you prefer?"
    valid_entities = []  # no DB query made
    print(f"\nResponse: {system_response}")
    print(f"Valid entities: {valid_entities}\n")

    hall_rate, entities_hallucinated, entities_mentioned = calculate_hallucination_rate(system_response, valid_entities)
    print(f"Hallucination Rate: {hall_rate:.2f} ({entities_hallucinated}/{entities_mentioned})")  # 0.00 (0/0) — skipped

    # Test 2: System mentions valid entity → no hallucination
    print_separator("TEST HALLUCINATION (2): VALID ENTITY MENTIONED")
    system_response = "I recommend Golden Wok, a chinese restaurant in the north."
    valid_entities = ["Golden Wok"]
    print(f"\nResponse: {system_response}")
    print(f"Valid entities: {valid_entities}\n")

    hall_rate, entities_hallucinated, entities_mentioned = calculate_hallucination_rate(system_response, valid_entities)
    print(f"Hallucination Rate: {hall_rate:.2f} ({entities_hallucinated}/{entities_mentioned})")  # 0.00 (0/1)

    # Test 3: System mentions entity NOT in valid_entities → hallucination
    print_separator("TEST HALLUCINATION (3): HALLUCINATED ENTITY")
    system_response = "I recommend Lovell Lodge for your stay."
    valid_entities = ["Golden Wok"]  # DB returned Golden Wok, not Lovell Lodge
    print(f"\nResponse: {system_response}")
    print(f"Valid entities: {valid_entities}\n")

    hall_rate, entities_hallucinated, entities_mentioned = calculate_hallucination_rate(system_response, valid_entities)
    print(f"Hallucination Rate: {hall_rate:.2f} ({entities_hallucinated}/{entities_mentioned})")  # 1.00 (1/1)

    # Test 4: System mentions both valid and hallucinated entity → partial hallucination
    print_separator("TEST HALLUCINATION (4): PARTIAL HALLUCINATION")
    system_response = "I recommend Golden Wok or Lovell Lodge for your stay."
    valid_entities = ["Golden Wok"]  # Lovell Lodge not in DB results
    print(f"\nResponse: {system_response}")
    print(f"Valid entities: {valid_entities}\n")

    hall_rate, entities_hallucinated, entities_mentioned = calculate_hallucination_rate(system_response, valid_entities)
    print(f"Hallucination Rate: {hall_rate:.2f} ({entities_hallucinated}/{entities_mentioned})")  # 0.50 (1/2)

    # Test 5: System mentions no entity at all → 0.0
    print_separator("TEST HALLUCINATION (5): NO ENTITY MENTIONED IN RESPONSE")
    system_response = "I have booked a table for you on Friday at 18:30."
    valid_entities = ["Golden Wok"]  # DB query was made but entity not mentioned by name
    print(f"\nResponse: {system_response}")
    print(f"Valid entities: {valid_entities}\n")

    hall_rate, entities_hallucinated, entities_mentioned = calculate_hallucination_rate(system_response, valid_entities)
    print(f"Hallucination Rate: {hall_rate:.2f} ({entities_hallucinated}/{entities_mentioned})")  # 0.00 (0/0)


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
    # Test 1: No hallucination, policy compliant → correct
    print_separator("TEST SYSTEM CORRECTNESS (1): NO HALLUCINATION + POLICY OK")
    hallucination_detected = False
    policy_compliant = True
    print(f"\nHallucination: {hallucination_detected} | Policy compliant: {policy_compliant}\n")
    correct, reason = calculate_system_correctness(hallucination_detected, policy_compliant)
    print(f"System Correct: {correct}")  # True
    print(f"Reason: {reason}")

    # Test 2: Hallucination detected → incorrect
    print_separator("TEST SYSTEM CORRECTNESS (2): HALLUCINATION DETECTED")
    hallucination_detected = True
    policy_compliant = True
    print(f"\nHallucination: {hallucination_detected} | Policy compliant: {policy_compliant}\n")
    correct, reason = calculate_system_correctness(hallucination_detected, policy_compliant)
    print(f"System Correct: {correct}")  # False
    print(f"Reason: {reason}")

    # Test 3: Policy violation → incorrect
    print_separator("TEST SYSTEM CORRECTNESS (3): POLICY VIOLATION")
    hallucination_detected = False
    policy_compliant = False
    print(f"\nHallucination: {hallucination_detected} | Policy compliant: {policy_compliant}\n")
    correct, reason = calculate_system_correctness(hallucination_detected, policy_compliant)
    print(f"System Correct: {correct}")  # False
    print(f"Reason: {reason}")

    # Test 4: Both hallucination and policy violation → incorrect
    print_separator("TEST SYSTEM CORRECTNESS (4): HALLUCINATION + POLICY VIOLATION")
    hallucination_detected = True
    policy_compliant = False
    print(f"\nHallucination: {hallucination_detected} | Policy compliant: {policy_compliant}\n")
    correct, reason = calculate_system_correctness(hallucination_detected, policy_compliant)
    print(f"System Correct: {correct}")  # False
    print(f"Reason: {reason}")


def test_booking_success() -> None:
    """Test booking success calculation."""
    # Test 1: Successful booking — all slots present + booking action occurred
    print_separator("TEST TASK SUCCESS (1): SUCCESSFUL BOOKING")
    turn_results = [
        {"domain": "hotel", "action": "find_hotel", "predicted_slots": {"hotel": {"area": "centre"}}},
        {"domain": "hotel", "action": "book_hotel", "predicted_slots": {"hotel": {"name": "hilton", "bookday": "monday", "bookpeople": "2", "bookstay": "3"}}}
    ]
    services = ["hotel"]
    requires_booking = True
    print(f"\nFinal slots: {turn_results[-1]['predicted_slots']}")
    print(f"Services: {services} | Requires booking: {requires_booking}\n")

    success, reason = calculate_booking_success(turn_results, services, requires_booking)
    print(f"Success: {success}")  # True
    print(f"Reason: {reason}")

    # Test 2: Failed — missing booking slots
    print_separator("TEST TASK SUCCESS (2): MISSING BOOKING SLOTS")
    turn_results = [
        {"domain": "hotel", "action": "find_hotel", "predicted_slots": {"hotel": {"area": "centre"}}},
        {"domain": "hotel", "action": "book_hotel", "predicted_slots": {"hotel": {"bookpeople": "2", "bookstay": "3"}}}  # missing name + bookday
    ]
    services = ["hotel"]
    requires_booking = True
    print(f"\nFinal slots: {turn_results[-1]['predicted_slots']}")
    print(f"Services: {services} | Requires booking: {requires_booking}\n")

    success, reason = calculate_booking_success(turn_results, services, requires_booking)
    print(f"Success: {success}")  # False
    print(f"Reason: {reason}")

    # Test 3: Failed — booking action never occurred
    print_separator("TEST TASK SUCCESS (3): NO BOOKING ACTION")
    turn_results = [
        {"domain": "hotel", "action": "find_hotel", "predicted_slots": {"hotel": {"name": "hilton", "bookday": "monday", "bookpeople": "2", "bookstay": "3"}}},
    ]
    services = ["hotel"]
    requires_booking = True
    print(f"\nFinal slots: {turn_results[-1]['predicted_slots']}")
    print(f"Services: {services} | Requires booking: {requires_booking}\n")

    success, reason = calculate_booking_success(turn_results, services, requires_booking)
    print(f"Success: {success}")  # False
    print(f"Reason: {reason}")

    # Test 4: Info-only dialogue — not applicable, returns None
    print_separator("TEST TASK SUCCESS (4): INFO-ONLY — NOT APPLICABLE")
    turn_results = [
        {"domain": "restaurant", "action": "find_restaurant", "predicted_slots": {"restaurant": {"area": "centre", "food": "italian"}}},
    ]
    services = ["restaurant"]
    requires_booking = False
    print(f"\nFinal slots: {turn_results[-1]['predicted_slots']}")
    print(f"Services: {services} | Requires booking: {requires_booking}\n")

    success, reason = calculate_booking_success(turn_results, services, requires_booking)
    print(f"Success: {success}")  # None — skipped in aggregation
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
    test_policy_compliance()
    test_system_correctness()
    test_booking_success()

    print_separator("PART B: SUBJECTIVE METRICS")
    test_judge_prompt_creation()
    test_judge_response_parsing()

    print_separator("ALL TESTS COMPLETED SUCCESSFULLY")


if __name__ == "__main__":

    capture_and_save(func=run_all_tests,
                     output_path="docs/evals_inspection/objective_and_judge_metrics.txt"
    )
