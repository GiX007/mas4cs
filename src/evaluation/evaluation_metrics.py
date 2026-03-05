"""Evaluation metrics for calculating MAS4CS performance."""
from typing import Any
from src.data import BOOKING_REQUIRED_SLOTS


def calculate_domain_accuracy(predicted_domain: str, ground_truth_intent: str) -> tuple[float, bool, str]:
    """
    Calculate domain routing accuracy for a single turn.

    Domain is extracted from the intent (e.g., "find_restaurant" → "restaurant").
    Domain accuracy is binary: 1.0 if correct domain, 0.0 otherwise.

    Args:
        predicted_domain: System's routed domain (e.g., "restaurant")
        ground_truth_intent: Annotated intent used to infer correct domain

    Returns:
        (accuracy, is_correct, ground_truth_domain): Score, boolean, and extracted domain
    """
    # Extract domain from intent (e.g., "find_restaurant" → "restaurant")
    # MultiWOZ format: intent is "{action}_{domain}"
    if "_" in ground_truth_intent:
        ground_truth_domain = ground_truth_intent.split("_")[-1]
    else:
        # Fallback: if no underscore, assume intent IS the domain
        ground_truth_domain = ground_truth_intent

    is_correct = predicted_domain.lower() == ground_truth_domain.lower()  # Add case insensitivity
    accuracy = 1.0 if is_correct else 0.0

    return accuracy, is_correct, ground_truth_domain


def calculate_set_based_accuracy(predicted: str | list[str], ground_truth: str | list[str], return_detailed: bool = False) -> tuple[float, bool] | tuple[float, bool, float, float, float, int, int, int]:
    """
    Generic accuracy calculator for set-based comparisons (intent, action-type, etc.).

    Handles both single values (str) and multiple values (list[str]).

    Args:
        predicted: System's prediction (single string or list)
        ground_truth: Ground truth annotation (single string or list)
        return_detailed: If True, also return recall/precision/f1 metrics

    Returns:
        If return_detailed=False: (accuracy, is_correct)
        If return_detailed=True:  (accuracy, is_correct, recall, precision, f1, num_correct, num_predicted, num_ground_truth)
    """
    # Convert to sets for uniform handling
    predicted_set = {predicted} if isinstance(predicted, str) else set(predicted)
    ground_truth_set = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)

    # Exact match (strict accuracy)
    is_correct = predicted_set == ground_truth_set
    accuracy = 1.0 if is_correct else 0.0

    if not return_detailed:
        return accuracy, is_correct

    # Detailed metrics
    correct_items = predicted_set & ground_truth_set
    num_correct = len(correct_items)
    num_predicted = len(predicted_set)
    num_ground_truth = len(ground_truth_set)

    recall = num_correct / num_ground_truth if num_ground_truth > 0 else 0.0
    precision = num_correct / num_predicted if num_predicted > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return accuracy, is_correct, recall, precision, f1, num_correct, num_predicted, num_ground_truth


def calculate_intent_accuracy(predicted_intent: str, ground_truth_intent: str, return_detailed: bool = False) -> tuple[float, bool] | tuple[float, bool, float, float, int, int, int]:
    """
    Calculate intent detection accuracy. Wrapper for set_based_accuracy specialized for intent comparison.

    Args:
        predicted_intent: System's detected intent (e.g., "find_restaurant")
        ground_truth_intent: Annotated ground truth intent
        return_detailed: If True, also return recall/precision metrics

    Returns:
        If return_detailed=False: (accuracy, is_correct)
        If return_detailed=True: (accuracy, is_correct, recall, precision, num_correct, num_predicted, num_ground_truth)
    """
    return calculate_set_based_accuracy(predicted_intent, ground_truth_intent, return_detailed)


def calculate_action_type_accuracy(predicted_act_type: list[str], ground_truth_act_type: list[str], return_detailed: bool = False) -> tuple[float, bool] | tuple[float, bool, float, float, int, int, int]:
    """
    Calculate action-type accuracy. Wrapper for set_based_accuracy specialized for dialogue act comparison.
    Acts are compared as sets since order doesn't matter.

    Args:
        predicted_act_type: System's dialogue acts (e.g., ["Restaurant-Inform"])
        ground_truth_act_type: Annotated ground truth acts
        return_detailed: If True, also return recall/precision metrics

    Returns:
        If return_detailed=False: (accuracy, is_correct)
        If return_detailed=True: (accuracy, is_correct, recall, precision, num_correct, num_predicted, num_ground_truth)
    """
    return calculate_set_based_accuracy(predicted_act_type, ground_truth_act_type, return_detailed)


def calculate_slot_accuracy(predicted_slots: dict[str, dict[str, str]], ground_truth_slots: dict[str, dict[str, str]]) -> tuple[float, float, float, int, int, int]:
    """
    Calculate slot-level recall, precision, and F1 across all domains.

    Slot Recall = correct / num_gt_slots (how many GT slots we got right)
    Slot Precision = correct / num_predicted_slots (how many predicted slots are correct)
    Slot F1 = 2 * precision * recall / (precision + recall)

    Args:
        predicted_slots: System's tracked slots, format: {"hotel": {"area": "south"}}
        ground_truth_slots: Annotated ground truth, same format

    Returns:
        (recall, precision, f1, num_correct, num_predicted, num_gt)
    """
    num_correct = 0
    num_gt = 0
    num_predicted = 0

    # Count all ground truth slot-value pairs
    for domain, slots in ground_truth_slots.items():
        for slot, value in slots.items():
            num_gt += 1
            if domain in predicted_slots:
                if predicted_slots[domain].get(slot) == value:
                    num_correct += 1

    # Count all predicted slot-value pairs
    for domain, slots in predicted_slots.items():
        for slot, value in slots.items():
            if value and value not in ("none", "not mentioned", "dontcare"):  # Filter out dontcare values before metric calculation
                num_predicted += 1

    recall = num_correct / num_gt if num_gt > 0 else 0.0
    precision = num_correct / num_predicted if num_predicted > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return recall, precision, f1, num_correct, num_predicted, num_gt


def calculate_jga(predicted_slots: dict[str, dict[str, str]], ground_truth_slots: dict[str, dict[str, str]]) -> tuple[float, dict[str, bool]]:
    """
    Calculate Joint Goal Accuracy (JGA) for dialogue state tracking.

    JGA = 1.0 if ALL slots across ALL domains match exactly, else 0.0.

    Args:
        predicted_slots: System's tracked slots, format: {"hotel": {"area": "south"}}
        ground_truth_slots: Annotated ground truth, same format

    Returns:
        (jga_score, per_domain_accuracy): Score is 1.0 or 0.0, plus domain breakdown
    """
    # Get all domains mentioned in either prediction or ground truth
    all_domains = set(predicted_slots.keys()) | set(ground_truth_slots.keys())
    # print(all_domains)

    per_domain_accuracy = {}
    all_match = True

    for domain in all_domains:
        predicted = predicted_slots.get(domain, {})
        truth = ground_truth_slots.get(domain, {})

        # Domain matches if both slot sets are identical
        domain_matches = predicted == truth
        per_domain_accuracy[domain] = domain_matches

        if not domain_matches:
            all_match = False

    jga_score = 1.0 if all_match else 0.0

    return jga_score, per_domain_accuracy


def calculate_hallucination_rate(system_response: str, valid_entities: list[str]) -> tuple[float, int, int]:
    """
    Calculate entity hallucination rate for a single turn.

    Measures whether the system response mentions hotel/restaurant names that were NOT returned by the DB query for this turn.

    Hallucination Rate = hallucinated_entities / total_entities_mentioned

    If no DB query was made this turn (valid_entities=[]), skip check → return (0.0, 0, 0).
    If DB query returned results but system mentions unknown entity → hallucination.

    Args:
        system_response: System's natural language response for this turn
        valid_entities:  Entity names returned by DB this turn (from find_entity/book_entity)

    Returns:
        (hallucination_rate, num_hallucinated, num_mentioned)
    """
    # No DB query made this turn — nothing to hallucinate about
    if not valid_entities:
        return 0.0, 0, 0

    response_lower = system_response.lower()

    # Check which valid entities are mentioned and which unknown ones appear
    num_mentioned = 0
    num_hallucinated = 0

    # Count valid entity mentions
    valid_mentioned = [e for e in valid_entities if e.lower() in response_lower]
    num_mentioned += len(valid_mentioned)

    # Load all known entity names from DB to detect unknown mentions
    from src.core.tools import load_db
    all_known = []
    for domain in ["hotel", "restaurant"]:
        try:
            db = load_db(domain)
            all_known.extend([e["name"].lower() for e in db if "name" in e])
        except Exception as e:
            print(f"Error loading DB for {domain}: {e}")
            pass

    # Any DB entity name mentioned in response that is NOT in valid_entities = hallucination
    for name in all_known:
        if name in response_lower and name not in [e.lower() for e in valid_entities]:
            num_hallucinated += 1
            num_mentioned += 1

    hallucination_rate = num_hallucinated / num_mentioned if num_mentioned > 0 else 0.0

    return hallucination_rate, num_hallucinated, num_mentioned


def calculate_policy_compliance(action_taken: str, required_slots: dict[str, list[str]], current_slots: dict[str, dict[str, str]]) -> tuple[bool, str]:
    """
    Check if the system complied with policy constraints.

    Policy compliance means:
    - If action requires slots, they must all be present
    - If slots are missing, action should be "request_slots" not the actual action

    Args:
        action_taken: The action the system took (e.g., "book_hotel", "request_slots")
        required_slots: Required slots per action, format: {"book_hotel": ["name", "bookday", "bookpeople", "bookstay"]}
        current_slots: Current dialogue state, format: {"hotel": {"name": "...", "bookday": "..."}}

    Returns:
        (is_compliant, reason): True if policy followed, plus explanation
    """
    # If no requirements for this action, always compliant
    if action_taken not in required_slots:
        return True, "No policy constraints for this action"

    # Extract domain from action (e.g., "book_hotel" → "hotel")
    domain = action_taken.split("_")[-1] if "_" in action_taken else None

    if domain is None:
        return True, "Action has no domain association"

    # Get current slots for this domain
    domain_slots = current_slots.get(domain, {})

    # Check if all required slots are present
    missing_slots = []
    for required_slot in required_slots[action_taken]:
        if required_slot not in domain_slots or not domain_slots[required_slot]:
            missing_slots.append(required_slot)

    # If slots are missing, action should have been "request_slots"
    if missing_slots:
        return False, f"Policy violation: Attempted {action_taken} with missing slots: {missing_slots}"


    # All required slots present, action is valid
    return True, f"All required slots present for {action_taken}"


def calculate_system_correctness(hallucination_detected: bool, policy_compliant: bool) -> tuple[bool, str]:
    """
    Determine if the system responded correctly for a single turn.

    System correctness is a composite binary metric: a turn is correct if and only if no hallucination occurred AND policy was respected.

    Args:
        hallucination_detected: Whether system mentioned entities not returned by DB
        policy_compliant: Whether system followed booking policy constraints

    Returns:
        (is_correct, reason): True if system behaved correctly, plus explanation
    """
    if hallucination_detected:
        return False, "Entity hallucination detected in response"
    if not policy_compliant:
        return False, "Policy violation detected"
    return True, "System response correct"


def calculate_booking_success(turn_results: list[dict[str, Any]], services: list[str], requires_booking: bool,) -> tuple[bool, str]:
    """
    Determine if the dialogue successfully completed a booking goal.

    Booking success requires:
    - All required domains were addressed
    - A booking action occurred
    - All required booking slots were filled by end of dialogue

    Args:
        turn_results: List of turn metric dicts from DialogueEvaluator
        services: Domains required for this dialogue (from dialogue["services"])
        requires_booking: True if any turn had a booking intent (from extract_booking())

    Returns:
        (is_successful, reason): True if booking completed, plus explanation
    """
    # Not a booking dialogue -> return None, skip
    if not requires_booking:
        return None, "No booking required — not applicable"

    addressed_domains = set()
    booking_action_occurred = False
    final_slots = turn_results[-1].get("predicted_slots", {}) if turn_results else {}

    for turn in turn_results:
        domain = turn.get("domain", "")
        action = turn.get("action", "")

        if domain and domain != "none":
            addressed_domains.add(domain)

        if "book" in action:
            booking_action_occurred = True

    # All required domains must be addressed
    missing_domains = set(services) - addressed_domains
    if missing_domains:
        return False, f"Missing domains: {missing_domains}"

    # Booking action must have occurred
    if not booking_action_occurred:
        return False, "Booking required but no booking action occurred"

    # All required slots must be filled at end of dialogue
    for domain in services:
        required = BOOKING_REQUIRED_SLOTS.get(f"book_{domain}", [])
        domain_slots = final_slots.get(domain, {})
        missing_slots = [s for s in required if s not in domain_slots]
        if missing_slots:
            return False, f"Missing booking slots in {domain}: {missing_slots}"

    return True, "Booking completed successfully"
