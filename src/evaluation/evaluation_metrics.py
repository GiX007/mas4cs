"""Evaluation metrics for calculating MAS4CS performance."""

from typing import Any
from src.data import BOOKING_REQUIRED_SLOTS, VALID_ACTION_TYPES


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


def calculate_set_based_accuracy(predicted: str | list[str], ground_truth: str | list[str], return_detailed: bool = False) -> tuple[float, bool] | tuple[float, bool, float, float, int, int, int]:
    """
    Generic accuracy calculator for set-based comparisons (intent, action-type, etc.).

    Handles both single values (str) and multiple values (List[str]).

    Args:
        predicted: System's prediction (single string or list)
        ground_truth: Ground truth annotation (single string or list)
        return_detailed: If True, also return recall/precision metrics

    Returns:
        If return_detailed=False: (accuracy, is_correct)
        If return_detailed=True: (accuracy, is_correct, recall, precision, num_correct, num_predicted, num_ground_truth)
    """
    # Convert to sets for uniform handling
    predicted_set = {predicted} if isinstance(predicted, str) else set(predicted)  # {predicted} -> set([predicted])
    ground_truth_set = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)  # {ground_truth} -> set([ground_truth])

    # Exact match (strict accuracy)
    is_correct = predicted_set == ground_truth_set
    accuracy = 1.0 if is_correct else 0.0

    if not return_detailed:
        return accuracy, is_correct

    # Detailed metrics (recall/precision)
    correct_items = predicted_set & ground_truth_set
    num_correct = len(correct_items)
    num_predicted = len(predicted_set)
    num_ground_truth = len(ground_truth_set)

    recall = num_correct / num_ground_truth if num_ground_truth > 0 else 0.0
    precision = num_correct / num_predicted if num_predicted > 0 else 0.0

    return accuracy, is_correct, recall, precision, num_correct, num_predicted, num_ground_truth


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


def calculate_slot_accuracy(predicted_slots: dict[str, dict[str, str]], ground_truth_slots: dict[str, dict[str, str]]) -> tuple[float, int, int]:
    """
    Calculate slot-level accuracy across all domains.

    Slot Accuracy = (correct slot-value pairs) / (total ground-truth pairs)

    Args:
        predicted_slots: System's tracked slots, format: {"hotel": {"area": "south"}}
        ground_truth_slots: Annotated ground truth, same format

    Returns:
        (accuracy, num_correct, num_total): Accuracy score, correct count, total count
    """
    num_correct = 0
    num_total = 0

    # Count all ground truth slot-value pairs
    for domain, slots in ground_truth_slots.items():
        for slot, value in slots.items():
            num_total += 1

            # Check if prediction has this exact slot-value pair
            if domain in predicted_slots:
                if predicted_slots[domain].get(slot) == value:
                    num_correct += 1

    # Calculate accuracy
    accuracy = num_correct / num_total if num_total > 0 else 0.0

    return accuracy, num_correct, num_total


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


def calculate_hallucination_rate(predicted_slots: dict[str, dict[str, str]], ground_truth_slots: dict[str, dict[str, str]], current_domain: str | None = None) -> tuple[float, int, int]:
    """
    Calculate hallucination rate (false positives in predictions).

    Hallucination Rate = (wrong/extra slot-value pairs) / (total predicted pairs)

    Args:
        predicted_slots: System's tracked slots, format: {"hotel": {"area": "south"}}
        ground_truth_slots: Annotated ground truth, same format
        current_domain: If specified, only evaluate hallucinations for this domain/turn (no consideration of accumulated slots over dialogue)

    Returns:
        (hallucination_rate, num_hallucinated, num_predicted): Rate, hallucination count, total predicted
    """
    num_hallucinated = 0
    num_predicted = 0

    # Filter to current domain only if specified
    if current_domain:
        predicted_slots = {current_domain: predicted_slots.get(current_domain, {})}
        ground_truth_slots = {current_domain: ground_truth_slots.get(current_domain, {})}

    # Count all predicted slot-value pairs
    for domain, slots in predicted_slots.items():
        for slot, value in slots.items():
            num_predicted += 1

            # Hallucination = predicting a slot key that doesn't exist in GT
            # Check if ground truth has this exact slot-value pair
            if domain not in ground_truth_slots:
                num_hallucinated += 1
            # elif ground_truth_slots[domain].get(slot) != value:
            #     num_hallucinated += 1
            elif slot not in ground_truth_slots[domain]:  # ← key check only, not value
                num_hallucinated += 1

    # Avoid division by zero
    hallucination_rate = num_hallucinated / num_predicted if num_predicted > 0 else 0.0

    return hallucination_rate, num_hallucinated, num_predicted


def calculate_memory_transfer_accuracy(dialogue_history: list[dict[str, Any]], transferable_slots: list[str] = None) -> tuple[float, int, int, list[dict[str, Any]]]:
    """
    Calculate cross-domain memory transfer accuracy.

    Detects domain switches and checks if shared constraints were carried over.
    Only evaluates turns where a domain switch occurred.

    Args:
        dialogue_history: List of turn states, each with:
            - "turn_id": int
            - "domain": str (current domain)
            - "predicted_slots": Dict[str, Dict[str, str]]
        transferable_slots: Slots that should transfer (default: ["area", "pricerange"])

    Returns:
        (accuracy, correct_transfers, total_transfers, transfer_events):
        Accuracy score, correct count, total count, and list of transfer event details
    """
    if transferable_slots is None:
        transferable_slots = ["area", "pricerange"]

    correct_transfers = 0
    total_transfers = 0
    transfer_events = []

    # Need at least 2 turns to detect a switch
    if len(dialogue_history) < 2:
        return 0.0, 0, 0, []

    for i in range(1, len(dialogue_history)):
        current_turn = dialogue_history[i]
        previous_turn = dialogue_history[i - 1]

        current_domain = current_turn.get("domain")
        previous_domain = previous_turn.get("domain")

        # Detect domain switch
        if current_domain != previous_domain and current_domain and previous_domain:
            # Check each transferable slot
            previous_slots = previous_turn.get("predicted_slots", {}).get(previous_domain, {})
            current_slots = current_turn.get("predicted_slots", {}).get(current_domain, {})

            for slot in transferable_slots:
                if slot in previous_slots and previous_slots[slot]:
                    # This slot was present in previous domain, should transfer
                    total_transfers += 1

                    transferred_correctly = (
                            slot in current_slots and
                            current_slots[slot] == previous_slots[slot]
                    )

                    if transferred_correctly:
                        correct_transfers += 1

                    transfer_events.append({
                        "turn_id": current_turn.get("turn_id"),
                        "from_domain": previous_domain,
                        "to_domain": current_domain,
                        "slot": slot,
                        "value": previous_slots[slot],
                        "transferred": transferred_correctly
                    })

    accuracy = correct_transfers / total_transfers if total_transfers > 0 else 0.0

    return accuracy, correct_transfers, total_transfers, transfer_events


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


def calculate_system_correctness(predicted_action: str, predicted_intent: str, predicted_slots: dict[str, dict[str, str]], hallucination_detected: bool, policy_compliant: bool, current_domain: str) -> tuple[bool, str]:
    """
    Determine if the system responded appropriately to the user's input.

    System is correct if it chose the right action given:
    - User's intent
    - Available slot information
    - Policy constraints
    - No hallucinations

    Args:
        predicted_action: Action taken by system ("search", "book", "request", "inform")
        predicted_intent: User's intent ("find_hotel", "book_hotel", etc.)
        predicted_slots: Current slot state for the domain
        hallucination_detected: Whether system hallucinated
        policy_compliant: Whether system followed policies
        current_domain: The active domain for this turn

    Returns:
        (is_correct, reason): True if system behaved correctly, plus explanation
    """
    # Rule 1: System must not hallucinate
    if hallucination_detected:
        return False, "System hallucinated slot values"

    # Rule 2: System must follow policies
    if not policy_compliant:
        # Special case: If action is "request" and policy not compliant, system is correctly asking for missing info
        if predicted_action == "request":
            return True, "System correctly requested missing policy-required slots"
        return False, "System violated policy"

    # Rule 3: Match action to the intent and slot completeness
    domain_slots = predicted_slots.get(current_domain, {})

    # For booking intents
    if "book" in predicted_intent or predicted_action.startswith("book_"):
        domain_required = BOOKING_REQUIRED_SLOTS.get(f"book_{current_domain}", [])
        missing_slots = [s for s in domain_required if s not in domain_slots]

        if missing_slots:
            # Should request missing info
            if predicted_action == "request" or predicted_action in VALID_ACTION_TYPES:
                return True, "System correctly requested missing booking slots"
            else:
                return False, f"System should request missing slots: {missing_slots}"
        else:
            # All slots present, should book
            if predicted_action == "book" or predicted_action.startswith("book_"):
                return True, "System correctly completed booking with all slots"
            else:
                return False, "System had all slots but didn't book"

    # For search/find intents
    if "find" in predicted_intent or "search" in predicted_intent:
        if predicted_action in ["search", "inform", "offer"] or predicted_action in VALID_ACTION_TYPES:
            return True, "System correctly handled search intent"
        else:
            return False, f"System used wrong action for search: {predicted_action}"

    # For general info intents
    if "inform" in predicted_intent or predicted_intent == "none":
        if predicted_action in ["inform", "request"]:
            return True, "System correctly handled info exchange"
        else:
            return False, f"Unexpected action for info intent: {predicted_action}"

    # Default: if we reach here, accept the action as reasonable
    return True, "System action reasonable for intent"


def calculate_task_success(turn_results: list[dict[str, Any]], ground_truth_goal: dict[str, Any]) -> tuple[bool, str]:
    """
    Determine if the dialogue successfully completed the user's goal.

    Task success requires:
    - All required domains were addressed
    - For booking intents: All required slots filled AND booking action occurred
    - For info intents: Search/inform action occurred

    Args:
        turn_results: List of turn metric dicts with 'domain', 'action', 'accumulated_slots'
        ground_truth_goal: User's goal, format: {"domains": ["hotel"], "requires_booking": True}

    Returns:
        (is_successful, reason): True if task completed, plus explanation
    """
    required_domains = ground_truth_goal.get("domains", [])
    requires_booking = ground_truth_goal.get("requires_booking", False)

    # Collect domains and check final state
    addressed_domains = set()
    booking_action_occurred = False
    info_provided = False

    # Get final accumulated slots from last turn
    final_slots = {}
    if turn_results:
        final_slots = turn_results[-1].get("predicted_slots", {})

    for turn in turn_results:
        domain = turn.get("domain", "")
        action = turn.get("action", "")

        # Track which domains were addressed (skip 'none' domain)
        if domain and domain != "none":
            addressed_domains.add(domain)

        # Check for booking actions
        if action in ["book", "reserve"] or "book" in action:
            booking_action_occurred = True

        # Check for info actions
        if action in ["search", "inform", "offer", "request"]:
            info_provided = True

    # Validate all domains were addressed
    missing_domains = set(required_domains) - addressed_domains
    if missing_domains:
        return False, f"Missing domains: {missing_domains}"

    # Validate booking requirement
    if requires_booking:
        # Check if booking action occurred
        if not booking_action_occurred:
            return False, "Booking was required but no booking action occurred"

        # Check if all required slots are filled
        required_slots = {
            "hotel": ["name", "bookday", "bookpeople", "bookstay"],
            "restaurant": ["name", "bookday", "bookpeople", "booktime"]
        }

        for domain in required_domains:
            if domain in required_slots:
                domain_slots = final_slots.get(domain, {})
                missing_slots = [s for s in required_slots[domain] if s not in domain_slots]
                if missing_slots:
                    return False, f"Booking required but missing slots in {domain}: {missing_slots}"

    # Validate information requirement
    if not requires_booking and not info_provided:
        return False, "Information was required but not provided"

    return True, "All goals achieved"







