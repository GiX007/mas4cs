"""
Feature extraction utilities for MultiWOZ 2.2 (official GitHub format).

Functions to extract ground truth features from official dialogue turns.
Used by experiment helpers for evaluation.
"""
from typing import Any


def extract_gt_intent(turn: dict[str, Any], services: list[str]) -> str:
    """
    Extract ground truth intent from a USER turn's frames.

    Iterates over frames list to find the first frame whose service matches an active domain and has a non-empty active_intent.

    Args:
        turn: Single turn dict from official GitHub MultiWOZ 2.2
        services: List of active domains (e.g. ['hotel', 'restaurant'])

    Returns:
        Ground truth intent string (e.g. 'find_restaurant') or '' if not found
    """
    for frame in turn.get("frames", []):
        service = frame.get("service", "")
        if service not in services:
            continue

        intent = frame.get("state", {}).get("active_intent", "")

        # NONE means no intent this turn (e.g. closing turns)
        if intent and intent != "NONE":
            return intent

    return ""


def extract_gt_slots(frames: list[dict]) -> dict[str, dict[str, str]]:
    """
    Extract and normalize slots from official GitHub MultiWOZ 2.2 frame annotations.

    Slot values in GitHub format are lists — we take the first value.
    Domain prefixes are stripped so both predicted and GT are comparable.

    Example:
        Input:  [{"service": "hotel", "state": {"slot_values": {"hotel-area": ["north"]}}}]
        Output: {"hotel": {"area": "north"}}

    Args:
        frames: Turn-level frames list from official GitHub MultiWOZ 2.2 turn

    Returns:
        Nested dict by domain with domain prefixes stripped from slot names
    """
    result = {}

    for frame in frames:
        service = frame.get("service", "")
        slot_values = frame.get("state", {}).get("slot_values", {})

        if not slot_values:
            continue

        domain_slots = {}
        for slot_name, slot_value_list in slot_values.items():
            # Strip domain prefix: "restaurant-area" → "area"
            slot = slot_name.split("-", 1)[1] if "-" in slot_name else slot_name

            # slot_values are lists in GitHub format — take first value
            value = slot_value_list[0] if slot_value_list else ""
            if value:
                domain_slots[slot] = value.lower()

        if domain_slots:
            result[service] = domain_slots

    return result


def extract_dialogue_acts(turn: dict[str, Any], services: list[str]) -> list[str]:
    """
    Extract dialogue act types from a turn's attached dialog_act annotation.

    Dialog acts come from dialog_acts.json attached during load_split().
    Only returns act types relevant to target services.

    Args:
        turn: Single turn dict with 'dialog_act' key attached
        services: List of active domains (e.g. ['hotel', 'restaurant'])

    Returns:
        List of act type strings (e.g. ['Restaurant-Inform', 'Booking-Book'])
    """
    dialog_act = turn.get("dialog_act", {})
    if not dialog_act:
        return []

    acts = []
    for act_type in dialog_act.keys():
        # act_type format: "Restaurant-Inform", "Hotel-Request", "Booking-Book"
        # keep acts relevant to our target services + booking acts
        act_lower = act_type.lower()
        is_relevant = (
            any(s in act_lower for s in services)
            or act_lower.startswith("booking")
            or act_lower.startswith("general")
        )
        if is_relevant and act_type not in acts:
            acts.append(act_type)

    return acts


def extract_booking(turns: list[dict], services: list[str]) -> bool:
    """
    Check if any USER turn in the dialogue has a booking intent.

    Replaces the inline generator in helpers.py that read frames as dict.

    Args:
        turns: All turns from a dialogue
        services: List of active domains

    Returns:
        True if any USER turn contains a book_ intent
    """
    for turn in turns:
        if turn.get("speaker") != "USER":
            continue
        intent = extract_gt_intent(turn, services)
        if intent.startswith("book_"):
            return True
    return False
