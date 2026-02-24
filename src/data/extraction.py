"""
Feature extraction utilities for MultiWOZ dataset.

Functions to extract ground truth features from preprocessed dialogue turns.
Used by experiment helpers for evaluation.
"""

from typing import Any


def extract_ground_truth_intent(turn: dict[str, Any], services: list[str]) -> str:
    """
    Extract ground truth intent from a USER turn's frames.

    Iterates over active services to find the first non-empty active_intent.
    Only looks in domains listed in `services` — ensures we stay within the
    dialogue's target domains and ignore any out-of-scope frame keys.
    Returns empty string for closing turns (e.g. 'goodbye') with no intent.

    Args:
        turn: Single turn dict from MultiWOZ dialogue
        services: List of active domains (e.g. ['hotel', 'restaurant'])

    Returns:
        Ground truth intent string (e.g. 'find_restaurant') or '' if not found
    """
    for domain in services:
        if domain in turn["frames"] and turn["frames"][domain]["active_intent"]:
            return turn["frames"][domain]["active_intent"]
    return ""


def extract_slots_from_frames(frames: dict[str, dict]) -> dict[str, dict[str, str]]:
    """
    Extract and normalize slots from MultiWOZ frame annotations.

    MultiWOZ stores slots with domain prefixes (e.g. 'hotel-area'), but our
    system uses prefix-free names (e.g. 'area'). This function strips prefixes
    from ground truth so both sides are comparable.

    Only looks in domains listed in `services` — ensures we stay within the
    dialogue's target domains and ignore any out-of-scope frame keys.

    Example:
        Input:  {"hotel": {"slots_values": {"hotel-area": "north", "hotel-bookday": "monday"}}}
        Output: {"hotel": {"area": "north", "bookday": "monday"}}

    Args:
        frames: Turn-level frames dict from preprocessed MultiWOZ dialogue

    Returns:
        Nested dict by domain with domain prefixes stripped from slot names
    """
    result = {}

    for domain, frame_data in frames.items():
        slots_values = frame_data.get("slots_values", {})

        if not slots_values:
            continue

        domain_slots = {}
        for slot_name, slot_value in slots_values.items():
            # "restaurant-area" → "area", "hotel-pricerange" → "pricerange"
            slot = slot_name.split("-", 1)[1] if "-" in slot_name else slot_name
            domain_slots[slot] = slot_value.lower() if isinstance(slot_value, str) else slot_value  # normalize case

        if domain_slots:
            result[domain] = domain_slots

    return result

def normalize_slot_value(value: str) -> str:
    """
    Normalize slot value to match MultiWOZ format.

    Args:
        value: Slot value to normalize

    Returns:
        Normalized value (e.g., "center" → "centre"), always lowercased
    """
    from src.data import SLOT_VALUE_NORMALIZATION

    if not value:
        return value

    # Lowercase first, then check normalization map
    value_str = str(value)  # handle integers e.g. bookpeople: 3
    value_lower = value_str.lower()
    return SLOT_VALUE_NORMALIZATION.get(value_lower, value_lower)  # always return lowercase

