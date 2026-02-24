"""
Policy Agent: Business rule validation and constraint enforcement.

Validates that required slots are present before executing actions (e.g., bookings require 'people', 'day', 'time').
Rule-based logic—no LLM call required.
"""

from src.core import AgentState
from src.data import BOOKING_REQUIRED_SLOTS


def policy_agent(state: AgentState) -> AgentState:
    """
    Validate that required slots are present for the current intent (Rule-based validation - no LLM call needed).

    Checks policy rules and populates policy_violations if slots are missing.

    Args:
        state: Current agent state with active_intent and slots_values

    Returns:
        Updated state with policy_violations populated
    """
    active_intent = state["active_intent"]
    current_domain = state["current_domain"]

    # Clear previous violations because it may be called multiple times in the retry loop/self-correction mechanism
    state["policy_violations"] = []

    # Check if intent requires policy validation
    if active_intent in BOOKING_REQUIRED_SLOTS:
        required_slots = BOOKING_REQUIRED_SLOTS[active_intent]
        current_slots = state["slots_values"].get(current_domain, {})

        # Check for missing required slots
        for required_slot in required_slots:
            if required_slot not in current_slots:
                violation = f"Missing required slot: {required_slot}"
                state["policy_violations"].append(violation)

    return state

