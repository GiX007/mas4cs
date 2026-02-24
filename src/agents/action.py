"""
Action Agent: Response generation and user interaction.

Generates customer service responses based on detected intent, extracted slots, and policy constraints. Handles retry logic for self-correction.
"""

from src.core import AgentState
from src.models import call_model
from src.utils import DEFAULT_ACTION_PROMPT


def action_agent(state: AgentState) -> AgentState:
    """
    Generate response to user based on domain, intent, slots, and policy violations.

    Args:
        state: Current agent state with all context

    Returns:
        Updated state with agent_response populated
    """
    domain = state["current_domain"]
    intent = state["active_intent"]
    slots = state["slots_values"].get(domain, {})
    violations = state["policy_violations"]
    user_message = state["user_utterance"]

    # Read model from config, fallback to default
    model_name = state.get("model_config", {}).get("action", "gpt-4o-mini")

    # Increment retry counter (tracks self-correction attempts)
    state["attempt_count"] += 1

    # Format slots and violations to str for prompt (LLMs work with text)
    slots_str = ", ".join([f"{k}={v}" for k, v in slots.items()]) if slots else "none"
    violations_str = "; ".join(violations) if violations else "none"

    prompt = DEFAULT_ACTION_PROMPT.format(
        domain=domain,
        intent=intent,
        slots=slots_str,
        violations=violations_str,
        user_message=user_message
    )

    # print(f"Prompt: {prompt}")
    # print(f"Slots BEFORE action: {state['slots_values']}")

    # Generate response
    response = call_model(model_name=model_name, prompt=prompt)

    # Determine action taken and dialogue acts
    action_taken = determine_action_type(intent, violations)
    dialogue_acts = map_action_to_dialogue_acts(action_taken, domain)

    # Update state
    state["agent_response"] = response.text.strip()
    state["action_taken"] = action_taken
    state["dialogue_acts"] = dialogue_acts

    # print(f"LLM response: {response}")
    # print(f"LLM response: {response.text}")
    # print(f"Slots AFTER action: {state['slots_values']}")

    return state


def determine_action_type(intent: str, violations: list[str]) -> str:
    """
    Determine what action the agent is taking based on intent and state.

    Args:
        intent: User's intent (e.g., "find_restaurant", "book_hotel")
        violations: List of policy violations

    Returns:
        Action type string (e.g., "search", "book", "request", "inform")
    """
    # If there are policy violations, we're requesting missing info
    if violations:
        return "request"

    # If intent is booking and no violations, we're booking
    if intent.startswith("book_"):
        return "book"  # e.g. "book_hotel", "book_restaurant" → matches BOOKING_REQUIRED_SLOTS

    # If intent is find/search, we're searching
    if intent.startswith("find_"):
        return "search"

    # Default: providing information
    return "inform"


def map_action_to_dialogue_acts(action_taken: str, domain: str) -> list[str]:
    """
    Map agent action to MultiWOZ dialogue act format.

    Args:
        action_taken: Action type (e.g., "search", "book", "request")
        domain: Current domain (e.g., "restaurant", "hotel")

    Returns:
        List of dialogue acts (e.g., ["Restaurant-Inform"])
    """
    domain_cap = domain.capitalize()  # "restaurant" → "Restaurant"

    if action_taken == "search":
        return [f"{domain_cap}-Inform"]

    elif action_taken == "book":
        return ["Booking-Book"]

    elif action_taken == "request":
        return [f"{domain_cap}-Request"]

    elif action_taken == "inform":
        return [f"{domain_cap}-Inform"]

    elif action_taken == "recommend":
        return [f"{domain_cap}-Recommend"]

    elif action_taken == "no_offer":
        return [f"{domain_cap}-NoOffer"]

    else:
        # Default fallback
        return [f"{domain_cap}-Inform"]

