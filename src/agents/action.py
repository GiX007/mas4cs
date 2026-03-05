"""
Action Agent: Response generation and user interaction.

Generates customer service responses based on detected intent, extracted slots, and policy constraints.
Integrates DB query tools for entity grounding.
Handles retry logic for self-correction.
"""
from src.core import AgentState, find_entity, book_entity
from src.models import call_model
from src.utils import DEFAULT_ACTION_PROMPT, format_agent_history


def action_agent(state: AgentState) -> AgentState:
    """
    Generate response to user based on domain, intent, slots, and policy violations.
    Calls DB tools when meaningful — find_entity for search, book_entity for booking.

    Args:
        state: Current agent state with all context

    Returns:
        Updated state
    """
    domain = state["current_domain"]
    intent = state["active_intent"]
    slots = state["slots_values"].get(domain, {})
    violations = state["policy_violations"]
    user_message = state["user_utterance"]
    model_name = state.get("model_config", {}).get("action", "gpt-4o-mini")
    state["attempt_count"] += 1  # For tracking self-correction attempts
    # history = state.get("conversation_history", [])

    # 1. Rule-based DB tool calls before generating response
    db_results = []
    booked_entity = None
    informed_entity = None

    # Prefix slot keys with domain for tools.py (e.g. "area" → "hotel-area")
    belief_state = {f"{domain}-{k}": v for k, v in slots.items()} if slots else {}

    # Check what history agent actually sees
    # print(f"\nState BEFORE Action agent: turn {state['turn_id']} | domain: {domain} | active intent: {intent} | slots: {slots} | belief state: {belief_state}| policy violations: {violations} | existing_booking={state.get('booked_entity')}")
    # print(f"  conversation_history length: {len(history)}")
    # for i, msg in enumerate(history):
    #     print(f"  [{i}] {msg['role']}: {msg['content'][:60]}...")
    # print(f"  user_utterance: {state['user_utterance'][:60]}")

    if intent and intent.startswith("book_") and not violations:
        # If booking already succeeded on a previous attempt, reuse it — avoid double booking
        existing_booking = state.get("booked_entity")
        if existing_booking and existing_booking.get("success"):
            booked_entity = existing_booking
            informed_entity = existing_booking["entity"]
            db_results = [existing_booking["entity"]]
        else:
            booked_entity = book_entity(domain, belief_state)
            if booked_entity["success"]:
                informed_entity = booked_entity["entity"]
                db_results = [booked_entity["entity"]]

    elif intent and intent.startswith("find_") and slots:
        # Search intent + at least one slot constraint → query DB
        db_results = find_entity(domain, belief_state)
        if db_results:
            informed_entity = db_results[0]  # recommend first match

    # Update valid_entities from real DB results
    if db_results:
        state["valid_entities"] = [e["name"] for e in db_results if "name" in e]

    # After find_entity call, save recommended entity name to slots, so the next turn's booking can use it without asking the user again
    if intent and intent.startswith("find_") and db_results:
        recommended_name = db_results[0].get("name", "")
        if recommended_name:
            if domain not in state["slots_values"]:
                state["slots_values"][domain] = {}
            state["slots_values"][domain]["name"] = recommended_name.lower()

    # 2. Format context for LLM prompt
    slots_str = ", ".join([f"{k}={v}" for k, v in slots.items()]) if slots else "none"
    violations_str = "; ".join(violations) if violations else "none"

    # Ground the LLM with real entity info if available
    entity_str = ", ".join([f"{k}={v}" for k, v in informed_entity.items()]) if informed_entity else "none"

    # Booking reference if available
    ref_str = booked_entity["ref"] if booked_entity and booked_entity["success"] else "none"

    # Get the Supervisor's feedback if available for self-correction (only after first attempt)
    feedback = state.get("supervisor_feedback") or "none"

    prompt = DEFAULT_ACTION_PROMPT.format(
        domain=domain,
        intent=intent,
        slots=slots_str,
        violations=violations_str,
        user_message=user_message,
        history=format_agent_history(state["conversation_history"]),
        entity=entity_str,
        ref=ref_str,
        supervisor_feedback=feedback,
    )
    # print(f"\nAction prompt: {prompt}")

    # 3. Generate response
    response = call_model(model_name=model_name, prompt=prompt)
    # print(f"\nAction LLM response:\n {response} | text: {response.text}")

    # Determine action taken and dialogue acts
    action_taken = determine_action_type(intent, violations)
    dialogue_acts = map_action_to_dialogue_acts(action_taken, domain)

    # 4. Update state
    state["agent_response"] = response.text.strip()
    state["action_taken"] = action_taken
    state["dialogue_acts"] = dialogue_acts
    state["db_results"] = db_results
    state["booked_entity"] = booked_entity
    state["informed_entity"] = informed_entity
    state["turn_cost"] += response.cost
    state["turn_response_time"] += response.response_time

    # print(f"\nState AFTER Action agent: turn {state['turn_id']} | action_taken: {state['action_taken']} | dialogue_acts: {state['dialogue_acts']} | db_results: {state['db_results']} | booked_entity: {state['booked_entity']} | informed_entity: {state['informed_entity']} | state['slots_values']: {state['slots_values']} | policy violations: {state['policy_violations']} | agent_response: {state['agent_response'][:80] if state['agent_response'] else None}")
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
    if violations:  # If there are policy violations, we're requesting missing info
        return "request"
    if intent.startswith("book_"):  # If intent is booking and no violations, we're booking
        return "book"  # e.g. "book_hotel", "book_restaurant" → matches BOOKING_REQUIRED_SLOTS
    if intent.startswith("find_"):  # If intent is find/search, we're searching
        return "search"
    return "inform"  # Default: providing information


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

    mapping = {
        "search": [f"{domain_cap}-Inform"],
        "book": ["Booking-Book"],
        "request": [f"{domain_cap}-Request"],
        "inform": [f"{domain_cap}-Inform"],
        "recommend": [f"{domain_cap}-Recommend"],
        "no_offer": [f"{domain_cap}-NoOffer"],
    }

    return mapping.get(action_taken, [f"{domain_cap}-Inform"])
