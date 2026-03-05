"""
Triage Agent: Intent detection and domain routing.

Entry point of the workflow. Analyzes user messages to identify the active domain (hotel/restaurant), intent (find/book), and extract initial slot values.
"""
from itertools import accumulate

from src.core import AgentState
from src.models import call_model
from src.utils import DEFAULT_TRIAGE_PROMPT, format_agent_history
from src.data import normalize_slot_value, INFORMABLE_SLOTS


def triage_agent(state: AgentState) -> AgentState:
    """
    Identify the active domain, intent, and extract slot values from user message.

    This is the entry point of the agent workflow.
    Extracts: current_domain, active_intent, slots_values

    Args:
        state: Current agent state with user_message

    Returns:
        Updated state with current_domain, active_intent, and slots_values populated
    """
    user_message = state["user_utterance"]
    services = state["services"]
    accumulated_slots = state["slots_values"]

    # Format accumulated slots for prompt readability
    belief_state_str = ""
    for domain, slots in accumulated_slots.items():
        if slots:  # only show domains with actual slots
            slots_str = ", ".join(f"{k}={v}" for k, v in slots.items())
            belief_state_str += f"{domain}: {slots_str}\n"
    if not belief_state_str:
        belief_state_str = "none"

    # history = state.get("conversation_history", [])

    # Check what history agent actually sees
    # print(f"\nState BEFORE Triage agent: turn {state['turn_id']} | current_domain= {state.get('current_domain')} | active_intent= {state.get('active_intent')} | slots_values= {state.get('slots_values')} | belief state: {belief_state_str}")
    # print(f"  conversation_history length: {len(history)}")
    # for i, msg in enumerate(history):
    #     print(f"  [{i}] {msg['role']}: {msg['content'][:60]}...")
    # print(f"  user_utterance: {state['user_utterance'][:60]}")

    # Read model from config, fallback to default
    model_name = state.get("model_config", {}).get("triage", "gpt-4o-mini")

    # Get and format prompt
    prompt = DEFAULT_TRIAGE_PROMPT.format(
        user_message=user_message,
        services=', '.join(services),
        history=format_agent_history(state["conversation_history"]),
        accumulated_slots=belief_state_str,
    )
    # print(f"\nTriage prompt: {prompt}")

    # Generate response
    response = call_model(model_name=model_name, prompt=prompt)
    # print(f"\nTriage LLM response:\n {response} | text: {response.text}")

    # Update state
    state["turn_cost"] += response.cost
    state["turn_response_time"] += response.response_time

    lines = response.text.strip().split('\n')
    for line in lines:
        if line.startswith("DOMAIN:"):
            state["current_domain"] = line.split("DOMAIN:")[1].strip()  # e.g., "restaurant"
        elif line.startswith("INTENT:"):
            intent = line.split("INTENT:")[1].strip()

            # Combine intent with domain to match VALID_INTENTS format
            domain = state.get("current_domain", "")
            if domain and intent:
                state["active_intent"] = f"{intent}_{domain}"  # e.g., "find_restaurant"
            else:
                state["active_intent"] = intent  # Fallback
        elif line.startswith("SLOTS:"):
            slots_str = line.split("SLOTS:")[1].strip()
            if slots_str.lower() != "none":  # {} when no slots extracted
                # Parse "key1=value1, key2=value2" format
                domain = state["current_domain"]
                if domain:  # if the LLM fails to identify a domain -> returns nothing

                    # Create domain key if it doesn't exist (handles LLM identifying domains not pre-initialized in state or outside original services list)
                    if domain not in state["slots_values"]:
                        state["slots_values"][domain] = {}

                    # Add slots
                    for pair in slots_str.split(','):
                        if '=' in pair:
                            key, value = pair.strip().split('=', 1)
                            key = key.strip()
                            value = value.strip()

                            # Skip unresolved placeholder values like <same day from history>
                            if value.startswith("<") and value.endswith(">"):
                                continue

                            # Normalize value using shared function
                            normalized_value = normalize_slot_value(value)

                            # Skip dontcare values (cleaner than explicit dontcare)
                            if normalized_value in ("dontcare", "none", "not mentioned"):
                                continue

                            state["slots_values"][domain][key] = normalized_value

            break  # stop after first complete DOMAIN/INTENT/SLOTS block

    # 1. Resolve co-references: if value contains "same", look up from accumulated belief state
    domain = state["current_domain"]
    if domain and domain in state["slots_values"]:
        for key, value in list(state["slots_values"][domain].items()):
            if "same" in str(value).lower():
                for acc_domain, acc_slots in accumulated_slots.items():
                    if key in acc_slots:
                        state["slots_values"][domain][key] = acc_slots[key]
                        break

    # 2. Name fallback: if booking intent but name missing, inherit from accumulated belief state
    domain = state["current_domain"]
    intent = state.get("active_intent", "")
    if intent and intent.startswith("book_") and domain:
        if "name" not in state["slots_values"].get(domain, {}):
            accumulated_name = accumulated_slots.get(domain, {}).get("name")
            if accumulated_name:
                state["slots_values"][domain]["name"] = accumulated_name

    # 3. Filter slots by intent to prevent LLM pre-extraction of booking slots during find_ turns
    domain = state["current_domain"]
    intent = state.get("active_intent", "")
    if domain and intent:
        if intent.startswith("find_"):
            state["slots_values"][domain] = {k: v for k, v in state["slots_values"].get(domain, {}).items() if k in INFORMABLE_SLOTS}

    # print(f"\nState AFTER Triage agent: turn {state['turn_id']} | domain= {state['current_domain']} | intent= {state['active_intent']} | slots= {state['slots_values'].get(state['current_domain'], {})}")
    return state
