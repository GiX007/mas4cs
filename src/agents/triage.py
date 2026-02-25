"""
Triage Agent: Intent detection and domain routing.

Entry point of the workflow. Analyzes user messages to identify the active domain (hotel/restaurant), intent (find/book), and extract initial slot values.
"""

from src.core import AgentState
from src.models import call_model
from src.utils import DEFAULT_TRIAGE_PROMPT, format_agent_history
from src.data import normalize_slot_value


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
    # Check what history agents actually see
    # history = state.get("conversation_history", [])
    # print(f"\n[DEBUG triage_agent] Turn {state['turn_id']}")
    # print(f"  conversation_history length: {len(history)}")
    # for i, msg in enumerate(history):
    #     print(f"  [{i}] {msg['role']}: {msg['content'][:60]}...")

    user_message = state["user_utterance"]
    services = state["services"]

    # Read model from config, fallback to default
    model_name = state.get("model_config", {}).get("triage", "gpt-4o-mini")

    # Get and format prompt
    prompt = DEFAULT_TRIAGE_PROMPT.format(
        user_message=user_message,
        services=', '.join(services),
        history=format_agent_history(state["conversation_history"])
    )

    # Generate response
    response = call_model(model_name=model_name, prompt=prompt)
    # print(f"User message: {user_message}")
    # print(f"LLM response: {response}")
    # print(f"LLM response text: {response.text}")

    # Update state (Parse response)
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

                    # Create domain key if it doesn't exist
                    # (handles LLM identifying domains not pre-initialized in state or outside original services list)
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
                            state["slots_values"][domain][key] = normalized_value

            break  # stop after first complete DOMAIN/INTENT/SLOTS block

    # print(f"[DEBUG triage RAW OUTPUT]: {response.text[:200]}")
    # print(f"[DEBUG triage PARSED]: domain={state['current_domain']} | intent={state['active_intent']} | slots={state['slots_values'].get(state['current_domain'], {})}")

    return state

