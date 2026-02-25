"""
Supervisor Agent: Quality control and hallucination detection.

Validates agent responses against database entities to prevent fabricated information. Uses LLM for fuzzy matching that handles name variations.
"""

from src.core import AgentState
from src.models import call_model
from src.utils import DEFAULT_SUPERVISOR_PROMPT, format_agent_history


def supervisor_agent(state: AgentState) -> AgentState:
    """
    Validate agent response against database to detect hallucinations.
    (Uses LLM to handle name variations and partial matches that regex cannot)

    Args:
        state: Current agent state with agent_response

    Returns:
        Updated state with validation_passed and hallucination_flags set
    """
    # Check what history agents actually see
    # history = state.get("conversation_history", [])
    # print(f"\n[DEBUG supervisor_agent] Turn {state['turn_id']}")
    # print(f"  conversation_history length: {len(history)}")
    # for i, msg in enumerate(history):
    #     print(f"  [{i}] {msg['role']}: {msg['content'][:60]}...")

    # print(f"\n[DEBUG supervisor IN]  response={state['agent_response'][:60] if state['agent_response'] else None}")

    valid_entities = state.get("valid_entities", [])
    user_message = state["user_utterance"]
    agent_response = state["agent_response"]

    # Read model from config, fallback to default
    model_name = state.get("model_config", {}).get("supervisor", "gpt-4o-mini")

    # Normalize entities to lowercase for case-insensitive comparison and format valid entities for prompt
    normalized_entities = [e.lower() for e in valid_entities]
    entities_str = ", ".join(valid_entities) if normalized_entities else "none"

    prompt = DEFAULT_SUPERVISOR_PROMPT.format(
        user_message=user_message,
        agent_response=agent_response,
        valid_entities=entities_str,
        history=format_agent_history(state["conversation_history"])
    )

    # Generate response
    response = call_model(model_name=model_name, prompt=prompt)

    # Update state (Parse response)
    state["hallucination_flags"] = []
    state["validation_passed"] = True  # Assume valid unless proven otherwise

    state["turn_cost"] += response.cost
    state["turn_response_time"] += response.response_time

    lines = response.text.strip().split('\n')
    for line in lines:
        if line.startswith("HALLUCINATION:"):
            has_hallucination = line.split("HALLUCINATION:")[1].strip().lower()
            if has_hallucination == "yes":
                state["validation_passed"] = False
        elif line.startswith("ENTITIES:"):
            entities_str = line.split("ENTITIES:")[1].strip()
            if entities_str.lower() != "none":
                # Parse comma-separated entities and normalize to lowercase
                hallucinated = [e.strip().lower() for e in entities_str.split(',')]
                state["hallucination_flags"].extend(hallucinated)

    # print(f"\n[DEBUG supervisor OUT/PARSED]: validation_passed={state['validation_passed']} | flags={state['hallucination_flags']}")

    return state
