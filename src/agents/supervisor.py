"""
Supervisor Agent: LLM-based quality control and reflection.

Acts as an internal judge that checks both Triage and Action outputs.
Catches errors from either agent and writes targeted correction feedback for retry.

WHY LLM and not rule-based: Rule-based string matching cannot handle natural language paraphrases ("check-in day" = bookday), reference resolution failures ("same day" not resolved by Triage), or semantic correctness.
An LLM judge understands meaning, not just exact strings.

WHAT IT CHECKS each turn:
1. Triage correctness: domain, intent, slots
2. Policy correctness: were the right slots flagged as missing?
3. Action correctness: did response handle violations, booking ref, no-offer correctly?

GRAPH POSITION: Triage → Policy → Action → Supervisor → Memory
                                     ↑            |
                                     |__(retry)___|

If VALID=no AND attempt_count < MAX_RETRIES → routes back to Action with feedback.
Memory only saves after VALID=yes OR max retries exhausted.
"""
from src.core import AgentState
from src.models import call_model
from src.utils import DEFAULT_SUPERVISOR_PROMPT, format_agent_history

MAX_RETRIES = 2


def supervisor_agent(state: AgentState) -> AgentState:
    """
    Validate Triage and Action outputs using LLM reasoning.

    Reads from state: current_domain, active_intent, slots_values, policy_violations, db_results, booked_entity, agent_response, valid_entities
    Writes to state: validation_passed, supervisor_feedback

    Args:
        state: Current agent state after Action agent has run

    Returns:
        Updated state with validation_passed and supervisor_feedback set
    """
    domain = state["current_domain"]
    intent = state["active_intent"]
    slots = state["slots_values"].get(domain, {})
    violations = state["policy_violations"]
    db_results = state.get("db_results", [])
    booked = state.get("booked_entity")
    user_message = state["user_utterance"]
    agent_response = state["agent_response"] or ""
    model_name = state.get("model_config", {}).get("supervisor", "gpt-4o-mini")

    # Check what history agent actually sees
    # print(f"\nState BEFORE Supervisor agent: turn {state['turn_id']} | current domain: {domain} | active intent: {intent} | slots: {slots} | policy violations: {violations} | booked_entity: {booked} | db_results: {db_results} | agent_response: {agent_response} | valid_entities: {state.get('valid_entities')}")

    # Format db_results and booking ref for prompt
    if db_results:
        db_str = ", ".join([e.get("name", "") for e in db_results if "name" in e])
    else:
        db_str = "none"

    ref_str = booked["ref"] if booked and booked.get("success") else "none"
    slots_str = ", ".join([f"{k}={v}" for k, v in slots.items()]) if slots else "none"
    violations_str = ", ".join(violations) if violations else "none"

    prompt = DEFAULT_SUPERVISOR_PROMPT.format(
        history=format_agent_history(state["conversation_history"]),
        user_message=user_message,
        domain=domain,
        intent=intent,
        slots=slots_str,
        violations=violations_str,
        db_results=db_str,
        ref=ref_str,
        agent_response=agent_response,
    )
    # print(f"\nSupervisor prompt: {prompt}")

    response = call_model(model_name=model_name, prompt=prompt)
    # print(f"\nSupervisor response:\n {response} | text: {response.text}")

    # Update cost and latency
    state["turn_cost"] += response.cost
    state["turn_response_time"] += response.response_time

    # Parse VALID and FEEDBACK from response
    state["validation_passed"] = True
    state["supervisor_feedback"] = None

    for line in response.text.strip().split("\n"):
        if line.startswith("VALID:"):
            valid_str = line.split("VALID:")[1].strip().lower()
            state["validation_passed"] = (valid_str == "yes")
        elif line.startswith("FEEDBACK:"):
            feedback = line.split("FEEDBACK:")[1].strip()
            if feedback.lower() != "none":
                state["supervisor_feedback"] = feedback

    # print(f"\nState AFTER Supervisor agent: turn {state['turn_id']} | validation_passed={state['validation_passed']} | attempt={state['attempt_count']} | supervisor_feedback={state['supervisor_feedback']}")
    return state
