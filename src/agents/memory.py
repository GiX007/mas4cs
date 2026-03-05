"""
Memory Agent: Conversation history management.

Saves each turn's user message and agent response to conversation_history so the next turn's agents (Triage, Action) have full dialogue context.

WHY: Each turn is a separate workflow.invoke() call — LangGraph forgets everything after each call. helpers.py carries conversation_history forward between turns manually.
Without this agent, every turn starts with empty history and agents cannot resolve references like "same group" or "same day".
"""
from src.core import AgentState


def memory_agent(state: AgentState) -> AgentState:
    """
    Save current turn's user message and agent response to conversation_history.

    Why: user_utterance and agent_response in state are overwritten each turn.
    Memory agent saves them before they are lost, so the next turn's Triage and Action agents have full dialogue context.
    Only saves after Supervisor approves (validation_passed=True) or when max retries are exhausted (attempt_count >= 2) — whichever comes first.

    Args:
        state: Current agent state with user_utterance and agent_response

    Returns:
        Updated state with conversation_history appended
    """
    # Check what history agent actually sees
    # history_before = state.get("conversation_history", [])
    # print(f"\nState history BEFORE Memory agent: turn id= {state['turn_id']} | history_length= {len(history_before)}")

    # Only append on first attempt — skip on retries to prevent duplicates
    if state["validation_passed"] or state["attempt_count"] >= 2:
        state["conversation_history"].append({
            "role": "user",
            "content": state["user_utterance"]
        })
        if state["agent_response"]:
            state["conversation_history"].append({
                "role": "assistant",
                "content": state["agent_response"]
            })

    # print(f"\nState history AFTER Memory agent: turn {state['turn_id']} | attempt={state['attempt_count']} | history_length= {len(state['conversation_history'])}")
    return state
