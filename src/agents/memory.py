"""
Memory Agent: Dialogue state and conversation history management.

Maintains conversation context across turns by tracking user-agent exchanges and preserving slot values. No LLM call required—pure state management.
"""

from src.core import AgentState


def memory_agent(state: AgentState) -> AgentState:
    """
    Update conversation history and maintain slot state across turns.
    (Simple state management - no LLM call needed)

    Adds current turn to history and ensures slots persist.

    Args:
        state: Current agent state

    Returns:
        Updated state with conversation_history appended
    """
    # Add user message to history
    state["conversation_history"].append({
        "role": "user",
        "content": state["user_utterance"]
    })

    # Add agent response to history
    if state["agent_response"]:  # in the workflow, memory agent might be called before action agenet generates a response
        state["conversation_history"].append({
            "role": "assistant",
            "content": state["agent_response"]
        })

    return state

