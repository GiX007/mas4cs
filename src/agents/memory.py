"""
Memory Agent: Dialogue state and conversation history management.

Maintains conversation context across turns by tracking user-agent exchanges and preserving slot values. No LLM call required—pure state management.
"""

from src.core import AgentState


def memory_agent(state: AgentState) -> AgentState:
    """
    Update conversation history and maintain slot state across turns.
    (Simple state management - no LLM call needed)

    Appends current turn to history only on first attempt (attempt_count == 1)
    to prevent duplicate entries during supervisor retry loops.

    Args:
        state: Current agent state

    Returns:
        Updated state with conversation_history appended
    """
    # Check what history agents actually see
    # history_before = state.get("conversation_history", [])
    # print(f"\n[DEBUG memory_agent] Turn {state['turn_id']}")
    # print(f"  history BEFORE append: {len(history_before)} messages")

    # # Add user message to history
    # state["conversation_history"].append({
    #     "role": "user",
    #     "content": state["user_utterance"]
    # })
    #
    # # Add agent response to history
    # if state["agent_response"]:  # in the workflow, memory agent might be called before action agenet generates a response
    #     state["conversation_history"].append({
    #         "role": "assistant",
    #         "content": state["agent_response"]
    #     })

    if state["attempt_count"] == 1:
        state["conversation_history"].append({
            "role": "user",
            "content": state["user_utterance"]
        })

        if state["agent_response"]:
            state["conversation_history"].append({
                "role": "assistant",
                "content": state["agent_response"]
            })

    # print(f"[DEBUG memory AFTER append/PARSED]: history_length={len(state['conversation_history'])}")

    return state
