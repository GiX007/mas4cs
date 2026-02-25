"""
Memory Agent: Conversation history management.

Updates the shared conversation history after each Action agent response, making the current turn visible to the Supervisor.

WHY THIS EXISTS:
- accumulated_history (in helpers.py) = memory BETWEEN turns
  Each turn is a separate workflow.invoke() call. After each call finishes, the workflow forgets everything. So we manually save history outside:

      # After turn 2 finishes:
      accumulated_history = final_state["conversation_history"].copy()
      # Before turn 3 starts:
      state["conversation_history"] = accumulated_history.copy()

  Without this → at Turn 3, agents don't know what happened in Turn 1 and 2.

- Memory agent = memory WITHIN a turn
  Triage → Policy → Action → Memory → Supervisor
  After Action generates a response, Memory appends it to history, so Supervisor can see what was just said in this turn.

  Without this → Supervisor cannot see what Action just said.

CURRENT IMPLEMENTATION:
  Passive bookkeeping only — appends user message and agent response to conversation history. No LLM call, no reasoning.

FUTURE EXTENSIONS:
  The Memory agent is kept as a separate node to allow future reasoning extensions without refactoring the rest of the architecture:

  1. Selective memory — analyze history and drop irrelevant turns to prevent context window overflow in long dialogues.
  2. Cross-domain summarization — when switching domains (e.g. restaurant to hotel), compress previous domain conversation into one summary line instead of carrying the full exchange forward.
  3. Explicit slot inheritance — reason about which slots from one domain should carry over to another (e.g. bookpeople from restaurant booking applies to hotel booking).
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
