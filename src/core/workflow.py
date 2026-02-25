"""
Workflow Orchestration: LangGraph-based multi-agent pipeline.

Defines the execution graph connecting all agents (triage → policy → action → memory → supervisor) with optional retry loop for self-correction.
"""

from src.core import AgentState
from src.agents import triage_agent, policy_agent, action_agent, memory_agent, supervisor_agent
from langgraph.graph import StateGraph, END


def should_retry(state: AgentState) -> str:
    """
    Routing function: decide if we retry Action or end workflow.

    Args:
        state: Current agent state

    Returns:
        "action" if retry needed, "end" if validation passed
    """
    # Check if validation failed and we haven't exceeded retry limit
    if not state["validation_passed"] and state["attempt_count"] < 2:  # Allows maximum 2 total attempts (initial + 1 retry)
        # print(f"\n[DEBUG retry] RETRYING | attempt={state['attempt_count']} | flags={state['hallucination_flags']}")
        return "action"
    # print(f"\n[DEBUG retry] ENDING | attempt={state['attempt_count']} | validation_passed={state['validation_passed']}")
    return "end"


def create_workflow(enable_retry: bool = True) -> StateGraph:
    """
    Build the MAS4CS workflow graph.

    Args:
        enable_retry: If True, add conditional retry loop; if False, simple linear flow

    Returns:
        Compiled LangGraph workflow
    """
    # Initialize graph with our state type
    workflow = StateGraph(AgentState)

    # Add agent nodes
    workflow.add_node("triage", triage_agent)
    workflow.add_node("policy", policy_agent)
    workflow.add_node("action", action_agent)
    workflow.add_node("memory", memory_agent)
    workflow.add_node("supervisor", supervisor_agent)

    # Linear flow through first 5 agents
    workflow.set_entry_point("triage")
    workflow.add_edge("triage", "policy")
    workflow.add_edge("policy", "action")
    workflow.add_edge("action", "memory")
    workflow.add_edge("memory", "supervisor")

    # Conditional or direct ending
    if enable_retry:
        # Add retry loop: supervisor can route back to action
        workflow.add_conditional_edges(
            "supervisor",
            should_retry,
            {
                "action": "action",  # Loop back for retry
                "end": END  # Finish workflow
            }
        )
    else:
        # Simple linear: supervisor goes directly to END
        workflow.add_edge("supervisor", END)

    compiled_graph = workflow.compile()

    return compiled_graph

