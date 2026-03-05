"""
Agent state definition for MAS4CS workflow.

Defines AgentState TypedDict structure that flows through the LangGraph workflow.
All agents read from and write to this shared state.
"""
from typing import TypedDict


class AgentState(TypedDict):
    """
    Shared state passed between agents in the workflow.
    Each agent reads from and writes to this state.
    This represents the complete context for processing one user turn.
    """
    # Core identifiers (dialogue metadata)
    dialogue_id: str
    turn_id: int
    services: list[str]  # e.g., ["restaurant", "hotel"]

    # Current turn (User) input message
    user_utterance: str

    # Domain & slot tracking (Triage outputs)
    current_domain: str | None  # Which service is currently active
    active_intent: str | None  # e.g., "find_restaurant", "book_hotel"

    # Slot tracking, e.g., {"restaurant": {"area": "centre"}}
    slots_values: dict[str, dict[str, str]]  # Accumulated across turns

    # Conversation state
    conversation_history: list[dict[str, str]]  # e.g., [{"role": "user", "content": "..."}]

    # Agent response
    agent_response: str | None
    action_taken: str  # e.g., "search", "book", "request", "inform"
    dialogue_acts: list[str]  # e.g., ["Restaurant-Inform", "Booking-Book"]

    # Policy validation
    policy_violations: list[str]  # e.g., ["Missing required slot: duration"]

    # Supervisor validation
    validation_passed: bool  # Did Supervisor approve the response?
    valid_entities: list[str]  # Valid entity names from database for hallucination detection
    supervisor_feedback: str | None  # Correction instructions for Action retry

    # DB query results (populated by Action agent via tools.py)
    db_results: list[dict]  # Matching entities returned by find_entity
    booked_entity: dict | None  # Booking confirmation returned by book_entity
    informed_entity: dict | None  # The specific entity the agent chose to recommend (First in DB results or None if no recommendation)

    # Retry control
    attempt_count: int  # Self-refinement mechanism

    # Model configuration (for heterogeneous experiments)
    model_config: dict[str, str]  # Maps agent name -> model name

    # Latency and cost tracking
    turn_cost: float  # Total API cost for this turn (all agents)
    turn_response_time: float  # Total response time for this turn (all agents)


def initialize_state(dialogue_id: str, turn_id: int, services: list[str], user_utterance: str,) -> AgentState:
    """
    Create a fresh AgentState for a new dialogue turn.

    Args:
        dialogue_id: Unique dialogue identifier
        turn_id: Turn number in the dialogue
        user_utterance: User's current utterance
        services: List of services involved (e.g., ["restaurant", "hotel"])

    Returns:
        AgentState with all fields initialized to safe defaults
    """
    return {
        "dialogue_id": dialogue_id,
        "turn_id": turn_id,
        "services": services,
        "user_utterance": user_utterance,
        "current_domain": None,
        "active_intent": None,
        "slots_values": {},
        "conversation_history": [],
        "agent_response": None,
        "action_taken": "",
        "dialogue_acts": [],
        "policy_violations": [],
        "validation_passed": False,
        "valid_entities": [],
        "supervisor_feedback": None,
        "db_results": [],
        "booked_entity": None,
        "informed_entity": None,
        "attempt_count": 0,
        "model_config": {},
        "turn_cost": 0.0,
        "turn_response_time": 0.0,
    }
