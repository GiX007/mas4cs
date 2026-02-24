"""
Object-oriented agent implementations for production deployment.

Refactored from function-based agents.
"""

from src.core import AgentState
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Defines common interface and shared functionality.
    All agents inherit from this and implement execute().
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name

    @abstractmethod
    def execute(self, state: AgentState) -> AgentState:
        """Process state and return updated state."""
        pass


class TriageAgent(BaseAgent):
    """Extracts domain, intent, and slots from user message."""

    def execute(self, state: AgentState) -> AgentState:
        pass


class PolicyAgent(BaseAgent):
    """Validates required slots (rule-based, no LLM)."""

    def execute(self, state: AgentState) -> AgentState:
        pass


class ActionAgent(BaseAgent):
    """Generates user-facing response."""

    def execute(self, state: AgentState) -> AgentState:
        pass


class MemoryAgent(BaseAgent):
    """Updates conversation history (rule-based, no LLM)."""

    def execute(self, state: AgentState) -> AgentState:
        pass


class SupervisorAgent(BaseAgent):
    """Validates response for hallucinations."""

    def execute(self, state: AgentState) -> AgentState:
        pass


