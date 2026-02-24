"""
Utility Functions and System Prompts.

This module provides:
- Agent prompts (Triage, Action, Supervisor, Judge) with versioning
- Mega-prompt for single-agent baseline (Experiment 1)
- Policy rules for constraint validation
- Cost calculation for API calls
- Dialogue formatting helpers for prompt construction
- JSON response parsing for single-agent baseline
"""

from .prompts import (
    DEFAULT_TRIAGE_PROMPT,
    DEFAULT_ACTION_PROMPT,
    DEFAULT_SUPERVISOR_PROMPT,
    DEFAULT_JUDGE_PROMPT,
    MEGA_PROMPTS,
    DEFAULT_MEGA_PROMPT,
)
from .utils import (
    search_multiwoz_datasets,
    calculate_cost,
    print_separator,
    save_graph_image,
    format_dialogue_history,
    format_policy_rules,
    parse_model_json_response,
    capture_and_save,
    MODEL_PRICING,
)

__all__ = [
    # Prompts
    "DEFAULT_TRIAGE_PROMPT",
    "DEFAULT_ACTION_PROMPT",
    "DEFAULT_SUPERVISOR_PROMPT",
    "DEFAULT_JUDGE_PROMPT",
    "MEGA_PROMPTS",
    "DEFAULT_MEGA_PROMPT",

    # Utility functions
    "search_multiwoz_datasets",
    "calculate_cost",
    "print_separator",
    "save_graph_image",
    "format_dialogue_history",
    "format_policy_rules",
    "parse_model_json_response",
    "capture_and_save",

    # Constants
    "MODEL_PRICING",
]

