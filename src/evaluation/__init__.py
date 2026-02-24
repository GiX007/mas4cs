"""
Evaluation framework for MAS4CS.

Provides:
- Objective metrics (JGA, slot accuracy, hallucination rate, policy compliance, etc.)
- LLM-as-judge infrastructure (prompt creation, response parsing)
- Hierarchical evaluators (turn-level, dialogue-level, dataset-level)

Not included:
- Utility demonstrations (sets, nested dicts, function vs class)
"""

from .evaluation_metrics import (
    calculate_intent_accuracy,
    calculate_action_type_accuracy,
    calculate_slot_accuracy,
    calculate_jga,
    calculate_hallucination_rate,
    calculate_policy_compliance,
    calculate_task_success,
    calculate_system_correctness,
    calculate_domain_accuracy,
    calculate_memory_transfer_accuracy,
    calculate_set_based_accuracy,
)
from .llm_judge import (
    create_judge_prompt,
    parse_judge_response,
    gpt4_judge,
    claude_judge,
    fake_judge
)
from .evaluator import (
    DialogueEvaluator,
    DatasetEvaluator,
)

__all__ = [
    # Evaluation metrics
    "calculate_intent_accuracy",
    "calculate_action_type_accuracy",
    "calculate_slot_accuracy",
    "calculate_jga",
    "calculate_hallucination_rate",
    "calculate_policy_compliance",
    "calculate_task_success",
    "calculate_system_correctness",
    "calculate_domain_accuracy",
    "calculate_memory_transfer_accuracy",
    "calculate_set_based_accuracy",

    # LLM judge
    "create_judge_prompt",
    "parse_judge_response",
    "gpt4_judge",
    "claude_judge",
    "fake_judge",

    # Evaluators
    "DialogueEvaluator",
    "DatasetEvaluator",

]


