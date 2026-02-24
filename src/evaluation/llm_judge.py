"""
LLM-as-judge evaluation infrastructure.

Provides rubric-based scoring (1-5 scale) for response quality assessment.
Uses prompts from src/utils/prompts.py for consistency.
"""

import json
from typing import Any
from src.models import call_model
from src.utils import DEFAULT_JUDGE_PROMPT


def create_judge_prompt(user_message: str, system_response: str, ground_truth_slots: dict[str, dict[str, str]], policy_rules: list[str]) -> str:
    """
    Create a prompt for LLM judge to evaluate response quality.

    Args:
        user_message: What the user said this turn
        system_response: System's response to evaluate
        ground_truth_slots: Expected slots from annotations
        policy_rules: Relevant policy constraints

    Returns:
        Formatted prompt string for the judge LLM
    """
    return DEFAULT_JUDGE_PROMPT.format(
        user_message=user_message,
        system_response=system_response,
        ground_truth_slots=json.dumps(ground_truth_slots, indent=2),
        policy_rules=json.dumps(policy_rules, indent=2)
    )


def parse_judge_response(response_text: str) -> dict[str, Any]:
    """
    Parse LLM judge response into structured format.

    LLMs often wrap JSON in Markdown fences or add extra text despite instructions.
    This function extracts the JSON, validates required fields, and ensures the score
    is in the valid 1-5 range. Returns an error dict instead of crashing if parsing fails.

    Args:
        response_text: Raw text response from judge LLM

    Returns:
        Dictionary with score and feedback, or error dict if parsing fails
    """
    try:
        # Try to extract JSON from response (handle Markdown code blocks)
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_str = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            json_str = response_text[json_start:json_end].strip()
        else:
            json_str = response_text.strip()

        result = json.loads(json_str)

        # Validate required fields
        if "score" not in result:
            return {"score": 0, "error": "Missing 'score' field in judge response"}

        # Ensure score is in valid range
        score = float(result["score"])
        if not (1 <= score <= 5):
            return {"score": 0, "error": f"Score {score} out of valid range [1-5]"}

        return result

    except json.JSONDecodeError as e:
        return {"score": 0, "error": f"JSON parsing failed: {str(e)}"}
    except Exception as e:
        return {"score": 0, "error": f"Unexpected error: {str(e)}"}


def gpt4_judge(temperature: float = 0.2):
    """Create GPT-4o-mini judge function for response quality evaluation."""
    return lambda prompt: call_model(model_name="gpt-4o-mini", prompt=prompt, temperature=temperature).text


def claude_judge(temperature: float = 0.2):
    """Create Claude Opus judge function for response quality evaluation."""
    return lambda prompt: call_model(model_name="claude-3-haiku-20240307", prompt=prompt, temperature=temperature).text


def fake_judge(score: int = 5):
    """Create fake judge for testing without API calls."""
    return lambda p: f'{{"score": {score}, "correctness": "Test"}}'
