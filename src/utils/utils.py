"""Utility functions for MAS4CS."""

import json
from pathlib import Path
from typing import Any

import io
import contextlib

from huggingface_hub import list_datasets
from langgraph.graph.state import CompiledGraph

# Pricing per 1M tokens (input, output) in USD (Feb 2026)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o-mini": (0.150, 0.600),

    # Anthropic
    "claude-3-haiku-20240307": (0.25, 1.25),
}


def search_multiwoz_datasets(verbose: bool = False) -> None:
    """
    Search HuggingFace Hub for MultiWOZ datasets.

    Args:
        verbose: If True, print loading messages (Default: False)
    """
    results = [d.id for d in list_datasets(search="multiwoz")]

    if verbose:
        print("Available MultiWOZ datasets:")
        for r in results[:10]:
            print(f"  - {r}")


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate API call cost in USD.

    Args:
        model_name: Model identifier
        input_tokens: Input token count
        output_tokens: Output token count

    Returns:
        Total cost in USD (0.0 for free models)
    """
    # Free models
    if model_name.startswith("unsloth/"):
        return 0.0

    # Paid models
    if model_name not in MODEL_PRICING:
        return 0.0  # Unknown model, return 0

    input_price, output_price = MODEL_PRICING[model_name]
    return (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price


def print_separator(message: str) -> None:
    """
    Print a consistent separator line for readable test output.

    Args:
        message: Title to display between separator lines
    """
    print("\n" + "=" * 60)
    print(message)
    print("=" * 60)


def save_graph_image(workflow: CompiledGraph, output_path: str) -> None:
    """
    Save LangGraph workflow graph as PNG image.

    Args:
        workflow: Compiled LangGraph workflow
        output_path: Path to save the PNG file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    png_bytes = workflow.get_graph().draw_mermaid_png()
    Path(output_path).write_bytes(png_bytes)


def format_dialogue_history(turns: list[dict[str, Any]]) -> str:
    """
    Format processed dialogue turns into readable history string for prompt.

    Args:
        turns: List of processed turn dicts with 'speaker' and 'utterance' keys

    Returns:
        Formatted history string ready for prompt injection.
        Empty string if no turns provided.
    """
    if not turns:
        return ""

    history_text = "**DIALOGUE HISTORY:**\n"
    for turn in turns:
        speaker = turn["speaker"]  # "USER" or "SYSTEM"
        utterance = turn["utterance"]
        history_text += f"{speaker}: {utterance}\n"

    return history_text


def format_policy_rules(policy_dict: dict[str, list[str]]) -> str:
    """
    Format policy rules dictionary into readable string for prompt.

    Args:
        policy_dict: Booking requirements e.g. {"book_hotel": ["name", "bookday", ...]}

    Returns:
        Formatted policy string ready for prompt injection.
    """
    policy_text = "**POLICY RULES:**\n"
    for action, required_slots in policy_dict.items():
        policy_text += f"- {action}: requires {', '.join(required_slots)}\n"

    return policy_text


def parse_model_json_response(response_text: str) -> dict[str, Any]:
    """
    Parse model's JSON response into Python dictionary.

    The model outputs structured text, but it's still a raw string.
    We need a Python dict to feed individual fields to the evaluator.

    Example:
        '{"domain": "hotel", "intent": "find_hotel"}'  # str
        → {"domain": "hotel", "intent": "find_hotel"}  # dict → parsed["domain"] works

    Handles common formatting issues like Markdown code blocks that some models add around JSON output.

    Args:
        response_text: Raw text response from model

    Returns:
        Parsed dictionary with keys: domain, intent, slots, action_type, policy_satisfied, response

    Raises:
        json.JSONDecodeError: If response cannot be parsed as JSON
    """
    # Strip Markdown code blocks if model adds them
    cleaned = response_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    return json.loads(cleaned.strip())


def capture_and_save(func, output_path: str) -> None:
    """
    Run func(), capture everything it prints, save to file AND print to terminal.

    Args:
        func: A callable that prints output
        output_path: Path to save the output .txt file
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        func()

    output = buffer.getvalue()
    print(output)  # print to terminal

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(output, encoding='utf-8')
    print(f"Saved to: {output_path}")

