"""
Unit tests for src/utils/python_fundamentals.py helper functions.
"""

import json

from src.utils import format_dialogue_history, format_policy_rules, parse_model_json_response, DEFAULT_MEGA_PROMPT, print_separator


def test_format_dialogue_history() -> None:
    """Test format_dialogue_history with real dataset turn structure."""
    turns = [
        {
            "turn_id": 0,
            "speaker": "USER",
            "utterance": "i need a place to dine in the center",
            "frames": {"restaurant": {"active_intent": "find_restaurant"}},
            "dialogue_acts": ["Restaurant-Inform"]
        },
        {
            "turn_id": 1,
            "speaker": "SYSTEM",
            "utterance": "I have several options. Do you prefer African or British food?",
            "frames": {},
            "dialogue_acts": ["Restaurant-Select"]
        }
    ]

    result = format_dialogue_history(turns)
    assert "USER: i need a place to dine in the center" in result
    assert "SYSTEM: I have several options" in result
    assert result.startswith("**DIALOGUE HISTORY:**")
    print("test_format_dialogue_history passed")

    empty_result = format_dialogue_history([])
    assert empty_result == ""
    print("test_format_dialogue_history (empty) passed")


def test_format_policy_rules() -> None:
    """Test format_policy_rules with BOOKING_REQUIRED_SLOTS structure."""
    from src.data import BOOKING_REQUIRED_SLOTS

    result = format_policy_rules(BOOKING_REQUIRED_SLOTS)
    assert "book_hotel" in result
    assert "book_restaurant" in result
    assert "**POLICY RULES:**" in result
    print("test_format_policy_rules passed")


def test_parse_model_json_response() -> None:
    """Test JSON parsing with clean and markdown-wrapped responses."""
    # Test 1: Clean JSON
    clean_json = '{"domain": "hotel", "intent": "find_hotel", "slots": {"area": "centre"}, "action_type": "Hotel-Inform", "policy_satisfied": true, "response": "I found some hotels."}'
    result = parse_model_json_response(clean_json)
    assert result["domain"] == "hotel"
    assert result["slots"] == {"area": "centre"}
    print("test_parse_model_json_response (clean JSON) passed")

    # Test 2: Markdown-wrapped JSON
    markdown_json = '```json\n{"domain": "restaurant", "intent": "find_restaurant", "slots": {}, "action_type": "Restaurant-Request", "policy_satisfied": false, "response": "What area?"}\n```'
    result2 = parse_model_json_response(markdown_json)
    assert result2["domain"] == "restaurant"
    assert result2["policy_satisfied"] == False
    print("test_parse_model_json_response (markdown JSON) passed")

    # Test 3: Invalid JSON raises error
    try:
        parse_model_json_response("This is not JSON at all")
        assert False, "Should have raised JSONDecodeError"
    except json.JSONDecodeError:
        print("test_parse_model_json_response (invalid JSON) raises correctly")

def test_mega_prompt() -> None:
    """Inspect DEFAULT_MEGA_PROMPT template before and after filling."""
    from src.utils import DEFAULT_MEGA_PROMPT
    services_str = "hotel, restaurant"
    history_str = "**DIALOGUE HISTORY:**\nUSER: I need a restaurant\nASSISTANT: What area?\n"
    user_msg = "In the centre, please"
    policy_str = "**POLICY RULES:**\n- book_hotel: requires name, bookday, bookpeople, bookstay\n"

    print_separator("TEST MEGA PROMPT — TEMPLATE")
    print("\nUnfilled prompt:\n")
    print(DEFAULT_MEGA_PROMPT)

    filled_prompt = DEFAULT_MEGA_PROMPT.format(
        services=services_str,
        history_text=history_str,
        user_message=user_msg,
        policy_text=policy_str
    )

    print("\n\nFilled prompt:\n")
    print(filled_prompt)
    print(f"\nFilled prompt length: {len(filled_prompt)}")

    print_separator("END OF TEST MEGA PROMPT — TEMPLATE")


if __name__ == "__main__":
    import sys
    test_map = {
        "history": test_format_dialogue_history,
        "policy": test_format_policy_rules,
        "json": test_parse_model_json_response,
        "prompt": test_mega_prompt,
    }
    keys = sys.argv[1:] if len(sys.argv) > 1 else list(test_map.keys())
    for key in keys:
        if key in test_map:
            test_map[key]()
        else:
            print(f"Unknown test '{key}'. Choose from: {list(test_map.keys())}")

