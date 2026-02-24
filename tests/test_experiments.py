"""Unit tests of modules of experiments directory."""

from src.utils import print_separator


def test_run_single_agent_turn() -> None:
    """
    Test run_single_agent_turn returns correct structure.

    To inspect internals: add temporary prints inside run_single_agent_turn()
    after each step (format prompt, call model, parse response).
    Remove prints when done.
    """
    import json
    from src.experiments import run_single_agent_turn
    from src.utils import parse_model_json_response

    result = run_single_agent_turn(
        user_message="I need a cheap hotel in the centre",
        services=["hotel", "restaurant"],
        dialogue_history=[],
        model_name="gpt-4o-mini"
    )

    # Verify structure
    assert result is not None
    required_keys = ["domain", "intent", "slots", "action_type", "policy_satisfied", "response", "input_tokens", "output_tokens", "cost", "response_time"]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"

    # Verify types
    assert isinstance(result["domain"], str)
    assert isinstance(result["intent"], str)
    assert isinstance(result["slots"], dict)
    assert isinstance(result["policy_satisfied"], bool)
    assert isinstance(result["response"], str)
    assert isinstance(result["cost"], float)

    print_separator("TEST run_single_agent_turn")
    print(f"\nExact output of run_single_agent_turn:\n {result}")
    print("\nStructure:")
    print(f"  domain: {result['domain']}")
    print(f"  intent: {result['intent']}")
    print(f"  slots: {result['slots']}")
    print(f"  action_type: {result['action_type']}")
    print(f"  policy_satisfied: {result['policy_satisfied']}")
    print(f"  response: {result['response'][:80]}...")
    print(f"  cost: ${result['cost']:.6f}")
    print(f"  response_time: {result['response_time']:.2f}s")
    print("\ntest_run_single_agent_turn works")

    # JSON failure simulation
    print_separator("TEST run_single_agent_turn ON JSON FAILURE SIMULATION")
    invalid = "Sorry, I cannot help with that."
    print(f"\nResponse: {invalid}")
    try:
        parse_model_json_response(invalid)  # ← this SHOULD fail
        assert False, "Should have raised JSONDecodeError"  # ← if it doesn't fail, we force a test failure
    except json.JSONDecodeError:
        print("\nJSON parse failed as expected (invalid input)")
        pass


def test_run_single_agent_dialogue() -> None:
    """
    Test that run_single_agent_dialogue returns correct structure.
    Uses first validation dialogue with gpt-4o-mini.
    """
    from src.data import load_split_data
    from src.experiments import run_single_agent_dialogue

    # Load first validation dialogue
    dialogues = load_split_data("dataset/mw22_filtered.json", "validation")
    dialogue = dialogues[0]

    print_separator("TEST run_single_dialogue")
    # print(f"\nTesting dialogue:\n {dialogue}")
    print(f"\nDialogue id: {dialogue['dialogue_id']}")
    print(f"Services: {dialogue['services']}")
    print(f"Turns: {len(dialogue['turns'])}")

    result = run_single_agent_dialogue(dialogue=dialogue, model_name="gpt-4o-mini")
    print(f"\nExact output of run_single_agent_dialogue:\n {result}")

    # Verify
    assert result is not None
    assert "task_success" in result
    assert "avg_intent_accuracy" in result
    assert "avg_slot_accuracy" in result
    assert "avg_jga" in result
    assert "num_turns" in result

    print("\nStructure:")
    print(f"   Turns evaluated: {result['num_turns']}")
    print(f"   Task success: {result['task_success']}")
    print(f"   Intent accuracy: {result['avg_intent_accuracy']:.2%}")
    print(f"   Slot accuracy: {result['avg_slot_accuracy']:.2%}")
    print(f"   JGA: {result['avg_jga']:.2%}")
    print("\ntest_run_single_agent_dialogue works")


def run_tests(test_keys: list[str]) -> None:
    """
    Run selected experiment tests by name.

    Args:
        test_keys: List of test name strings

    CLI usage:
        python -m tests.test_experiments single_turn
        python -m tests.test_experiments  # runs all
    """
    test_map = {
        "single_turn": test_run_single_agent_turn,
        "single_dialogue": test_run_single_agent_dialogue,
    }

    for key in test_keys:
        if key in test_map:
            test_map[key]()
        else:
            print(f"Unknown test '{key}'. Choose from: {list(test_map.keys())}")


if __name__ == "__main__":
    import sys

    keys = sys.argv[1:] if len(sys.argv) > 1 else ["single_turn", "single_dialogue"]
    run_tests(keys)

