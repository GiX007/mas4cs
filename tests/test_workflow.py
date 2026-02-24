"""Integration tests for the full Multi-Agent System with LangGraph."""

import json
from pathlib import Path

from src.core import initialize_state, create_workflow
from src.data import load_split_data
from src.utils import print_separator, save_graph_image


def test_workflow_creation() -> None:
    """Verify workflow compiles without errors."""
    workflow = create_workflow()
    print("Workflow structure created!")
    print(f"Type: {type(workflow)}")

    # Check workflow was created
    assert workflow is not None
    print("\nWorkflow compiled successfully")

    # Verify it's callable/executable (structural sanity check)
    assert callable(workflow.invoke)
    print("Workflow has invoke method")


def test_workflow_single_turn(enable_retry: bool = True) -> None:
    """
    Test workflow with one real MultiWOZ turn.

    Args:
        enable_retry: If True, test with retry loop; if False, test linear flow
    """
    # Load sample dialogue
    data = load_split_data("dataset/mw22_filtered.json", "train")
    first_dialogue = data[0]
    first_turn = first_dialogue["turns"][0]

    # Initialize state
    initial_state = initialize_state(
        dialogue_id=first_dialogue["dialogue_id"],
        turn_id=first_turn["turn_id"],
        services=first_dialogue["services"],
        user_utterance=first_turn["utterance"]
    )

    # Add model_config
    initial_state["model_config"] = {
        "triage": "gpt-4o-mini",
        "action": "gpt-4o-mini",
        "supervisor": "gpt-4o-mini"
    }

    # Run workflow with specified mode
    workflow = create_workflow(enable_retry=enable_retry)
    final_state = workflow.invoke(initial_state)

    # Verify all agents executed
    assert final_state["current_domain"] is not None, "Triage failed: domain not set"
    assert final_state["active_intent"] is not None, "Triage failed: intent not set"
    assert final_state["agent_response"] is not None, "Action failed: no response"
    assert len(final_state["conversation_history"]) > 0, "Memory failed: history empty"

    mode = "TEST CONDITIONAL RETRY WORKFLOW" if enable_retry else "TEST LINEAR WORKFLOW (NO RETRY)"
    filename = "linear_workflow.png" if not enable_retry else "conditional_workflow.png"

    # Save the graph as image
    save_graph_image(workflow, f"docs/images/{filename}")

    # Print results
    print_separator(mode)
    print(f"\nUser message: {final_state['user_utterance']}")
    print("\nMAS State:")
    print(f"Domain: {final_state['current_domain']}")
    print(f"Intent: {final_state['active_intent']}")
    print(f"Slots: {final_state['slots_values']}")
    print(f"MAS Response: {final_state['agent_response']}")
    print(f"Policy Violations: {final_state['policy_violations']}")
    print(f"Validation Passed: {final_state['validation_passed']}")
    print(f"Hallucinations: {final_state['hallucination_flags']}")
    print(f"Attempt Count: {final_state['attempt_count']}")
    print("\nAll agents executed successfully!")


def find_dialogue_with_entities() -> tuple[dict | None, dict | None, list[str]]:
    """
    Find a dialogue turn that mentions specific entities (for hallucination testing).

    Returns:
        Tuple of (dialogue_dict, turn_dict, list_of_valid_entities)
    """
    data_path = Path("dataset") / "mw22_filtered.json"
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Search for turns with entity mentions in span_info
    print_separator("SEARCH FOR ENTITIES")
    for i, dialogue in enumerate(data["train"][:50]):  # Check first 50 dialogues
        # print(f"i = {i}")
        for turn in dialogue["turns"]:
            if turn["speaker"] == "USER" and turn.get("span_info"):
                # Extract entity names from span_info
                entities = []
                for span in turn["span_info"]:
                    # Look for 'name' slot which contains hotel/restaurant names
                    if span.get("slot") == "name":
                        entities.append(span.get("value"))

                if entities:
                    print(f"\nFound dialogue with entities:")
                    print(f"Dialogue ID: {dialogue['dialogue_id']}")
                    print(f"Turn {turn['turn_id']}: {turn['utterance']}")
                    print(f"Entities mentioned: {entities}")

                    return dialogue, turn, entities

    print("No suitable dialogue found with entity names in first 50 dialogues")
    return None, None, []


def test_workflow_with_entity_validation() -> None:
    """Test workflow with valid entities passed to supervisor for hallucination detection."""
    # Find dialogue with entity mentions
    dialogue, turn, valid_entities = find_dialogue_with_entities()

    if not dialogue or not valid_entities:
        print("Skipping test - no suitable dialogue found")
        return

    # Initialize state
    initial_state = initialize_state(
        dialogue_id=dialogue["dialogue_id"],
        turn_id=turn["turn_id"],
        services=dialogue["services"],
        user_utterance=turn["utterance"]
    )

    # Add model_config
    initial_state["model_config"] = {
        "triage": "gpt-4o-mini",
        "action": "gpt-4o-mini",
        "supervisor": "gpt-4o-mini"
    }

    # Run workflow with retry enabled
    workflow = create_workflow(enable_retry=True)
    final_state = workflow.invoke(initial_state)

    # Verify all agents executed
    assert final_state["current_domain"] is not None, "Triage failed: domain not set"
    assert final_state["active_intent"] is not None, "Triage failed: intent not set"
    assert final_state["agent_response"] is not None, "Action failed: no response"
    assert len(final_state["conversation_history"]) > 0, "Memory failed: history empty"

    # Print results
    print_separator("TEST WORKFLOW WITH ENTITY MENTIONED")
    print(f"\nUser message: {final_state['user_utterance']}")
    print("\nMAS State:")
    print(f"Valid Entities: {valid_entities}")
    print(f"MAS Response: {final_state['agent_response']}")
    print(f"Validation Passed: {final_state['validation_passed']}")
    print(f"Hallucinations Detected: {final_state['hallucination_flags']}")
    print(f"Attempt Count: {final_state['attempt_count']}")
    print("\nAll agents executed successfully!")


if __name__ == "__main__":
    # test_workflow_creation()

    # Test 1: Linear workflow
    test_workflow_single_turn(enable_retry=False)

    # Test 2: Conditional retry workflow
    test_workflow_single_turn(enable_retry=True)

    # Test 3: Entity validation
    test_workflow_with_entity_validation()

