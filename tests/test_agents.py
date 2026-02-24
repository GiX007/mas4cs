"""Unit tests for the state and all individual agents."""
from src.core import AgentState, initialize_state
from src.agents import triage_agent, policy_agent, action_agent, memory_agent, supervisor_agent
from src.utils import print_separator
from pprint import pprint


def test_state_usage() -> None:
    """Test creating and inspecting AgentState."""

    # Use the simplest way to create a dummy state
    state: AgentState = {
        "dialogue_id": "PMUL4398.json",
        "turn_id": 0,
        "services": ["restaurant", "hotel"],
        "user_utterance": "i need a place to dine in the center that is expensive",
        "current_domain": "restaurant",
        "slots_values": {
            "restaurant": {"area": "centre", "pricerange": "expensive"},
            "hotel": {}
        },
        "conversation_history": [
            {"role": "user", "content": "i need a place to dine in the center that is expensive"}
        ],
        "agent_response": "",
    }

    # Inspect it
    print_separator("TEST 1: 'AgentState' TYPE & FIELDS")
    print(f"\nType: {type(state)}")
    print(f"Fields (keys): {list(state.keys())}")

    print_separator("FULL STATE")
    print()
    pprint(state, indent=2, sort_dicts=False)

    # Access fields like normal dict
    print_separator("ACCESSING FIELDS")
    print(f"\nCurrent domain: {state['current_domain']}")
    print(f"Restaurant slots: {state['slots_values']['restaurant']}")

    # Modify fields (this is how agents will update state)
    state["agent_response"] = "I found 5 expensive restaurants in the centre."
    print_separator("AFTER AGENT UPDATE")
    print(f"\nUpdate agent's response: {state['agent_response']}")
    print("\nFull State:\n")
    pprint(state, indent=2, sort_dicts=False)


def test_extended_state() -> None:
    """Test AgentState with policy and validation fields."""

    # Simplest way to create a state
    state: AgentState = {
        "dialogue_id": "PMUL4398.json",
        "turn_id": 1,
        "services": ["hotel"],
        "user_utterance": "book it for 3 nights",
        "current_domain": "hotel",
        "active_intent": "book_hotel",
        "slots_values": {"hotel": {"area": "centre"}},
        "conversation_history": [
            {"role": "user", "content": "find a hotel in the centre"},
            {"role": "assistant", "content": "I found Hotel XYZ."}
        ],
        "agent_response": "",
        "policy_violations": ["Missing required slot: stay duration"],
        "validation_passed": False,
        "hallucination_flags": [],
        "attempt_count": 0
    }

    print_separator("TEST 2: EXTENDED 'AgentState' FIELDS")
    print(f"\nActive Intent: {state['active_intent']}")
    print(f"Policy Violations: {state['policy_violations']}")
    print(f"Validation Passed: {state['validation_passed']}")
    print(f"Attempts Count: {state['attempt_count']}")

    print_separator("FULL EXTENDED STATE")
    print()
    pprint(state, indent=2, sort_dicts=False)

    # Simulate supervisor correction
    state["attempt_count"] += 1
    state["policy_violations"].append("Hallucinated hotel name")

    print_separator("AFTER SUPERVISOR FLAGS ISSUES")
    print(f"\nAttempt Count: {state['attempt_count']}")
    print(f"Policy Violations: {state['policy_violations']}")


def test_prompt_and_state_parsing() -> None:
    """Validate prompt formatting and structured LLM state parsing."""

    print_separator("TEST PROMPT FORMATING")
    from src.utils import DEFAULT_TRIAGE_PROMPT
    prompt = DEFAULT_TRIAGE_PROMPT.format(  # .format() replaces the variables inside prompt's {}
        user_message="Book a hotel",
        services=', '.join(["hotel", "taverna"])
    )
    print(f"Prompt: {prompt}\n")

    lines = prompt.strip().split("\n")
    print(f"Agent converts LLM response to lists[str] to process:\n{lines}")

    print_separator("TEST PARSED STATE")
    dummy_response = """DOMAIN: hotel
INTENT: find
SLOTS: area=centre, pricerange=cheap, internet=yes"""

    print(f"\nDummy LLM response:\n {dummy_response}")

    lines = dummy_response.strip().split('\n')
    print(f"\nAgent converts LLM response to lists[str] to process:\n{lines}")

    print("Parsed State:")
    for line in lines:
        if line.startswith("DOMAIN:"):
            current_domain = line.split("DOMAIN:")[1].strip()
            print(f"\nDomain: {current_domain}")
        elif line.startswith("INTENT:"):
            active_intent = line.split("INTENT:")[1].strip()
            print(f"Intent: {active_intent}")
        elif line.startswith("SLOTS"):
            slots_str = line.split("SLOTS:")[1].strip()
            # print(slots_str)
            print("Slots:")
            for pair in slots_str.split(','):
                if '=' in pair:
                    key, value = pair.strip().split('=', 1)
                    print(key, value)


def test_triage_agent() -> None:
    """Test triage agent with real LLM call."""

    # Initialize a dummy state
    state = initialize_state(dialogue_id="TEST01",turn_id=0, services=["hotel", "restaurant"], user_utterance="i need a cheap hotel in the centre")

    print_separator("TEST TRIAGE AGENT")
    print(f"\nAvailable services: {state['services']}")
    print(f"User message: {state['user_utterance']}")

    # Run triage
    updated_state = triage_agent(state)

    print("\nTriage agent's output:")
    print(f"Detected domain: {updated_state['current_domain']}")
    print(f"Detected intent: {updated_state['active_intent']}")
    print(f"Extracted slots: {updated_state['slots_values']}")


def test_policy_agent() -> None:
    """Test policy agent detects missing required slots."""

    # Simulate state after triage: user wants to book but missing slots (initial state)
    state = initialize_state(dialogue_id="TEST02", turn_id=0, services=["hotel"], user_utterance="book a hotel")

    # Manually set what triage would extract
    state["current_domain"] = "hotel"
    state["active_intent"] = "book_hotel"
    state["slots_values"]["hotel"] = {"area": "centre"}  # Missing 'stay' and 'people'

    print_separator("TEST POLICY AGENT")
    print("\nBefore policy check:")
    print(f"Intent: {state['active_intent']}")
    print(f"Current slots: {state['slots_values']['hotel']}")

    updated_state = policy_agent(state)

    print("\nAfter policy check:")
    print(f"Violations: {updated_state['policy_violations']}")


def test_action_agent_with_violations() -> None:
    """Test action agent asks for missing slots."""

    # Initialize a dummy state
    state = initialize_state(dialogue_id="TEST03", turn_id=0, services=["hotel"], user_utterance="book a hotel for me")

    state["current_domain"] = "hotel"
    state["active_intent"] = "book_hotel"
    state["slots_values"]["hotel"] = {"area": "centre"}
    state["policy_violations"] = ["Missing required slot: stay", "Missing required slot: people"]

    print_separator("TEST ACTION AGENT (WITH VIOLATIONS)")
    print(f"\nUser message: {state['user_utterance']}")
    print(f"Violations: {state['policy_violations']}")

    updated_state = action_agent(state)

    print("\nAction agent's output:")
    print(f"{updated_state['agent_response']}")


def test_memory_agent() -> None:
    """Test memory agent updates conversation history."""

    # Initialize a dummy state
    state = initialize_state(dialogue_id="TEST04", turn_id=0, services=["hotel"], user_utterance="i need a hotel")

    state["agent_response"] = "I can help you find a hotel. What area would you prefer?"

    print_separator("TEST MEMORY AGENT")
    print(f"\nInitial memory state : {len(state['conversation_history'])} messages")

    updated_state = memory_agent(state)

    print(f"\nMemory state after a dummy conversation: {len(updated_state['conversation_history'])} messages")
    for i, msg in enumerate(updated_state['conversation_history']):
        print(f"\n[{i}] {msg['role']}: {msg['content']}")


def test_supervisor_detects_hallucination() -> None:
    """Test supervisor flags hallucinated entity names."""

    # Initialize a dummy state
    state = initialize_state(dialogue_id="TEST05", turn_id=0, services=["hotel"], user_utterance="find me a hotel")

    state["agent_response"] = "I found the Grand Plaza Hotel for you in the centre."

    # Simulate database has different hotels
    valid_hotels = ["City Hotel", "Budget Inn", "Luxury Suites"]
    state["valid_entities"] = valid_hotels

    print_separator("TEST SUPERVISOR AGENT (HALLUCINATION)")
    print(f"\nAction agent's response: {state['agent_response']}")
    print(f"Valid hotels in DB: {valid_hotels}")

    updated_state = supervisor_agent(state)

    print("\nSupervisor agent's response:")
    print(f"Validation passed: {updated_state['validation_passed']}")
    print(f"Hallucinated entities: {updated_state['hallucination_flags']}")


def run_tests(test_keys: list[str]) -> None:
    """
    Run selected agent tests by name.

    Args:
        test_keys: List of test name strings

    CLI usage:
        python -m tests.test_agents triage policy
        python -m tests.test_agents  # runs all agent tests
    """
    test_map = {
        "state": test_state_usage,
        "extended_state": test_extended_state,
        "prompt_parsing": test_prompt_and_state_parsing,
        "triage": test_triage_agent,
        "policy": test_policy_agent,
        "action": test_action_agent_with_violations,
        "memory": test_memory_agent,
        "supervisor": test_supervisor_detects_hallucination,
    }

    for key in test_keys:
        if key in test_map:
            test_map[key]()
        else:
            print(f"Unknown test '{key}'. Choose from: {list(test_map.keys())}")


if __name__ == "__main__":
    import sys

    keys = sys.argv[1:] if len(sys.argv) > 1 else ["triage", "policy", "action", "memory", "supervisor"]
    run_tests(keys)

