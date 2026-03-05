"""Several tests to understand the Tomiinek evaluator."""
import json

from src.evaluation import build_tomiinek_input
from src.utils import print_separator, capture_and_save
from mwzeval.metrics import Evaluator


def dummy_tomiinek_input() -> dict[str, list[dict]]:
    """
    Build a minimal dummy input matching the Tomiinek evaluator format (https://github.com/Tomiinek/MultiWOZ_Evaluation/tree/master?tab=readme-ov-file).

    The evaluator expects:
        - Keys: dialogue ids in lowercase, no .json suffix (e.g. 'mul0001')
        - Values: list of turn dicts, one per SYSTEM turn
        - Each turn dict must have 'response' (the system's delexicalized reply), 'state' and 'active_domains' are optional

    Returns:
        Dict matching the Tomiinek evaluator's expected input format
    """
    fake_predictions = {
        "mul0001": [
            {
                "response": "I found a [restaurant_name] in the [restaurant_area] area.",
                "state": {
                    "restaurant": {
                        "area": "centre",
                        "food": "chinese"
                    }
                },
                "active_domains": ["restaurant"]
            },
            {
                "response": "The phone number is [restaurant_phone].",
                "state": {
                    "restaurant": {
                        "area": "centre",
                        "food": "chinese"
                    }
                },
                "active_domains": ["restaurant"]
            }
        ]
    }
    # Inspect the structure
    print("\nDummy Tomiinek input structure:")
    for dialogue_id, turns in fake_predictions.items():
        print(f"Dialogue ID: {dialogue_id}")
        for i, turn in enumerate(turns):
            print(f"  Turn {i}: response: {turn['response']} | state: {turn['state']} | active_domains: {turn['active_domains']}")


def test_tomiinek_adapter() -> None:
    """Load a minimal real-shaped sample and print before/after the adapter."""
    # Minimal sample with only the fields the adapter actually reads
    sample_dialogue_results = [
        {
            "dialogue_id": "MUL1271.json",
            "turn_metrics": [
                {
                    "system_response": "I found a nice restaurant in the centre.",
                    "predicted_slots": {"restaurant": {"area": "centre", "food": "chinese"}},
                    "domain": "restaurant",
                },
                {
                    "system_response": "The phone number is 01223 323737.",
                    "predicted_slots": {"restaurant": {"area": "centre", "food": "chinese"}},
                    "domain": "restaurant",
                },
            ],
        }
    ]

    print("\nBefore tomiinek adapter (MAS4CS format):")
    for d in sample_dialogue_results:
        print(f"dialogue_id : {d['dialogue_id']}")
        for i, t in enumerate(d["turn_metrics"]):
            print(f"  Turn {i}: response: {t['system_response']} | predicted_slots: {t['predicted_slots']} | domain: {t['domain']}")

    result = build_tomiinek_input(sample_dialogue_results)

    print("\nAfter tomiinek adapter (Tomiinek format)")
    for dialogue_id, turns in result.items():
        print(f"dialogue_id : {dialogue_id}")
        for i, t in enumerate(turns):
            print(f"  turn {i}: response: {t['response']} | state: {t['state']} | active_domains: {t['active_domains']}")


def test_tomiinek_evaluator_single_output() -> None:
    """Call the real Tomiinek evaluator on a minimal dummy input and print raw output."""
    sample = {
        "mul1271": [
            {
                "response": "I found a nice restaurant in the centre.",
                "state": {"restaurant": {"area": "centre", "food": "chinese"}},
                "active_domains": ["restaurant"],
            },
            {
                "response": "The phone number is 01223 323737.",
                "state": {"restaurant": {"area": "centre", "food": "chinese"}},
                "active_domains": ["restaurant"],
            },
        ]
    }
    print(f"\nRunning Tomiinek Evaluator on sample input:\n {sample}")

    e = Evaluator(bleu=True, success=True, richness=False)
    results = e.evaluate(sample)

    print("\nTomiinek Evaluator raw output:\n")
    print(json.dumps(results, indent=2))


def test_tomiinek_evaluator_two_outputs() -> None:
    """Call the real Tomiinek evaluator on two dummy dialogues and print raw output."""
    sample = {
        "mul1271": [
            {
                "response": "I found a nice restaurant in the centre.",
                "state": {"restaurant": {"area": "centre", "food": "chinese"}},
                "active_domains": ["restaurant"],
            },
            {
                "response": "The phone number is 01223 323737.",
                "state": {"restaurant": {"area": "centre", "food": "chinese"}},
                "active_domains": ["restaurant"],
            },
        ],
        "sng0551": [
            {
                "response": "I found a hotel in the north with free parking.",
                "state": {"hotel": {"area": "north", "parking": "yes"}},
                "active_domains": ["hotel"],
            },
            {
                "response": "I have booked it for you. Your reference is ABC123.",
                "state": {"hotel": {"area": "north", "parking": "yes", "book stay": "2"}},
                "active_domains": ["hotel"],
            },
        ],
    }

    print(f"\nRunning Tomiinek Evaluator on two-dialogue sample:\n {sample}")

    e = Evaluator(bleu=True, success=True, richness=False)
    results = e.evaluate(sample)

    print("\nTomiinek Evaluator raw output:\n")
    print(json.dumps(results, indent=2))


def run_all_tests():
    """Run all tests."""
    print_separator("TEST TOMIINEK ADAPTER")
    dummy_tomiinek_input()
    test_tomiinek_adapter()
    test_tomiinek_evaluator_single_output()
    test_tomiinek_evaluator_two_outputs()
    print_separator("END TEST TOMIINEK ADAPTER")

if __name__ == "__main__":
    capture_and_save(
        func=run_all_tests,
        output_path="docs/evals_inspection/tomiinek_evaluator_tests.txt"
    )
