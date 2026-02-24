"""Tests for data preprocessing functionality."""

from src.data import load_multiwoz, select_dialogue_sample, transform_dialogue, run_preprocessing_pipeline
from src.utils import print_separator


def test_slot_normalization() -> None:
    """
    Dummy test for slot normalization logic.

    Assumption:
    - If a slot contains multiple values in slots_values_list, only the first value is retained.
    """
    # Simulated raw MultiWOZ state structure
    raw_state = {
        "active_intent": "find_restaurant",
        "requested_slots": [],
        "slots_values": {
            "slots_values_name": [
                "restaurant-area",
                "restaurant-pricerange",
                "restaurant-food"
            ],
            "slots_values_list": [
                ["centre"],  # single value
                ["expensive"],  # single value
                ["African", "Asian"]  # multiple values
            ]
        }
    }

    # Normalization logic
    normalized_slots = {}

    slot_names = raw_state["slots_values"]["slots_values_name"]
    slot_values_lists = raw_state["slots_values"]["slots_values_list"]

    for name, values in zip(slot_names, slot_values_lists):
        if isinstance(values, list) and len(values) > 0:
            normalized_slots[name] = values[0]  # keep only first value
        else:
            normalized_slots[name] = None

    # Expected result
    expected_output = {
        "restaurant-area": "centre",
        "restaurant-pricerange": "expensive",
        "restaurant-food": "African"  # only first kept
    }

    print_separator("TEST SLOT NORMALIZATION")

    print("\nRaw slots:")
    print(raw_state)

    print("\nNormalized slots:")
    print(normalized_slots)

    print("\nExpected output:")
    print(expected_output)

    assert normalized_slots == expected_output, "Normalization failed"
    print_separator("END OF TEST SLOT NORMALIZATION")


def test_sample_transformation() -> None:
    """
    Inspect one dialogue sample before and after transformation.
    Run: python -m tests.test_preprocess
    """
    # Dummy test for slot normalization
    test_slot_normalization()

    # Load dataset
    dataset = load_multiwoz()

    # Select and print one sample (Default to 0)
    test_sample = select_dialogue_sample(dataset, verbose=True)

    # Transform and print processed sample
    transform_dialogue(test_sample, verbose=True)

    print("\nNotice how slots appear in turn 0 — before: parallel lists, after: flat dict keyed by slot name.\n")


if __name__ == "__main__":

    # Inspect a dialogue before and after processing
    test_sample_transformation()

    # Run Preprocess -> Pipeline(Transform + Domain Filtering)
    run_preprocessing_pipeline(load_multiwoz(), verbose=True)

