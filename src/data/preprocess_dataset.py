"""
Dataset Preprocessing: MultiWOZ 2.2 transformation pipeline.

Transforms raw MultiWOZ format into simplified turn-by-turn structure for MAS4CS.
Includes filtering (hotel/restaurant only), caching, and documentation generation.
"""

import os
import json
from typing import Any
from collections import defaultdict

from src.data import load_split_data
from src.utils import print_separator


def select_dialogue_sample(dataset_dict: dict[str, Any], split: str = "train", index: int = 0, verbose: bool = False) -> dict[str, Any]:
    """
    Select one dialogue sample from the dataset for inspection.

    Args:
        dataset_dict: Loaded MultiWOZ dataset from load_multiwoz()
        split: Dataset split ("train", "validation", or "test")
        index: Index of dialogue to select
        verbose: Whether or not to print the raw sample (Default: False)

    Returns:
        Single dialogue dictionary with all its raw fields
    """
    dialogues = dataset_dict[split]
    dialogue_sample = dialogues[index]

    if verbose:
        print_separator("RAW MULTIWOZ 2.2 SAMPLE")
        print(json.dumps(dialogue_sample, indent=2))
        print("=" * 60)

    return dialogue_sample


def transform_dialogue(raw_sample: dict[str, Any], verbose: bool = False) -> dict[str, Any]:
    """
    Transform raw MultiWOZ sample into MAS4CS format.

    MultiWOZ stores turns as parallel lists — turn_ids[0], speakers[0], utterances[0]
    all belong to turn 0 but are stored separately. This function zips them into
    turn-by-turn dictionaries, which is the format every agent will work with.

    Args:
        raw_sample: Raw dialogue from MultiWOZ 2.2
        verbose: If True, print the processed sample (Default: False)

    Returns:
        Processed dialogue with simplified turn-by-turn structure

    Example:
        Input:  {"turns": {"turn_id": ["0", "1"], "speaker": [0, 1], ...}}  # parallel lists
        Output: {"turns": [{"turn_id": 0, "speaker": "USER", ...}, {...}]}  # list of dicts
    """
    dialogue_id = raw_sample["dialogue_id"]
    services = raw_sample["services"]

    # Extract parallel lists from raw structure
    turn_ids = raw_sample["turns"]["turn_id"]
    speakers = raw_sample["turns"]["speaker"]
    utterances = raw_sample["turns"]["utterance"]
    frames_list = raw_sample["turns"]["frames"]
    dialogue_acts_list = raw_sample["turns"]["dialogue_acts"]

    # Build turn-by-turn structure
    turns = []
    for i in range(len(turn_ids)):
        # Extract dialogue types, acts for this turn
        act_types = dialogue_acts_list[i]["dialog_act"]["act_type"]
        act_slots_list = dialogue_acts_list[i]["dialog_act"]["act_slots"]

        # Flatten act_slots into list of {slot, value} dicts
        dialogue_act_slots = []
        for act_slot_dict in act_slots_list:
            slot_names = act_slot_dict["slot_name"]
            slot_values = act_slot_dict["slot_value"]
            for name, value in zip(slot_names, slot_values):
                dialogue_act_slots.append({"slot": name, "value": value})

        # Extract span_info (positions in utterance)
        span_info_data = dialogue_acts_list[i]["span_info"]
        span_info = []
        act_slot_names = span_info_data["act_slot_name"]
        act_slot_values = span_info_data["act_slot_value"]
        for name, value in zip(act_slot_names, act_slot_values):
            span_info.append({"slot": name, "value": value})

        # Extract all frames, organized by service
        frame = frames_list[i]
        frames_by_service = {}

        if frame.get("service") and frame.get("state"):
            for idx, service in enumerate(frame["service"]):
                state = frame["state"][idx]

                # Normalize slots_values from parallel lists to dict
                slots_values = {}
                slots_data = state.get("slots_values", {})
                slot_names = slots_data.get("slots_values_name", [])
                slot_value_lists = slots_data.get("slots_values_list", [])
                for name, value_list in zip(slot_names, slot_value_lists):
                    # ASSUMPTION: When a slot has multiple values in slots_values_list, we take only the first value
                    slots_values[name] = value_list[0] if value_list else None

                # Store frame data per service
                frames_by_service[service] = {
                    "active_intent": state.get("active_intent"),
                    "requested_slots": state.get("requested_slots", []),
                    "slots_values": slots_values
                }

        turn = {
            "turn_id": int(turn_ids[i]) if turn_ids[i].isdigit() else turn_ids[i],
            "speaker": "USER" if speakers[i] == 0 else "SYSTEM",
            "utterance": utterances[i],
            "frames": frames_by_service,
            "dialogue_acts": act_types,
            "dialogue_act_slots": dialogue_act_slots,
            "span_info": span_info
        }
        turns.append(turn)

    processed_sample = {
        "dialogue_id": dialogue_id,
        "services": services,
        "turns": turns
    }

    if verbose:
        print_separator("PROCESSED SAMPLE (COMPLETE - ALL FRAMES)")
        print(json.dumps(processed_sample, indent=2))
        print_separator("END OF PROCESSED SAMPLE (COMPLETE - ALL FRAMES)")

    return processed_sample


def transform_dataset(mw22_dataset: dict[str, Any], force_reprocess: bool = False, verbose: bool = False) -> dict[str, list]:
    """
    Process all dialogues across all splits with caching.
    Checks if processed dataset exists; if yes, loads it; if no, processes and saves.

    Args:
        mw22_dataset: Loaded MultiWOZ dataset from load_multiwoz()
        force_reprocess: If True, ignore cached version and reprocess
        verbose: If True, print pipeline messages (Default: False)

    Returns:
        Dictionary with processed dialogues per split:
        {"train": [...], "validation": [...], "test": [...]}
    """
    # Define cache file path
    cache_dir = "dataset"
    cache_file = os.path.join(cache_dir, "mw22_transformed.json")

    if verbose:
        print_separator("PROCESS FULL DATASET")
        print()

    # Check if cached version exists
    if os.path.exists(cache_file) and not force_reprocess:
        if verbose:
            print(f"Loading cached processed dataset from {cache_file}...")
        with open(cache_file, "r", encoding="utf-8") as f:
            processed_mw22_dataset = json.load(f)
        if verbose:
            print("Cached processed dataset loaded successfully!")
        return processed_mw22_dataset

    # Process dataset from scratch
    if verbose:
        print("No cached dataset found. Processing from scratch...")
    processed_mw22_dataset = {}

    for split in ["train", "validation", "test"]:
        if verbose:
            print(f"\nProcessing {split} split...")
        dialogues = mw22_dataset[split]
        processed_dialogues = []

        for idx, raw_dialogue in enumerate(dialogues):
            # Transform each dialogue without printing
            processed = transform_dialogue(raw_dialogue)
            processed_dialogues.append(processed)

            # Progress indicator every 100 dialogues
            if verbose and (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(dialogues)} dialogues")

        processed_mw22_dataset[split] = processed_dialogues
        if verbose:
            print(f"  Completed {split}: {len(processed_dialogues)} dialogues")

    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Save processed dataset
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(processed_mw22_dataset, f, indent=2)
    if verbose:
        print("\nProcessed dataset saved successfully!")

    return processed_mw22_dataset


def _filter_dialogue_domains(processed_dialogues: list) -> list:
    """
    Filter dialogues to keep only those with services subset of allowed domains.

    Args:
        processed_dialogues: List of processed dialogue dictionaries

    Returns:
        Filtered list of dialogues where all services are in allowed set
    """
    filtered = []

    # Set of allowed domain names (Default: {"hotel", "restaurant"})
    allowed = {"hotel", "restaurant"}

    for dialogue in processed_dialogues:
        dialogue_services = set(dialogue["services"])
        if dialogue_services.issubset(allowed):
            filtered.append(dialogue)

    return filtered


def filter_dataset_domains(processed_mw22_dataset: dict[str, list], force_reprocess: bool = False, verbose: bool = False) -> dict[str, list]:
    """
    Apply domain filtering to all splits and print statistics.

    Args:
        processed_mw22_dataset: Dictionary with splits as keys and dialogue lists as values
        force_reprocess: If True, ignore cached version and reprocess
        verbose: If True, print pipeline messages (Default: False)

    Returns:
        Filtered dataset with same structure as input
    """
    # Define cache file path
    cache_dir = "dataset"
    cache_file = os.path.join(cache_dir, "mw22_filtered.json")

    # Check if cached version exists
    if os.path.exists(cache_file) and not force_reprocess:
        if verbose:
            print_separator("APPLYING DOMAIN FILTERING")
            print(f"\nLoading cached filtered (processed) dataset from {cache_file}...")
        with open(cache_file, "r", encoding="utf-8") as f:
            filtered_mw22_dataset = json.load(f)
        if verbose:
            print("Cached filtered (processed) dataset loaded successfully!")
            print("\n" + "=" * 60)
        return filtered_mw22_dataset

    # Filter from scratch
    if verbose:
        print_separator("APPLYING DOMAIN FILTERING")

    filtered_mw22_dataset = {}
    for split in ["train", "validation", "test"]:
        original_count = len(processed_mw22_dataset[split])
        filtered_mw22_dataset[split] = _filter_dialogue_domains(processed_mw22_dataset[split])
        filtered_count = len(filtered_mw22_dataset[split])
        retention_rate = (filtered_count / original_count * 100) if original_count > 0 else 0
        if verbose:
            print(f"{split}: {original_count} → {filtered_count} dialogues "
                  f"({retention_rate:.1f}% retained)")

    # Save filtered dataset
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(filtered_mw22_dataset, f, indent=2)
    if verbose:
        print("\nFiltered processed dataset saved successfully!")

    return filtered_mw22_dataset


def explore_processed_dialogue(split: str = 'train', index: int = 0) -> None:
    """
    Print one processed dialogue as a reference example.

    Args:
        split: Dataset split to use ('train', 'validation', 'test')
        index: Index of dialogue to display (Default: 0)
    """
    data = load_split_data("dataset/mw22_filtered.json", split)
    dialogue = data[index]

    print_separator(f"PROCESSED DIALOGUE EXAMPLE — {split.upper()} [{index}]")
    print(json.dumps(dialogue, indent=2))


def run_preprocessing_pipeline(mw22_dataset: dict[str, Any], force_reprocess: bool = False, verbose: bool = False) -> dict[str, list]:
    """
    Run full preprocessing pipeline: transformation + domain filtering.

    Args:
        mw22_dataset: Loaded MultiWOZ dataset from load_multiwoz()
        force_reprocess: If True, ignore cached versions and reprocess
        verbose: If True, print pipeline messages (Default: False)

    Returns:
        Filtered and transformed dataset ready for agent use
    """
    transformed = transform_dataset(mw22_dataset, force_reprocess=force_reprocess, verbose=verbose)
    filtered = filter_dataset_domains(transformed, force_reprocess=force_reprocess, verbose=verbose)

    return filtered


def extract_schema(dataset_path: str) -> dict:
    """
    Extract complete schema from filtered MultiWOZ dataset.

    Args:
        dataset_path: Path to mw22_filtered.json

    Returns:
        Dictionary containing all schema information
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    domains = set()
    intents_by_domain = defaultdict(set)  # no KeyError even though a key, e.g., "hotel" doesn't exist
    action_types = set()
    slots_by_domain = defaultdict(set)
    slot_values = defaultdict(set)
    booking_intents = set()
    info_intents = set()

    # print(f"Extracting schema from {len(data['train'])} training dialogues...")

    for dialogue in data['train']:
        for turn in dialogue.get('turns', []):
            if turn['speaker'] == 'USER':
                for domain, frame_data in turn.get('frames', {}).items():
                    intent = frame_data.get('active_intent')
                    if not intent:
                        continue

                    domains.add(domain)
                    intents_by_domain[domain].add(intent)

                    if 'book' in intent:
                        booking_intents.add(intent)
                    elif 'find' in intent:
                        info_intents.add(intent)

                    for slot_name, slot_value in frame_data.get('slots_values', {}).items():
                        slot = slot_name.split('-', 1)[1] if '-' in slot_name else slot_name
                        slots_by_domain[domain].add(slot)
                        if slot_value:
                            slot_values[slot].add(str(slot_value))

            for act in turn.get('dialogue_acts', []):
                if any(d in act for d in ['Hotel', 'Restaurant', 'Booking', 'general']):
                    action_types.add(act)

    return {
        "domains": sorted(domains),
        "intents_by_domain": {d: sorted(i) for d, i in sorted(intents_by_domain.items())},
        "action_types": sorted(action_types),
        "slots_by_domain": {d: sorted(s) for d, s in sorted(slots_by_domain.items())},
        "slot_values": {s: sorted(v) for s, v in sorted(slot_values.items())},
        "booking_intents": sorted(booking_intents),
        "info_intents": sorted(info_intents)
    }


def create_domain_summary(schema: dict, target_domains: list[str]) -> dict:
    """
    Create focused schema summary for specific domains only.

    Args:
        schema: Full schema from extract_schema()
        target_domains: Domains to focus on (e.g., ["hotel", "restaurant"])

    Returns:
        Filtered schema for target domains only
    """
    filtered_intents = {
        d: i for d, i in schema['intents_by_domain'].items()
        if d in target_domains
    }

    filtered_action_types = [
        act for act in schema['action_types']
        if any(d.capitalize() in act for d in target_domains)
        or 'Booking' in act
        or 'general' in act
    ]

    filtered_slots = {
        d: s for d, s in schema['slots_by_domain'].items()
        if d in target_domains
    }

    all_target_slots = {slot for slots in filtered_slots.values() for slot in slots}

    filtered_slot_values = {
        s: v for s, v in schema['slot_values'].items()
        if s in all_target_slots
    }

    all_target_intents = {i for intents in filtered_intents.values() for i in intents}

    return {
        "target_domains": sorted(target_domains),
        "intents_by_domain": filtered_intents,
        "action_types": sorted(filtered_action_types),
        "slots_by_domain": filtered_slots,
        "slot_values": filtered_slot_values,
        "booking_intents": sorted(i for i in schema['booking_intents'] if i in all_target_intents),
        "info_intents": sorted(i for i in schema['info_intents'] if i in all_target_intents),
        "all_intents": sorted(all_target_intents)
    }

