"""Script to explore MultiWOZ 2.2 dataset structure."""

import json
from pathlib import Path
from typing import Any
from collections import Counter

from src.data import load_multiwoz, load_split_data, run_preprocessing_pipeline, extract_schema, create_domain_summary
from src.utils import print_separator, capture_and_save


def explore_multiwoz22_v0(n_dialogues: int = 1) -> None:
    """
    Quick and Dirty exploration of multiwoz 2.2.

    Args:
        n_dialogues: Number of dialogues to inspect from the train split (Default=5)

    Resources:
        - https://aclanthology.org/2020.nlp4convai-1.13.pdf
        - https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2
        - https://huggingface.co/datasets/pfb30/multi_woz_v22
    """
    dataset = load_multiwoz()
    print("1. dataset:"); print(type(dataset)); print(dataset.keys()); print(); print(dataset); print()

    split = "train"
    print(f"2. {split} split:")
    data = dataset[split]
    print(type(data)); print(); print(data); print()
    print(f"Dialogues (length of split): {len(data)}"); print(); print(f"Type of Features: {type(data.features)}"); print(); print(f"Features: {data.features}"); print(); print(data.features.keys()); print()

    print(f"3. first sample (0 / {len(data)}):")
    first = data[0]
    print(type(first)); print(first.keys()); print(type(first['dialogue_id']), type(first['services']), type(first['turns'])); print()
    print("dialogue_id:", first['dialogue_id']); print("services:", first['services']); print("turns keys:", first['turns'].keys()); print()
    first_turns = first['turns']
    print("Unpack 'turns':")
    for k, v in first_turns.items():
        print(k, v); print()

    print("4. Exploring frames")
    first_frames = first_turns['frames']
    print(type(first_frames)); print(len(first_frames)); print()
    first_frames_first = first_frames[0]

    print("First frame (index 0):", type(first_frames_first), first_frames_first.keys())
    print(type(first_frames_first['service']), type(first_frames_first['state']), type(first_frames_first['slots'])); print()

    print("5. Exploring dialogue_acts")
    first_dialogue_acts = first_turns['dialogue_acts']
    print(type(first_dialogue_acts)); print(len(first_dialogue_acts)); print()
    first_dial_acts_first = first_dialogue_acts[0]

    print("First dialogue act (index 0):", type(first_dial_acts_first), first_dial_acts_first.keys())
    print(type(first_dial_acts_first['dialog_act']), type(first_dial_acts_first['span_info'])); print()

    # Unpack n_dialogues considering some basic info like intent and action
    domain_counter = Counter()
    intent_counter = Counter()
    action_counter = Counter()

    for i in range(min(n_dialogues, len(data))):
        d = data[i]
        print("=" * 60)
        print(f"Dialogue ID: {d['dialogue_id']}")
        print(f"Services: {d['services']}")


        num_turns  = len(d['turns']["turn_id"])
        for turn_idx in range(num_turns):
            speaker = d["turns"]["speaker"][turn_idx]
            utterance = d["turns"]["utterance"][turn_idx]
            frames = d["turns"]["frames"][turn_idx]
            dialogue_acts = d["turns"]["dialogue_acts"][turn_idx]

            speaker_label = "USER" if speaker == 0 else "SYSTEM"
            print(f"\n  Turn {turn_idx} [{speaker_label}]: {utterance}")

            # Services (frames["service"] is a list of strings)
            services = frames.get("service", [])
            for svc in services:
                domain_counter[svc] += 1
                print(f"    Service: {svc}")

            # States (list of state dicts)
            states = frames.get("state", [])
            for state in states:
                intent = state.get("active_intent", "")
                if intent:
                    intent_counter[intent] += 1
                    print(f"    Intent: {intent}")

                # Slot values
                slots_vals = state.get("slots_values", {})
                if slots_vals.get("slots_values_name"):
                    print(f"    Slots: {dict(zip(slots_vals['slots_values_name'], slots_vals['slots_values_list']))}")

            # Dialogue acts
            dialog_act = dialogue_acts.get("dialog_act", {})
            act_types = dialog_act.get("act_type", [])
            for act in act_types:
                action_counter[act] += 1
                print(f"    Action: {act}")

        print("\n" + "=" * 60)
        print("\nSummary:")
        print("Top domains:", domain_counter.most_common(5))
        print("Top intents:", intent_counter.most_common(10))
        print("Top actions:", action_counter.most_common(10))


def explore_basic_structure() -> None:
    """Show basic dataset structure and splits."""
    dataset = load_multiwoz()

    print_separator("BASIC DATASET STRUCTURE EXPLORATION")
    print(f"Dataset type: {type(dataset)}")
    print("\nDataset:", dataset)
    print(f"\nSplits: {list(dataset.keys())}")
    print(f"Train size: {len(dataset['train'])}")
    print(f"Validation size: {len(dataset['validation'])}")
    print(f"Test size: {len(dataset['test'])}")
    print(f"\nExample of features:\n{dataset['train'].features}")


def explore_single_dialogue(dialogue_idx: int = 0) -> None:
    """
    Explore the structure of a single dialogue.

    Args:
        dialogue_idx: Index of dialogue in the train split (Default=0)
    """
    dataset = load_multiwoz()
    dialogue = dataset['train'][dialogue_idx]

    print_separator(f"SINGLE DIALOGUE STRUCTURE (Index {dialogue_idx})")
    print(f"Top-level keys: {dialogue.keys()}")
    print(f"Dialogue ID: {dialogue['dialogue_id']}")
    print(f"Services: {dialogue['services']}\n")

    turns = dialogue['turns']
    print(f"Turns type: {type(turns)}")
    print(f"Turns keys: {turns.keys()}")
    print(f"\nNumber of turns: {len(turns['turn_id'])}")
    print(f"turn_id: {turns['turn_id']}")
    print(f"\nNumber of speakers: {len(turns['speaker'])}")
    print(f"speaker: {turns['speaker']}")
    print(f"\nNumber of utterances: {len(turns['utterance'])}")
    print(f"utterance: {turns['utterance']}")
    print(f"\nNumber of frames: {len(turns['frames'])}")
    print(f"Number of dialogue_acts: {len(turns['dialogue_acts'])}")
    print("\nNote: all 5 turn keys are parallel lists — same length, indexed by dialogue_id.")


def explore_turn_details(dialogue_idx: int = 0, turn_idx: int = 0) -> None:
    """
    Explore a single turn in detail.

    Args:
        dialogue_idx: Index of dialogue in the train split (Default=0)
        turn_idx: Index of turn within the selected dialogue (Default=0)
    """
    dataset = load_multiwoz()
    dialogue = dataset['train'][dialogue_idx]
    turns = dialogue['turns']

    print_separator(f"TURN DETAILS (Dialogue {dialogue_idx}, Turn {turn_idx})")

    turn_id = turns['turn_id'][turn_idx]
    speaker = turns['speaker'][turn_idx]
    utterance = turns['utterance'][turn_idx]
    frames = turns['frames'][turn_idx]
    dialogue_acts = turns['dialogue_acts'][turn_idx]

    speaker_label = "USER" if speaker == 0 else "SYSTEM"
    print(f"Turn id: {turn_id}")
    print(f"Speaker: {speaker_label}")
    print(f"Utterance: {utterance}")

    print(f"\nFrames keys: {frames.keys()}")
    print(f"Services: {frames['service']}")

    print(f"\nDialogue acts keys: {dialogue_acts.keys()}")
    print(f"Dialog act: {dialogue_acts['dialog_act']}")


def explore_frames_structure(dialogue_idx: int = 0, turn_idx: int = 0) -> None:
    """
    Deep exploration of frames structure with types.

    Args:
        dialogue_idx: Index of dialogue in the train split (Default=0)
        turn_idx: Index of turn within the selected dialogue (Default=0)

    """
    dataset = load_multiwoz()
    dialogue = dataset['train'][dialogue_idx]
    frames = dialogue['turns']['frames'][turn_idx]

    print_separator(f"FRAMES DEEP DIVE (Dialogue {dialogue_idx}, Turn {turn_idx}, Frame {turn_idx})")

    print(f"Frame type: {type(frames)}")
    print(f"Frames keys: {frames.keys()}\n")

    # Service
    print("--- SERVICE ---")
    print(f"Type: {type(frames['service'])}")
    print(f"Length: {len(frames['service'])}")
    print(f"Value: {frames['service']}\n")

    # State
    print("--- STATE ---")
    print(f"Type: {type(frames['state'])}")
    print(f"Length: {len(frames['state'])}")
    if frames['state']:
        state = frames['state'][0]
        print(f"First state type: {type(state)}")
        print(f"First state keys: {state.keys()}\n")

        print(f"  active_intent type: {type(state['active_intent'])}")
        print(f"  active_intent value: {state['active_intent']}\n")

        print(f"  requested_slots type: {type(state['requested_slots'])}")
        print(f"  requested_slots value: {state['requested_slots']}\n")

        print(f"  slots_values type: {type(state['slots_values'])}")
        print(f"  slots_values keys: {state['slots_values'].keys()}")
        print(f"  slots_values_name: {state['slots_values']['slots_values_name']}")
        print(f"  slots_values_list: {state['slots_values']['slots_values_list']}\n")

    # Slots
    print("--- SLOTS ---")
    print(f"Type: {type(frames['slots'])}")
    print(f"Length: {len(frames['slots'])}")
    if frames['slots']:
        slot = frames['slots'][0]
        print(f"First slot type: {type(slot)}")
        print(f"First slot keys: {slot.keys()}")
        print(f"  slot: {slot['slot']}")
        print(f"  value: {slot['value']}")
        print(f"  start: {slot['start']}")
        print(f"  exclusive_end: {slot['exclusive_end']}")


def explore_dialogue_acts_structure(dialogue_idx: int = 0, turn_idx: int = 0) -> None:
    """
    Deep exploration of dialogue_acts structure with types.

    Args:
        dialogue_idx: Index of dialogue in the train split (Default=0)
        turn_idx: Index of turn within the selected dialogue (Default=0)

    """
    dataset = load_multiwoz()
    dialogue = dataset['train'][dialogue_idx]
    dialogue_acts = dialogue['turns']['dialogue_acts'][turn_idx]

    print_separator(f"DIALOGUE_ACTS DEEP DIVE (Dialogue {dialogue_idx}, Turn {turn_idx}, Dialogue Act{turn_idx})")
    print(f"Dialogue_acts type: {type(dialogue_acts)}")
    print(f"Dialogue_acts keys: {dialogue_acts.keys()}\n")

    # Dialog act
    print("--- DIALOG_ACT ---")
    dialog_act = dialogue_acts['dialog_act']
    print(f"Type: {type(dialog_act)}")
    print(f"Keys: {dialog_act.keys()}\n")

    print(f"  act_type type: {type(dialog_act['act_type'])}")
    print(f"  act_type value: {dialog_act['act_type']}\n")

    print(f"  act_slots type: {type(dialog_act['act_slots'])}")
    print(f"  act_slots length: {len(dialog_act['act_slots'])}")
    if dialog_act['act_slots']:
        act_slot = dialog_act['act_slots'][0]
        print(f"  First act_slot type: {type(act_slot)}")
        print(f"  First act_slot keys: {act_slot.keys()}")
        print(f"    slot_name: {act_slot['slot_name']}")
        print(f"    slot_value: {act_slot['slot_value']}")

    # Span info
    print("\n--- SPAN_INFO ---")
    span_info = dialogue_acts['span_info']
    print(f"Type: {type(span_info)}")
    print(f"Keys: {span_info.keys()}\n")

    print(f"  act_type type: {type(span_info['act_type'])}")
    print(f"  act_type value: {span_info['act_type']}\n")

    print(f"  act_slot_name type: {type(span_info['act_slot_name'])}")
    print(f"  act_slot_name value: {span_info['act_slot_name']}\n")

    print(f"  act_slot_value type: {type(span_info['act_slot_value'])}")
    print(f"  act_slot_value value: {span_info['act_slot_value']}\n")

    print(f"  span_start type: {type(span_info['span_start'])}")
    print(f"  span_start value: {span_info['span_start']}\n")

    print(f"  span_end type: {type(span_info['span_end'])}")
    print(f"  span_end value: {span_info['span_end']}")


def explore_annotations(n_dialogues: int = 5) -> dict[str, Any]:
    """
    Analyze annotations across multiple dialogues.

    Args:
        n_dialogues: Number of dialogues to inspect from the train split (Default=5).

    Returns:
        Dictionary summarizing discovered annotation structures.

    """
    dataset = load_multiwoz()
    data = dataset['train']

    domain_counter = Counter()
    intent_counter = Counter()
    action_counter = Counter()

    print_separator(f"ANNOTATION ANALYSIS ({n_dialogues} dialogues)")

    for i in range(min(n_dialogues, len(data))):
        dialogue = data[i]
        num_turns = len(dialogue['turns']['turn_id'])

        for turn_idx in range(num_turns):
            frames = dialogue['turns']['frames'][turn_idx]
            dialogue_acts = dialogue['turns']['dialogue_acts'][turn_idx]

            for service in frames.get('service', []):
                domain_counter[service] += 1

            for state in frames.get('state', []):
                intent = state.get('active_intent', '')
                if intent:
                    intent_counter[intent] += 1

            dialog_act = dialogue_acts.get('dialog_act', {})
            for act_type in dialog_act.get('act_type', []):
                action_counter[act_type] += 1

    print(f"\nTop 5 Domains: {domain_counter.most_common(5)}")
    print(f"\nTop 10 Intents: {intent_counter.most_common(10)}")
    print(f"\nTop 10 Actions: {action_counter.most_common(10)}")

    print_separator("END OF BASIC DATASET STRUCTURE EXPLORATION")


def run_exploration(n_dialogues: int = 5) -> None:
    """
    Run all exploration steps in sequence.

    Args:
        n_dialogues: Number of dialogues for annotation analysis
    """
    explore_basic_structure()
    explore_single_dialogue()
    explore_turn_details()
    explore_frames_structure()
    explore_dialogue_acts_structure()
    explore_annotations(n_dialogues)


def explore_conversation_examples(n_examples: int = 3, split: str = 'train') -> None:
    """
    Print readable conversation examples with full annotations.

    Args:
        n_examples: Number of dialogues to print
        split: Dataset split to use ('train', 'validation', 'test')
    """
    dataset = load_multiwoz()
    data = dataset[split]

    print_separator(f"MULTIWOZ 2.2 CONVERSATION EXAMPLES — {split.upper()} SPLIT")
    print(f"\nNumber of examples: {n_examples}")

    for i in range(min(n_examples, len(data))):
        d = data[i]
        print("\n" + "=" * 60)
        print(f"Dialogue ID: {d['dialogue_id']}")
        print(f"Services: {d['services']}")

        num_turns = len(d['turns']['turn_id'])
        for turn_idx in range(num_turns):
            speaker = d['turns']['speaker'][turn_idx]
            utterance = d['turns']['utterance'][turn_idx]
            frames = d['turns']['frames'][turn_idx]
            dialogue_acts = d['turns']['dialogue_acts'][turn_idx]

            speaker_label = "USER" if speaker == 0 else "SYSTEM"
            print(f"\n  Turn {turn_idx} [{speaker_label}]: {utterance}")

            # Services
            for svc in frames.get('service', []):
                print(f"    Service: {svc}")

            # States
            for state in frames.get('state', []):
                intent = state.get('active_intent', '')
                if intent:
                    print(f"    Intent: {intent}")

                slots_vals = state.get('slots_values', {})
                if slots_vals.get('slots_values_name'):
                    print(f"    Slots: {dict(zip(slots_vals['slots_values_name'], slots_vals['slots_values_list']))}")

                requested = state.get('requested_slots', [])
                if requested:
                    print(f"    Requested slots: {requested}")

            # Dialogue acts
            dialog_act = dialogue_acts.get('dialog_act', {})
            for act in dialog_act.get('act_type', []):
                print(f"    Act type: {act}")

            # Span info
            # span_info = dialogue_acts.get('span_info', {})
            # names = span_info.get('act_slot_name', [])
            # values = span_info.get('act_slot_value', [])
            # if names:
            #     print(f"    Span slots: {dict(zip(names, values))}")

        print_separator(f"END OF MULTIWOZ 2.2 CONVERSATION EXAMPLES")


def explore_dataset_statistics(split: str = 'train', filepath: str = None) -> None:
    """
    Print statistics about MultiWOZ 2.2 dataset (raw or filtered processed).

    Args:
        split: Dataset split to analyze ('train', 'validation', 'test')
        filepath: Path to processed .json file. If None, loads raw from HuggingFace
    """
    if filepath:
        data = load_split_data(filepath, split)
        title = f"FILTERED MULTIWOZ 2.2 STATISTICS — {split.upper()}"
    else:
        data = load_multiwoz()[split]
        title = f"MULTIWOZ 2.2 STATISTICS — {split.upper()}"

    print_separator(title)
    print(f"\nAnalyzing {len(data)} dialogues of {split} split...")

    dialogue_ids = set()
    services_counter = Counter()
    speaker_counter = Counter()
    intent_counter = Counter()
    requested_slots_counter = Counter()
    slot_names_counter = Counter()
    act_type_counter = Counter()
    act_slot_names_counter = Counter()
    num_turns_list = []
    num_services_per_dialogue = []
    total_turn_ids = 0

    for dialogue in data:
        dialogue_ids.add(dialogue['dialogue_id'])

        services = dialogue['services']
        num_services_per_dialogue.append(len(services))
        for service in services:
            services_counter[service] += 1

        # Handle both raw (parallel lists) and processed (list of dicts)
        turns = dialogue['turns']
        if isinstance(turns, dict):
            num_turns = len(turns['turn_id'])
            num_turns_list.append(num_turns)
            total_turn_ids += num_turns

            for turn_idx in range(num_turns):
                speaker = turns['speaker'][turn_idx]
                speaker_counter["USER" if speaker == 0 else "SYSTEM"] += 1

                frames = turns['frames'][turn_idx]
                dialogue_acts = turns['dialogue_acts'][turn_idx]

                for state in frames.get('state', []):
                    intent = state.get('active_intent', '')
                    if intent:
                        intent_counter[intent] += 1
                    for slot in state.get('requested_slots', []):
                        requested_slots_counter[slot] += 1
                    for slot_name in state.get('slots_values', {}).get('slots_values_name', []):
                        slot_names_counter[slot_name] += 1

                dialog_act = dialogue_acts.get('dialog_act', {})
                for act in dialog_act.get('act_type', []):
                    act_type_counter[act] += 1
                for act_slots in dialog_act.get('act_slots', []):
                    for slot_name in act_slots.get('slot_name', []):
                        act_slot_names_counter[slot_name] += 1

        else:  # processed: list of dicts
            num_turns = len(turns)
            num_turns_list.append(num_turns)
            total_turn_ids += num_turns

            for turn in turns:
                speaker_counter[turn['speaker']] += 1

                for service, frame in turn['frames'].items():
                    intent = frame.get('active_intent', '')
                    if intent:
                        intent_counter[intent] += 1
                    for slot in frame.get('requested_slots', []):
                        requested_slots_counter[slot] += 1
                    for slot_name in frame.get('slots_values', {}).keys():
                        slot_names_counter[slot_name] += 1

                for act in turn.get('dialogue_acts', []):
                    act_type_counter[act] += 1
                for act_slot in turn.get('dialogue_act_slots', []):
                    act_slot_names_counter[act_slot['slot']] += 1

    print_separator("1. DIALOGUE-LEVEL STATISTICS")
    print(f"Total dialogues: {len(data)}")
    print(f"Unique dialogue_ids: {len(dialogue_ids)}")
    print(f"Average turns per dialogue: {sum(num_turns_list) / len(num_turns_list):.2f}")
    print(f"Total turns: {total_turn_ids}")
    print(f"Min turns: {min(num_turns_list)}")
    print(f"Max turns: {max(num_turns_list)}")
    print(f"Average services per dialogue: {sum(num_services_per_dialogue) / len(num_services_per_dialogue):.2f}")

    print_separator("2. SPEAKER DISTRIBUTION")
    for speaker, count in speaker_counter.most_common():
        print(f"  {speaker:10s} : {count} turns")

    print_separator("3. SERVICES (DOMAINS) DISTRIBUTION")
    print(f"Total unique services: {len(services_counter)}\n")
    for service, count in services_counter.most_common():
        print(f"  {service:10s} : {count} dialogues ({(count / len(data)) * 100:.2f}%)")

    print_separator("4. ACTIVE INTENTS DISTRIBUTION")
    print(f"Total unique intents: {len(intent_counter)}\n")
    for intent, count in intent_counter.most_common(15):
        print(f"  {intent:15s} : {count} occurrences")

    print_separator("5. REQUESTED SLOTS DISTRIBUTION (Top 15)")
    print(f"Total unique requested slots: {len(requested_slots_counter)}\n")
    for slot, count in requested_slots_counter.most_common(15):
        print(f"  {slot:25s} : {count} occurrences")

    print_separator("6. SLOT NAMES DISTRIBUTION (Top 20)")
    print(f"Total unique slot names: {len(slot_names_counter)}\n")
    for slot, count in slot_names_counter.most_common(20):
        print(f"  {slot:25s} : {count} occurrences")

    print_separator("7. ACTION TYPES DISTRIBUTION (Top 20)")
    print(f"Total unique action types: {len(act_type_counter)}\n")
    for act_type, count in act_type_counter.most_common(20):
        print(f"  {act_type:20s} : {count} occurrences")

    print_separator("8. ACTION SLOT NAMES DISTRIBUTION (Top 15)")
    print(f"Total unique action slot names: {len(act_slot_names_counter)}\n")
    for slot, count in act_slot_names_counter.most_common(15):
        print(f"  {slot:15s} : {count} occurrences")

    if filepath:
        print_separator("OVERVIEW — ALL SPLITS")
        total_dialogues = 0
        total_turns = 0
        for s in ["train", "validation", "test"]:
            d = load_split_data(filepath, s)
            turns = sum(len(dialogue['turns']) for dialogue in d)
            print(f"  {s:12s}: {len(d)} dialogues, {turns} turns")
            total_dialogues += len(d)
            total_turns += turns
        print(f"\n  Total: {total_dialogues} dialogues, {total_turns} turns")

    print_separator("STATISTICS COMPLETE")


def explore_evaluation_guide() -> None:
    """
    Print a guide explaining core MultiWOZ features for MAS4CS.

    Returns: None
    """
    dataset = load_multiwoz()
    data = dataset['train']
    example = data[0]
    example_turn = example['turns']
    example_frame = example_turn['frames'][0]
    example_acts = example_turn['dialogue_acts'][0]

    print_separator("MULTIWOZ 2.2 FEATURES GUIDE FOR MAS4CS")
    print("\nThis guide explains which dataset features we are going to use and how.")
    print("Focus: Building agents that understand, track, and respond to user requests.")
    print("This is an initial estimate. During implementation, we re-consider our options.")

    print_separator("DATASET OVERVIEW")
    print(f"MultiWOZ 2.2 contains {len(data)} training dialogues about:")
    print("  - Hotels, Restaurants, Trains, Taxis, Attractions")
    print(f"  - Average {sum(len(d['turns']['turn_id']) for d in data) / len(data):.1f} turns per conversation")
    print("  - Multi-domain conversations (users can switch between services)")

    print_separator("9 CORE FEATURES WE USE")

    print("1. dialogue_id")
    print("   What: Unique name for each conversation")
    print(f"   Example: '{example['dialogue_id']}'")
    print("   How we use it: Track which conversation we're processing")

    print("\n2. turn_id")
    print("   What: Order of turn in conversation")
    print(f"   Example: {example_turn['turn_id'][:3]}")
    print("   How we use it: Keep conversation in correct order")

    print("\n3. services")
    print("   What: Which domains are involved (hotel, restaurant, etc.)")
    print(f"   Example: {example['services']}")
    print("   How we use it: Detect and Validate the correct domain")

    print("\n4. speaker + utterance")
    print("   What: Who said what (USER or SYSTEM)")
    print(f"   Example speaker: {'USER' if example_turn['speaker'][0] == 0 else 'SYSTEM'}")
    print(f"   Example utterance: '{example_turn['utterance'][0]}'")
    print("   How we use it: Understand what user wants and generate appropriate responses and actions")

    states = example_frame.get('state', [])
    example_intent = states[0]['active_intent'] if states and states[0].get('active_intent') else 'find_restaurant'
    print("\n5. active_intent")
    print("   What: What task the user wants to do")
    print(f"   Example: '{example_intent}'")
    print("   How we use it: Decide which route to follow (e.g., action or policy) and also Validate intent detection")

    print("\n6. requested_slots")
    print("   What: Information the user asks for")
    print("   Example: ['hotel-phone', 'hotel-address']")
    print("   How we use it: Know what info to provide in response")

    slots_vals = states[0].get('slots_values', {}) if states else {}
    if slots_vals.get('slots_values_name'):
        example_slots = dict(zip(slots_vals['slots_values_name'][:2], slots_vals['slots_values_list'][:2]))
    else:
        example_slots = {'restaurant-area': ['centre'], 'restaurant-pricerange': ['cheap']}
    print("\n7. slots_values")
    print("   What: User's specific requirements (the belief state)")
    print(f"   Example: {example_slots}")
    print("   How we use it: Validate our predictions for hotels/restaurants matching these criteria")

    dialog_act = example_acts.get('dialog_act', {})
    example_act = dialog_act.get('act_type', ['Restaurant-Inform'])[0] if dialog_act.get('act_type') else 'Restaurant-Inform'
    print("\n8. act_type (dialogue_acts)")
    print("   What: Type of action being performed")
    print(f"   Example: '{example_act}'")
    print("   How we use it: Validate we're doing the right type of action")

    print("\n9. act_slots (dialogue_acts)")
    print("   What: Which slot-value pairs go with the action")
    print("   Example: slot='area', value='centre'")
    print("   How we use it: Ensure we extracted the right information")

    print_separator("EXAMPLES OF HOW THESE MAY CONNECT TO EVALUATION")
    print("slots_values → Joint Goal Accuracy (JGA)")
    print("  Did we track all user requirements correctly?")
    print("\nactive_intent + act_type → Task Success Rate (One point of view)")
    print("  Did we complete the right task?")
    print("\nrequested_slots + act_slots → Policy Compliance")
    print("  Did we ask for required information before booking?")

    print_separator("GUIDE COMPLETE")


def explore_processed_dialogue(idx: int = 0, split: str = "train") -> None:
    """
    Print one processed dialogue as a reference example.

    Args:
        idx: Index of the sample in the split (Default: first dialogue)
        split: Dataset split to use ('train', 'validation', 'test')
    """
    data = load_split_data("dataset/mw22_filtered.json", split)
    dialogue = data[idx]

    print_separator(f"PROCESSED DIALOGUE EXAMPLE — {split.upper()} [{idx}]")
    print(json.dumps(dialogue, indent=2))
    print_separator("END OF DIALOGUE EXAMPLE")


def explore_full_schema() -> None:
    """Print full schema extracted from filtered dataset."""
    schema = extract_schema("dataset/mw22_filtered.json")
    print_separator("MULTIWOZ 2.2 FULL SCHEMA")
    print(json.dumps(schema, indent=2))


def explore_hotel_restaurant_schema() -> None:
    """Print hotel & restaurant focused schema."""
    schema = extract_schema("dataset/mw22_filtered.json")
    summary = create_domain_summary(schema, ["hotel", "restaurant"])
    print_separator("MULTIWOZ 2.2 SCHEMA — HOTEL & RESTAURANT")
    print(json.dumps(summary, indent=2))


def main():
    """Simple Exploration Workflow Orchestrator."""
    # Quick and dirty exploration
    # explore_multiwoz22_v0()

    # Raw dataset exploration
    ds = load_multiwoz()  # Ensure the dataset is here
    capture_and_save(func=run_exploration,
                     output_path="docs/dataset_inspection/mw22_structure.txt")  # Basic exploration pipeline
    capture_and_save(func=explore_conversation_examples,
                     output_path="docs/dataset_inspection/mw22_conversation_examples.txt")  # Raw conversation examples
    capture_and_save(func=explore_dataset_statistics,
                     output_path="docs/dataset_inspection/mw22_statistics.txt")  # Basic statistics
    capture_and_save(func=explore_evaluation_guide,
                     output_path="docs/dataset_inspection/mw22_features_guide.txt")  # Basic feature space exploration

    # Filtered dataset exploration
    run_preprocessing_pipeline(ds)  # Ensure the dataset is preprocessed
    capture_and_save(func=lambda: explore_dataset_statistics(filepath="dataset/mw22_filtered.json"),
                     output_path="docs/dataset_inspection/mw22_filtered_statistics.txt")  # Filtered basic statistics
    capture_and_save(func=explore_processed_dialogue,
                     output_path="docs/dataset_inspection/mw22_processed_dialogue_example.txt")  # Processed dialogue example

    # Deeper exploration
    capture_and_save(func=explore_full_schema,
                     output_path="docs/dataset_inspection/mw22_schema_full.txt")  # Extract filtered dataset schema
    capture_and_save(func=explore_hotel_restaurant_schema,
                     output_path="docs/dataset_inspection/mw22_schema_hotel_restaurant.txt")  # Extract domain summary for hotel, restaurant only

    # Save JSON files for schemas
    schema = extract_schema("dataset/mw22_filtered.json")
    summary = create_domain_summary(schema, ["hotel", "restaurant"])
    Path("dataset/mw22_schema_full.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    Path("dataset/mw22_schema_hotel_restaurant.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print_separator("END OF DATASET EXPLORATION")


if __name__ == "__main__":
    main()

