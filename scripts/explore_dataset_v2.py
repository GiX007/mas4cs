"""Exploration of MultiWOZ 2.2 dataset loaded from the official GitHub repository (budzianowski/multiwoz)."""

"""Exploration of MultiWOZ 2.2 dataset loaded from the official GitHub repository (budzianowski/multiwoz)."""

import json
from pathlib import Path

from src.utils import print_separator,capture_and_save


# Path to official MultiWOZ 2.2 data
MULTIWOZ_DIR = Path("data/multiwoz_github/data/MultiWOZ_2.2")
DB_DIR = Path("data/multiwoz_github/db")

# One file per split (each split has multiple files — we start with one)
TRAIN_FILE = MULTIWOZ_DIR / "train" / "dialogues_001.json"
DEV_FILE = MULTIWOZ_DIR / "dev" / "dialogues_001.json"
TEST_FILE = MULTIWOZ_DIR / "test" / "dialogues_001.json"


def load_dialogues(file_path: Path) -> list[dict]:
    """
    Load dialogues from one official MultiWOZ 2.2 JSON file.

    Args:
        file_path: path to a dialogues_XXX.json file
    Returns:
        list of dialogue dicts
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def explore_basic_structure() -> None:
    """
    Show basic structure of one official MultiWOZ 2.2 dialogue file.

    Mirrors explore_basic_structure() from explore_dataset.py (HF version).
    """
    dialogues = load_dialogues(TRAIN_FILE)

    print_separator("BASIC STRUCTURE — OFFICIAL MULTIWOZ 2.2 (GitHub)")

    print(f"\nFile: {TRAIN_FILE}")
    print(f"Type: {type(dialogues)} | is a list of: {type(dialogues[0])}")
    print(f"Number of dialogues in this file: {len(dialogues)}")

    first = dialogues[0]
    print(f"\nTop-level keys: {list(first.keys())}")
    print(f"dialogue_id: {type(first['dialogue_id']).__name__} | {first['dialogue_id']}")
    print(f"services: {type(first['services']).__name__} | length={len(first['services'])} | {first['services']}")
    print(f"turns: {type(first['turns']).__name__} of {type(first['turns'][0]).__name__} | length={len(first['turns'])}")

    first_turn = first["turns"][0]
    print(f"\nFirst turn keys: {list(first_turn.keys())}")
    print(f"speaker: {type(first_turn['speaker']).__name__} | {first_turn['speaker']}")
    print(f"turn_id: {type(first_turn['turn_id']).__name__} | {first_turn['turn_id']}")
    print(f"utterance: {type(first_turn['utterance']).__name__} | {first_turn['utterance']}")
    print(f"frames: list of {type(first_turn['frames'][0]).__name__} | length={len(first_turn['frames'])}")

    first_frame = first_turn["frames"][0]
    print(f"\nFirst frame keys: {list(first_frame.keys())}")
    print(f"actions: list | length={len(first_frame['actions'])}")
    print(f"service: {type(first_frame['service']).__name__} | {first_frame['service']}")
    print(f"slots: list | length={len(first_frame['slots'])}")
    print(f"state: {type(first_frame['state']).__name__} | keys={list(first_frame['state'].keys())}")

    state = first_frame["state"]
    print(f"\nUnpacking state:")
    print(f"  active_intent: {type(state['active_intent']).__name__} | {state['active_intent']}")
    print(f"  requested_slots: list | length={len(state['requested_slots'])} | {state['requested_slots']}")
    print(f"  slot_values: {type(state['slot_values']).__name__} | length={len(state['slot_values'])} | {state['slot_values']}")

    print_separator("END OF BASIC STRUCTURE")


def explore_single_dialogue(dialogue_idx: int = 0) -> None:
    """
    Print a full dialogue turn by turn with intents and slot values.

    Mirrors explore_single_dialogue() from explore_dataset.py (HF version).

    Params:
        dialogue_idx: index of dialogue to inspect (Default=0)
    Returns:
        None
    """
    dialogues = load_dialogues(TRAIN_FILE)
    dialogue = dialogues[dialogue_idx]

    print_separator(f"SINGLE DIALOGUE (Index {dialogue_idx})")
    print(f"\ndialogue_id: {dialogue['dialogue_id']}")
    print(f"services: {dialogue['services']}")
    print(f"num turns: {len(dialogue['turns'])}\n")

    for turn in dialogue["turns"]:
        speaker = turn["speaker"]
        print(f"  Turn {turn['turn_id']} [{speaker}]: {turn['utterance']}")

        # Only USER turns have state annotations
        if speaker == "USER":
            for frame in turn["frames"]:
                intent = frame["state"]["active_intent"]
                slots = frame["state"]["slot_values"]
                requested = frame["state"]["requested_slots"]

                # Skip inactive services
                if intent == "NONE":
                    continue

                print(f"           service: {frame['service']}")
                print(f"           intent: {intent}")
                print(f"           slots: {slots}")
                print(f"           requested: {requested}")

    print_separator("END OF SINGLE DIALOGUE")


def explore_turn_details(dialogue_idx: int = 0, turn_idx: int = 4) -> None:
    """
    Deep dive into one specific turn — all fields with types.

    Mirrors explore_turn_details() from explore_dataset.py (HF version).

    Params:
        dialogue_idx: index of dialogue in the file (Default=0)
        turn_idx:     index of turn within the dialogue (Default=4, first turn with full slots)
    Returns:
        None
    """
    dialogues = load_dialogues(TRAIN_FILE)
    dialogue = dialogues[dialogue_idx]
    turn = dialogue["turns"][turn_idx]

    print_separator(f"TURN DETAILS (Dialogue {dialogue_idx}, Turn {turn_idx})")
    print(f"\nturn_id: {type(turn['turn_id']).__name__} | {turn['turn_id']}")
    print(f"speaker: {type(turn['speaker']).__name__} | {turn['speaker']}")
    print(f"utterance: {type(turn['utterance']).__name__} | {turn['utterance']}")
    print(f"frames: list of dict | length={len(turn['frames'])}")

    print("\nActive frames (active_intent != NONE):")
    for frame in turn["frames"]:
        intent = frame["state"]["active_intent"] if frame.get("state") else "N/A"
        if intent == "NONE" or intent == "N/A":
            continue
        print(f"\n  service: {type(frame['service']).__name__} | {frame['service']}")
        print(f"  active_intent: {type(intent).__name__} | {intent}")
        print(f"  slot_values: {type(frame['state']['slot_values']).__name__} | {frame['state']['slot_values']}")
        print(f"  requested_slots: list | length={len(frame['state']['requested_slots'])} | {frame['state']['requested_slots']}")
        print(f"  slots (spans): list | length={len(frame['slots'])}")

    print("\nNote on USER turns:")
    print("  frames = 8 dicts → one frame per service (hotel, restaurant, taxi, ...)")
    print("  intent = NONE → service not active in this turn (skipped)")
    print("  intent = find_X → user is searching for an entity")
    print("  intent = book_X → user wants to make a booking")
    print("  slot_values → constraints user has specified so far (accumulate across turns)")
    print("  requested_slots → information user is asking the system to provide")
    print("  Example: 'I need a cheap hotel in the north' → slot_values={hotel-area: north, hotel-pricerange: cheap}")
    print("  Example: 'What is the phone number?' → requested_slots=[hotel-phone]")

    print("\nNote on SYSTEM turns:")
    print("  frames = [] → system is asking a question (no entity mentioned)")
    print("  frames = [...] → system mentioned a specific entity (span annotations present)")
    print("  Example: 'Do you prefer African or British food?' → frames=[]")
    print(
        "  Example: 'I recommend Bedouin in the centre.' → frames=[{slot: restaurant-name, value: Bedouin, start: 31, end: 38}]")

    print_separator("END OF TURN DETAILS")


def explore_db_structure() -> None:
    """
    Explore hotel and restaurant DB files — fields, types, and differences.

    Directly informs db_query.py design.

    Returns:
        None
    """
    hotel_path      = DB_DIR / "hotel_db.json"
    restaurant_path = DB_DIR / "restaurant_db.json"

    with open(hotel_path, "r", encoding="utf-8") as f:
        hotel_db = json.load(f)
    with open(restaurant_path, "r", encoding="utf-8") as f:
        restaurant_db = json.load(f)

    print_separator("DB STRUCTURE — HOTEL & RESTAURANT")

    # Hotel db
    print(f"\nHotel DB type: {type(hotel_db)} | is a list of: {type(hotel_db[0])}")
    print(f"Hotel DB length: {len(hotel_db)} entries\n")

    h = hotel_db[0]
    print("First hotel entry keys:")
    print(f"  address: {type(h['address']).__name__} | {h['address']}")
    print(f"  area: {type(h['area']).__name__} | {h['area']}")
    print(f"  internet: {type(h['internet']).__name__} | {h['internet']}")
    print(f"  parking: {type(h['parking']).__name__} | {h['parking']}")
    print(f"  id: {type(h['id']).__name__} | {h['id']}")
    print(f"  location: {type(h['location']).__name__} | {h['location']}")
    print(f"  name: {type(h['name']).__name__} | {h['name']}")
    print(f"  phone: {type(h['phone']).__name__} | {h['phone']}")
    print(f"  postcode: {type(h['postcode']).__name__} | {h['postcode']}")
    print(f"  price: {type(h['price']).__name__} | {h['price']}")
    print(f"  pricerange: {type(h['pricerange']).__name__} | {h['pricerange']}")
    print(f"  stars: {type(h['stars']).__name__} | {h['stars']}")
    print(f"  takesbookings: {type(h['takesbookings']).__name__} | {h['takesbookings']}")
    print(f"  type: {type(h['type']).__name__} | {h['type']}")

    print("\nFull first hotel entry:")
    print(json.dumps(h, indent=4))

    # Restaurant db
    print(f"\nRestaurant DB type: {type(restaurant_db)} | is a list of: {type(restaurant_db[0])}")
    print(f"Restaurant DB length: {len(restaurant_db)} entries\n")

    r = restaurant_db[0]
    print("First restaurant entry keys:")
    print(f"  address: {type(r['address']).__name__} | {r['address']}")
    print(f"  area: {type(r['area']).__name__} | {r['area']}")
    print(f"  food: {type(r['food']).__name__} | {r['food']}")
    print(f"  id: {type(r['id']).__name__} | {r['id']}")
    print(f"  introduction: {type(r['introduction']).__name__} | {r['introduction'][:60]}...")
    print(f"  location: {type(r['location']).__name__} | {r['location']}")
    print(f"  name: {type(r['name']).__name__} | {r['name']}")
    print(f"  phone: {type(r['phone']).__name__} | {r['phone']}")
    print(f"  postcode: {type(r['postcode']).__name__} | {r['postcode']}")
    print(f"  pricerange: {type(r['pricerange']).__name__} | {r['pricerange']}")
    print(f"  type: {type(r['type']).__name__} | {r['type']}")

    print("\nFull first restaurant entry:")
    print(json.dumps(r, indent=4))

    print("\nKey differences hotel vs restaurant:")
    print("  Hotel only: stars, parking, internet, takesbookings, type, price (dict)")
    print("  Restaurant only: food, introduction")
    print("  Shared: name, area, pricerange, phone, address, postcode, location, id")

    print_separator("END OF DB STRUCTURE")


def explore_conversation_examples(n_examples: int = 2) -> None:
    """
    Print readable conversation examples filtered to hotel/restaurant dialogues only.

    Mirrors explore_conversation_examples() from explore_dataset.py (HF version).

    Params:
        n_examples: number of hotel/restaurant dialogues to print (Default=2)
    Returns:
        None
    """
    dialogues = load_dialogues(TRAIN_FILE)

    # Filter to hotel/restaurant only — same scope as our experiments
    filtered = [
        d for d in dialogues
        if all(s in ("hotel", "restaurant") for s in d["services"])
    ]

    print_separator(f"CONVERSATION EXAMPLES — HOTEL & RESTAURANT ONLY")
    print(f"\nTotal dialogues in file: {len(dialogues)}")
    print(f"Hotel/restaurant only dialogues: {len(filtered)}")
    print(f"Showing first {n_examples}:")

    for i, dialogue in enumerate(filtered[:n_examples]):
        print("\n" + "-" * 60)
        print(f"Dialogue ID: {dialogue['dialogue_id']}")
        print(f"Services: {dialogue['services']}")
        print(f"Num turns: {len(dialogue['turns'])}")

        for turn in dialogue["turns"]:
            speaker  = turn["speaker"]
            print(f"\n  Turn {turn['turn_id']} [{speaker}]: {turn['utterance']}")

            if speaker == "USER":
                for frame in turn["frames"]:
                    intent = frame["state"]["active_intent"]
                    slots = frame["state"]["slot_values"]
                    requested = frame["state"]["requested_slots"]

                    if intent == "NONE":
                        continue

                    print(f"           service: {frame['service']}")
                    print(f"           intent: {intent}")
                    print(f"           slots: {slots}")
                    print(f"           requested: {requested}")

    print_separator("END OF CONVERSATION EXAMPLES")


def count_hotel_restaurant_dialogues() -> None:
    """
    Count hotel/restaurant-only dialogues across all splits.

    Returns:
        None
    """
    print_separator("HOTEL/RESTAURANT DIALOGUE COUNTS ACROSS SPLITS")
    print()
    for split in ("train", "dev", "test"):
        split_dir = MULTIWOZ_DIR / split
        all_files = sorted(split_dir.glob("dialogues_*.json"))
        all_dialogues = []
        for f in all_files:
            all_dialogues.extend(load_dialogues(f))

        filtered = [
            d for d in all_dialogues
            if all(s in ("hotel", "restaurant") for s in d["services"])
        ]
        print(f"{split}: {len(all_dialogues)} total → {len(filtered)} hotel/restaurant only ({len(filtered)/len(all_dialogues)*100:.1f}%)")

    print_separator("END OF DIALOGUE COUNTS")


def main() -> None:
    """Exploration workflow orchestrator."""
    explore_basic_structure()
    explore_single_dialogue()
    explore_turn_details()
    explore_db_structure()
    explore_conversation_examples()
    count_hotel_restaurant_dialogues()


if __name__ == "__main__":

    capture_and_save(func=main,
                     output_path="docs/dataset_inspection/mw22_github_exploration.txt")

