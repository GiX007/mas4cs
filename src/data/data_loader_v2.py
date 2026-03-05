"""
Load and filter MultiWOZ 2.2 dialogues from the official GitHub repository.

Dataset downloaded locally via:
    git clone https://github.com/budzianowski/multiwoz.git data/multiwoz_github
"""
import json

def _get_config():
    """Lazy import to avoid circular imports at module load time."""
    from src.experiments.config import MULTIWOZ_DIR, TARGET_DOMAINS
    return MULTIWOZ_DIR, TARGET_DOMAINS


def _load_dialog_acts() -> dict:
    """
    Load dialog_acts.json from official MultiWOZ 2.2.

    Returns:
        dict keyed by dialogue_id → turn_id → act data
    """
    MULTIWOZ_DIR, _ = _get_config()
    path = MULTIWOZ_DIR / "dialog_acts.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _attach_dialog_acts(dialogues: list[dict], dialog_acts: dict) -> list[dict]:
    """
    Attach dialog act annotations to each turn from dialog_acts.json.

    Adds 'dialog_act' key to each turn dict directly.

    Args:
        dialogues: list of dialogue dicts from load_all_dialogues()
        dialog_acts: dict loaded from dialog_acts.json

    Returns:
        dialogues with 'dialog_act' attached to each turn
    """
    for dialogue in dialogues:
        dialogue_id = dialogue["dialogue_id"]
        acts_for_dialogue = dialog_acts.get(dialogue_id, {})

        for turn in dialogue["turns"]:
            turn_id = str(turn["turn_id"])
            act_data = acts_for_dialogue.get(turn_id, {})
            # Attach act keys only (ignore span_info)
            turn["dialog_act"] = act_data.get("dialog_act", {})

    return dialogues


def load_all_dialogues(split: str) -> list[dict]:
    """
    Load all dialogue files from a split folder into one flat list.

    Args:
        split: one of 'train', 'dev', 'test'
    Returns:
        list of all dialogue dicts in that split
    """
    MULTIWOZ_DIR, _ = _get_config()
    split_dir = MULTIWOZ_DIR / split
    files = sorted(split_dir.glob("dialogues_*.json"))

    all_dialogues = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            all_dialogues.extend(json.load(f))

    return all_dialogues


def filter_by_domains(dialogues: list[dict], domains: set[str] = None) -> list[dict]:
    """
    Filter dialogues to those whose services are all within the target domains.

    Args:
        dialogues: list of dialogue dicts from load_all_dialogues()
        domains: set of allowed domain strings (Default: TARGET_DOMAINS from config)
    Returns:
        filtered list of dialogue dicts
    """
    if domains is None:
        _, domains = _get_config()

    return [
        d for d in dialogues
        if all(s in domains for s in d["services"])
    ]


def load_split(split: str, verbose=False) -> list[dict]:
    """
    Load, filter, and annotate one split. Main entry point for all v2 code.

    Args:
        split: one of 'train', 'dev', 'test'
        verbose: If True, print loading messages (Default: False)
    Returns:
        filtered list of hotel/restaurant dialogue dicts with dialog_act annotations attached to each turn
    """
    dialogues = load_all_dialogues(split)
    filtered = filter_by_domains(dialogues)
    dialogue_acts = _load_dialog_acts()
    filtered = _attach_dialog_acts(filtered, dialogue_acts)

    if verbose:
        print(f"{split}: {len(dialogues)} total → {len(filtered)} hotel/restaurant only ({len(filtered)/len(dialogues)*100:.1f}%)")

    return filtered
