"""
DB query tools for MAS4CS agents.

Two public tools called by agents:
    find_entity → search hotel/restaurant DB, return matching entities
    book_entity → validate booking slots, return booking confirmation
"""
import json
import random
import string

from src.experiments import DB_DIR, MAX_DB_RESULTS
from src.data import normalize_slot_value


_DB_CACHE: dict[str, list[dict]] = {}

def load_db(domain: str) -> list[dict]:
    """
    Load hotel or restaurant DB JSON into memory. Cached after first load.

    Args:
        domain: one of 'hotel', 'restaurant'
    Returns:
        list of entity dicts from the DB
    """
    if domain not in _DB_CACHE:
        db_path = DB_DIR / f"{domain}_db.json"
        with open(db_path, "r", encoding="utf-8") as f:
            _DB_CACHE[domain] = json.load(f)
    return _DB_CACHE[domain]


def _normalize_belief_state(belief_state: dict) -> dict:
    """
    Normalize all slot values in a belief state dict.

    Args:
        belief_state: dict of {slot_name: value} e.g. {"hotel-area": "Center"}
    Returns:
        normalized belief state dict e.g. {"hotel-area": "centre"}
    """
    return {
        slot: normalize_slot_value(value)
        for slot, value in belief_state.items()
    }


def _match_entity(entity: dict, constraints: dict) -> bool:
    """
    Check if a single DB entity satisfies all constraints.

    Skips constraints where value is 'dontcare' or slot is missing from entity.

    Args:
        entity: one entity dict from the DB
        constraints: normalized belief state dict {slot_name: value}
    Returns:
        True if entity matches all constraints, False otherwise
    """
    for slot, value in constraints.items():
        # Skip empty or meaningless values as they not useful for DB filtering
        if value in ("dontcare", "none", "", "not mentioned", "any"):
            continue

        # Extract the field name from slot (e.g. "hotel-area" → "area")
        field = slot.split("-")[-1]

        # Skip if entity doesn't have this field
        if field not in entity:
            continue

        # Normalize entity value for fair comparison
        entity_value = normalize_slot_value(str(entity[field]))

        if entity_value != value:
            return False

    return True


def _generate_ref() -> str:
    """
    Generate a random booking reference number.

    Returns:
        8-character alphanumeric string e.g. 'AB3X9K2M'
    """
    chars = string.ascii_uppercase + string.digits
    return "".join(random.choices(chars, k=8))


# DB Tools
def find_entity(domain: str, belief_state: dict) -> list[dict]:
    """
    Search the DB for entities matching the belief state constraints.

    Args:
        domain: one of 'hotel', 'restaurant'
        belief_state: dict of {slot_name: value} from agent prediction, e.g. {"hotel-area": "north", "hotel-pricerange": "cheap"}
    Returns:
        list of matching entity dicts (max MAX_DB_RESULTS)
    """
    db = load_db(domain)
    normalized = _normalize_belief_state(belief_state)
    matches = [e for e in db if _match_entity(e, normalized)]
    return matches[:MAX_DB_RESULTS]


def book_entity(domain: str, belief_state: dict) -> dict:
    """
    Validate booking and return a booking confirmation with reference number.

    Calls find_entity internally to verify a matching entity exists before booking.

    Args:
        domain: one of 'hotel', 'restaurant'
        belief_state: dict of {slot_name: value} including booking slots
                      e.g. {"hotel-name": "acorn guest house", "hotel-bookday": "monday", "hotel-bookpeople": "2", "hotel-bookstay": "3"}
    Returns:
        dict with keys:
            success → bool
            ref → booking reference string (if success)
            entity → matched entity dict (if success)
            reason → failure reason string (if not success)
    """
    matches = find_entity(domain, belief_state)

    if not matches:
        return {
            "success": False,
            "ref": None,
            "entity": None,
            "reason": f"No {domain} found matching the given constraints."
        }

    # Book the first matching entity
    entity = matches[0]
    ref = _generate_ref()

    return {
        "success": True,
        "ref": ref,
        "entity": entity,
        "reason": None
    }
