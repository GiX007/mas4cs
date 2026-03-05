"""
MultiWOZ schema constants for hotel & restaurant domains.

Used by agents, evaluation metrics, and validation.
"""
# Valid domains
VALID_DOMAINS: list[str] = ["hotel", "restaurant"]

# Valid intents
VALID_INTENTS: list[str] = ["book_hotel", "book_restaurant", "find_hotel", "find_restaurant"]

# Intent categorization
BOOKING_INTENTS: list[str] = ["book_hotel", "book_restaurant"]
INFO_INTENTS: list[str] = ["find_hotel", "find_restaurant"]

# Valid action types (dialogue acts)
VALID_ACTION_TYPES: list[str] = [
    "Booking-Book",
    "Booking-Inform",
    "Booking-NoBook",
    "Booking-Request",
    "Hotel-Inform",
    "Hotel-NoOffer",
    "Hotel-Recommend",
    "Hotel-Request",
    "Hotel-Select",
    "Restaurant-Inform",
    "Restaurant-NoOffer",
    "Restaurant-Recommend",
    "Restaurant-Request",
    "Restaurant-Select",
    "general-bye",
    "general-greet",
    "general-reqmore",
    "general-thank",
    "general-welcome"
]

# Acts excluded from ActType evaluation (too generic, not domain-specific)
GENERAL_ACTS: set[str] = {
    "general-bye",
    "general-greet",
    "general-reqmore",
    "general-thank",
    "general-welcome"
}

# Slots by domain
SLOTS_BY_DOMAIN: dict[str, list[str]] = {
    "hotel": [
        "area",
        "bookday",
        "bookpeople",
        "bookstay",
        "internet",
        "name",
        "parking",
        "pricerange",
        "stars",
        "type"
    ],
    "restaurant": [
        "area",
        "bookday",
        "bookpeople",
        "booktime",
        "food",
        "name",
        "pricerange"
    ]
}

# Slots used for DB search constraints (what the user wants)
INFORMABLE_SLOTS = {"area", "name", "pricerange", "food", "type", "stars", "internet", "parking"}

# Slots used for booking details (when/how many)
BOOKING_SLOTS = {"bookday", "bookpeople", "booktime", "bookstay"}

# Policy requirements for booking actions -> These slots MUST be present before allowing a booking to proceed
BOOKING_REQUIRED_SLOTS: dict[str, list[str]] = {
    "book_hotel": ["name", "bookday", "bookpeople", "bookstay"],
    "book_restaurant": ["name", "bookday", "bookpeople", "booktime"]
}

# Slot value normalization (slot value cleaning): used by normalize_slot_value (triage, run_sa_turn) and tools.py (_normalize_value)
SLOT_VALUE_NORMALIZATION: dict[str, str] = {
    # British spelling
    "center": "centre",
    # Dontcare variants
    "any": "dontcare",
    "doesn't matter": "dontcare",
    "do not care": "dontcare",
    "don't care": "dontcare",
    "not mentioned": "dontcare",
    # Food=none variants (LLM returns "none" when user says "no preference")
    "none": "dontcare",
}
