"""
MultiWOZ schema constants for hotel & restaurant domains.

Used by agents, evaluation metrics, and validation.
"""


# Valid domains
VALID_DOMAINS: list[str] = ["hotel", "restaurant"]

# Valid intents (all)
VALID_INTENTS: list[str] = [
    "book_hotel",
    "book_restaurant",
    "find_hotel",
    "find_restaurant"
]

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

# Policy requirements for booking actions -> These slots MUST be present before allowing a booking to proceed
BOOKING_REQUIRED_SLOTS: dict[str, list[str]] = {
    "book_hotel": ["name", "bookday", "bookpeople", "bookstay"],
    "book_restaurant": ["name", "bookday", "bookpeople", "booktime"]
}


# General policy requirements
# POLICY_RULES = {
#
# }


# Slot value normalization (American → British spelling)
SLOT_VALUE_NORMALIZATION: dict[str, str] = {
    "center": "centre",
}

