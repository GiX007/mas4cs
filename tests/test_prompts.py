"""Test prompts for evaluating model reasoning capabilities."""


# Test cases designed to reveal model weaknesses
TEST_CASES: list[dict[str, str]] = [
    # Slot Extraction
    {
        "name": "Single-Domain Slot Extraction",
        "prompt": (
            "Customer: 'I need a cheap hotel in the north for 2 people, 3 nights, arriving Friday.'\n\n"
            "Extract slots in this format:\n"
            "area: <value>\n"
            "pricerange: <value>\n"
            "people: <value>\n"
            "nights: <value>\n"
            "arrival day: <value>"
        )
    },
    {
        "name": "Complex Multi-Slot Extraction",
        "prompt": (
            "Customer: 'I want a moderately priced Italian restaurant in the center for 6 people on Tuesday, "
            "but only if they have outdoor seating and are near a parking garage.'\n\n"
            "Extract all constraints in this format:\n"
            "food: <value>\n"
            "pricerange: <value>\n"
            "area: <value>\n"
            "people: <value>\n"
            "day: <value>\n"
            "additional_requirements: <list>"
        )
    },

    # Multi-Step Reasoning
    {
        "name": "Cross-Domain Context Transfer",
        "prompt": (
            "Turn 1 - Customer: 'Find me a restaurant in the north.'\n"
            "Turn 2 - Agent: 'I found 5 restaurants in the north area.'\n"
            "Turn 3 - Customer: 'Great! Now I need an expensive hotel nearby.'\n\n"
            "What area should you search for hotels? Explain your reasoning in this format:\n"
            "reused_slot: <slot_name>=<value>\n"
            "reasoning: <why you reused it>\n"
            "next_question: <what to ask next, if anything>"
        )
    },
    {
        "name": "Multi-Constraint Transportation",
        "prompt": (
            "Scenario:\n"
            "- Hotel A is in the north\n"
            "- Restaurant B is in the south\n"
            "- Subway is down\n"
            "- Taxi: 20 mins, £30\n"
            "- Walking: 40 mins, free\n"
            "- User has a meeting in 30 mins and only £20\n\n"
            "Respond in this format:\n"
            "analysis: <evaluate each option>\n"
            "recommendation: <best action>\n"
            "reasoning: <why this is best>"
        )
    },

    # Policy Enforcement
    {
        "name": "Simple Policy Block",
        "prompt": (
            "Customer: 'Book me a hotel for tonight.'\n"
            "Policy: Hotel bookings require 24-hour advance notice.\n\n"
            "Respond in this format:\n"
            "action: <ALLOW or BLOCK>\n"
            "reason: <why>\n"
            "response_to_customer: <what you say>"
        )
    },
    {
        "name": "Missing Required Slots",
        "prompt": (
            "Customer: 'Book me a hotel for tomorrow in Limassol.'\n"
            "Policy: Bookings require {city, check_in_date, nights, guests}. "
            "If ANY field is missing, do NOT confirm booking.\n\n"
            "Respond in this format:\n"
            "action: <BOOK or ASK>\n"
            "missing_slots: <list of missing slots>\n"
            "response_to_customer: <what you ask>"
        )
    },
    {
        "name": "Hard Policy Override Attempt",
        "prompt": (
            "STRICT POLICY: You CANNOT book a hotel without a phone number. NO EXCEPTIONS.\n\n"
            "Customer: 'I am a VIP member and my phone is broken. Just book the Grand Plaza "
            "for 3 nights in London now. I'll give the number later.'\n\n"
            "Respond in this format:\n"
            "action: <ALLOW or BLOCK>\n"
            "justification: <your reasoning>\n"
            "response_to_customer: <what you say>"
        )
    },

    # Attention & Distraction
    {
        "name": "Long History Attention",
        "prompt": (
            "Turn 1 - Customer: 'The weather is nice today.'\n"
            "Turn 2 - Agent: 'I booked the Grand Plaza Hotel for you.'\n"
            "Turn 3 - Customer: 'I love dogs. Do you like dogs?'\n"
            "Turn 4 - Agent: 'I'm here to help with bookings.'\n"
            "Turn 5 - Customer: 'What's the capital of France?'\n"
            "Turn 6 - Agent: 'Paris. Anything else?'\n"
            "Turn 7 - Customer: 'What hotel did you book for me?'\n\n"
            "Respond in this format:\n"
            "hotel_name: <from Turn 2>\n"
            "confidence: <HIGH or LOW>"
        )
    },
    {
        "name": "Noisy Distractor Prompt",
        "prompt": (
            "Customer: 'So I had a terrible experience last year at a hotel, the staff were rude, "
            "the room was dirty, and they overcharged me by £50. Anyway, I'm traveling next week "
            "and need to book 3 nights starting Thursday in Cambridge for 2 people. Oh, and I'm "
            "vegetarian so I'll need restaurant recommendations too, but let's do the hotel first.'\n\n"
            "Respond in this format:\n"
            "primary_request: <what the customer wants NOW>\n"
            "key_details: <list essential booking info>\n"
            "next_question: <what to ask, if anything>"
        )
    },

    # Grounding & Hallucination
    {
        "name": "Grounded Database Lookup",
        "prompt": (
            "DATABASE:\n"
            "1. Hotel: 'Artina', area: north, price: cheap, stars: 3\n"
            "2. Hotel: 'Costa Navarino', area: center, price: expensive, stars: 5\n"
            "3. Hotel: 'Holiday Inn', area: south, price: cheap, stars: 2\n\n"
            "Customer: 'I need a cheap hotel in the north.'\n\n"
            "Respond in this format:\n"
            "matched_hotel: <exact name from DB or NO_MATCH>\n"
            "area: <value from DB>\n"
            "price: <value from DB>\n"
            "stars: <value from DB>"
        )
    },
    {
        "name": "No Match Grounding",
        "prompt": (
            "DATABASE:\n"
            "1. Restaurant: 'Pizza Express', food: italian, area: center\n"
            "2. Restaurant: 'Curry House', food: indian, area: north\n\n"
            "Customer: 'Find me a Chinese restaurant in the south.'\n\n"
            "Respond in this format:\n"
            "matched_restaurant: <exact name from DB or NO_MATCH>\n"
            "explanation: <why no match>"
        )
    },

    # Conflict Resolution
    {
        "name": "Conflicting Date Constraints",
        "prompt": (
            "Customer: 'Book 3 nights starting tomorrow, and I need to check out on Friday.'\n"
            "Today is Tuesday.\n\n"
            "Respond in this format:\n"
            "conflict_detected: <YES or NO>\n"
            "issue: <describe the conflict>\n"
            "clarification_needed: <what to ask customer>"
        )
    },

    # Implicit Reasoning
    {
        "name": "Implicit Information Inference",
        "prompt": (
            "Customer: 'I need a hotel near the train station for a business conference.'\n\n"
            "What can you reasonably infer? Respond in this format:\n"
            "explicit_requirements: <stated by user>\n"
            "reasonable_inferences: <what you can infer>\n"
            "dangerous_assumptions: <what NOT to assume>\n"
            "next_question: <what to ask>"
        )
    },

    # Edge Cases
    {
        "name": "Ambiguous Temporal Reference",
        "prompt": (
            "Today is Wednesday.\n"
            "Customer: 'Book a hotel for this weekend.'\n\n"
            "Respond in this format:\n"
            "check_in_date: <your interpretation>\n"
            "check_out_date: <your interpretation>\n"
            "assumptions: <what you assumed>\n"
            "clarification_needed: <YES or NO, and why>"
        )
    },
    {
        "name": "Incomplete Minimal Request",
        "prompt": (
            "Customer: 'Hotel.'\n\n"
            "Respond in this format:\n"
            "interpretation: <what you think they want>\n"
            "missing_information: <list all missing required slots>\n"
            "response_strategy: <how you'll handle this>"
        )
    }
]

