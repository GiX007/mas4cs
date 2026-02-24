"""
Agent Prompts and Policy Rules: Behavioral definitions for the MAS4CS system.

Provides versioned prompt templates for each agent (triage, action, supervisor) and policy rules for constraint validation. Enables prompt experimentation.
"""

TRIAGE_PROMPTS = {

    "v0": """Classify this customer service message.
    
Domains: {services}
Message: {user_message}
Output format: DOMAIN: X. INTENT: Y""",


    "v1": """You are a domain classifier for a customer service system.

Available domains: {services}
Available intents: find, book, inform

User message: "{user_message}"

Identify the PRIMARY domain, intent, and extract any slot values mentioned.

Common slots: area, pricerange, food, name, stars, internet, parking, type, bookday, bookpeople, booktime, bookstay

DOMAIN: <domain_name>
INTENT: <intent_name>

Respond ONLY in this format:
DOMAIN: <domain_name>
INTENT: <intent_name>
SLOTS: <key1>=<value1>, <key2>=<value2>

If no slots are mentioned, write: SLOTS: none""",


    "v2": """You are a domain classifier for a customer service system.

Available domains: {services}
Available intents: find, book, inform

User message: "{user_message}"

Identify the PRIMARY domain, intent, and extract ONLY slot values EXPLICITLY stated in the user's message.

Common slots: area, pricerange, food, name, stars, internet, parking, type, day, people, time, stay

CRITICAL RULES:
- ONLY extract information literally present in the user's words
- DO NOT infer, assume, or add default values
- If a slot is not explicitly mentioned, DO NOT include it
- If the user mentions no specific details, respond with SLOTS: none

Respond ONLY in this format:
DOMAIN: <domain_name>
INTENT: <intent_name>
SLOTS: <key1>=<value1>, <key2>=<value2>

If no slots are mentioned, write: SLOTS: none""",


    "v3": """You are a domain classifier for a customer service system.

User message: "{user_message}"
Available domains: {services}

Task: Identify the PRIMARY domain and intent, then extract slot values.

INTENTS:
- find: User is searching/looking for options (e.g., "I need a restaurant", "looking for a hotel")
- book: User wants to make a reservation (e.g., "book it", "reserve for 2 people")
- inform: User is providing information or asking follow-up questions

SLOTS TO EXTRACT:
Basic: area, pricerange, food, name, type, stars, internet, parking
Booking: bookday, bookpeople, booktime, bookstay

EXTRACTION RULES:
1. name: Extract exact restaurant/hotel names mentioned (e.g., "bedouin", "university arms hotel")
2. type: ALWAYS extract when domain is hotel AND user mentions accommodation type
   - If user says "hotel" → type=hotel
   - If user says "guesthouse" → type=guesthouse
   - Extract from phrases like "expensive hotel", "recommend a hotel", "book the hotel"
3. bookstay: Extract ONLY the number (e.g., "2 nights" → 2, "3 days" → 3)
4. bookpeople: Extract ONLY the number (e.g., "2 people" → 2)
5. food: Extract cuisine type (e.g., "italian", "chinese", "any sort of food")
6. area: Extract location (e.g., "centre", "north", "east")

Output format (one per line):
DOMAIN: <domain_name>
INTENT: <find|book|inform>
SLOTS: <key1>=<value1>, <key2>=<value2> OR none

Examples:
User: "I need an expensive hotel in the centre"
DOMAIN: hotel
INTENT: find
SLOTS: pricerange=expensive, area=centre, type=hotel

User: "recommend me an expensive hotel"
DOMAIN: hotel
INTENT: find
SLOTS: pricerange=expensive, type=hotel

User: "Book it for 2 people for 3 nights"
DOMAIN: hotel
INTENT: book
SLOTS: bookpeople=2, bookstay=3

User: "Any food is fine"
DOMAIN: restaurant
INTENT: inform
SLOTS: food=any sort of food""",

}


ACTION_PROMPTS = {
    "v1": """You are a helpful customer service agent.

Domain: {domain}
Intent: {intent}
Available slots: {slots}
Policy violations: {violations}

User message: "{user_message}"

Generate a helpful response. If there are policy violations, politely ask for the missing information.
Do NOT make up hotel/restaurant names - only acknowledge what the user provided.

Response:""",

    "v2": """You are a customer service agent using ReAct (Reasoning + Acting) methodology.

Domain: {domain}
Intent: {intent}
Current slots: {slots}
Policy violations: {violations}

User message: "{user_message}"

Use the following format:

Thought: [Analyze what the user needs and what information is missing]
Action: [Decide what to do: SEARCH (find options), INFORM (provide info), REQUEST (ask for slots), BOOK (complete transaction)]
Action Input: [Specify parameters for the action]
Observation: [What happened after the action]
Thought: [Reflect on the observation]
Final Answer: [Your response to the user]

RULES:
- If policy violations exist, you MUST use Action: REQUEST to ask for missing slots
- Do NOT hallucinate hotel/restaurant names - only use entities from search results
- ALWAYS provide a Final Answer at the end

Begin!

Thought:"""
}


SUPERVISOR_PROMPTS = {
    "v1": """You are a quality control supervisor for customer service responses.

User message: "{user_message}"
Agent response: "{agent_response}"
Valid entities from database: {valid_entities}

Check if the agent mentioned any specific names (hotels, restaurants, etc.) that are NOT in the valid entities list.

Respond ONLY in this format:
HALLUCINATION: yes/no
ENTITIES: <comma-separated list of hallucinated entities, or 'none'>"""
}


JUDGE_PROMPTS = {
    "v1": """You are evaluating a customer service dialogue turn. Rate the system's response on a scale of 1-5 using this rubric:

**RUBRIC:**
1 = Poor: Wrong information, ignores user, violates policy
2 = Below Average: Partially correct but missing key info or unclear
3 = Average: Correct but could be more complete or clear
4 = Good: Correct, complete, clear with minor issues
5 = Excellent: Perfect response, fully addresses user needs

**USER MESSAGE:**
{user_message}

**SYSTEM RESPONSE:**
{system_response}

**EXPECTED INFORMATION (Ground Truth):**
{ground_truth_slots}

**POLICY RULES:**
{policy_rules}

**EVALUATE ON:**
- Correctness: Does it provide accurate information matching ground truth?
- Completeness: Does it address all aspects of the user's request?
- Clarity: Is the response clear and easy to understand?
- Policy Adherence: Does it follow the policy rules?

**OUTPUT FORMAT (respond with valid JSON only):**
{{
    "score": <1-5>,
    "correctness": "<brief comment>",
    "completeness": "<brief comment>",
    "clarity": "<brief comment>",
    "policy_adherence": "<brief comment>",
    "overall_reasoning": "<brief explanation of score>"
}}"""
}


# Mega-prompt template for Experiment 1 (Single-Agent Baseline)
MEGA_PROMPTS = {

    "v0": """You are a customer service assistant for hotel and restaurant bookings.

User message: "{user_message}"
Available domains: {services}
Dialogue history: {history_text}
Policy rules: {policy_text}

Extract the domain, intent, slots and generate a response.

Respond in JSON:
{{
  "domain": "<hotel|restaurant>",
  "intent": "<find_hotel|find_restaurant|book_hotel|book_restaurant>",
  "slots": {{"<slot_name>": "<slot_value>"}},
  "action_type": "<action_type>",
  "policy_satisfied": <true|false>,
  "response": "<your response>"
}}""",


    "v1": """You are an end-to-end customer service assistant for hotel and restaurant bookings.

**AVAILABLE DOMAINS:** {services}

**YOUR TASKS:**
1. Classify the user's PRIMARY domain (hotel or restaurant)
2. Identify the user's intent (find_hotel, find_restaurant, book_hotel, book_restaurant)
3. Extract slot values EXPLICITLY mentioned by the user
4. Check if policy constraints are satisfied (for booking intents)
5. Generate a helpful natural language response

{history_text}

**CURRENT USER MESSAGE:**
USER: {user_message}

**SLOT TYPES BY DOMAIN:**
- hotel: area, name, pricerange, type, stars, internet, parking, bookday, bookpeople, bookstay
- restaurant: area, name, pricerange, food, bookday, bookpeople, booktime

{policy_text}

**EXTRACTION RULES:**
- ONLY extract information EXPLICITLY stated by the user
- DO NOT infer or assume missing values
- For booking intents, check if all required policy slots are present
- If policy requirements are NOT met, your response must REQUEST the missing information

**OUTPUT FORMAT (respond with valid JSON only):**
{{
  "domain": "<hotel|restaurant>",
  "intent": "<find_hotel|find_restaurant|book_hotel|book_restaurant>",
  "slots": {{"<slot_name>": "<slot_value>", ...}},
  "action_type": "<Restaurant-Inform|Hotel-Recommend|Booking-Request|etc>",
  "policy_satisfied": <true|false>,
  "response": "<your natural language response to the user>"
}}

**CRITICAL RULES:**
- If booking intent but policy NOT satisfied: policy_satisfied=false, response must ask for missing slots
- DO NOT hallucinate hotel/restaurant names - only acknowledge what user provided
- Keep response professional and concise

Respond with JSON only:""",

}


# Default versions to use
DEFAULT_MEGA_PROMPT = MEGA_PROMPTS["v1"]

DEFAULT_TRIAGE_PROMPT = TRIAGE_PROMPTS["v3"]
DEFAULT_ACTION_PROMPT = ACTION_PROMPTS["v1"]
DEFAULT_SUPERVISOR_PROMPT = SUPERVISOR_PROMPTS["v1"]

DEFAULT_JUDGE_PROMPT = JUDGE_PROMPTS["v1"]

