## Evaluation Choices for MAS4CS

Having detected the most important features of the dataset, let's see now how MAS4CS can be evaluated.

Evaluation is performed at three levels: turn-level (micro), dialogue-level (meso), and dataset-level (macro). 
Metrics are aggregated hierarchically to ensure reproducibility and comparability. 
The hierarchical evaluation is orchestrated by `DialogueEvaluator` (turn + dialogue levels) and `DatasetEvaluator` (macro level), which aggregate individual metric calculations from `evaluation_metrics.py`.

**Test examples and detailed metric behavior can be found in `docs/evals_inspection/metrics_test_results.txt` and `docs/evals_inspection/evaluator_test_results.txt`.**

---

### 1. Domain Routing Accuracy
- **What it measures**: Did the system route the turn to the correct domain (hotel vs restaurant etc.)?
- **Required features**: `services`, `active_intent`, `turn_id`, `dialogue_id`
- **Method**: At each turn, compare predicted domain vs. domain inferred from ground truth `active_intent`
- **Example**:
  - predicted domain: `hotel`
  - ground truth active intent: `find_restaurant` (restaurant domain)
  - mismatch → incorrect (0)
- **Formula**:
  - Step 1 (Turn-level): DomainAcc_turn = 1 if correct domain, else 0
  - Step 2 (Dialogue-level): DomainAcc_dialogue = average(DomainAcc_turn over all turns in dialogue)
  - Step 3 (Dataset-level): DomainAcc_dataset = average(DomainAcc_dialogue over all dialogues)
- **MAS4CS Component Tested**: Triage (domain routing)

---

### 2. Intent Accuracy
- **What it measures**: Did the system identify the correct user task (intent) at each turn?
- **Required features**: `active_intent`, `turn_id`, `dialogue_id`
- **Method**: At each turn, compare predicted intent vs. ground-truth `active_intent`
- **Example**:
  - predicted: `active_intent`: `find_restaurant`
  - ground truth: `active_intent`: `find_restaurant`
  - match → correct (1). If predicted find_hotel → incorrect (0)
- **Formula**:
  - Step 1 (Turn-level): IntentAcc_turn = 1 if exact match, else 0
  - Step 2 (Dialogue-level): IntentAcc_dialogue = average(IntentAcc_turn over all turns in dialogue)
  - Step 3 (Dataset-level): IntentAcc_dataset = average(IntentAcc_dialogue over all dialogues)
- **MAS4CS Component Tested**: Triage (intent detection)
- **Note on Multiple Intents**: Multiple intents per turn are rare in MultiWOZ but possible in real systems, so:
  - **Detailed Metrics (Optional)**: Recall and precision can be calculated for diagnostic purposes:
    - **Missing prediction** (predicted: `find_hotel`, truth: `[find_hotel, book_hotel]`): Recall drops (0.5), precision stays perfect (1.0) - system didn't detect all intents
    - **Extra prediction** (predicted: `[find_hotel, book_hotel]`, truth: `find_hotel`): Precision drops (0.5), recall stays perfect (1.0) - system hallucinated an intent

---

### 3. Action-Type Accuracy
- **What it measures**: Did the system choose the correct dialogue act type (the right kind of action) at each turn?
- **Required features**: `predicted_act_type`, `ground_truth_act_type`
- **Method**: At each turn, compare predicted act type(s) vs. ground-truth act type(s) as sets (order doesn't matter)
  Exact match → correct (1), else incorrect (0)
- **Example**:
  - predicted `act_type`: `["Restaurant-Inform"]`
  - ground truth `act_type`: `["Restaurant-Inform"]`
  - match → correct (1). If predicted `["Restaurant-Request"]` → incorrect (0)
- **Formula**: 
  - Step 1 (Turn-level): ActAcc_turn = 1 if exact match, else 0
  - Step 2 (Dialogue-level): ActAcc_dialogue = average(ActAcc_turn over all turns in dialogue)
  - Step 3 (Dataset-level): ActAcc_dataset = average(ActAcc_dialogue over all dialogues)
  
  **Note**: `accuracy=1.0` only when predicted and GT sets are identical. Partial matches (recall/precision) are tracked but don't contribute to accuracy score
- **MAS4CS Component Tested**: Action (tool/act selection), Supervisor (validation)
- **Handling Multiple Action Types**: Since multiple action types per turn are common, we use set-based comparison:
  - **Missing act** (predicted: `[Restaurant-Inform]`, truth: `[Restaurant-Inform, Restaurant-Request]`): Recall = 0.5 (got 1/2), Precision = 1.0 (no false positives) - incomplete but accurate
  - **Extra act** (predicted: `[Hotel-Inform, Hotel-Request, Hotel-Book]`, truth: `[Hotel-Inform, Hotel-Request]`): Recall = 1.0 (got all required), Precision = 0.67 (2/3 correct) - complete but with hallucination
- **Future Extension**: Can be extended to include act slots (e.g. `Restaurant-Inform(area=north)`), enabling finer-grained evaluation of slot-level action correctness. 

---

### 4. Slot-level Accuracy
- **What it measures**: How many individual slot–value pairs were correctly tracked at each turn (less strict than JGA)?
- **Required features**: `slots_values`, `turn_id`
- **Method**: At each turn, compare each predicted slot–value pair with ground truth. Count how many match
- **Example**:
  - predicted: `restaurant-area = centre`, `restaurant-pricerange = expensive`
  - ground truth: `restaurant-area = centre`, `restaurant-pricerange = cheap`
  - 1 correct out of 2 → score = 0.5
  - **Implementation Note**: Both predicted and ground truth use **accumulated slots** (all slots mentioned up to this turn)
- **Formula**: 
  - Step 1 (Turn-level): SlotAcc_turn = (correct slot–value pairs at turn) / (total ground-truth slot–value pairs at turn)
  - Step 2 (Dialogue-level): SlotAcc_dialogue = average(SlotAcc_turn over all turns in dialogue)
  - Step 3 (Dataset-level): SlotAcc_dataset = average(SlotAcc_turn over all evaluated turns in dataset)
- **MAS4CS Component Tested**: Triage (slot extraction), Memory (state update)
- **Why Slot Accuracy Doesn't Penalize Extra Predictions**: 
  - Slot Accuracy is a **recall metric** - it measures "did we capture what the user said?"
  - It counts: (correct predictions) / (total ground-truth slots)
  - **Missing slots** reduce accuracy: If truth has 3 slots, but we predict only 2 correct ones → 2/3 = 0.67
  - **Extra slots are ignored**: If truth has 2 slots, and we predict those 2 plus 1 hallucinated → still 2/2 = 1.0
  - **Why this design?** Penalizing hallucinations is the job of the separate "Hallucination Rate" metric. This separation helps diagnose whether the system is:
    - Missing user inputs (low Slot Accuracy) 
    - Making things up (high Hallucination Rate)

---

### 5. Joint Goal Accuracy (JGA)
- **What it measures**: Did the system correctly track all user requirements/constraints (belief state) at each turn?
- **Required features**: `slots_values`, `turn_id`, `dialogue_id`
- **Method**: At each turn, compare predicted belief state vs. ground-truth `slots_values`
    If all slot–value pairs match exactly → turn = correct (1), otherwise turn → incorrect (0)
- **Example**:
  - predicted `slots_values`: `{'slots_values_name': ['restaurant-area', 'restaurant-pricerange'], 'slots_values_list': [['centre'], ['expensive']]}`
  - ground truth `slots_values`: `{'slots_values_name': ['restaurant-area', 'restaurant-pricerange'], 'slots_values_list': [['centre'], ['expensive']]}`
  - predicted matches exactly the ground truth → correct (1). If instead predicted `restaurant-pricerange = cheap` → incorrect (0)
  - **Implementation Note**: Both use accumulated slots
- **Formula**:
  - Step 1 (Turn-level): JGA_turn = 1 if exact match, else 0
  - Step 2 (Dialogue-level): JGA_dialogue = average(JGA_turn over all turns in dialogue)
  - Step 3 (Dataset-level): JGA_dataset = average(JGA_turn over all evaluated turns in dataset)
- **MAS4CS Component Tested**: Memory (state update), Triage (slot extraction)
- **Strictness**: JGA is an all-or-nothing metric. Both missing and extra predictions result in JGA = 0 for that turn. Use Slot Accuracy and Hallucination Rate for fine-grained diagnosis

---

### 6. Hallucination Rate
- **What it measures**: Did the system output slot values that are not supported by the dialogue text/annotations?
- **Required features**: `slots_values` (predicted accumulated), `frames.slots_values` (GT accumulated)
- **Method**: For each predicted slot-value pair, check if it exists exactly in GT
    
  Wrong domain, wrong slot name, or wrong value all count as hallucination.
- **Implementation Note**: Hallucination is checked per active domain only (not across all accumulated domains)
    
    This avoids penalizing correct cross-domain memory from previous turns.
- **What Counts as Hallucination**: Any predicted slot-value pair that doesn't exactly match ground truth:
  - **Case 1: Wrong domain** - User mentioned restaurant, system predicted hotel slots → hallucination
  - **Case 2: Wrong slot name** - Truth has `area`, system predicted `parking` → hallucination
  - **Case 3: Wrong slot value** - Truth has `area=centre`, system predicted `area=north` → hallucination
- **Example**:
  - predicted slot: `restaurant-area = north`
  - span_info contains only `south`
  - → hallucination (1 hallucinated value)
- **Formula**:
  - Step 1 (Turn-level): Hall_turn = (hallucinated values at turn) / (total predicted values at turn)
  - Step 2 (Dialogue-level): Hall_dialogue = average(Hall_turn over all turns in dialogue)
  - Step 3 (Dataset-level): Hall_dataset = average(Hall_turn over all evaluated turns in dataset)
- **MAS4CS Component Tested**: Supervisor (validation), Action
- **Relation to Slot Accuracy**: Hallucination Rate is the **precision** complement to Slot Accuracy's **recall**. Together they provide a complete picture of slot tracking quality

---

### 7. Cross-Domain Memory Transfer Accuracy
- **What it measures**: When the dialogue switches domain, did the system correctly carry over relevant constraints?
- **Required features**: `dialogue_history` (list of turn dicts with `domain`, `predicted_slots`, `turn_id`)
- **Method**:
  - Detect a domain shift (e.g., restaurant → hotel)
  - Check whether shared constraints (e.g., `area`, `pricerange`) that should carry over are present in the new domain's state
- **Example**:
  - earlier restaurant state: `restaurant-area = north`
  - later hotel search uses: `hotel-area = north` (carried over)
  - → correct transfer (1)
- **Transferable Slots**: By default, only `area` and `pricerange` are considered transferable (search constraints). Booking-specific slots (`bookday`, `bookpeople`, etc.) are NOT transferred as they are transaction-specific
- **Formula**:
  - Step 1 (Transfer Event-level): Transfer_event = 1 if expected carry-over happens, else 0
  - Step 2 (Dialogue-level): MemoryAcc_dialogue = (correct transfers) / (total transfer opportunities) in dialogue
  - Step 3 (Dataset-level): MemoryAcc_dataset = average(MemoryAcc_dialogue over dialogues with domain switches)
- **MAS4CS Component Tested**: Memory (state persistence), Triage (domain-shift handling)
- **Validation**: This metric directly measures success in multi-domain context carry-over
- **Note**: Only dialogues with domain switches contribute to this metric. Dialogues without switches are excluded from the average
- **Important**: Memory Transfer will only be non-zero when the user explicitly reuses constraints across domains (e.g. "find a hotel in the same area"). If user switches domain without reusing constraints, 0.0 is correct — not a bug

---

### 8. Policy Compliance Rate
- **What it measures**: Did the system follow required business rules (hard blocks), especially before booking?
- **Required features**: `active_intent`, `slots_values`, `dialogue_acts.dialog_act.act_type`, `turn_id`, `dialogue_id`
- **Method**: Define policy rules (example for hotel booking):
  - If intent is `book_hotel`, then before any `Booking-Book`, the state must include required slots (e.g., `hotel-bookstay`, `hotel-bookday`, `hotel-bookpeople`, `hotel-name`)
  - If the system books without them → violation
- **Example**:
  - predicted action: `book_hotel` (normalized from intent)
  - predicted state missing `bookstay`
  - → policy violation (0 = non-compliant)
- **Formula**:
  - Step 1 (Turn-level): Compliance_turn = 1 if policy followed, else 0
  - Step 2 (Dialogue-level): Violations_dialogue = count of turns where Compliance_turn = 0
  - Step 3 (Dataset-level): ViolationRate_dataset = (total violations across all dialogues) / (total turns across all dialogues)
- **MAS4CS Component Tested**: Policy (hard blocks)
- **Policy Requirements**: Defined in `src/data/dataset_constants.py` as `BOOKING_REQUIRED_SLOTS` based on the dataset schema
- **Note**: Policy compliance is tracked as a count of violations per dialogue, then aggregated as a rate across the dataset

---

### 9. System Correctness (System Perspective)
- **What it measures:** Did the system respond appropriately to what the user provided?
- **Purpose:** Measures whether the multi-agent system behaves correctly under all conditions (complete info, incomplete info, policy violations)
- **Success Criteria:**
  The system is correct if it performs the RIGHT action given the user's input:
  - **User provides search criteria** → System searches (action: search)
  - **User requests booking with complete slots** → System books (action: book)
  - **User requests booking with incomplete slots** → System requests missing info (action: request)
  - **User violates policy** → System blocks and explains (policy_compliant: False)
  - **User provides info** → System acknowledges (action: inform)
  - **Examples**:
    - **Example 1 - SYSTEM CORRECT (Handles Incomplete Booking):**
    - User: "I want to book a hotel for 2 people"
    - System: "I need the hotel name, check-in date, and number of nights to complete your booking"
    - Expected behavior: Request missing slots
    - Actual behavior: action = "request", asks for name/bookday/bookstay
    - → System Correctness = TRUE
  - **Example 2 - SYSTEM CORRECT (Enforces Policy):**
    - User: "Book the hotel, but I don't know how many nights"
    - System: "I cannot complete the booking without the duration. Please provide the number of nights."
    - Expected behavior: Block booking, request required slot
    - Actual behavior: policy_compliant = False, action = "request"
    - → System Correctness = TRUE
  - **Example 3 - SYSTEM INCORRECT (Hallucinates):**
    - User: "Find me an expensive restaurant in the center"
    - System: "I recommend The Golden Dragon, phone: 123-456-7890"
    - Expected behavior: Search and inform (no hallucination)
    - Actual behavior: Provides name/phone not in database
    - Hallucination detected: True
    - → System Correctness = FALSE
  - **Example 4 - SYSTEM INCORRECT (Wrong Action):**
    - User: "I want to book the Hilton for 2 people, Saturday, 2 nights"
    - System: "Let me search for hotels in that area..."
    - Expected behavior: Complete booking (all slots present) 
    - Actual behavior: action = "search" (should be "book")
    - → System Correctness = FALSE 
- **Formula:**
  - **Policy Gate:** If any policy violation detected → Task Completion = FALSE immediately
  - For each turn:
  
    SystemCorrect_turn = 1 if (expected_action == actual_action AND no_hallucination AND policy_compliant) else 0

  - For dialogue:
  
    SystemCorrectness = (correct_turns) / (total_turns)

**Note:** This metric tests the MAS4CS architecture's ability to handle real-world complexity (incomplete info, policy constraints).
High system correctness with low task completion indicates users are abandoning tasks, not that the system is failing.

---

### 10. Task Completion Rate (Customer Perspective)
- **What it measures:** Did the customer achieve their stated goal by the end of the dialogue?
- **Purpose:** Measures end-to-end dialogue success from the user's viewpoint
- **Success Criteria:**
  - **For booking goals:** All required slots are filled AND a booking action occurred
    - Hotel booking requires: `name`, `bookday`, `bookpeople`, `bookstay`
    - Restaurant booking requires: `name`, `bookday`, `bookpeople`, `booktime`
  - **For information goals:** All requested information was provided to the user
- **Examples**:
  - **Example 1 - TASK COMPLETED:**
    - User: "I want to book the University Arms Hotel for 2 people, 2 nights, starting Saturday"
    - System: "Booking confirmed for University Arms Hotel, 2 guests, Saturday for 2 nights"
    - Final slots: {hotel: {name: "university arms", bookday: "Saturday", bookpeople: "2", bookstay: "2"}}
    - Action: book_hotel
    - → Task Completion = TRUE (all slots + booking action)
  - **Example 2 - TASK INCOMPLETE (Missing Info):**
    - User: "I want to book a hotel for 2 people and 2 nights starting Saturday"
    - System: "I need the hotel name to complete your booking"
    - User: "That's all, goodbye"
    - Final slots: {hotel: {bookpeople: "2", bookstay: "2", bookday: "Saturday"}}
    - Action: request_info
    - → Task Completion = FALSE (missing required slot: name)
  - **Example 3 - TASK INCOMPLETE (No Booking Action):**
    - User: "I want to book the Hilton for 2 people, Saturday, 2 nights"
    - System: "Great choice! The Hilton is available. Would you like to proceed?"
    - User: "Actually, never mind. Goodbye"
    - Final slots: {hotel: {name: "Hilton", bookpeople: "2", bookstay: "2", bookday: "Saturday"}}
    - Action: inform
    - → Task Completion = FALSE (user had all slots but no booking action occurred)
- **Formula:**
  - Step 0 (Policy gate): If any policy violation detected → TaskCompletion = FALSE immediately
  - Step 1 (Dialogue-level): TaskCompletion_dialogue = 1 if goal completed, else 0
  - Step 2 (Dataset-level): TaskCompletionRate = (completed dialogues) / (total dialogues)

**Note:** This metric depends on user behavior (providing complete info). A low rate may indicate user abandonment, not necessarily system failure.

---

#### Task Completion Rate vs System Correctness

Key Difference Summary

| Scenario                                     | Task Completion | System Correctness |
|----------------------------------------------|-----------------|--------------------|
| User provides all info, system books         | ✅ TRUE          | ✅ TRUE             |
| User missing info, system asks for it        | ❌ FALSE         | ✅ TRUE             |
| User provides all info, system fails to book | ❌ FALSE         | ❌ FALSE            |
| System hallucinates during search            | ❌ FALSE         | ❌ FALSE            |

**Both metrics matter:**
- **Task Completion** = Customer satisfaction
- **System Correctness** = System reliability

**Design Rationale:** MAS4CS evaluation uses both perspectives because automated customer service must both satisfy customers (Task Completion) AND operate reliably under all conditions (System Correctness). This dual-metric approach distinguishes between user abandonment and system failure.

---

### 11. LLM-Judged Response Quality
- **What it measures**: Overall quality of the system's response (helpful, correct, clear)
- **Required features**: `utterance` (context), generated system response
- **Method**: Use an LLM judge to score each response on a fixed scale (e.g., 1–5) using a rubric (correctness, completeness, clarity, policy adherence)
- **Example**: judge score: 4/5 (correct and clear, minor missing detail)
- **Formula**:
  - Step 1 (Turn-level): Score_turn = judge score (1–5)
  - Step 2 (Dialogue-level): Score_dialogue = average(Score_turn over all turns in dialogue)
  - Step 3 (Dataset-level): Score_dataset = average(Score_dialogue over all dialogues)
- **MAS4CS Component Tested**: End-to-end MAS (final output quality)
- **Implementation**: Uses `DEFAULT_JUDGE_PROMPT` from `src/utils/prompts.py` with robust JSON parsing in `llm_judge.py` to handle Markdown-wrapped responses
- **Note**: Only available if `judge_llm_fn` is provided to `DialogueEvaluator`. Otherwise, this metric is skipped

---

## Summary

The 11 metrics provide comprehensive coverage:
- **Routing & Intent** (metrics 1, 2, 3): Does the system understand what the user wants and where to handle it, and what action to take?
- **Slot Tracking** (metrics 4, 5, 6): Does the system accurately, maintain, and avoid inventing user requirements?
- **Memory** (metric 7): Does the system maintain context across domain switches?
- **Policy & Task Completion** (metrics 6, 7, 8): Does the system follow rules, complete goals, and respond correctly to user inputs?
- **Policy & Correctness** (metrics 8, 9): Does the system follow rules and respond correctly to user inputs?
- **Task & Quality** (metrics 10, 11): Does the system complete user goals and produce high-quality responses?

All metrics are tested in `tests/test_evals.py` with detailed examples in `docs/evals_inspection/objective_and_judge_metrics.txt`. 

The hierarchical evaluation workflow is tested in `tests/test_evaluator.py` with results in `docs/evals_inspection/evaluator_test_results.txt`.

---
