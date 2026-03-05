## Evaluation Choices for MAS4CS

Having detected the most important features of the dataset, let's see now how MAS4CS can be evaluated.

Evaluation is performed at three levels: turn-level (micro), dialogue-level (meso), and dataset-level (macro). 
Metrics are aggregated hierarchically to ensure reproducibility and comparability. 
The hierarchical evaluation is orchestrated by `DialogueEvaluator` (turn + dialogue levels) and `DatasetEvaluator` (macro level), which aggregate individual metric calculations from `evaluation_metrics.py`.

MAS4CS uses two complementary evaluation frameworks:

1. **Custom Metrics (Metrics 1–10):** Designed specifically for MAS4CS to measure dimensions not covered by standard benchmarks, such as policy enforcement and booking completion.
2. **Official MultiWOZ Evaluation:** The standard community benchmark script by Tomiinek et al., providing **Inform Rate**, **Success Rate**, **BLEU**, and **Combined Score** for direct comparison with existing literature.

---

## MAS4CS Custom Metrics

Metrics 1–10 are custom metrics designed specifically for MAS4CS. They target dimensions of customer service performance not covered by standard MultiWOZ benchmarks, such as policy enforcement, entity hallucination rate, and booking completion.

---

### 1. Domain Routing Accuracy
- **What it measures:** Did the system route the turn to the correct domain (hotel vs restaurant etc.)?
- **Required features:** `services`, `active_intent`, `turn_id`, `dialogue_id`
- **Method:** At each turn, compare predicted domain vs. domain inferred from ground truth `active_intent`
- **Example:**
  - predicted domain: `hotel`
  - ground truth active intent: `find_restaurant` (restaurant domain)
  - mismatch → incorrect (0)
- **Formula:**
  - Step 1 (Turn-level): DomainAcc_turn = 1 if correct domain, else 0
  - Step 2 (Dialogue-level): DomainAcc_dialogue = average(DomainAcc_turn over all turns in dialogue)
  - Step 3 (Dataset-level): DomainAcc_dataset = average(DomainAcc_dialogue over all dialogues)
- **MAS4CS Component Tested:** Triage (domain routing)

---

### 2. Intent Accuracy
- **What it measures:** Did the system identify the correct user task (intent) at each turn?
- **Required features:** `active_intent`, `turn_id`, `dialogue_id`
- **Method:** At each turn, compare predicted intent vs. ground-truth `active_intent`
- **Example:**
  - predicted: `active_intent`: `find_restaurant`
  - ground truth: `active_intent`: `find_restaurant`
  - match → correct (1). If predicted find_hotel → incorrect (0)
- **Formula:**
  - Step 1 (Turn-level): IntentAcc_turn = 1 if exact match, else 0
  - Step 2 (Dialogue-level): IntentAcc_dialogue = average(IntentAcc_turn over all turns in dialogue)
  - Step 3 (Dataset-level): IntentAcc_dataset = average(IntentAcc_dialogue over all dialogues)
- **MAS4CS Component Tested:** Triage (intent detection)
- **Note on Multiple Intents:** Multiple intents per turn are rare in MultiWOZ but possible in real systems, so:
  - **Detailed Metrics (Optional):** Recall and precision can be calculated for diagnostic purposes:
    - **Missing prediction** (predicted: `find_hotel`, truth: `[find_hotel, book_hotel]`): Recall drops (0.5), precision stays perfect (1.0) - system didn't detect all intents
    - **Extra prediction** (predicted: `[find_hotel, book_hotel]`, truth: `find_hotel`): Precision drops (0.5), recall stays perfect (1.0) - system hallucinated an intent

---

### 3. Action-Type Accuracy
- **What it measures:** Did the system choose the correct dialogue act type (the right kind of action) at each turn?
- **Required features:** `predicted_act_type`, `ground_truth_act_type`
- **Method**: At each turn, compare predicted act type(s) vs. ground-truth act type(s) as sets (order doesn't matter). Two scores are computed: strict exact match (ActType-R%) and F1 (ActType-F1%)
  Exact match → correct (1), else incorrect (0)
- **Example:**
  - predicted `act_type`: `["Restaurant-Inform"]`
  - ground truth `act_type`: `["Restaurant-Inform", "Restaurant-Request"]`
  - strict match → incorrect (0). F1 = 0.67 (partial credit)
- **Formula:** 
  - Step 1 (Turn-level): ActAcc_turn = 1 if exact match, else 0 and ActF1_turn = F1(predicted set, GT set) (partial credit)
  - Step 2 (Dialogue-level): average(turn scores)
  - Step 3 (Dataset-level): average(turn scores)
- **MAS4CS Component Tested:** Action (tool/act selection), Supervisor (validation)
- **Handling Multiple Action Types:** 
  - **Missing act** (predicted: `[Restaurant-Inform]`, truth: `[Restaurant-Inform, Restaurant-Request]`): Recall = 0.5, Precision = 1.0 
  - **Extra act** (predicted: `[Hotel-Inform, Hotel-Request, Hotel-Book]`, truth: `[Hotel-Inform, Hotel-Request]`): Recall = 1.0, Precision = 0.67
- **Future Extension:** Can be extended to include act slot-value pairs (e.g. `Restaurant-Inform(area=north)`), for finer-grained evaluation

---

### 4. Slot-level Accuracy
- **What it measures:** How many individual slot–value pairs were correctly tracked at each turn (less strict than JGA)?
- **Required features:** `slots_values`, `turn_id`
- **Method:** At each turn, compare each predicted slot–value pair with ground truth. Count how many match. Both predicted and ground truth use **accumulated slots** (all slots mentioned up to this turn)
- **Example:**
  - predicted: `restaurant-area = centre`, `restaurant-pricerange = expensive`
  - ground truth: `restaurant-area = centre`, `restaurant-pricerange = cheap`
  - 1 correct out of 2 GT slots → recall = 0.5
  - **Implementation Note:** Both predicted and ground truth use **accumulated slots** (all slots mentioned up to this turn)
- **Formula:** 
  - Step 1 (Turn-level): SlotRecall_turn = correct / GT slots, SlotPrecision_turn = correct / predicted slots, SlotF1_turn = F1(recall, precision)
  - Step 2 (Dialogue-level): average(turn scores)
  - Step 3 (Dataset-level): average(turn scores)
- **MAS4CS Component Tested:** Triage (slot extraction), Memory (state update)
- **Design note:** Slot Accuracy is a recall metric. It measures "did we capture what the user said?". 
  Extra predicted slots are not penalized here. That is the job of the Hallucination Rate metric. 
  This separation helps diagnose whether the system is missing user inputs (low Slot-R%) or inventing information (high Hall%)

---

### 5. Joint Goal Accuracy (JGA)
- **What it measures:** Did the system correctly track ALL user requirements across ALL domains at each turn? (strict, all-or-nothing)
- **Required features:** `slots_values`, `turn_id`, `dialogue_id`
- **Method:** At each turn, compare predicted belief state vs. ground-truth `slots_values`. 
    If all slot–value pairs match exactly → correct (1), otherwise → incorrect (0). Both use accumulated slots
- **Example:**
  - predicted: `{restaurant: {area: centre, pricerange: expensive}}`
  - ground truth `{restaurant: {area: centre, pricerange: expensive}}`
  - exact match → correct (1). If `pricerange = cheap` → incorrect (0)
- **Formula:**
  - Step 1 (Turn-level): JGA_turn = 1 if exact match across all domains, else 0
  - Step 2 (Dialogue-level): JGA_dialogue = average(JGA_turn over all turns in a dialogue)
  - Step 3 (Dataset-level): JGA_dataset = average(JGA_dialogue over all dialogues in the dataset)
- **MAS4CS Component Tested:** Memory (state update), Triage (slot extraction)
- **Strictness:** Any single wrong slot → JGA = 0 for that turn. Use Slot Accuracy for partial credit diagnosis

---

### 6. Hallucination Rate
- **What it measures:** Did the system mention hotel or restaurant names in its response that were NOT returned by the database for that turn?
- **Novel contribution:** Not a standard MultiWOZ metric. Enabled by our DB integration and measures response grounding and trustworthiness directly in the customer service context
- **Method:** For each turn where a DB query was made, check if any entity name in the system response is absent from the DB results returned for that turn
- **Example:**
  - DB returned: `["The Varsity Restaurant"]`
  - System response mentions: `"The Golden Dragon"` (not in DB results)
  - → hallucination detected
- **Formula:**
  - Step 1 (Turn-level): Hall_turn = hallucinated_entities / entities_mentioned. Skipped if no DB query was made this turn (valid_entities = [])
  - Step 2 (Dialogue-level): average(Hall_turn). Only over the turns where entities were mentioned
  - Step 3 (Dataset-level): average(Hall_dialogue). Only over the dialogues with at least one DB query turn
- **MAS4CS Component Tested:** Action (DB grounding), Supervisor (response validation)

---

### 8. Policy Compliance Rate
- **What it measures:** Did the system attempt a booking only when all required slots were present?
- **Novel contribution:** Not a standard MultiWOZ metric. Enforces business rules: system must not attempt booking without checking required slots (directly relevant to safe deployment of customer service automation)
- **Required features:** `active_intent`, `slots_values`, `turn_id`, `dialogue_id`
- **Method:** If intent is `book_hotel` or `book_restaurant`, check that all required slots are present in the accumulated state before allowing the booking action
- **Example:**
  - predicted action: `book_hotel`
  - predicted state missing `bookstay`
  - → policy violation (non-compliant)
- **Formula:**
  - Step 1 (Turn-level): Compliance_turn = 1 if policy followed, else 0
  - Step 2 (Dialogue-level): Violations_dialogue = count of turns where Compliance_turn = 0
  - Step 3 (Dataset-level): ViolationRate_dataset = total_violations / total_turns
- **MAS4CS Component Tested:** Policy (hard blocks)
- **Policy Requirements:** Defined in `src/data/dataset_constants.py` as `BOOKING_REQUIRED_SLOTS` 

---

### 9. System Correctness (System Perspective)
- **What it measures:** Did the system respond correctly for this turn?
- **Novel contribution:** Composite binary metric: a turn is correct if and only if the system neither invented entities nor violated booking policy
- **Method:** A turn is correct if both conditions hold: no hallucination detected AND policy compliant
- **Formula:**
  - Step 1 (Turn-level): SystemCorrect_turn = 1 if (no_hallucination AND policy_compliant), else 0
  - Step 2 (Dialogue-level): SystemCorrectness_dialogue = average(SystemCorrect_turn)
  - Step 3 (Dataset-level): SystemCorrectness_dataset = average(SystemCorrectness_dialogue)
- **Examples:**
  - System recommends a restaurant not in DB → hallucination → SystemCorrect = FALSE
  - System attempts booking with missing slots → policy violation → SystemCorrect = FALSE
  - No hallucination, no violation → SystemCorrect = TRUE
- **MAS4CS Component Tested:** End-to-end MAS reliability

---

### 10. Task Completion Rate (Customer Perspective)
- **What it measures:** Did the system successfully complete a booking when one was required by the end of the dialogue?
- **Novel contribution:** Not a standard MultiWOZ metric. Directly measures end-to-end booking completion. The primary success criterion for customer service automation. Complements Policy Compliance: policy checks per-turn slot requirements, Booking Success checks the final dialogue outcome
- **Method:** For dialogues where a booking intent was detected, check that: all required domains were addressed, a booking action occurred, and all required slots were filled in the final state
- **Formula:**
  - Step 1 (Dialogue-level): BookingSuccess_dialogue = True/False/None
    - None if dialogue had no booking intent → excluded from aggregation
    - False if booking action never occurred or required slots missing at end
    - True if all conditions met
  - Step 2 (Dataset-level): BookingSuccessRate = sum(True) / count(non-None dialogues)
- **Examples:**
  - Info-only dialogue (find_restaurant, no booking) → None → excluded from rate
  - Booking dialogue, all slots filled, `book_hotel` action occurred → True
  - Booking dialogue, `book_hotel` action occurred but `bookstay` missing → False
- **MAS4CS Component Tested:** End-to-end MAS 

---

### 11. LLM-Judged Response Quality
- **What it measures:** Overall quality of the system's natural language response (correct, complete, clear)
- **Required features:** `utterance` (context), generated system response
- **Method:** Use an LLM judge to score each response on a 1–5 scale using a fixed rubric (correctness, completeness, clarity, policy adherence)
- **Example:** judge score: 4/5 (correct and clear, minor missing detail)
- **Formula:**
  - Step 1 (Turn-level): Score_turn = judge score (1–5)
  - Step 2 (Dialogue-level): Score_dialogue = average(Score_turn over all turns in dialogue)
  - Step 3 (Dataset-level): Score_dataset = average(Score_dialogue over all dialogues)
- **MAS4CS Component Tested:** End-to-end MAS (final output quality)
- **Implementation:** Uses `DEFAULT_JUDGE_PROMPT` from `src/utils/prompts.py` with robust JSON parsing in `llm_judge.py` 
- **Note:** Disabled by default (`judge_llm_fn=None`) to avoid extra API costs. Intended to run separately on a subset of dialogues for qualitative analysis

---

## Official MultiWOZ Evaluation (Tomiinek et al.)

MAS4CS is also evaluated using the official MultiWOZ evaluation script by Tomiinek et al. (2021), which provides three standard community metrics for direct comparison with existing literature.

Official script: https://github.com/Tomiinek/MultiWOZ_Evaluation

### What MAS4CS passes to Tomiinek
For each dialogue turn, MAS4CS provides:
- **response:** the system's natural language response (lexicalized — contains real entity names like "Oasis" instead of placeholders like [restaurant_name])
- **active_domains:** the domain predicted by the Triage agent for that turn (e.g. ["restaurant"])
- **state:** (optional) the ground truth belief state

The predicted belief state is omitted. Tomiinek uses the **ground truth belief state from MultiWOZ 2.2** to determine what constraints the user had and what attributes they requested.
It then evaluates whether the system responses satisfied those constraints and requests.

**How entity matching works:**
- Leaderboard models generate delexicalized responses like `"The restaurant is [restaurant_name]"`. Tomiinek replaces the placeholder with the real entity name from its internal map, then checks the DB. BLEU is computed against the delexicalized reference  (high overlap)
- MAS4CS generates lexicalized responses like `"I recommend The Nirala"`. Tomiinek searches directly for known entity names from the DB in the response text. Inform/Success work correctly. 
  BLEU fails because `"The Nirala"` has zero overlap with `"[restaurant_name]"`.

---

### Inform Rate
**"Did the system find/offer the RIGHT entity?"**

Measures whether the system mentioned an entity (hotel or restaurant) that matches the user's goal constraints, as defined in the MultiWOZ 2.2 ground truth goals.

- Tomiinek scans all system responses in the dialogue and checks whether at any point the system mentioned an entity matching ALL the user's constraints (area, pricerange, food type etc.) from the GT belief state
- Entity matching is performed against the **official MultiWOZ database** (same DB used during training and evaluation by all leaderboard models)
- Binary per dialogue: 1 if a matching entity was mentioned at any turn, else 0
- Example: user wants a cheap restaurant in the north → system mentions "Oasis" 
  (cheap, north) → Inform = 1. If system mentions a non-matching restaurant → 0
- Dataset score = (correct dialogues / total dialogues) × 100. A score of 40 means 40% of dialogues had a correctly informed entity

---

### Success Rate
**"Did the system answer EVERYTHING the user asked for?"**

Measures whether the system answered all attributes the user explicitly requested during the dialogue (e.g. phone number, address, postcode, reference number).

- Requires Inform = 1 first. It cannot succeed without first offering the right entity
- For **find intents**: checks whether all requested attributes were provided (e.g., user asks for phone + address → both must appear in responses)
- For **book intents**: checks whether a booking reference number was provided
- Binary per dialogue: 1 if all requested attributes were answered, else 0
- Example: user requests phone and address → system provides only phone → Success = 0
- Dataset score = (correct dialogues / total dialogues) × 100

**Key relationship:** Success ≤ Inform always. A system can Inform without Succeeding (mentioned right entity but didn't give phone number), but cannot Succeed without Informing.

---

### BLEU
Measures n-gram overlap between system responses and MultiWOZ 2.2 reference responses.

**Important limitation for MAS4CS:** The MultiWOZ 2.2 reference responses are delexicalized (use placeholders like [restaurant_name], [restaurant_phone]), while 
MAS4CS generates lexicalized responses (real entity names from the DB). This causes near-zero word overlap even for correct responses, making BLEU scores not comparable to leaderboard numbers. 
BLEU is reported for completeness but should **not** be used as a primary evaluation metric for MAS4CS.

---

### Combined Score
Standard MultiWOZ formula: Combined = 0.5 × (Inform + Success) + BLEU.
Due to the BLEU limitation above, Combined is also **not** directly comparable to the leaderboard. Inform and Success are the primary Tomiinek metrics for MAS4CS.

---

### Comparison with Leaderboard
State-of-the-art models on the MultiWOZ 2.2 leaderboard (DiactTOD, KRLS, TOATOD) report Inform ~85–90%, Success ~75–85%. 
Inform and Success remain directly comparable since both use the same GT belief state and the same official DB for entity matching. 
BLEU and Combined are not comparable due to the lexicalization mismatch described above.

---

## Summary

The 10 metrics provide comprehensive coverage:
- **Routing & Intent** (1, 2, 3): Does the system understand what the user wants, route to the correct domain, and choose the right action type?
- **Slot Tracking** (4, 5): Does the system accurately extract and accumulate user constraints?
- **Response Grounding** (6): Does the system avoid inventing entity names not returned by the DB?
- **Policy & Correctness** (8, 9): Does the system follow booking rules and respond correctly at each turn?
- **Booking Completion** (10): Does the system complete the user's booking goal end-to-end?
- **Response Quality** (11): Does the system produce helpful, correct, and clear natural language responses?
- **Official Benchmark Metrics** (Inform, Success, BLEU, Combined): Computed via the Tomiinek evaluation script using ground truth belief state. 
    Inform and Success are directly comparable to literature. 
    BLEU is reported with a caveat due to lexicalization mismatch.

All custom metrics are tested in `tests/test_evaluation.py` with detailed examples in `docs/evals_inspection/objective_and_judge_metrics.txt`.
The hierarchical evaluation workflow is tested in `tests/test_evaluator.py` with results in `docs/evals_inspection/evaluator_test_results.txt`.
More details about the metrics can be found in `docs/evals_inspection/metrics_quick_reference.txt`, and `docs/evals_inspection/tomiinek__evaluator_tests.txt`.

---
