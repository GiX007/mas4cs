# Dataset: MultiWOZ 2.2

The **MultiWOZ 2.2** dataset is the backbone of the MAS4CS system, providing a structured multi-domain environment (hotel, restaurant, taxi) to test agentic coordination and state tracking.

Detailed structural exploration results (raw prints, inspection logs, and structural details) are available in `docs/dataset_inspection/` directory.

---

## Data Overview

The dataset is loaded via HuggingFace `datasets` as a `DatasetDict`.

- **Total Dialogues**: 10,437 dialogues across 
  - Train: 8,437 
  - Validation: 1,000 
  - Test: 1,000
- Each dialogue contains **multi-domain conversations** of multiple turns between `USER` and `SYSTEM` 
- **Average dialogue length**: ~ 13-14 turns

---

## General Structure

Each dialogue contains:
- `dialogue_id`
- `services`
- `turns`

Each turn contains:
- `turn_id`
- `speaker`
- `utterance`
- `frames`
- `dialogue_acts`

Each frame contains:
- `service`
- `state`: 
  - `active_intent`
  - `requested_slots`
  - `slots_values`
- `slots`

- `frames` vs. `dialogue_acts`
  - `frames` represent the dialogue state layer. They store the current belief state of the conversation: the `active intent`, `requested slots`, and accumulated constraints (`slots_values`). They answer "What is the current task and what does the system believe so far?"
  - `dialogue_acts` represent the action layer. They describe the communicative action performed in a specific turn (e.g., inform, request, book). They answer "What action happened in this turn?"

Each dialogue act contains:
- `dialogue_act`
  - `act_type`: the semantic action performed in the turn (e.g., `Restaurant-Inform`, `Booking-Book`)
  - `act_slots`: the structured slot–value pairs associated with that action (e.g., `area = north`, `bookday = Saturday`)
- `span_info`
  - `act_type`
  - `act_slot_name`
  - `act_slot_value`
  - `span_start`
  - `span_end`

- `dialogue_act` vs. `span_info`
  - `dialogue_act` represents the semantic action annotation of the turn. It describes what action occurred and which slot–value pairs are associated with that action at the intent level. It answers "What does this turn mean?"
  - `span_info` represents the text-grounded evidence of slot values. It indicates where in the utterance a slot value explicitly appears, using character offsets. It answers "Where is that meaning located in the text?"
  - Not every semantic slot in `dialogue_act` must appear in `span_info`. When present, `span_info` enables grounding validation and hallucination detection 

- Three conceptual layers in MultiWOZ 2.2
  - State layer → `frames` → What does the system currently believe about the user’s goal?
  - Action layer → `dialogue_acts` → What action happened in this turn?
  - Text-grounding layer → `span_info` → Where in the text does this value appear?

---

## Feature Selection 

### 1. dialogue_id
- Name: `dialogue_id`
- What it represents: the unique name of a full conversation
- Example: `PMUL4398.json`
- Where we use it: logging, tracing errors, grouping results
- Which agent: all agents, evaluator
- How we use it in evaluation: pointer to a specific dialogue

### 2. turn_id
- Name: `turns[].turn_id`
- What it represents: the order of turns inside a dialogue
- Example: `0`, `1`, `2`, ...
- Where we use it: maintain correct chronological state
- Which agent: Memory, Evaluator
- How we use it for evaluation: ensure per-turn predictions are aligned correctly with ground truth

### 3. services (dialogue-level)
- Name: `services`
- What it represents: the list of domains involved in the dialogue
- Example: `["restaurant", "hotel"]`
- Where we use it: domain routing and limiting search space
- Which agent: Triage, Policy 
- How we use it in evaluation: check if correct domain was selected (or verify if the system operates only within the dialogue's permissible domains)

### 4. speaker + utterance
- Names:
  - `turns[].speaker`
  - `turns[].utterance`
- What it represents: who is speaking (`USER` or `SYSTEM`) and what they say
- Example: `[USER]: "I want an expensive hotel in the south."`
- Where we use it: build conversation history and generate responses
- Which agent: Triage, Action, Memory, Supervisor
- How we use it in evaluation: calculate task success and detect hallucinations

### 5. active_intent
- Name: `turns[].frames[].state.active_intent`
- What it represents: the task the user wants to perform
- Example: `find_restaurant`, `book_hotel`
- Where we use it: decide which tool to call (search, book, inform)
- Which agent: Triage, Action, Policy
- How we use it in evaluation: measure intent accuracy

### 6. requested_slots
- Name: `turns[].frames[].state.requested_slots`
- What it represents: information the user asks for
- Example: `["hotel-phone"]`
- Where we use it: detect what data to fetch from the database (identify specific information the user expects the system to provide)
- Which agent: Policy, Action
- How we use it in evaluation: measure if the system provided all requested information

### 7. slots_values
- Name: `turns[].frames[].state.slots_values`
- What it represents: the requirements (constraints) the user has specified so far
- Example: `restaurant-area = north`, `restaurant-pricerange = expensive`
- Where we use it: state tracking, booking inputs
- Which agent: Triage (extraction), Memory (state update), Action (tool input), Supervisor (validation)
- How we use it in evaluation: calculate Joint Goal Accuracy (JGA), slot accuracy, state tracking correctness

### 8. dialogue_acts (act_type)
- Name: `turns[].dialogue_acts.dialog_act.act_type`
- What it represents: the type of communicative action
- Example: `Restaurant-Inform`, `Hotel-Book`, `Restaurant-Request`
- Where we use it: simulate tool calls and validate chosen action
- Which agent: Action, Supervisor
- How we will use it for evaluation: measure action-level correctness

### 9. dialogue_acts (act_slots)
- Name: `turns[].dialogue_acts.dialog_act.act_slots`
- What it represents: slot-value pairs associated with an action
- Example: `slot_name=area`, `slot_value=south`
- Where we use it: validate slot extraction and tool inputs 
- Which agent: Triage, Memory, Supervisor
- How we will use it in evaluation: slot extraction accuracy

### 10. span_info
- Name: `turns[].dialogue_acts.span_info`
- What it represents: exact character positions where slot values appear in text
- Example: `north` appears at character 32–37
- Where we use it: check if slot values are grounded in text 
- Which agent: Supervisor
- How we use it in evaluation: hallucination detection

---

## Dataset Preprocessing

The original MultiWOZ structure is nested and complex. For MAS4CS, we apply lightweight preprocessing to simplify access and stabilize evaluation.

### Goals of preprocessing
- Make selected features directly accessible
- Avoid repeated parsing inside agents
- Ensure consistent internal state format
- Align dataset structure with evaluation metrics

### 1. Transformations
- **Slot-pair normalization**
  - Convert slots_values from parallel name/value lists into a direct slot → value mapping
  - Example:
    - Raw: `{'slots_values_name': ['restaurant-area'], 'slots_values_list': [['north']]}`
    - Normalized: `{'restaurant-area': 'north'}`
    - If a slot contains multiple values in `slots_values_list`, **only the first value is retained** during preprocessing (multivalued slots are not explicitly modeled in MAS4CS), e.g., we keep `north` from [`north`, `east`]
- **Unified Turn Record construction**
  - Flatten each turn into a clean internal structure:
    - turn_id
    - speaker
    - utterance
    - active_intent
    - requested_slots
    - normalized slots_values
    - dialogue acts
    - span_info
- **No full database reconstruction**
  - We do not rebuild the official MultiWOZ database
  - Tool simulation relies only on annotated state and dialogue acts

### 2. Dataset Filtering

MAS4CS utilizes a controlled subset of the MultiWOZ 2.2 dataset to ensure consistent domain structure and evaluation conditions.

The system is restricted to the **Hotel** and **Restaurant** domains. These domains are chosen because they provide rich slot structures and natural cross-domain transitions (e.g., restaurant ↔ hotel).

We retain dialogues where the set of services satisfies: `services ⊆ {hotel, restaurant}`. This means:
- Dialogues containing only `hotel`
- Dialogues containing only `restaurant`
- Dialogues containing both `hotel` and `restaurant`

We exclude dialogues that include any additional domains such as: `taxi`, `train`, `attraction` even if they also contain `hotel` or `restaurant` (although in frames, you can see slots from other domains; this is the structure of multiwoz).

This restriction:
- Reduces schema complexity
- Avoids introducing unused policy and slot types
- Ensures consistent slot structure
- Preserves multi-domain behavior

All filtering is applied consistently across train, validation, and test splits.


### Example of processed dialogue format

After preprocessing, each dialogue is converted into a simplified internal structure:

```
{
  "dialogue_id": "PMUL4398.json",
  "services": ["restaurant", "hotel"],
  "turns": [
    {
      "turn_id": 0,
      "speaker": "USER",
      "utterance": "i need a place to dine in the center thats expensive",
      "frames": {
        "restaurant": {
          "active_intent": "find_restaurant",
          "requested_slots": [],
          "slots_values": {
            "restaurant-area": "centre",
            "restaurant-pricerange": "expensive"
          }
        },
        "hotel": {
          "active_intent": "find_hotel",
          "requested_slots": [],
          "slots_values": {}
        }
      },
      "dialogue_acts": ["Restaurant-Inform"],
      "dialogue_act_slots": [
        {"slot": "area", "value": "centre"},
        {"slot": "pricerange", "value": "expensive"}
      ],
      "span_info": [
        {"slot": "area", "value": "centre"},
        {"slot": "pricerange", "value": "expensive"}
      ]
    }
  ]
}
```
This structure preserves all essential semantic information (state, actions, and text grounding) while removing nested complexity. It separates the state layer (`frames`), action layer (`dialogue_acts`), and text-grounding layer (`span_info`) into a clean, directly accessible format, making the dataset easier to use for MAS4CS reasoning and evaluation without altering the original annotations.

---
