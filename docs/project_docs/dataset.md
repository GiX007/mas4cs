# Dataset: MultiWOZ 2.2

The **MultiWOZ 2.2** dataset is the backbone of the MAS4CS system, providing a structured multi-domain environment to test agentic coordination and state tracking.

Detailed structural exploration results are available in `docs/dataset_inspection/`.

---

## Data Overview

- **Total Dialogues**: 10.437 across Train: 8.437 / Validation: 1.000 / Test: 1.000
- Each dialogue contains multi-domain conversations of multiple turns between `USER` and `SYSTEM`
- **Average dialogue length**: ~13–14 turns

---

## Loading Strategy

### HuggingFace (Phase 1, initial development)

The dataset was first loaded via HuggingFace `datasets` (`load_dataset("multi_woz_v22")`) due to its simplicity as a starting point: one function call, automatic caching, and familiar `DatasetDict` structure. 
Although the dialogue samples are identical to the official GitHub version, the HF version differs in two important ways:

- **Structure**: turns are stored as a **dict of parallel lists** (`turns['utterance'][i]`, `turns['speaker'][i]`), whereas the official version uses a **list of dicts** (each turn is its own dict)
- **Missing DB files and eval scripts**: the HF version packages dialogue annotations only. It does not include the domain database files (`hotel_db.json`, `restaurant_db.json`) or the official evaluation scripts needed to compute Inform Rate, Success Rate, and BLEU

The HF dataset lives in `dataset/mw22_filtered.json` (preprocessed) and `hf_cache/` (raw). It was used for Experiment 1 and Experiment 2 with custom metrics only.

### Official GitHub (Phase 1 Revisited, DB integration + official metrics)

The official repository (`budzianowski/multiwoz`) was cloned locally to provide full access for EuroHPC (no internet available on the cluster):

```bash
git clone https://github.com/budzianowski/multiwoz.git data/multiwoz_github
```

Key additions over the HF version: domain DB files in `data/multiwoz_github/db/` (`hotel_db.json`, `restaurant_db.json`), official evaluation script (`evaluate.py`), and `schema.json` defining all slot types. 
The official version is required for computing Inform Rate, Success Rate, BLEU, and Combined Score, and for building role-specific fine-tuning datasets for Experiment 3.

The loader for this version lives in `src/data/load_dataset_v2.py`. Splits: train 8.437 / dev 1.000 / test 1.000 (same counts as HF). 
After filtering to hotel + restaurant only: train 2.850 / dev 171 / test 186.

---

## General Structure

Each dialogue contains: `dialogue_id`, `services`, `turns`.
Each turn contains: `turn_id`, `speaker`, `utterance`, `frames`, `dialogue_acts`.
Each frame contains: `service`, `state` (`active_intent`, `requested_slots`, `slots_values`), `slots`.

**`frames` vs. `dialogue_acts`**
- `frames` = the state layer. Stores the current belief state: active intent, requested slots, accumulated constraints. Answers "What is the current task and what does the system believe so far?"
- `dialogue_acts` = the action layer. Describes the communicative action performed in a turn (e.g., inform, request, book). Answers "What action happened in this turn?"

Each dialogue act contains:
- `dialogue_act`: `act_type` (e.g., `Restaurant-Inform`, `Booking-Book`) and `act_slots` (slot–value pairs)
- `span_info`: `act_type`, `act_slot_name`, `act_slot_value`, `span_start`, `span_end`

**`dialogue_act` vs. `span_info`**
- `dialogue_act` = semantic annotation. Describes what action occurred and which slots are involved. Answers "What does this turn mean?"
- `span_info` = text-grounded evidence. Indicates where in the utterance a slot value explicitly appears using character offsets. Answers "Where is that meaning in the text?" Not every semantic slot must appear in `span_info`

- Three conceptual layers in MultiWOZ 2.2
  - State layer → `frames` → What does the system currently believe about the user’s goal?
  - Action layer → `dialogue_acts` → What action happened in this turn?
  - Text-grounding layer → `span_info` → Where in the text does this value appear?

**Three conceptual layers:**
- State layer → `frames` → What does the system currently believe about the user's goal?
- Action layer → `dialogue_acts` → What action happened in this turn?
- Text-grounding layer → `span_info` → Where in the text does this value appear?

---

## Basic Features 

### 1. dialogue_id
- **Represents:** unique name of a full conversation, e.g., `PMUL4398.json`
- **Used by:** evaluator for logging, tracing, grouping results

### 2. turn_id
- **Represents:** chronological order of turns inside a dialogue, e.g., `0`, `1`, `2`
- **Used by:** evaluator and Memory to align per-turn predictions with ground truth

### 3. services (dialogue-level)
- **Represents:** domains involved in the dialogue, e.g., `["restaurant", "hotel"]`
- **Used by:** Triage, Policy when domain routing, search space restriction

### 4. speaker + utterance
- **Represents:** who speaks (`USER`/`SYSTEM`) and what they say
- **Used by:** Triage, Action, Memory, Supervisor to build conversation history, generate responses, detect hallucinations

### 5. active_intent
- **Represents:** the task the user wants to perform, e.g., `find_restaurant`, `book_hotel`
- **Used by:** Triage, Action, Policy to decide which tool to call and measure intent accuracy

### 6. requested_slots
- **Represents:** information the user explicitly asks for, e.g., `["hotel-phone"]`
- **Used by:** Policy, Action to fetch specific data from DB, drives Success Rate evaluation

### 7. slots_values
- **Represents:** constraints the user has specified so far, e.g., `restaurant-area = north`
- **Used by:** Triage (extraction), Action (tool usage), Supervisor (validation), drives JGA and slot accuracy

### 8. dialogue_acts (act_type)
- **Represents:** type of action, e.g., `Restaurant-Inform`, `Booking-Book`
- **Used by:** Action, Supervisor to simulate tool calls and validate action correctness

---

## Dataset Preprocessing

The original MultiWOZ structure is nested and complex. Lightweight preprocessing makes selected features directly accessible, avoids repeated parsing inside agents, ensures consistent internal state format, and aligns structure with evaluation metrics.

### HuggingFace version

Slot-pair normalization converts `slots_values` from parallel name/value lists into a direct `slot → value` mapping. Example: `{'hotel': {'slots_values_name': ['restaurant-area'], 'slots_values_list': [['north']]}}` → `{'hotel': {'restaurant-area': 'north'}}`. If a slot has multiple values, only the first is retained.
Each turn is flattened into a unified turn record with: turn_id, speaker, utterance, active_intent, requested_slots, normalized slots_values, dialogue acts, act_slots, span_info.
No full database reconstruction as tool simulation relies only on annotated state and dialogue acts.

### Official GitHub version

The official version uses a **list of dicts** structure. Each turn is already a self-contained dict, so no parallel-list flattening is needed. 
The key transformation is extracting the active frame per domain from `frames` (one frame per service, skipping `active_intent == "NONE"`) and normalizing `slot_values` from `{slot: [value_list]}` to `{slot: value}` (first value retained).
Additionally, dialogue acts are not embedded inside each turn's dict in the official format. They live in a separate file, `data/multiwoz_github/data/MultiWOZ_2.2/dialog_acts.json`, keyed by dialogue_id and turn_id. 
During preprocessing, we load this file separately and join the relevant acts onto each turn by matching on those two keys, producing the same unified turn record as the HF version.
`span_info` is not extracted in the v2 pipeline since hallucination detection in MAS4CS is handled by the Action and Supervisor agents.
Each turn is flattened to: turn_id, speaker, utterance, active_intent, requested_slots, normalized slots_values, dialogue acts.

### Dataset Filtering

MAS4CS is restricted to **Hotel** and **Restaurant** domains with rich slot structures, natural cross-domain transitions, consistent schema. 
We retain dialogues where `services ⊆ {hotel, restaurant}`, excluding any dialogue that includes taxi, train, attraction, or other domains even if it also contains hotel or restaurant. 
Filtering is applied consistently across all splits.

| Split | Total | Hotel + Restaurant only |
|-------|-------|-------------------------|
| Train | 8.437 | 2.850 (33.8%)           |
| Dev   | 1.000 | 171 (17.1%)             |
| Test  | 1.000 | 186 (18.6%)             |

---

### Example of processed dialogue format

### HuggingFace version

```json
{
  "dialogue_id": "PMUL4398.json",
  "services": ["restaurant", "hotel"],
  "turns": [
    {
      "turn_id": 0,
      "speaker": "USER",
      "utterance": "i need a place to dine in the center that's expensive",
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

### Official GitHub version

```json
{
  "dialogue_id": "PMUL4398.json",
  "services": ["restaurant", "hotel"],
  "turns": [
    {
      "turn_id": "0",
      "speaker": "USER",
      "utterance": "i need a place to dine in the center that's expensive",
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
          "active_intent": "NONE",
          "requested_slots": [],
          "slots_values": {}
        }
      },
      "dialogue_acts": ["Restaurant-Inform"],
      "dialogue_act_slots": [
        {"slot": "area", "value": "centre"},
        {"slot": "pricerange", "value": "expensive"}
      ]
    }
  ]
}
```

Key differences from HF version: `turn_id` is a `str` (not `int`), frames with `active_intent == "NONE"` are kept in the raw structure but skipped during processing, `span_info` is not extracted. 

---
