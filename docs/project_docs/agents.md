# MAS4CS Agents 

## Agent Fundamentals

Before building MAS4CS, three agent architectures were explored to understand the progression from raw LLM to structured reasoning. See `scripts/explore_agents.py`.

### Agent Types

- **Standard LLM (No Tools):** Pure text generation from training data. No tool access, no reasoning loop. Use case: general questions, policy explanations
- **Simple Agent (Single-Turn Tool Use):** Flow: User query → LLM decides → Execute tool(s) once → Final answer. Cannot iterate or reason across multiple steps. Use case: single lookup tasks ("Find cheap hotels in south")
- **ReAct Agent (Multi-Turn Reasoning):** Flow: User query → [LLM thinks → Execute tools → Observe]* → Final answer. Can iterate multiple times until task is complete. Maintains conversation context across iterations. Use case: complex multi-step tasks ("Compare cheap vs expensive hotels")

### What is an Iteration?

An iteration in ReAct is one complete reasoning cycle: one LLM call (THINK), zero or more tool executions based on the LLM's decision (ACTION), and results added to conversation history (OBSERVE). 
Tool executions happen within iterations. An iteration is a "thinking moment" for the agent, which may trigger multiple actions.

---

## MAS4CS Workflow Architecture

```
User Message → Triage → Policy → Action  → Supervisor → Memory
                                    ↑             |
                                    |___(retry)___|
```

The MAS4CS workflow is defined as a LangGraph StateGraph in `src/core/workflow.py`, connecting all agents as nodes with edges encoding both normal flow and the conditional retry loop.

### Why LangGraph over sequential Python calls?

LangGraph provides three capabilities that plain function calls cannot: automatic state passing and updating between nodes, conditional routing that dynamically chooses the next agent based on state flags, and graph visualization for debugging execution flow.

### Retry Loop

If the Supervisor marks `validation_passed=False` and `attempt_count < 2`, the workflow loops back to Action for self-correction. The full retry path is:

```
Action → Supervisor → (if invalid and attempt_count < 2) → Action → Supervisor → Memory
```

Memory only saves after Supervisor approves or max retries are exhausted. If both attempts fail, the last Action response enters history uncorrected.

---

## Tool Usage Design by Experiment

Tool routing is handled differently in Exp1 and Exp2, by design.

**Exp2 (MAS Graph):** Tool usage is rule-based and deterministic. The Action agent's code reads the intent (`find_` or `book_`) and calls `find_entity` or `book_entity` directly, no LLM decision needed. The LLM only generates the natural language response using the DB results it receives. This is intentional: tool routing is architectural (code decides), not generative (LLM decides), which eliminates one source of hallucination.
**Exp1 (Single-Agent Baseline):** The single LLM cannot call tools itself, so tool usage is simulated with two LLM calls per turn. Call 1 extracts slots and intent without DB results. The code then runs the DB query using those slots. Call 2 receives the actual DB results and generates the final grounded response. This two-call design is the honest and fair way to include tool usage in a single-agent system that mirrors what a real LLM with tool calling would do and keeps Exp1 directly comparable to Exp2.

---

## Agent Roles

- **Triage (LLM)**
  - entry point of the workflow
  - detects `current_domain`, `active_intent`, and extracts `slot_values` from the `user_utterance`
  - uses `conversation_history` for reference resolution
  - no policy validation, no DB calls
  
- **Policy (rule-based, no LLM)** 
  - validates required slots for booking intents  
  - checks if required slots are present for booking intents and populates `policy_violations` if slots are missing
  - acts as a hard gate before book/search execution

- **Action (LLM + internal DB tools** 
  - generates the user's response 
  - calls `find_entity` or `book_entity` based on `intent`
  - prevents double booking
  - sets `action_taken` and `dialogue_acts`
  - supports retry via `attempt_count` and `supervisor_feedback`

- **Supervisor (LLM judge)** 
  - check correctness of Triage, Policy, and Action outputs
  - detects hallucinations using `db_results` and `valid_entities`
  - sets `validation_passed` if all checks pass 
  - triggers retry if validation fails (max 2 attempts))

- **Memory (no LLM)**
  - appends user + assistant messages to `conversation_history` within a turn
  - saves only after validation passes or max retries exhausted
  - prevents duplicate history entries during retries

All agents are implemented as single-purpose functions in `src/agents/`.

**Important note on "agents"**: 
Agents are structured workflow nodes that read/write shared state and are structured LLM nodes with  distinct roles, not autonomous tool-using agents in the classical sense. 
No external tools or APIs are called. The goal is to study whether architectural structure alone improves reliability, before introducing the added complexity of real tool use.

---

## DB Tools

Two public tools in `src/tools/db_tools.py`, called by Action via rule-based routing:

- **`find_entity(domain, belief_state)`**
  - Loads the domain DB (`hotel_db.json` / `restaurant_db.json`) with caching
  - Normalizes slot values
  - Filters entities by constraints and returns up to `MAX_DB_RESULTS` matches
- **`book_entity(domain, belief_state)`**
  - Calls `find_entity` to check for matches
  - If matches exist, returns a success message with the first match's name

---

## Shared State

All workflow nodes read from and write to a single `AgentState` TypedDict defined in `src/core/state.py`.
It carries the full context for one user turn, plus accumulated cross-turn memory.

Key fields:

- **IDs:** `dialogue_id`, `turn_id`, `services`
- **Turn input:** `user_utterance`
- **Triage outputs:** `current_domain`, `active_intent`, `slots_values` (accumulated across turns)
- **Conversation:** `conversation_history`
- **Action outputs:** `agent_response`, `action_taken`, `dialogue_acts`
- **DB grounding:** `db_results`, `booked_entity`, `informed_entity`
- **Policy:** `policy_violations`
- **Supervisor:** `validation_passed`, `valid_entities`, `supervisor_feedback`
- **Control/config:** `attempt_count`, `model_config`

---

