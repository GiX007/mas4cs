# MAS4CS Agents 

## 1: Agent Fundamentals

Before building MAS4CS, a thorough exploration of three agent architectures took place to understand the progression from raw LLM to structured reasoning, see `scripts/explore_agents.py`.

### Agent Types

- **Standard LLM (No Tools)**
  - Pure text generation from training data
  - No tool access, no reasoning loop
  - Use case: general questions, policy explanations

- **Simple Agent (Single-Turn Tool Use)**
  - Flow: User query → LLM decides → Execute tool(s) once → Final answer
  - Total LLM calls: 2 (decision + response)
  - Cannot iterate or reason across multiple steps
  - Use case: single lookup tasks ("Find cheap hotels in south")

- **ReAct Agent (Multi-Turn Reasoning)**
  - Flow: User query → [LLM thinks → Execute tools → Observe]* → Final answer
  - Can iterate multiple times until task is complete
  - Maintains conversation context across **iterations**
  - Use case: complex multi-step tasks ("Compare cheap vs expensive hotels")

### What is an Iteration?

An **iteration** in ReAct is one complete reasoning cycle:
- One call to the LLM -> `THINK`
- Zero or more tool executions (based on LLM's decision) -> `ACTION`
- Results added to conversation history -> `OBSERVE`

Example Breakdown
```
Iteration 1:
  └─ LLM Call 1 → "I need to search for cheap AND expensive hotels"
      ├─ Tool execution 1: search cheap
      └─ Tool execution 2: search expensive

Iteration 2:
  └─ LLM Call 2 → "Here's the comparison: [answer]"
```

Tool executions happen **within** iterations. An iteration = "thinking moment" for the agent, which may trigger multiple actions.

The three agent types differ in reasoning depth: 
- a Standard LLM makes one call with no tools
- a Simple Agent makes two calls (one to decide, one to respond) with a single tool execution
- a ReAct Agent makes N calls iterating between reasoning and tool use until the task is complete

---

## 2. MAS4CS Workflow Architecture

```
User Message → Triage → Policy → Action → Memory → Supervisor
                                    ↑                   |
                                    |_____(retry)_______|
```

The MAS4CS workflow is defined as a LangGraph StateGraph in `src/core/workflow.py`, connecting all agents as nodes with edges encoding both normal flow and the conditional retry loop.

### Why LangGraph over sequential Python calls?

LangGraph provides three capabilities that plain function calls cannot:
- **State Management**: automatically passes and updates shared state between nodes
- **Conditional Routing**: dynamically chooses next agent based on state flags (retry loop)
- **Visualization & Debugging**: generates graph diagrams and tracks execution flow

### Retry Loop

```
USER turn → Triage → Policy → Action → Memory → Supervisor
                                                     ↓
                                         If valid: workflow ends
                                         If invalid: retry → Action
```

The Supervisor validates the Action agent's response. If validation fails and `attempt_count < 2`, the workflow loops back to Action for self-correction.

---

## 3. Agent Roles

- **Triage**
  - entry point of the workflow
  - detects active domain, intent, and extracts slot values from the user message
  - no policy enforcement here — only parsing and routing

- **Policy** 
  - rule-based validation (no LLM) 
  - checks if required slots are present for booking intents and populates `policy_violations` if slots are missing
  - acts as a hard block before Action executes

- **Action** 
  - generates the customer service response using domain, intent, slots, and policy violations as context 
  - determines `action_taken` and `dialogue_acts` 
  - handles retry logic via `attempt_count`

- **Memory**
  - pure state management (no LLM) 
  - appends current turn to `conversation_history` 
  - ensures user message and agent response persist across turns

- **Supervisor** 
  - quality control via LLM 
  - validates agent response against known valid entities to detect hallucinations. Sets `validation_passed` and `hallucination_flags` 
  - triggers retry if validation fails

All agents are implemented as single-purpose functions in `src/agents/`.

**Important note on "agents":** In MAS4CS, agents are structured LLM nodes with  distinct roles, not autonomous tool-using agents in the classical sense. 
No external tools or APIs are called. Each agent reads from and writes to  a shared state, with the workflow graph enforcing execution order and retry logic. 
This is an intentional design choice: the goal is to study whether architectural structure alone improves reliability, before introducing the added complexity of real tool use.

---

## 4. Shared State

All agents read from and write to a single `AgentState` TypedDict defined in `src/core/state.py`.

Key fields by category:

- **Identifiers**: `dialogue_id`, `turn_id`, `services`
- **Input**: `user_utterance`
- **Triage outputs**: `current_domain`, `active_intent`, `slots_values`
- **Conversation**: `conversation_history`
- **Action outputs**: `agent_response`, `action_taken`, `dialogue_acts`
- **Policy**: `policy_violations`
- **Supervisor**: `validation_passed`, `hallucination_flags`, `valid_entities`
- **Control**: `attempt_count`, `model_config`

---

