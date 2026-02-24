# MAS4CS: A Multi-Agent System for Customer Service Automation

A modular, hierarchical multi-agent system for end-to-end customer service automation using LangGraph and MultiWOZ 2.2.

---

## Project Structure

- `mas4cs/`
  - `src/` - Source code
    - `agents/` - Triage, Policy, Action, Memory, Supervisor
    - `core/` - AgentState, workflow, dialogue runner
    - `data` - Data loading, preprocessing, constants
    - `evaluation` - Metrics, evaluators, LLM judge
    - `experiments` - Experiment runners, helpers, config
    - `utils` - Prompts, utilities, cost calculation
  - `scripts/` - Exploration scripts (dataset, models, agents)
  - `tests/` - Unit and integration tests
  - `lab` - Learning references and concept demos
  - `dataset` - Processed MultiWOZ data (generated)
  - `docs/` - Documentation and inspection outputs
  - `results` - Experiment results (generated)
  - `hf_cache/` - HuggingFace cache

---

## Why This Project

A perfect customer service system understands the user's goal, tracks all constraints without losing context, enforces business rules before acting, avoids hallucinations, handles multi-domain conversations, and completes the task reliably from start to finish.

Most single-LLM approaches fail at one or more of these requirements under real-world complexity. MAS4CS investigates whether a structured, hierarchical multi-agent architecture rather than a larger model can close this gap reliably.

---

## Setup

```bash
git clone <repo>
cd mas4cs

python -m venv venv
source venv/bin/activate     # macOS/Linux
.\venv\Scripts\Activate.ps1  # Windows

pip install -r requirements.txt
```

---

## Quickstart
```bash
python -m main
```

This will load and preprocess the dataset, then run all configured experiments. Results are saved to `results/experiments/`.

---

## How It Works

MAS4CS is built on one core question: does **architectural structure** alone improve reliability, over single giant models?

Five specialized agents are connected in a deterministic LangGraph workflow, each with a single responsibility. A shared `AgentState` flows through the graph. Every agent reads from it and writes back to it.

```
User Message → Triage → Policy → Action → Memory → Supervisor
                                    ↑                    |
                                    |_____(retry)________|
```

- **Triage** detects domain, intent, and extracts slots
- **Policy** enforces hard rules (e.g. no booking without required slots)
- **Action** generates the response
- **Memory** updates conversation history
- **Supervisor** validates the response and triggers retry if needed

**A note on "agents":** In MAS4CS, agents are structured LLM nodes with  distinct roles — not autonomous tool-using agents in the classical sense. 
No external tools or APIs are called. Each agent reads from and writes to  a shared state, with the workflow graph enforcing execution order and retry logic. 
This is an intentional design choice: the goal is to study whether architectural structure alone improves reliability, before introducing the added complexity of real tool use.

Full details in `docs/agents.md`.

---

### Technology

- Python 3.10+
- MultiWOZ 2.2 — dialogue dataset
- OpenAI / Anthropic / Unsloth — model providers
- LangGraph — multi-agent workflow orchestration

---

## Experiments

Three experiments are run to isolate the effect of architecture and model specialization:

- **Experiment 1** — Single-agent baseline: one LLM, one prompt, no architecture
- **Experiment 2** — MAS graph: same models, structured multi-agent workflow
- **Experiment 3** — Fine-tuned MAS: role-specialized open-source models

Full details in `docs/experiments.md`.

---

## Evaluation

11 metrics across three levels (turn, dialogue, dataset):

- Objective: 
  - Routing & Intent: Intent Accuracy, Domain Accuracy
  - Slot Tracking: Slot Accuracy, JGA, Hallucination Rate
  - Action: Action-Type Accuracy 
  - Policy & Task: Policy Compliance, Task Completion, System Correctness
  - Memory: Cross-Domain Transfer Accuracy
- Subjective
  - LLM-judged Response Quality

Full details in `docs/project_docs/evaluation.md`.

---

## Results

*(To be filled after experiments are complete — see `docs/project_docs/results.md`)*
*(add plots..)*

---

## Development Roadmap 

This section documents the exact steps to reproduce the MAS4CS pipeline, from data loading to evaluation.

- Step 1: Data
  - `load_multiwoz()` — loads or downloads dataset, caches to `src/hf_cache/`. Called in `main.py`
  - `python -m tests.test_loader` — validates dataset loading and split sizes
  - `run_preprocessing_pipeline()` — applies preprocessing (parallel lists -> dicts + domain filtering) and caches to `dataset/mw22_filtered.json`. Called in `main.py`
  - `python -m tests.test_preprocess` — validates transformation and slot normalization on one sample before and after processing. Validates preprocessing pipeline.
  - `python -m scripts.explore_dataset` — full exploration pipeline for both raw and filtered dataset including focused schema references. 
     Saves 8 inspection files to `docs/dataset_inspection/` and 4 schema/data files to `dataset/`
  - `docs/project_docs/dataset.md` — full dataset documentation: structure, feature selection, preprocessing transformations, and processed format schema

- Step 2: Models
  - `docs/project_docs/models.md` - model arsenal reference: open-source and commercial models used across experiments, stack configurations, and fine-tuning strategy
  - `src/models.py` — unified LLM interface (OpenAI, Anthropic, Unsloth). Shared utility called directly by agents, experiments, and tests. No changes needed to agent code when swapping models
  - `python -m tests.test_models` — tests API and local model calls. Saves results to `docs/model_inspection/` (2 JSON + 2 TXT files per run)
  - `python -m scripts.explore_models` — raw output exploration across all available models

- Step 3: Agents
  - `docs/project_docs/agents.md` — full agent documentation: fundamentals, workflow architecture, agent roles, and shared state reference
  - `python -m scripts.explore_agents` — demonstrates agent architecture progression: raw LLM → tool-calling agent → ReAct reasoning loop. Saves output to `docs/agents_inspection/`
  - `src/core/state.py` — `AgentState` TypedDict and `initialize_state()` factory. Shared state flowing through all agents
  - `src/agents/triage.py` — entry point: domain detection, intent classification, slot extraction
  - `src/agents/policy.py` — rule-based hard block: validates required slots before booking actions (no LLM)
  - `src/agents/action.py` — response generation based on domain, intent, slots, and policy violations
  - `src/agents/memory.py` — conversation history management across turns (no LLM)
  - `src/agents/supervisor.py` — hallucination detection and response validation via LLM
  - `python -m tests.test_agents` — unit tests for all individual agents with real LLM calls
  - `src/core/workflow.py` — LangGraph workflow graph connecting all agents with optional retry loop. Called internally by experiments and `main.py`
  - `python -m tests.test_workflow` — tests full multi-agent pipeline on one real MultiWOZ turn. Saves workflow graph images to `docs/images/`

- Step 4: Evaluation
  - `docs/evaluation.md` — full evaluation documentation: 11 metrics, formulas, examples, and component-to-metric mapping
  - `src/evaluation/evaluation_metrics.py` — all objective metric functions (JGA, slot accuracy, hallucination rate, policy compliance, etc.)
  - `src/evaluation/llm_judge.py` — LLM-as-judge infrastructure
  - `src/evaluation/evaluator.py` — `DialogueEvaluator` (turn + dialogue levels) and `DatasetEvaluator` (dataset level). Called internally by experiments and `dialogue_runner.py`
  - `python -m tests.test_evals` — tests all 11 metrics with detailed examples. Saves output to `docs/evals_inspection/objective_and_judge_metrics.txt`
  - `python -m tests.test_evaluator` — integration tests for `DialogueEvaluator` and `DatasetEvaluator`. Saves output to `docs/evals_inspection/evaluator_test_results.txt`

- Step 5: Experiments
  - `docs/project_docs/experiments.md` — full experiment documentation: setup, models, architecture configurations, and results structure
  - `src/experiments/config.py` — single source of truth for all experiment parameters (models, splits, limits)
  - `python -m experiments.run_experiment_1` — single-agent baseline across all configured models
  - `python -m experiments.run_experiment_2` — MAS graph with homogeneous and heterogeneous configurations
  - `python -m experiments.run_experiment_3` — MAS graph with fine-tuned models (requires fine-tuned model paths in config)
  - `python -m tests.test_experiments` — validates single-agent turn and dialogue execution
  - Results saved to `results/experiments/` (3 JSON files per run + running `leaderboard.txt`)
  - Results analysis saved to `docs/project_docs/results.md`

- Step 6: References & Entry Point
  - `python -m main` — full pipeline entry point: loads dataset, runs preprocessing, executes all experiments
  - `docs/project_docs/references.md` — academic and technical work related to this project

---

## Future Work

- **Prompt Engineering** — explore more structured or chain-of-thought prompts per agent role to improve extraction and reasoning quality
- **Moderation Layer** — add a pre-Triage moderation agent for safety checks, urgency detection, and language handling
- **Real Tool Integration** — replace simulated tool calls with actual database queries and booking APIs
- **OOP Refactor** — reimplement agents as classes for production-grade state encapsulation and extensibility
- **Production Prototype** — deploy as an interactive demo (e.g., HuggingFace Spaces)
- **MAS4BPO** - extend the architecture to other large-scale domain-specific (e.g., hospitality) processes

---

## Contributing

Contributions are welcome. Improvements to prompt design, evaluation metrics, experiment configurations, or model support are highly encouraged.

---

