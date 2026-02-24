# References

---

## Core Academic References

1. **Zang et al. (2020) - MultiWOZ 2.2**
   - A corrected version of the large-scale multi-domain dialogue dataset MultiWOZ
   - Relation to MAS4CS: Serves as our primary dataset for simulating customer service domains and evaluating slot tracking (JGA) and task success

2. **Yao et al. (2022) - ReAct** 
   - A framework that combines reasoning (Chain-of-Thought) with action (tool use) in LLMs
   - Relation to MAS4CS: Forms the reasoning backbone (in one of the implementation variants evaluated) of the Action Agent (reason → act → observe loop)

3. **Schick et al. (2023) - Toolformer**
   - Demonstrates self-supervised tool-use learning in LLMs
   - Relation to MAS4CS: Conceptual foundation for structured tool invocation in Triage and Action agents

4. **Shinn et al. (2023) - Reflexion**
   - Introduces self-reflection and iterative correction in language agents
   - Relation to MAS4CS: Guides the Supervisor Agent to detect hallucinations and trigger corrective loops

5. **Masterman et al. (2024) - AI Agent Architectures Survey**
   - Comprehensive survey of modern agent architectures and orchestration patterns
   - Relation to MAS4CS: Justifies hierarchical, modular agent decomposition

6. **Hong et al. (2023) - MetaGPT** 
   - Role-based collaborative multi-agent framework for complex tasks
   - Relation to MAS4CS: Inspires role separation (Triage, Policy, Action, Supervisor)

7. **Park et al. (2025) - Workflow Graphs**
   - Graph-based orchestration for reliable conversational systems
   - Relation to MAS4CS: Direct architectural blueprint for LangGraph workflow design

8. **Wang et al. (2025) - TalkHier**
   - Structured communication and hierarchical coordination for multi-agent systems
   - Relation to MAS4CS: Supports controlled message passing between agents

9. **Park et al. (2023) - Generative Agents**
   - Introduces memory architectures and "reflection" for long-term agent behavior
   - Relation to MAS4CS: Influences Memory Agent and history manipulation strategy

10. **Valentini et al. (2025) - MAS for Customer Experience**
    - Applies multi-agent LLM systems in customer service contexts
    - Relation to MAS4CS: Domain validation for customer experience automation

11. **Choubey et al. (2025) - Turning Conversations into Workflows**
    - Extracts structured workflows from dialogue interactions
    - Relation to MAS4CS: Supports dialogue-to-graph execution mapping

12. **Shen et al. (2023) - HuggingGPT** 
    - Orchestrates multiple AI models for task execution
    - Relation to MAS4CS: Conceptual reference for heterogeneous model orchestration

13. **Mohammadi et al. (2025) - Evaluation of LLM Agents Survey**
    - Survey of evaluation methodologies for LLM-based agents
    - Relation to MAS4CS: Guides hybrid evaluation framework design

14. **Liu et al. (2023) - G-Eval** 
    - A framework for using GPT-4 as a judge for NLG evaluation
    - Relation to MAS4CS: Used for subjective policy-compliance scoring

15. **Zheng et al. (2023) - Judging LLM-as-a-Judge** 
    - Benchmarking LLM quality using model-as-judge evaluation
    - Relation to MAS4CS: Supports LLM-based evaluation

16. **Balaji et al. (2026) - Beyond IVR**
    - A benchmark specifically for customer support LLM agents
    - Relation to MAS4CS: Provides business-policy evaluation framework

17. **Baidya et al. (2025) - The Behavior Gap**
    - Evaluates zero-shot LLM agents in complex task-oriented dialogues
    - Relation to MAS4CS: Highlights the gap between reasoning and task success

18. **Liu et al. (2023) - BOLAA**
    - Benchmarking and orchestrating LLM-augmented autonomous agents
    - Relation to MAS4CS: Supports structured evaluation of multi-agent orchestration

---

## Technical & Educational References

1. Hugging Face (Datasets Library)
   - URL: https://huggingface.co/docs/datasets/index
   - Documentation for dataset loading, processing, and evaluation pipelines
   - Relation to MAS4CS: Used to load and process the MultiWOZ 2.2 dataset and manage train/validation/test splits

2. Hugging Face (Transformers Library)
   - URL: https://huggingface.co/docs/transformers/index
   - Official documentation for loading, training, fine-tuning, and evaluating transformer-based language models
   - Relation to MAS4CS: Used for loading base LLMs, fine-tuning with LoRA, and performing inference during experiments

3. PyTorch
   - URL: https://docs.pytorch.org/docs/stable/index.html
   - Deep learning framework for tensor computation and GPU acceleration
   - Relation to MAS4CS: Backend framework used for model training, fine-tuning, and inference

4. DeepLearning.AI - Agentic AI Course (Andrew Ng)
    - URL: https://learn.deeplearning.ai/courses/agentic-ai/information
    - Practical course covering agent design patterns like reflection, tool usage, planning, and multi-agent collaboration
    - Relation to MAS4CS: Conceptual and implementation-level guidance for agent design decisions

5. LangChain - LangGraph Documentation
    - URL: https://docs.langchain.com/oss/python/langgraph/overview
    - Official documentation for graph-based LLM workflow orchestration
    - Relation to MAS4CS: Direct implementation framework for the MAS4CS state machine

6. Unsloth - Efficient LLM Fine-Tuning
   - URL: https://unsloth.ai/
   - Lightweight library enabling efficient LoRA fine-tuning of large language models
   - Relation to MAS4CS: Used for parameter-efficient fine-tuning of open-source LLMs

7. EuroHPC Joint Undertaking
   - URL: https://www.eurohpc-ju.europa.eu/index_en
   - European high-performance computing infrastructure supporting large-scale GPU workloads
   - Relation to MAS4CS: Provides computational resources for large-scale model fine-tuning experiments

---


