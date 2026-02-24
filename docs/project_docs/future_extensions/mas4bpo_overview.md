# Project Overview  

## A Multi-Agent System for Large-Scale Business Process Optimization

---

## Overview

Restaurant operations involve many interconnected processes such as purchasing, inventory control, pricing, customer service, and marketing. In practice, these processes are supported by different software tools (ERP, POS, CRM) and rely heavily on human coordination across roles and departments.

While specialized systems exist for individual tasks (e.g. inventory software, reservation systems), they typically operate in isolation. As a result, information flows through manual communication, decisions are delayed, and small errors propagate across the organization, especially in high-volume, seasonal restaurant environments.

The idea of this project is **not** to replace existing POS/ERP/CRM systems, but to introduce an **agentic AI architecture** that introduces a coordination layer on top of existing systems. Specialized AI agents analyze different business domains, while a central **General Manager Agent** aggregates insights, prioritizes actions, and supports human decision-making. The system is designed as **human-in-the-loop**, meaning agents never execute strategic actions autonomously.

The goal is selective automation where appropriate, together with improved coordination, consistency, and explainability in operational decision-making.

---

### Core Questions

- Can LLM-based agents, when orchestrated as a multi-agent system, outperform traditional rule-based systems in cross-domain business process coordination for hospitality?
- Can fully automated agentic approaches (e.g., in invoice processing or in customer service) match or exceed human performance in accuracy, speed, and cost-efficiency under well-defined operational constraints?

---

### Key Innovation

While vendors provide isolated solutions (e.g., invoice OCR, inventory heuristics, marketing automation), they typically lack intelligent cross-domain reasoning. The key innovation of this project is the introduction of a **General Manager Agent** that synthesizes outputs from multiple specialized agents, detects cross-domain patterns, and produces prioritized, explainable recommendations for human managers.

In addition, several operational domains such as invoice processing, customer service, reservation management, inventory monitoring, and feedback analysis can be fully automated through specialized agents. In these cases, automation is applied to well-defined, repetitive tasks, while oversight, exception handling, and strategic decisions remain under human control.

This combination of selective automation and centralized agentic coordination enables improved responsiveness, reduced human workload, and more consistent decision-making compared to traditional rule-based or single-domain systems.

---

## Problem Statement

Modern restaurant operations, especially large-scale and seasonal ones, face persistent operational and decision-making challenges:

- **Fragmented data and independent tools**: Even when robust ERP, POS, CRM, and review platforms are in place, these systems rarely communicate in a way that supports cross-domain reasoning. Vendor solutions are typically effective at optimizing individual tasks (e.g. invoice processing, inventory tracking, reservations) but perform poorly when decisions require integrating signals across functions.
- **High reliance on manual coordination**: Many operational workflows assume that multiple people will execute tasks correctly, on time, and with shared context across systems. In practice, this assumption breaks down under realistic operating conditions such as long operating periods, high staff turnover, and limited technical expertise, leading to inconsistent data and delayed reactions.
- **Delayed or inconsistent decisions and issue detection**: Because information is distributed across systems and roles, management decisions are often based on partial, delayed, or subjective inputs rather than integrated, real-time signals. As a result, issues such as stockouts, margin erosion, or customer dissatisfaction are frequently detected late.

Even when advanced software systems are deployed, these structural problems persist due to the lack of coordinated, cross-domain intelligence and the limited application of full automation to isolated tasks.

---

## Proposed Methodology

### High-Level System Architecture

Three-Layer Design:

- Data Layer: ERP,POS, CRM systems as shared sources of truth  
- Agent Layer: Specialized domain agents 
- Orchestration Layer: General Manager Agent for coordination, prioritization and explainability.

### Core Agents

- Invoice Processing Agent
- Inventory Manager Agent
- Revenue Manager Agent
- Customer Review Analysis Agent
- Post-Visit Feedback Agent
- Customer Service & Reservation Manager Agent
- Scout Agent
- Marketing Agent
- General Manager Agent

See details of each agent's role, inputs, outputs, and responsibilities in README.md.

---

### Data

Use of real and synthetic restaurant data including ERP records, POS transactions, CRM customer profiles, and review data. Data preprocessing will ensure consistency and quality for agent training and evaluation.

### Evaluation Strategy

- Individual Agent Performance: Accuracy and cost-efficiency across multiple LLM configurations. Execution speed is acknowledged but not treated as a primary evaluation metric.
- Multi-Agent Coordination: Quality of recommendations, explainability, alignment with human judgment.

---

### Technology Stack

- Data: PostgreSQL for ERP/POS simulation
- Agent Framework: LangGraph or CrewAI for agent orchestration
- LLMs: GPT, Claude, and some open-source alternatives
- Evaluation: Custom and standard metrics

---

## Indicative Timeline (Now – 15 June)

**Phase 1**: Literature review completion, Architecture finalization, and Data Collection (February)

**Phase 2**: Implementation of all Agents and Coordination Logic (March)

**Phase 3**: Evaluation, Analysis, and Reporting (April - May)

**Phase 4**: Final Revisions, Writing and Delivery (Early June)

---

## Challenges

- **Depth and scope of individual agents**:
  
  Each agent represents a complex problem domain that could independently require separate dedicated modeling choices, datasets, and independent experiments and evaluations. For example: 
  - *Invoice processing*: comparison between LLM-only approaches and hybrid OCR + LLM pipelines.
  - *Inventory Management*: evaluation of demand forecasting (statistical methods vs time-series ML vs LLM-based reasoning), anomaly detection, and waste-related signals.
  - *Revenue Management*: assessment of pricing and upselling recommendations using simulations and comparison against human decision baselines.
  - *Customer Review Analysis*: sentiment classification, topic extraction, and trend detection across unstructured feedback.
  - *Marketing*: content generation and campaign suggestions, potentially involving vision-language models and other evaluation techniques.
  - *Scout Agent*: continuous data collection (web scraping), integration of external sources, and market trend analysis.

- **Agent orchestration and prioritization** 
  
  Designing a supervisory agent capable of **resolving and prioritizing** potentially conflicting signals remains a key challenge (e.g. inventory recommendations to reorder versus margin pressure identified by revenue analysis). Beyond individual agent performance, the effectiveness of the system as a whole, including cross-domain reasoning, prioritization quality, and the role of the General Manager Agent, requires additional layers of experimentation.
  
- **Evaluation methodology and metrics** 
  
  System performance cannot be assessed solely through single-domain metrics such as prediction accuracy or cost reduction. A broader evaluation framework is required to capture coordination quality, decision coherence across agents, and alignment with human judgment.

---
