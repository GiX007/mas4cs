# MAS4BPO: A Multi-Agent System for Large-Scale Business Processes Optimization

_First Draft — Simple Overview_

## Introduction

This project explores the use of agentic AI workflows to improve restaurant operations by reducing manual coordination, human error, and operational complexity. Built from real-world experience in hospitality, it introduces a multi-agent system with specialized agents for supporting key business functions such as invoice processing, inventory control, revenue management, customer service, reviews analysis, and marketing.

Each agent has a clearly defined role and interacts through shared systems (ERP, POS, CRM), while keeping humans in control of critical decisions. The result is a coordinated, data-driven workflow that helps restaurants operate more efficiently, make better decisions under pressure, and scale more reliably, especially in high-volume, seasonal environments.

The architecture is modular and extensible, making it adaptable to different types of hospitality venues including restaurants, bars, and hotels.

---

## Why This Project Exists

Restaurant operations rely on many interconnected processes: purchasing, inventory control, pricing, service, customer feedback, and marketing. In practice, these processes are handled by different tools and people, often under high pressure and with limited technical support. As a result, information flows through manual communication, errors propagate across departments, and important decisions are delayed or based on incomplete context.

This project exists to address that gap. Instead of replacing existing systems, it introduces an agentic coordination layer that connects them. By using specialized AI agents that share a common understanding of the business state (via ERP, POS, CRM), the system reduces reliance on perfect human coordination and helps managers operate consistently mainly in high-volume, seasonal environments.

The goal is not full automation, but **decision support with human control**, where routine work is handled by agents and humans focus on supervision and strategy.

---

## How the Agents Work Together (End-to-End Flow)

The system operates as a coordinated workflow rather than isolated tools.

First, goods are purchased and supplier invoices arrive. The **Invoice Processing Agent** reads each invoice, validates the data, and records clean purchase entries in the ERP. This ensures that inventory quantities, costs, and supplier information are accurate and up to date.

The **Inventory Manager Agent** continuously reads inventory and sales data from the ERP and POS. It tracks stock levels and valuation, updates recipe and menu costs, detects anomalies (such as unusual usage or cost increases), and predicts future consumption based on sales history and upcoming reservations. Based on these signals, it generates reorder suggestions, which are reviewed and approved by a human operator (typically the chef or manager).

The **Revenue Manager Agent** combines sales performance with cost data to analyze margins, identify high- and low-performing items, and propose pricing, menu, and upselling adjustments. It also leverages customer data from the CRM (e.g., repeat visits, preferences, VIP tags) together with real-time inventory availability and current pricing to recommend customer-specific upselling strategies during service (e.g., what to suggest to a specific customer at a specific moment). All recommendations are presented with explanations and expected impact, and are applied only after human approval.

Customer feedback is handled by two agents. The **Post-Visit Feedback Agent** collects structured feedback shortly after a visit, while the **Customer Review Analysis Agent** processes public reviews and groups sentiment and issues. Both feed structured insights to the **General Manager Agent**, which decides whether operational, pricing, or service actions are required.

The **Customer Service & Reservation Manager Agent** manages guest communication and reservations, tracks guest preferences and VIPs, and detects patterns in cancellations and demand. Its insights help refine service flow and capacity planning.

The **Marketing Agent** uses performance data, customer segments, and seasonal trends to propose campaigns and content ideas, while the **Scout Agent** monitors competitors and market trends to provide external context.

Finally, the **General Manager Agent** acts as the orchestration layer. It aggregates insights from all agents, prioritizes issues, and presents clear, contextual recommendations to human decision-makers. Approved actions are routed back to the relevant agents and their impact is tracked over time.

---

### Important Note on Existing Tools and System Modularity

There are mature vendors in the market that already provide heuristic-based or rule-based solutions for specific domains such as invoice processing, inventory management, marketing automation, and customer relationship management. In environments where these systems reliably produce transactional truth (e.g., accurate purchases, stock levels, reservations, or customer profiles), the corresponding agent may not be required.

The proposed architecture is intentionally modular and composable. It does not assume that all agents must be deployed at once. Instead, the system can be introduced at any point in the workflow, using existing tools as inputs. For example, if a third-party ERP already handles invoice ingestion and inventory valuation correctly, the system can start directly from the Inventory Manager Agent, Revenue Manager Agent, or higher-level agents.

The core value of the architecture is therefore not replacement, but coordination. Even when external tools are used, the General Manager Agent can consume their outputs and combine them with signals from other domains to support cross-functional reasoning, prioritization, and human-in-the-loop decision-making.

Agentic workflows remain critical wherever the business requires:
- cross-domain reasoning
- prioritization across competing objectives
- human-in-the-loop governance
- explainable, contextual recommendations

This design allows organizations to adopt the system incrementally, preserve existing investments, and gradually move from basic automation to coordinated intelligence with minimal disruption.

---

## Architecture Overview (Agent Interaction)

High-level description:
- Core business systems (ERP, POS, CRM) act as shared sources of truth
- Specialized agents read from and write to these systems within clearly defined roles
- Agents do not act autonomously; all strategic decisions flow through the General Manager Agent
- Humans remain in control through approval and supervision loops

Conceptual flow:

    ERP / POS / CRM 
            ↓
    Domain Agents (Inventory, Revenue, Reviews, Customer Service, Marketing, Scout)
            ↓
     General Manager Agent
            ↓
      Human Decision-Maker
            ↓
    Approved actions routed back to agents

This architecture emphasizes coordination over automation, ensuring robustness, explainability, and scalability.

---

## 1. Invoice Processing Agent

**Purpose** 

Convert supplier invoices into accurate, structured purchase records that the ERP can reliably use.

**Interfaces (Reads)** 
- Supplier invoices (PDF or images) received via email or upload
- ERP data (item master, suppliers, historical prices, etc.)

**Interfaces (Writes)** 
- Validated purchase records stored in the ERP (items, quantities, costs, taxes, supplier, date)
- Exception records for invoices or line items requiring human review

**Responsibilities**
- Extract key invoice fields (supplier, date, invoice ID, line items, quantities, prices, taxes, totals)
- Normalize units (kg/pcs/lt) and formats where needed
- Match invoice line items to ERP items
- Validate data integrity (totals, tax calculations, duplicates, abnormal price changes)
- Record the finalized purchase transaction in the ERP
- Escalate only low-confidence or anomalous cases to a human operator

**Real Workflow**
- Before: invoice arrives → manual (human) reading → manual (human) ERP entry → frequent delays and errors
- After: invoice arrives → automated extraction and validation → ERP updated → human involved only for exceptions

---

## 2. Inventory Manager Agent  

**Purpose** 

Maintain a reliable view of inventory levels and costs, and support purchasing and cost-control decisions using sales and demand signals.

**Interfaces (Reads)**
- Inventory data from ERP (stock levels, purchases, costs, adjustments)  
- Recipes and menu items from ERP
- Sales data from POS 
- Historical inventory and sales data from ERP
- Upcoming reservations and events (date, time, pax, notes) from CRM
- Customer profile signals (VIP tags, allergies, strong preferences) from CRM

**Interfaces (Writes / Produces)**
- Current inventory status (quantity and value)
- Cost per recipe and menu item
- Reorder suggestions
- Alerts for inventory-related anomalies (stock shortages, waste, overstock, unusual cost changes)
- Periodic inventory and cost reports (weekly, monthly)  

**Responsibilities**
- Track inventory quantities and valuation using ERP and POS data
- Build and maintain recipe definitions and their links to inventory items
- Recalculate item and recipe costs from ERP purchase records
- Compute food and beverage cost per dish and per time period
- Detect inventory anomalies (unexpected usage, missing stock, abnormal cost changes)
- Estimate short-term demand using sales history and upcoming reservations (e.g., more wine sold on Saturdays)
- Generate reorder quantities based on stock levels and expected demand (upcoming reservations, customer preferences)
- Present reorder suggestions to a human operator (e.g., chef or manager) for approval 
- Apply approved decisions and track outcomes over time
- Monitor waste indicators and identify cost leakage
- Produce periodic summaries and comparisons (week-over-week, month-over-month)

**Real Workflow**
- Before: stock updated manually during invoice processing, costs are often outdated, shortages noticed too late (usually at the end of month)
- After: ERP data continuously analyzed → demand estimated → reorder suggestions reviewed by humans → issues flagged early

**Why this agent is critical**

This agent connects purchasing data with daily operations:
- Without it: ordering and cost control rely on manual checks or static ERP heuristics
- With it: inventory decisions become consistent, explainable, and aligned with real demand

---

### Important Note on Existing Inventory Systems

There are already vendors (or inventory management platforms/systems) that cover the roles of the Invoice Processing Agent and Inventory Manager Agent. These systems are typically rule-based and rely on predefined heuristics, rather than agentic AI. Their main focus is on:
- Recording purchases and stock levels
- Performing inventory counts and cost calculations
- Integrating with the POS to retrieve sales data and produce relevant reports

However, these platforms operate largely in isolation. Their databases or ERPs do not communicate with customer review analysis tools, customer service or reservation systems, or marketing platforms. As a result, coordination across departments depends heavily on manual communication between people.

Different departments (purchasing, operations, service, marketing, management) must manually exchange data, interpretations, and decisions. In practice, a correct operational workflow requires many people across different roles to perform their tasks correctly and simultaneously. For example, purchasing, receiving, invoice entry, and inventory updates must be aligned with service operations, pricing decisions, customer feedback, upcoming reservations, and management planning—both in time and interpretation. This leads to:

- Delays in decision-making
- Inconsistent or conflicting information
- Miscommunication between teams
- Increased probability of human error

This level of coordination is difficult to maintain consistently, especially in seasonal restaurant environments. The core problem is not the lack of inventory software, but the lack of intelligent coordination across business functions.

Seasonal restaurants often operate under extreme conditions:
- Long operating periods with little or no rest
- Staff working for up to 100 consecutive days
- High workload, pressure, and frequent staff turnover

In addition, many employees involved in purchasing, invoice processing, and inventory control do not have the technical background or the patience required to operate complex ERP or inventory systems consistently. In practice, these responsibilities are often handled by a single person (F&B Manager) responsible for both invoice processing and inventory control, alongside other operational duties.

Under these constraints, even well-designed systems become error-prone. Data may be entered late, partially, or incorrectly, and small mistakes quickly propagate across departments, affecting inventory accuracy, pricing decisions, and profitability.

**What the agentic AI workflow solves**

The proposed agentic AI workflow does not aim to replace existing ERPs or inventory tools. Instead, it introduces an intelligent coordination layer on top of them. By treating the ERP together with POS and CRM systems as shared sources of truth and enabling autonomous agents to exchange structured information, the system reduces reliance on human technical expertise and manual coordination.

This approach:
- Reduces the need for deep technical knowledge from operational staff
- Removes dependence on perfect, simultaneous human execution
- Minimizes miscommunication between departments
- Limits error propagation during high-stress and seasonal periods

As a result, human effort shifts from repetitive data entry and system management to supervision and strategic decision-making, where human judgment provides the greatest value.

---

## 3. Revenue Manager Agent

**Purpose**

Support pricing, menu, and upselling decisions by analyzing demand, costs, and customer behavior (in order to maximize revenue and profit), while keeping final control with human decision-makers.

**Interfaces (Reads)**
- Sales data from POS (items sold, time, quantity, revenue)
- Cost data and inventory levels from ERP
- Menu structure and recipe costs from ERP
- Historical sales and revenue data from ERP
- Reservations and demand signals (pax, time, events) from CRM
- Customer signals from CRM (VIP tags, preferences, spending patterns)

**Interfaces (Writes)**
- Pricing adjustment suggestions per item (suggested price adjustments)
- Menu optimization suggestions (reprice, add, remove items)
- Upselling and cross-selling suggestions
- Revenue and performance reports (daily, weekly, monthly)
- Alerts for unusual revenue patterns or margin erosion

**Responsibilities**
- Analyze sales and demand patterns across items, time periods, and seasons
- Combine sales and cost data to compute margins and item contribution
- Identify high-performing, low-performing, and high-risk items
- Generate pricing and menu suggestions based on demand, cost changes, and performance trends
- Generate upselling suggestions using:
  - Temporal patterns (time of day, day of week, season)
  - Historical best-performing combinations
  - Customer-level signals (when available)
- Compare revenue and margin performance across periods (WoW, MoM, YoY)
- Present recommendations with clear explanations and expected impact

**Human-in-the-Loop (Decision Control)**
- Present all pricing, menu, and upselling suggestions for human review
- Allow approval, adjustment, or rejection before any change is applied
- Track outcomes and incorporate human feedback into future recommendations

**Real Workflow**
- Before: prices, menu decisions, and sales strategy are based on intuition and static, delayed reports
- After: continuous analysis → targeted recommendations (pricing, menu structure, sales recommendations) → human approval → impact measured over time

**Why this agent is critical**

This agent connects demand with profitability:
- Without it: pricing and menu changes are subjective and intuition-driven
- With it: revenue optimization becomes continuous, data-driven, explainable, and supervised

**Note**: This agent is not an autonomous pricing engine. It is a **decision-support agent with supervised autonomy**, designed to reduce managerial workload, surface non-obvious opportunities, provide real-time recommendations on what to sell, and preserve human judgement in sensitive decisions

---

## 4. Customer Review Analysis Agent

**Purpose**

Transform customer reviews and feedback into structured insights and alerts that support management decisions, without taking action itself.

**Interfaces (Reads)**
- Public reviews (e.g., Google, TripAdvisor, Instagram, etc.)
- Internal feedback (post-visit surveys, emails, messages)
- Current menu and item list (including recent changes)

**Interfaces (Writes)**
- Overall and topic-based sentiment summaries
- Issue clusters (e.g., slow service, price complaints, food quality issues)
- Item-level signals (dish or drink mentions with associated sentiment)
- Alerts for repeated or high-severity issues
- Periodic review summaries and trend comparisons (weekly, monthly)

**Responsibilities**
- Collect and consolidate feedback from multiple sources
- Clean, filter, and deduplicate content (repeated posts by the same person)
- Classify sentiment and intensity
- Extract and group topics (food, service, price/value, cleanliness, atmosphere)
- Link feedback to specific menu items when possible
- Generate concise, actionable summaries with examples
- Route all findings to the General Manager Agent for interpretation and decision

**Real Workflow**
- Before: reviews read manually requiring much time → issues noticed late or interpreted subjectively → actions are based on personal estimations
- After: feedback analyzed continuously → clear summaries and alerts sent to GM → relevant agent involved → human-approved actions tracked over time

**Why this agent is useful**

This agent turns unstructured customer voice into usable signals:
- Without it: feedback remains noisy, fragmented, and subjective
- With it: recurring problems are detected early, trends are visible, and decisions are more consistent and evidence-based

---

## 5. Post-Visit Feedback Agent

**Purpose**

Collect timely, structured feedback from guests after their visit, before issues escalate to public reviews, and enrich the customer database with satisfaction signals.

**Interfaces (Reads)**
- Reservation history (date, time, party size)
- Customer contact details (email, phone, messaging channel) from CRM (customer DB)

**Interfaces (Writes)**
- Personalized post-visit messages (email, SMS, WhatsApp, etc.)
- Structured feedback records stored in CRM 
- Early alerts for negative or low-satisfaction experiences

**Responsibilities**
- Trigger post-visit feedback requests automatically (e.g., next day)
- Personalize messages based on visit context (first visit, repeat guest, VIP)
- Store feedback in a structured format
- Detect low scores or negative feedback early
- Route critical feedback for human review and follow-up

**Real Workflow**
- Before: feedback collected sporadically via emails or only through public platforms, with no structured link to customer data or operations
- After: structured feedback collected systematically → issues detected early → human intervention occurs before public escalation

**Why this agent is useful**
- Without it: dissatisfaction often surfaces late as public reviews
- With it: negative experiences are detected early, managed privately, and used to improve operations

**Note**: While existing feedback tools can collect responses and basic insights, they are not intelligently connected to the enterprise’s decision-making core. As a result, feedback still requires manual interpretation and cross-department communication to trigger meaningful action.

---

## 6. Customer Service & Reservation Manager Agent

**Purpose**

Handle customer communication and reservations efficiently, while improving service quality, table utilization, and guest experience through intelligent support and human oversight.

**Interfaces (Reads)**
- Customer messages (chat, WhatsApp, email, phone transcripts)
- Reservation data (date, time, party size, table allocation)
- Customer profiles from CRM (visit history, preferences, VIP tags, notes) 
- Restaurant rules (opening hours, table capacity, overbooking limits)

**Interfaces (Writes)**
- Reservation confirmations, modifications, and cancellations
- Answers to general customer questions (menu, hours, policies)
- Alerts for special cases (VIP guests, allergies, special requests)
- Approved customer follow-up actions (e.g., apology messages, recovery gestures) executed after General Manager approval
- Summaries on reservations, cancellations, no-shows, and guest behavior patterns

**Responsibilities**
- Handle routine customer requests for reservations and general information
- Support moderation policies, multilingual communication, and basic sentiment detection
- Confirm, modify, or cancel reservations according to availability rules
- Maintain an up-to-date view of table availability and seating constraints
- Identify repeat guests, VIPs, and high-value customers using CRM data
- Capture and store special requests (allergies, celebrations, seating preferences)
- Execute approved customer follow-up actions in cases of dissatisfaction, only after evaluation and authorization by the General Manager Agent
- Escalate complex, sensitive, or ambiguous interactions to human staff
- Learn from human corrections to improve future handling

**Real Workflow**
- Before: reservations handled via basic tools; guest context often lost; no systematic use of customer data
- After: agent manages routine communication → identifies guests via CRM → flags important cases → humans decide → agent executes approved actions

**Why this agent is super critical**
- Without it: service quality depends heavily on individual staff memory and availability, and reservations ignore customer history
- With it: guest interactions become consistent, informed, and scalable, while preserving human judgment and strategic control

**Important Note**: There are existing vendors that integrate reservations with POS and basic customer data. However, these systems typically operate in isolation and are not intelligently connected to inventory, revenue, feedback analysis, or centralized decision support. In this architecture, such tools can be used as inputs, while the agentic workflow provides coordination and decision support across the entire operation.

---

## 7. Scout Agent

**Purpose**

Monitor the external environment (competitors, market trends, supplier signals) and provide early insights that support strategic and tactical decisions.

**Interfaces (Reads)**
- Public competitor information (menus, prices, promotions)
- Online reviews and ratings of competitors
- Public social media content and food or hospitality trends
- Optional: supplier-related signals and cost trends

**Interfaces (Writes)**
- Competitor comparisons (menu structure, pricing, positioning)
- Detected market and trend signals (e.g., rising demand for a specific dish)
- Alerts for relevant external changes
- Periodic market intelligence summaries

**Responsibilities**
- Continuously scan publicly available competitor and market data
- Track competitor menu updates, pricing shifts, and promotions
- Identify emerging food, drink, and experience trends
- Compare competitor offerings with the restaurant’s current menu and pricing
- Detect potential opportunities (new dishes, price adjustments, differentiation)
- Flag risks (price pressure, trend shifts, supplier dependencies)
- Summarize findings in concise, actionable reports
- Route all insights and alerts to the General Manager Agent for evaluation

**Real Workflow**
- Before: competitor monitoring done informally and inconsistently
- After: continuous external scanning → structured insights sent to GM → humans decide → strategy adapts proactively

**Why this agent is critical**

This agent ensures the business looks outward as well as inward:
- Without it: strategy reacts late to external changes
- With it: management gains early awareness and can act proactively

---

## 8. Marketing Agent

**Purpose**

Support marketing and promotional activities by generating data-driven content and campaign recommendations aligned with revenue goals, customer behavior, and operational constraints.

**Interfaces (Reads)**
- Menu items and current pricing
- Sales and performance data (from POS and Revenue Manager outputs)
- Customer behavior signals (visit frequency, preferences, VIP tags)
- Seasonal trends and calendar events
- Social media and past campaign performance data (engagement, reach, conversions)
- Optional: competitor and market signals (from Scout Agent)

**Interfaces (Writes)**
- Content drafts (social posts, short texts, emails, blog ideas)
- Campaign and promotion suggestions
- Recommendations on which items or offers to promote
- Campaign performance summaries and insights

**Responsibilities**
- Generate marketing content ideas aligned with brand tone and business goals
- Identify promotable items based on margin, demand, seasonality, and availability
- Propose campaigns and promotions supported by performance data
- Monitor campaign performance and summarize results
- Present content drafts and campaign ideas for human review
- Never publish content or launch campaigns autonomously

**Real Workflow**
- Before: marketing decisions driven by intuition and limited feedback
- After: data-backed campaign ideas generated → humans approve → performance measured and refined

**Why this agent is useful**

This agent connects marketing activity with operations and revenue:
- Without it: promotions may boost demand while harming margins or operations
- With it: marketing becomes coordinated, measurable, and aligned with business strategy

---

## 9. General Manager Agent 

**Purpose**

Act as the central coordination and decision-support layer that integrates insights from all agents, prioritizes actions, and supports management with clear, contextual recommendations. It functions as the coordination and reasoning layer of the system, not a replacement for human leadership.

**Interfaces (Reads)** 
- Reports, alerts, and summaries from all other agents
- Core business data from ERP, POS, and CRM
- Direct prompts and questions from owners or managers

**Interfaces (Writes)** 
- Consolidated management reports (daily, weekly, monthly)
- Prioritized alerts and recommendations
- Action proposals routed to the appropriate agent (after approval)
- Real-time dashboards and key business indicators

**Responsibilities**
- Aggregate and organize outputs from all agents into a unified view
- Combine financial, operational, customer, and market signals
- Detect cross-domain patterns (e.g., rising costs combined with negative feedback and margin pressure)
- Prioritize issues and opportunities based on impact and urgency
- Decide which agent(s) should analyze or execute a given task
- Present clear, explainable recommendations to human decision-makers
- Track approved decisions and monitor outcomes over time
- Adapt reporting and communication based on manager preferences and feedback

**Human-in-the-Loop**
- Never execute strategic or sensitive actions autonomously
- Support decisions by presenting options, context, and expected impact
- Allow owners or managers to approve, modify, or reject proposals
- Learn from human decisions to improve prioritization and communication style
- Operate strictly as an advisory and coordination layer
- Route approved actions to execution agents and ensure system-wide alignment

**Real Workflow**
- Before: managers manually combine reports, emails, reviews, and intuition
- After: agents generate insights → General Manager Agent synthesizes and prioritizes → humans decide → actions are executed and tracked

**Why this agent is critical**

This agent provides coherence to the entire system:
- Without it: agents operate as disconnected tools producing fragmented insights
- With it: the business benefits from a unified, explainable, and human-centered intelligence layer

---

## Future Agents (Planned Extensions)

The following agents are not part of the initial system but represent natural extensions of the architecture. They further automate front-of-house operations, service coordination, and workforce analytics, building on the same agentic and human-in-the-loop principles.

### 10. Transaction Agent (Voice Assistant for Bar & Floor)
- Captures orders via voice interaction
- Confirms and logs transactions in real time
- Reduces order errors, fraud, and unrecorded sales
- Minimizes manual POS (human) interaction during peak hours

### 11. Kitchen & Service Coordinator Agent
- Tracks dish preparation status in the kitchen
- Synchronizes kitchen output with service staff
- Detects delays and bottlenecks early
- Reduces miscommunication between kitchen and floor teams

### 12. Seating Manager Agent
- Optimizes table layout and seating allocation
- Dynamically adjusts seating plans based on reservations and walk-ins
- Improves capacity utilization without harming guest experience

### 13. Workforce Evaluation Agent
- Analyzes staff performance using objective metrics (e.g., covers served, revenue per shift, upselling effectiveness)
- Produces consistent and unbiased performance summaries
- Supports decisions on training, incentives, and staffing optimization

---

## Implementation

- PostgreSQL for ERP, POS, CRM
- Reflection and evaluation steps can be built into every agent
- ...

---

## Experiments

...

---

## Results

...

### Discussion on some edge cases

Example 1:

    Customer Review 
            ↓
     Customer Review Analysis Agent
            ↓
      General Manager Agent
            ↓
       Revenue Manager Agent
            ↓
        Human Approval (Owner/Manager)

Example 2:

...

---

## Notes  

...

---

## Sources  
- "9 Genius Ways Restaurants Are Using AI": https://sevenrooms.com/blog/restaurant-AI/
- "15 Ways AI Is Impacting the Restaurant Industry": https://www.netsuite.com/portal/resource/articles/business-strategy/ai-in-restaurants.shtml?utm_source=chatgpt.com
- Agentic workflows: LangChain and LangGraph docs  
- https://sevenrooms.com/platform/artificial-intelligence/

- https://www.salesforce.com/retail/artificial-intelligence/ai-agents-in-restaurants/
- https://www.zendesk.com/blog/ai-for-restaurants/
- https://digiqt.com/blog/ai-agents-in-restaurant-tech/
- https://www.domo.com/glossary/inventory-management-ai-agents
- https://www.highradius.com/resources/Blog/agentic-ai-invoice-processing/

---
