# MAS4CS Vision

MAS4CS is built around a clear definition of what reliable automated customer service looks like. 
This document captures that definition from the user's perspective and serves as the design target for all architectural decisions.

---

## Definition of the Perfect Customer Service

This subsection defines, in simple terms, what a successful MAS4CS system looks like from the user’s perspective.

A perfect customer service system:
- **Understands the user’s goal correctly**
  - User asks to find, book, cancel, or request information
  - System identifies the correct intent and domain
- **Tracks all user requirements accurately**
  - User specifies constraints (area, price, number of people, dates)
  - System stores and updates them without losing information
- **Provides exactly what the user asked for**
  - If user requests phone, address, email address → system provides them
  - If user requests booking → system completes booking correctly
- **Enforces policy and business rules**
  - Does not book without required information
  - Does not skip mandatory steps
  - Asks for missing data before proceeding
- **Handles multi-domain conversations correctly**
  - User switches from restaurant to hotel
  - System keeps relevant constraints and does not forget context
- **Does not hallucinate**
  - Does not invent slot values
  - Does not provide unsupported information
- **Produces clear and helpful responses**
  - Answers are concise, correct, and aligned with user intent
  - No contradictions or irrelevant content
- **Completes the dialogue successfully**
  - The final user goal is satisfied
  - No unresolved requests remain
  - No policy violations occurred

In simple terms: 

A successful MAS4CS system correctly understands the user, tracks all constraints, follows rules, avoids hallucinations, and completes the requested task reliably from start to finish.

---

