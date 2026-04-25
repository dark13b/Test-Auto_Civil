# AutoCivil MVP Scope

## Product
Inverse concrete mix design assistant: enter target strength and constraints, get suggested mix proportions with predictions and warnings.

## Target User
Civil or materials engineers and small testing labs needing quick mix design starting points.

## Core Workflow
1. User inputs target compressive strength and constraints (w/c ratio limits, aggregate type, admixtures, etc.)
2. System predicts concrete mixes that meet the target using trained models
3. Output: ranked mix suggestions, predicted strength, uncertainty warnings, downloadable report

## In Scope
- Strength-driven inverse mix design
- Input validation and constraint handling
- Mix prediction with confidence/uncertainty indicators
- Warning flags for out-of-range or high-uncertainty results
- Simple report export (PDF or similar)

## Out of Scope for MVP
- Automated model research or literature ingestion
- LLM-based proposal generation system
- Academic benchmarking or model comparison dashboards
- Advanced analytics dashboards
- Any feature not directly part of the engineer's mix design workflow

## Internal / Admin Features
Some internal tooling (model management, experiment tracking, admin dashboards) remains in the codebase but is not part of the customer-facing MVP and should not be presented to end users.

---

> **Not ready for production engineering approval without lab validation.**
> All mix suggestions are model predictions and must be verified through physical trial batches and standard testing before use in real construction.
