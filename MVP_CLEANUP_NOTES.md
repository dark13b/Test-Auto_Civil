# MVP Cleanup Notes

Generated: 2026-04-24

This note lists items that look stale or misleading and should be reviewed later. It is intentionally non-destructive.

| Item | Issue | Risk | Recommended action |
| --- | --- | --- | --- |
| `PROJECT_INDEX.md` | Contains status/model claims that may lag the current artifact state. | High: review trust can break when the index conflicts with live outputs. | Update later |
| `program.md` | Appears to be a legacy pointer document rather than the current source of truth. | Medium: readers may follow an outdated entry point. | Archive later |
| `test_track_b.py` | Root-level test file may not be covered by the configured pytest path. | Medium: creates a false sense of test coverage. | Archive later |
| `config_track_b_additions.yaml` | Track B config may only exist for legacy or excluded test paths. | Medium: can confuse current runtime and test ownership. | Update later |
| `proposal_engine_patch.py` | Patch/reference module may overlap with active proposal code and still looks implementation-like. | Medium: source-of-truth ambiguity. | Archive later |
| `research_loop_patch.py` | Patch/reference module may be a legacy branch artifact rather than current runtime. | Medium: maintenance confusion and accidental reuse risk. | Archive later |
| Docs mentioning `final_metrics` as source of truth | Older docs may still describe deprecated artifact semantics. | High: can misstate the authoritative outputs and acceptance flow. | Update later |
| Customer-facing research/LLM claims if any remain | External-facing language may overstate research maturity, model status, or readiness. | High: can mislead users about scope and validation state. | Update later |

Notes:

- None of the items above should be deleted as part of this cleanup pass.
- If an item is still needed for history, prefer archiving or marking it explicitly as legacy.
