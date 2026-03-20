# Whole-Repository Review, Written By Someone Who Should Have Closed The IDE Hours Ago

## Scope

This is a repo-wide review of the main runtime surfaces, not just the current diff.

Reviewed areas:

- `train.py`
- `search.py`
- `research_loop.py`
- `research_protocol.py`
- `research_lab.py`
- `validator.py`
- `uncertainty.py`
- `report.py`
- `llm_backend.py`
- `proposal_engine.py`
- `llm_proposer.py`
- `dashboard.py`
- `benchmark.py`
- `design_tool.py`
- `config.yaml`
- `requirements.txt`
- `launch_auto_research.bat`
- every file under `tests/`

This is not a style pass. This is the "what undermines correctness, scientific credibility, reproducibility, safety, or future maintenance" pass.

## Findings

### P0 - The project burns the holdout set as a search tool, then still calls it a final evaluation

**Files:** `train.py`, `research_loop.py`, `report.py`, `uncertainty.py`

The code creates one train/test split in `train.py:179-195`, then spends the rest of the repository treating that same `x_test` and `y_test` as if they are free public infrastructure.

Key places:

- `train.py:494-510` evaluates every candidate on `x_test`
- `research_loop.py:632-640` does it again for every scout candidate
- `research_loop.py:717-730` re-evaluates the current best reference on the same holdout before confirm
- `research_loop.py:742-759` evaluates every confirm candidate on the same holdout
- `report.py:269-350` uses the same split again for the final report
- `uncertainty.py:81-96` and `uncertainty.py:360-441` use the same held-out test partition again for interval auditing

Academic translation:

- there is no untouched final test set left
- there is no nested CV
- there is no honest separation between model selection and final reporting

This means the reported "final" numbers are not final. They are the result of repeated exposure to the same holdout. By the time the dashboard is admiring them, the holdout has already been demoted from evaluation set to lab assistant.

### P0 - The stacking ensemble is outright test leakage with fake CV labels

**Files:** `train.py`, `search.py`

`build_stacking_ensemble()` fits on `x_train`, predicts on `x_test`, and then writes those holdout metrics into both the primary score fields and the `cv_*` aliases:

- `train.py:610-640`

Then `search.py` compares that ensemble score against the current best and can promote it as the new best model:

- `search.py:1009-1035`

That is bad enough on its own. The extra insult is that the result object labels holdout metrics as CV metrics. So the code is not only leaking the test set, it is also lying about what kind of number it just produced.

At that point this stops being "messy evaluation" and becomes "results laundering with extra steps."

### P1 - The uncertainty module calibrates a different model than the one the project reports as deployed

**Files:** `uncertainty.py`, `report.py`

`UncertaintyEstimator` loads the kept model, but then for conformal intervals it clones that model and refits a new copy on a reduced split:

- `uncertainty.py:79-96`
- `uncertainty.py:141-171`

Later, interval predictions come from that refit conformal model:

- `uncertainty.py:273-297`

But the report's main predictions come from the saved best model:

- `report.py:272-299`

So the report mixes:

- point predictions from the saved full model
- interval widths and confidence labels from a different retrained submodel

That is not a small implementation detail. It means the interval artifact is not actually describing uncertainty around the predictor the rest of the repo is celebrating. If this were presented in a methods section, Reviewer #2 would start sharpening the knife before the abstract ended.

### P1 - The LLM control loop explicitly asks for hidden reasoning and then treats hidden reasoning as executable output

**Files:** `proposal_engine.py`, `llm_backend.py`, `tests/test_proposal_engine.py`

The compact prompt path tells the model:

- `proposal_engine.py:521-522`

to think step by step internally.

Then `_perform_interaction()` and `_extract_structured_output()` will accept structured JSON from `thinking_text` if the visible response is empty:

- `proposal_engine.py:438-449`
- `proposal_engine.py:775-804`

The backend helpers actively surface `thinking`, `thought`, and `reasoning` channels:

- `llm_backend.py:145-205`

And the tests explicitly approve this behavior:

- `tests/test_proposal_engine.py:240-268`

That means the system is using hidden-reasoning channels as control-plane data for experiment proposals. This is exactly the kind of thing that becomes impossible to reason about across providers, transport modes, and future model versions.

You do not get to say "output only JSON" and then, when that fails, quietly rummage through the model's internal scratchpad like a raccoon in a dumpster.

### P1 - The backend abstraction is not actually abstract; it leaks local-model assumptions and environment-dependent behavior

**Files:** `proposal_engine.py`, `llm_backend.py`, `llm_proposer.py`, `config.yaml`, `tests/test_proposal_engine.py`, `tests/test_llm_backend.py`

`ProposalEngine._resolve_model_hint()` treats `model_hint=None` as "use `default_local_proposal_model`":

- `proposal_engine.py:43-47`

That value is a local Ollama model by default:

- `llm_backend.py:15-60`
- `config.yaml` under the `llm` section

`OpenAIBackend.generate_text()` then uses any supplied model string verbatim:

- `llm_backend.py:428-479`

So in `backend_mode: openai`, a no-hint proposal call can try to send `qwen3:8b` to OpenAI. Meanwhile `LLMProposer._resolve_model_hints()` actually does backend-aware model resolution:

- `llm_proposer.py:140-155`

So the repo contains two different ideas of what "default model" means, depending on which wrapper you came through. Excellent. Two clocks, no truth.

It gets worse:

- default config enables LLMs and hybrid fallback, so environment variables can change whether the repo stays local or goes remote
- Ollama runtime `options` are forwarded over HTTP but ignored by the CLI fallback path (`llm_backend.py:337-349` vs `llm_backend.py:372-409`)
- tests mostly cover the easy path and bless the wrong default-model behavior

This is not backend abstraction. It is backend mood swings.

### P1 - The dashboard and reporting layer mix incompatible metrics, flatten data incorrectly, and expose browser-side injection surfaces

**Files:** `dashboard.py`, `report.py`, `research_loop.py`

The scientific problem:

- the overview cards compare baseline CV numbers against best-model holdout numbers (`dashboard.py:870-915`)
- the improvement card uses `composite_improvement_pct`, which is built from `composite_score` in `research_loop.py:178-197` and `report.py:304-307`
- for ordinary candidates `composite_score` is CV-based (`train.py:511-518`)
- for stacking ensembles it is holdout-based while still wearing a fake CV nametag (`train.py:615-637`)

So the dashboard is not presenting one metric story. It is stitching together whichever numbers happened to be lying around and hoping the gradients distract you.

The data-integrity problem:

- `normalize_result_payload()` overwrites `suspicious_count` with `dataset_anomaly_count` in `dashboard.py:123-128`

Those are not synonyms. That is corrupting the validator summary before the UI even starts improvising.

The security problem:

- auth is disabled whenever `DASHBOARD_PASSWORD` is unset (`dashboard.py:228-239`)
- multiple views dump raw values into `innerHTML`, including record notes and JSON-derived strings (`dashboard.py:1011-1031`, `dashboard.py:1308-1321`, `dashboard.py:1375-1387`)

If this ever escapes localhost, the dashboard becomes a stored-XSS scrapbook with optional authentication.

### P2 - The deterministic "exploit current best" logic is mathematically wrong for log-scaled search spaces

**Files:** `research_lab.py`, `config.yaml`, `tests/test_research_lab.py`

The exploit mutator adds or subtracts a linear delta for float parameters:

- `research_lab.py:199-211`

But the search space contains multiple parameters explicitly marked `log: true`, including:

- `Ridge.alpha` in `config.yaml:163-167`
- `ElasticNet.alpha` in `config.yaml:178-182`
- `SVR.C`, `SVR.epsilon`, `SVR.gamma` in `config.yaml:194-208`

Linear perturbation in a log space is not "local search." It is scale confusion with confidence.

The test coverage does not protect you here:

- `tests/test_research_lab.py:7-43`

checks only that multiple exploit variants are produced, not that they make any numerical sense in log-scaled domains.

### P2 - Reproducibility is weak before the experiments even start

**Files:** `llm_backend.py`, `requirements.txt`, `launch_auto_research.bat`

`llm_backend.py` imports `requests` at import time:

- `llm_backend.py:12`

But `requirements.txt` still does not declare it:

- `requirements.txt:1-10`

Attempting to run the current unit suite with the repo's virtualenv fails on import for the LLM modules because `requests` is missing.

Then the Windows launcher picks `venv\\Scripts\\python.exe` if present and otherwise falls back to bare `python`, even though this repo already has `.venv` in use as well. That is not an execution policy. That is interpreter roulette.

The whole project talks a big game about autonomous research. A clean environment currently replies with a dependency error and a coin flip.

### P2 - The persistent research state is allowed to rot

**Files:** `research_lab.py`, `research_protocol.py`

The checked-in `LAB_STATE` already contains duplicate accepted experiment IDs with different scores:

- `research_lab.py:28-37`

And `apply_keep_to_research_surface()` just appends new accepted entries without any dedupe or integrity check:

- `research_protocol.py:694-706`

If accepted experiments are supposed to be a ratcheting source of truth, then duplicate IDs should be treated as corruption, not as quirky personality.

### P2 - The test suite is aimed at plumbing, not at the places where the repo can actually lie

**Files:** `tests/`

There are only seven test files, and the distribution is telling:

- no `test_uncertainty.py`
- no `test_report.py`
- no direct tests for `search.py`
- no end-to-end test proving the training/search/report/dashboard artifact chain stays scientifically consistent

Specific weak spots:

- `tests/test_dashboard.py:6-15` is basically an import pulse check
- `tests/test_proposal_engine.py:240-268` blesses thinking-channel fallback instead of rejecting it
- `tests/test_research_lab.py:7-43` checks count and uniqueness of exploit variants, not log-space correctness
- `tests/test_llm_backend.py:111-140` covers Ollama HTTP options, not CLI fallback semantics

The suite is much better at verifying that JSON-shaped things remain JSON-shaped than at verifying that the project still deserves to call itself a research system.

## Edge Cases You Are Still Missing

- Any serious long-running search overfits the one holdout set because scout, confirm, ensemble selection, report generation, and uncertainty audit all touch it.
- A best-model artifact selected after many cycles has no honest untouched benchmark left.
- `backend_mode: openai` plus `model_hint=None` can route an Ollama model name into the OpenAI path.
- `backend_mode: hybrid` can silently change local-vs-remote behavior depending on machine environment.
- Ollama HTTP and CLI transport paths do not honor the same runtime configuration.
- Log-scaled hyperparameters can be "exploited" with numerically meaningless linear jumps.
- Uncertainty intervals and displayed point predictions can disagree because they come from different fitted models.
- Validator summary counts shown in the dashboard can be numerically wrong even when the underlying artifact is correct.
- Any untrusted note or JSON field rendered through `innerHTML` can become a browser injection vector.
- Duplicate experiment IDs can accumulate in ratcheted state without any guardrail.

## Verification Notes

I attempted to run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_llm_backend tests.test_proposal_engine tests.test_research_lab tests.test_dashboard
```

Result:

- `tests.test_llm_backend` and `tests.test_proposal_engine` fail to import because `requests` is not installed and is not declared in `requirements.txt`
- the suite therefore does not currently provide a clean verification baseline for the LLM stack

This review is based on source inspection plus that failed verification attempt. The code did not earn a green badge, so I am not pretending it did.

## Bottom Line

There are good ideas in this repository:

- governed search instead of blind trial spam
- engineering-aware validation
- uncertainty and inverse-design ambitions
- a provider-agnostic LLM layer in principle

The implementation is not yet operating at the level those ideas require.

Right now the biggest themes are:

- the evaluation protocol is scientifically compromised
- the LLM control loop is too willing to improvise with hidden channels
- the artifact/report/dashboard layer cannot keep one metric story straight
- reproducibility depends too much on ambient machine state
- the tests mostly guard convenience paths, not credibility

If this were an academic submission, the rejection reason would be simple:

the system is ambitious, but the experimental protocol does not support the confidence of its presentation.

If this were a production review, the reason would be even simpler:

too many core decisions still depend on wishful thinking instead of hard boundaries.
