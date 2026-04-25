# Integration Guide

Source: [INTEGRATION_GUIDE.md](<E:\Random IDEA\AutoResearch\auto-civil-lab\INTEGRATION_GUIDE.md>)

## Source Summary

This is a legacy Track B integration guide that describes a proposed patch set for the research loop, proposal engine, and governance layer. The document is explicitly marked as legacy reference only.

## Source Facts

- It proposes new modules such as `knowledge_base.py`, `novelty_scorer.py`, and `hypothesis_archive.py`.
- It proposes adding Track B orchestration hooks to `research_loop.py`.
- It proposes enhanced proposal parsing and a richer research results schema.
- It contains many patch snippets and implementation notes rather than experimental results.

## Research Objective

Extend the research loop with knowledge-base, novelty, and hypothesis-tracking behavior.

## Materials and Mix Details

- Not a mix-material source in itself.
- It is relevant only because it changes how concrete experiments are proposed and recorded.

## Methodology

- Hook the research loop.
- Add failure context and knowledge context.
- Filter proposals by novelty.
- Record hypotheses before and after each trial.
- Review scientific reports during each cycle.

## Key Results

- No empirical results are reported.
- The document is a design and patch plan.

## Limitations

- It is not the authoritative implementation state.
- It is historical and may diverge from the current code.
- Some proposed hooks may already have been superseded by later refactors.

## Relevance To The Current Research Direction

- Useful for tracking how the repository started to think about knowledge-base structure, novelty tracking, and scientific reporting.
- Useful as a conceptual predecessor to the wiki itself.

## Related Pages

- [Top-level overview](<E:\Random IDEA\AutoResearch\auto-civil-lab\wiki\overview\top-level-overview.md>)
- [Knowledge gaps](<E:\Random IDEA\AutoResearch\auto-civil-lab\wiki\overview\gaps.md>)

