# Phase 3 Distill Local Merge Sweep

- Benchmark: `digits_pairs`
- Distill strength: `0.50`
- Best avg delta vs `adapt_only`: merge `0.02`, avg delta +5.4%, forgetting delta -4.5%
- Closest to fixed: merge `0.02`, avg gain vs fixed -11.4%

## Sweep

- merge `0.01`: avg vs adapt +5.3%, forgetting vs adapt -4.5%, vs fixed -11.5%
- merge `0.02`: avg vs adapt +5.4%, forgetting vs adapt -4.5%, vs fixed -11.4%
- merge `0.03`: avg vs adapt +5.3%, forgetting vs adapt -4.4%, vs fixed -11.5%
