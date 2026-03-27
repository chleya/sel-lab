# Phase 3 Distill Joint Sweep

- Benchmark: `digits_pairs`
- Best avg delta vs `adapt_only`: merge `0.02`, distill `0.25`, avg delta +5.4%, forgetting delta -4.5%
- Closest to fixed: merge `0.02`, distill `0.25`, avg gain vs fixed -11.4%

## Grid

- merge `0.02`, distill `0.25`: avg vs adapt +5.4%, forgetting vs adapt -4.5%, vs fixed -11.4%
- merge `0.02`, distill `0.50`: avg vs adapt +5.4%, forgetting vs adapt -4.5%, vs fixed -11.4%
- merge `0.02`, distill `0.75`: avg vs adapt +5.4%, forgetting vs adapt -4.5%, vs fixed -11.4%
- merge `0.05`, distill `0.25`: avg vs adapt +5.0%, forgetting vs adapt -4.4%, vs fixed -11.8%
- merge `0.05`, distill `0.50`: avg vs adapt +5.0%, forgetting vs adapt -4.4%, vs fixed -11.8%
- merge `0.05`, distill `0.75`: avg vs adapt +5.0%, forgetting vs adapt -4.5%, vs fixed -11.8%
- merge `0.10`, distill `0.25`: avg vs adapt +5.0%, forgetting vs adapt -4.1%, vs fixed -11.8%
- merge `0.10`, distill `0.50`: avg vs adapt +4.9%, forgetting vs adapt -4.1%, vs fixed -11.9%
- merge `0.10`, distill `0.75`: avg vs adapt +4.9%, forgetting vs adapt -4.2%, vs fixed -11.9%
