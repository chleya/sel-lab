# Phase 3 Two-Stage Selector Sweep

Sparse-route threshold sweep for the two-stage selector on `selector_full_map`.

- Family: `selector_full_map`
- Focused benchmarks: `plasticity_stress`, `mixed_regime_stress`, `selector_adversarial_gate`, `selector_sparse_adversarial_gate`
- Source task-ranking payload: `F:\sel-lab\results\canonical\phase3_task_ranking_selector_benchmark.json`

## Best Candidate

- `zero_ratio_floor=0.385`
- `conflict_delta_floor=-0.005`
- aggregate gain vs fixed `+1.5%`
- aggregate delta vs adapt_only `+4.1%`
- embedded gain vs fixed `+0.0%`
- sparse gain vs fixed `-0.8%`
- feasible `False`

## Top Candidates

- `zr=0.385`, `cd=-0.005`: aggregate +1.5% vs fixed, embedded +0.0%, sparse -0.8%, feasible=False
- `zr=0.385`, `cd=-0.015`: aggregate +1.5% vs fixed, embedded +0.0%, sparse -0.8%, feasible=False
- `zr=0.365`, `cd=-0.005`: aggregate +1.5% vs fixed, embedded -0.1%, sparse -0.8%, feasible=False
- `zr=0.385`, `cd=0.005`: aggregate +1.5% vs fixed, embedded +0.0%, sparse -0.9%, feasible=False
- `zr=0.405`, `cd=-0.005`: aggregate +1.3% vs fixed, embedded +0.0%, sparse -1.6%, feasible=False
