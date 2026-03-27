# Phase 3 Digits Benchmark Transfer

- Digits benchmark family: `digits_pairs, digits_pairs_noisy, digits_pairs_permuted`
- Shared tuned configuration: merge scale `0.02`, distill strength `0.50`
- Overall best reuse policy by mean avg delta vs adapt_only: `task_specialist_clone_limited_merge`
- Hardest suite for that policy vs fixed: `digits_pairs_noisy`

## Aggregate Policy Comparison

- `task_specialist_clone_distill_merge`: mean avg delta +4.9%, mean forgetting delta -3.8%, mean gain vs fixed -12.4%, suite wins 1
- `task_specialist_clone_confidence_gate_merge`: mean avg delta +4.9%, mean forgetting delta -3.8%, mean gain vs fixed -12.4%, suite wins 0
- `task_specialist_clone_limited_merge`: mean avg delta +5.0%, mean forgetting delta -3.8%, mean gain vs fixed -12.3%, suite wins 2

## Suite Comparison

| Task Suite | Best Reuse Policy | Avg Delta vs adapt_only | Forgetting Delta | Gain vs Fixed | Tied Best Policies |
|---|---|---:|---:|---:|---|
| digits_pairs | task_specialist_clone_distill_merge | +5.4% | -4.5% | -11.4% | task_specialist_clone_distill_merge, task_specialist_clone_confidence_gate_merge |
| digits_pairs_noisy | task_specialist_clone_limited_merge | +3.5% | -2.9% | -17.6% | task_specialist_clone_limited_merge |
| digits_pairs_permuted | task_specialist_clone_limited_merge | +6.3% | -4.1% | -7.7% | task_specialist_clone_limited_merge |
