# Phase 3 Asymmetric Retention Benchmark

- Digits suites: `digits_pairs, digits_pairs_noisy, digits_pairs_permuted`
- Shared tuned configuration: merge scale `0.02`, distill strength `0.50`, retention strength `0.12`
- Overall best policy by mean avg delta vs adapt_only: `task_specialist_clone_limited_merge`

## Aggregate Comparison

- `task_specialist_clone_limited_merge`: mean avg delta +5.0%, mean forgetting delta -3.8%, mean gain vs fixed -12.3%, suite wins 2
- `task_specialist_clone_distill_merge`: mean avg delta +4.9%, mean forgetting delta -3.8%, mean gain vs fixed -12.4%, suite wins 1
- `task_specialist_clone_asymmetric_retention_merge`: mean avg delta +4.8%, mean forgetting delta -3.9%, mean gain vs fixed -12.5%, suite wins 0

## Suite Winners

| Task Suite | Best Policy | limited_merge | distill_merge | asymmetric_retention_merge |
|---|---|---:|---:|---:|
| digits_pairs | task_specialist_clone_distill_merge | +5.3% | +5.4% | +5.2% |
| digits_pairs_noisy | task_specialist_clone_limited_merge | +3.5% | +3.4% | +3.5% |
| digits_pairs_permuted | task_specialist_clone_limited_merge | +6.3% | +6.0% | +5.8% |
