# Phase 3 Source-Aware Benchmark

- Digits suites: `digits_pairs, digits_pairs_noisy, digits_pairs_permuted`
- Shared tuned configuration: merge scale `0.02`, distill strength `0.50`
- Overall best policy by mean avg delta vs adapt_only: `task_specialist_clone_limited_merge`

## Aggregate Comparison

- `task_specialist_clone_limited_merge`: mean avg delta +5.0%, mean forgetting delta -3.8%, mean gain vs fixed -12.3%, suite wins 2
- `task_specialist_clone_distill_merge`: mean avg delta +4.9%, mean forgetting delta -3.8%, mean gain vs fixed -12.4%, suite wins 1
- `task_specialist_clone_source_aware_merge`: mean avg delta +4.9%, mean forgetting delta -3.8%, mean gain vs fixed -12.4%, suite wins 0

## Suite Winners

| Task Suite | Best Policy | limited_merge | distill_merge | source_aware_merge |
|---|---|---:|---:|---:|
| digits_pairs | task_specialist_clone_distill_merge | +5.3% | +5.4% | +5.4% |
| digits_pairs_noisy | task_specialist_clone_limited_merge | +3.5% | +3.4% | +3.4% |
| digits_pairs_permuted | task_specialist_clone_limited_merge | +6.3% | +6.0% | +6.1% |
