# Phase 3 Current Path Boost Benchmark

- Digits suites: `digits_pairs, digits_pairs_noisy, digits_pairs_permuted`
- Shared tuned configuration: merge scale `0.02`, distill strength `0.50`, current-path LR boost `2.00`
- Overall best policy by mean avg delta vs adapt_only: `task_specialist_clone_current_path_boost_merge`

## Aggregate Comparison

- `task_specialist_clone_limited_merge`: mean avg delta +5.0%, mean current delta +13.1%, mean forgetting delta -3.8%, mean gain vs fixed -12.3%, suite wins 0
- `task_specialist_clone_distill_merge`: mean avg delta +4.9%, mean current delta +12.8%, mean forgetting delta -3.8%, mean gain vs fixed -12.4%, suite wins 0
- `task_specialist_clone_current_path_boost_merge`: mean avg delta +11.9%, mean current delta +31.4%, mean forgetting delta -10.1%, mean gain vs fixed -5.4%, suite wins 3

## Suite Winners

| Task Suite | Best Policy | limited_merge | distill_merge | current_path_boost_merge |
|---|---|---:|---:|---:|
| digits_pairs | task_specialist_clone_current_path_boost_merge | +5.3% | +5.4% | +11.0% |
| digits_pairs_noisy | task_specialist_clone_current_path_boost_merge | +3.5% | +3.4% | +10.3% |
| digits_pairs_permuted | task_specialist_clone_current_path_boost_merge | +6.3% | +6.0% | +14.5% |
