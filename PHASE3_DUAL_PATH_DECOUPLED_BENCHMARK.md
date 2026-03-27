# Phase 3 Dual Path Decoupled Benchmark

- Digits suites: `digits_pairs, digits_pairs_noisy, digits_pairs_permuted`
- Shared tuned configuration: current-path boost `2.50`, retention strength `0.12`
- Overall best policy by mean avg delta vs adapt_only: `task_specialist_clone_current_path_boost_merge`

## Aggregate Comparison

- `task_specialist_clone_limited_merge`: mean avg delta +5.0%, mean current delta +13.1%, mean forgetting delta -3.8%, mean gain vs fixed -12.3%, suite wins 0
- `task_specialist_clone_current_path_boost_merge`: mean avg delta +13.2%, mean current delta +34.0%, mean forgetting delta -12.8%, mean gain vs fixed -4.1%, suite wins 2
- `task_specialist_clone_dual_path_boost_merge`: mean avg delta +10.9%, mean current delta +30.7%, mean forgetting delta -13.5%, mean gain vs fixed -6.4%, suite wins 0
- `task_specialist_clone_dual_path_decoupled_merge`: mean avg delta +13.2%, mean current delta +34.5%, mean forgetting delta -13.3%, mean gain vs fixed -4.1%, suite wins 1
