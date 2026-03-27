# Phase 3 Decoupled Objective Boost Benchmark

- Digits suites: `digits_pairs, digits_pairs_noisy, digits_pairs_permuted`
- Shared tuned configuration: current-path boost `2.50`, consistency strength `0.50`
- Overall best policy by mean avg delta vs adapt_only: `task_specialist_clone_current_path_boost_merge`

## Aggregate Comparison

- `task_specialist_clone_limited_merge`: mean avg delta +5.0%, mean current delta +13.1%, mean forgetting delta -3.8%, mean gain vs fixed -12.3%, suite wins 0
- `task_specialist_clone_current_path_boost_merge`: mean avg delta +13.2%, mean current delta +34.0%, mean forgetting delta -12.8%, mean gain vs fixed -4.1%, suite wins 2
- `task_specialist_clone_decoupled_objective_boost_merge`: mean avg delta +13.2%, mean current delta +34.0%, mean forgetting delta -12.8%, mean gain vs fixed -4.1%, suite wins 1
