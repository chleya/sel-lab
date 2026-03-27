# Phase 3 Quantized Memory Path Benchmark

- Digits suites: `digits_pairs, digits_pairs_noisy, digits_pairs_permuted`
- Archived memory precision: `4`-bit
- Overall best policy by mean avg delta vs `adapt_only`: `task_specialist_clone_current_path_boost_merge`

## Aggregate Comparison

- `task_specialist_clone_current_path_boost_merge`: mean avg delta +13.3%, mean current delta +34.1%, mean forgetting delta -12.8%, mean gain vs fixed -4.0%, suite wins 3
- `task_specialist_clone_external_memory_boost_merge`: mean avg delta +10.7%, mean current delta +30.1%, mean forgetting delta -13.6%, mean gain vs fixed -6.6%, suite wins 0

## Suite Winners

- `digits_pairs`: best `task_specialist_clone_current_path_boost_merge`, internal=+12.9%, external=+9.7%
- `digits_pairs_noisy`: best `task_specialist_clone_current_path_boost_merge`, internal=+11.9%, external=+8.3%
- `digits_pairs_permuted`: best `task_specialist_clone_current_path_boost_merge`, internal=+15.1%, external=+14.1%
