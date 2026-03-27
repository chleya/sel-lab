# Phase 3 Regime Switch Oracle

Upper-bound check for a high-level regime switch that selects one policy per stress benchmark.

- Source family: `expanded_stress_map`

## Aggregate

- `task_specialist_clone_current_path_boost_merge`: mean avg delta +6.7%, current delta +21.6%, prior delta +1.8%, forgetting delta -8.8%, gain vs fixed +2.8%
- `task_specialist_clone_limited_merge`: mean avg delta +2.9%, current delta +6.9%, prior delta +1.6%, forgetting delta -2.5%, gain vs fixed -1.0%
- `task_specialist_clone_mode_switch_boost_merge`: mean avg delta -1.0%, current delta -2.0%, prior delta -0.7%, forgetting delta -5.3%, gain vs fixed +6.0%
- `task_specialist_clone_dual_mode_update_merge`: mean avg delta -1.0%, current delta -2.0%, prior delta -0.7%, forgetting delta -5.3%, gain vs fixed +6.0%
- `regime_switch_oracle`: mean avg delta +7.5%, current delta +20.9%, prior delta +3.0%, forgetting delta -6.7%, gain vs fixed +3.6%

## Per Benchmark Choice

### plasticity_stress

- Task suite: `digits_pairs_noisy`
- Oracle choice: `task_specialist_clone_current_path_boost_merge`
- Benchmark winner: `task_specialist_clone_current_path_boost_merge`
- Chosen result: avg delta +13.6%, current delta +24.0%, prior delta +10.1%, forgetting delta -9.7%, gain vs fixed -9.5%

### interference_stress

- Task suite: `feature_shift`
- Oracle choice: `task_specialist_clone_limited_merge`
- Benchmark winner: `task_specialist_clone_limited_merge`
- Chosen result: avg delta -0.1%, current delta +0.7%, prior delta -0.3%, forgetting delta -0.6%, gain vs fixed +12.8%

### interference_stress_strong

- Task suite: `feature_shift_noisy`
- Oracle choice: `task_specialist_clone_limited_merge`
- Benchmark winner: `task_specialist_clone_limited_merge`
- Chosen result: avg delta +0.7%, current delta -2.0%, prior delta +1.6%, forgetting delta -1.0%, gain vs fixed +7.7%

### retrieval_ambiguity_stress

- Task suite: `digits_pairs_permuted`
- Oracle choice: `task_specialist_clone_current_path_boost_merge`
- Benchmark winner: `task_specialist_clone_current_path_boost_merge`
- Chosen result: avg delta +15.8%, current delta +61.0%, prior delta +0.7%, forgetting delta -15.7%, gain vs fixed +3.4%
