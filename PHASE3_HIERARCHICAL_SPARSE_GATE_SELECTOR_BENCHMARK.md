# Phase 3 Hierarchical Sparse-Gate Selector Benchmark

Hierarchical segmented selector with quadratic region heads and explicit sparse-gate-aware supervision.

- Train family: `selector_full_map`
- Eval family: `selector_full_map`
- Overall best policy: `task_specialist_clone_two_stage_selector_merge`

## Region Heads

- `default`: samples=60, mean utility=-0.038, bias=-0.229
- `embedded`: samples=30, mean utility=-0.005, bias=+7.040
- `sparse_embedded`: samples=6, mean utility=-0.068, bias=+70.444

## Training Regions

- `plasticity_stress` / `digits_pairs_noisy`: mean utility -0.125, mean adjusted utility -0.125, regions default=12, embedded=0, sparse_embedded=0
- `interference_stress` / `feature_shift`: mean utility +0.003, mean adjusted utility +0.003, regions default=12, embedded=0, sparse_embedded=0
- `interference_stress_strong` / `feature_shift_noisy`: mean utility +0.022, mean adjusted utility +0.022, regions default=12, embedded=0, sparse_embedded=0
- `retrieval_ambiguity_stress` / `digits_pairs_permuted`: mean utility -0.086, mean adjusted utility -0.086, regions default=12, embedded=0, sparse_embedded=0
- `mixed_regime_stress` / `feature_shift_mixed`: mean utility -0.005, mean adjusted utility -0.005, regions default=12, embedded=0, sparse_embedded=0
- `selector_generalization_gate` / `feature_shift_nonnegative`: mean utility -0.001, mean adjusted utility -0.001, regions default=0, embedded=12, sparse_embedded=0
- `selector_adversarial_gate` / `feature_shift_embedded`: mean utility +0.004, mean adjusted utility +0.010, regions default=0, embedded=12, sparse_embedded=0
- `selector_sparse_adversarial_gate` / `feature_shift_sparse_embedded`: mean utility -0.046, mean adjusted utility -0.056, regions default=0, embedded=6, sparse_embedded=6

## Aggregate

- `adapt_only`: mean avg delta +0.0%, current delta +0.0%, prior delta +0.0%, forgetting delta +0.0%, gain vs fixed +0.9%, wins 0
- `task_specialist_clone_limited_merge`: mean avg delta +1.7%, current delta +3.7%, prior delta +1.0%, forgetting delta -2.1%, gain vs fixed +2.7%, wins 0
- `task_specialist_clone_current_path_boost_merge`: mean avg delta +3.6%, current delta +13.4%, prior delta +0.4%, forgetting delta -6.8%, gain vs fixed +4.6%, wins 3
- `task_specialist_clone_regime_switch_merge`: mean avg delta +4.0%, current delta +10.7%, prior delta +1.7%, forgetting delta -4.2%, gain vs fixed +4.9%, wins 0
- `task_specialist_clone_signature_selector_merge`: mean avg delta +3.9%, current delta +11.8%, prior delta +1.2%, forgetting delta -5.9%, gain vs fixed +4.8%, wins 1
- `task_specialist_clone_task_ranking_selector_merge`: mean avg delta +4.2%, current delta +11.5%, prior delta +1.7%, forgetting delta -4.9%, gain vs fixed +5.1%, wins 3
- `task_specialist_clone_two_stage_selector_merge`: mean avg delta +4.4%, current delta +11.5%, prior delta +2.0%, forgetting delta -4.7%, gain vs fixed +5.3%, wins 1
- `task_specialist_clone_hierarchical_quadratic_selector_merge`: mean avg delta +1.7%, current delta +3.7%, prior delta +1.0%, forgetting delta -2.1%, gain vs fixed +2.7%, wins 0
- `task_specialist_clone_hierarchical_sparse_gate_selector_merge`: mean avg delta +4.1%, current delta +11.3%, prior delta +1.7%, forgetting delta -5.0%, gain vs fixed +5.0%, wins 0
