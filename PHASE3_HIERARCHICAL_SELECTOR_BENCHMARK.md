# Phase 3 Hierarchical Selector Benchmark

Hierarchical segmented selector with region-specific ranking heads over `default`, `embedded`, and `sparse_embedded` regions.

- Train family: `selector_full_map`
- Eval family: `selector_full_map`
- Overall best policy: `task_specialist_clone_two_stage_selector_merge`
- Task-ranking payload source: `F:\sel-lab\results\canonical\phase3_task_ranking_selector_benchmark.json`
- Two-stage payload source: `F:\sel-lab\results\canonical\phase3_two_stage_selector_benchmark.json`
- Learned-router payload source: `F:\sel-lab\results\canonical\phase3_learned_router_selector_benchmark.json`

## Region Heads

- `default`: samples=60, mean utility=-0.038, weights conflict_score=+0.198, conflict_peak=-0.280, confidence=-0.675, input_abs_mean=+0.699, input_nonnegative_ratio=-2.094, input_zero_ratio=+1.456, loss_mean=+0.733, loss_delta=+5.586, conflict_delta=+0.532, confidence_delta=-0.452, bias=+1.306
- `embedded`: samples=30, mean utility=-0.008, weights conflict_score=-0.094, conflict_peak=-1.051, confidence=-2.974, input_abs_mean=-1.596, input_nonnegative_ratio=+0.000, input_zero_ratio=+1.414, loss_mean=+0.305, loss_delta=+9.960, conflict_delta=-9.427, confidence_delta=+0.349, bias=+6.327
- `sparse_embedded`: samples=6, mean utility=-0.048, weights conflict_score=+1.954, conflict_peak=+1.524, confidence=+10.241, input_abs_mean=+33.533, input_nonnegative_ratio=+0.000, input_zero_ratio=-466.725, loss_mean=+0.172, loss_delta=+1.001, conflict_delta=+99.791, confidence_delta=-9.470, bias=+136.022

## Training Regions

- `plasticity_stress` / `digits_pairs_noisy`: mean utility -0.125, regions default=12, embedded=0, sparse_embedded=0
- `interference_stress` / `feature_shift`: mean utility +0.003, regions default=12, embedded=0, sparse_embedded=0
- `interference_stress_strong` / `feature_shift_noisy`: mean utility +0.022, regions default=12, embedded=0, sparse_embedded=0
- `retrieval_ambiguity_stress` / `digits_pairs_permuted`: mean utility -0.086, regions default=12, embedded=0, sparse_embedded=0
- `mixed_regime_stress` / `feature_shift_mixed`: mean utility -0.005, regions default=12, embedded=0, sparse_embedded=0
- `selector_generalization_gate` / `feature_shift_nonnegative`: mean utility -0.001, regions default=0, embedded=12, sparse_embedded=0
- `selector_adversarial_gate` / `feature_shift_embedded`: mean utility +0.004, regions default=0, embedded=12, sparse_embedded=0
- `selector_sparse_adversarial_gate` / `feature_shift_sparse_embedded`: mean utility -0.046, regions default=0, embedded=6, sparse_embedded=6

## Aggregate

- `adapt_only`: mean avg delta +0.0%, current delta +0.0%, prior delta +0.0%, forgetting delta +0.0%, gain vs fixed +0.9%, wins 0
- `task_specialist_clone_limited_merge`: mean avg delta +1.7%, current delta +3.7%, prior delta +1.0%, forgetting delta -2.1%, gain vs fixed +2.7%, wins 0
- `task_specialist_clone_current_path_boost_merge`: mean avg delta +3.6%, current delta +13.4%, prior delta +0.4%, forgetting delta -6.8%, gain vs fixed +4.6%, wins 3
- `task_specialist_clone_regime_switch_merge`: mean avg delta +4.0%, current delta +10.7%, prior delta +1.7%, forgetting delta -4.2%, gain vs fixed +4.9%, wins 0
- `task_specialist_clone_signature_selector_merge`: mean avg delta +3.9%, current delta +11.8%, prior delta +1.2%, forgetting delta -5.9%, gain vs fixed +4.8%, wins 1
- `task_specialist_clone_task_ranking_selector_merge`: mean avg delta +4.2%, current delta +11.5%, prior delta +1.7%, forgetting delta -4.9%, gain vs fixed +5.1%, wins 3
- `task_specialist_clone_two_stage_selector_merge`: mean avg delta +4.4%, current delta +11.5%, prior delta +2.0%, forgetting delta -4.7%, gain vs fixed +5.3%, wins 1
- `task_specialist_clone_learned_router_selector_merge`: mean avg delta +3.9%, current delta +12.3%, prior delta +1.1%, forgetting delta -5.9%, gain vs fixed +4.8%, wins 0
- `task_specialist_clone_hierarchical_selector_merge`: mean avg delta +4.2%, current delta +11.4%, prior delta +1.7%, forgetting delta -4.8%, gain vs fixed +5.1%, wins 0
