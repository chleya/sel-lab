# Phase 3 Learned Router Selector Benchmark

Learned three-way router over regime features for boost, interference control, and sparse-boost routing.

- Train family: `selector_full_map`
- Eval family: `selector_full_map`
- Probe policy for feature collection: `task_specialist_clone_current_path_boost_merge`
- Task-ranking fallback payload: `F:\sel-lab\results\canonical\phase3_task_ranking_selector_benchmark.json`
- Overall best policy: `task_specialist_clone_two_stage_selector_merge`
- Router classes: `plasticity_boost`, `interference_control`, `sparse_plasticity_boost`

## Router Weights

- `plasticity_boost`: conflict_score=-1.283, conflict_peak=+0.368, confidence=+0.519, input_abs_mean=-3.137, input_nonnegative_ratio=-1.021, input_zero_ratio=-9.123, loss_mean=-1.124, loss_delta=-3.319, conflict_delta=-1.028, confidence_delta=+0.538, bias=+7.621
- `interference_control`: conflict_score=+2.656, conflict_peak=-1.726, confidence=-2.920, input_abs_mean=+1.256, input_nonnegative_ratio=-0.880, input_zero_ratio=+1.781, loss_mean=-0.733, loss_delta=+5.617, conflict_delta=-3.677, confidence_delta=-0.221, bias=+2.371
- `sparse_plasticity_boost`: conflict_score=-1.373, conflict_peak=+1.358, confidence=+2.402, input_abs_mean=+1.882, input_nonnegative_ratio=+1.901, input_zero_ratio=+7.342, loss_mean=+1.857, loss_delta=-2.298, conflict_delta=+4.706, confidence_delta=-0.317, bias=-9.992

## Training Targets

- `plasticity_stress` / `digits_pairs_noisy`: labels plasticity_boost=12, interference_control=0, sparse_plasticity_boost=0, mean sample weight 2.50
- `interference_stress` / `feature_shift`: labels plasticity_boost=6, interference_control=6, sparse_plasticity_boost=0, mean sample weight 1.33
- `interference_stress_strong` / `feature_shift_noisy`: labels plasticity_boost=5, interference_control=7, sparse_plasticity_boost=0, mean sample weight 1.55
- `retrieval_ambiguity_stress` / `digits_pairs_permuted`: labels plasticity_boost=11, interference_control=1, sparse_plasticity_boost=0, mean sample weight 2.23
- `mixed_regime_stress` / `feature_shift_mixed`: labels plasticity_boost=8, interference_control=4, sparse_plasticity_boost=0, mean sample weight 1.18
- `selector_generalization_gate` / `feature_shift_nonnegative`: labels plasticity_boost=10, interference_control=2, sparse_plasticity_boost=0, mean sample weight 1.07
- `selector_adversarial_gate` / `feature_shift_embedded`: labels plasticity_boost=7, interference_control=5, sparse_plasticity_boost=0, mean sample weight 3.94
- `selector_sparse_adversarial_gate` / `feature_shift_sparse_embedded`: labels plasticity_boost=0, interference_control=3, sparse_plasticity_boost=9, mean sample weight 5.65

## Aggregate

- `adapt_only`: mean avg delta +0.0%, current delta +0.0%, prior delta +0.0%, forgetting delta +0.0%, gain vs fixed +0.9%, wins 0
- `task_specialist_clone_limited_merge`: mean avg delta +1.7%, current delta +3.7%, prior delta +1.0%, forgetting delta -2.1%, gain vs fixed +2.7%, wins 0
- `task_specialist_clone_current_path_boost_merge`: mean avg delta +3.6%, current delta +13.4%, prior delta +0.4%, forgetting delta -6.8%, gain vs fixed +4.6%, wins 3
- `task_specialist_clone_regime_switch_merge`: mean avg delta +4.0%, current delta +10.7%, prior delta +1.7%, forgetting delta -4.2%, gain vs fixed +4.9%, wins 0
- `task_specialist_clone_fingerprint_selector_merge`: mean avg delta +4.0%, current delta +12.9%, prior delta +1.1%, forgetting delta -5.6%, gain vs fixed +5.0%, wins 0
- `task_specialist_clone_signature_selector_merge`: mean avg delta +3.9%, current delta +11.8%, prior delta +1.2%, forgetting delta -5.9%, gain vs fixed +4.8%, wins 1
- `task_specialist_clone_task_ranking_selector_merge`: mean avg delta +4.2%, current delta +11.5%, prior delta +1.7%, forgetting delta -4.9%, gain vs fixed +5.1%, wins 3
- `task_specialist_clone_two_stage_selector_merge`: mean avg delta +4.4%, current delta +11.5%, prior delta +2.0%, forgetting delta -4.7%, gain vs fixed +5.3%, wins 1
- `task_specialist_clone_learned_router_selector_merge`: mean avg delta +3.9%, current delta +12.3%, prior delta +1.1%, forgetting delta -5.9%, gain vs fixed +4.8%, wins 0
