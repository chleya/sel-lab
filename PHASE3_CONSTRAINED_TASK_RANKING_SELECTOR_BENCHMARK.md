# Phase 3 Constrained Task Ranking Selector Benchmark

Constrained task-level pairwise-ranking selector with explicit sparse-gate protection.

- Train family: `selector_full_map`
- Eval family: `selector_full_map`
- Probe policy for feature collection: `task_specialist_clone_current_path_boost_merge`
- Overall best policy: `task_specialist_clone_constrained_task_ranking_selector_merge`
- Ranking weights: conflict_score=+0.496, conflict_peak=-0.247, confidence=-0.772, input_abs_mean=+1.106, input_nonnegative_ratio=-0.393, input_zero_ratio=+0.986, loss_mean=-0.890, loss_delta=+6.212, conflict_delta=+1.177, confidence_delta=-0.463, bias=+0.530

## Training Targets

- `plasticity_stress` / `digits_pairs_noisy`: mean task utility -0.125, mean pair scale 1.00
- `interference_stress` / `feature_shift`: mean task utility +0.003, mean pair scale 1.00
- `interference_stress_strong` / `feature_shift_noisy`: mean task utility +0.022, mean pair scale 1.00
- `retrieval_ambiguity_stress` / `digits_pairs_permuted`: mean task utility -0.086, mean pair scale 1.00
- `mixed_regime_stress` / `feature_shift_mixed`: mean task utility -0.005, mean pair scale 1.00
- `selector_generalization_gate` / `feature_shift_nonnegative`: mean task utility -0.001, mean pair scale 1.00
- `selector_adversarial_gate` / `feature_shift_embedded`: mean task utility -0.007, mean pair scale 1.95
- `selector_sparse_adversarial_gate` / `feature_shift_sparse_embedded`: mean task utility -0.061, mean pair scale 2.80

## Aggregate

- `adapt_only`: mean avg delta +0.0%, current delta +0.0%, prior delta +0.0%, forgetting delta +0.0%, gain vs fixed +0.9%, wins 0
- `task_specialist_clone_limited_merge`: mean avg delta +1.7%, current delta +3.7%, prior delta +1.0%, forgetting delta -2.1%, gain vs fixed +2.7%, wins 0
- `task_specialist_clone_current_path_boost_merge`: mean avg delta +3.6%, current delta +13.4%, prior delta +0.4%, forgetting delta -6.8%, gain vs fixed +4.6%, wins 4
- `task_specialist_clone_regime_switch_merge`: mean avg delta +4.0%, current delta +10.7%, prior delta +1.7%, forgetting delta -4.2%, gain vs fixed +4.9%, wins 0
- `task_specialist_clone_fingerprint_selector_merge`: mean avg delta +4.0%, current delta +12.9%, prior delta +1.1%, forgetting delta -5.6%, gain vs fixed +5.0%, wins 0
- `task_specialist_clone_signature_selector_merge`: mean avg delta +3.9%, current delta +11.8%, prior delta +1.2%, forgetting delta -5.9%, gain vs fixed +4.8%, wins 1
- `task_specialist_clone_task_ranking_selector_merge`: mean avg delta +1.7%, current delta +3.7%, prior delta +1.0%, forgetting delta -2.1%, gain vs fixed +2.7%, wins 0
- `task_specialist_clone_constrained_task_ranking_selector_merge`: mean avg delta +4.2%, current delta +11.8%, prior delta +1.6%, forgetting delta -5.0%, gain vs fixed +5.1%, wins 3
