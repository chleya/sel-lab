# Phase 3 Dynamics Selector Benchmark

Linear fitted selector over richer online regime features benchmarked on the full selector stress map.

- Train family: `expanded_stress_map`
- Eval family: `selector_full_map`
- Probe policy for feature collection: `task_specialist_clone_current_path_boost_merge`
- Overall best policy: `task_specialist_clone_fingerprint_selector_merge`
- Fitted weights: conflict_score=+0.271, conflict_peak=-0.189, confidence=+0.016, input_abs_mean=+0.615, input_nonnegative_ratio=-2.601, input_zero_ratio=-0.010, loss_mean=-0.159, loss_delta=-0.216, conflict_delta=+0.039, confidence_delta=-0.043, bias=+1.422

## Aggregate

- `adapt_only`: mean avg delta +0.0%, current delta +0.0%, prior delta +0.0%, forgetting delta +0.0%, gain vs fixed +0.9%, wins 0
- `task_specialist_clone_limited_merge`: mean avg delta +1.7%, current delta +3.7%, prior delta +1.0%, forgetting delta -2.1%, gain vs fixed +2.7%, wins 3
- `task_specialist_clone_current_path_boost_merge`: mean avg delta +3.6%, current delta +13.4%, prior delta +0.4%, forgetting delta -6.8%, gain vs fixed +4.6%, wins 4
- `task_specialist_clone_regime_switch_merge`: mean avg delta +4.0%, current delta +10.7%, prior delta +1.7%, forgetting delta -4.2%, gain vs fixed +4.9%, wins 0
- `task_specialist_clone_fingerprint_selector_merge`: mean avg delta +4.0%, current delta +12.9%, prior delta +1.1%, forgetting delta -5.6%, gain vs fixed +5.0%, wins 0
- `task_specialist_clone_signature_selector_merge`: mean avg delta +3.9%, current delta +11.8%, prior delta +1.2%, forgetting delta -5.9%, gain vs fixed +4.8%, wins 1
- `task_specialist_clone_fitted_selector_merge`: mean avg delta +1.7%, current delta +3.7%, prior delta +1.0%, forgetting delta -2.1%, gain vs fixed +2.7%, wins 0
- `task_specialist_clone_prototype_selector_merge`: mean avg delta +1.7%, current delta +3.7%, prior delta +1.0%, forgetting delta -2.1%, gain vs fixed +2.7%, wins 0
- `task_specialist_clone_dynamics_selector_merge`: mean avg delta +4.0%, current delta +12.9%, prior delta +1.1%, forgetting delta -5.6%, gain vs fixed +5.0%, wins 0

## Per Benchmark

### plasticity_stress

- Task suite: `digits_pairs_noisy`
- Intent: stress current-task learning under noisy sequential digits
- Best policy: `task_specialist_clone_current_path_boost_merge`

- `adapt_only`: avg delta +0.0%, current delta +0.0%, prior delta +0.0%, forgetting delta +0.0%, gain vs fixed -23.1%
- `task_specialist_clone_limited_merge`: avg delta +4.7%, current delta +6.7%, prior delta +4.1%, forgetting delta -2.3%, gain vs fixed -18.3%
- `task_specialist_clone_current_path_boost_merge`: avg delta +13.6%, current delta +24.0%, prior delta +10.1%, forgetting delta -9.7%, gain vs fixed -9.5%
- `task_specialist_clone_regime_switch_merge`: avg delta +13.6%, current delta +24.0%, prior delta +10.1%, forgetting delta -9.7%, gain vs fixed -9.5%
- `task_specialist_clone_fingerprint_selector_merge`: avg delta +13.6%, current delta +24.0%, prior delta +10.1%, forgetting delta -9.7%, gain vs fixed -9.5%
- `task_specialist_clone_signature_selector_merge`: avg delta +13.6%, current delta +24.0%, prior delta +10.1%, forgetting delta -9.7%, gain vs fixed -9.5%
- `task_specialist_clone_fitted_selector_merge`: avg delta +4.7%, current delta +6.7%, prior delta +4.1%, forgetting delta -2.3%, gain vs fixed -18.3%
- `task_specialist_clone_prototype_selector_merge`: avg delta +4.7%, current delta +6.7%, prior delta +4.1%, forgetting delta -2.3%, gain vs fixed -18.3%
- `task_specialist_clone_dynamics_selector_merge`: avg delta +13.6%, current delta +24.0%, prior delta +10.1%, forgetting delta -9.7%, gain vs fixed -9.5%

### interference_stress

- Task suite: `feature_shift`
- Intent: separate raw specialist interference from low-merge control
- Best policy: `task_specialist_clone_limited_merge`

- `adapt_only`: avg delta +0.0%, current delta +0.0%, prior delta +0.0%, forgetting delta +0.0%, gain vs fixed +12.8%
- `task_specialist_clone_limited_merge`: avg delta -0.1%, current delta +0.7%, prior delta -0.3%, forgetting delta -0.6%, gain vs fixed +12.8%
- `task_specialist_clone_current_path_boost_merge`: avg delta -0.9%, current delta +3.0%, prior delta -2.2%, forgetting delta -3.6%, gain vs fixed +11.9%
- `task_specialist_clone_regime_switch_merge`: avg delta -0.1%, current delta +0.7%, prior delta -0.3%, forgetting delta -0.6%, gain vs fixed +12.8%
- `task_specialist_clone_fingerprint_selector_merge`: avg delta -0.1%, current delta +0.7%, prior delta -0.3%, forgetting delta -0.6%, gain vs fixed +12.8%
- `task_specialist_clone_signature_selector_merge`: avg delta -0.9%, current delta +3.0%, prior delta -2.2%, forgetting delta -3.8%, gain vs fixed +11.9%
- `task_specialist_clone_fitted_selector_merge`: avg delta -0.1%, current delta +0.7%, prior delta -0.3%, forgetting delta -0.6%, gain vs fixed +12.8%
- `task_specialist_clone_prototype_selector_merge`: avg delta -0.1%, current delta +0.7%, prior delta -0.3%, forgetting delta -0.6%, gain vs fixed +12.8%
- `task_specialist_clone_dynamics_selector_merge`: avg delta -0.1%, current delta +0.7%, prior delta -0.3%, forgetting delta -0.6%, gain vs fixed +12.8%

### interference_stress_strong

- Task suite: `feature_shift_noisy`
- Intent: stress interference control under stronger feature-shift noise
- Best policy: `task_specialist_clone_limited_merge`

- `adapt_only`: avg delta +0.0%, current delta +0.0%, prior delta +0.0%, forgetting delta +0.0%, gain vs fixed +7.0%
- `task_specialist_clone_limited_merge`: avg delta +0.7%, current delta -2.0%, prior delta +1.6%, forgetting delta -1.0%, gain vs fixed +7.7%
- `task_specialist_clone_current_path_boost_merge`: avg delta -1.5%, current delta -1.7%, prior delta -1.4%, forgetting delta -6.3%, gain vs fixed +5.5%
- `task_specialist_clone_regime_switch_merge`: avg delta +0.7%, current delta -2.0%, prior delta +1.6%, forgetting delta -1.0%, gain vs fixed +7.7%
- `task_specialist_clone_fingerprint_selector_merge`: avg delta +0.7%, current delta -2.0%, prior delta +1.6%, forgetting delta -1.0%, gain vs fixed +7.7%
- `task_specialist_clone_signature_selector_merge`: avg delta -0.8%, current delta -2.0%, prior delta -0.4%, forgetting delta -5.2%, gain vs fixed +6.2%
- `task_specialist_clone_fitted_selector_merge`: avg delta +0.7%, current delta -2.0%, prior delta +1.6%, forgetting delta -1.0%, gain vs fixed +7.7%
- `task_specialist_clone_prototype_selector_merge`: avg delta +0.7%, current delta -2.0%, prior delta +1.6%, forgetting delta -1.0%, gain vs fixed +7.7%
- `task_specialist_clone_dynamics_selector_merge`: avg delta +0.7%, current delta -2.0%, prior delta +1.6%, forgetting delta -1.0%, gain vs fixed +7.7%

### retrieval_ambiguity_stress

- Task suite: `digits_pairs_permuted`
- Intent: test whether routing/retrieval helps under ambiguous digit remapping
- Best policy: `task_specialist_clone_current_path_boost_merge`

- `adapt_only`: avg delta +0.0%, current delta +0.0%, prior delta +0.0%, forgetting delta +0.0%, gain vs fixed -12.3%
- `task_specialist_clone_limited_merge`: avg delta +6.4%, current delta +22.3%, prior delta +1.0%, forgetting delta -6.3%, gain vs fixed -6.0%
- `task_specialist_clone_current_path_boost_merge`: avg delta +15.8%, current delta +61.0%, prior delta +0.7%, forgetting delta -15.7%, gain vs fixed +3.4%
- `task_specialist_clone_regime_switch_merge`: avg delta +15.8%, current delta +61.0%, prior delta +0.7%, forgetting delta -15.7%, gain vs fixed +3.4%
- `task_specialist_clone_fingerprint_selector_merge`: avg delta +15.8%, current delta +61.0%, prior delta +0.7%, forgetting delta -15.7%, gain vs fixed +3.4%
- `task_specialist_clone_signature_selector_merge`: avg delta +15.8%, current delta +61.0%, prior delta +0.7%, forgetting delta -15.7%, gain vs fixed +3.4%
- `task_specialist_clone_fitted_selector_merge`: avg delta +6.4%, current delta +22.3%, prior delta +1.0%, forgetting delta -6.3%, gain vs fixed -6.0%
- `task_specialist_clone_prototype_selector_merge`: avg delta +6.4%, current delta +22.3%, prior delta +1.0%, forgetting delta -6.3%, gain vs fixed -6.0%
- `task_specialist_clone_dynamics_selector_merge`: avg delta +15.8%, current delta +61.0%, prior delta +0.7%, forgetting delta -15.7%, gain vs fixed +3.4%

### mixed_regime_stress

- Task suite: `feature_shift_mixed`
- Intent: stress selector behavior when acquisition and interference pressure are mixed within the same suite
- Best policy: `task_specialist_clone_limited_merge`

- `adapt_only`: avg delta +0.0%, current delta +0.0%, prior delta +0.0%, forgetting delta +0.0%, gain vs fixed +16.9%
- `task_specialist_clone_limited_merge`: avg delta -0.9%, current delta -2.0%, prior delta -0.6%, forgetting delta -1.6%, gain vs fixed +16.0%
- `task_specialist_clone_current_path_boost_merge`: avg delta -1.2%, current delta -0.7%, prior delta -1.4%, forgetting delta -3.0%, gain vs fixed +15.7%
- `task_specialist_clone_regime_switch_merge`: avg delta -0.9%, current delta -2.0%, prior delta -0.6%, forgetting delta -1.6%, gain vs fixed +16.0%
- `task_specialist_clone_fingerprint_selector_merge`: avg delta -0.9%, current delta -2.0%, prior delta -0.6%, forgetting delta -1.6%, gain vs fixed +16.0%
- `task_specialist_clone_signature_selector_merge`: avg delta -1.0%, current delta -0.3%, prior delta -1.2%, forgetting delta -2.8%, gain vs fixed +15.9%
- `task_specialist_clone_fitted_selector_merge`: avg delta -0.9%, current delta -2.0%, prior delta -0.6%, forgetting delta -1.6%, gain vs fixed +16.0%
- `task_specialist_clone_prototype_selector_merge`: avg delta -0.9%, current delta -2.0%, prior delta -0.6%, forgetting delta -1.6%, gain vs fixed +16.0%
- `task_specialist_clone_dynamics_selector_merge`: avg delta -0.9%, current delta -2.0%, prior delta -0.6%, forgetting delta -1.6%, gain vs fixed +16.0%

### selector_generalization_gate

- Task suite: `feature_shift_nonnegative`
- Intent: check whether selector logic overreacts to digits-like nonnegative inputs without true adversarial embedding
- Best policy: `task_specialist_clone_current_path_boost_merge`

- `adapt_only`: avg delta +0.0%, current delta +0.0%, prior delta +0.0%, forgetting delta +0.0%, gain vs fixed +10.7%
- `task_specialist_clone_limited_merge`: avg delta +1.4%, current delta -1.3%, prior delta +2.3%, forgetting delta +2.0%, gain vs fixed +12.2%
- `task_specialist_clone_current_path_boost_merge`: avg delta +1.5%, current delta -1.0%, prior delta +2.3%, forgetting delta +2.0%, gain vs fixed +12.3%
- `task_specialist_clone_regime_switch_merge`: avg delta +1.4%, current delta -1.3%, prior delta +2.3%, forgetting delta +2.0%, gain vs fixed +12.2%
- `task_specialist_clone_fingerprint_selector_merge`: avg delta +1.5%, current delta -1.0%, prior delta +2.3%, forgetting delta +2.0%, gain vs fixed +12.3%
- `task_specialist_clone_signature_selector_merge`: avg delta +1.5%, current delta -1.0%, prior delta +2.3%, forgetting delta +2.0%, gain vs fixed +12.3%
- `task_specialist_clone_fitted_selector_merge`: avg delta +1.4%, current delta -1.3%, prior delta +2.3%, forgetting delta +2.0%, gain vs fixed +12.2%
- `task_specialist_clone_prototype_selector_merge`: avg delta +1.4%, current delta -1.3%, prior delta +2.3%, forgetting delta +2.0%, gain vs fixed +12.2%
- `task_specialist_clone_dynamics_selector_merge`: avg delta +1.5%, current delta -1.0%, prior delta +2.3%, forgetting delta +2.0%, gain vs fixed +12.3%

### selector_adversarial_gate

- Task suite: `feature_shift_embedded`
- Intent: stress selector robustness when interference is embedded in digits-like nonnegative inputs
- Best policy: `task_specialist_clone_signature_selector_merge`

- `adapt_only`: avg delta +0.0%, current delta +0.0%, prior delta +0.0%, forgetting delta +0.0%, gain vs fixed -3.3%
- `task_specialist_clone_limited_merge`: avg delta +2.8%, current delta +7.0%, prior delta +1.3%, forgetting delta -1.0%, gain vs fixed -0.6%
- `task_specialist_clone_current_path_boost_merge`: avg delta +1.7%, current delta +17.3%, prior delta -3.6%, forgetting delta -8.7%, gain vs fixed -1.7%
- `task_specialist_clone_regime_switch_merge`: avg delta +2.8%, current delta +7.0%, prior delta +1.3%, forgetting delta -1.0%, gain vs fixed -0.6%
- `task_specialist_clone_fingerprint_selector_merge`: avg delta +1.7%, current delta +17.3%, prior delta -3.6%, forgetting delta -8.7%, gain vs fixed -1.7%
- `task_specialist_clone_signature_selector_merge`: avg delta +3.3%, current delta +8.7%, prior delta +1.6%, forgetting delta -2.9%, gain vs fixed +0.0%
- `task_specialist_clone_fitted_selector_merge`: avg delta +2.8%, current delta +7.0%, prior delta +1.3%, forgetting delta -1.0%, gain vs fixed -0.6%
- `task_specialist_clone_prototype_selector_merge`: avg delta +2.8%, current delta +7.0%, prior delta +1.3%, forgetting delta -1.0%, gain vs fixed -0.6%
- `task_specialist_clone_dynamics_selector_merge`: avg delta +1.7%, current delta +17.3%, prior delta -3.6%, forgetting delta -8.7%, gain vs fixed -1.7%

### selector_sparse_adversarial_gate

- Task suite: `feature_shift_sparse_embedded`
- Intent: stress selector robustness under sparse embedded interference cues rather than dense digits-like statistics
- Best policy: `task_specialist_clone_current_path_boost_merge`

- `adapt_only`: avg delta +0.0%, current delta +0.0%, prior delta +0.0%, forgetting delta +0.0%, gain vs fixed -1.2%
- `task_specialist_clone_limited_merge`: avg delta -1.2%, current delta -1.3%, prior delta -1.2%, forgetting delta -5.8%, gain vs fixed -2.4%
- `task_specialist_clone_current_path_boost_merge`: avg delta +0.2%, current delta +5.0%, prior delta -1.4%, forgetting delta -9.4%, gain vs fixed -1.0%
- `task_specialist_clone_regime_switch_merge`: avg delta -1.2%, current delta -1.3%, prior delta -1.2%, forgetting delta -5.8%, gain vs fixed -2.4%
- `task_specialist_clone_fingerprint_selector_merge`: avg delta +0.2%, current delta +5.0%, prior delta -1.4%, forgetting delta -9.4%, gain vs fixed -1.0%
- `task_specialist_clone_signature_selector_merge`: avg delta -0.4%, current delta +1.3%, prior delta -1.0%, forgetting delta -8.9%, gain vs fixed -1.6%
- `task_specialist_clone_fitted_selector_merge`: avg delta -1.2%, current delta -1.3%, prior delta -1.2%, forgetting delta -5.8%, gain vs fixed -2.4%
- `task_specialist_clone_prototype_selector_merge`: avg delta -1.2%, current delta -1.3%, prior delta -1.2%, forgetting delta -5.8%, gain vs fixed -2.4%
- `task_specialist_clone_dynamics_selector_merge`: avg delta +0.2%, current delta +5.0%, prior delta -1.4%, forgetting delta -9.4%, gain vs fixed -1.0%

