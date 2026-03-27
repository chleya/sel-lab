# Phase 3 Diagnostic Benchmark Family

Formal stress-map benchmark family for screening Phase 3 mechanisms across plasticity, interference, and retrieval pressure.

- Family: `expanded_stress_map`
- Benchmarks: `plasticity_stress, interference_stress, interference_stress_strong, retrieval_ambiguity_stress`

## Policy Wins

- `task_specialist_clone_current_path_boost_merge`: 2
- `task_specialist_clone_limited_merge`: 2

## Family Wins

- `plasticity_boost`: 2
- `specialist_reuse`: 2

## Aggregate

- `adapt_only`: mean avg delta +0.0%, current delta +0.0%, prior delta +0.0%, forgetting delta +0.0%, gain vs fixed -3.9%, wins 0/4
- `task_specialist_clone`: mean avg delta -2.3%, current delta +2.3%, prior delta -3.9%, forgetting delta -3.8%, gain vs fixed +10.5%, wins 0/1
- `task_specialist_clone_current_path_boost_merge`: mean avg delta +6.7%, current delta +21.6%, prior delta +1.8%, forgetting delta -8.8%, gain vs fixed +2.8%, wins 2/4
- `task_specialist_clone_distill_merge`: mean avg delta +4.5%, current delta +6.5%, prior delta +3.9%, forgetting delta -2.5%, gain vs fixed -18.5%, wins 0/1
- `task_specialist_clone_dual_mode_update_merge`: mean avg delta -1.0%, current delta -2.0%, prior delta -0.7%, forgetting delta -5.3%, gain vs fixed +6.0%, wins 0/1
- `task_specialist_clone_limited_merge`: mean avg delta +2.9%, current delta +6.9%, prior delta +1.6%, forgetting delta -2.5%, gain vs fixed -1.0%, wins 2/4
- `task_specialist_clone_mode_switch_boost_merge`: mean avg delta -1.0%, current delta -2.0%, prior delta -0.7%, forgetting delta -5.3%, gain vs fixed +6.0%, wins 0/1
- `task_specialist_clone_routed_merge`: mean avg delta +7.4%, current delta +29.2%, prior delta +0.1%, forgetting delta -6.3%, gain vs fixed -4.9%, wins 0/1
- `task_specialist_clone_source_aware_merge`: mean avg delta +6.4%, current delta +21.7%, prior delta +1.3%, forgetting delta -6.0%, gain vs fixed -5.9%, wins 0/1

## Per Benchmark

### plasticity_stress

- Task suite: `digits_pairs_noisy`
- Intent: stress current-task learning under noisy sequential digits
- Best policy: `task_specialist_clone_current_path_boost_merge`
- Best family: `plasticity_boost`

- `adapt_only`: avg delta +0.0%, current delta +0.0%, prior delta +0.0%, forgetting delta +0.0%, gain vs fixed -23.1%
- `task_specialist_clone_limited_merge`: avg delta +4.7%, current delta +6.7%, prior delta +4.1%, forgetting delta -2.3%, gain vs fixed -18.3%
- `task_specialist_clone_distill_merge`: avg delta +4.5%, current delta +6.5%, prior delta +3.9%, forgetting delta -2.5%, gain vs fixed -18.5%
- `task_specialist_clone_current_path_boost_merge`: avg delta +13.6%, current delta +24.0%, prior delta +10.1%, forgetting delta -9.7%, gain vs fixed -9.5%

### interference_stress

- Task suite: `feature_shift`
- Intent: separate raw specialist interference from low-merge control
- Best policy: `task_specialist_clone_limited_merge`
- Best family: `specialist_reuse`

- `adapt_only`: avg delta +0.0%, current delta +0.0%, prior delta +0.0%, forgetting delta +0.0%, gain vs fixed +12.8%
- `task_specialist_clone`: avg delta -2.3%, current delta +2.3%, prior delta -3.9%, forgetting delta -3.8%, gain vs fixed +10.5%
- `task_specialist_clone_limited_merge`: avg delta -0.1%, current delta +0.7%, prior delta -0.3%, forgetting delta -0.6%, gain vs fixed +12.8%
- `task_specialist_clone_current_path_boost_merge`: avg delta -0.9%, current delta +3.0%, prior delta -2.2%, forgetting delta -3.6%, gain vs fixed +11.9%

### interference_stress_strong

- Task suite: `feature_shift_noisy`
- Intent: stress interference control under stronger feature-shift noise
- Best policy: `task_specialist_clone_limited_merge`
- Best family: `specialist_reuse`

- `adapt_only`: avg delta +0.0%, current delta +0.0%, prior delta +0.0%, forgetting delta +0.0%, gain vs fixed +7.0%
- `task_specialist_clone_limited_merge`: avg delta +0.7%, current delta -2.0%, prior delta +1.6%, forgetting delta -1.0%, gain vs fixed +7.7%
- `task_specialist_clone_current_path_boost_merge`: avg delta -1.5%, current delta -1.7%, prior delta -1.4%, forgetting delta -6.3%, gain vs fixed +5.5%
- `task_specialist_clone_mode_switch_boost_merge`: avg delta -1.0%, current delta -2.0%, prior delta -0.7%, forgetting delta -5.3%, gain vs fixed +6.0%
- `task_specialist_clone_dual_mode_update_merge`: avg delta -1.0%, current delta -2.0%, prior delta -0.7%, forgetting delta -5.3%, gain vs fixed +6.0%

### retrieval_ambiguity_stress

- Task suite: `digits_pairs_permuted`
- Intent: test whether routing/retrieval helps under ambiguous digit remapping
- Best policy: `task_specialist_clone_current_path_boost_merge`
- Best family: `plasticity_boost`

- `adapt_only`: avg delta +0.0%, current delta +0.0%, prior delta +0.0%, forgetting delta +0.0%, gain vs fixed -12.3%
- `task_specialist_clone_limited_merge`: avg delta +6.4%, current delta +22.3%, prior delta +1.0%, forgetting delta -6.3%, gain vs fixed -6.0%
- `task_specialist_clone_source_aware_merge`: avg delta +6.4%, current delta +21.7%, prior delta +1.3%, forgetting delta -6.0%, gain vs fixed -5.9%
- `task_specialist_clone_routed_merge`: avg delta +7.4%, current delta +29.2%, prior delta +0.1%, forgetting delta -6.3%, gain vs fixed -4.9%
- `task_specialist_clone_current_path_boost_merge`: avg delta +15.8%, current delta +61.0%, prior delta +0.7%, forgetting delta -15.7%, gain vs fixed +3.4%
