# Phase 3 Digits Gap Decomposition

- Digits suites: `digits_pairs, digits_pairs_noisy, digits_pairs_permuted`
- Best policy by mean avg gap vs fixed: `task_specialist_clone_current_path_boost_merge`

## Aggregate Gap Summary

- `adapt_only`: avg gap -17.3%, current gap -40.9%, prior gap -9.5%, forgetting delta -29.0%, classification `current_task_gap_dominates`
- `task_specialist_clone_limited_merge`: avg gap -12.3%, current gap -27.8%, prior gap -7.1%, forgetting delta -25.2%, classification `current_task_gap_dominates`
- `task_specialist_clone_distill_merge`: avg gap -12.4%, current gap -28.1%, prior gap -7.2%, forgetting delta -25.2%, classification `current_task_gap_dominates`
- `task_specialist_clone_role_separation_merge`: avg gap -12.4%, current gap -27.8%, prior gap -7.3%, forgetting delta -25.1%, classification `current_task_gap_dominates`
- `task_specialist_clone_source_aware_merge`: avg gap -12.4%, current gap -28.0%, prior gap -7.2%, forgetting delta -25.2%, classification `current_task_gap_dominates`
- `task_specialist_clone_asymmetric_retention_merge`: avg gap -12.5%, current gap -28.4%, prior gap -7.2%, forgetting delta -25.1%, classification `current_task_gap_dominates`
- `task_specialist_clone_current_path_boost_merge`: avg gap -5.4%, current gap -9.5%, prior gap -4.0%, forgetting delta -19.0%, classification `current_task_gap_dominates`

## Suite Breakdown

### digits_pairs

- `adapt_only`: avg gap -16.8%, current gap -34.0%, prior gap -11.1%, forgetting delta -28.1%, classification `current_task_gap_dominates`
- `task_specialist_clone_limited_merge`: avg gap -11.5%, current gap -22.4%, prior gap -7.9%, forgetting delta -23.5%, classification `current_task_gap_dominates`
- `task_specialist_clone_distill_merge`: avg gap -11.4%, current gap -22.3%, prior gap -7.8%, forgetting delta -23.6%, classification `current_task_gap_dominates`
- `task_specialist_clone_role_separation_merge`: avg gap -11.7%, current gap -22.4%, prior gap -8.1%, forgetting delta -23.3%, classification `current_task_gap_dominates`
- `task_specialist_clone_source_aware_merge`: avg gap -11.4%, current gap -22.3%, prior gap -7.8%, forgetting delta -23.6%, classification `current_task_gap_dominates`
- `task_specialist_clone_asymmetric_retention_merge`: avg gap -11.6%, current gap -22.4%, prior gap -8.0%, forgetting delta -23.5%, classification `current_task_gap_dominates`
- `task_specialist_clone_current_path_boost_merge`: avg gap -5.8%, current gap -10.1%, prior gap -4.4%, forgetting delta -18.0%, classification `current_task_gap_dominates`

### digits_pairs_noisy

- `adapt_only`: avg gap -21.1%, current gap -31.6%, prior gap -17.6%, forgetting delta -25.5%, classification `current_task_gap_dominates`
- `task_specialist_clone_limited_merge`: avg gap -17.6%, current gap -24.4%, prior gap -15.4%, forgetting delta -22.7%, classification `current_task_gap_dominates`
- `task_specialist_clone_distill_merge`: avg gap -17.7%, current gap -24.2%, prior gap -15.5%, forgetting delta -22.5%, classification `current_task_gap_dominates`
- `task_specialist_clone_role_separation_merge`: avg gap -17.8%, current gap -24.4%, prior gap -15.5%, forgetting delta -22.5%, classification `current_task_gap_dominates`
- `task_specialist_clone_source_aware_merge`: avg gap -17.7%, current gap -24.2%, prior gap -15.5%, forgetting delta -22.4%, classification `current_task_gap_dominates`
- `task_specialist_clone_asymmetric_retention_merge`: avg gap -17.7%, current gap -24.4%, prior gap -15.4%, forgetting delta -22.5%, classification `current_task_gap_dominates`
- `task_specialist_clone_current_path_boost_merge`: avg gap -10.8%, current gap -11.5%, prior gap -10.6%, forgetting delta -18.5%, classification `current_task_gap_dominates`

### digits_pairs_permuted

- `adapt_only`: avg gap -14.1%, current gap -57.0%, prior gap +0.2%, forgetting delta -33.4%, classification `current_task_gap_dominates`
- `task_specialist_clone_limited_merge`: avg gap -7.7%, current gap -36.5%, prior gap +1.9%, forgetting delta -29.3%, classification `current_task_gap_dominates`
- `task_specialist_clone_distill_merge`: avg gap -8.1%, current gap -37.7%, prior gap +1.8%, forgetting delta -29.5%, classification `current_task_gap_dominates`
- `task_specialist_clone_role_separation_merge`: avg gap -7.8%, current gap -36.7%, prior gap +1.9%, forgetting delta -29.4%, classification `current_task_gap_dominates`
- `task_specialist_clone_source_aware_merge`: avg gap -8.0%, current gap -37.6%, prior gap +1.9%, forgetting delta -29.5%, classification `current_task_gap_dominates`
- `task_specialist_clone_asymmetric_retention_merge`: avg gap -8.3%, current gap -38.4%, prior gap +1.8%, forgetting delta -29.5%, classification `current_task_gap_dominates`
- `task_specialist_clone_current_path_boost_merge`: avg gap +0.5%, current gap -6.9%, prior gap +2.9%, forgetting delta -20.3%, classification `current_task_gap_dominates`
