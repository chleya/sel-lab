# Phase 3 Dual-Mode Update Diagnostic

Conflict-aware dual-mode update: downshift current-path boost and switch archived specialists to retention-only when conflict rises.

- Task suites: `feature_shift_noisy, digits_pairs_noisy, digits_pairs_permuted`
- Overall best policy: `task_specialist_clone_current_path_boost_merge`
- Config: threshold `0.35`, min boost `1.10`, retention `0.12`

## Aggregate

- `adapt_only`: avg delta +0.0%, current delta +0.0%, prior delta +0.0%, forgetting delta +0.0%, gain vs fixed -9.5%, suite wins 0
- `task_specialist_clone_limited_merge`: avg delta +3.9%, current delta +9.0%, prior delta +2.2%, forgetting delta -3.2%, gain vs fixed -5.6%, suite wins 1
- `task_specialist_clone_current_path_boost_merge`: avg delta +9.3%, current delta +27.8%, prior delta +3.1%, forgetting delta -10.6%, gain vs fixed -0.2%, suite wins 2
- `task_specialist_clone_dual_mode_update_merge`: avg delta +6.9%, current delta +15.5%, prior delta +4.0%, forgetting delta -5.4%, gain vs fixed -2.6%, suite wins 0

## Per Suite

### feature_shift_noisy

- Best policy: `task_specialist_clone_limited_merge`
- `adapt_only`: avg delta +0.0%, current delta +0.0%, prior delta +0.0%, forgetting delta +0.0%, gain vs fixed +7.0%
- `task_specialist_clone_limited_merge`: avg delta +0.7%, current delta -2.0%, prior delta +1.6%, forgetting delta -1.0%, gain vs fixed +7.7%
- `task_specialist_clone_current_path_boost_merge`: avg delta -1.5%, current delta -1.7%, prior delta -1.4%, forgetting delta -6.3%, gain vs fixed +5.5%
- `task_specialist_clone_dual_mode_update_merge`: avg delta -1.0%, current delta -2.0%, prior delta -0.7%, forgetting delta -5.3%, gain vs fixed +6.0%

### digits_pairs_noisy

- Best policy: `task_specialist_clone_current_path_boost_merge`
- `adapt_only`: avg delta +0.0%, current delta +0.0%, prior delta +0.0%, forgetting delta +0.0%, gain vs fixed -23.1%
- `task_specialist_clone_limited_merge`: avg delta +4.7%, current delta +6.7%, prior delta +4.1%, forgetting delta -2.3%, gain vs fixed -18.3%
- `task_specialist_clone_current_path_boost_merge`: avg delta +13.6%, current delta +24.0%, prior delta +10.1%, forgetting delta -9.7%, gain vs fixed -9.5%
- `task_specialist_clone_dual_mode_update_merge`: avg delta +9.7%, current delta +14.8%, prior delta +8.1%, forgetting delta -4.5%, gain vs fixed -13.3%

### digits_pairs_permuted

- Best policy: `task_specialist_clone_current_path_boost_merge`
- `adapt_only`: avg delta +0.0%, current delta +0.0%, prior delta +0.0%, forgetting delta +0.0%, gain vs fixed -12.3%
- `task_specialist_clone_limited_merge`: avg delta +6.4%, current delta +22.3%, prior delta +1.0%, forgetting delta -6.3%, gain vs fixed -6.0%
- `task_specialist_clone_current_path_boost_merge`: avg delta +15.8%, current delta +61.0%, prior delta +0.7%, forgetting delta -15.7%, gain vs fixed +3.4%
- `task_specialist_clone_dual_mode_update_merge`: avg delta +12.0%, current delta +33.8%, prior delta +4.7%, forgetting delta -6.3%, gain vs fixed -0.4%
