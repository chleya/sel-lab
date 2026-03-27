# Phase 3 Interference Diagnostic

Focused comparison for raw specialist interference vs low-merge control vs current-path boost.

- Task suites: `feature_shift, feature_shift_noisy`
- Overall best policy: `task_specialist_clone_limited_merge`

## Aggregate

- `adapt_only`: avg delta +0.0%, current delta +0.0%, prior delta +0.0%, forgetting delta +0.0%, gain vs fixed +9.9%, suite wins 0
- `task_specialist_clone`: avg delta -1.8%, current delta +0.3%, prior delta -2.6%, forgetting delta -4.2%, gain vs fixed +8.1%, suite wins 0
- `task_specialist_clone_limited_merge`: avg delta +0.3%, current delta -0.7%, prior delta +0.6%, forgetting delta -0.8%, gain vs fixed +10.2%, suite wins 2
- `task_specialist_clone_current_path_boost_merge`: avg delta -1.2%, current delta +0.7%, prior delta -1.8%, forgetting delta -4.9%, gain vs fixed +8.7%, suite wins 0

## Per Suite

### feature_shift

- Best policy: `task_specialist_clone_limited_merge`
- `adapt_only`: avg delta +0.0%, current delta +0.0%, prior delta +0.0%, forgetting delta +0.0%, gain vs fixed +12.8%
- `task_specialist_clone`: avg delta -2.3%, current delta +2.3%, prior delta -3.9%, forgetting delta -3.8%, gain vs fixed +10.5%
- `task_specialist_clone_limited_merge`: avg delta -0.1%, current delta +0.7%, prior delta -0.3%, forgetting delta -0.6%, gain vs fixed +12.8%
- `task_specialist_clone_current_path_boost_merge`: avg delta -0.9%, current delta +3.0%, prior delta -2.2%, forgetting delta -3.6%, gain vs fixed +11.9%

### feature_shift_noisy

- Best policy: `task_specialist_clone_limited_merge`
- `adapt_only`: avg delta +0.0%, current delta +0.0%, prior delta +0.0%, forgetting delta +0.0%, gain vs fixed +7.0%
- `task_specialist_clone`: avg delta -1.3%, current delta -1.7%, prior delta -1.2%, forgetting delta -4.7%, gain vs fixed +5.7%
- `task_specialist_clone_limited_merge`: avg delta +0.7%, current delta -2.0%, prior delta +1.6%, forgetting delta -1.0%, gain vs fixed +7.7%
- `task_specialist_clone_current_path_boost_merge`: avg delta -1.5%, current delta -1.7%, prior delta -1.4%, forgetting delta -6.3%, gain vs fixed +5.5%
