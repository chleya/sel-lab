# Phase 3 Diagnostic Suite

Compact diagnostic suite for separating plasticity, interference, and retrieval pressure.

## Family Wins

- `plasticity_boost`: 2
- `specialist_reuse`: 1

## Diagnostics

### plasticity_stress

- Task suite: `digits_pairs_noisy`
- Intent: stress current-task learning under noisy sequential digits
- Best policy: `task_specialist_clone_current_path_boost_merge`
- Best family: `plasticity_boost`

- `adapt_only`: avg delta vs adapt_only +0.0%, current delta +0.0%, prior delta +0.0%, forgetting delta +0.0%, gain vs fixed -22.8%
- `task_specialist_clone_limited_merge`: avg delta vs adapt_only +2.6%, current delta +2.7%, prior delta +2.5%, forgetting delta +0.3%, gain vs fixed -20.2%
- `task_specialist_clone_distill_merge`: avg delta vs adapt_only +2.3%, current delta +2.1%, prior delta +2.4%, forgetting delta +0.5%, gain vs fixed -20.4%
- `task_specialist_clone_current_path_boost_merge`: avg delta vs adapt_only +13.8%, current delta +24.0%, prior delta +10.4%, forgetting delta -2.1%, gain vs fixed -9.0%

### interference_stress

- Task suite: `feature_shift`
- Intent: separate raw specialist interference from low-merge control
- Best policy: `task_specialist_clone_limited_merge`
- Best family: `specialist_reuse`

- `adapt_only`: avg delta vs adapt_only +0.0%, current delta +0.0%, prior delta +0.0%, forgetting delta +0.0%, gain vs fixed +10.8%
- `task_specialist_clone`: avg delta vs adapt_only -1.9%, current delta +0.3%, prior delta -2.7%, forgetting delta -2.9%, gain vs fixed +8.8%
- `task_specialist_clone_limited_merge`: avg delta vs adapt_only +0.3%, current delta -1.0%, prior delta +0.8%, forgetting delta +1.1%, gain vs fixed +11.1%
- `task_specialist_clone_current_path_boost_merge`: avg delta vs adapt_only -0.4%, current delta +0.0%, prior delta -0.6%, forgetting delta -1.3%, gain vs fixed +10.3%

### retrieval_ambiguity_stress

- Task suite: `digits_pairs_permuted`
- Intent: test whether routing/retrieval helps under ambiguous digit remapping
- Best policy: `task_specialist_clone_current_path_boost_merge`
- Best family: `plasticity_boost`

- `adapt_only`: avg delta vs adapt_only +0.0%, current delta +0.0%, prior delta +0.0%, forgetting delta +0.0%, gain vs fixed -11.9%
- `task_specialist_clone_limited_merge`: avg delta vs adapt_only +1.8%, current delta -1.0%, prior delta +2.8%, forgetting delta -1.4%, gain vs fixed -10.1%
- `task_specialist_clone_source_aware_merge`: avg delta vs adapt_only +1.9%, current delta -1.3%, prior delta +2.9%, forgetting delta -1.1%, gain vs fixed -10.1%
- `task_specialist_clone_routed_merge`: avg delta vs adapt_only +3.7%, current delta +11.7%, prior delta +1.0%, forgetting delta -2.8%, gain vs fixed -8.2%
- `task_specialist_clone_current_path_boost_merge`: avg delta vs adapt_only +15.9%, current delta +47.5%, prior delta +5.3%, forgetting delta -6.4%, gain vs fixed +4.0%
