# Phase 3 Guarded Expert Simplification Benchmark

Checks whether the guarded-expert default-region quadratic learner is actually needed, or whether a hardcoded default fallback is enough once the sparse and embedded branches are preserved.

- Train family: `selector_full_map`
- Eval family: `selector_full_map`
- Overall best policy: `task_specialist_clone_two_stage_selector_merge`

## Aggregate

- `adapt_only`: mean avg delta +0.0%, current delta +0.0%, prior delta +0.0%, forgetting delta +0.0%, gain vs fixed +0.9%, wins 0
- `task_specialist_clone_signature_selector_merge`: mean avg delta +3.9%, current delta +11.8%, prior delta +1.2%, forgetting delta -5.9%, gain vs fixed +4.8%, wins 4
- `task_specialist_clone_task_ranking_selector_merge`: mean avg delta +4.2%, current delta +11.5%, prior delta +1.7%, forgetting delta -4.9%, gain vs fixed +5.1%, wins 3
- `task_specialist_clone_two_stage_selector_merge`: mean avg delta +4.4%, current delta +11.5%, prior delta +2.0%, forgetting delta -4.7%, gain vs fixed +5.3%, wins 1
- `task_specialist_clone_guarded_expert_selector_merge`: mean avg delta +4.3%, current delta +11.5%, prior delta +1.9%, forgetting delta -4.7%, gain vs fixed +5.3%, wins 0
- `task_specialist_clone_guarded_expert_boost_default_merge`: mean avg delta +3.9%, current delta +12.0%, prior delta +1.2%, forgetting delta -5.9%, gain vs fixed +4.8%, wins 0
- `task_specialist_clone_guarded_expert_merge_default_merge`: mean avg delta +2.0%, current delta +4.5%, prior delta +1.2%, forgetting delta -2.6%, gain vs fixed +2.9%, wins 0
