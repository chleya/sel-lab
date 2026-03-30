# Phase 3 Hierarchical Quadratic Selector Benchmark

Hierarchical segmented selector with region-specific quadratic ranking heads over `default`, `embedded`, and `sparse_embedded` regions.

- Train family: `selector_full_map`
- Eval family: `selector_full_map`
- Overall best policy: `task_specialist_clone_two_stage_selector_merge`
- Quadratic feature count: `65`

## Region Heads

- `default`: samples=60, mean utility=-0.038, bias=-0.229
- `embedded`: samples=30, mean utility=-0.008, bias=+7.273
- `sparse_embedded`: samples=6, mean utility=-0.048, bias=+62.160

## Aggregate

- `adapt_only`: mean avg delta +0.0%, current delta +0.0%, prior delta +0.0%, forgetting delta +0.0%, gain vs fixed +0.9%, wins 0
- `task_specialist_clone_limited_merge`: mean avg delta +1.7%, current delta +3.7%, prior delta +1.0%, forgetting delta -2.1%, gain vs fixed +2.7%, wins 0
- `task_specialist_clone_current_path_boost_merge`: mean avg delta +3.6%, current delta +13.4%, prior delta +0.4%, forgetting delta -6.8%, gain vs fixed +4.6%, wins 3
- `task_specialist_clone_regime_switch_merge`: mean avg delta +4.0%, current delta +10.7%, prior delta +1.7%, forgetting delta -4.2%, gain vs fixed +4.9%, wins 0
- `task_specialist_clone_signature_selector_merge`: mean avg delta +3.9%, current delta +11.8%, prior delta +1.2%, forgetting delta -5.9%, gain vs fixed +4.8%, wins 1
- `task_specialist_clone_task_ranking_selector_merge`: mean avg delta +4.2%, current delta +11.5%, prior delta +1.7%, forgetting delta -4.9%, gain vs fixed +5.1%, wins 3
- `task_specialist_clone_two_stage_selector_merge`: mean avg delta +4.4%, current delta +11.5%, prior delta +2.0%, forgetting delta -4.7%, gain vs fixed +5.3%, wins 1
- `task_specialist_clone_hierarchical_selector_merge`: mean avg delta +1.7%, current delta +3.7%, prior delta +1.0%, forgetting delta -2.1%, gain vs fixed +2.7%, wins 0
- `task_specialist_clone_hierarchical_quadratic_selector_merge`: mean avg delta +4.1%, current delta +11.3%, prior delta +1.7%, forgetting delta -5.0%, gain vs fixed +5.0%, wins 0
