# Phase 3 Selector Generalization Benchmark

Generalization check on an interference-style benchmark with nonnegative, digits-like inputs.

- Task suite: `feature_shift_nonnegative`
- Best policy: `task_specialist_clone_current_path_boost_merge`
- Scalar selector thresholds: conflict `0.30`, confidence `0.22`
- Fingerprint thresholds: nonnegative ratio `0.85`, abs-mean `0.80`

## Policies

- `adapt_only`: avg delta +0.0%, current delta +0.0%, prior delta +0.0%, forgetting delta +0.0%, gain vs fixed +10.7%
- `task_specialist_clone_limited_merge`: avg delta +1.4%, current delta -1.3%, prior delta +2.3%, forgetting delta +2.0%, gain vs fixed +12.2%
- `task_specialist_clone_current_path_boost_merge`: avg delta +1.5%, current delta -1.0%, prior delta +2.3%, forgetting delta +2.0%, gain vs fixed +12.3%
- `task_specialist_clone_regime_switch_merge`: avg delta +1.4%, current delta -1.3%, prior delta +2.3%, forgetting delta +2.0%, gain vs fixed +12.2%
- `task_specialist_clone_diagnostic_selector_merge`: avg delta +1.5%, current delta -1.0%, prior delta +2.3%, forgetting delta +2.0%, gain vs fixed +12.3%
- `task_specialist_clone_fingerprint_selector_merge`: avg delta +1.5%, current delta -1.0%, prior delta +2.3%, forgetting delta +2.0%, gain vs fixed +12.3%
