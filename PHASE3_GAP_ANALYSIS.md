# Phase 3 Gap Analysis

- Baseline: `adapt_only`
- Leading true-reuse policy: `task_specialist_clone_distill_merge` (+1.1% vs adapt_only, forgetting delta +0.7%)
- Lowest-forgetting true-reuse policy: `task_specialist_clone_limited_merge` (mean forgetting 1.3%, avg delta +1.0%)

## best_clone_limited_adapt

- Avg accuracy delta vs adapt_only: -0.6%
- Current-task delta vs adapt_only: -1.2%
- Prior-task delta vs adapt_only: -0.4%
- Forgetting delta vs adapt_only: -0.9%
- Unit delta vs adapt_only: +11.0
- Classification: below_adapt_only, current_task_gap_is_larger_than_retention_gap, forgets_more_than_adapt_only, uses_more_units_than_adapt_only

## benefit_limited_clone

- Avg accuracy delta vs adapt_only: -0.6%
- Current-task delta vs adapt_only: -1.2%
- Prior-task delta vs adapt_only: -0.4%
- Forgetting delta vs adapt_only: -0.9%
- Unit delta vs adapt_only: +11.0
- Classification: below_adapt_only, current_task_gap_is_larger_than_retention_gap, forgets_more_than_adapt_only, uses_more_units_than_adapt_only

## best_clone_delayed_limited_adapt

- Avg accuracy delta vs adapt_only: -0.4%
- Current-task delta vs adapt_only: +2.2%
- Prior-task delta vs adapt_only: -1.3%
- Forgetting delta vs adapt_only: -2.1%
- Unit delta vs adapt_only: +11.0
- Classification: below_adapt_only, retention_gap_is_larger_or_equal, forgets_more_than_adapt_only, uses_more_units_than_adapt_only

## capped_delayed_limited_clone

- Avg accuracy delta vs adapt_only: -1.4%
- Current-task delta vs adapt_only: -2.2%
- Prior-task delta vs adapt_only: -1.1%
- Forgetting delta vs adapt_only: -0.6%
- Unit delta vs adapt_only: +3.0
- Classification: below_adapt_only, current_task_gap_is_larger_than_retention_gap, forgets_more_than_adapt_only, uses_more_units_than_adapt_only

## task_specialist_clone

- Avg accuracy delta vs adapt_only: +0.5%
- Current-task delta vs adapt_only: +1.0%
- Prior-task delta vs adapt_only: +0.4%
- Forgetting delta vs adapt_only: -1.5%
- Unit delta vs adapt_only: +11.0
- Classification: matches_or_exceeds_adapt_only, retention_gap_is_larger_or_equal, forgets_more_than_adapt_only, uses_more_units_than_adapt_only

## task_specialist_clone_limited_merge

- Avg accuracy delta vs adapt_only: +1.0%
- Current-task delta vs adapt_only: -1.0%
- Prior-task delta vs adapt_only: +1.7%
- Forgetting delta vs adapt_only: +0.8%
- Unit delta vs adapt_only: +11.0
- Classification: matches_or_exceeds_adapt_only, current_task_gap_is_larger_than_retention_gap, forgets_less_or_equal_than_adapt_only, uses_more_units_than_adapt_only

## task_specialist_clone_confidence_gate_merge

- Avg accuracy delta vs adapt_only: +1.0%
- Current-task delta vs adapt_only: -1.2%
- Prior-task delta vs adapt_only: +1.7%
- Forgetting delta vs adapt_only: +0.7%
- Unit delta vs adapt_only: +11.0
- Classification: matches_or_exceeds_adapt_only, current_task_gap_is_larger_than_retention_gap, forgets_less_or_equal_than_adapt_only, uses_more_units_than_adapt_only

## task_specialist_clone_routed_merge

- Avg accuracy delta vs adapt_only: +0.3%
- Current-task delta vs adapt_only: -1.0%
- Prior-task delta vs adapt_only: +0.8%
- Forgetting delta vs adapt_only: -1.1%
- Unit delta vs adapt_only: +11.0
- Classification: matches_or_exceeds_adapt_only, current_task_gap_is_larger_than_retention_gap, forgets_more_than_adapt_only, uses_more_units_than_adapt_only

## task_specialist_clone_soft_route_merge

- Avg accuracy delta vs adapt_only: +0.7%
- Current-task delta vs adapt_only: +0.4%
- Prior-task delta vs adapt_only: +0.8%
- Forgetting delta vs adapt_only: -0.7%
- Unit delta vs adapt_only: +11.0
- Classification: matches_or_exceeds_adapt_only, current_task_gap_is_larger_than_retention_gap, forgets_more_than_adapt_only, uses_more_units_than_adapt_only

## task_specialist_clone_budgeted_merge

- Avg accuracy delta vs adapt_only: +0.5%
- Current-task delta vs adapt_only: +0.6%
- Prior-task delta vs adapt_only: +0.5%
- Forgetting delta vs adapt_only: -1.0%
- Unit delta vs adapt_only: +11.0
- Classification: matches_or_exceeds_adapt_only, retention_gap_is_larger_or_equal, forgets_more_than_adapt_only, uses_more_units_than_adapt_only

## task_specialist_clone_anchor_merge

- Avg accuracy delta vs adapt_only: +1.0%
- Current-task delta vs adapt_only: -1.2%
- Prior-task delta vs adapt_only: +1.7%
- Forgetting delta vs adapt_only: +0.7%
- Unit delta vs adapt_only: +11.0
- Classification: matches_or_exceeds_adapt_only, current_task_gap_is_larger_than_retention_gap, forgets_less_or_equal_than_adapt_only, uses_more_units_than_adapt_only

## task_specialist_clone_distance_anchor_merge

- Avg accuracy delta vs adapt_only: +1.0%
- Current-task delta vs adapt_only: -1.2%
- Prior-task delta vs adapt_only: +1.7%
- Forgetting delta vs adapt_only: +0.7%
- Unit delta vs adapt_only: +11.0
- Classification: matches_or_exceeds_adapt_only, current_task_gap_is_larger_than_retention_gap, forgets_less_or_equal_than_adapt_only, uses_more_units_than_adapt_only

## task_specialist_clone_distill_merge

- Avg accuracy delta vs adapt_only: +1.1%
- Current-task delta vs adapt_only: -0.4%
- Prior-task delta vs adapt_only: +1.6%
- Forgetting delta vs adapt_only: +0.7%
- Unit delta vs adapt_only: +11.0
- Classification: matches_or_exceeds_adapt_only, current_task_gap_is_larger_than_retention_gap, forgets_less_or_equal_than_adapt_only, uses_more_units_than_adapt_only

## task_specialist_clone_freeze_source

- Avg accuracy delta vs adapt_only: -1.4%
- Current-task delta vs adapt_only: +2.6%
- Prior-task delta vs adapt_only: -2.8%
- Forgetting delta vs adapt_only: -4.1%
- Unit delta vs adapt_only: +11.0
- Classification: below_adapt_only, retention_gap_is_larger_or_equal, forgets_more_than_adapt_only, uses_more_units_than_adapt_only

