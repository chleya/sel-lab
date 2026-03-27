# Phase 3 Digits Transfer

- Benchmark: sequential digits-pair binary tasks `(0,1) -> (2,3) -> (4,5) -> (6,7)`
- Tuned specialist merge scale: `0.02`
- Tuned distill strength: `0.50`
- Best reuse policy in this comparison: `task_specialist_clone_distill_merge`
- Tied best reuse policies at reported precision: `task_specialist_clone_distill_merge, task_specialist_clone_confidence_gate_merge`
- Best avg delta vs adapt_only: +5.4%
- Best forgetting delta vs adapt_only: -4.5%
- Best current-task delta vs adapt_only: +11.8%
- Best prior-task delta vs adapt_only: +3.3%
- Best unit delta vs adapt_only: +10.4

## Candidate Comparison

- `task_specialist_clone_limited_merge`: avg +5.3%, forgetting -4.5%, current +11.6%, prior +3.2%
- `task_specialist_clone_confidence_gate_merge`: avg +5.4%, forgetting -4.5%, current +11.8%, prior +3.3%
- `task_specialist_clone_distill_merge`: avg +5.4%, forgetting -4.5%, current +11.8%, prior +3.3%
