# SEL-Lab Handoff

## Current Project Position

SEL-Lab is currently a continual-learning research codebase centered on one narrow question:

When does forward-only structure reuse help, and which mechanism actually drives that gain?

Use this priority order when interpreting the repository:

1. `Phase 3` as the main result line
2. `Phase 4` as the validation line
3. `Phase 2` as the mechanism-screening bench
4. exploratory utilities as side work only

## Source Of Truth

Use files in this order:

1. `UNIFIED_REPORT.md` for canonical scientific read
2. `README.md` and `PROJECT_REFRAME.md` for project framing
3. `AUTONOMY_PROTOCOL.md` for the autonomous work loop
4. `HANDOFF.md` for current execution state only
5. `memory/YYYY-MM-DD.md` for chronological lab notes

`HANDOFF.md` is intentionally not a historical log. Old narratives belong in `memory/` or archived reports.

## Fast Start

If you are the next agent, do this first:

1. Read `UNIFIED_REPORT.md` once for the stable project narrative.
2. Read this file only through:
   - `Current Leading Mechanism`
   - `Selector State`
   - `Immediate Next Step`
   - `Do Not Repeat`
3. Read `memory/2026-03-27.md` for the latest selector-line work.
4. Run `python tests\smoke_test.py` before doing new edits.

If you only need the current main read:

- broader-digits leader: `task_specialist_clone_current_path_boost_merge`
- formal stress-map leader: `task_specialist_clone_regime_switch_merge`
- best less-hardcoded selector on the formal family: `task_specialist_clone_fingerprint_selector_merge`
- best non-hardcoded selector on the embedded adversarial gate: `task_specialist_clone_signature_selector_merge`
- current best learned aggregate selector on `selector_full_map`: `task_specialist_clone_two_stage_selector_merge`
- current selector bottleneck: no longer just objective structure or local threshold choice; the remaining bottleneck is routing target / segmented selector form, because neither a single global score surface nor a flat learned multiclass router can jointly satisfy `feature_shift_embedded` and `feature_shift_sparse_embedded`

## Selector State

The selector line has already ruled out several easy stories.

- `task_specialist_clone_diagnostic_selector_merge`
  - too weak; collapses to boost-side behavior
- `task_specialist_clone_fingerprint_selector_merge`
  - recovers the formal family
  - still fails the embedded adversarial gate
- `task_specialist_clone_signature_selector_merge`
  - best current non-hardcoded selector on `feature_shift_embedded`
  - still not the best aggregate selector
- `task_specialist_clone_fitted_selector_merge`
  - linear fit over static signature features is not enough
- `task_specialist_clone_prototype_selector_merge`
  - exemplar retrieval is not enough
- `task_specialist_clone_dynamics_selector_merge`
  - richer online dynamics features still mostly reproduce the fingerprint frontier
- `task_specialist_clone_outcome_selector_merge`
  - outcome-driven supervision helps on the main embedded gate
  - but degrades the broader full-map aggregate and fails the sparse adversarial gate
- `task_specialist_clone_ranking_selector_merge`
  - suite-level pairwise/ranking supervision is better aligned than plain outcome regression
  - but still does not solve the embedded gate
- `task_specialist_clone_task_ranking_selector_merge`
  - task-level structured ranking becomes the strongest single-head learned aggregate selector
  - but it degrades `feature_shift_sparse_embedded`
- `task_specialist_clone_constrained_task_ranking_selector_merge`
  - adding sparse-gate-aware weighting or constraints does not rescue the single-head ranking frontier
- `task_specialist_clone_two_stage_selector_merge`
  - current best learned selector on `selector_full_map`
  - matches `signature_selector` on `feature_shift_embedded`
  - improves `feature_shift_sparse_embedded` to `-0.8%` vs fixed
  - improves aggregate to `+5.3%` vs fixed, above `task_ranking_selector` / `fingerprint_selector` / `signature_selector`
- `task_specialist_clone_learned_router_selector_merge`
  - flat learned three-way router over the same regime features does not beat the current two-stage split
  - aggregate falls back to `+4.8%` vs fixed
  - gate performance regresses to `-0.8%` on `feature_shift_embedded` and `-1.5%` on `feature_shift_sparse_embedded`
- `task_specialist_clone_hierarchical_selector_merge`
  - keeps segmented routing but replaces the shared fallback with region-specific ranking heads
  - aggregate reaches about `+5.1%` vs fixed in the dedicated benchmark, but remains below `task_specialist_clone_two_stage_selector_merge`
  - gate performance still regresses relative to `two_stage_selector`:
    - `feature_shift_embedded`: `-0.2%` vs fixed
    - `feature_shift_sparse_embedded`: `-1.6%` vs fixed
- `task_specialist_clone_hierarchical_quadratic_selector_merge`
  - keeps the same segmentation but upgrades each region head to a quadratic ranking surface
  - aggregate reaches about `+5.0%` vs fixed on `selector_full_map`
  - this improves over the weaker linear hierarchical variant but still stays below `task_specialist_clone_two_stage_selector_merge`
  - gate performance still regresses relative to `two_stage_selector`:
    - `feature_shift_embedded`: `-0.5%` vs fixed
    - `feature_shift_sparse_embedded`: `-2.0%` vs fixed
- `task_specialist_clone_hierarchical_sparse_gate_selector_merge`
  - keeps the same segmented quadratic form but adds explicit sparse-gate-aware supervision weighting
  - aggregate still lands around `+5.0%` vs fixed on `selector_full_map`
  - this does not recover the `two_stage_selector` frontier
  - gate performance still regresses relative to `two_stage_selector`:
    - `feature_shift_embedded`: `-0.5%` vs fixed
    - `feature_shift_sparse_embedded`: `-2.0%` vs fixed
- `task_specialist_clone_guarded_router_selector_merge`
  - preserves the hard sparse branch from `two_stage_selector` and learns only the remaining segmented heads
  - aggregate still lands around `+5.0%` vs fixed on `selector_full_map`
  - this narrows the gap relative to the sparse-weighted segmented fit, but still does not recover the `two_stage_selector` frontier
  - gate performance still regresses relative to `two_stage_selector`:
    - `feature_shift_embedded`: `-0.2%` vs fixed
    - `feature_shift_sparse_embedded`: `-1.2%` vs fixed
- focused two-stage threshold sweep
  - local calibration around `selector_sparse_zero_ratio_floor=0.385` and `selector_sparse_conflict_delta_floor=-0.005` did not find a better nearby point
  - current default remains the best local threshold pair among the tested neighbors

Current selector conclusion:

- formal-family recovery is solved
- structured ranking objectives were necessary but not sufficient
- adversarial-gate robustness is now partially solved only by segmented routing
- the current learned-selector frontier is a two-stage router, not a single learned score surface
- local threshold tuning around the current two-stage sparse route does not reveal an easy free win
- a region-specific linear-head variant has now been tested and does not beat the current two-stage frontier
- a region-specific quadratic-head variant has also now been tested and still does not beat the current two-stage frontier
- an explicit sparse-gate-aware segmented objective has also now been tested and still does not beat the current two-stage frontier
- even preserving the hard sparse branch while learning the rest still does not beat the current two-stage frontier
- the next credible upgrade is no longer just "add segmented heads", "make the same segmented heads more nonlinear", "reweight the same segmented objective", or "preserve only the sparse branch"; it is better routing targets or a closer recovery of the exact hand-designed embedded branch logic inside a more expressive router

## Immediate Next Step

The next agent should not go back to another single-head selector over the same regime features.

Best next experiment:

- keep the task-level ranking signal
- keep `selector_full_map` as the screening family
- upgrade the selector form beyond one global boundary

Concretely:

1. Start from `task_specialist_clone_two_stage_selector_merge`, not from `outcome_selector` or the older fitted selectors.
2. Treat the current result as evidence for segmented routing:
   - non-sparse embedded-like tasks need the `signature_selector`-style interference-control branch
   - sparse embedded-like tasks need the boost-side branch
   - the remaining tasks benefit from the task-ranking fallback
3. Preferred next direction:
   - improve the routing target itself, not just the threshold pair
   - preferred concrete form:
     routing targets that explicitly preserve both the sparse branch and the embedded branch logic as separate decision objects
   - or replace the hard threshold split with a more expressive segmented router than flat, per-region linear, or per-region quadratic fits
   - or fit a mixture-of-experts / Pareto-style selector that preserves the sparse-gate branch while keeping the task-ranking aggregate
4. Treat a candidate as interesting only if it beats the current two-stage frontier on at least one axis without breaking the others:
   - `feature_shift_embedded` must stay at least at `+0.0%` vs fixed
   - `feature_shift_sparse_embedded` should improve beyond `-0.8%` vs fixed
   - aggregate on `selector_full_map` should stay near or above `+5.3%` vs fixed

## Do Not Repeat

Do not spend another loop on:

- local hybrid policy tweaks like more `mode_switch` / `dual_mode` variants
- another boolean handcrafted selector
- another plain linear fit over nearly the same feature vector
- another single-head ranking or constrained-ranking fit over the same regime vector
- another flat learned multiclass router over the same regime vector without changing the routing target or selector form
- another region-specific linear-head selector behind the same current segmentation
- another region-specific quadratic-head selector behind the same current segmentation
- another sparse-weighted variant of the same current segmented quadratic objective
- another guarded-router variant that only preserves the sparse branch while relearning the rest from the same targets
- another local threshold retune of the current two-stage sparse-route pair unless a new benchmark family or new router target changes the evidence
- another prototype / nearest-neighbor selector over the same feature bank
- weak generalization suites that only perturb nonnegativity without preserving regime ambiguity

Those directions are already information-saturated in this repo.

## Files To Open

For selector work, open these first:

- `core/phase3_model.py`
- `core/phase3_policies.py`
- `core/phase3_registry.py`
- `analysis/phase3_benchmark_family.py`
- `analysis/phase3_dynamics_selector_benchmark.py`
- `analysis/phase3_outcome_selector_benchmark.py`
- `analysis/phase3_task_ranking_selector_benchmark.py`
- `analysis/phase3_two_stage_selector_benchmark.py`
- `analysis/phase3_learned_router_selector_benchmark.py`
- `analysis/phase3_two_stage_selector_sweep.py`
- `results/canonical/phase3_dynamics_selector_benchmark.json`
- `results/canonical/phase3_outcome_selector_benchmark.json`
- `results/canonical/phase3_task_ranking_selector_benchmark.json`
- `results/canonical/phase3_two_stage_selector_benchmark.json`
- `results/canonical/phase3_learned_router_selector_benchmark.json`
- `results/canonical/phase3_two_stage_selector_sweep.json`

## Cheap Verification

Use these before any heavy rerun:

- `python tests\smoke_test.py`
- `python -c "from core.phase3_registry import resolve_phase3_diagnostic_family; print(resolve_phase3_diagnostic_family('selector_full_map'))"`

Heavy selector reruns that are worth it:

- `python analysis\phase3_dynamics_selector_benchmark.py`
- `python analysis\phase3_outcome_selector_benchmark.py`
- `python analysis\phase3_task_ranking_selector_benchmark.py`
- `python analysis\phase3_two_stage_selector_benchmark.py`
- `python analysis\phase3_learned_router_selector_benchmark.py`
- `python analysis\phase3_two_stage_selector_sweep.py`

## Current Leading Mechanism

- Leading single-policy broader-digits mechanism:
  `task_specialist_clone_current_path_boost_merge`
- Leading formal stress-map mechanism:
  `task_specialist_clone_regime_switch_merge`
- Leading less-hardcoded formal stress-map selector:
  `task_specialist_clone_fingerprint_selector_merge`
- Current tuned boundary default:
  - merge scale `0.02`
  - distill strength `0.50`
  - current-path boost `2.50`
- Canonical read:
  current-task plasticity, not retention alone, is the main bottleneck on broader digits, but the best current control strategy is now an explicit regime switch between plasticity-first and interference-control behavior.

## Current Boundary Benchmark

- Primary boundary family:
  - `digits_pairs`
  - `digits_pairs_noisy`
  - `digits_pairs_permuted`
- Strongest current aggregate result on that family:
  - `task_specialist_clone_current_path_boost_merge`
  - mean `+13.2%` vs `adapt_only`
  - mean current-task delta `+34.0%`
  - mean `-4.1%` vs fixed
  - suite wins `3/3`
- Hardest suite remains:
  `digits_pairs_noisy`

## Most Important Recent Results

Positive:

- Gap decomposition established that the main remaining deficit vs fixed is `current-task gap`, not `prior-task gap`.
- `task_specialist_clone_current_path_boost_merge` replaced `limited_merge` as the leading broader-digits mechanism.
- Local boost sweeps showed the effect is not a single-point accident; best tested point moved to `boost=2.50`.
- Archived-specialist quantization stress test was unexpectedly positive:
  - `4-bit` archived specialists improved the current mainline from mean `-5.4%` vs fixed to mean `-4.0%`
  - `2-bit` archived specialists also held up, improving mean fixed gap to `-4.2%`
  - current read: archived specialists appear to function as coarse reusable memory, not precision-critical paths
- Quantized replacement check stayed negative:
  - even at `4-bit`, `task_specialist_clone_external_memory_boost_merge` still loses to internal quantized specialists
  - internal quantized current-path boost: mean `-4.0%` vs fixed
  - external quantized memory boost: mean `-6.6%` vs fixed
  - current read: compression helps archived specialists inside the main reuse structure, but does not rescue external-memory rearrangements
- Internal-memory budget check was negative:
  - under `4-bit` archived-specialist quantization, full-budget internal memory remains best
  - budget `2`: mean `+11.8%` vs `adapt_only`, mean `-5.5%` vs fixed
  - budget `1`: mean `+11.3%` vs `adapt_only`, mean `-6.1%` vs fixed
  - current read: low-bit archived memory is viable, but the mechanism still needs more than one archived specialist; archive-count compression is not free
- Archive-selection check was also negative:
  - at budget `2`, swapping recency selection for coverage selection stays effectively flat
  - `recency_2`: mean `-5.5%` vs fixed
  - `coverage_2`: mean `-5.6%` vs fixed
  - current read: the budget-compression loss is not mainly caused by a naive recency selector; the mechanism appears to need more archive capacity, not just a different simple selector
- Archive-representation check was also negative:
  - at budget `2`, replacing direct archived selection with simple prototype compression also stays effectively flat
  - `prototype_2`: mean `-5.6%` vs fixed
  - current read: the budget-compression loss is not rescued by a minimal output-space prototype representation either
- Weight-space archive compression check was also negative:
  - at budget `2`, compressing archived specialists into weight-space prototype units also stays effectively flat
  - `weight_prototype_2`: mean `-5.6%` vs fixed
  - current read: even a more structured lightweight prototype compression does not recover the archive-count loss
- Key-value retrieval check was also negative:
  - at budget `2`, sample-conditioned key-value retrieval also stays effectively flat
  - `key_value_2`: mean `-5.5%` vs fixed
  - current read: even a more structured input-conditioned memory query does not recover the archive-count loss
- Diagnostic split is now sharper than the old single-leader story:
  - `task_specialist_clone_current_path_boost_merge` still wins plasticity-heavy and retrieval-heavy digits stresses
  - `task_specialist_clone_limited_merge` still wins explicit interference-heavy stresses
  - current read: the broader-digits story is now "plasticity leader with a separate interference-control specialist baseline", not "one mechanism wins every stress family"
- Structured mode-switch check was informative but still negative as a replacement:
  - new policy: `task_specialist_clone_mode_switch_boost_merge`
  - mixed diagnostic aggregate: mean `+7.0%` vs `adapt_only`, mean `-2.4%` vs fixed
  - it partially recovers interference/forgetting pressure relative to plain `current_path_boost`, but stays behind `limited_merge` on `feature_shift_noisy` and behind plain `current_path_boost` on `digits_pairs_noisy` / `digits_pairs_permuted`
  - current read: even a more structured conflict-aware boost fallback does not unify the plasticity/interference frontier
- Explicit dual-mode update check also stayed negative as a replacement:
  - new policy: `task_specialist_clone_dual_mode_update_merge`
  - mixed diagnostic aggregate: mean `+6.9%` vs `adapt_only`, mean `-2.6%` vs fixed
  - it switches both current-path boost and archived-specialist updates under conflict, but lands almost exactly on the same frontier as `mode_switch_boost`
  - current read: switching archived specialists to retention-only under conflict still gives back too much digits plasticity and still does not beat `limited_merge` on the explicit interference stress
- Formal stress-map benchmark family is now in place:
  - canonical result: `results/canonical/phase3_diagnostic_benchmark_family.json`
  - canonical report: `PHASE3_DIAGNOSTIC_BENCHMARK_FAMILY.md`
  - current read:
    - `plasticity_boost` wins `2` stresses
    - `specialist_reuse` wins `2` stresses
    - `mode_switch_boost` wins `0`
    - `dual_mode_update` wins `0`
  - interpretation: the repo now has a cleaner screening layer, and it confirms at family level that the recent mixed candidates still do not create a new frontier
- Regime-switch oracle is now available:
  - canonical result: `results/canonical/phase3_regime_switch_oracle.json`
  - canonical report: `PHASE3_REGIME_SWITCH_ORACLE.md`
  - benchmark-conditioned oracle chooses:
    - `current_path_boost` for `plasticity_stress`
    - `limited_merge` for `interference_stress`
    - `limited_merge` for `interference_stress_strong`
    - `current_path_boost` for `retrieval_ambiguity_stress`
  - aggregate result:
    - oracle mean `+7.5%` vs `adapt_only`, mean `+3.6%` vs fixed
    - plain `current_path_boost` mean `+6.7%` vs `adapt_only`, mean `+2.8%` vs fixed
  - interpretation: higher-level regime switching is now a credible next direction; the upside is real enough that future work should target explicit regime selection rather than another local hybrid tweak
- Explicit regime-switch benchmark is now positive:
  - new policy: `task_specialist_clone_regime_switch_merge`
  - canonical result: `results/canonical/phase3_regime_switch_benchmark.json`
  - canonical report: `PHASE3_REGIME_SWITCH_BENCHMARK.md`
  - result:
    - mean `+7.5%` vs `adapt_only`
    - mean `+3.6%` vs fixed
    - wins `4/4` stresses in the formal benchmark family
  - interpretation:
    - the regime-switch direction is no longer just oracle headroom
    - the first explicit benchmark-conditioned regime policy already matches that headroom on the current stress-map family
    - the main open question is no longer "should we do regime selection?" but "how do we make regime selection less benchmark-hardcoded and more learnable/general?"
- Simple diagnostic-driven selector is now tested and negative:
  - new policy: `task_specialist_clone_diagnostic_selector_merge`
  - canonical result: `results/canonical/phase3_diagnostic_selector_benchmark.json`
  - canonical report: `PHASE3_DIAGNOSTIC_SELECTOR_BENCHMARK.md`
  - result:
    - it matches `current_path_boost` on the digits stresses
    - but it fails to recover the interference wins of the hardcoded regime switch on `feature_shift` / `feature_shift_noisy`
    - aggregate falls back to the `current_path_boost` frontier instead of the stronger regime-switch frontier
  - interpretation:
    - moving from benchmark-hardcoded switching to simple task-wise conflict/confidence thresholds is not enough
    - the next selector must use richer diagnostics or a more explicit regime signal than the current scalar stats
- Task-fingerprint-driven selector is now positive:
  - new policy: `task_specialist_clone_fingerprint_selector_merge`
  - canonical result: `results/canonical/phase3_fingerprint_selector_benchmark.json`
  - canonical report: `PHASE3_FINGERPRINT_SELECTOR_BENCHMARK.md`
  - result:
    - matches `task_specialist_clone_regime_switch_merge` on all `4/4` stresses
    - mean `+7.5%` vs `adapt_only`
    - mean `+3.6%` vs fixed
  - interpretation:
    - regime selection no longer depends on explicit `task_suite` hardcoding alone
    - richer task fingerprints are sufficient to recover the hardcoded frontier
    - the next selector question is now "how do we learn or generalize the fingerprint?", not "can any non-hardcoded selector work?"
- Adversarial selector benchmark is now negative in a useful way:
  - new suite: `feature_shift_embedded`
  - canonical result: `results/canonical/phase3_selector_adversarial_benchmark.json`
  - canonical report: `PHASE3_SELECTOR_ADVERSARIAL_BENCHMARK.md`
  - result:
    - `task_specialist_clone_regime_switch_merge` stays on the stronger interference-control frontier: `-0.6%` vs fixed
    - `task_specialist_clone_diagnostic_selector_merge` falls back to the weaker boost frontier: `-1.2%` vs fixed
    - `task_specialist_clone_fingerprint_selector_merge` also falls back to the weaker boost frontier: `-1.2%` vs fixed
  - interpretation:
    - simple task fingerprints still key too much on superficial input statistics
    - the next selector must learn causal regime structure, not just detect easy family-level fingerprints
    - current less-hardcoded selector story is now "works on the current formal family, but fails under adversarial regime ambiguity"
- Multi-signal task-signature selector is now tested and still negative as a replacement:
  - new policy: `task_specialist_clone_signature_selector_merge`
  - canonical result: `results/canonical/phase3_signature_selector_benchmark.json`
  - canonical report: `PHASE3_SIGNATURE_SELECTOR_BENCHMARK.md`
  - result:
    - it preserves the strong boost-side behavior on `digits_pairs_noisy` and `digits_pairs_permuted`
    - and on the adversarial selector suite it partially recovers the interference frontier, improving from mean `-1.2%` vs fixed for `diagnostic_selector` / `fingerprint_selector` to mean `-0.5%`
    - but on the formal stress-map family it still falls short of both `task_specialist_clone_regime_switch_merge` and `task_specialist_clone_fingerprint_selector_merge`
    - aggregate remains between the weaker diagnostic frontier and the stronger hardcoded/fingerprint frontier:
      - `signature_selector`: mean `+6.9%` vs `adapt_only`, mean `+3.0%` vs fixed
      - `fingerprint_selector`: mean `+7.5%` vs `adapt_only`, mean `+3.6%` vs fixed
  - interpretation:
    - adding a richer handcrafted signature helps with the adversarial ambiguity failure mode
    - but handcrafted multi-signal rules are still too brittle to become the new main selector story
    - the next selector step should shift from better thresholds to a learned or explicitly fit selector over task-signature features
- Linear fitted task-signature selector is now tested and also negative as a replacement:
  - new policy: `task_specialist_clone_fitted_selector_merge`
  - canonical result: `results/canonical/phase3_fitted_selector_benchmark.json`
  - canonical report: `PHASE3_FITTED_SELECTOR_BENCHMARK.md`
  - adversarial check: `results/canonical/phase3_selector_adversarial_benchmark_fitted.json`
  - result:
    - on the formal stress-map family, it matches the stronger hardcoded/fingerprint frontier:
      - `fitted_selector`: mean `+7.5%` vs `adapt_only`, mean `+3.6%` vs fixed
    - but on `feature_shift_embedded` it collapses back to the weaker boost frontier:
      - `fitted_selector`: mean `-1.2%` vs fixed
      - `signature_selector`: mean `-0.5%` vs fixed
  - interpretation:
    - simply replacing boolean rules with a linear fit over the current task-signature features is not enough
    - the current feature set is sufficient to recover the formal family, but still not sufficient to survive adversarial regime ambiguity
    - the next selector step should target better regime features or outcome-grounded / learned supervision, not just a cleaner linear decision surface
- Richer dynamics-aware selector is now tested and also negative as a replacement:
  - new policy: `task_specialist_clone_dynamics_selector_merge`
  - canonical result: `results/canonical/phase3_dynamics_selector_benchmark.json`
  - canonical report: `PHASE3_DYNAMICS_SELECTOR_BENCHMARK.md`
  - mechanism:
    - extends the selector feature set with online task dynamics:
      - `loss_mean`
      - `loss_delta`
      - `conflict_delta`
      - `confidence_delta`
  - result:
    - on `selector_full_map`, it effectively ties the current `fingerprint_selector` aggregate:
      - `dynamics_selector`: mean `+4.0%` vs `adapt_only`, mean `+5.0%` vs fixed
      - `fingerprint_selector`: mean `+4.0%` vs `adapt_only`, mean `+5.0%` vs fixed
    - but it still fails the main adversarial gate:
      - `selector_adversarial_gate` (`feature_shift_embedded`): `-1.7%` vs fixed
      - `selector_sparse_adversarial_gate` (`feature_shift_sparse_embedded`): `-1.0%` vs fixed
      - `signature_selector` still remains better on the embedded adversarial gate at `+0.0%` vs fixed
  - interpretation:
    - adding learning-dynamics features alone is not enough
    - the main selector bottleneck now looks more like supervision / objective mismatch than a missing simple online feature
- Outcome-supervised selector is now tested and also negative as a replacement:
  - new policy: `task_specialist_clone_outcome_selector_merge`
  - canonical result: `results/canonical/phase3_outcome_selector_benchmark.json`
  - canonical report: `PHASE3_OUTCOME_SELECTOR_BENCHMARK.md`
  - mechanism:
    - fits a selector directly to the real outcome margin between `task_specialist_clone_limited_merge` and `task_specialist_clone_current_path_boost_merge`
    - supervision is result-driven, not task-family-labeled
  - result:
    - aggregate on `selector_full_map`:
      - `outcome_selector`: mean `+2.8%` vs `adapt_only`, mean `+3.8%` vs fixed
      - this is weaker than `fingerprint_selector` and `signature_selector`
    - selector gates:
      - `selector_adversarial_gate` (`feature_shift_embedded`): `-0.6%` vs fixed
      - `selector_sparse_adversarial_gate` (`feature_shift_sparse_embedded`): `-2.4%` vs fixed
      - `signature_selector` still remains better on the embedded adversarial gate at `+0.0%` vs fixed
  - interpretation:
    - stronger supervision helps recover the embedded gate relative to the boost-side selectors
    - but a simple linear outcome fit still overcommits to the interference-control side and degrades the broader full-map aggregate
    - the next selector move should target a more structured target or objective, not just swap in another linear supervision surface
- Selector generalization benchmark using `feature_shift_nonnegative` is now available but not strong enough to upgrade into the main stress-map family:
  - canonical result: `results/canonical/phase3_selector_generalization_benchmark.json`
  - canonical report: `PHASE3_SELECTOR_GENERALIZATION_BENCHMARK.md`
  - result:
    - all main candidates become nearly tied on this suite
    - `limited_merge`, `current_path_boost`, `regime_switch`, `diagnostic_selector`, and `fingerprint_selector` all land around `+12.2%` to `+12.3%` vs fixed
  - interpretation:
    - the suite is too easy / too weakly diagnostic to meaningfully test selector generalization
    - nonnegative-input transformation alone is not a good adversarial regime-generalization benchmark
    - next benchmark work should target stronger regime ambiguity, not just surface input-stat shifts

Negative but decisive:

- `task_specialist_clone_role_separation_merge`
- `task_specialist_clone_source_aware_merge`
- `task_specialist_clone_asymmetric_retention_merge`
- `task_specialist_clone_dual_path_boost_merge`
- `task_specialist_clone_dual_path_decoupled_merge`
- `task_specialist_clone_external_memory_boost_merge`
- `task_specialist_clone_external_memory_routed_merge`
- `task_specialist_clone_two_speed_boost_merge`
- `task_specialist_clone_decoupled_objective_boost_merge`
- `task_specialist_clone_mode_switch_boost_merge`
- `task_specialist_clone_dual_mode_update_merge`

Current interpretation of that negative set:

- simple retention tweaks are saturated
- simple memory splitting is saturated
- light retrieval heuristics are saturated
- trivial optimizer/objective splitting is saturated
- conflict-aware local mode switching is still too weak to merge the plasticity and interference frontiers
- even a more explicit dual-mode update rule still collapses to the same plasticity/interference tradeoff rather than creating a new frontier
- simple scalar diagnostics are too weak, but explicit task fingerprints are already strong enough to recover regime switching
- richer handcrafted task signatures partially improve adversarial ambiguity, but still do not replace the stronger hardcoded/fingerprint frontier
- linear fits over the current task-signature features still collapse on the adversarial gate, so the bottleneck is now feature sufficiency as much as classifier form
- even richer online dynamics features still collapse on the adversarial gate, so the next selector move should shift toward stronger supervision or a more structured selector objective
- even stronger outcome supervision does not solve the problem by itself; the selector target likely needs structure beyond a single linear margin fit

Current interpretation of the new quantization result:

- low-bit compression is not a replacement for the main mechanism
- but it is now a credible memory-representation direction
- archived specialists may not need high precision to remain useful
- internal compressed memory currently looks better than external compressed memory
- precision compression is promising, but count compression is now a constrained negative
- future memory work should treat precision budget and archive-count budget as separate questions
- simple archive-selection heuristics are now close to saturated as well
- simple archive-representation heuristics are now close to saturated as well
- lightweight weight-space prototype compression also looks saturated
- lightweight sample-conditioned memory retrieval also looks saturated

## Open Technical Risks

1. `core/stability_test.py` was previously overstating project-wide success.
   It has now been downgraded to an SC-4-only script, but its old outputs should not be cited as full validation.

2. `Phase 3` is no longer a single monolith, but policy semantics are still not fully declarative.
   The runner now consumes a registry, and `ReusePolicyNetwork.evolve()` no longer owns the full policy branch chain by itself. Both the special-case preclone dispatch and the clone-application tail now delegate into [core/phase3_policies.py](/F:/sel-lab/core/phase3_policies.py), which uses explicit handler dispatch, family-level grouping, and registry-derived family assignment instead of hand-maintained parallel grouping tables. The next cleanup step should move from registry-backed grouping to stronger registry-backed policy dispatch or shared family scaffolds.

3. Visualization consistency is still uncertain.
   Some visualization code paths assume fields that may no longer match the normalized metrics pipeline.

4. `Phase 4` remains secondary evidence.
   It is useful, but still less settled than `Phase 3`.

5. The current quantization result is only a stress test, not TurboQuant-style outlier-aware PTQ.
   It supports a direction, but should not be overstated as a full low-bit inference result.

## Immediate Next Work

Repository cleanup still has higher priority than more local mechanism variants.

1. `UNIFIED_REPORT.md` has been realigned so the canonical broader-digits leader is now `task_specialist_clone_current_path_boost_merge`, while `task_specialist_clone_distill_merge` is explicitly treated as the narrower base-`digits_pairs` point rather than the broader benchmark main line.
2. `tests/smoke_test.py` now includes Phase 3 structural consistency checks:
   - registry policy names vs policy-layer handlers
   - handler family grouping vs registry family metadata
   - canonical `UNIFIED_REPORT.md` wording for the current broader-digits main line
3. `analysis/generate_unified_report.py` now has an explicit `CANONICAL_RESULT_FILES` manifest.
   `UNIFIED_REPORT.md` is no longer fed by scattered implicit loads alone, and `tests/smoke_test.py` now checks that every canonical result file in that manifest exists.
4. Result layering has now started in the filesystem, not just in docs:
   - canonical generated summaries now save under `results/canonical/`
   - smoke artifacts now save under `results/smoke/`
   - exploratory one-off scripts can now save under `results/exploratory/`
   - root-level legacy files are still readable for backward compatibility
5. The main default canonical producers now actually write into `results/canonical/`:
   - `phase2_results.json`
   - `phase2_hard_results.json`
   - `phase2_improved_results.json`
   - `phase4_results.json`
   - `phase4_real_results.json`
   - `unified_summary.json`
6. Several old standalone scripts no longer hard-code `F:/skill/sel-lab/results/...`:
   - `core/convergence_analysis.py`
   - `core/convergence_quick.py`
   - `core/data_augmentation.py`
   - `core/learning_rate_schedule.py`
   - `core/phase4_quick.py`
   - `core/phase4_mnist_full.py`
   - `core/phase_d_scale.py`
   - `core/structural_evolution_analysis.py`
   They now save to `results/exploratory/` through `runtime.py`, and their imports were adjusted so `python core\\...py` still works.
7. Canonical-first result loading is now shared in `core/runtime.py`:
   - `resolve_existing_results_path(...)`
   - `load_json_result(...)`
   `analysis/generate_unified_report.py` and key Phase 3 analysis scripts can now prefer `results/canonical/` while still reading root-level legacy files during migration.
8. `analysis/phase3_digits_gap_decomposition.py` now writes its canonical JSON to `results/canonical/`.
   `analysis/phase3_gap_analysis.py` now writes exploratory JSON to `results/exploratory/`.
9. Phase 4 result routing is cleaner now:
   - `core/phase4_failure_analysis.py` no longer writes a root-level JSON on every function call
   - it only saves when an explicit entrypoint requests output
   - `core/phase4_repair_experiment.py` now saves to `results/canonical/phase4_repair_sweep.json`
   - `core/phase4_expansion_sweep.py` is also configured to save future runs to `results/canonical/phase4_expansion_sweep.json`
10. Phase 3 analysis scripts are now being split by result intent:
   - benchmark/transfer outputs are being moved to `results/canonical/`
   - sweep outputs are being moved to `results/exploratory/`
   - confirmed by rerunning `analysis/phase3_digits_transfer.py`, which now saves to `results/canonical/phase3_digits_transfer.json`
   - one old exploratory sweep (`analysis/phase3_distill_sweep.py`) exceeded a 5-minute timeout, so these scripts should now be treated as heavier jobs rather than lightweight maintenance entrypoints
11. New `Phase 3 diagnostic suite` is now in place:
   - canonical result: `results/canonical/phase3_diagnostic_suite.json`
   - canonical report: `PHASE3_DIAGNOSTIC_SUITE.md`
   - diagnostic wins:
     - `plasticity_stress` (`digits_pairs_noisy`) -> `task_specialist_clone_current_path_boost_merge`
     - `interference_stress` (`feature_shift`) -> `task_specialist_clone_limited_merge`
     - `retrieval_ambiguity_stress` (`digits_pairs_permuted`) -> `task_specialist_clone_current_path_boost_merge`
   - current read: `plasticity_boost` wins 2/3 diagnostic stresses, but low-merge `specialist_reuse` still wins the interference-focused stress test
12. Stronger interference benchmark family is now in place:
   - new suite: `feature_shift_noisy`
   - canonical result: `results/canonical/phase3_interference_diagnostic.json`
   - canonical report: `PHASE3_INTERFERENCE_DIAGNOSTIC.md`
   - result: `task_specialist_clone_limited_merge` is the overall winner across `feature_shift` and `feature_shift_noisy`
   - interpretation: once interference pressure is made more explicit, low-merge control beats both raw specialist cloning and `current_path_boost`
13. Minimal hybrid mechanism check is now available:
   - new policy: `task_specialist_clone_confidence_boost_merge`
   - benchmark: `PHASE3_HYBRID_BOOST_DIAGNOSTIC.md`
   - canonical result: `results/canonical/phase3_hybrid_boost_diagnostic.json`
   - result: the hybrid is only a marginal tweak, not a new phase change
     - it slightly improves over `task_specialist_clone_current_path_boost_merge` on the mixed diagnostic aggregate
     - but it does not recover the stronger interference robustness of `task_specialist_clone_limited_merge`
   - interpretation: a small confidence gate on top of current-path boost is not enough to unify the plasticity and interference frontiers
14. `analysis/phase3_current_path_boost_benchmark.py` has now been rerun after the canonical-path migration and writes to `results/canonical/phase3_current_path_boost_benchmark.json`.
    The broader-digits leader remains `task_specialist_clone_current_path_boost_merge`.
15. A full rerun of `analysis/phase3_digits_benchmark_transfer.py` still exceeds a 10-minute timeout and should be treated as a heavier batch job.
16. Keep using the Phase 3 registry as the source of truth for:
   - canonical policy order
   - policy family / mechanism intent
   - diagnostic suite families
17. Continue normalizing the new Phase 3 policy layer so policy-specific behavior is easier to inspect outside `core/phase3_model.py`.
18. Push the current registry-backed family grouping toward stronger registry-backed policy dispatch or shared family scaffolds.
19. Stop treating this whole compressed-memory micro-variant family as the default next move. Shift either to genuinely richer memory representations or back to core learner cleanup.
20. Keep the selector line on the main path, but stop spending loops on more handcrafted boolean rule variants.
    - `task_specialist_clone_signature_selector_merge` was useful because it improved the adversarial selector suite
    - but it did not replace the stronger hardcoded/fingerprint frontier on the formal stress-map family
21. The most credible next selector step is now a learned or explicitly fit selector over task-signature features.
    - good candidate feature set:
      `confidence`, `conflict_score`, `conflict_peak`, `input_abs_mean`, `input_nonnegative_ratio`, `input_zero_ratio`
    - the immediate target is not to beat the hardcoded oracle everywhere
    - it is to preserve the formal-family fingerprint frontier while not collapsing on `feature_shift_embedded`
    - note:
      a simple linear fit over the current feature set already recovers the formal family and still fails the adversarial gate, so the next upgrade likely needs richer regime features, better supervision, or both
22. Treat `feature_shift_embedded` as the current selector-adversarial gate.
    - if a new selector only matches `phase3_fingerprint_selector_benchmark.json` but still collapses on `phase3_selector_adversarial_benchmark.json`, it is not a narrative upgrade
23. Selector benchmarking now has a stronger registry-backed screening layer, not just separate point reports.
    - shared runner: `analysis/phase3_benchmark_family.py`
    - new formal families in `core/phase3_registry.py`:
      - `selector_gate_map`
      - `selector_full_map`
    - new candidate entrypoint:
      - `analysis/phase3_selector_stress_map_benchmark.py`
    - interpretation:
      future selector work should be screened on `selector_full_map`, not only on `expanded_stress_map`
24. `python tests\smoke_test.py` now passes cleanly again after fixing the direct-entry fixture bug.
    - current smoke status: `15 passed, 0 failed`
    - this matters because the benchmark-family infrastructure and selector-gate registry additions now have a stable cheap verification path
25. New learned selector candidate `task_specialist_clone_prototype_selector_merge` has now been tested.
    - benchmark:
      - `analysis/phase3_prototype_selector_benchmark.py`
      - `results/canonical/phase3_prototype_selector_benchmark.json`
      - `PHASE3_PROTOTYPE_SELECTOR_BENCHMARK.md`
    - training family: `expanded_stress_map`
    - evaluation family: `selector_full_map`
    - result:
      exemplar-style nearest-prototype selection still collapses to the same frontier as `fingerprint_selector`
    - key adversarial read:
      - `prototype_selector`: `-1.7%` vs fixed
      - `fingerprint_selector`: `-1.7%` vs fixed
      - `signature_selector`: `+0.0%` vs fixed
      - hardcoded `regime_switch`: `-0.6%` vs fixed
    - interpretation:
      nonlinearity alone over the current feature set is not enough; the remaining gap is likely feature quality, supervision quality, or both
26. Shared stress-benchmark config now correctly routes `feature_shift_embedded` through the high-dimensional path.
    - practical reason:
      that suite uses embedded 64-D inputs and previously would fail under the generic family helper
    - verification:
      full prototype-selector benchmark now completes, and `python tests\smoke_test.py` still passes cleanly
27. New learned selector candidate `task_specialist_clone_dynamics_selector_merge` has now been tested.
    - benchmark:
      - `analysis/phase3_dynamics_selector_benchmark.py`
      - `results/canonical/phase3_dynamics_selector_benchmark.json`
      - `PHASE3_DYNAMICS_SELECTOR_BENCHMARK.md`
    - training family: `expanded_stress_map`
    - evaluation family: `selector_full_map`
    - result:
      richer task-dynamics features let a linear selector recover the same aggregate frontier as `fingerprint_selector`
    - key adversarial read:
      - `dynamics_selector`: `-1.7%` vs fixed on `feature_shift_embedded`
      - `dynamics_selector`: `-1.0%` vs fixed on `feature_shift_sparse_embedded`
      - `signature_selector`: `+0.0%` vs fixed on `feature_shift_embedded`
    - interpretation:
      better online features alone are not enough; next selector work should target supervision or objective structure, not just another feature family
28. New learned selector candidate `task_specialist_clone_outcome_selector_merge` has now been tested.
    - benchmark:
      - `analysis/phase3_outcome_selector_benchmark.py`
      - `results/canonical/phase3_outcome_selector_benchmark.json`
      - `PHASE3_OUTCOME_SELECTOR_BENCHMARK.md`
    - training family: `expanded_stress_map`
    - evaluation family: `selector_full_map`
    - result:
      outcome-supervised fitting does not become the new main selector frontier
    - key adversarial read:
      - `outcome_selector`: `-0.6%` vs fixed on `feature_shift_embedded`
      - `outcome_selector`: `-2.4%` vs fixed on `feature_shift_sparse_embedded`
      - `signature_selector`: `+0.0%` vs fixed on `feature_shift_embedded`
    - interpretation:
      stronger supervision helps on the main embedded gate but still does not produce a robust selector; next work should focus on structured selector objectives or richer target formation
29. New preservation-style selector candidate `task_specialist_clone_guarded_expert_selector_merge` has now been tested.
    - benchmark:
      - `analysis/phase3_guarded_expert_selector_benchmark.py`
      - `results/canonical/phase3_guarded_expert_selector_benchmark.json`
      - `PHASE3_GUARDED_EXPERT_SELECTOR_BENCHMARK.md`
    - training family: `selector_full_map`
    - evaluation family: `selector_full_map`
    - mechanism:
      preserve the `two_stage_selector` hard sparse branch and embedded signature branch; learn only a quadratic default-region head
    - result:
      it nearly matches `two_stage_selector` end-to-end and matches it exactly on the two key adversarial gates
    - key adversarial read:
      - `guarded_expert_selector`: `+0.0%` vs fixed on `feature_shift_embedded`
      - `guarded_expert_selector`: `-0.8%` vs fixed on `feature_shift_sparse_embedded`
      - `two_stage_selector`: `+0.0%` / `-0.8%` on the same gates
    - aggregate read:
      - `guarded_expert_selector`: mean `+4.3%` vs `adapt_only`, mean `+5.3%` vs fixed
      - `two_stage_selector`: mean `+4.4%` vs `adapt_only`, mean `+5.3%` vs fixed
    - interpretation:
      the meaningful selector frontier is now much clearer: the nontrivial advantage comes from preserving both hand-designed branches, not from relearning them with another segmented objective
      the remaining difference versus `two_stage_selector` is a tiny tail effect on non-gate suites, not a collapse on the important adversarial checks
30. Direct simplification of `guarded_expert` has now been tested.
    - benchmark:
      - `analysis/phase3_guarded_expert_simplification_benchmark.py`
      - `results/canonical/phase3_guarded_expert_simplification_benchmark.json`
      - `PHASE3_GUARDED_EXPERT_SIMPLIFICATION_BENCHMARK.md`
    - tested simplifications:
      - `task_specialist_clone_guarded_expert_boost_default_merge`
      - `task_specialist_clone_guarded_expert_merge_default_merge`
    - mechanism:
      preserve the hard sparse branch and embedded signature branch, but remove the learned default-region head and replace it with a hardcoded default fallback
    - result:
      the adversarial gates stay intact, but aggregate quality drops unless the default region still has a learned fallback
    - key read:
      - `guarded_expert_selector`: mean `+4.3%` vs `adapt_only`, mean `+5.3%` vs fixed
      - `guarded_expert_boost_default`: mean `+3.9%` vs `adapt_only`, mean `+4.8%` vs fixed
      - `guarded_expert_merge_default`: mean `+2.0%` vs `adapt_only`, mean `+2.9%` vs fixed
      - all three variants still match `two_stage_selector` on `feature_shift_embedded` and `feature_shift_sparse_embedded`
    - interpretation:
      simplifying away the default-region learner is not free
      the minimal effective selector story is now sharper: preserve the hard sparse branch, preserve the embedded signature branch, and keep a learned default fallback
      hardcoding the default region to boost regresses to roughly the `signature_selector` frontier, while hardcoding it to interference control gives away too much plasticity

## Working Rule

If a change does not improve:

- `Phase 3` evidence quality
- report/document consistency
- experiment credibility
- or mechanism interpretability

it is probably not part of the current main project.
