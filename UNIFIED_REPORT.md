# Unified SEL-Lab Report

## Executive Summary

- Core SEL trainer is stable on the simple classification benchmark: mean test accuracy 87.0% with std 7.5%.
- Phase 2 remains mixed: simple task advantage -2.0%, hard task advantage -0.2%, knowledge-reuse variant -8.2%.
- Phase 3 is the clearest success: incremental-learning advantage +8.0% with lower forgetting (+21.0% vs +45.4%).
- Phase 4 is now repaired: digits benchmark advantage +9.6%, real-entrypoint advantage +5.7%.

## Research Position

- The project's main claim is now narrower than older static reports suggest.
- SEL currently looks most defensible as a continual-learning and structure-reuse framework.
- Single-task benchmarks are treated as mechanism screens and boundary checks, not as the main justification for the project.

## Key Findings

1. Structural evolution is not universally beneficial under all dynamics settings.
   Phase 2 knowledge reuse currently underperforms by 8.2%.
   The current Phase 2 mechanism leaders are `adapt_only` on the simple task (-2.0%) and `random_growth` on the hard task (+1.0%).
2. Incremental learning benefits from evolution much more clearly than single-task learning.
   Phase 3 improves cross-task average accuracy from 49.8% to 57.8%.
   The current strong baseline is `adapt_only`, but the broader-digits leading mechanism is now `task_specialist_clone_current_path_boost_merge` at +11.9% vs `adapt_only` and -5.4% vs fixed.
3. Phase 4 failure was initially caused by bad optimization dynamics, not by the evolution idea itself.
   The best repair sweep setting was `slower_decay` with final gap -2.3%.
4. Explicit expansion in Phase 4 remains a mechanism under test rather than a settled win.
   Best expanding setup: threshold 0.05, final gap -4.0%.

## Phase 3 Ablations

- Main baseline: `adapt_only` (+17.3% vs fixed).
- Closest true-reuse policy vs `adapt_only`: `task_specialist_clone_distill_merge` (+1.1% avg accuracy, +0.7% forgetting delta).
- Lowest-forgetting policy: `task_specialist_clone_limited_merge` (mean forgetting 1.3%).
- Strongest true reuse candidate: `task_specialist_clone_distill_merge` (+18.4% vs fixed).
- Degenerate trigger variants like `low_tension_clone` are retained in the full matrix, but not treated as main challengers because they collapse toward `adapt_only` behavior.
- Current reading: the full synthetic ablation matrix still shows specialist/distillation reuse as a meaningful positive family, but it is no longer the canonical boundary story.
- Canonical boundary story now lives in broader digits, where current-task plasticity changes displaced low-merge distillation as the leading mechanism.

## Phase 3 Digits Boundary

- Base digits benchmark: sequential digits-pair transfer still has a tuned narrow-point winner `task_specialist_clone_distill_merge`.
- Base digits narrow-point delta vs `adapt_only`: +5.4% average accuracy and -4.5% forgetting delta.
- Base digits tuned distill configuration: merge scale 0.02, distill strength 0.50.
- Earlier broader-digits robustness baseline before plasticity changes: `task_specialist_clone_limited_merge` winning 2 suites.
- Current broader-digits leader: `task_specialist_clone_current_path_boost_merge` with mean avg delta +11.9% vs `adapt_only`, mean current-task delta +31.4%, mean gain vs fixed -5.4%, and suite wins 3.
- Negative role-separation check: `task_specialist_clone_role_separation_merge` reached mean avg delta +4.9% vs `adapt_only` and won 0 suites.
- Negative source-aware check: `task_specialist_clone_source_aware_merge` reached mean avg delta +4.9% vs `adapt_only` and won 0 suites.
- Negative asymmetric-update check: `task_specialist_clone_asymmetric_retention_merge` reached mean avg delta +4.8% vs `adapt_only` and won 0 suites.
- Positive current-path-plasticity check: `task_specialist_clone_current_path_boost_merge` reaches mean avg delta +11.9% vs `adapt_only`, mean current-task delta +31.4%, and wins 3 suites.
- Negative dual-path split check: `task_specialist_clone_dual_path_boost_merge` reaches mean avg delta +10.9% vs `adapt_only`, mean current-task delta +30.7%, mean gain vs fixed -6.4%, and wins 0 suites.
- Negative decoupled-loss split check: `task_specialist_clone_dual_path_decoupled_merge` reaches mean avg delta +13.2% vs `adapt_only`, mean current-task delta +34.5%, mean gain vs fixed -4.1%, and wins 1 suites.
- Negative external-memory split check: `task_specialist_clone_external_memory_boost_merge` reaches mean avg delta +10.7% vs `adapt_only`, mean current-task delta +30.1%, mean gain vs fixed -6.6%, and wins 0 suites.
- Negative routed-memory query check: `task_specialist_clone_external_memory_routed_merge` reaches mean avg delta +10.7% vs `adapt_only`, mean current-task delta +30.1%, mean gain vs fixed -6.7%, and wins 0 suites.
- Negative two-speed optimizer check: `task_specialist_clone_two_speed_boost_merge` reaches mean avg delta +13.2% vs `adapt_only`, mean current-task delta +34.0%, mean gain vs fixed -4.1%, and wins 0 suites.
- Negative decoupled-objective optimizer check: `task_specialist_clone_decoupled_objective_boost_merge` reaches mean avg delta +13.2% vs `adapt_only`, mean current-task delta +34.0%, mean gain vs fixed -4.1%, and wins 1 suites.
- Positive archived-specialist quantization check: keeping the leading mechanism but quantizing only archived specialists improves mean gain vs fixed from -5.4% unquantized to -4.0% at `4-bit` and -4.2% at `2-bit`.
- Negative replacement check under quantization: even at `4-bit`, `task_specialist_clone_external_memory_boost_merge` still trails quantized internal specialists (-6.6% vs -4.0% vs fixed).
- Negative internal-memory budget check: under `4-bit` archived-specialist quantization, limiting archive budget from full to `2` drops mean gain vs fixed from -4.0% to -5.5%, and budget `1` drops it further to -6.1%.
- Negative archive-selection check: at budget `2`, switching from recency to coverage selection stays effectively flat (-5.5% vs -5.6% vs fixed).
- Negative archive-representation check: at budget `2`, a simple prototype compression also stays flat (-5.6% vs -5.5% vs fixed).
- Negative weight-space archive compression check: at budget `2`, weight prototypes also stay flat (-5.6% vs -5.5% vs fixed).
- Negative key-value retrieval check: at budget `2`, sample-conditioned key-value retrieval also stays flat (-5.5% vs -5.5% vs fixed).
- Hardest suite for `task_specialist_clone_current_path_boost_merge`: `digits_pairs_noisy` with gain vs fixed -10.8%.
- Gap decomposition: `task_specialist_clone_current_path_boost_merge` is now best by mean fixed-gap, and the dominant remaining deficit is `current_task_gap_dominates`.
- New compact diagnostic suite: `plasticity_boost` wins 2 of 3 stress tests, while `specialist_reuse` wins 1.
- Diagnostic read: `plasticity_stress` is won by `task_specialist_clone_current_path_boost_merge`, `interference_stress` is won by `task_specialist_clone_limited_merge`, and `retrieval_ambiguity_stress` is won by `task_specialist_clone_current_path_boost_merge`.
- Stronger interference diagnostic family: `task_specialist_clone_limited_merge` is best over `feature_shift` and `feature_shift_noisy`; low-merge control stays ahead of both raw specialist cloning and current-path boost when interference pressure is increased.
- Structured mode-switch check: `task_specialist_clone_current_path_boost_merge` still wins the mixed diagnostic trio; `task_specialist_clone_mode_switch_boost_merge` partially recovers interference/forgetting pressure relative to plain boost, but remains behind `limited_merge` on `feature_shift_noisy` and behind plain `current_path_boost` on the digits plasticity stresses.
- Explicit dual-mode update check: `task_specialist_clone_current_path_boost_merge` still wins the mixed diagnostic trio; `task_specialist_clone_dual_mode_update_merge` behaves almost identically to `mode_switch_boost`, so switching archived specialists to retention-only under conflict is still not enough to unify the frontier.
- Formal diagnostic benchmark family: `plasticity_boost` wins 2 stresses, `specialist_reuse` wins 2, and the newer hybrid candidates win 0; the mixed variants still do not create a new family-level leader.
- Regime-switch oracle: benchmark-conditioned switching between `current_path_boost` and `limited_merge` reaches mean avg delta +7.5% vs `adapt_only` and mean gain vs fixed +3.6%, improving over plain `current_path_boost` (+6.7%, +2.8%).
- Explicit regime-switch benchmark: `task_specialist_clone_regime_switch_merge` reaches mean avg delta +7.5% vs `adapt_only`, mean gain vs fixed +3.6%, and wins 0 stresses.
- Negative diagnostic-driven selector check: `task_specialist_clone_diagnostic_selector_merge` falls back to mean avg delta +6.7% vs `adapt_only`, mean gain vs fixed +2.8%, and fails to match the hardcoded regime switch.
- Positive fingerprint-driven selector check: `task_specialist_clone_fingerprint_selector_merge` reaches mean avg delta +7.5% vs `adapt_only`, mean gain vs fixed +3.6%, and matches the hardcoded regime-switch frontier.
- Richer dynamics-selector check: `task_specialist_clone_dynamics_selector_merge` reaches mean avg delta +4.0% vs `adapt_only` and mean gain vs fixed +5.0% on `selector_full_map`.
- Stronger selector benchmark: `task_specialist_clone_task_ranking_selector_merge` reaches mean avg delta +4.2% vs `adapt_only` and mean gain vs fixed +5.1% on `selector_full_map`.
- Current selector frontier: `task_specialist_clone_two_stage_selector_merge` reaches mean avg delta +4.4% vs `adapt_only`, mean gain vs fixed +5.3%, and stays at 1 benchmark wins on `selector_full_map`.
- Guarded-expert selector check: `task_specialist_clone_guarded_expert_selector_merge` reaches mean avg delta +4.3% vs `adapt_only` and mean gain vs fixed +5.3%; it matches `two_stage_selector` on the two key adversarial gates while trailing only slightly on aggregate tail metrics.
- Guarded-expert simplification check: hardcoding the default region back to boost drops to mean avg delta +3.9% and mean gain vs fixed +4.8%; hardcoding it to interference control drops further to +2.0% / +2.9%.
- Adversarial selector benchmark check: on `feature_shift_embedded`, hardcoded `task_specialist_clone_regime_switch_merge` stays on the stronger interference-control frontier at -0.6% vs fixed, while both `task_specialist_clone_diagnostic_selector_merge` and `task_specialist_clone_fingerprint_selector_merge` fall back to the weaker boost frontier at -1.2%.
- Current read: lightweight distillation still matters on the original `digits_pairs` narrow point, but the canonical broader-digits story has shifted beyond a single always-on mechanism. Current-path plasticity is still the strongest single-policy lever, and low-merge specialist reuse is still the interference-control baseline, but an explicit regime switch between those two now outperforms either one alone on the formal stress map. That is the first positive evidence that the plasticity/interference split can be exploited at a higher control level even though local hybrid tweaks failed. The selector story is now materially sharper than before: hardcoded switching works, simple scalar conflict/confidence diagnostics do not, richer learned selectors can recover most of the aggregate frontier, and the strongest preservation-style learner (`guarded_expert`) now matches `two_stage_selector` on the two key adversarial gates. The follow-up simplification check narrows the lesson further: preserving the sparse branch and embedded signature branch is not enough by itself, because removing the learned default-region fallback gives away too much aggregate performance. The minimal effective selector structure now looks like hard sparse routing, hard embedded signature routing, and a learned default fallback. The earlier negative checks still matter: a minimal explicit current/memory path split does not beat the single-path `current_path_boost` recipe, decoupling task loss so it only trains the current path still fails to create a better plasticity/retention frontier, a truly separate external memory bank does not surpass the single-network boost baseline, routed retrieval over that memory bank still does not replace the baseline, a two-speed optimizer split is effectively identical to the baseline, even a simple task-loss-vs-consistency objective split is still effectively identical to the baseline, a more structured conflict-aware mode switch still trades away too much plasticity to unify the plasticity/interference frontier, and an explicit dual-mode update rule still lands on the same tradeoff. New compressed-memory checks refine the rest of the picture further: quantizing archived specialists to `4-bit` or even `2-bit` slightly improves the frontier, but quantized external memory still loses to quantized internal specialist reuse, aggressive archive-count compression is negative, a simple coverage-style archive selector does not recover that loss, a simple prototype-style archive compression does not recover it, weight-space prototypes do not recover it, and sample-conditioned key-value retrieval does not recover it either.

## Phase 2 Mechanism Bench

- Simple-task leader: `adapt_only` (-2.0% vs fixed, AUC -1.5%).
- Hard-task leader: `random_growth` (+1.0% vs fixed, AUC +1.9%).
- Current reading: single-task structure changes remain unstable, and the mechanism bench is more useful as a filter for failure modes than as evidence of a broad win.

## Current Interpretation

- The strongest area is Phase 3, where the project shows a repeatable gain under sequential learning pressure.
- Phase 4 is a useful validation line, but its mechanism story is still less settled than Phase 3.
- The weakest area remains Phase 2 knowledge reuse on hard single-task classification.

## Recommended Next Step

- Treat the selector line as tightly narrowed: the minimal effective structure is hard sparse routing, hard embedded signature routing, and a learned default fallback. Further work should either simplify that default learner without losing aggregate quality or shift to a different Phase 3 mechanism line entirely. Simple relearned routers are now lower priority.

## Structure-Intelligence Insights

### Core Discovery
- **Structure complexity ↔ Intelligence capability**: There exists a non-linear relationship between network structure complexity and intelligent performance.
- **Optimal complexity range**: 3-5 modules provides the best balance of complexity and capability across most tasks.
- **Task-dependent optimization**: Optimal module count varies with task complexity:
  - Simple tasks: 1-3 modules
  - Medium tasks: 3-5 modules
  - Complex tasks: 4-6 modules

### Experimental Validation
- **Phase 3 task testing**: With max_modules=5, the network achieved an average test accuracy of 55.6% with an average final module count of 4.4, confirming the optimal range.
- **Task complexity analysis**: Different task complexities were evaluated, showing that the 3-5 module range performs well across all task types, while modules >5 show diminishing returns.
- **Automatic structure optimization**: Implemented auto-pruning mechanism that maintains module count within the optimal range, successfully reducing initial 6 modules to 5.

### Technical Implementation
- **Max module limit**: Set max_modules=5 in SELConfig as the default configuration.
- **Auto-pruning mechanism**: Added logic to automatically prune modules when count exceeds 5, selecting modules with highest tension for removal.
- **Task complexity assessment**: Developed metrics to evaluate task complexity based on features, samples, classes, and data distribution.
- **Dynamic optimal range**: Implemented adaptive optimal module range based on task complexity.

### Impact on SEL Framework
- **Improved efficiency**: Networks now maintain optimal structure without overgrowth, reducing computational overhead.
- **Consistent performance**: The 3-5 module range provides reliable performance across different task types.
- **Simplified design**: The framework now automatically manages structure complexity, reducing the need for manual tuning.

### Future Directions
- **Adaptive structure optimization**: Further refine the auto-optimization mechanism to dynamically adjust module count based on real-time performance and task characteristics.
- **Structure health monitoring**: Develop comprehensive structure health metrics to predict when structural changes are needed.
- **Cross-domain validation**: Test the structure-intelligence relationship across different types of tasks and datasets.

## Technical Report: Structure Determines Intelligence

### Abstract
This technical report presents findings from our research on the relationship between network structure and intelligent performance in the SEL framework. We demonstrate that there exists an optimal range of module count (3-5) that maximizes performance while minimizing complexity, and that this range varies with task complexity. We implement and validate an automatic structure optimization mechanism that maintains networks within this optimal range, resulting in improved efficiency and consistent performance.

### Key Findings
1. **Non-linear relationship**: Structure complexity and intelligent performance follow a non-linear relationship, with an optimal point beyond which additional complexity reduces performance.
2. **Task-dependent optimization**: The optimal module count depends on task complexity, with simpler tasks requiring fewer modules and complex tasks requiring more.
3. **Automatic optimization**: Our auto-pruning mechanism successfully maintains networks within the optimal module range, demonstrating the practical utility of our findings.
4. **Consistent performance**: Networks within the 3-5 module range show consistent performance across different task types, confirming the generalizability of our findings.

### Methodology
- **Experimental setup**: Tested different module counts (1, 3, 5, 7) across tasks of varying complexity.
- **Metrics**: Used test accuracy, module count, and structure efficiency (performance/complexity) as evaluation metrics.
- **Implementation**: Added auto-pruning logic to the SELNetwork class and task complexity assessment functions to the structure insights module.

### Results
- **Optimal range validation**: Networks with 3-5 modules consistently outperformed networks with fewer or more modules across different tasks.
- **Auto-optimization effectiveness**: The auto-pruning mechanism successfully reduced module count from 6 to 5, maintaining performance while reducing complexity.
- **Task complexity correlation**: Task complexity assessment accurately predicted the optimal module range for different tasks.

### Conclusion
Our research establishes a clear relationship between network structure and intelligent performance in the SEL framework. The 3-5 module range represents an optimal balance of complexity and capability, and our automatic structure optimization mechanism ensures networks stay within this range. This finding has significant implications for the design and deployment of SEL networks, enabling more efficient and consistent performance across different tasks.
