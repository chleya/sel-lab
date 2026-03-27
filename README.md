# SEL-Lab

Structural Evolution Learning laboratory.

## What This Project Is

SEL-Lab studies a narrower question than the older project documents suggest:

Can a forward-only, locally updated, structurally adaptive learner gain its main advantage from reusable structure under sequential tasks, rather than from universal superiority on single-task benchmarks?

The current evidence supports that narrower framing:

- `Phase 1`: forward-only learning is viable
- `Phase 2`: structural evolution is mixed on single-task benchmarks
- `Phase 3`: continual learning and reduced forgetting are the strongest positive results
- `Phase 4`: digits-scale perception tasks can benefit once learning dynamics are tuned

This repository is therefore best understood as a research codebase for continual learning and structure reuse, not as a general replacement for backpropagation.

The current architecture separates:

- `core/`: algorithm kernel, task generation, training state, shared runtime helpers
- `orchestrator/`: repeated experiment execution and result aggregation
- `analysis/`: post-hoc metrics
- `visualization/`: dashboards, topology animation, comparison views
- `results/`: generated artifacts

See [ARCHITECTURE.md](/F:/sel-lab/ARCHITECTURE.md) for the architectural rules.

See [RESEARCH_ROADMAP.md](/F:/sel-lab/RESEARCH_ROADMAP.md) for the current project direction.

See [AUTONOMY_PROTOCOL.md](/F:/sel-lab/AUTONOMY_PROTOCOL.md) for the default autonomous work loop and stop conditions.
See [FULL_AUTO_PROMPT.md](/F:/sel-lab/FULL_AUTO_PROMPT.md) for a repository-specific `codex --full-auto` prompt template.

See [REPORT_STATUS.md](/F:/sel-lab/REPORT_STATUS.md) for which reports are current versus archived.

## Main Entry Points

Run the core demo:

```powershell
python core/sel_core.py
```

Run repeated experiments:

```powershell
python orchestrator/experiment_runner.py
```

Run visualization demos:

```powershell
python visualization/main.py --mode full
```

Run smoke tests:

```powershell
python tests/smoke_test.py
```

Run Phase 4 mechanism analysis:

```powershell
python core/phase4_failure_analysis.py
```

Run Phase 3 structure-reuse ablations:

```powershell
python core/phase3_ablation.py
```

Run Phase 2 mechanism bench:

```powershell
python core/phase2_ablation.py
```

Run Phase 4 repair and expansion sweeps:

```powershell
python core/phase4_repair_experiment.py
python core/phase4_expansion_sweep.py
```

Generate the current unified report:

```powershell
python analysis/generate_unified_report.py
```

Analyze why true reuse still trails `adapt_only`:

```powershell
python analysis/phase3_gap_analysis.py
```

Sweep the specialist merge scale:

```powershell
python analysis/phase3_specialist_merge_sweep.py
```

```powershell
python analysis/phase3_suite_transfer.py
```

```powershell
python analysis/phase3_digits_transfer.py
```

## Current Research Position

Primary line:

- `Phase 3` is the main result path. This is where SEL currently has the clearest claim: structure reuse helps sequential learning and reduces forgetting.
- `Phase 4` is the main external validation path. It tests whether tuned forward-only evolution can help on non-trivial perception tasks.

Secondary line:

- `Phase 2` is now best treated as a mechanism-screening bench. Its role is to answer which reuse strategy is stable, not to prove that evolution always helps.

Exploratory side tracks:

- `core/multi_agent.py`
- `core/edge_optimization.py`
- `core/stability_test.py`

These are still useful, but they should not drive the main narrative until they are migrated onto the shared runtime/reporting path.

## Current Design Rules

- No hard-coded absolute result paths in new code
- Use `core.runtime` for result paths, JSON saving, train/test split, and history normalization
- Use `SELTrainer.metrics` as the typed in-memory record
- Convert histories at boundaries with `history_to_records()`
- Keep algorithm logic in `core`, not in visualization or orchestration scripts

## Near-Term Refactor Target

The immediate goal is not to add more branches of experimentation. It is to make the current main line trustworthy:

- keep `sel_core.py` as the canonical kernel
- migrate old standalone scripts onto the shared runtime layer
- remove hard-coded output paths and stale summary logic
- make report generation reflect current results rather than historical claims
- strengthen `Phase 3` and `Phase 4` as the core evidence chain
