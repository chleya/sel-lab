# SEL-Lab Research Roadmap

## Thesis

SEL-Lab should be framed as a research project about forward-only learning with reusable structure under sequential pressure.

The strongest current claim is not:

"SEL beats fixed networks everywhere."

The strongest current claim is:

"SEL is most promising when tasks arrive sequentially and previously learned structure can be reused."

## Main Question

When does structure reuse help a forward-only learner more than a fixed architecture?

## Current Interpretation

- `Phase 1` establishes feasibility for forward-only learning.
- `Phase 2` shows that naive evolution is not enough and that reuse policies matter.
- `Phase 3` is the core positive result: evolution helps continual learning and reduces forgetting.
- `Phase 4` is the main validation path outside toy sequential tasks, but its analysis chain still needs tightening.

## Keep / Demote

Keep as main line:

- `core/sel_core.py`
- `core/phase3.py`
- `core/phase4_mnist.py`
- `core/phase4_real.py`
- `analysis/generate_unified_report.py`

Keep as mechanism bench:

- `core/phase2.py`
- `core/phase2_hard.py`
- `core/phase2_improved.py`

Demote to exploratory side work:

- `core/multi_agent.py`
- `core/edge_optimization.py`
- `core/stability_test.py`

## Highest-Value Next Steps

1. Make the evidence chain trustworthy.

- Remove hard-coded result paths.
- Fix Phase 4 analysis so diagnostics use the same dynamics as training.
- Keep one current report path and stop relying on stale static summaries.

2. Strengthen `Phase 3`.

- Add explicit forgetting and transfer metrics.
- Add ablations for clone policy, mutation policy, and freeze-vs-adapt behavior.
- Treat `Phase 3` as the primary result section of the project.

3. Reposition `Phase 2`.

- Use it to test reuse mechanisms rather than to defend evolution as a universal win.
- Compare random growth, best-unit cloning, low-tension cloning, and constrained adaptation.

4. Use `Phase 4` as a validation gate, not a narrative shortcut.

- Only claim perception-task success when analysis, repair sweeps, and benchmark runs agree.
- Prefer a small number of reproducible digits/MNIST-scale results over many weak extensions.

## What To Avoid

- Do not expand into more side experiments before cleaning the current main line.
- Do not present all positive results as equally strong.
- Do not let static reports outrun the actual result files.
- Do not treat single-task wins as the main justification for SEL.

## Working Claim

Forward-only structural adaptation appears most defensible as a continual-learning and structure-reuse framework.
