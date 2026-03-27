# SEL-Lab Architecture

## Research Posture

SEL-Lab is now organized around a narrower claim:

Forward-only structural adaptation is most promising as a continual-learning and structure-reuse framework, not as a universal replacement for fixed architectures or backpropagation.

That posture should shape both code organization and experimental priority.

Primary research line:

- `Phase 3`: continual learning, transfer, and reduced forgetting
- `Phase 4`: validation on non-trivial perception tasks once dynamics are tuned

Secondary research line:

- `Phase 2`: mechanism screening for structure reuse policies

Exploratory side work:

- `multi_agent.py`
- `edge_optimization.py`
- `stability_test.py`

Exploratory scripts may remain in the repository, but they should not define the main narrative or drive the top-level reporting path.

## Layers

`core/`
The algorithmic substrate. This layer owns learning rules, task generation, training state, and shared runtime helpers. It should not know about visualization widgets or hard-coded result paths.

`orchestrator/`
Top-level experiment composition. This layer chooses configs, repeats runs, aggregates summaries, and persists outputs. It should call `core`, not reimplement training logic.

`analysis/`
Post-hoc metrics and report calculations. This layer consumes experiment outputs and histories.

`visualization/`
Rendering and exploration only. It consumes normalized history records and saved artifacts; it should not need to know whether histories came from dataclasses or dicts.

`results/`
Generated artifacts only. No source logic belongs here.

## Shared Conventions

- Use `core.runtime` for project-root resolution, result-file paths, JSON saving, and train/test splitting.
- Use `core.runtime.history_to_records()` when a component needs plain dictionaries.
- Use `SELTrainer.metrics` for typed in-memory history and `history_to_records()` for serialization boundaries.
- Avoid `sys.path.insert()` in new code unless a script absolutely must be executable as a loose file. Prefer module imports from the project root.
- Avoid hard-coded absolute paths such as `F:/skill/...`.

## Refactor Direction

1. Keep `sel_core.py` as the canonical algorithm kernel.
2. Treat `phase3.py` and `phase4_*` as the main evidence chain.
3. Reposition `phase2*` scripts as a mechanism bench for reuse policies.
4. Migrate old standalone scripts to the shared runtime helpers before extending them further.
5. Move repeated experiment patterns into orchestrator-level helpers instead of copying loops across files.
6. Treat visualization as a consumer of normalized history records, not trainer internals.
7. Add smoke coverage whenever a new entrypoint or analysis path is introduced.

## Current Empirical Status

- `sel_core.py` is the canonical algorithm kernel and currently the healthiest experimental path.
- Phase 2 scripts are structurally unified, but single-task knowledge reuse remains unstable and should be treated as mechanism exploration.
- Phase 3 currently shows the clearest positive signal: evolution helps incremental learning and reduces forgetting.
- Phase 4 now provides a useful validation path, but its conclusions should only be trusted when training runs, repair sweeps, and analysis agree.
