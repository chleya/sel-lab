# Report Status

## Current Sources Of Truth

- [UNIFIED_REPORT.md](/F:/sel-lab/UNIFIED_REPORT.md): current experiment summary
- [README.md](/F:/sel-lab/README.md): repository framing and entrypoints
- [PROJECT_REFRAME.md](/F:/sel-lab/PROJECT_REFRAME.md): research goal reset
- [RESEARCH_ROADMAP.md](/F:/sel-lab/RESEARCH_ROADMAP.md): near-term direction
- [HANDOFF.md](/F:/sel-lab/HANDOFF.md): current implementation state and risks
- [analysis/generate_unified_report.py](/F:/sel-lab/analysis/generate_unified_report.py): canonical report builder and canonical result manifest

## Canonical Result Files

`UNIFIED_REPORT.md` is generated from the explicit `CANONICAL_RESULT_FILES` manifest in [analysis/generate_unified_report.py](/F:/sel-lab/analysis/generate_unified_report.py).

Those files are the current canonical evidence chain. Sweep artifacts, smoke outputs, and exploratory one-offs in [results](/F:/sel-lab/results) should not be treated as canonical unless they are added to that manifest.

Canonical-first result loading now goes through shared helpers in [core/runtime.py](/F:/sel-lab/core/runtime.py), so analysis scripts can prefer `results/canonical/` while still reading legacy root-level files during migration.

## Result Layers

- [results/canonical](/F:/sel-lab/results/canonical): canonical generated outputs such as `unified_summary.json`
- [results/smoke](/F:/sel-lab/results/smoke): smoke-test artifacts
- [results/exploratory](/F:/sel-lab/results/exploratory): exploratory and quick-script outputs that should not be cited as canonical evidence
- [results](/F:/sel-lab/results): legacy root-level artifacts and existing canonical inputs that have not yet been migrated

Practical rule:

- New canonical outputs should prefer `results/canonical`
- New smoke outputs should prefer `results/smoke`
- New exploratory or quick-script outputs should prefer `results/exploratory`
- Existing root-level files remain readable for backward compatibility until migration is complete

## Archived Documents

- [FINAL_REPORT.md](/F:/sel-lab/FINAL_REPORT.md)
- [PROJECT_SUMMARY.md](/F:/sel-lab/PROJECT_SUMMARY.md)
- [REVIEW.md](/F:/sel-lab/REVIEW.md)

These archived files remain in the repository only for history. They should not be cited as current evidence.

## Practical Rule

If a report conflicts with `UNIFIED_REPORT.md`, trust `UNIFIED_REPORT.md`.
