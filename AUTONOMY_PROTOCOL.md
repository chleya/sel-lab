# SEL-Lab Autonomy Protocol

## Purpose

This protocol defines how SEL-Lab should be advanced without requiring step-by-step human confirmation.

The goal is not infinite motion. The goal is bounded autonomous progress with explicit stop conditions.

## North Star

The project should optimize for this research question:

When does forward-only structure reuse improve continual learning, and which mechanisms are responsible for that gain?

## Current Main Objective

The current implementation objective is narrower, but should remain answer-oriented rather than mechanism-locked:

Improve the best available structure-reuse approach on `digits_pairs` and close the remaining gap to the fixed baseline without giving back its gain over `adapt_only`.

The current leading candidate is `task_specialist_clone_limited_merge`, but it is not a fixed truth source.
If a stronger mechanism appears, or a strong negative result invalidates it, autonomous work may replace it as the main candidate as long as the reason is recorded in results, reports, and handoff notes.

## Priority Order

When choosing work autonomously, use this order:

1. Strengthen the `Phase 3` evidence chain
2. Extend or validate the `Phase 4` line when it materially supports the main thesis
3. Use `Phase 2` only as a mechanism screen
4. Treat all other scripts as exploratory unless they are migrated into the main evidence path

## Default Work Loop

Autonomous work should follow this loop:

1. Read the latest `HANDOFF.md`, `UNIFIED_REPORT.md`, and current result artifacts
2. Select the smallest high-value next step on the main path
3. Implement the change
4. Run the relevant verification or experiments
5. Update result artifacts, reports, and handoff files
6. Classify the outcome

Outcome classes:

- `positive`: strengthens the main claim
- `negative_but_useful`: closes off a direction or clarifies a boundary
- `neutral`: little or no new information

7. Continue to the next loop unless a stop condition is hit

The important constraint is:

- keep the question stable
- allow the answer to change

## Allowed Autonomous Scope

Without asking first, autonomous work may:

- add or refine experiments on the current main path
- adjust local analysis scripts and reports
- update smoke tests
- update handoff and memory files
- refine synthetic or benchmark task setup if it directly supports the current objective
- replace the current leading mechanism if evidence shows a better one or invalidates the old one

Autonomous work should not:

- redefine the project thesis silently
- replace the main benchmark family without reporting
- revive archived reports as active truth sources
- do unrelated side-track work while the main bottleneck is unresolved

## Stop Conditions

Autonomous work must stop and report when any of these occur:

1. The main conclusion changes
2. A strong positive result upgrades the project narrative
3. A strong negative result invalidates the current direction
4. The current leading mechanism is no longer the best-supported candidate
5. Two to three consecutive loops produce no meaningful information gain
6. A change requires major architecture or benchmark redesign
7. Tests fail or results conflict
8. Report files and result files disagree
9. The next useful step is ambiguous enough that multiple directions are equally plausible

## Reporting Requirements

Each autonomous loop should leave behind at least one durable artifact:

- code change
- new result file
- updated report
- updated handoff or memory note

At every stop condition report, include:

- what was tried
- what changed
- what the result means for the main objective
- the next one or two best options

## Escalation Rule

If a result is strong enough to change the project story, prefer stopping early and reporting rather than silently continuing.

Examples:

- a reuse policy finally beats fixed on `digits_pairs`
- the current main policy collapses on a richer benchmark
- `Phase 4` overtakes `Phase 3` as the strongest line
- a newly tested reuse mechanism clearly overtakes `task_specialist_clone_limited_merge`

## Current Concrete Guidance

Right now the most valuable autonomous work is:

1. Use `digits_pairs` as the boundary benchmark for `Phase 3`
2. Target retention-oriented changes that preserve current-task gains, but stay open to stronger routing or role-separation ideas
3. Compare new reuse variants primarily against:
   - `adapt_only`
   - the current leading reuse candidate
   - fixed baseline

## Success Standard For The Current Stage

The current stage is a success if at least one of these happens:

- the current leading reuse candidate beats fixed on `digits_pairs`
- the gap to fixed shrinks materially while preserving its gain over `adapt_only`
- a clear mechanism explanation emerges for why the fixed gap remains

## Failure Standard For The Current Stage

The current stage should be reconsidered if:

- repeated retention tweaks do not improve the `digits_pairs` boundary
- the main effect only survives on synthetic suites
- benchmark expansion repeatedly removes the specialist-merge advantage
