# SEL-Lab Project Reframe

## New Goal

SEL-Lab is being refocused from a broad "prove a new universal learning paradigm" project into a narrower and stronger research program:

Characterize when forward-only structure reuse improves continual learning, and isolate the mechanisms responsible for that gain.

## What Changes

Old framing:

- prove structural evolution is generally beneficial
- show SEL as a broad replacement for backpropagation
- expand into many side experiments in parallel

New framing:

- treat continual learning as the main target
- treat structure reuse as the main mechanism
- use single-task benchmarks as boundary conditions, not the main story
- keep perception tasks as validation, not as a shortcut to generality

## Research Tracks

### Track A: Main Result Track

- `core/phase3.py`
- future `Phase 3` ablations

Primary questions:

- which reuse policy best reduces forgetting?
- when should cloning happen?
- when should reused structure stay fixed versus continue adapting?

### Track B: Validation Track

- `core/phase4_mnist.py`
- `core/phase4_real.py`
- `core/phase4_failure_analysis.py`

Primary questions:

- do the same reuse mechanisms help outside toy sequential tasks?
- which optimization dynamics make forward-only evolution viable at all?

### Track C: Mechanism Bench

- `core/phase2.py`
- `core/phase2_hard.py`
- `core/phase2_improved.py`

Primary questions:

- which reuse policy works best under controlled settings?
- which policies fail, and why?

### Track D: Exploratory Side Work

- `core/multi_agent.py`
- `core/edge_optimization.py`
- `core/stability_test.py`

These tracks stay in the repository, but they are no longer first-class evidence for the project thesis.

## Mechanism Matrix

The long-term organization should move from phase-centric comparisons to mechanism-centric comparisons.

Core mechanism families:

- fixed structure
- random growth
- best-unit cloning
- low-tension cloning
- clone then freeze
- clone then adapt
- adapt without cloning

Core evaluation worlds:

- sequential tasks
- single-task controls
- perception-task validation

## Immediate Milestones

1. Make the current evidence chain trustworthy.

- shared runtime paths
- consistent analysis logic
- one current reporting path

2. Strengthen the `Phase 3` story.

- forgetting metrics
- transfer metrics
- reuse-policy ablations

3. Make `Phase 2` answer mechanism questions.

- stop treating it as a universal pass/fail for evolution
- use it to compare reuse rules directly

4. Use `Phase 4` conservatively.

- validate claims only when benchmark, repair, and analysis all align

## Non-Goals For Now

- do not broaden into more side experiments
- do not claim universal superiority on static benchmarks
- do not let stale reports define project conclusions

## Working Statement

SEL-Lab is a research codebase for forward-only continual learning via reusable structure.
