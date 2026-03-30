# SEL-Lab Research Roadmap

## Thesis

SEL-Lab is a research project about **forward-only learning with reusable structure under sequential pressure**.

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

## Research Lines

### Core Evidence Chain (Main Line)

These form the primary scientific contribution:

- `core/sel_core.py` - Canonical algorithm kernel
- `core/phase3.py` - Continual learning and structure reuse
- `core/phase4_mnist.py` - Perception task validation
- `core/phase4_real.py` - Real-world validation
- `analysis/generate_unified_report.py` - Unified reporting

### Mechanism Bench (Secondary Line)

Used to test reuse mechanisms:

- `core/phase2.py`
- `core/phase2_hard.py`
- `core/phase2_improved.py`

### Exploration Directions (Frontier Research)

**Active exploration areas:**

1. **Structure-Intelligence Relationship** (`exploration/structure_insights.py`)
   - ✅ **Key Finding**: Structure complexity ↔ Intelligence capability (negative correlation -0.51)
   - ✅ **Practical Insight**: Optimal range is 3-5 modules, not "more is better"
   - **Next Step**: Validate on real Phase 3/4 tasks
   - **Status**: Core insight preserved, implementation simplified

2. **Multi-Agent Collaboration** (`core/multi_agent.py`)
   - Multiple SEL agents learning in parallel
   - Inter-agent knowledge sharing
   - Collaborative problem solving
   - **Status**: Framework ready, needs validation

3. **Edge Deployment** (`core/edge_optimization.py`)
   - Model compression and quantization
   - Memory and compute constraints
   - Real-time inference optimization
   - **Status**: Framework ready

**Archived explorations** (moved to `archive/exploration_archive/`):
- Neuromorphic computing (high theory, low practical value)
- Meta-learning (interesting but not integrated)
- Foundations research (information theory, complex systems, energy functions)
  - **Reason**: High maintenance cost, low output ratio
  - **Decision**: Keep core insights, remove complex implementations

## Highest-Value Next Steps

### Core Line (Stability)

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

### Exploration Line (Innovation)

5. **Neuromorphic Integration**
   - Validate SNN-based SEL on temporal tasks
   - Explore spike-time-dependent plasticity (STDP) for learning
   - Test on neuromorphic hardware simulators

6. **Meta-Learning Enhancement**
   - Extend to more diverse task distributions
   - Explore architecture search via meta-learning
   - Connect to Phase 3's continual learning

7. **Multi-Agent Scaling**
   - Test with larger agent populations
   - Explore different knowledge sharing topologies
   - Study emergent specialization

8. **Edge Optimization**
   - Quantization-aware evolution
   - Dynamic architecture pruning
   - Latency-constrained learning

9. **Foundations Research**
   - Develop information-theoretic metrics for structure evolution
   - Model SEL as a complex adaptive system
   - Apply Bayesian optimization to SEL hyperparameters
   - Explore causal structure discovery in evolving networks
   - Develop energy-based objective functions for SEL

## Running Exploration Experiments

```bash
# Run all exploration directions (quick mode)
python exploration/exploration_runner.py --quick

# Run specific directions
python exploration/exploration_runner.py --directions neuromorphic meta_learning

# Full exploration run
python exploration/exploration_runner.py

# Run individual modules
python exploration/neuromorphic.py
python exploration/meta_learning.py
```

## What To Avoid

- Do not expand into more side experiments before cleaning the current main line.
- Do not present all positive results as equally strong.
- Do not let static reports outrun the actual result files.
- Do not treat single-task wins as the main justification for SEL.

## Working Claim

Forward-only structural adaptation appears most defensible as a continual-learning and structure-reuse framework.

The exploration directions represent **frontier research** that may or may not integrate into the core claim. They are kept active but separate from the main evidence chain until they produce reproducible, significant results.
