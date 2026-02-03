# SEL-Lab
Structural Evolution Learning Laboratory

## Project: Structural Evolution Learning (SEL)

### Goal:
Develop a learning system where intelligence emerges from structural evolution driven by local forward dynamics, not global backward optimization.

## Core Principles:
1. **No mandatory backpropagation** - Learning can occur without gradient descent
2. **Learning as dynamics, not minimization** - Focus on process, not just outcome  
3. **Structure as a first-class entity** - Representation is central to learning
4. **Evaluation by adaptability, not accuracy** - Success measured by flexibility and reuse

## Research Hypothesis:
> Learning is a dynamical process in which a structured system,
> under environmental constraints and feedback,
> evolves toward lower structural tension and higher adaptability.

This project is NOT:
- a deep learning framework
- a model zoo  
- a benchmark-chasing system

This project IS:
- a laboratory for testing learning-as-dynamics
- an agent-driven experimental system
- a framework for studying non-backprop, structure-first learning

---

## Research Phases

### Phase 0: Analyze the structural role of backpropagation ✅
**Status**: Conceptual analysis complete

### Phase 1: Demonstrate learning via forward feedback only 🔄
**Status**: Protocol defined, implementation in progress
**Protocol**: `protocols/phase1_forward_feedback.yaml`

### Phase 2: Validate structural evolution as a learning accelerator ⏳
**Status**: Awaiting Phase 1 results

### Phase 3: Establish long-term evolutionary accumulation ⏳
**Status**: Long-term goal

---

## Success Criteria:
1. **Locality of updates** - Changes remain local to relevant structures
2. **Emergence of reusable structures** - Modular components form naturally
3. **Reduced learning cost over time** - Adaptation becomes more efficient
4. **Structural stability under perturbation** - System maintains coherence
5. **Forward-only dynamics** - No reliance on backward optimization

---

## Conceptual Architecture
SEL-Lab follows a five-layer conceptual model:
1. **Environment** - Task and feedback interface
2. **Representation (Structure)** - Graph-based structural encoding
3. **Learning Dynamics** - Forward-only update rules
4. **Structural Change** - Adaptive reconfiguration
5. **Evolution Loop** - Iterative improvement cycle

Each layer is explicit, observable, and replaceable.

---

## Agent-Centric Design
SEL-Lab is designed to be operated by an agent (e.g. clawdbot).

### Human researchers define:
- Research goals and hypotheses
- Experimental constraints
- Protocol specifications

### Agents execute:
- Experiments under protocol constraints
- Parameter sweeps and variations
- Logging, metrics collection, and visualization
- Preliminary analysis and reporting

### Standard Experiment Flow:
```
[START]
↓
Read protocol.yaml
↓  
Initialize experiment environment
↓
Run N experiments under constraints
↓
Collect logs and metrics
↓
Analyze trends and failure modes
↓
Generate report.md
↓
Recommend next protocol changes (if allowed)
[END]
```

---

## Project Structure
```
sel-lab/
├── README.md                    # This file
├── PROJECT_MANIFESTO.md         # Complete project manifesto
├── RESEARCH_ROADMAP.md          # Detailed research roadmap
├── SUCCESS_CRITERIA.md          # Success metrics and evaluation
├── EXPERIMENT_FLOW.md           # Standard experiment workflow
├── protocols/                   # Research protocols (human-authored)
│   └── phase1_forward_feedback.yaml  # Phase 1 protocol
├── orchestrator/                # Agent-driven experiment runners
│   └── experiment_runner.py     # Main experiment orchestrator
├── core/                        # World, structure, dynamics, evolution
│   ├── __init__.py
│   └── environment.py           # Environment implementations
├── analysis/                    # Metrics and visualization
│   ├── __init__.py
│   └── metrics.py              # Metrics calculation
├── logs/                        # Experiment logs
├── results/                     # Experiment results
├── reports/                     # Generated reports
└── test_experiment_flow.py      # Test script
```

---

## Evaluation Philosophy
Success is NOT defined by peak performance.

### Primary evaluation metrics include:
- Adaptation speed to new environments
- Structural stability under change  
- Reuse and modularity of substructures
- Locality of updates
- Long-term evolutionary trends
- Learning cost reduction over time

### Against:
- Gradient descent as the only learning mechanism
- Global loss minimization as the primary goal
- Accuracy as the sole success metric
- Black-box optimization

### For:
- Multiple paths to intelligence
- Process-oriented evaluation
- Explicit, observable structures
- Biologically plausible mechanisms

---

## Getting Started

### 1. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Test the experiment flow:
```bash
python test_experiment_flow.py
```

### 3. Run Phase 1 experiment:
```bash
python orchestrator/experiment_runner.py protocols/phase1_forward_feedback.yaml
```

### 4. Explore results:
- Check `logs/` directory for experiment logs
- Check `results/` for aggregated results
- Check `reports/` for generated reports

---

## Current Status
- **Framework**: ✅ Established and operational
- **Phase 1**: 🔄 Implementation in progress
- **Code**: ✅ Basic framework complete
- **Experiments**: ⏳ Ready for execution

### Immediate next steps:
1. Complete core module implementations
2. Run Phase 1 experiments
3. Analyze results and generate reports
4. Decide Phase 2 direction based on evidence

---

## Contributing
This is exploratory research into alternative foundations for intelligence. We welcome:
- Protocol designs for new experiments
- Implementations of alternative learning dynamics
- Analysis tools and visualization
- Documentation and tutorials

Join us in asking: *What if learning doesn't require backpropagation?*

---
*SEL-Lab: Where structure evolves, intelligence emerges.*