# SEL: Structural Evolution Learning
## 项目纲要

### Project: Structural Evolution Learning (SEL)

### Goal:
Develop a learning system where intelligence emerges from structural evolution driven by local forward dynamics, not global backward optimization.

### Core Principles:
1. **No mandatory backpropagation** - Learning can occur without gradient descent
2. **Learning as dynamics, not minimization** - Focus on process, not just outcome
3. **Structure as a first-class entity** - Representation is central to learning
4. **Evaluation by adaptability, not accuracy** - Success measured by flexibility and reuse

### Research Hypothesis:
> Intelligence emerges when a structured system, under environmental pressure,
> evolves toward configurations that reduce internal tension while increasing
> external adaptability—all through local, forward-only dynamics.

### Phases:

#### **Phase 0: Analyze the structural role of backpropagation**
- **Objective**: Understand what backpropagation actually does structurally
- **Method**: Decompose backprop into structural operations
- **Outcome**: Identify replaceable components with forward alternatives
- **Status**: ✅ Conceptual analysis complete

#### **Phase 1: Demonstrate learning via forward feedback only**
- **Objective**: Prove learning can occur without backward passes
- **Method**: Local updates driven by output correctness feedback
- **Constraints**: No gradients, no global loss, local updates only
- **Status**: 🔄 Protocol defined, implementation in progress
- **Protocol**: `protocols/phase1_forward_feedback.yaml`

#### **Phase 2: Validate structural evolution as a learning accelerator**
- **Objective**: Show that structural changes speed up learning
- **Method**: Compare fixed-structure vs evolving-structure systems
- **Metrics**: Adaptation speed, structural reuse, learning cost
- **Status**: ⏳ Awaiting Phase 1 results

#### **Phase 3: Establish long-term evolutionary accumulation**
- **Objective**: Demonstrate knowledge accumulation across tasks
- **Method**: Sequential learning with structural inheritance
- **Metrics**: Transfer efficiency, structural preservation
- **Status**: ⏳ Long-term goal

### Success Criteria:
1. **Locality of updates** - Changes remain local to relevant structures
2. **Emergence of reusable structures** - Modular components form naturally
3. **Reduced learning cost over time** - Adaptation becomes more efficient
4. **Structural stability under perturbation** - System maintains coherence
5. **Forward-only dynamics** - No reliance on backward optimization

### Technical Approach:

#### **Representation: Graph-Based Structures**
- Nodes: Local state vectors with parameters
- Edges: Directional influence relationships
- Hierarchy: Nested structural organization

#### **Learning Dynamics: Forward Feedback**
- Signal sources: Local mismatch, output correctness
- Update form: Modulation strength, not gradient
- Scope: Local neighborhood only

#### **Structural Evolution: Adaptive Reconfiguration**
- Operations: Split, merge, rewire
- Triggers: High local tension, sustained mismatch
- Constraints: Energy minimization, coherence preservation

#### **Evaluation Framework:**
- Primary: Adaptability metrics
- Secondary: Structural metrics
- Tertiary: Efficiency metrics

### Philosophical Stance:

#### **Against:**
- Gradient descent as the only learning mechanism
- Global loss minimization as the primary goal
- Accuracy as the sole success metric
- Black-box optimization

#### **For:**
- Multiple paths to intelligence
- Process-oriented evaluation
- Explicit, observable structures
- Biologically plausible mechanisms

### Implementation Strategy:

#### **Bottom-Up Development:**
1. Simple environments (Phase 1)
2. Basic forward dynamics
3. Limited structural changes
4. Progressive complexity

#### **Agent-Centric Operation:**
- Humans define research questions
- Agents execute experiments
- Collaborative interpretation
- Iterative refinement

### Expected Contributions:

#### **Theoretical:**
- Formal framework for structure-first learning
- Taxonomy of non-backprop learning mechanisms
- Metrics for structural intelligence

#### **Practical:**
- Open-source experimental framework
- Reusable components for alternative AI
- Benchmarks for structural learning

#### **Philosophical:**
- Expanded view of what learning can be
- Challenge to gradient-centric AI
- Bridge to biological learning systems

### Project Status:
- **Framework**: ✅ Established (SEL-Lab)
- **Phase 1**: 🔄 Implementation in progress
- **Community**: 🌱 Early stage
- **Impact**: 🔮 Exploratory research

### Call to Action:
This is not just another AI project. This is an exploration of alternative foundations for intelligence. Join us in asking: *What if learning doesn't require backpropagation?*

---
*SEL-Lab: Where structure evolves, intelligence emerges.*