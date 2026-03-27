# -*- coding: utf-8 -*-
"""
SEL-L
Multiple agents collaboratingab Multi-Agent Evolution and sharing knowledge
"""

import numpy as np
from dataclasses import dataclass
import json


@dataclass
class MultiAgentConfig:
    """Multi-agent configuration"""
    num_agents: int = 4          # Number of agents
    input_size: int = 4
    hidden_size: int = 8
    output_size: int = 2
    learning_rate: float = 0.03
    max_units: int = 4           # Per agent
    epochs: int = 50
    sharing_interval: int = 10   # Share knowledge every N epochs


class DFMAgent:
    """Single agent with DFA and evolution"""
    
    def __init__(self, agent_id, config, seed=None):
        if seed:
            np.random.seed(seed)
        self.agent_id = agent_id
        self.config = config
        self.units = []
        self.add_unit()
        
        # Performance tracking
        self.accuracy_history = []
        self.knowledge_shared = 0
        self.knowledge_received = 0
    
    def add_unit(self, clone_from=-1, source_agent=None):
        """Add unit with optional cloning from another agent"""
        scale1 = np.sqrt(2.0 / self.config.input_size) * 0.5
        scale2 = np.sqrt(2.0 / self.config.hidden_size) * 0.5
        
        if clone_from >= 0 and clone_from < len(self.units):
            source = self.units[clone_from]
            new_unit = {
                'W1': source['W1'] + np.random.randn(self.config.input_size, self.config.hidden_size) * 0.05,
                'W2': source['W2'] + np.random.randn(self.config.hidden_size, self.config.output_size) * 0.05,
                'feedback': source['feedback'] + np.random.randn(self.config.output_size, self.config.hidden_size) * 0.02,
                'tension': source['tension'],
                'age': 0,
                'active': True
            }
        elif source_agent is not None:
            # Clone from another agent's best unit
            source_agent_units = [u for u in source_agent.units if u['active']]
            if source_agent_units:
                best = min(source_agent_units, key=lambda u: u['tension'])
                new_unit = {
                    'W1': best['W1'] + np.random.randn(self.config.input_size, self.config.hidden_size) * 0.1,
                    'W2': best['W2'] + np.random.randn(self.config.hidden_size, self.config.output_size) * 0.1,
                    'feedback': best['feedback'] + np.random.randn(self.config.output_size, self.config.hidden_size) * 0.05,
                    'tension': 0.5,
                    'age': 0,
                    'active': True
                }
                self.knowledge_received += 1
            else:
                return -1
        else:
            new_unit = {
                'W1': np.random.randn(self.config.input_size, self.config.hidden_size) * scale1,
                'W2': np.random.randn(self.config.hidden_size, self.config.output_size) * scale2,
                'feedback': np.random.randn(self.config.output_size, self.config.hidden_size) * 0.05,
                'tension': 0.5,
                'age': 0,
                'active': True
            }
        
        self.units.append(new_unit)
        return len(self.units) - 1
    
    def forward(self, x):
        outputs = []
        for u in self.units:
            if u['active']:
                h = np.tanh(x @ u['W1'])
                out = h @ u['W2']
                outputs.append(out)
        return np.mean(outputs, axis=0) if outputs else np.zeros(self.config.output_size)
    
    def learn(self, x, target):
        out = self.forward(x)
        error = target - out
        
        for u in self.units:
            if u['active']:
                h = np.tanh(x @ u['W1'])
                fb_error = error @ u['feedback']
                lr = self.config.learning_rate * np.exp(-u['age'] * 0.01)
                u['W2'] += lr * np.outer(h, error)
                u['W1'] += lr * np.outer(x, fb_error)
                np.clip(u['W1'], -2, 2, out=u['W1'])
                np.clip(u['W2'], -2, 2, out=u['W2'])
                u['age'] += 1
                u['tension'] = 0.9 * u['tension'] + 0.1 * np.mean(error ** 2)
        
        return np.mean(error ** 2)
    
    def evolve(self):
        """Local evolution"""
        changes = []
        active = [u for u in self.units if u['active']]
        if not active:
            return changes
        
        avg_tension = np.mean([u['tension'] for u in active])
        
        if avg_tension > 0.2 and len(self.units) < self.config.max_units:
            best_idx = np.argmin([u['tension'] for u in active])
            self.add_unit(clone_from=best_idx)
            changes.append('add')
        
        return changes
    
    def predict(self, x):
        return int(np.argmax(self.forward(x)))
    
    def accuracy(self, X, y):
        correct = 0
        for i in range(len(X)):
            if self.predict(X[i]) == int(np.argmax(y[i])):
                correct += 1
        return correct / len(X)
    
    def get_avg_tension(self):
        active = [u for u in self.units if u['active']]
        return np.mean([u['tension'] for u in active]) if active else 1.0


class MultiAgentSystem:
    """System of collaborating agents"""
    
    def __init__(self, config, seed=None):
        if seed:
            np.random.seed(seed)
        self.config = config
        self.agents = []
        
        # Initialize agents
        for i in range(config.num_agents):
            agent = DFMAgent(i, config, seed=seed * 100 + i)
            self.agents.append(agent)
    
    def share_knowledge(self):
        """Knowledge sharing between agents"""
        sharing_events = []
        
        # Find best and worst performing agents
        tensions = [(i, a.get_avg_tension()) for i, a in enumerate(self.agents)]
        tensions.sort(key=lambda x: x[1])  # Low tension = better
        
        # Best agent shares with worst
        if len(tensions) >= 2:
            best_idx = tensions[0][0]
            worst_idx = tensions[-1][0]
            
            best_agent = self.agents[best_idx]
            worst_agent = self.agents[worst_idx]
            
            # Worst agent receives unit from best
            received = worst_agent.add_unit(source_agent=best_agent)
            
            if received >= 0:
                sharing_events.append({
                    'from': best_idx,
                    'to': worst_idx,
                    'type': 'unit_transfer'
                })
                best_agent.knowledge_shared += 1
        
        return sharing_events
    
    def collective_predict(self, x):
        """Majority vote across agents"""
        votes = [a.predict(x) for a in self.agents]
        return max(set(votes), key=votes.count)
    
    def collective_accuracy(self, X, y):
        """Evaluate collective intelligence"""
        correct = 0
        for i in range(len(X)):
            if self.collective_predict(X[i]) == int(np.argmax(y[i])):
                correct += 1
        return correct / len(X)
    
    def get_stats(self):
        """Get system statistics"""
        total_units = sum(len(a.units) for a in self.agents)
        avg_units = total_units / len(self.agents)
        total_shared = sum(a.knowledge_shared for a in self.agents)
        total_received = sum(a.knowledge_received for a in self.agents)
        
        return {
            'num_agents': len(self.agents),
            'total_units': total_units,
            'avg_units_per_agent': avg_units,
            'knowledge_shared': total_shared,
            'knowledge_received': total_received
        }


def create_task(task_id=0, seed=42):
    """Create learning task"""
    np.random.seed(seed + task_id)
    X = np.random.randn(100, 4) * 2
    y = np.zeros((100, 2))
    
    for i in range(100):
        if X[i, 0] + X[i, 1] > 0:
            y[i, 0] = 1
        else:
            y[i, 1] = 1
    
    return X, y


def run_multi_agent():
    """Run multi-agent experiment"""
    print("\n" + "=" * 60)
    print("Multi-Agent Evolution Experiment")
    print("=" * 60)
    
    config = MultiAgentConfig()
    X, y = create_task()
    X_train, y_train = X[:50], y[:50]
    X_test, y_test = X[50:], y[50:]
    
    print(f"\nTask: {config.num_agents} agents collaborating")
    print(f"Knowledge sharing every {config.sharing_interval} epochs")
    
    # Initialize systems
    multi_agent = MultiAgentSystem(config, seed=42)
    single_agent = DFMAgent(0, config, seed=42)  # Baseline: single agent
    
    print(f"\nInitialized {config.num_agents} agents")
    
    results = {
        'individual': [],
        'collective': [],
        'single': []
    }
    
    for epoch in range(config.epochs):
        # Train all agents
        for i in range(len(X_train)):
            # Multi-agent training
            for agent in multi_agent.agents:
                agent.learn(X_train[i], y_train[i])
            
            # Single agent training
            single_agent.learn(X_train[i], y_train[i])
        
        # Knowledge sharing
        if (epoch + 1) % config.sharing_interval == 0:
            sharing = multi_agent.share_knowledge()
            if sharing:
                print(f"  Epoch {epoch+1}: Knowledge shared {len(sharing)} times")
        
        # Local evolution
        for agent in multi_agent.agents:
            agent.evolve()
        single_agent.evolve()
        
        # Evaluation
        if (epoch + 1) % 10 == 0:
            # Individual accuracy (average across agents)
            ind_accs = [a.accuracy(X_test, y_test) for a in multi_agent.agents]
            ind_acc = np.mean(ind_accs)
            
            # Collective accuracy
            col_acc = multi_agent.collective_accuracy(X_test, y_test)
            
            # Single agent accuracy
            sng_acc = single_agent.accuracy(X_test, y_test)
            
            results['individual'].append(ind_acc)
            results['collective'].append(col_acc)
            results['single'].append(sng_acc)
            
            print(f"Epoch {epoch+1}: Individual={ind_acc:.0%}, Collective={col_acc:.0%}, Single={sng_acc:.0%}")
    
    # Final evaluation
    final_ind = np.mean(results['individual'][-3:])
    final_col = np.mean(results['collective'][-3:])
    final_sng = np.mean(results['single'][-3:])
    
    stats = multi_agent.get_stats()
    
    print(f"\n" + "=" * 60)
    print("Multi-Agent Results Summary")
    print("=" * 60)
    
    print(f"\nFinal Accuracy (last 3 evaluations):")
    print(f"  Single Agent:    {final_sng:.1%}")
    print(f"  Multi-Agent (individual avg): {final_ind:.1%}")
    print(f"  Multi-Agent (collective):     {final_col:.1%}")
    
    print(f"\nKnowledge Sharing:")
    print(f"  Total units created: {stats['total_units']}")
    print(f"  Units shared: {stats['knowledge_shared']}")
    print(f"  Units received: {stats['knowledge_received']}")
    
    advantage_ind = final_ind - final_sng
    advantage_col = final_col - final_sng
    
    print(f"\nAdvantages vs Single Agent:")
    print(f"  Individual avg: {advantage_ind:+.1%}")
    print(f"  Collective: {advantage_col:+.1%}")
    
    success = advantage_ind > 0 or advantage_col > 0
    print(f"\n{'[SUCCESS] Multi-agent collaboration helps!' if success else '[NEUTRAL] No clear advantage'}")
    
    # Save results
    with open("F:/skill/sel-lab/results/multi_agent_results.json", 'w', encoding='utf-8') as f:
        json.dump({
            'config': {
                'num_agents': config.num_agents,
                'sharing_interval': config.sharing_interval
            },
            'single_final': float(final_sng),
            'individual_final': float(final_ind),
            'collective_final': float(final_col),
            'advantage_individual': float(advantage_ind),
            'advantage_collective': float(advantage_col),
            'knowledge_shared': stats['knowledge_shared'],
            'decision': 'success' if success else 'neutral'
        }, f, indent=2)
    
    print(f"\nResults saved: results/multi_agent_results.json")


if __name__ == "__main__":
    run_multi_agent()
