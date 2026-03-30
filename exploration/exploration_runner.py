# -*- coding: utf-8 -*-
"""
SEL-Lab Exploration Runner
探索性实验统一运行框架

运行所有前沿探索方向:
1. 多智能体协作 (multi_agent)
2. 边缘优化 (edge_optimization)
3. 神经形态计算 (neuromorphic)
4. 元学习 (meta_learning)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime import resolve_results_path, save_json


def run_exploration(
    directions: Optional[List[str]] = None,
    quick: bool = False,
    verbose: bool = True
) -> Dict:
    """
    运行探索性实验
    
    Args:
        directions: 要运行的探索方向列表，None 表示运行所有
        quick: 是否使用快速模式（减少迭代次数）
        verbose: 是否打印详细信息
    
    Returns:
        所有实验结果的字典
    """
    
    all_directions = {
        'neuromorphic': '神经形态计算与脉冲神经网络',
        'meta_learning': '元学习与自适应机制',
        'multi_agent': '多智能体协作进化',
        'edge': '边缘部署优化',
        'foundations': '基础研究深度探索',
    }
    
    directions_to_run = directions or list(all_directions.keys())
    
    print("=" * 70)
    print(" SEL-Lab Exploration Framework ")
    print("=" * 70)
    print(f"\n将运行以下探索方向:")
    for d in directions_to_run:
        print(f"  - {d}: {all_directions.get(d, '未知')}")
    print()
    
    results = {}
    
    # 1. 神经形态计算
    if 'neuromorphic' in directions_to_run:
        print("\n" + "=" * 70)
        print("运行: 神经形态计算探索")
        print("=" * 70)
        try:
            from exploration.neuromorphic import run_neuromorphic_exploration
            
            epochs = 20 if quick else 50
            runs = 2 if quick else 3
            
            results['neuromorphic'] = run_neuromorphic_exploration(
                runs=runs,
                epochs=epochs,
                verbose=verbose
            )
            print("✅ 神经形态计算探索完成")
        except Exception as e:
            print(f"❌ 神经形态计算探索失败: {e}")
            results['neuromorphic'] = {'error': str(e)}
    
    # 2. 元学习
    if 'meta_learning' in directions_to_run:
        print("\n" + "=" * 70)
        print("运行: 元学习探索")
        print("=" * 70)
        try:
            from exploration.meta_learning import run_meta_learning_exploration
            
            iterations = 50 if quick else 100
            
            results['meta_learning'] = run_meta_learning_exploration(
                meta_iterations=iterations,
                eval_interval=10,
                verbose=verbose
            )
            print("✅ 元学习探索完成")
        except Exception as e:
            print(f"❌ 元学习探索失败: {e}")
            results['meta_learning'] = {'error': str(e)}
    
    # 3. 多智能体
    if 'multi_agent' in directions_to_run:
        print("\n" + "=" * 70)
        print("运行: 多智能体协作探索")
        print("=" * 70)
        try:
            # 导入并运行多智能体实验
            from core.multi_agent import MultiAgentConfig, DFMAgent
            
            print("多智能体实验框架已加载")
            print("注意: 完整的多智能体实验需要额外的配置")
            
            results['multi_agent'] = {
                'status': 'framework_loaded',
                'message': '多智能体框架已准备就绪，可运行详细实验'
            }
        except Exception as e:
            print(f"❌ 多智能体探索失败: {e}")
            results['multi_agent'] = {'error': str(e)}
    
    # 4. 边缘优化
    if 'edge' in directions_to_run:
        print("\n" + "=" * 70)
        print("运行: 边缘部署优化探索")
        print("=" * 70)
        try:
            from core.edge_optimization import EdgeConfig, EdgeDFA, EdgeEvolving
            
            print("边缘优化实验框架已加载")
            print("注意: 完整的边缘优化实验需要额外的配置")
            
            results['edge'] = {
                'status': 'framework_loaded',
                'message': '边缘优化框架已准备就绪，可运行详细实验'
            }
        except Exception as e:
            print(f"❌ 边缘优化探索失败: {e}")
            results['edge'] = {'error': str(e)}
    
    # 5. 基础研究
    if 'foundations' in directions_to_run:
        print("\n" + "=" * 70)
        print("运行: 基础研究深度探索")
        print("=" * 70)
        try:
            from exploration.foundations_research import run_foundations_exploration
            
            results['foundations'] = run_foundations_exploration(
                verbose=verbose
            )
            print("✅ 基础研究探索完成")
        except Exception as e:
            print(f"❌ 基础研究探索失败: {e}")
            results['foundations'] = {'error': str(e)}
    
    # 保存汇总结果
    print("\n" + "=" * 70)
    print("探索实验汇总")
    print("=" * 70)
    
    for direction, result in results.items():
        if 'error' in result:
            print(f"  {direction}: ❌ 失败")
        elif 'summary' in result:
            print(f"  {direction}: ✅ 成功")
        else:
            print(f"  {direction}: ⚠️ 部分完成")
    
    summary_path = resolve_results_path("exploration_summary.json")
    save_json(results, summary_path)
    print(f"\n汇总结果保存至: {summary_path}")
    
    return results


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='SEL-Lab 探索性实验框架',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行所有探索方向（快速模式）
  python exploration/exploration_runner.py --quick
  
  # 运行特定方向
  python exploration/exploration_runner.py --directions neuromorphic meta_learning
  
  # 完整运行
  python exploration/exploration_runner.py
        """
    )
    
    parser.add_argument(
        '--directions',
        nargs='+',
        choices=['neuromorphic', 'meta_learning', 'multi_agent', 'edge', 'foundations', 'all'],
        default=['all'],
        help='要运行的探索方向'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='快速模式（减少迭代次数）'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='安静模式（减少输出）'
    )
    
    args = parser.parse_args()
    
    directions = None if 'all' in args.directions else args.directions
    
    run_exploration(
        directions=directions,
        quick=args.quick,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
