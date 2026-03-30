# -*- coding: utf-8 -*-
"""
Phase 4 age_decay 参数扫描脚本
用于诊断结构扩展失效问题
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.phase4_common import Phase4Config, load_real_mnist_or_digits, split_indices, run_phase4_suite


def run_age_decay_scan():
    """扫描不同 age_decay 值的效果"""
    # 加载数据集
    X, y, dataset_name = load_real_mnist_or_digits()
    n = len(X)
    train_idx, test_idx = split_indices(n, n_train=int(n * 0.8))
    
    # 扫描的 age_decay 值范围
    age_decay_values = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    
    results = []
    
    for age_decay in age_decay_values:
        print(f"\n{'='*70}")
        print(f"Scanning age_decay = {age_decay}")
        print(f"{'='*70}")
        
        # 配置
        config = Phase4Config(
            input_size=X.shape[1],
            hidden_size=100,
            learning_rate=0.05,
            runs=3,
            epochs=30,
            age_decay=age_decay,
            initial_tension=0.5,
            clone_noise_scale=0.1
        )
        
        # 运行测试
        result = run_phase4_suite(
            title=f"Phase 4 - age_decay={age_decay}",
            config=config,
            X=X,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            result_filename=f"phase4_age_decay_scan_{age_decay:.3f}.json",
            dataset_name=dataset_name,
            evolution_threshold=0.3
        )
        
        # 记录结果
        results.append({
            "age_decay": age_decay,
            "fixed_final": result["fixed_final"],
            "evolving_final": result["evolving_final"],
            "advantage": result["advantage"],
            "decision": result["decision"],
            "max_units": max(r["max_units"] for r in result["results"])
        })
    
    # 保存扫描结果
    scan_results = {
        "dataset": dataset_name,
        "scan_results": results
    }
    
    output_path = PROJECT_ROOT / "results" / "phase4_age_decay_scan.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scan_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print("Age Decay Scan Summary")
    print(f"{'='*70}")
    print("age_decay | fixed_final | evolving_final | advantage | max_units | decision")
    print("-" * 70)
    
    for r in results:
        print(f"{r['age_decay']:.3f} | {r['fixed_final']:.3f} | {r['evolving_final']:.3f} | {r['advantage']:+.3f} | {r['max_units']:d} | {r['decision']}")
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_age_decay_scan()
