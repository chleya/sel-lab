# -*- coding: utf-8 -*-
"""
精简版 Phase 3 策略注册表

根据审查报告建议，只保留以下核心策略：
1. adapt_only - 基线策略
2. task_specialist_clone_limited_merge - 遗忘最低
3. task_specialist_clone_current_path_boost_merge - 最强单一策略
4. task_specialist_clone_two_stage_selector_merge - selector 家族最佳

其他策略已归档到 archive/phase3_policy_registry_archive.py
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence


# 精简版策略列表（只保留核心策略）
PHASE3_POLICY_ORDER = (
    "adapt_only",
    "task_specialist_clone_limited_merge",
    "task_specialist_clone_current_path_boost_merge",
    "task_specialist_clone_two_stage_selector_merge",
)

# 精简版策略规格
PHASE3_POLICY_SPECS: Dict[str, Dict[str, str]] = {
    "adapt_only": {
        "family": "baseline",
        "forward_change": "none",
        "update_rule_change": "single active path only",
        "memory_change": "none",
        "benchmark_intent": "reference policy",
    },
    "task_specialist_clone_limited_merge": {
        "family": "specialist_reuse",
        "forward_change": "low merge from archived specialists",
        "update_rule_change": "limited specialist adaptation",
        "memory_change": "specialist residual path",
        "benchmark_intent": "robustness baseline - lowest forgetting",
    },
    "task_specialist_clone_current_path_boost_merge": {
        "family": "plasticity_boost",
        "forward_change": "low specialist merge",
        "update_rule_change": "boost active current path",
        "memory_change": "specialist residual path",
        "benchmark_intent": "current-task plasticity leader - strongest single policy",
    },
    "task_specialist_clone_two_stage_selector_merge": {
        "family": "regime_switch",
        "forward_change": "switch between low-merge specialist reuse and current-path boost by a two-stage selector",
        "update_rule_change": "embedded-gate routing followed by task-level ranking fallback",
        "memory_change": "mixture-style selector over signature and task-ranking routes",
        "benchmark_intent": "two-stage regime-selection - best selector family",
    },
}

# 精简版策略集
PHASE3_POLICY_SETS = {
    "core_only": PHASE3_POLICY_ORDER,
    "baseline_vs_best": (
        "adapt_only",
        "task_specialist_clone_current_path_boost_merge",
    ),
    "forgetting_comparison": (
        "adapt_only",
        "task_specialist_clone_limited_merge",
        "task_specialist_clone_current_path_boost_merge",
    ),
}

# 精简版诊断套件
PHASE3_DIAGNOSTIC_SUITES = {
    "core_digits": ("digits_pairs", "digits_pairs_noisy", "digits_pairs_permuted"),
    "feature_shift": ("feature_shift", "feature_shift_noisy"),
}

# 精简版诊断基准
PHASE3_DIAGNOSTIC_BENCHMARKS: Dict[str, Dict[str, object]] = {
    "plasticity_stress": {
        "task_suite": "digits_pairs_noisy",
        "policies": PHASE3_POLICY_ORDER,
        "benchmark_intent": "stress current-task learning under noisy sequential digits",
    },
    "interference_stress": {
        "task_suite": "feature_shift",
        "policies": PHASE3_POLICY_ORDER,
        "benchmark_intent": "test interference control under feature shift",
    },
}

PHASE3_DIAGNOSTIC_FAMILIES: Dict[str, tuple[str, ...]] = {
    "core_stress_map": (
        "plasticity_stress",
        "interference_stress",
    ),
}


def get_phase3_policy_spec(policy_name: str) -> Dict[str, str]:
    """获取策略规格"""
    if policy_name in PHASE3_POLICY_SPECS:
        return dict(PHASE3_POLICY_SPECS[policy_name])
    raise KeyError(f"Unknown Phase 3 policy: {policy_name}. "
                  f"Available policies: {list(PHASE3_POLICY_SPECS.keys())}")


def build_phase3_policy_metadata(policies: Iterable[str]) -> Dict[str, Dict[str, str]]:
    """构建策略元数据"""
    return {policy_name: get_phase3_policy_spec(policy_name) for policy_name in policies}


def group_phase3_policies_by_family(policies: Iterable[str]) -> Dict[str, tuple[str, ...]]:
    """按家族分组策略"""
    grouped: Dict[str, list[str]] = {}
    for policy_name in policies:
        family = get_phase3_policy_spec(policy_name)["family"]
        grouped.setdefault(family, []).append(policy_name)
    return {
        family_name: tuple(policy_names)
        for family_name, policy_names in grouped.items()
    }


def resolve_phase3_policies(config, policies: Sequence[str] | None = None) -> list[str]:
    """解析策略列表"""
    if policies is not None:
        # 验证所有策略都在精简列表中
        for policy in policies:
            if policy not in PHASE3_POLICY_SPECS:
                raise ValueError(f"Policy '{policy}' not in core policy set. "
                               f"Available: {list(PHASE3_POLICY_SPECS.keys())}")
        return list(policies)
    return list(PHASE3_POLICY_ORDER)


def resolve_phase3_policy_set(name: str) -> tuple[str, ...]:
    """解析策略集"""
    return tuple(PHASE3_POLICY_SETS[name])


def diagnostic_family_for_task_suite(task_suite: str) -> str | None:
    """查找任务套件对应的诊断家族"""
    for family_name, suites in PHASE3_DIAGNOSTIC_SUITES.items():
        if task_suite in suites:
            return family_name
    return None


def resolve_phase3_diagnostic_benchmark(name: str) -> Dict[str, object]:
    """解析诊断基准"""
    return dict(PHASE3_DIAGNOSTIC_BENCHMARKS[name])


def resolve_phase3_diagnostic_family(name: str) -> tuple[str, ...]:
    """解析诊断家族"""
    return tuple(PHASE3_DIAGNOSTIC_FAMILIES[name])
