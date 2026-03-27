"""
Policy application layer for phase-3 clone/reuse mechanisms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List

from core.phase3_registry import group_phase3_policies_by_family

if TYPE_CHECKING:
    from core.phase3_model import ReusePolicyNetwork


PolicyHandler = Callable[["ReusePolicyNetwork", int, int, int], List[str]]
SpecialCaseHandler = Callable[["ReusePolicyNetwork", List[int], float], List[str] | None]


def resolve_max_units(network: "ReusePolicyNetwork") -> int:
    max_units = network.config.max_units
    if network.policy == "capped_delayed_limited_clone":
        max_units = min(max_units, network.config.capped_clone_max_units)
    return max_units


def _handle_adapt_only(
    network: "ReusePolicyNetwork",
    active_indices: List[int],
    avg_tension: float,
) -> List[str]:
    return []


def _handle_random_clone(
    network: "ReusePolicyNetwork",
    active_indices: List[int],
    avg_tension: float,
) -> List[str]:
    new_idx = network.add_unit()
    return [f"add_random:{new_idx}"]


def _handle_benefit_gate(
    network: "ReusePolicyNetwork",
    active_indices: List[int],
    avg_tension: float,
) -> List[str] | None:
    if not network.benefit_gate_open:
        return []
    return None


def _handle_trend_limited_clone(
    network: "ReusePolicyNetwork",
    active_indices: List[int],
    avg_tension: float,
) -> List[str]:
    return network._try_trend_limited_clone(active_indices, avg_tension)


def _handle_margin_clone(
    network: "ReusePolicyNetwork",
    active_indices: List[int],
    avg_tension: float,
) -> List[str]:
    return network._try_margin_clone(active_indices, avg_tension)


SPECIAL_EVOLVE_HANDLERS: dict[str, SpecialCaseHandler] = {
    "adapt_only": _handle_adapt_only,
    "random_clone": _handle_random_clone,
    "benefit_limited_clone": _handle_benefit_gate,
    "trend_limited_clone": _handle_trend_limited_clone,
    "margin_clone": _handle_margin_clone,
}


def maybe_handle_special_evolve_case(
    network: "ReusePolicyNetwork",
    active_indices: List[int],
    avg_tension: float,
) -> List[str] | None:
    handler = SPECIAL_EVOLVE_HANDLERS.get(network.policy)
    if handler is None:
        return None
    return handler(network, active_indices, avg_tension)


def _limited_adapt_only(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    network.units[new_idx]["lr_scale"] = network.config.limited_adapt_scale
    return [f"limited_adapt:{new_idx}"]


def _benefit_limited_clone(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    changes = _limited_adapt_only(network, source_idx, new_idx, max_units)
    network.pending_benefit_eval = True
    changes.append(f"benefit_gate:{source_idx}")
    return changes


def _delayed_limited_clone(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    network.units[new_idx]["frozen_tasks"] = network.config.delayed_adapt_tasks
    changes = [f"delayed_adapt:{new_idx}"]
    changes.extend(_limited_adapt_only(network, source_idx, new_idx, max_units))
    return changes


def _capped_delayed_limited_clone(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    changes = _delayed_limited_clone(network, source_idx, new_idx, max_units)
    changes.append(f"cap:{max_units}")
    return changes


def _task_specialist_clone(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    return network._activate_task_specialist(new_idx)


def _task_specialist_clone_limited_merge(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    return network._activate_merged_specialist(new_idx)


def _task_specialist_clone_retention_merge(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    return network._activate_merged_specialist(new_idx, retention_bonus=True)


def _task_specialist_clone_confidence_gate_merge(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    changes = network._activate_merged_specialist(new_idx, confidence_gate=True)
    changes.append(
        f"confidence_gate:{network.config.specialist_confidence_threshold:.2f}/{network.config.specialist_min_gate_scale:.2f}"
    )
    return changes


def _task_specialist_clone_source_aware_merge(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    return network._activate_merged_specialist(
        new_idx,
        confidence_gate=True,
        source_aware_gate=True,
    )


def _task_specialist_clone_routed_merge(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    return network._activate_merged_specialist(
        new_idx,
        confidence_gate=True,
        route_by_similarity=True,
    )


def _task_specialist_clone_soft_route_merge(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    return network._activate_merged_specialist(
        new_idx,
        confidence_gate=True,
        soft_route=True,
    )


def _task_specialist_clone_budgeted_merge(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    return network._activate_merged_specialist(
        new_idx,
        confidence_gate=True,
        budgeted_gate=True,
    )


def _task_specialist_clone_anchor_merge(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    return network._activate_merged_specialist(
        new_idx,
        confidence_gate=True,
        anchor_regularized=True,
    )


def _task_specialist_clone_distance_anchor_merge(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    return network._activate_merged_specialist(
        new_idx,
        confidence_gate=True,
        anchor_regularized=True,
        distance_anchor=True,
    )


def _task_specialist_clone_distill_merge(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    changes = network._activate_task_specialist(
        new_idx,
        merge_scale=network.config.specialist_merge_scale,
        confidence_gate=True,
    )
    network.units[new_idx]["distill_regularized"] = True
    network._capture_anchor(new_idx)
    changes.append(
        f"distill:{network.config.specialist_confidence_threshold:.2f}/{network.config.specialist_distill_strength:.2f}"
    )
    return changes


def _task_specialist_clone_current_path_boost_merge(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    return network._activate_boosted_specialist(new_idx)


def _task_specialist_clone_regime_switch_merge(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    task_suite = network.config.task_suite
    if task_suite.startswith("digits_pairs"):
        changes = network._activate_boosted_specialist(new_idx)
        changes.append("regime_switch:plasticity_boost")
        return changes
    changes = network._activate_merged_specialist(new_idx)
    changes.append("regime_switch:interference_control")
    return changes


def _task_specialist_clone_diagnostic_selector_merge(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    if network._task_selector_prefers_interference_control():
        changes = network._activate_merged_specialist(new_idx)
        changes.append("diagnostic_selector:interference_control")
        return changes
    changes = network._activate_boosted_specialist(new_idx)
    changes.append("diagnostic_selector:plasticity_boost")
    return changes


def _task_specialist_clone_fingerprint_selector_merge(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    if network._task_fingerprint_prefers_interference_control():
        changes = network._activate_merged_specialist(new_idx)
        changes.append("fingerprint_selector:interference_control")
        return changes
    changes = network._activate_boosted_specialist(new_idx)
    changes.append("fingerprint_selector:plasticity_boost")
    return changes


def _task_specialist_clone_signature_selector_merge(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    if network._task_signature_prefers_interference_control():
        changes = network._activate_merged_specialist(new_idx)
        changes.append("signature_selector:interference_control")
        return changes
    changes = network._activate_boosted_specialist(new_idx)
    changes.append("signature_selector:plasticity_boost")
    return changes


def _task_specialist_clone_fitted_selector_merge(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    if network._task_fitted_selector_prefers_interference_control():
        changes = network._activate_merged_specialist(new_idx)
        changes.append("fitted_selector:interference_control")
        return changes
    changes = network._activate_boosted_specialist(new_idx)
    changes.append("fitted_selector:plasticity_boost")
    return changes


def _task_specialist_clone_prototype_selector_merge(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    if network._task_prototype_selector_prefers_interference_control():
        changes = network._activate_merged_specialist(new_idx)
        changes.append("prototype_selector:interference_control")
        return changes
    changes = network._activate_boosted_specialist(new_idx)
    changes.append("prototype_selector:plasticity_boost")
    return changes


def _task_specialist_clone_confidence_boost_merge(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    changes = network._activate_boosted_specialist(new_idx)
    network.units[new_idx]["confidence_gate"] = True
    changes.append(
        f"confidence_gate:{network.config.specialist_confidence_threshold:.2f}/{network.config.specialist_min_gate_scale:.2f}"
    )
    return changes


def _task_specialist_clone_mode_switch_boost_merge(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    changes = network._activate_boosted_specialist(new_idx)
    network.units[new_idx]["mode_switch_boost"] = True
    changes.append(
        "mode_switch_boost:"
        f"{network.config.current_path_lr_boost:.2f}->"
        f"{network.config.mode_switch_min_current_boost:.2f}@"
        f"{network.config.mode_switch_conflict_threshold:.2f}"
    )
    return changes


def _task_specialist_clone_dual_mode_update_merge(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    changes = network._activate_boosted_specialist(new_idx, capture_anchor=True)
    network.units[new_idx]["mode_switch_boost"] = True
    network.units[new_idx]["dual_mode_archived_update"] = True
    changes.append(
        "dual_mode_update:"
        f"{network.config.current_path_lr_boost:.2f}->"
        f"{network.config.mode_switch_min_current_boost:.2f}@"
        f"{network.config.mode_switch_conflict_threshold:.2f}/"
        f"{network.config.specialist_retention_strength:.2f}"
    )
    return changes


def _task_specialist_clone_decoupled_objective_boost_merge(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    changes = network._activate_boosted_specialist(
        new_idx,
        capture_anchor=True,
        archived_mode={"consistency_only": True},
    )
    changes.append(f"consistency_only:{network.config.specialist_distill_strength:.2f}")
    return changes


def _task_specialist_clone_two_speed_boost_merge(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    changes = network._activate_boosted_specialist(
        new_idx,
        archived_mode={"old_path_slow": True},
    )
    changes.append(f"old_path_lr:{network.config.old_path_lr_scale:.2f}")
    return changes


def _task_specialist_clone_external_memory_boost_merge(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    return network._activate_external_memory_boosted_specialist(new_idx)


def _task_specialist_clone_external_memory_routed_merge(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    return network._activate_external_memory_boosted_specialist(
        new_idx,
        routed_memory=True,
    )


def _task_specialist_clone_current_path_boost_retention_split_merge(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    changes = network._activate_boosted_specialist(
        new_idx,
        capture_anchor=True,
        archived_mode={"retention_only": True},
    )
    changes.append(f"retention_split:{network.config.specialist_retention_strength:.2f}")
    return changes


def _task_specialist_clone_dual_path_boost_merge(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    changes = network._activate_boosted_specialist(
        new_idx,
        capture_anchor=True,
        archived_mode={"memory_path_only": True, "retention_only": True},
    )
    changes.append(f"dual_path_split:{network.config.specialist_retention_strength:.2f}")
    return changes


def _task_specialist_clone_dual_path_decoupled_merge(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    changes = network._activate_boosted_specialist(
        new_idx,
        capture_anchor=True,
        archived_mode={"memory_path_only": True, "retention_only": True},
        decoupled_current_loss=True,
    )
    changes.append(f"dual_path_decoupled:{network.config.specialist_retention_strength:.2f}")
    return changes


def _task_specialist_clone_asymmetric_retention_merge(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    changes = network._activate_merged_specialist(
        new_idx,
        confidence_gate=True,
    )
    network.units[new_idx]["retention_only_update"] = True
    network._capture_anchor(new_idx)
    changes.append(f"asymmetric_retention:{network.config.specialist_retention_strength:.2f}")
    return changes


def _task_specialist_clone_role_separation_merge(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    changes = network._activate_merged_specialist(new_idx)
    network.units[new_idx]["role_separated"] = True
    changes.append("role_separation:archive_specialists")
    return changes


def _task_specialist_clone_freeze_source(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    changes = network._activate_task_specialist(new_idx)
    network.units[source_idx]["frozen_tasks"] = max(network.units[source_idx]["frozen_tasks"], 1)
    changes.append(f"freeze_source:{source_idx}")
    return changes


def _margin_limited_clone(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    changes = _limited_adapt_only(network, source_idx, new_idx, max_units)
    changes.append(f"margin_gate:{source_idx}")
    return changes


BASE_CLONE_POLICY_HANDLERS: dict[str, PolicyHandler] = {
    "best_clone_limited_adapt": _limited_adapt_only,
    "benefit_limited_clone": _benefit_limited_clone,
    "best_clone_delayed_limited_adapt": _delayed_limited_clone,
    "capped_delayed_limited_clone": _capped_delayed_limited_clone,
    "task_specialist_clone": _task_specialist_clone,
    "task_specialist_clone_limited_merge": _task_specialist_clone_limited_merge,
    "task_specialist_clone_retention_merge": _task_specialist_clone_retention_merge,
    "task_specialist_clone_confidence_gate_merge": _task_specialist_clone_confidence_gate_merge,
    "task_specialist_clone_source_aware_merge": _task_specialist_clone_source_aware_merge,
    "task_specialist_clone_routed_merge": _task_specialist_clone_routed_merge,
    "task_specialist_clone_soft_route_merge": _task_specialist_clone_soft_route_merge,
    "task_specialist_clone_budgeted_merge": _task_specialist_clone_budgeted_merge,
    "task_specialist_clone_anchor_merge": _task_specialist_clone_anchor_merge,
    "task_specialist_clone_distance_anchor_merge": _task_specialist_clone_distance_anchor_merge,
    "task_specialist_clone_distill_merge": _task_specialist_clone_distill_merge,
    "task_specialist_clone_asymmetric_retention_merge": _task_specialist_clone_asymmetric_retention_merge,
    "task_specialist_clone_role_separation_merge": _task_specialist_clone_role_separation_merge,
    "task_specialist_clone_current_path_boost_merge": _task_specialist_clone_current_path_boost_merge,
    "task_specialist_clone_regime_switch_merge": _task_specialist_clone_regime_switch_merge,
    "task_specialist_clone_diagnostic_selector_merge": _task_specialist_clone_diagnostic_selector_merge,
    "task_specialist_clone_fingerprint_selector_merge": _task_specialist_clone_fingerprint_selector_merge,
    "task_specialist_clone_signature_selector_merge": _task_specialist_clone_signature_selector_merge,
    "task_specialist_clone_fitted_selector_merge": _task_specialist_clone_fitted_selector_merge,
    "task_specialist_clone_prototype_selector_merge": _task_specialist_clone_prototype_selector_merge,
    "task_specialist_clone_confidence_boost_merge": _task_specialist_clone_confidence_boost_merge,
    "task_specialist_clone_mode_switch_boost_merge": _task_specialist_clone_mode_switch_boost_merge,
    "task_specialist_clone_dual_mode_update_merge": _task_specialist_clone_dual_mode_update_merge,
    "task_specialist_clone_decoupled_objective_boost_merge": _task_specialist_clone_decoupled_objective_boost_merge,
    "task_specialist_clone_two_speed_boost_merge": _task_specialist_clone_two_speed_boost_merge,
    "task_specialist_clone_current_path_boost_retention_split_merge": _task_specialist_clone_current_path_boost_retention_split_merge,
    "task_specialist_clone_external_memory_boost_merge": _task_specialist_clone_external_memory_boost_merge,
    "task_specialist_clone_external_memory_routed_merge": _task_specialist_clone_external_memory_routed_merge,
    "task_specialist_clone_dual_path_boost_merge": _task_specialist_clone_dual_path_boost_merge,
    "task_specialist_clone_dual_path_decoupled_merge": _task_specialist_clone_dual_path_decoupled_merge,
    "task_specialist_clone_freeze_source": _task_specialist_clone_freeze_source,
    "margin_limited_clone": _margin_limited_clone,
}

def _build_family_handler_maps() -> dict[str, dict[str, PolicyHandler]]:
    grouped_policies = group_phase3_policies_by_family(BASE_CLONE_POLICY_HANDLERS.keys())
    return {
        family_name: {
            policy_name: BASE_CLONE_POLICY_HANDLERS[policy_name]
            for policy_name in policy_names
        }
        for family_name, policy_names in grouped_policies.items()
    }


CLONE_POLICY_FAMILY_HANDLERS = _build_family_handler_maps()

CLONE_POLICY_HANDLERS: dict[str, PolicyHandler] = {}
for family_handlers in CLONE_POLICY_FAMILY_HANDLERS.values():
    CLONE_POLICY_HANDLERS.update(family_handlers)


def apply_clone_policy(
    network: "ReusePolicyNetwork",
    source_idx: int,
    new_idx: int,
    max_units: int,
) -> List[str]:
    handler = CLONE_POLICY_HANDLERS.get(network.policy)
    if handler is None:
        return []
    return handler(network, source_idx, new_idx, max_units)
