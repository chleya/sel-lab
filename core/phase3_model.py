# -*- coding: utf-8 -*-
"""
Learner core for phase-3 structure-reuse ablations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from core.phase3_common import Phase3Config
from core.phase3_policies import (
    apply_clone_policy,
    maybe_handle_special_evolve_case,
    resolve_max_units,
)

TASK_SIGNATURE_FEATURE_NAMES = (
    "conflict_score",
    "conflict_peak",
    "confidence",
    "input_abs_mean",
    "input_nonnegative_ratio",
    "input_zero_ratio",
)

TASK_REGIME_FEATURE_NAMES = TASK_SIGNATURE_FEATURE_NAMES + (
    "loss_mean",
    "loss_delta",
    "conflict_delta",
    "confidence_delta",
)
TASK_REGIME_QUADRATIC_FEATURE_COUNT = len(TASK_REGIME_FEATURE_NAMES) + (
    len(TASK_REGIME_FEATURE_NAMES) * (len(TASK_REGIME_FEATURE_NAMES) + 1)
) // 2


@dataclass
class AblationConfig(Phase3Config):
    max_units: int = 12
    capped_clone_max_units: int = 4
    tension_threshold: float = 0.15
    evolve_interval: int = 10
    low_tension_ratio: float = 0.75
    clone_margin_ratio: float = 0.2
    trend_window: int = 3
    trend_max_increase: float = 0.01
    benefit_score_threshold: float = 0.45
    benefit_reopen_margin: float = 0.05
    limited_adapt_scale: float = 0.35
    specialist_merge_scale: float = 0.2
    retention_merge_bonus: float = 0.15
    specialist_confidence_threshold: float = 0.35
    specialist_min_gate_scale: float = 0.15
    specialist_route_min_scale: float = 0.05
    specialist_softmax_temperature: float = 0.75
    specialist_budget_scale: float = 0.6
    specialist_source_agreement_bonus: float = 0.75
    specialist_source_conflict_floor: float = 0.25
    specialist_anchor_strength: float = 0.08
    specialist_distance_anchor_bonus: float = 0.08
    specialist_distill_strength: float = 0.35
    specialist_retention_strength: float = 0.12
    current_path_lr_boost: float = 1.75
    mode_switch_conflict_threshold: float = 0.35
    mode_switch_min_current_boost: float = 1.0
    selector_conflict_threshold: float = 0.30
    selector_confidence_threshold: float = 0.22
    selector_input_nonnegative_threshold: float = 0.85
    selector_input_abs_mean_threshold: float = 0.80
    selector_input_zero_ratio_threshold: float = 0.55
    selector_conflict_peak_scale: float = 1.10
    selector_fit_conflict_weight: float = 0.0
    selector_fit_conflict_peak_weight: float = 0.0
    selector_fit_confidence_weight: float = 0.0
    selector_fit_abs_mean_weight: float = 0.0
    selector_fit_nonnegative_weight: float = 0.0
    selector_fit_zero_ratio_weight: float = 0.0
    selector_fit_bias: float = 0.0
    selector_feature_mean: tuple[float, ...] = (0.0,) * len(TASK_SIGNATURE_FEATURE_NAMES)
    selector_feature_std: tuple[float, ...] = (1.0,) * len(TASK_SIGNATURE_FEATURE_NAMES)
    selector_prototype_vectors: tuple[tuple[float, ...], ...] = ()
    selector_prototype_labels: tuple[float, ...] = ()
    selector_prototype_top_k: int = 3
    selector_prototype_bandwidth: float = 1.0
    selector_regime_fit_weights: tuple[float, ...] = (0.0,) * len(TASK_REGIME_FEATURE_NAMES)
    selector_regime_fit_bias: float = 0.0
    selector_outcome_fit_weights: tuple[float, ...] = (0.0,) * len(TASK_REGIME_FEATURE_NAMES)
    selector_outcome_fit_bias: float = 0.0
    selector_ranking_fit_weights: tuple[float, ...] = (0.0,) * len(TASK_REGIME_FEATURE_NAMES)
    selector_ranking_fit_bias: float = 0.0
    selector_task_ranking_fit_weights: tuple[float, ...] = (0.0,) * len(TASK_REGIME_FEATURE_NAMES)
    selector_task_ranking_fit_bias: float = 0.0
    selector_constrained_task_ranking_fit_weights: tuple[float, ...] = (0.0,) * len(TASK_REGIME_FEATURE_NAMES)
    selector_constrained_task_ranking_fit_bias: float = 0.0
    selector_router_fit_weights: tuple[float, ...] = (0.0,) * (3 * len(TASK_REGIME_FEATURE_NAMES))
    selector_router_fit_bias: tuple[float, ...] = (0.0, 0.0, 0.0)
    selector_hierarchical_default_fit_weights: tuple[float, ...] = (0.0,) * len(TASK_REGIME_FEATURE_NAMES)
    selector_hierarchical_default_fit_bias: float = 0.0
    selector_hierarchical_embedded_fit_weights: tuple[float, ...] = (0.0,) * len(TASK_REGIME_FEATURE_NAMES)
    selector_hierarchical_embedded_fit_bias: float = 0.0
    selector_hierarchical_sparse_fit_weights: tuple[float, ...] = (0.0,) * len(TASK_REGIME_FEATURE_NAMES)
    selector_hierarchical_sparse_fit_bias: float = 0.0
    selector_hierarchical_quadratic_default_fit_weights: tuple[float, ...] = (0.0,) * TASK_REGIME_QUADRATIC_FEATURE_COUNT
    selector_hierarchical_quadratic_default_fit_bias: float = 0.0
    selector_hierarchical_quadratic_embedded_fit_weights: tuple[float, ...] = (0.0,) * TASK_REGIME_QUADRATIC_FEATURE_COUNT
    selector_hierarchical_quadratic_embedded_fit_bias: float = 0.0
    selector_hierarchical_quadratic_sparse_fit_weights: tuple[float, ...] = (0.0,) * TASK_REGIME_QUADRATIC_FEATURE_COUNT
    selector_hierarchical_quadratic_sparse_fit_bias: float = 0.0
    selector_hierarchical_sparse_gate_default_fit_weights: tuple[float, ...] = (0.0,) * TASK_REGIME_QUADRATIC_FEATURE_COUNT
    selector_hierarchical_sparse_gate_default_fit_bias: float = 0.0
    selector_hierarchical_sparse_gate_embedded_fit_weights: tuple[float, ...] = (0.0,) * TASK_REGIME_QUADRATIC_FEATURE_COUNT
    selector_hierarchical_sparse_gate_embedded_fit_bias: float = 0.0
    selector_hierarchical_sparse_gate_sparse_fit_weights: tuple[float, ...] = (0.0,) * TASK_REGIME_QUADRATIC_FEATURE_COUNT
    selector_hierarchical_sparse_gate_sparse_fit_bias: float = 0.0
    selector_guarded_router_default_fit_weights: tuple[float, ...] = (0.0,) * TASK_REGIME_QUADRATIC_FEATURE_COUNT
    selector_guarded_router_default_fit_bias: float = 0.0
    selector_guarded_router_embedded_fit_weights: tuple[float, ...] = (0.0,) * TASK_REGIME_QUADRATIC_FEATURE_COUNT
    selector_guarded_router_embedded_fit_bias: float = 0.0
    selector_guarded_expert_default_fit_weights: tuple[float, ...] = (0.0,) * TASK_REGIME_QUADRATIC_FEATURE_COUNT
    selector_guarded_expert_default_fit_bias: float = 0.0
    selector_guarded_expert_linear_default_fit_weights: tuple[float, ...] = (0.0,) * len(TASK_REGIME_FEATURE_NAMES)
    selector_guarded_expert_linear_default_fit_bias: float = 0.0
    selector_sparse_zero_ratio_floor: float = 0.385
    selector_sparse_conflict_delta_floor: float = -0.005
    old_path_lr_scale: float = 0.5
    archived_specialist_quantization_bits: int = 0
    archived_specialist_max_count: int = 0
    archived_specialist_selection_mode: str = "recency"
    delayed_adapt_tasks: int = 1
    freeze_windows: tuple[int, ...] = (1, 2, 3)


class FixedNetwork:
    """Fixed-structure baseline for sequential-learning comparisons."""

    def __init__(self, config: AblationConfig, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.W1 = self.rng.normal(0.0, 0.5, size=(config.input_size, config.hidden_size))
        self.W2 = self.rng.normal(0.0, 0.5, size=(config.hidden_size, config.output_size))
        self.feedback = self.rng.normal(0.0, 0.5, size=(config.output_size, config.hidden_size))

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x @ self.W1) @ self.W2

    def learn(self, x: np.ndarray, target: np.ndarray, lr: float = 0.01) -> float:
        h = np.tanh(x @ self.W1)
        out = h @ self.W2
        error = target - out
        fb_error = self.feedback.T @ error
        self.W2 += lr * np.outer(h, error)
        self.W1 += lr * np.outer(x, fb_error)
        return float(np.mean(error ** 2))

    def predict(self, x: np.ndarray) -> int:
        return int(np.argmax(self.forward(x)))

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        correct = sum(1 for i in range(len(X)) if self.predict(X[i]) == int(np.argmax(y[i])))
        return correct / len(X)


class ReusePolicyNetwork:
    """Sequential learner used for phase-3 ablations."""

    def __init__(self, config: AblationConfig, policy: str, seed: int | None = None):
        self.config = config
        self.policy = policy
        self.rng = np.random.default_rng(seed)
        self.units: List[Dict] = []
        self.memory_units: List[Dict] = []
        self.benefit_gate_open = True
        self.pending_benefit_eval = False
        self.current_task_id = 0
        self.last_forward_stats = {
            "confidence": 0.0,
            "conflict_score": 0.0,
            "conflict_peak": 0.0,
        }
        self.task_stat_sums = {
            "confidence": 0.0,
            "conflict_score": 0.0,
            "conflict_peak": 0.0,
            "input_abs_mean": 0.0,
            "input_nonnegative_ratio": 0.0,
            "input_zero_ratio": 0.0,
            "steps": 0,
        }
        self.task_stat_trace = {
            "loss": [],
            "confidence": [],
            "conflict_score": [],
        }
        self.add_unit()

    def _freeze_window(self) -> int:
        if self.policy == "best_clone_freeze_source":
            return 1
        if self.policy.startswith("best_clone_freeze_"):
            return int(self.policy.rsplit("_", 1)[1])
        return 0

    def add_unit(self, clone_from: int = -1) -> int:
        if 0 <= clone_from < len(self.units):
            source = self.units[clone_from]
            anchor_w1 = source.get("anchor_W1")
            anchor_w2 = source.get("anchor_W2")
            unit = {
                "W1": source["W1"].copy() + self.rng.normal(0.0, 0.1, size=source["W1"].shape),
                "W2": source["W2"].copy() + self.rng.normal(0.0, 0.1, size=source["W2"].shape),
                "feedback": source["feedback"].copy()
                + self.rng.normal(0.0, 0.1, size=source["feedback"].shape),
                "active": True,
                "task_specialist": False,
                "specialist_merge_scale": 0.0,
                "retention_merge_bonus": 0.0,
                "confidence_gate": False,
                "route_by_similarity": False,
                "soft_route": False,
                "budgeted_gate": False,
                "source_aware_gate": False,
                "external_memory_route": False,
                "anchor_regularized": False,
                "distance_anchor": False,
                "distill_regularized": False,
                "role_separated": False,
                "retention_only_update": False,
                "current_path_boost": False,
                "mode_switch_boost": False,
                "dual_mode_archived_update": False,
                "memory_path_only": False,
                "decoupled_current_loss": False,
                "old_path_slow": False,
                "consistency_only_update": False,
                "anchor_W1": anchor_w1.copy() if anchor_w1 is not None else source["W1"].copy(),
                "anchor_W2": anchor_w2.copy() if anchor_w2 is not None else source["W2"].copy(),
                "tension": source["tension"],
                "age": 0,
                "frozen": False,
                "frozen_tasks": 0,
                "lr_scale": 1.0,
                "origin": f"clone:{clone_from}",
                "tension_history": list(source.get("tension_history", [source["tension"]])),
                "specialist_task": self.current_task_id,
            }
        else:
            unit = {
                "W1": self.rng.normal(0.0, 0.5, size=(self.config.input_size, self.config.hidden_size)),
                "W2": self.rng.normal(0.0, 0.5, size=(self.config.hidden_size, self.config.output_size)),
                "feedback": self.rng.normal(
                    0.0, 0.5, size=(self.config.output_size, self.config.hidden_size)
                ),
                "active": True,
                "task_specialist": False,
                "specialist_merge_scale": 0.0,
                "retention_merge_bonus": 0.0,
                "confidence_gate": False,
                "route_by_similarity": False,
                "soft_route": False,
                "budgeted_gate": False,
                "source_aware_gate": False,
                "external_memory_route": False,
                "anchor_regularized": False,
                "distance_anchor": False,
                "distill_regularized": False,
                "role_separated": False,
                "retention_only_update": False,
                "current_path_boost": False,
                "mode_switch_boost": False,
                "dual_mode_archived_update": False,
                "memory_path_only": False,
                "decoupled_current_loss": False,
                "old_path_slow": False,
                "consistency_only_update": False,
                "anchor_W1": None,
                "anchor_W2": None,
                "tension": 0.5,
                "age": 0,
                "frozen": False,
                "frozen_tasks": 0,
                "lr_scale": 1.0,
                "origin": "random",
                "tension_history": [0.5],
                "specialist_task": -1,
            }
        self.units.append(unit)
        return len(self.units) - 1

    def _source_idx_from_active(self, active_indices: List[int]) -> int:
        return min(active_indices, key=lambda idx: self.units[idx]["tension"])

    def _try_trend_limited_clone(self, active_indices: List[int], avg_tension: float) -> List[str]:
        source_idx = self._source_idx_from_active(active_indices)
        history = self.units[source_idx]["tension_history"][-self.config.trend_window :]
        if len(history) < self.config.trend_window:
            return []
        if history[-1] >= (avg_tension * self.config.low_tension_ratio):
            return []
        deltas = [history[i + 1] - history[i] for i in range(len(history) - 1)]
        if max(deltas, default=0.0) > self.config.trend_max_increase:
            return []
        new_idx = self.add_unit(clone_from=source_idx)
        self.units[new_idx]["lr_scale"] = self.config.limited_adapt_scale
        return [f"add_trend_clone:{source_idx}->{new_idx}", f"limited_adapt:{new_idx}"]

    def _try_margin_clone(self, active_indices: List[int], avg_tension: float) -> List[str]:
        tensions = [self.units[idx]["tension"] for idx in active_indices]
        source_idx = self._source_idx_from_active(active_indices)
        source_tension = self.units[source_idx]["tension"]
        second_best = min((t for idx, t in zip(active_indices, tensions) if idx != source_idx), default=avg_tension)
        required_margin = avg_tension * self.config.clone_margin_ratio
        if (second_best - source_tension) < required_margin:
            return []
        new_idx = self.add_unit(clone_from=source_idx)
        return [f"add_margin_clone:{source_idx}->{new_idx}"]

    def _passes_preclone_gate(self, active_indices: List[int], avg_tension: float, source_idx: int) -> bool:
        source_tension = self.units[source_idx]["tension"]
        if self.policy == "low_tension_clone" and source_tension >= (avg_tension * self.config.low_tension_ratio):
            return False
        if self.policy == "margin_limited_clone":
            second_best = min((self.units[idx]["tension"] for idx in active_indices if idx != source_idx), default=avg_tension)
            required_margin = avg_tension * self.config.clone_margin_ratio
            if (second_best - source_tension) < required_margin:
                return False
        return True

    def _activate_task_specialist(
        self,
        new_idx: int,
        *,
        merge_scale: float | None = None,
        confidence_gate: bool = False,
    ) -> List[str]:
        self.units[new_idx]["task_specialist"] = True
        self.units[new_idx]["specialist_task"] = self.current_task_id
        self.units[new_idx]["lr_scale"] = self.config.limited_adapt_scale
        changes = [f"task_specialist:{new_idx}", f"limited_adapt:{new_idx}"]
        if merge_scale is not None:
            self.units[new_idx]["specialist_merge_scale"] = merge_scale
            changes.append(f"merge_scale:{merge_scale:.2f}")
        if confidence_gate:
            self.units[new_idx]["confidence_gate"] = True
        return changes

    def _capture_anchor(self, new_idx: int) -> None:
        self.units[new_idx]["anchor_W1"] = self.units[new_idx]["W1"].copy()
        self.units[new_idx]["anchor_W2"] = self.units[new_idx]["W2"].copy()

    def _activate_merged_specialist(
        self,
        new_idx: int,
        *,
        confidence_gate: bool = False,
        retention_bonus: bool = False,
        source_aware_gate: bool = False,
        route_by_similarity: bool = False,
        soft_route: bool = False,
        budgeted_gate: bool = False,
        anchor_regularized: bool = False,
        distance_anchor: bool = False,
    ) -> List[str]:
        changes = self._activate_task_specialist(
            new_idx,
            merge_scale=self.config.specialist_merge_scale,
            confidence_gate=confidence_gate,
        )
        unit = self.units[new_idx]
        if retention_bonus:
            unit["retention_merge_bonus"] = self.config.retention_merge_bonus
            changes.append(f"retention_bonus:{self.config.retention_merge_bonus:.2f}")
        if source_aware_gate:
            unit["source_aware_gate"] = True
            changes.append(
                "source_aware:"
                f"{self.config.specialist_source_agreement_bonus:.2f}/"
                f"{self.config.specialist_source_conflict_floor:.2f}"
            )
        if route_by_similarity:
            unit["route_by_similarity"] = True
            changes.append(
                f"routed_gate:{self.config.specialist_confidence_threshold:.2f}/{self.config.specialist_route_min_scale:.2f}"
            )
        if soft_route:
            unit["soft_route"] = True
            changes.append(
                f"soft_route:{self.config.specialist_confidence_threshold:.2f}/{self.config.specialist_softmax_temperature:.2f}"
            )
        if budgeted_gate:
            unit["budgeted_gate"] = True
            changes.append(
                f"budgeted_gate:{self.config.specialist_confidence_threshold:.2f}/{self.config.specialist_budget_scale:.2f}"
            )
        if anchor_regularized:
            unit["anchor_regularized"] = True
            self._capture_anchor(new_idx)
            if distance_anchor:
                unit["distance_anchor"] = True
                changes.append(
                    f"distance_anchor:{self.config.specialist_anchor_strength:.2f}/{self.config.specialist_distance_anchor_bonus:.2f}"
                )
            else:
                changes.append(
                    f"anchor:{self.config.specialist_confidence_threshold:.2f}/{self.config.specialist_anchor_strength:.2f}"
                )
        return changes

    def _activate_boosted_specialist(
        self,
        new_idx: int,
        *,
        capture_anchor: bool = False,
        archived_mode: Dict[str, bool] | None = None,
        decoupled_current_loss: bool = False,
    ) -> List[str]:
        changes = self._activate_task_specialist(new_idx, merge_scale=self.config.specialist_merge_scale)
        self.units[new_idx]["current_path_boost"] = True
        if capture_anchor:
            self._capture_anchor(new_idx)
        if archived_mode:
            self._set_archived_units_mode(**archived_mode)
        self._enable_current_path_boost_for_nonarchived()
        if decoupled_current_loss:
            for unit in self.units:
                unit["decoupled_current_loss"] = True
        changes.append(f"current_path_boost:{self.config.current_path_lr_boost:.2f}")
        return changes

    def _activate_external_memory_boosted_specialist(
        self,
        new_idx: int,
        *,
        routed_memory: bool = False,
    ) -> List[str]:
        changes = self._activate_boosted_specialist(new_idx)
        self._archive_prior_specialists_to_memory(routed=routed_memory)
        if routed_memory:
            changes.append(
                f"external_memory_route:{len(self.memory_units)}/{self.config.specialist_confidence_threshold:.2f}"
            )
        else:
            changes.append(f"external_memory:{len(self.memory_units)}")
        return changes

    def _task_selector_prefers_interference_control(self) -> bool:
        steps = max(1, int(self.task_stat_sums.get("steps", 0)))
        mean_conflict = self.task_stat_sums.get("conflict_score", 0.0) / steps
        mean_confidence = self.task_stat_sums.get("confidence", 0.0) / steps
        return (
            mean_conflict >= self.config.selector_conflict_threshold
            and mean_confidence <= self.config.selector_confidence_threshold
        )

    def _task_signature_feature_dict(self) -> Dict[str, float]:
        steps = max(1, int(self.task_stat_sums.get("steps", 0)))
        return {
            "conflict_score": float(self.task_stat_sums.get("conflict_score", 0.0) / steps),
            "conflict_peak": float(self.task_stat_sums.get("conflict_peak", 0.0) / steps),
            "confidence": float(self.task_stat_sums.get("confidence", 0.0) / steps),
            "input_abs_mean": float(self.task_stat_sums.get("input_abs_mean", 0.0) / steps),
            "input_nonnegative_ratio": float(self.task_stat_sums.get("input_nonnegative_ratio", 0.0) / steps),
            "input_zero_ratio": float(self.task_stat_sums.get("input_zero_ratio", 0.0) / steps),
        }

    def _task_signature_feature_vector(self) -> np.ndarray:
        features = self._task_signature_feature_dict()
        return np.array(
            [
                features["conflict_score"],
                features["conflict_peak"],
                features["confidence"],
                features["input_abs_mean"],
                features["input_nonnegative_ratio"],
                features["input_zero_ratio"],
            ],
            dtype=float,
        )

    def _task_regime_feature_dict(self) -> Dict[str, float]:
        features = dict(self._task_signature_feature_dict())
        loss_trace = self.task_stat_trace.get("loss", [])
        conflict_trace = self.task_stat_trace.get("conflict_score", [])
        confidence_trace = self.task_stat_trace.get("confidence", [])
        features.update(
            {
                "loss_mean": self._trace_mean(loss_trace),
                "loss_delta": self._trace_delta(loss_trace, invert=False),
                "conflict_delta": self._trace_delta(conflict_trace, invert=True),
                "confidence_delta": self._trace_delta(confidence_trace, invert=False),
            }
        )
        return features

    def _task_regime_feature_vector(self) -> np.ndarray:
        features = self._task_regime_feature_dict()
        return np.array([features[name] for name in TASK_REGIME_FEATURE_NAMES], dtype=float)

    def _task_regime_quadratic_feature_vector(self) -> np.ndarray:
        linear = self._task_regime_feature_vector()
        pairwise_terms = []
        for idx in range(len(linear)):
            for jdx in range(idx, len(linear)):
                pairwise_terms.append(linear[idx] * linear[jdx])
        return np.concatenate([linear, np.array(pairwise_terms, dtype=float)])

    def _trace_mean(self, values: List[float]) -> float:
        if not values:
            return 0.0
        return float(np.mean(values))

    def _trace_delta(self, values: List[float], *, invert: bool) -> float:
        if not values:
            return 0.0
        split = max(1, len(values) // 2)
        early_mean = float(np.mean(values[:split]))
        late_mean = float(np.mean(values[split:])) if split < len(values) else early_mean
        if invert:
            return early_mean - late_mean
        return late_mean - early_mean

    def _task_fingerprint_prefers_interference_control(self) -> bool:
        steps = max(1, int(self.task_stat_sums.get("steps", 0)))
        mean_abs = self.task_stat_sums.get("input_abs_mean", 0.0) / steps
        mean_nonnegative = self.task_stat_sums.get("input_nonnegative_ratio", 0.0) / steps
        return (
            mean_nonnegative < self.config.selector_input_nonnegative_threshold
            and mean_abs > self.config.selector_input_abs_mean_threshold
        )

    def _task_signature_prefers_interference_control(self) -> bool:
        features = self._task_signature_feature_dict()
        mean_conflict = features["conflict_score"]
        mean_conflict_peak = features["conflict_peak"]
        mean_confidence = features["confidence"]
        mean_abs = features["input_abs_mean"]
        mean_nonnegative = features["input_nonnegative_ratio"]
        mean_zero_ratio = features["input_zero_ratio"]

        strong_interference_signal = (
            mean_conflict >= self.config.selector_conflict_threshold
            and mean_abs >= self.config.selector_input_abs_mean_threshold
        )
        sparse_embedded_interference_signal = (
            mean_zero_ratio >= self.config.selector_input_zero_ratio_threshold
            and mean_abs >= self.config.selector_input_abs_mean_threshold
            and mean_conflict_peak
            >= self.config.selector_conflict_threshold * self.config.selector_conflict_peak_scale
        )
        classic_interference_signal = (
            mean_conflict >= self.config.selector_conflict_threshold
            and mean_confidence <= self.config.selector_confidence_threshold
        )
        easy_digits_like_signal = (
            mean_nonnegative >= self.config.selector_input_nonnegative_threshold
            and mean_abs < self.config.selector_input_abs_mean_threshold
            and mean_confidence > self.config.selector_confidence_threshold
        )
        if easy_digits_like_signal:
            return False
        return (
            strong_interference_signal
            or sparse_embedded_interference_signal
            or classic_interference_signal
        )

    def _task_fitted_selector_prefers_interference_control(self) -> bool:
        feature_vector = self._task_signature_feature_vector()
        weight_vector = np.array(
            [
                self.config.selector_fit_conflict_weight,
                self.config.selector_fit_conflict_peak_weight,
                self.config.selector_fit_confidence_weight,
                self.config.selector_fit_abs_mean_weight,
                self.config.selector_fit_nonnegative_weight,
                self.config.selector_fit_zero_ratio_weight,
            ],
            dtype=float,
        )
        score = float(np.dot(weight_vector, feature_vector) + self.config.selector_fit_bias)
        return score >= 0.0

    def _task_prototype_selector_prefers_interference_control(self) -> bool:
        if not self.config.selector_prototype_vectors or not self.config.selector_prototype_labels:
            return self._task_fitted_selector_prefers_interference_control()
        feature_vector = self._task_signature_feature_vector()
        feature_mean = np.array(self.config.selector_feature_mean, dtype=float)
        feature_std = np.array(self.config.selector_feature_std, dtype=float)
        feature_std = np.where(feature_std <= 1e-8, 1.0, feature_std)
        normalized = (feature_vector - feature_mean) / feature_std
        prototype_vectors = np.array(self.config.selector_prototype_vectors, dtype=float)
        prototype_labels = np.array(self.config.selector_prototype_labels, dtype=float)
        distances = np.linalg.norm(prototype_vectors - normalized[None, :], axis=1)
        top_k = max(1, min(int(self.config.selector_prototype_top_k), len(distances)))
        top_indices = np.argsort(distances)[:top_k]
        bandwidth = max(1e-6, float(self.config.selector_prototype_bandwidth))
        weights = np.exp(-(distances[top_indices] ** 2) / (2.0 * bandwidth * bandwidth))
        score = float(np.sum(weights * prototype_labels[top_indices]))
        return score > 0.0

    def _task_dynamics_selector_prefers_interference_control(self) -> bool:
        feature_vector = self._task_regime_feature_vector()
        weight_vector = np.array(self.config.selector_regime_fit_weights, dtype=float)
        if len(weight_vector) < len(feature_vector):
            weight_vector = np.pad(weight_vector, (0, len(feature_vector) - len(weight_vector)))
        elif len(weight_vector) > len(feature_vector):
            weight_vector = weight_vector[: len(feature_vector)]
        score = float(np.dot(weight_vector, feature_vector) + self.config.selector_regime_fit_bias)
        return score >= 0.0

    def _task_outcome_selector_prefers_interference_control(self) -> bool:
        feature_vector = self._task_regime_feature_vector()
        weight_vector = np.array(self.config.selector_outcome_fit_weights, dtype=float)
        if len(weight_vector) < len(feature_vector):
            weight_vector = np.pad(weight_vector, (0, len(feature_vector) - len(weight_vector)))
        elif len(weight_vector) > len(feature_vector):
            weight_vector = weight_vector[: len(feature_vector)]
        score = float(np.dot(weight_vector, feature_vector) + self.config.selector_outcome_fit_bias)
        return score >= 0.0

    def _task_ranking_selector_prefers_interference_control(self) -> bool:
        feature_vector = self._task_regime_feature_vector()
        weight_vector = np.array(self.config.selector_ranking_fit_weights, dtype=float)
        if len(weight_vector) < len(feature_vector):
            weight_vector = np.pad(weight_vector, (0, len(feature_vector) - len(weight_vector)))
        elif len(weight_vector) > len(feature_vector):
            weight_vector = weight_vector[: len(feature_vector)]
        score = float(np.dot(weight_vector, feature_vector) + self.config.selector_ranking_fit_bias)
        return score >= 0.0

    def _task_task_ranking_selector_prefers_interference_control(self) -> bool:
        feature_vector = self._task_regime_feature_vector()
        weight_vector = np.array(self.config.selector_task_ranking_fit_weights, dtype=float)
        if len(weight_vector) < len(feature_vector):
            weight_vector = np.pad(weight_vector, (0, len(feature_vector) - len(weight_vector)))
        elif len(weight_vector) > len(feature_vector):
            weight_vector = weight_vector[: len(feature_vector)]
        score = float(np.dot(weight_vector, feature_vector) + self.config.selector_task_ranking_fit_bias)
        return score >= 0.0

    def _task_constrained_task_ranking_selector_prefers_interference_control(self) -> bool:
        feature_vector = self._task_regime_feature_vector()
        weight_vector = np.array(self.config.selector_constrained_task_ranking_fit_weights, dtype=float)
        if len(weight_vector) < len(feature_vector):
            weight_vector = np.pad(weight_vector, (0, len(feature_vector) - len(weight_vector)))
        elif len(weight_vector) > len(feature_vector):
            weight_vector = weight_vector[: len(feature_vector)]
        score = float(
            np.dot(weight_vector, feature_vector) + self.config.selector_constrained_task_ranking_fit_bias
        )
        return score >= 0.0

    def _task_selector_region(self) -> str:
        signature = self._task_signature_feature_dict()
        regime = self._task_regime_feature_dict()
        embedded_like = (
            signature["input_nonnegative_ratio"] >= self.config.selector_input_nonnegative_threshold
            and signature["input_abs_mean"] >= self.config.selector_input_abs_mean_threshold
        )
        sparse_embedded_like = (
            embedded_like
            and signature["input_zero_ratio"] >= self.config.selector_sparse_zero_ratio_floor
            and regime["conflict_delta"] >= self.config.selector_sparse_conflict_delta_floor
        )
        if sparse_embedded_like:
            return "sparse_embedded"
        if embedded_like:
            return "embedded"
        return "default"

    def _task_two_stage_selector_mode(self) -> str:
        region = self._task_selector_region()
        if region == "sparse_embedded":
            return "plasticity_boost"
        if region == "embedded":
            if self._task_signature_prefers_interference_control():
                return "interference_control"
            return "plasticity_boost"
        if self._task_task_ranking_selector_prefers_interference_control():
            return "interference_control"
        return "plasticity_boost"

    def _task_hierarchical_selector_mode(self) -> str:
        region = self._task_selector_region()
        feature_vector = self._task_regime_feature_vector()
        if region == "embedded":
            raw_weights = self.config.selector_hierarchical_embedded_fit_weights
            raw_bias = self.config.selector_hierarchical_embedded_fit_bias
        elif region == "sparse_embedded":
            raw_weights = self.config.selector_hierarchical_sparse_fit_weights
            raw_bias = self.config.selector_hierarchical_sparse_fit_bias
        else:
            raw_weights = self.config.selector_hierarchical_default_fit_weights
            raw_bias = self.config.selector_hierarchical_default_fit_bias
        weight_vector = np.array(raw_weights, dtype=float)
        if len(weight_vector) < len(feature_vector):
            weight_vector = np.pad(weight_vector, (0, len(feature_vector) - len(weight_vector)))
        elif len(weight_vector) > len(feature_vector):
            weight_vector = weight_vector[: len(feature_vector)]
        score = float(np.dot(weight_vector, feature_vector) + raw_bias)
        if score >= 0.0:
            return "interference_control"
        return "plasticity_boost"

    def _task_hierarchical_quadratic_selector_mode(self) -> str:
        region = self._task_selector_region()
        feature_vector = self._task_regime_quadratic_feature_vector()
        if region == "embedded":
            raw_weights = self.config.selector_hierarchical_quadratic_embedded_fit_weights
            raw_bias = self.config.selector_hierarchical_quadratic_embedded_fit_bias
        elif region == "sparse_embedded":
            raw_weights = self.config.selector_hierarchical_quadratic_sparse_fit_weights
            raw_bias = self.config.selector_hierarchical_quadratic_sparse_fit_bias
        else:
            raw_weights = self.config.selector_hierarchical_quadratic_default_fit_weights
            raw_bias = self.config.selector_hierarchical_quadratic_default_fit_bias
        weight_vector = np.array(raw_weights, dtype=float)
        if len(weight_vector) < len(feature_vector):
            weight_vector = np.pad(weight_vector, (0, len(feature_vector) - len(weight_vector)))
        elif len(weight_vector) > len(feature_vector):
            weight_vector = weight_vector[: len(feature_vector)]
        score = float(np.dot(weight_vector, feature_vector) + raw_bias)
        if score >= 0.0:
            return "interference_control"
        return "plasticity_boost"

    def _task_hierarchical_sparse_gate_selector_mode(self) -> str:
        region = self._task_selector_region()
        feature_vector = self._task_regime_quadratic_feature_vector()
        if region == "embedded":
            raw_weights = self.config.selector_hierarchical_sparse_gate_embedded_fit_weights
            raw_bias = self.config.selector_hierarchical_sparse_gate_embedded_fit_bias
        elif region == "sparse_embedded":
            raw_weights = self.config.selector_hierarchical_sparse_gate_sparse_fit_weights
            raw_bias = self.config.selector_hierarchical_sparse_gate_sparse_fit_bias
        else:
            raw_weights = self.config.selector_hierarchical_sparse_gate_default_fit_weights
            raw_bias = self.config.selector_hierarchical_sparse_gate_default_fit_bias
        weight_vector = np.array(raw_weights, dtype=float)
        if len(weight_vector) < len(feature_vector):
            weight_vector = np.pad(weight_vector, (0, len(feature_vector) - len(weight_vector)))
        elif len(weight_vector) > len(feature_vector):
            weight_vector = weight_vector[: len(feature_vector)]
        score = float(np.dot(weight_vector, feature_vector) + raw_bias)
        if score >= 0.0:
            return "interference_control"
        return "plasticity_boost"

    def _task_guarded_router_selector_mode(self) -> str:
        region = self._task_selector_region()
        if region == "sparse_embedded":
            return "plasticity_boost"
        feature_vector = self._task_regime_quadratic_feature_vector()
        if region == "embedded":
            raw_weights = self.config.selector_guarded_router_embedded_fit_weights
            raw_bias = self.config.selector_guarded_router_embedded_fit_bias
        else:
            raw_weights = self.config.selector_guarded_router_default_fit_weights
            raw_bias = self.config.selector_guarded_router_default_fit_bias
        weight_vector = np.array(raw_weights, dtype=float)
        if len(weight_vector) < len(feature_vector):
            weight_vector = np.pad(weight_vector, (0, len(feature_vector) - len(weight_vector)))
        elif len(weight_vector) > len(feature_vector):
            weight_vector = weight_vector[: len(feature_vector)]
        score = float(np.dot(weight_vector, feature_vector) + raw_bias)
        if score >= 0.0:
            return "interference_control"
        return "plasticity_boost"

    def _task_guarded_expert_selector_mode(self) -> str:
        region = self._task_selector_region()
        if region == "sparse_embedded":
            return "plasticity_boost"
        if region == "embedded":
            if self._task_signature_prefers_interference_control():
                return "interference_control"
            return "plasticity_boost"
        feature_vector = self._task_regime_quadratic_feature_vector()
        weight_vector = np.array(self.config.selector_guarded_expert_default_fit_weights, dtype=float)
        if len(weight_vector) < len(feature_vector):
            weight_vector = np.pad(weight_vector, (0, len(feature_vector) - len(weight_vector)))
        elif len(weight_vector) > len(feature_vector):
            weight_vector = weight_vector[: len(feature_vector)]
        score = float(np.dot(weight_vector, feature_vector) + self.config.selector_guarded_expert_default_fit_bias)
        if score >= 0.0:
            return "interference_control"
        return "plasticity_boost"

    def _task_guarded_expert_boost_default_selector_mode(self) -> str:
        region = self._task_selector_region()
        if region == "sparse_embedded":
            return "plasticity_boost"
        if region == "embedded":
            if self._task_signature_prefers_interference_control():
                return "interference_control"
            return "plasticity_boost"
        return "plasticity_boost"

    def _task_guarded_expert_merge_default_selector_mode(self) -> str:
        region = self._task_selector_region()
        if region == "sparse_embedded":
            return "plasticity_boost"
        if region == "embedded":
            if self._task_signature_prefers_interference_control():
                return "interference_control"
            return "plasticity_boost"
        return "interference_control"

    def _task_guarded_expert_linear_default_selector_mode(self) -> str:
        region = self._task_selector_region()
        if region == "sparse_embedded":
            return "plasticity_boost"
        if region == "embedded":
            if self._task_signature_prefers_interference_control():
                return "interference_control"
            return "plasticity_boost"
        feature_vector = self._task_regime_feature_vector()
        weight_vector = np.array(self.config.selector_guarded_expert_linear_default_fit_weights, dtype=float)
        if len(weight_vector) < len(feature_vector):
            weight_vector = np.pad(weight_vector, (0, len(feature_vector) - len(weight_vector)))
        elif len(weight_vector) > len(feature_vector):
            weight_vector = weight_vector[: len(feature_vector)]
        score = float(
            np.dot(weight_vector, feature_vector) + self.config.selector_guarded_expert_linear_default_fit_bias
        )
        if score >= 0.0:
            return "interference_control"
        return "plasticity_boost"

    def _task_learned_router_selector_mode(self) -> str:
        feature_vector = self._task_regime_feature_vector()
        class_count = 3
        raw_weights = np.array(self.config.selector_router_fit_weights, dtype=float)
        expected_size = class_count * len(feature_vector)
        if len(raw_weights) < expected_size:
            raw_weights = np.pad(raw_weights, (0, expected_size - len(raw_weights)))
        elif len(raw_weights) > expected_size:
            raw_weights = raw_weights[:expected_size]
        weight_matrix = raw_weights.reshape(class_count, len(feature_vector))
        bias_vector = np.array(self.config.selector_router_fit_bias, dtype=float)
        if len(bias_vector) < class_count:
            bias_vector = np.pad(bias_vector, (0, class_count - len(bias_vector)))
        elif len(bias_vector) > class_count:
            bias_vector = bias_vector[:class_count]
        scores = weight_matrix @ feature_vector + bias_vector
        mode_idx = int(np.argmax(scores))
        if mode_idx == 1:
            return "interference_control"
        if mode_idx == 2:
            return "sparse_plasticity_boost"
        return "plasticity_boost"

    def _enable_current_path_boost_for_nonarchived(self) -> None:
        for unit in self.units:
            if not (unit["task_specialist"] and unit["specialist_task"] != self.current_task_id):
                unit["current_path_boost"] = True

    def _set_archived_units_mode(
        self,
        *,
        memory_path_only: bool = False,
        current_path_boost: bool = False,
        consistency_only: bool = False,
        old_path_slow: bool = False,
        retention_only: bool = False,
    ) -> None:
        for unit in self.units:
            is_archived_specialist = unit["task_specialist"] and unit["specialist_task"] != self.current_task_id
            if is_archived_specialist:
                unit["memory_path_only"] = memory_path_only
                unit["current_path_boost"] = current_path_boost
                unit["consistency_only_update"] = consistency_only
                unit["old_path_slow"] = old_path_slow
                unit["retention_only_update"] = retention_only
            else:
                unit["memory_path_only"] = False
                if not current_path_boost:
                    unit["current_path_boost"] = unit.get("current_path_boost", False)
                unit["consistency_only_update"] = False
                unit["old_path_slow"] = False
                unit["retention_only_update"] = False

    def _quantize_tensor(self, weights: np.ndarray, bits: int) -> np.ndarray:
        if bits <= 0:
            return weights
        max_abs = float(np.max(np.abs(weights)))
        if max_abs <= 1e-12:
            return weights
        max_int = max(1, (2 ** (bits - 1)) - 1)
        scale = max_abs / max_int
        quantized = np.clip(np.round(weights / scale), -max_int, max_int)
        return quantized * scale

    def _forward_with_optional_quantization(self, x: np.ndarray, unit: Dict, bits: int = 0) -> np.ndarray:
        W1 = self._quantize_tensor(unit["W1"], bits)
        W2 = self._quantize_tensor(unit["W2"], bits)
        h = np.tanh(x @ W1)
        return h @ W2

    def _limit_archived_entries(self, entries: List[tuple], task_index: int = 0) -> List[tuple]:
        max_count = self.config.archived_specialist_max_count
        if max_count <= 0 or len(entries) <= max_count:
            return entries
        if self.config.archived_specialist_selection_mode == "prototype":
            return self._prototype_compress_entries(entries, task_index, max_count)
        if self.config.archived_specialist_selection_mode == "coverage":
            return self._coverage_select_entries(entries, task_index, max_count)
        return sorted(entries, key=lambda item: item[task_index], reverse=True)[:max_count]

    def _coverage_select_entries(self, entries: List[tuple], task_index: int, max_count: int) -> List[tuple]:
        sorted_entries = sorted(entries, key=lambda item: item[task_index])
        if max_count >= len(sorted_entries):
            return sorted_entries
        selected = [sorted_entries[0], sorted_entries[-1]]
        while len(selected) < max_count:
            remaining = [entry for entry in sorted_entries if entry not in selected]
            if not remaining:
                break
            next_entry = max(
                remaining,
                key=lambda candidate: min(
                    abs(candidate[task_index] - chosen[task_index]) for chosen in selected
                ),
            )
            selected.append(next_entry)
        return sorted(selected, key=lambda item: item[task_index], reverse=True)[:max_count]

    def _prototype_compress_entries(self, entries: List[tuple], task_index: int, max_count: int) -> List[tuple]:
        sorted_entries = sorted(entries, key=lambda item: item[task_index])
        if max_count >= len(sorted_entries):
            return sorted_entries
        n = len(sorted_entries)
        base_size, remainder = divmod(n, max_count)
        groups = []
        start = 0
        for group_idx in range(max_count):
            group_size = base_size + (1 if group_idx < remainder else 0)
            end = start + group_size
            groups.append(sorted_entries[start:end])
            start = end
        prototypes = []
        for group in groups:
            if not group:
                continue
            first = group[0]
            mean_task = int(round(float(np.mean([item[task_index] for item in group]))))
            if len(first) == 2:
                mean_out = np.mean([item[1] for item in group], axis=0)
                prototypes.append((mean_task, mean_out))
                continue
            if len(first) == 4:
                mean_score = float(np.mean([item[1] for item in group]))
                mean_out = np.mean([item[2] for item in group], axis=0)
                mean_merge = float(np.mean([item[3] for item in group]))
                prototypes.append((mean_task, mean_score, mean_out, mean_merge))
                continue
            if len(first) == 8:
                mean_out = np.mean([item[1] for item in group], axis=0)
                mean_merge = float(np.mean([item[2] for item in group]))
                gate_flags = tuple(bool(any(item[idx] for item in group)) for idx in range(3, 8))
                prototypes.append((mean_task, mean_out, mean_merge, *gate_flags))
                continue
            prototypes.append(first)
        return sorted(prototypes, key=lambda item: item[task_index], reverse=True)[:max_count]

    def _limit_archived_units_for_forward(self, units: List[Dict]) -> List[Dict]:
        max_count = self.config.archived_specialist_max_count
        if max_count <= 0 or len(units) <= max_count:
            return units
        if self.config.archived_specialist_selection_mode != "weight_prototype":
            return units
        return self._weight_prototype_units(units, max_count)

    def _key_value_select_units(self, x: np.ndarray, units: List[Dict], max_count: int) -> List[Dict]:
        if max_count <= 0 or len(units) <= max_count:
            return units
        query = np.abs(x)
        query_norm = float(np.linalg.norm(query)) + 1e-8
        scored = []
        for unit in units:
            key = np.mean(np.abs(unit["W1"]), axis=1)
            key_norm = float(np.linalg.norm(key)) + 1e-8
            score = float(np.dot(query, key) / (query_norm * key_norm))
            scored.append((score, unit))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [unit for _, unit in scored[:max_count]]

    def _weight_prototype_units(self, units: List[Dict], max_count: int) -> List[Dict]:
        sorted_units = sorted(units, key=lambda unit: unit.get("specialist_task", -1))
        if max_count >= len(sorted_units):
            return sorted_units
        n = len(sorted_units)
        base_size, remainder = divmod(n, max_count)
        groups = []
        start = 0
        for group_idx in range(max_count):
            group_size = base_size + (1 if group_idx < remainder else 0)
            end = start + group_size
            groups.append(sorted_units[start:end])
            start = end
        prototypes: List[Dict] = []
        for group in groups:
            if not group:
                continue
            first = group[0]
            prototype = dict(first)
            prototype["W1"] = np.mean([unit["W1"] for unit in group], axis=0)
            prototype["W2"] = np.mean([unit["W2"] for unit in group], axis=0)
            prototype["feedback"] = np.mean([unit["feedback"] for unit in group], axis=0)
            prototype["specialist_task"] = int(
                round(float(np.mean([unit.get("specialist_task", -1) for unit in group])))
            )
            prototype["specialist_merge_scale"] = float(
                np.mean([unit.get("specialist_merge_scale", 0.0) for unit in group])
            )
            prototype["retention_merge_bonus"] = float(
                np.mean([unit.get("retention_merge_bonus", 0.0) for unit in group])
            )
            prototype["tension"] = float(np.mean([unit.get("tension", 0.0) for unit in group]))
            prototype["age"] = float(np.mean([unit.get("age", 0.0) for unit in group]))
            prototype["task_specialist"] = any(unit.get("task_specialist", False) for unit in group)
            for key in (
                "confidence_gate",
                "route_by_similarity",
                "soft_route",
                "budgeted_gate",
                "source_aware_gate",
                "memory_path_only",
                "external_memory_route",
                "anchor_regularized",
                "distance_anchor",
                "distill_regularized",
                "role_separated",
                "retention_only_update",
                "current_path_boost",
                "mode_switch_boost",
                "dual_mode_archived_update",
                "decoupled_current_loss",
                "old_path_slow",
                "consistency_only_update",
            ):
                prototype[key] = any(unit.get(key, False) for unit in group)
            if all(unit.get("anchor_W1") is not None for unit in group):
                prototype["anchor_W1"] = np.mean([unit["anchor_W1"] for unit in group], axis=0)
                prototype["anchor_W2"] = np.mean([unit["anchor_W2"] for unit in group], axis=0)
            else:
                prototype["anchor_W1"] = first.get("anchor_W1")
                prototype["anchor_W2"] = first.get("anchor_W2")
            prototypes.append(prototype)
        return sorted(prototypes, key=lambda unit: unit.get("specialist_task", -1), reverse=True)[:max_count]

    def forward(self, x: np.ndarray, inference_mode: bool = True) -> np.ndarray:
        base_outputs = []
        gated_specialists = []
        dual_path_memory_outputs = []
        active_units = [unit for unit in self.units if unit["active"]]
        if (
            self.config.archived_specialist_selection_mode in {"weight_prototype", "key_value"}
            and self.config.archived_specialist_max_count > 0
        ):
            current_units = []
            archived_units = []
            for unit in active_units:
                if unit["task_specialist"] and unit["specialist_task"] != self.current_task_id:
                    archived_units.append(unit)
                else:
                    current_units.append(unit)
            if self.config.archived_specialist_selection_mode == "weight_prototype":
                archived_units = self._limit_archived_units_for_forward(archived_units)
                memory_units = self._limit_archived_units_for_forward(list(self.memory_units))
            else:
                archived_units = self._key_value_select_units(
                    x, archived_units, self.config.archived_specialist_max_count
                )
                memory_units = self._key_value_select_units(
                    x, list(self.memory_units), self.config.archived_specialist_max_count
                )
            active_units = current_units + archived_units
        else:
            memory_units = self.memory_units
        for unit in active_units:
            if unit["active"]:
                quant_bits = 0
                if (
                    self.config.archived_specialist_quantization_bits > 0
                    and unit["task_specialist"]
                    and unit["specialist_task"] != self.current_task_id
                ):
                    quant_bits = self.config.archived_specialist_quantization_bits
                out = self._forward_with_optional_quantization(x, unit, bits=quant_bits)
                if unit["task_specialist"] and unit["specialist_task"] != self.current_task_id:
                    merge_scale = unit.get("specialist_merge_scale", 0.0)
                    if unit.get("retention_merge_bonus", 0.0) > 0.0:
                        task_distance = max(0, self.current_task_id - unit["specialist_task"] - 1)
                        merge_scale *= 1.0 + unit["retention_merge_bonus"] * task_distance
                    if merge_scale <= 0.0:
                        continue
                    if unit.get("memory_path_only", False) and inference_mode:
                        dual_path_memory_outputs.append((unit["specialist_task"], out * merge_scale))
                        continue
                    gated_specialists.append(
                        (
                            unit["specialist_task"],
                            out,
                            merge_scale,
                            unit.get("confidence_gate", False),
                            unit.get("route_by_similarity", False),
                            unit.get("soft_route", False),
                            unit.get("budgeted_gate", False),
                            unit.get("source_aware_gate", False),
                        )
                    )
                    continue
                base_outputs.append(out)
        gated_specialists = self._limit_archived_entries(gated_specialists)
        outputs = list(base_outputs)
        base_mean = (
            np.mean(base_outputs, axis=0)
            if base_outputs
            else np.zeros(self.config.output_size)
        )
        memory_route_candidates = []
        for unit in memory_units:
            quant_bits = max(0, self.config.archived_specialist_quantization_bits)
            out = self._forward_with_optional_quantization(x, unit, bits=quant_bits)
            merge_scale = unit.get("specialist_merge_scale", 0.0)
            if merge_scale <= 0.0:
                continue
            if unit.get("external_memory_route", False):
                similarity = float(np.max(out) - np.min(out))
                alignment = float(np.dot(out, base_mean)) if np.any(base_mean) else 0.0
                memory_route_candidates.append((unit.get("specialist_task", -1), similarity + 0.5 * alignment, out, merge_scale))
            else:
                dual_path_memory_outputs.append((unit.get("specialist_task", -1), out * merge_scale))
        memory_route_candidates = self._limit_archived_entries(memory_route_candidates)
        dual_path_memory_outputs = self._limit_archived_entries(dual_path_memory_outputs)
        confidence = float(np.max(base_mean) - np.min(base_mean))
        routed_candidates = []
        soft_route_candidates = []
        budgeted_candidates = []
        conflict_scores = []
        for (
            _task_id,
            out,
            merge_scale,
            confidence_gate,
            route_by_similarity,
            soft_route,
            budgeted_gate,
            source_aware_gate,
        ) in gated_specialists:
            conflict_scores.append(float(np.mean(np.abs(out - base_mean))))
            if confidence_gate:
                threshold = max(self.config.specialist_confidence_threshold, 1e-6)
                gate_scale = 1.0 - min(confidence / threshold, 1.0)
                gate_scale = max(self.config.specialist_min_gate_scale, gate_scale)
                merge_scale *= gate_scale
            if source_aware_gate:
                disagreement = float(np.mean(np.abs(out - base_mean)))
                specialist_strength = float(np.max(out) - np.min(out))
                source_score = disagreement + self.config.specialist_source_agreement_bonus * specialist_strength
                threshold = max(self.config.specialist_confidence_threshold, 1e-6)
                normalized_score = min(source_score / threshold, 1.0)
                source_scale = max(self.config.specialist_source_conflict_floor, normalized_score)
                merge_scale *= source_scale
            if budgeted_gate:
                similarity = float(np.max(out) - np.min(out))
                budgeted_candidates.append((similarity, out, merge_scale))
                continue
            if route_by_similarity:
                similarity = float(np.max(out) - np.min(out))
                routed_candidates.append((similarity, out, merge_scale))
                continue
            if soft_route:
                similarity = float(np.max(out) - np.min(out))
                alignment = float(np.dot(out, base_mean)) if np.any(base_mean) else 0.0
                score = similarity + 0.5 * alignment
                soft_route_candidates.append((score, out, merge_scale))
                continue
            outputs.append(out * merge_scale)
        if routed_candidates:
            _, out, merge_scale = max(routed_candidates, key=lambda item: item[0])
            threshold = max(self.config.specialist_confidence_threshold, 1e-6)
            route_gate = 1.0 - min(confidence / threshold, 1.0)
            route_gate = max(self.config.specialist_route_min_scale, route_gate)
            outputs.append(out * merge_scale * route_gate)
        if soft_route_candidates:
            scores = np.array([item[0] for item in soft_route_candidates], dtype=float)
            temperature = max(self.config.specialist_softmax_temperature, 1e-6)
            scaled = scores / temperature
            scaled -= np.max(scaled)
            weights = np.exp(scaled)
            weights /= np.sum(weights)
            threshold = max(self.config.specialist_confidence_threshold, 1e-6)
            route_gate = 1.0 - min(confidence / threshold, 1.0)
            route_gate = max(self.config.specialist_route_min_scale, route_gate)
            combined = np.zeros(self.config.output_size)
            for weight, (_, out, merge_scale) in zip(weights, soft_route_candidates):
                combined += weight * out * merge_scale
            outputs.append(combined * route_gate)
        if budgeted_candidates:
            threshold = max(self.config.specialist_confidence_threshold, 1e-6)
            route_gate = 1.0 - min(confidence / threshold, 1.0)
            route_gate = max(self.config.specialist_route_min_scale, route_gate)
            budget_scale = max(0.0, min(1.0, self.config.specialist_budget_scale))
            budget = route_gate * budget_scale
            scores = np.array([item[0] for item in budgeted_candidates], dtype=float)
            total = float(np.sum(scores))
            if total <= 0.0:
                weights = np.full(len(budgeted_candidates), 1.0 / len(budgeted_candidates))
            else:
                weights = scores / total
            combined = np.zeros(self.config.output_size)
            for weight, (_, out, merge_scale) in zip(weights, budgeted_candidates):
                combined += weight * out * merge_scale
            outputs.append(combined * budget)
        mean_conflict = float(np.mean(conflict_scores)) if conflict_scores else 0.0
        max_conflict = float(np.max(conflict_scores)) if conflict_scores else 0.0
        self.last_forward_stats = {
            "confidence": confidence,
            "conflict_score": mean_conflict,
            "conflict_peak": max_conflict,
        }
        if memory_route_candidates:
            _, _, out, merge_scale = max(memory_route_candidates, key=lambda item: item[1])
            threshold = max(self.config.specialist_confidence_threshold, 1e-6)
            route_gate = 1.0 - min(confidence / threshold, 1.0)
            route_gate = max(self.config.specialist_route_min_scale, route_gate)
            dual_path_memory_outputs.append((-1, out * merge_scale * route_gate))
        if dual_path_memory_outputs:
            current_path = (
                np.mean(base_outputs, axis=0)
                if base_outputs
                else np.zeros(self.config.output_size)
            )
            memory_path = np.mean([item[1] for item in dual_path_memory_outputs], axis=0)
            if outputs:
                # Preserve existing gated-path contributions, but keep the current and memory routes structurally separate.
                return np.mean(outputs + [current_path, memory_path], axis=0)
            return np.mean([current_path, memory_path], axis=0)
        if outputs:
            return np.mean(outputs, axis=0)
        return np.zeros(self.config.output_size)

    def learn(self, x: np.ndarray, target: np.ndarray, lr: float = 0.01) -> float:
        out = self.forward(x)
        self.task_stat_sums["confidence"] += self.last_forward_stats.get("confidence", 0.0)
        self.task_stat_sums["conflict_score"] += self.last_forward_stats.get("conflict_score", 0.0)
        self.task_stat_sums["conflict_peak"] += self.last_forward_stats.get("conflict_peak", 0.0)
        self.task_stat_sums["input_abs_mean"] += float(np.mean(np.abs(x)))
        self.task_stat_sums["input_nonnegative_ratio"] += float(np.mean(x >= 0.0))
        self.task_stat_sums["input_zero_ratio"] += float(np.mean(np.isclose(x, 0.0)))
        self.task_stat_sums["steps"] += 1
        train_out = out
        if any(unit.get("decoupled_current_loss", False) for unit in self.units):
            train_out = self.forward(x, inference_mode=False)
        error = target - train_out
        loss_value = float(np.mean(error ** 2))
        self.task_stat_trace["loss"].append(loss_value)
        self.task_stat_trace["confidence"].append(self.last_forward_stats.get("confidence", 0.0))
        self.task_stat_trace["conflict_score"].append(self.last_forward_stats.get("conflict_score", 0.0))
        for unit in self.units:
            if not unit["active"]:
                continue
            unit["tension"] = 0.9 * unit["tension"] + 0.1 * loss_value
            unit["tension_history"].append(unit["tension"])
            if len(unit["tension_history"]) > max(self.config.trend_window, 2):
                unit["tension_history"] = unit["tension_history"][-max(self.config.trend_window, 2) :]
            if unit["frozen"] or unit["frozen_tasks"] > 0:
                continue
            if (
                unit.get("role_separated", False)
                and unit["task_specialist"]
                and unit["specialist_task"] != self.current_task_id
            ):
                continue

            h = np.tanh(x @ unit["W1"])
            unit_error = error
            if (
                unit.get("consistency_only_update", False)
                and unit["task_specialist"]
                and unit["specialist_task"] != self.current_task_id
                and unit.get("anchor_W1") is not None
                and unit.get("anchor_W2") is not None
            ):
                anchor_h = np.tanh(x @ unit["anchor_W1"])
                anchor_out = anchor_h @ unit["anchor_W2"]
                current_out = h @ unit["W2"]
                unit_error = self.config.specialist_distill_strength * (anchor_out - current_out)
            elif (
                (
                    unit.get("retention_only_update", False)
                    or (
                        unit.get("dual_mode_archived_update", False)
                        and self.last_forward_stats.get("conflict_score", 0.0)
                        >= self.config.mode_switch_conflict_threshold
                    )
                )
                and unit["task_specialist"]
                and unit["specialist_task"] != self.current_task_id
                and unit.get("anchor_W1") is not None
                and unit.get("anchor_W2") is not None
            ):
                anchor_h = np.tanh(x @ unit["anchor_W1"])
                anchor_out = anchor_h @ unit["anchor_W2"]
                current_out = h @ unit["W2"]
                unit_error = self.config.specialist_retention_strength * (anchor_out - current_out)
            if (
                unit.get("distill_regularized", False)
                and unit["task_specialist"]
                and unit["specialist_task"] != self.current_task_id
                and unit.get("anchor_W1") is not None
                and unit.get("anchor_W2") is not None
            ):
                anchor_h = np.tanh(x @ unit["anchor_W1"])
                anchor_out = anchor_h @ unit["anchor_W2"]
                current_out = h @ unit["W2"]
                distill_error = anchor_out - current_out
                unit_error = unit_error + self.config.specialist_distill_strength * distill_error
            fb_error = unit["feedback"].T @ unit_error
            unit_lr = lr * unit["lr_scale"] * (1.0 / (1.0 + unit["age"] * 0.1))
            if (
                unit.get("current_path_boost", False)
                and not (
                    unit["task_specialist"]
                    and unit["specialist_task"] != self.current_task_id
                )
            ):
                boost = self.config.current_path_lr_boost
                if unit.get("mode_switch_boost", False):
                    threshold = max(self.config.mode_switch_conflict_threshold, 1e-6)
                    conflict_ratio = min(
                        self.last_forward_stats.get("conflict_score", 0.0) / threshold,
                        1.0,
                    )
                    min_boost = min(
                        self.config.current_path_lr_boost,
                        self.config.mode_switch_min_current_boost,
                    )
                    boost = self.config.current_path_lr_boost - (
                        self.config.current_path_lr_boost - min_boost
                    ) * conflict_ratio
                unit_lr *= boost
            if (
                unit.get("old_path_slow", False)
                and unit["task_specialist"]
                and unit["specialist_task"] != self.current_task_id
            ):
                unit_lr *= self.config.old_path_lr_scale
            unit["W2"] += unit_lr * np.outer(h, unit_error)
            unit["W1"] += unit_lr * np.outer(x, fb_error)
            if (
                unit.get("anchor_regularized", False)
                and unit["task_specialist"]
                and unit["specialist_task"] != self.current_task_id
                and unit.get("anchor_W1") is not None
                and unit.get("anchor_W2") is not None
            ):
                anchor_strength = self.config.specialist_anchor_strength
                if unit.get("distance_anchor", False):
                    task_distance = max(0, self.current_task_id - unit["specialist_task"] - 1)
                    anchor_strength *= 1.0 + self.config.specialist_distance_anchor_bonus * task_distance
                pull = anchor_strength * unit_lr
                unit["W1"] += pull * (unit["anchor_W1"] - unit["W1"])
                unit["W2"] += pull * (unit["anchor_W2"] - unit["W2"])
            unit["age"] += 1
        return float(np.mean(error ** 2))

    def evolve(self) -> List[str]:
        active_indices = [idx for idx, unit in enumerate(self.units) if unit["active"]]
        if not active_indices:
            return []

        avg_tension = float(np.mean([self.units[idx]["tension"] for idx in active_indices]))
        max_units = resolve_max_units(self)

        if avg_tension <= self.config.tension_threshold or len(self.units) >= max_units:
            return []

        special_case = maybe_handle_special_evolve_case(self, active_indices, avg_tension)
        if special_case is not None:
            return special_case

        source_idx = self._source_idx_from_active(active_indices)
        if not self._passes_preclone_gate(active_indices, avg_tension, source_idx):
            return []

        new_idx = self.add_unit(clone_from=source_idx)
        changes = [f"add_clone:{source_idx}->{new_idx}"]

        freeze_window = self._freeze_window()
        if freeze_window > 0:
            self.units[source_idx]["frozen_tasks"] = freeze_window
            changes.append(f"freeze_source:{source_idx}")
        changes.extend(apply_clone_policy(self, source_idx, new_idx, max_units))

        return changes

    def reset_after_task(self) -> None:
        self.current_task_id += 1
        self.task_stat_sums = {
            "confidence": 0.0,
            "conflict_score": 0.0,
            "conflict_peak": 0.0,
            "input_abs_mean": 0.0,
            "input_nonnegative_ratio": 0.0,
            "input_zero_ratio": 0.0,
            "steps": 0,
        }
        self.task_stat_trace = {
            "loss": [],
            "confidence": [],
            "conflict_score": [],
        }
        for unit in self.units:
            unit["tension"] = 0.3
            if unit["frozen_tasks"] > 0:
                unit["frozen_tasks"] -= 1

    def _archive_prior_specialists_to_memory(self, routed: bool = False) -> None:
        retained_units = []
        for unit in self.units:
            if unit["task_specialist"] and unit["specialist_task"] != self.current_task_id:
                archived = {
                    "W1": unit["W1"].copy(),
                    "W2": unit["W2"].copy(),
                    "specialist_merge_scale": unit.get("specialist_merge_scale", 0.0),
                    "specialist_task": unit["specialist_task"],
                    "external_memory_route": routed,
                }
                self.memory_units.append(archived)
                continue
            retained_units.append(unit)
        self.units = retained_units

    def register_task_outcome(self, task_row: Dict) -> None:
        if self.policy != "benefit_limited_clone":
            return

        score = task_row["current_accuracy"] - task_row["mean_forgetting"]
        if self.pending_benefit_eval:
            self.benefit_gate_open = score >= self.config.benefit_score_threshold
            self.pending_benefit_eval = False
            return

        if not self.benefit_gate_open:
            reopen_threshold = self.config.benefit_score_threshold + self.config.benefit_reopen_margin
            if score >= reopen_threshold:
                self.benefit_gate_open = True

    def predict(self, x: np.ndarray) -> int:
        return int(np.argmax(self.forward(x)))

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        correct = sum(1 for i in range(len(X)) if self.predict(X[i]) == int(np.argmax(y[i])))
        return correct / len(X)

    @property
    def active_units(self) -> int:
        return sum(1 for unit in self.units if unit["active"]) + len(self.memory_units)

    @property
    def frozen_units(self) -> int:
        return sum(
            1 for unit in self.units if unit["active"] and (unit["frozen"] or unit["frozen_tasks"] > 0)
        )


