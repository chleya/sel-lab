# -*- coding: utf-8 -*-
"""
Shared runtime helpers for SEL-Lab.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def ensure_dir(path: Path | str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def results_dir(*parts: str) -> Path:
    return ensure_dir(project_root() / "results" / Path(*parts))


def resolve_results_path(filename: str, *parts: str) -> Path:
    directory = results_dir(*parts) if parts else results_dir()
    return directory / filename


def resolve_canonical_results_path(filename: str) -> Path:
    return resolve_results_path(filename, "canonical")


def resolve_smoke_results_path(filename: str) -> Path:
    return resolve_results_path(filename, "smoke")


def resolve_exploratory_results_path(filename: str) -> Path:
    return resolve_results_path(filename, "exploratory")


def resolve_existing_results_path(filename: str, preferred_parts: Sequence[str] = ("canonical",)) -> Path:
    candidates = [resolve_results_path(filename, *preferred_parts)]
    candidates.append(resolve_results_path(filename))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def save_json(data: Dict[str, Any], filepath: Path | str) -> Path:
    target = Path(filepath)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    return target


def load_json(filepath: Path | str) -> Dict[str, Any]:
    with Path(filepath).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json_result(filename: str, preferred_parts: Sequence[str] = ("canonical",)) -> Dict[str, Any]:
    return load_json(resolve_existing_results_path(filename, preferred_parts))


def split_train_test(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split = int(len(X) * train_ratio)
    return X[:split], y[:split], X[split:], y[split:]


def dataclass_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


def history_to_records(history: Sequence[Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for item in history:
        converted = dataclass_to_dict(item)
        if not isinstance(converted, dict):
            raise TypeError(f"History items must serialize to dict, got {type(converted)!r}")
        records.append(converted)
    return records


def summarize_scalar_runs(results: Iterable[Dict[str, Any]], key: str) -> Dict[str, float]:
    values = [float(item[key]) for item in results]
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
