from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


APPLIANCE_THRESHOLDS = {
    "Washing_Machine_Usage": 6.0,  # hours / day
    "AC_Usage": 6.0,               # hours / day
    "Heater_Usage": 2.0,           # hours / day
}


@dataclass
class ApplianceInsight:
    name: str
    usage_hours: float
    relative_importance: float
    suggestion: str


def appliance_contributions(
    feature_importances: Dict[str, float],
    input_features: Dict[str, float],
) -> List[ApplianceInsight]:
    """
    Map feature importances and user input into per‑appliance insights.
    """
    results: List[ApplianceInsight] = []

    for feature_name, importance in feature_importances.items():
        if feature_name not in input_features:
            continue
        usage = float(input_features[feature_name])
        if "Usage" not in feature_name and "AC_Usage" not in feature_name:
            # Skip non‑appliance features like Units_Consumed, Temperature here
            continue

        threshold = APPLIANCE_THRESHOLDS.get(feature_name, None)
        suggestion_parts: List[str] = []

        if threshold is not None and usage > threshold:
            # Simple heuristic: suggest cutting down ~20–30%
            reduce_by = max(1.0, round((usage - threshold) * 0.5, 1))
            suggestion_parts.append(
                f"Reduce {feature_name.replace('_', ' ').lower()} by about {reduce_by:.1f} hours per day."
            )

        if feature_name == "Washing_Machine_Usage":
            suggestion_parts.append("Run only full loads and avoid half‑loads.")
        if feature_name == "AC_Usage":
            suggestion_parts.append("Increase thermostat by 1–2°C and use fans where possible.")
        if feature_name == "Heater_Usage":
            suggestion_parts.append("Improve room insulation and limit heater to peak cold hours.")

        if not suggestion_parts:
            suggestion_parts.append("Current usage looks reasonable; maintain efficient habits.")

        results.append(
            ApplianceInsight(
                name=feature_name,
                usage_hours=usage,
                relative_importance=float(importance),
                suggestion=" ".join(suggestion_parts),
            )
        )

    # Sort descending by importance
    results.sort(key=lambda x: x.relative_importance, reverse=True)
    return results


def normalize_importances(raw_importances: Dict[str, float]) -> Dict[str, float]:
    vals = np.array(list(raw_importances.values()), dtype=float)
    total = float(vals.sum())
    if total == 0.0:
        return {k: 0.0 for k in raw_importances}
    return {k: float(v / total) for k, v in raw_importances.items()}

