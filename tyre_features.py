"""Transparent tyre-wear feature engineering shared by the model and portal.

The model learns the circuit and tyre-age baseline.  The wearable-load formula
then exposes the variables an engineer can observe or tune on the car, rather
than treating a setup change as an unexplained model output.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


PROFILE_COLUMNS = [
    "traction", "asphalt_grip", "asphalt_abrasion", "track_evolution",
    "tyre_stress", "braking", "lateral", "downforce",
]

ENGINEERING_FEATURE_COLUMNS = [
    "Tyre_Thermal_Stress", "Pressure_Deviation_Index", "Slip_Energy_Index",
    "Lateral_Load_Index", "Brake_Thermal_Index", "Aero_Load_Index",
]

# Values are deliberately normalised around 1.0. A score of 1.0 represents a
# healthy reference lap; >1 raises degradation and <1 reduces it.
WEAR_WEIGHTS = {
    "thermal": 0.16,
    "pressure": 0.10,
    "slip": 0.16,
    "lateral": 0.14,
    "brake": 0.08,
    "vertical": 0.08,
    "camber": 0.06,
    "toe": 0.04,
    "differential": 0.05,
    "brake_bias": 0.04,
    "aero": 0.09,
}

# These are priors, not claims of exact vehicle-specific coefficients.  Their
# uncertainty is intentionally exposed so a weekend update can move only when
# supported by clean observations.
WEAR_WEIGHT_PRIOR_STD = {
    "thermal": 0.045, "pressure": 0.035, "slip": 0.045, "lateral": 0.040,
    "brake": 0.030, "vertical": 0.030, "camber": 0.025, "toe": 0.020,
    "differential": 0.025, "brake_bias": 0.020, "aero": 0.030,
}


def _numeric_column(frame: pd.DataFrame, name: str, fallback: float) -> pd.Series:
    """Return a numeric frame column, including a safe scalar fallback."""
    raw = frame[name] if name in frame else pd.Series(fallback, index=frame.index)
    return pd.to_numeric(raw, errors="coerce").fillna(fallback)


def derive_race_sensor_proxies(frame: pd.DataFrame) -> pd.DataFrame:
    """Add reproducible sensor-load proxies when historical CAN data is absent.

    FastF1 timing data does not expose a team's raw tyre CAN channels.  These
    features use *only* the supplied track CSV plus weather/fuel fields so a
    retrained model has a consistent, auditable baseline. Live or simulated
    sensor values in the portal replace these proxies at decision time.
    """
    out = frame.copy()
    track_temp = _numeric_column(out, "TrackTemp", 35.0)
    traction = _numeric_column(out, "traction", 3.0)
    abrasion = _numeric_column(out, "asphalt_abrasion", 3.0)
    stress = _numeric_column(out, "tyre_stress", 3.0)
    braking = _numeric_column(out, "braking", 3.0)
    lateral = _numeric_column(out, "lateral", 3.0)
    downforce = _numeric_column(out, "downforce", 3.0)

    out["Tyre_Thermal_Stress"] = 1.0 + (track_temp.sub(34.0).abs() / 20.0) + 0.08 * stress.sub(3.0).clip(lower=0)
    out["Pressure_Deviation_Index"] = 1.0 + (track_temp.sub(32.0).abs() / 28.0) + 0.05 * abrasion.sub(3.0).clip(lower=0)
    out["Slip_Energy_Index"] = (0.45 * traction + 0.35 * stress + 0.20 * abrasion) / 3.0
    out["Lateral_Load_Index"] = (lateral * downforce) / 9.0
    out["Brake_Thermal_Index"] = (braking / 3.0) * (1.0 + (track_temp - 30.0).clip(lower=0) / 80.0)
    out["Aero_Load_Index"] = downforce / 3.0
    return out


def default_sensor_state(profile: Mapping[str, float], track_temp: float = 35.0) -> dict[str, float]:
    """Return a plausible, neutral reference state for a simulated F1 car."""
    return {
        "carcass_temp_c": round(91.0 + (track_temp - 30.0) * 0.22 + (float(profile["tyre_stress"]) - 3) * 1.2, 1),
        "pressure_psi": 21.5,
        "wheel_slip_pct": round(5.0 + (float(profile["traction"]) - 3) * 0.45, 1),
        "lateral_g": round(4.25 + (float(profile["lateral"]) - 3) * 0.18, 2),
        "brake_temp_c": round(640 + (float(profile["braking"]) - 3) * 32, 0),
        "vertical_load_kg": round(2980 + (float(profile["downforce"]) - 3) * 110, 0),
        "camber_deg": -3.0,
        "toe_deg": 0.10,
        "diff_lock_pct": 55.0,
        "brake_bias_pct": 56.0,
        "aero_balance_pct": 70.0,
    }


def wear_load_breakdown(sensor: Mapping[str, float], weights: Mapping[str, float] | None = None) -> dict[str, float]:
    """Calculate the transparent wear-load multiplier and its components.

    Formula: multiplier = clip(0.70 + 0.30 * sum(weight_i * score_i), .70,
    1.55).  Each score is centred at 1.0 at its reference operating point.
    """
    value = lambda key: float(sensor[key])
    scores = {
        "thermal": 1.0 + (abs(value("carcass_temp_c") - 92.0) / 12.0) ** 1.55,
        "pressure": 1.0 + (abs(value("pressure_psi") - 21.5) / 1.25) ** 1.35,
        "slip": max(0.45, value("wheel_slip_pct") / 5.5) ** 1.25,
        "lateral": max(0.50, value("lateral_g") / 4.35) ** 1.20,
        "brake": 1.0 + max(0.0, value("brake_temp_c") - 650.0) / 260.0,
        "vertical": max(0.50, value("vertical_load_kg") / 3000.0),
        "camber": max(0.55, abs(value("camber_deg")) / 3.0),
        "toe": max(0.50, abs(value("toe_deg")) / 0.10),
        "differential": max(0.55, value("diff_lock_pct") / 55.0),
        "brake_bias": 1.0 + abs(value("brake_bias_pct") - 56.0) / 3.0,
        "aero": max(0.55, value("aero_balance_pct") / 70.0),
    }
    active_weights = WEAR_WEIGHTS if weights is None else weights
    if set(active_weights) != set(WEAR_WEIGHTS):
        raise ValueError("weights must contain exactly the virtual-sensor wear components")
    total_weight = float(sum(active_weights.values()))
    if total_weight <= 0:
        raise ValueError("weights must have a positive sum")
    wear_score = sum((active_weights[name] / total_weight) * scores[name] for name in WEAR_WEIGHTS)
    multiplier = float(np.clip(0.70 + 0.30 * wear_score, 0.70, 1.55))
    return {"wear_score": float(wear_score), "multiplier": multiplier, **scores}


def update_wear_weight_priors(
    prior_weights: Mapping[str, float],
    evidence: Mapping[str, float],
    learning_rate: float = 0.12,
) -> dict[str, float]:
    """Make a bounded, explainable weekend update to the physical priors.

    ``evidence`` is a relative component signal inferred from clean FP laps;
    values above one strengthen a component, values below one weaken it.  The
    update is deliberately conservative and renormalised to preserve a 1.0
    total weight, rather than overfitting a handful of Friday observations.
    """
    if not 0.0 <= learning_rate <= 1.0:
        raise ValueError("learning_rate must be between 0 and 1")
    updated = {
        name: max(0.005, float(prior_weights[name]) * (1.0 + learning_rate * (float(evidence.get(name, 1.0)) - 1.0)))
        for name in WEAR_WEIGHTS
    }
    scale = sum(WEAR_WEIGHTS.values()) / sum(updated.values())
    return {name: value * scale for name, value in updated.items()}
