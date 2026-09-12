"""FastF1 race-lap extraction and transparent fuel correction.

Only green-flag, in-stint dry-compound laps are admitted.  The correction is
kept as a named column so every latent-state observation can be audited back
to its timing source and assumed fuel sensitivity.
"""

from __future__ import annotations

from typing import Mapping

import fastf1
import pandas as pd


RELATIVE_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}


def fuel_penalty_seconds_per_kg(profile: Mapping[str, float]) -> float:
    """Track-scaled public-data starting point for fuel-mass correction."""
    return float(0.018 + 0.0015 * float(profile["downforce"]) + 0.0012 * float(profile["braking"]))


def apply_fuel_correction(laps: pd.DataFrame, profile: Mapping[str, float]) -> pd.DataFrame:
    """Add estimated fuel mass and fuel-corrected lap time to race laps."""
    out = laps.copy()
    race_laps = max(int(pd.to_numeric(out["LapNumber"], errors="coerce").max()), 1)
    penalty = fuel_penalty_seconds_per_kg(profile)
    out["Fuel_Penalty_s_per_kg"] = penalty
    out["Fuel_Weight_kg"] = 110.0 - (out["LapNumber"] - 1) * (110.0 / race_laps)
    out["LapTime_s"] = out["LapTime"].dt.total_seconds()
    out["Fuel_Corrected_LapTime"] = out["LapTime_s"] - out["Fuel_Weight_kg"] * penalty
    return out


def fetch_race_laps(year: int, round_number: int, profile: Mapping[str, float]) -> pd.DataFrame:
    """Fetch usable Sunday race laps with weather matched to each lap."""
    session = fastf1.get_session(year, round_number, "R")
    session.load(telemetry=False, weather=True, messages=False)
    laps = session.laps.reset_index(drop=True).copy()
    laps = laps[
        laps["Compound"].isin(RELATIVE_COMPOUNDS)
        & laps["LapTime"].notna()
        & laps["TyreLife"].notna()
        & laps["LapNumber"].notna()
        & laps["Driver"].notna()
        & laps["PitOutTime"].isna()
        & laps["PitInTime"].isna()
        & (laps["TrackStatus"].astype(str) == "1")
    ].copy()
    if laps.empty:
        return laps
    weather = session.weather_data[["Time", "TrackTemp", "AirTemp"]].dropna(subset=["Time"]).sort_values("Time")
    laps = pd.merge_asof(laps.sort_values("Time"), weather, on="Time", direction="backward")
    return apply_fuel_correction(laps, profile)


def clean_stint_observations(laps: pd.DataFrame, minimum_laps: int = 6) -> pd.DataFrame:
    """Derive robust degradation observations from fuel-corrected stint pace."""
    clean: list[pd.DataFrame] = []
    for _, stint in laps.groupby("Stint_ID", sort=False):
        work = stint.sort_values("TyreLife").copy()
        if len(work) < minimum_laps:
            continue
        median = float(work["Fuel_Corrected_LapTime"].median())
        work = work[work["Fuel_Corrected_LapTime"] <= median * 1.03].copy()
        if len(work) < minimum_laps:
            continue
        warm = work[work["TyreLife"].between(2, 5)]
        base = float((warm if not warm.empty else work.head(3))["Fuel_Corrected_LapTime"].min())
        work["Degradation_Delta"] = (work["Fuel_Corrected_LapTime"] - base).clip(lower=0.0)
        work["Stint_Length"] = len(work)
        clean.append(work)
    return pd.concat(clean, ignore_index=True) if clean else pd.DataFrame()
