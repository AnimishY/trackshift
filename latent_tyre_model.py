"""Probabilistic latent tyre-state inference.

The observed signal is a fuel-corrected lap-time degradation delta.  It is
noisy: traffic, a lock-up, DRS trains and driver errors are not tyre wear.  A
two-state Kalman filter therefore estimates the unobserved degradation and
its per-lap growth rate, carrying a covariance matrix with every estimate.

This deliberately uses NumPy rather than FilterPy so the hackathon demo has a
small, reproducible deployment surface.  The equations are the standard
linear Gaussian state-space update; innovation clipping provides a practical
robust observation model for one-sided slow-lap outliers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, sqrt
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


@dataclass
class LatentTyreState:
    """Posterior state at a tyre age, in seconds and seconds per lap."""

    tyre_life: float
    mean_s: float
    rate_s_per_lap: float
    covariance: np.ndarray = field(repr=False)

    @property
    def std_s(self) -> float:
        return float(sqrt(max(float(self.covariance[0, 0]), 1e-12)))


@dataclass
class LatentTyreModel:
    """A fitted prior and online Kalman filter for tyre degradation."""

    prior_rate_s_per_lap: float = 0.045
    initial_std_s: float = 0.18
    initial_rate_std: float = 0.025
    process_std_s: float = 0.018
    process_rate_std: float = 0.006
    observation_std_s: float = 0.16
    innovation_clip_sigma: float = 2.8

    def initial_state(self, tyre_life: float = 1.0, wear_multiplier: float = 1.0) -> LatentTyreState:
        rate = max(0.0, self.prior_rate_s_per_lap * float(wear_multiplier))
        covariance = np.diag([self.initial_std_s**2, self.initial_rate_std**2])
        return LatentTyreState(float(tyre_life), 0.0, rate, covariance)

    def predict(self, state: LatentTyreState, tyre_life: float) -> LatentTyreState:
        """Propagate posterior to ``tyre_life`` without seeing a lap time."""
        step = max(0.0, float(tyre_life) - state.tyre_life)
        transition = np.array([[1.0, step], [0.0, 1.0]])
        # Process uncertainty grows with the distance forecast.
        q = np.diag([
            (self.process_std_s * max(step, 1.0)) ** 2,
            (self.process_rate_std * sqrt(max(step, 1.0))) ** 2,
        ])
        mean = transition @ np.array([state.mean_s, state.rate_s_per_lap])
        covariance = transition @ state.covariance @ transition.T + q
        return LatentTyreState(float(tyre_life), float(max(0.0, mean[0])), float(max(0.0, mean[1])), covariance)

    def update(self, predicted: LatentTyreState, observed_delta_s: float) -> LatentTyreState:
        """Assimilate a fuel-corrected degradation observation robustly."""
        observation = max(0.0, float(observed_delta_s))
        h = np.array([1.0, 0.0])
        innovation = observation - predicted.mean_s
        innovation_std = sqrt(max(float(h @ predicted.covariance @ h + self.observation_std_s**2), 1e-12))
        # A clipped innovation has the same intention as a heavy-tailed/
        # skewed observation model: do not let an anomalously slow lap set the
        # tyre state.  The original lap remains visible in validation charts.
        innovation = float(np.clip(innovation, -self.innovation_clip_sigma * innovation_std, self.innovation_clip_sigma * innovation_std))
        gain = (predicted.covariance @ h) / (float(h @ predicted.covariance @ h) + self.observation_std_s**2)
        mean = np.array([predicted.mean_s, predicted.rate_s_per_lap]) + gain * innovation
        identity = np.eye(2)
        # Joseph form keeps covariance numerically positive semi-definite.
        residual = identity - np.outer(gain, h)
        covariance = residual @ predicted.covariance @ residual.T + np.outer(gain, gain) * self.observation_std_s**2
        return LatentTyreState(predicted.tyre_life, float(max(0.0, mean[0])), float(max(0.0, mean[1])), covariance)

    def infer_stint(
        self,
        observations: pd.DataFrame,
        *,
        tyre_life_column: str = "TyreLife",
        observed_column: str = "Degradation_Delta",
        wear_multiplier: float = 1.0,
    ) -> pd.DataFrame:
        """Run chronological posterior updates and return uncertainty columns."""
        if observations.empty:
            return observations.copy()
        
        # Remove any existing inference columns to avoid duplicates on re-inference
        inference_cols = ["Latent_Degradation_Mean", "Latent_Degradation_Std", "Prior_Degradation_Mean", "Prior_Degradation_Std", "Latent_Degradation_Rate"]
        work = observations.drop(columns=[col for col in inference_cols if col in observations.columns]).sort_values(tyre_life_column).copy()
        
        state = self.initial_state(float(work[tyre_life_column].iloc[0]), wear_multiplier)
        results: list[dict[str, float]] = []
        for _, row in work.iterrows():
            predicted = self.predict(state, float(row[tyre_life_column]))
            prior_mean, prior_std = predicted.mean_s, predicted.std_s
            state = self.update(predicted, float(row[observed_column]))
            results.append({
                "Latent_Degradation_Mean": state.mean_s,
                "Latent_Degradation_Std": state.std_s,
                "Prior_Degradation_Mean": prior_mean,
                "Prior_Degradation_Std": prior_std,
                "Latent_Degradation_Rate": state.rate_s_per_lap,
            })
        return pd.concat([work.reset_index(drop=True), pd.DataFrame(results)], axis=1)

    def forecast(self, state: LatentTyreState, future_tyre_lives: Iterable[float], cliff_threshold_s: float = 1.5) -> pd.DataFrame:
        """Forecast a fan chart and probability of crossing a tyre-cliff limit."""
        rows = []
        for life in future_tyre_lives:
            future = self.predict(state, float(life))
            std = future.std_s
            z = (float(cliff_threshold_s) - future.mean_s) / std
            probability = 1.0 - _normal_cdf(z)
            rows.append({
                "TyreLife": float(life),
                "mean_s": future.mean_s,
                "std_s": std,
                "p15_s": max(0.0, future.mean_s - 1.036 * std),
                "p85_s": future.mean_s + 1.036 * std,
                "p05_s": max(0.0, future.mean_s - 1.645 * std),
                "p95_s": future.mean_s + 1.645 * std,
                "cliff_probability": float(np.clip(probability, 0.0, 1.0)),
            })
        return pd.DataFrame(rows)


def fit_latent_tyre_model(dataset: pd.DataFrame) -> LatentTyreModel:
    """Estimate conservative global priors from clean historical stints.

    A robust median slope supplies the prior transition rate. Residual spread
    determines state/observation uncertainty, avoiding a misleading fixed
    confidence band across all tracks.
    """
    slopes: list[float] = []
    residuals: list[float] = []
    for _, stint in dataset.groupby("Stint_ID", sort=False):
        work = stint.sort_values("TyreLife")
        x = pd.to_numeric(work["TyreLife"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(work["Degradation_Delta"], errors="coerce").to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 4 or np.ptp(x[valid]) == 0:
            continue
        slope, intercept = np.polyfit(x[valid], y[valid], 1)
        slopes.append(max(0.002, float(slope)))
        residuals.extend((y[valid] - (intercept + slope * x[valid])).tolist())
    if not slopes:
        return LatentTyreModel()
    rate = float(np.median(slopes))
    residual_scale = float(1.4826 * np.median(np.abs(np.asarray(residuals) - np.median(residuals)))) if residuals else 0.16
    observation = float(np.clip(residual_scale, 0.08, 0.40))
    return LatentTyreModel(
        prior_rate_s_per_lap=rate,
        observation_std_s=observation,
        initial_rate_std=float(np.clip(np.std(slopes), 0.01, 0.10)),
        process_std_s=max(0.012, observation * 0.11),
        process_rate_std=max(0.003, float(np.std(slopes)) * 0.20),
    )


def first_cliff_forecast(forecast: pd.DataFrame, probability: float = 0.80) -> float | None:
    """Return the first predicted tyre age with at least ``probability`` risk."""
    matches = forecast[forecast["cliff_probability"] >= probability]
    return None if matches.empty else float(matches["TyreLife"].iloc[0])


def infer_dataset(dataset: pd.DataFrame, model: LatentTyreModel) -> pd.DataFrame:
    """Infer every stint independently, retaining chronological observations."""
    inferred = [model.infer_stint(stint) for _, stint in dataset.groupby("Stint_ID", sort=False)]
    return pd.concat(inferred, ignore_index=True) if inferred else dataset.copy()
