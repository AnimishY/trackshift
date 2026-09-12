"""Train and evaluate a tyre-pace residual model from FastF1 lap data.

Run ``python 13.py`` for a short evaluation using the included 2023 parquet
file. Use ``python 13.py --source fastf1 --years 2022 2023 --full`` when the
FastF1 backends are reachable and a full refresh is wanted.

The code uses FastF1's session-relative ``Time`` rather than ``LapStartDate``.
The latter is optional metadata and is often NaT in cached historical sessions.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Iterable

import fastf1
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

CACHE_DIR = Path("cache")
DEFAULT_PARQUET = Path("ultimate_pure_xgboost_degradation.parquet")
COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}


class PirelliCompoundMapper:
    """Translate FastF1's relative tyre names into the nominated C-scale."""

    ALLOCATIONS = {
        "Bahrain": ("C1", "C2", "C3"), "Saudi Arabia": ("C2", "C3", "C4"),
        "Australia": ("C3", "C4", "C5"), "Azerbaijan": ("C3", "C4", "C5"),
        "Miami": ("C2", "C3", "C4"), "Monaco": ("C3", "C4", "C5"),
        "Spain": ("C1", "C2", "C3"), "Canada": ("C3", "C4", "C5"),
        "Austria": ("C3", "C4", "C5"), "Great Britain": ("C1", "C2", "C3"),
        "Hungary": ("C3", "C4", "C5"), "Belgium": ("C2", "C3", "C4"),
        "Netherlands": ("C1", "C2", "C3"), "Italy": ("C3", "C4", "C5"),
        "Singapore": ("C3", "C4", "C5"), "Japan": ("C1", "C2", "C3"),
        "Qatar": ("C1", "C2", "C3"), "United States": ("C2", "C3", "C4"),
        "Mexico": ("C3", "C4", "C5"), "Brazil": ("C2", "C3", "C4"),
        "Las Vegas": ("C3", "C4", "C5"), "Abu Dhabi": ("C3", "C4", "C5"),
    }

    @classmethod
    def get_absolute_compound(cls, country: str, relative_compound: str) -> str:
        allocation = cls.ALLOCATIONS.get(country, ("C2", "C3", "C4"))
        return {"HARD": allocation[0], "MEDIUM": allocation[1], "SOFT": allocation[2]}.get(
            str(relative_compound).upper(), str(relative_compound)
        )


class F1FeatureEngine:
    """Load FastF1 data and derive features available before a lap ends."""

    def __init__(self, years: Iterable[int]):
        self.years = list(years)

    def extract_data(self) -> pd.DataFrame:
        """Fetch sessions from FastF1. This needs internet access on first use."""
        CACHE_DIR.mkdir(exist_ok=True)
        fastf1.Cache.enable_cache(str(CACHE_DIR))
        all_laps: list[pd.DataFrame] = []

        for year in self.years:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
            for _, event in schedule.iterrows():
                for session_id in ("FP1", "FP2", "FP3", "S", "R"):
                    try:
                        session = event.get_session(session_id)
                        session.load(telemetry=False, weather=True, messages=False)
                        laps = session.laps.reset_index(drop=True).copy()
                        laps = laps[laps["Compound"].isin(COMPOUNDS)].copy()
                        if laps.empty:
                            continue

                        event_key = f"{year}_{event['EventName']}_{session_id}"
                        laps["Year"] = year
                        laps["Country"] = event["Country"]
                        laps["Circuit"] = event["Location"]
                        laps["Session"] = session_id
                        laps["EventKey"] = event_key
                        laps["Stint_ID"] = (
                            event_key + "_" + laps["Driver"].astype(str) + "_" + laps["Stint"].astype(str)
                        )
                        laps["AbsoluteCompound"] = laps.apply(
                            lambda row: PirelliCompoundMapper.get_absolute_compound(row["Country"], row["Compound"]),
                            axis=1,
                        )
                        laps = self._merge_weather(laps, session.weather_data)
                        all_laps.append(self._calculate_traffic_o_n_log_n(laps))
                    except Exception as exc:
                        logging.warning("Skipping %s %s %s: %s", year, event["EventName"], session_id, exc)

        if not all_laps:
            raise RuntimeError("FastF1 returned no usable sessions. Check the connection and cache.")
        return self._engineer_causal_features(pd.concat(all_laps, ignore_index=True))

    @staticmethod
    def _merge_weather(laps: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
        """Attach the latest known weather measurement to each lap start."""
        laps = laps.copy()
        laps["LapStartTime"] = laps["Time"] - laps["LapTime"]
        laps = laps.dropna(subset=["LapStartTime"]).sort_values("LapStartTime")
        weather = weather.dropna(subset=["Time"]).sort_values("Time")
        if weather.empty:
            laps["TrackTemp"] = np.nan
            laps["AirTemp"] = np.nan
            return laps
        merged = pd.merge_asof(
            laps,
            weather[["Time", "TrackTemp", "AirTemp"]],
            left_on="LapStartTime",
            right_on="Time",
            direction="backward",
            suffixes=("", "_weather"),
        )
        return merged.drop(columns=["Time_weather"], errors="ignore")

    @staticmethod
    def _calculate_traffic_o_n_log_n(laps: pd.DataFrame) -> pd.DataFrame:
        """Count overlapping laps with a vectorized sweep-line calculation."""
        laps = laps.copy()
        valid = laps["LapStartTime"].notna() & laps["Time"].notna()
        laps["Traffic_Density"] = np.nan
        if not valid.any():
            return laps
        starts = laps.loc[valid, "LapStartTime"].dt.total_seconds().to_numpy()
        ends = laps.loc[valid, "Time"].dt.total_seconds().to_numpy()
        starts_sorted, ends_sorted = np.sort(starts), np.sort(ends)
        ends_before = np.searchsorted(ends_sorted, starts, side="right")
        starts_after = len(starts) - np.searchsorted(starts_sorted, ends, side="left")
        laps.loc[valid, "Traffic_Density"] = np.maximum(len(starts) - ends_before - starts_after - 1, 0)
        return laps

    @staticmethod
    def _engineer_causal_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features without backfilling from future laps."""
        df = df.copy()
        df["LapTime_s"] = df["LapTime"].dt.total_seconds()
        required = ["LapTime_s", "TyreLife", "Sector1Time", "Sector2Time", "Sector3Time", "Time"]
        df = df.dropna(subset=required)
        if df.empty:
            return df

        df["SessionElapsed_s"] = df["Time"].dt.total_seconds()
        df["S1_s"] = df["Sector1Time"].dt.total_seconds()
        df["S2_s"] = df["Sector2Time"].dt.total_seconds()
        df["S3_s"] = df["Sector3Time"].dt.total_seconds()
        df = df.sort_values(["EventKey", "Session", "SessionElapsed_s", "Driver", "LapNumber"]).reset_index(drop=True)

        compound_groups = ["EventKey", "Session", "AbsoluteCompound"]
        grouped = df.groupby(compound_groups, sort=False)
        # shift() leaves the first row unavailable instead of bfill(), which
        # would leak a future sector time into an earlier prediction.
        df["S1_min"] = grouped["S1_s"].transform(lambda values: values.shift().cummin())
        df["S2_min"] = grouped["S2_s"].transform(lambda values: values.shift().cummin())
        df["S3_min"] = grouped["S3_s"].transform(lambda values: values.shift().cummin())
        df["Theoretical_Baseline"] = df["S1_min"] + df["S2_min"] + df["S3_min"]

        session_groups = df.groupby(["EventKey", "Session"], sort=False)
        df["Session_Best_Pace"] = session_groups["LapTime_s"].transform(lambda values: values.shift().cummin())
        df["Session_Cumulative_Laps"] = session_groups.cumcount()
        df["Fuel_Burn_Proxy"] = df["LapNumber"].fillna(0).astype(float) * 0.03
        df["Residual"] = df["LapTime_s"] - df["Theoretical_Baseline"]

        # The previous TrackEvolution_Pace used the target lap time itself.
        # SessionElapsed_s is known at lap start and captures the same trend safely.
        df = df.dropna(subset=["Theoretical_Baseline", "Session_Best_Pace"])
        df = df[(df["TrackStatus"].astype(str) == "1") & df["Residual"].between(-1.5, 12.0)]
        return df.reset_index(drop=True)


def load_local_parquet(path: Path, year: int) -> pd.DataFrame:
    """Adapt the repository's FastF1-derived parquet data to this model."""
    if not path.is_file():
        raise FileNotFoundError(f"Local data file does not exist: {path}")
    df = pd.read_parquet(path).copy()
    required = {"Time", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time", "Compound", "Driver", "Track"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} is not compatible; it is missing: {', '.join(missing)}")

    df = df[df["Compound"].isin(COMPOUNDS)].copy()
    df["Year"] = df.get("Year", year).fillna(year).astype(int) if "Year" in df else year
    df["Country"] = df["Track"]
    df["Circuit"] = df["Track"]
    df["Session"] = df.get("Session", "R")
    df["Session"] = df["Session"].fillna("R").astype(str)
    df["EventKey"] = df["Year"].astype(str) + "_" + df["Track"].astype(str) + "_" + df["Session"]
    if "Stint_ID" in df:
        df["Stint_ID"] = df["EventKey"] + "_" + df["Stint_ID"].astype(str)
    else:
        df["Stint_ID"] = df["EventKey"] + "_" + df["Driver"].astype(str) + "_" + df["Stint"].astype(str)

    if "Compound_C_Rating" in df:
        rating = pd.to_numeric(df["Compound_C_Rating"], errors="coerce")
        df["AbsoluteCompound"] = rating.map(lambda value: f"C{int(value)}" if pd.notna(value) else np.nan)
    else:
        df["AbsoluteCompound"] = np.nan
    df["AbsoluteCompound"] = df["AbsoluteCompound"].fillna(
        df.apply(lambda row: PirelliCompoundMapper.get_absolute_compound(row["Country"], row["Compound"]), axis=1)
    )
    if "AirTemp" not in df:
        df["AirTemp"] = np.nan
    if "TrackTemp" not in df:
        df["TrackTemp"] = np.nan
    if "LapStartTime" not in df:
        df["LapStartTime"] = df["Time"] - df["LapTime"]

    engine = F1FeatureEngine([year])
    traffic_parts = [engine._calculate_traffic_o_n_log_n(part) for _, part in df.groupby("EventKey", sort=False)]
    return engine._engineer_causal_features(pd.concat(traffic_parts, ignore_index=True))


class CausalPostProcessor:
    """Apply per-stint smoothing using only previous predictions in that stint."""

    @staticmethod
    def apply_causal_smoothing(df: pd.DataFrame, predictions: np.ndarray) -> np.ndarray:
        work = df.copy()
        work["Raw_Pred"] = predictions
        result = pd.Series(index=work.index, dtype=float)
        for _, stint in work.groupby("Stint_ID", sort=False):
            stint = stint.sort_values("SessionElapsed_s")
            smoothed = np.maximum.accumulate(stint["Raw_Pred"].to_numpy())
            result.loc[stint.index] = pd.Series(smoothed).ewm(span=3, adjust=False).mean().to_numpy()
        return result.reindex(df.index).to_numpy()


class F1ModelTuner:
    """Tune a compact ensemble and report leave-one-circuit-out error."""

    NUMERIC_FEATURES = [
        "TyreLife", "TrackTemp", "AirTemp", "Traffic_Density", "Session_Cumulative_Laps",
        "Fuel_Burn_Proxy", "SessionElapsed_s",
    ]
    CATEGORICAL_FEATURES = ["AbsoluteCompound", "Driver"]

    def __init__(self, df: pd.DataFrame, trials: int, tuning_folds: int, evaluation_folds: int | None, stacking_cv: int):
        self.df = df.reset_index(drop=True)
        self.features = self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES
        self.X = self.df[self.features]
        self.y = self.df["Residual"]
        self.groups = self.df["Circuit"]
        self.trials = trials
        self.tuning_folds = tuning_folds
        self.evaluation_folds = evaluation_folds
        self.stacking_cv = stacking_cv
        if self.groups.nunique() < 2:
            raise ValueError("At least two circuits are required for leave-one-circuit-out validation.")

    @staticmethod
    def _preprocessor() -> ColumnTransformer:
        return ColumnTransformer([
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", RobustScaler())]), F1ModelTuner.NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), F1ModelTuner.CATEGORICAL_FEATURES),
        ])

    @staticmethod
    def _xgb_params(trial: optuna.Trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("xgb_n_estimators", 60, 140),
            "learning_rate": trial.suggest_float("xgb_lr", 0.02, 0.12, log=True),
            "max_depth": trial.suggest_int("xgb_max_depth", 3, 6),
            "subsample": trial.suggest_float("xgb_subsample", 0.7, 1.0),
        }

    def objective(self, trial: optuna.Trial) -> float:
        model = Pipeline([
            ("preprocessor", self._preprocessor()),
            ("xgb", xgb.XGBRegressor(**self._xgb_params(trial), objective="reg:absoluteerror", n_jobs=1, random_state=42)),
        ])
        scores = []
        for fold, (train_idx, test_idx) in enumerate(LeaveOneGroupOut().split(self.X, self.y, self.groups)):
            if fold >= self.tuning_folds:
                break
            model.fit(self.X.iloc[train_idx], self.y.iloc[train_idx])
            scores.append(mean_absolute_error(self.y.iloc[test_idx], model.predict(self.X.iloc[test_idx])))
        return float(np.mean(scores))

    def _make_final_model(self, best: dict) -> Pipeline:
        return Pipeline([
            ("preprocessor", self._preprocessor()),
            ("stacking", StackingRegressor(
                estimators=[
                    ("xgb", xgb.XGBRegressor(
                        n_estimators=best["xgb_n_estimators"], learning_rate=best["xgb_lr"],
                        max_depth=best["xgb_max_depth"], subsample=best["xgb_subsample"],
                        objective="reg:absoluteerror", n_jobs=1, random_state=42,
                    )),
                    ("lgb", lgb.LGBMRegressor(n_estimators=100, learning_rate=0.04, max_depth=5,
                                               objective="mae", verbose=-1, n_jobs=1, random_state=42)),
                    ("mlp", MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=100, early_stopping=True,
                                         validation_fraction=0.15, random_state=42)),
                ],
                final_estimator=Ridge(alpha=1.0), cv=self.stacking_cv, n_jobs=1,
            )),
        ])

    def tune_and_evaluate(self) -> Pipeline:
        logging.info("Tuning XGBoost with %s trial(s)...", self.trials)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(self.objective, n_trials=self.trials)
        logging.info("Best XGBoost parameters: %s", study.best_params)
        model = self._make_final_model(study.best_params)
        self._evaluate_loocv(model)
        model.fit(self.X, self.y)
        return model

    def _evaluate_loocv(self, model: Pipeline) -> None:
        logging.info("Executing leave-one-circuit-out validation...")
        scores = []
        for fold, (train_idx, test_idx) in enumerate(LeaveOneGroupOut().split(self.X, self.y, self.groups)):
            if self.evaluation_folds is not None and fold >= self.evaluation_folds:
                break
            circuit = self.groups.iloc[test_idx].iloc[0]
            model.fit(self.X.iloc[train_idx], self.y.iloc[train_idx])
            residual_predictions = CausalPostProcessor.apply_causal_smoothing(
                self.df.iloc[test_idx], model.predict(self.X.iloc[test_idx])
            )
            predictions = self.df["Theoretical_Baseline"].iloc[test_idx].to_numpy() + residual_predictions
            actual = self.df["LapTime_s"].iloc[test_idx].to_numpy()
            score = mean_absolute_error(actual, predictions)
            scores.append(score)
            logging.info("Held out %-16s | MAE %.4fs", circuit, score)
        logging.info("Mean LOCO MAE across %d circuit(s): %.4fs (+/- %.4fs)", len(scores), np.mean(scores), np.std(scores))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("auto", "parquet", "fastf1"), default="auto")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--local-year", type=int, default=2023)
    parser.add_argument("--years", type=int, nargs="+", default=[2022, 2023])
    parser.add_argument("--full", action="store_true", help="Use 10 tuning trials and evaluate every circuit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_parquet = args.source == "parquet" or (args.source == "auto" and args.data_file.is_file())
    if use_parquet:
        logging.info("Loading local FastF1-derived data: %s", args.data_file)
        df = load_local_parquet(args.data_file, args.local_year)
    else:
        logging.info("Loading FastF1 sessions for %s", args.years)
        df = F1FeatureEngine(args.years).extract_data()
    if df.empty:
        raise RuntimeError("No valid green-flag laps remained after feature engineering.")
    logging.info("Dataset ready: %d laps across %d circuit(s)", len(df), df["Circuit"].nunique())

    # The default is intentionally short and reproducible. --full retains the
    # original project's all-circuit/ten-trial objective.
    F1ModelTuner(
        df,
        trials=10 if args.full else 1,
        tuning_folds=4 if args.full else 2,
        evaluation_folds=None if args.full else 4,
        stacking_cv=5 if args.full else 2,
    ).tune_and_evaluate()


if __name__ == "__main__":
    main()
