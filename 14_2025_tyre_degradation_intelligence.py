"""Tyre degradation intelligence pipeline for the 2025 F1 season.

This script:
1) downloads race laps for every available 2025 grand prix via FastF1,
2) merges external track/tyre-allocation factors from a CSV,
3) trains a combined-driver degradation model,
4) reports MAE and saves tyre cliff / loss-per-lap analysis,
5) saves a reusable model bundle for later race validation.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
from pathlib import Path
from typing import Iterable

import fastf1
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

CACHE_DIR = Path("cache")
DEFAULT_MODEL_PATH = Path("models/tyre_degradation_2025.pkl")
DEFAULT_OUTPUT_DIR = Path("outputs")
RELATIVE_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}

PROFILE_COLUMNS = [
    "traction",
    "asphalt_grip",
    "asphalt_abrasion",
    "track_evolution",
    "tyre_stress",
    "braking",
    "lateral",
    "downforce",
]

ALIAS_MAP = {
    "GREAT BRITAIN": "BRITISH",
    "NETHERLANDS": "DUTCH",
    "ITALY": "ITALIAN",
    "BELGIUM": "BELGIAN",
    "AUSTRIA": "AUSTRIAN",
    "SPAIN": "SPANISH",
    "JAPAN": "JAPANESE",
    "AUSTRALIA": "AUSTRALIAN",
    "MEXICO": "MEXICO CITY",
    "UNITED STATES": "TEXAS",
    "BRAZIL": "SAO PAULO",
    "CHINA": "CHINESE",
}


def normalize_name(value: str) -> str:
    text = re.sub(r"[^A-Z0-9]+", " ", str(value).upper()).strip()
    return re.sub(r"\s+", " ", text)


def parse_compound_triplet(value: str) -> dict[str, int]:
    digits = [int(ch) for ch in str(value) if ch.isdigit()]
    if len(digits) < 3:
        raise ValueError(f"Invalid tyre compound triplet: {value}")
    return {"HARD": digits[0], "MEDIUM": digits[1], "SOFT": digits[2]}


def load_track_profiles(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    rename_map = {
        "RACE": "race",
        "race": "race",
        "tyre compounds": "tyre_compounds",
        "tyre_compounds": "tyre_compounds",
        "traction": "traction",
        "asphalt grip": "asphalt_grip",
        "asphalt_grip": "asphalt_grip",
        "asphalt abrasion": "asphalt_abrasion",
        "asphalt_abrasion": "asphalt_abrasion",
        "track evolution": "track_evolution",
        "track_evolution": "track_evolution",
        "tyre stress": "tyre_stress",
        "tyre_stress": "tyre_stress",
        "braking": "braking",
        "lateral": "lateral",
        "downforce": "downforce",
    }
    df.columns = [rename_map.get(col, col.strip().lower()) for col in df.columns]
    required = {"race", "tyre_compounds", *PROFILE_COLUMNS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Track-factor CSV missing required columns: {missing}")

    df["race_key"] = df["race"].map(normalize_name)
    for col in PROFILE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[PROFILE_COLUMNS].isna().any().any():
        raise ValueError("Track-factor CSV has non-numeric values in profile columns.")

    df["allocation"] = df["tyre_compounds"].map(parse_compound_triplet)
    return df


def resolve_profile(event_name: str, country: str, profiles: pd.DataFrame) -> pd.Series:
    race_index = profiles.set_index("race_key")
    candidates = [normalize_name(event_name), normalize_name(country)]
    alias_candidates = []
    for candidate in candidates:
        alias = ALIAS_MAP.get(candidate)
        if alias:
            alias_candidates.append(normalize_name(alias))
    for key in [*candidates, *alias_candidates]:
        if key in race_index.index:
            return race_index.loc[key]

    default_row = profiles[PROFILE_COLUMNS].median(numeric_only=True)
    default_row["allocation"] = {"HARD": 2, "MEDIUM": 3, "SOFT": 4}
    logging.warning("No track profile for %s / %s. Using fallback profile.", event_name, country)
    return default_row


def prepare_fastf1_cache() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))


def fetch_event_race_laps(year: int, event_row: pd.Series, profile: pd.Series) -> pd.DataFrame:
    event_name = str(event_row["EventName"])
    round_number = int(event_row["RoundNumber"])
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
    laps = laps.sort_values("Time")
    laps = pd.merge_asof(
        laps,
        weather,
        on="Time",
        direction="backward",
    )

    race_laps = int(laps["LapNumber"].max())
    race_laps = max(race_laps, 1)
    fuel_penalty = float(0.018 + 0.0015 * profile["downforce"] + 0.0012 * profile["braking"])
    laps["Fuel_Weight_kg"] = 110.0 - (laps["LapNumber"] - 1) * (110.0 / race_laps)
    laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()
    laps["Fuel_Corrected_LapTime"] = laps["LapTime_s"] - laps["Fuel_Weight_kg"] * fuel_penalty

    allocation = profile["allocation"]
    laps["Compound_C_Rating"] = laps["Compound"].map(allocation).astype(float)

    laps["Race"] = event_name
    laps["Country"] = event_row["Country"]
    laps["Circuit"] = event_row["Location"]
    laps["Stint_ID"] = (
        str(year) + "_" + event_name + "_" + laps["Driver"].astype(str) + "_" + laps["Stint"].astype(str)
    )
    for col in PROFILE_COLUMNS:
        laps[col] = float(profile[col])
    return laps


def build_clean_degradation_dataset(df: pd.DataFrame) -> pd.DataFrame:
    clean_stints: list[pd.DataFrame] = []
    for stint_id, stint in df.groupby("Stint_ID", sort=False):
        stint = stint.sort_values("TyreLife").copy()
        if len(stint) < 6:
            continue
        median_pace = float(stint["Fuel_Corrected_LapTime"].median())
        stint = stint[stint["Fuel_Corrected_LapTime"] <= median_pace * 1.03].copy()
        if len(stint) < 6:
            continue
        warmup = stint[(stint["TyreLife"] >= 2) & (stint["TyreLife"] <= 5)]
        if warmup.empty:
            warmup = stint.head(3)
        base = float(warmup["Fuel_Corrected_LapTime"].min())
        stint["Degradation_Delta"] = (stint["Fuel_Corrected_LapTime"] - base).clip(lower=0.0)
        stint["Stint_Length"] = len(stint)
        clean_stints.append(stint)

    if not clean_stints:
        return pd.DataFrame()
    clean = pd.concat(clean_stints, ignore_index=True)
    clean["TyreLife"] = pd.to_numeric(clean["TyreLife"], errors="coerce")
    clean["TrackTemp"] = clean["TrackTemp"].fillna(clean["TrackTemp"].median())
    clean["AirTemp"] = clean["AirTemp"].fillna(clean["AirTemp"].median())
    return clean.dropna(subset=["TyreLife", "Compound_C_Rating", "Degradation_Delta"])


def get_features() -> tuple[list[str], list[str]]:
    numeric = [
        "TyreLife",
        "Compound_C_Rating",
        "TrackTemp",
        "AirTemp",
        "Fuel_Weight_kg",
        "Stint_Length",
        "TyreLife_x_Compound",
        "TyreLife_x_Abrasion",
        "TyreLife_x_Stress",
        "Temp_x_Stress",
        "Grip_x_Traction",
        *PROFILE_COLUMNS,
    ]
    categorical = ["Driver", "Race"]
    return numeric, categorical


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["TyreLife_x_Compound"] = enriched["TyreLife"] * enriched["Compound_C_Rating"]
    enriched["TyreLife_x_Abrasion"] = enriched["TyreLife"] * enriched["asphalt_abrasion"]
    enriched["TyreLife_x_Stress"] = enriched["TyreLife"] * enriched["tyre_stress"]
    enriched["Temp_x_Stress"] = enriched["TrackTemp"] * enriched["tyre_stress"]
    enriched["Grip_x_Traction"] = enriched["asphalt_grip"] * enriched["traction"]
    return enriched


def make_onehot_encoder() -> OneHotEncoder:
    """Build a dense OneHotEncoder across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class DynamicWeightedRegressor(BaseEstimator, RegressorMixin):
    """Learns ensemble weights from cross-validated base-model MAE."""

    def __init__(self, estimators: list[tuple[str, BaseEstimator]], cv_splits: int = 4, random_state: int = 42):
        self.estimators = estimators
        self.cv_splits = cv_splits
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DynamicWeightedRegressor":
        X_arr = np.asarray(X)
        y_arr = np.asarray(y, dtype=float)
        self.estimators_ = [(name, clone(model)) for name, model in self.estimators]
        maes_used = [np.nan] * len(self.estimators_)
        if len(self.estimators_) == 1 or len(y_arr) < 20:
            self.weights_ = np.array([1.0 / len(self.estimators_)] * len(self.estimators_), dtype=float)
        else:
            splits = max(2, min(self.cv_splits, len(y_arr) // 8))
            kf = KFold(n_splits=splits, shuffle=True, random_state=self.random_state)
            maes = []
            for _, model in self.estimators_:
                fold_maes = []
                for train_idx, test_idx in kf.split(X_arr):
                    model_fold = clone(model)
                    model_fold.fit(X_arr[train_idx], y_arr[train_idx])
                    pred = np.clip(model_fold.predict(X_arr[test_idx]), 0.0, None)
                    fold_maes.append(mean_absolute_error(y_arr[test_idx], pred))
                maes.append(float(np.mean(fold_maes)))
            maes_used = maes
            mae_arr = np.array(maes, dtype=float)
            inv = 1.0 / np.clip(mae_arr, 1e-6, None)
            self.weights_ = inv / inv.sum()
        self.model_cv_mae_ = {
            name: (float(mae) if not np.isnan(mae) else np.nan) for (name, _), mae in zip(self.estimators_, maes_used)
        }
        self.dynamic_weights_ = {name: float(weight) for (name, _), weight in zip(self.estimators_, self.weights_)}
        for _, model in self.estimators_:
            model.fit(X_arr, y_arr)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X)
        preds = np.column_stack([np.clip(model.predict(X_arr), 0.0, None) for _, model in self.estimators_])
        return preds @ self.weights_


def build_model_pipeline() -> Pipeline:
    numeric, categorical = get_features()
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_onehot_encoder())]), categorical),
        ]
    )
    model = DynamicWeightedRegressor(
        estimators=[
            ("gbr", GradientBoostingRegressor(random_state=42, n_estimators=450, learning_rate=0.03, max_depth=3)),
            ("hgb", HistGradientBoostingRegressor(random_state=42, max_depth=8, learning_rate=0.03, max_iter=450, min_samples_leaf=20)),
            ("rf", RandomForestRegressor(random_state=42, n_estimators=500, min_samples_leaf=3, n_jobs=-1)),
            ("etr", ExtraTreesRegressor(random_state=42, n_estimators=500, min_samples_leaf=2, n_jobs=-1)),
        ],
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def evaluate_group_cv(df: pd.DataFrame, pipeline: Pipeline) -> tuple[float, pd.DataFrame]:
    numeric, categorical = get_features()
    features = numeric + categorical
    X = df[features]
    y = df["Degradation_Delta"].to_numpy()
    groups = df["Race"]
    n_groups = groups.nunique()
    splits = min(5, n_groups)
    if splits < 2:
        predictions = np.clip(pipeline.fit(X, y).predict(X), 0.0, None)
        mae = float(mean_absolute_error(y, predictions))
        race_mae = pd.DataFrame({"Race": [groups.iloc[0]], "MAE": [mae], "Laps": [len(df)]})
        return mae, race_mae

    gkf = GroupKFold(n_splits=splits)
    oof = np.full(shape=len(df), fill_value=np.nan, dtype=float)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]
        pipeline.fit(X_train, y_train)
        pred = np.clip(pipeline.predict(X_test), 0.0, None)
        oof[test_idx] = pred
        fold_mae = float(mean_absolute_error(y_test, pred))
        race_name = groups.iloc[test_idx].mode().iat[0]
        rows.append({"Fold": fold + 1, "Race": race_name, "MAE": fold_mae, "Laps": len(test_idx)})

    overall_mae = float(mean_absolute_error(y, oof))
    race_mae = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    return overall_mae, race_mae


def detect_cliff_and_loss(stint: pd.DataFrame, predictions: np.ndarray) -> dict[str, float | str]:
    work = stint.sort_values("TyreLife").copy()
    work["Pred"] = np.clip(predictions, 0.0, None)
    life = work["TyreLife"].to_numpy(dtype=float)
    pred = work["Pred"].to_numpy(dtype=float)
    if len(work) < 5:
        return {
            "cliff_lap": np.nan,
            "avg_loss_per_lap": np.nan,
            "peak_loss_per_lap": np.nan,
        }

    slope = np.gradient(pred, life)
    valid_slope = np.clip(slope, 0.0, None)
    warm_count = max(2, int(len(valid_slope) * 0.4))
    baseline = float(np.median(valid_slope[:warm_count]))
    threshold = max(float(np.quantile(valid_slope, 0.80)), baseline + float(np.std(valid_slope)))
    cliff_idx = np.where((valid_slope >= threshold) & (life >= life.min() + 3))[0]
    cliff_lap = float(life[cliff_idx[0]]) if len(cliff_idx) else np.nan
    return {
        "cliff_lap": cliff_lap,
        "avg_loss_per_lap": float(np.mean(valid_slope)),
        "peak_loss_per_lap": float(np.max(valid_slope)),
    }


def build_cliff_report(df: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
    numeric, categorical = get_features()
    features = numeric + categorical
    rows = []
    for stint_id, stint in df.groupby("Stint_ID", sort=False):
        pred = pipeline.predict(stint[features])
        metrics = detect_cliff_and_loss(stint, pred)
        rows.append(
            {
                "Race": stint["Race"].iloc[0],
                "Driver": stint["Driver"].iloc[0],
                "Compound": stint["Compound"].iloc[0],
                "Stint_ID": stint_id,
                "Laps": len(stint),
                **metrics,
            }
        )
    report = pd.DataFrame(rows)
    if report.empty:
        return report
    return report.sort_values(["Race", "Driver", "Stint_ID"]).reset_index(drop=True)


def build_tyre_wise_cliff_report(cliff_report: pd.DataFrame) -> pd.DataFrame:
    if cliff_report.empty:
        return pd.DataFrame()
    tyre_summary = (
        cliff_report.groupby("Compound", as_index=False)
        .agg(
            avg_cliff_lap=("cliff_lap", "mean"),
            median_cliff_lap=("cliff_lap", "median"),
            avg_loss_per_lap=("avg_loss_per_lap", "mean"),
            peak_loss_per_lap=("peak_loss_per_lap", "max"),
            stints=("Stint_ID", "count"),
            races=("Race", "nunique"),
        )
        .sort_values("avg_loss_per_lap", ascending=False)
        .reset_index(drop=True)
    )
    return tyre_summary


def save_model_bundle(path: Path, pipeline: Pipeline, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fp:
        pickle.dump({"pipeline": pipeline, "metadata": metadata}, fp)


def load_model_bundle(path: Path) -> tuple[Pipeline, dict]:
    with path.open("rb") as fp:
        bundle = pickle.load(fp)
    return bundle["pipeline"], bundle.get("metadata", {})


def race_mae_table(df: pd.DataFrame, prediction_column: str = "Pred") -> pd.DataFrame:
    rows = []
    for race, part in df.groupby("Race", sort=True):
        rows.append(
            {
                "Race": race,
                "MAE": float(mean_absolute_error(part["Degradation_Delta"], part[prediction_column])),
                "Laps": int(len(part)),
            }
        )
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def log_dynamic_weights(pipeline: Pipeline) -> None:
    model = pipeline.named_steps.get("model")
    weights = getattr(model, "dynamic_weights_", None)
    if weights:
        ordered = ", ".join(f"{name}={value:.3f}" for name, value in weights.items())
        logging.info("Dynamic ensemble weights: %s", ordered)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-factors-csv", type=Path, required=True, help="CSV containing race-level factors and tyre allocations.")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validate-races", type=str, nargs="*", default=[], help="Optional race names held out for validation.")
    parser.add_argument("--load-model", action="store_true", help="Load an existing model and only run validation reports.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_fastf1_cache()
    profiles = load_track_profiles(args.track_factors_csv)

    schedule = fastf1.get_event_schedule(args.year, include_testing=False)
    race_events = schedule[schedule["EventFormat"].isin(["conventional", "sprint", "sprint_shootout"])].copy()
    race_events = race_events.sort_values("RoundNumber")
    if race_events.empty:
        raise RuntimeError(f"No race events found in {args.year} schedule.")

    all_laps = []
    for _, event_row in race_events.iterrows():
        profile = resolve_profile(str(event_row["EventName"]), str(event_row["Country"]), profiles)
        try:
            laps = fetch_event_race_laps(args.year, event_row, profile)
            if not laps.empty:
                all_laps.append(laps)
                logging.info("Loaded %-24s | laps=%d", event_row["EventName"], len(laps))
            else:
                logging.warning("No valid green-flag race laps for %s", event_row["EventName"])
        except Exception as exc:
            logging.warning("Skipping %s due to error: %s", event_row["EventName"], exc)

    if not all_laps:
        raise RuntimeError("No usable race laps were retrieved from FastF1.")

    raw = pd.concat(all_laps, ignore_index=True)
    dataset = build_clean_degradation_dataset(raw)
    if dataset.empty:
        raise RuntimeError("No valid stints remained after cleaning.")
    dataset = enrich_features(dataset)

    validate_set = {normalize_name(name) for name in args.validate_races}
    if validate_set:
        validation_mask = dataset["Race"].map(normalize_name).isin(validate_set)
    else:
        validation_mask = pd.Series(False, index=dataset.index)

    if args.load_model:
        pipeline, metadata = load_model_bundle(args.model_path)
        logging.info("Loaded model from %s", args.model_path)
        logging.info("Stored metadata: %s", json.dumps(metadata, indent=2))
        log_dynamic_weights(pipeline)
    else:
        pipeline = build_model_pipeline()
        if validation_mask.any():
            train_df = dataset[~validation_mask].reset_index(drop=True)
            val_df = dataset[validation_mask].reset_index(drop=True)
            numeric, categorical = get_features()
            features = numeric + categorical
            pipeline.fit(train_df[features], train_df["Degradation_Delta"])
            log_dynamic_weights(pipeline)
            val_pred = np.clip(pipeline.predict(val_df[features]), 0.0, None)
            val_mae = float(mean_absolute_error(val_df["Degradation_Delta"], val_pred))
            race_val = race_mae_table(val_df.assign(Pred=val_pred))
            cv_mae = np.nan
            logging.info("Validation MAE on held-out races: %.4fs", val_mae)
        else:
            cv_mae, race_cv = evaluate_group_cv(dataset, pipeline)
            numeric, categorical = get_features()
            features = numeric + categorical
            pipeline.fit(dataset[features], dataset["Degradation_Delta"])
            log_dynamic_weights(pipeline)
            logging.info("Cross-race MAE (GroupKFold): %.4fs", cv_mae)
            logging.info("Best race-level CV MAE snapshot:\n%s", race_cv.head(5).to_string(index=False))

        metadata = {
            "year": args.year,
            "dataset_laps": int(len(dataset)),
            "races": sorted(dataset["Race"].unique().tolist()),
            "drivers": sorted(dataset["Driver"].unique().tolist()),
            "cv_or_validation_mae": None if np.isnan(cv_mae) else float(cv_mae),
        }
        save_model_bundle(args.model_path, pipeline, metadata)
        logging.info("Saved model bundle to %s", args.model_path)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if validation_mask.any():
        analysis_df = dataset[validation_mask].reset_index(drop=True)
    else:
        analysis_df = dataset.copy()

    cliff_report = build_cliff_report(analysis_df, pipeline)
    cliff_path = output_dir / f"tyre_cliff_report_{args.year}.csv"
    cliff_report.to_csv(cliff_path, index=False)

    numeric, categorical = get_features()
    features = numeric + categorical
    pred_all = np.clip(pipeline.predict(analysis_df[features]), 0.0, None)
    mae_all = float(mean_absolute_error(analysis_df["Degradation_Delta"], pred_all))
    race_mae = race_mae_table(analysis_df.assign(Pred=pred_all))
    avg_race_mae = float(race_mae["MAE"].mean())
    race_mae_path = output_dir / f"race_mae_{args.year}.csv"
    race_mae.to_csv(race_mae_path, index=False)

    logging.info("Combined-driver MAE on analysis set: %.4fs", mae_all)
    logging.info("Average race MAE: %.4fs", avg_race_mae)
    logging.info("Saved race MAE report to %s", race_mae_path)
    logging.info("Saved cliff report to %s", cliff_path)
    tyre_wise_cliff = build_tyre_wise_cliff_report(cliff_report)
    tyre_wise_path = output_dir / f"tyre_wise_cliff_report_{args.year}.csv"
    tyre_wise_cliff.to_csv(tyre_wise_path, index=False)
    logging.info("Saved tyre-wise cliff report to %s", tyre_wise_path)
    if not cliff_report.empty:
        summary = (
            cliff_report.groupby("Race", as_index=False)
            .agg(
                avg_cliff_lap=("cliff_lap", "mean"),
                avg_loss_per_lap=("avg_loss_per_lap", "mean"),
                peak_loss_per_lap=("peak_loss_per_lap", "max"),
                stints=("Stint_ID", "count"),
            )
            .sort_values("avg_loss_per_lap", ascending=False)
        )
        summary_path = output_dir / f"race_falloff_summary_{args.year}.csv"
        summary.to_csv(summary_path, index=False)
        logging.info("Saved fall-off summary to %s", summary_path)


if __name__ == "__main__":
    main()
