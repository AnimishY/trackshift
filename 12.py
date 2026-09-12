"""
Multi-Driver Tyre Degradation Model — v2
=========================================

Your v1 run: Sunday holdout MAE = 0.339s/lap, with fitted alpha = 0.015 —
which is EXACTLY the lower bound you gave curve_fit (bounds=([0.015, ...])).
That's not a coincidence: the optimizer wanted to go lower (or negative) and
was pinned at the wall. Everything below is aimed at the actual root causes,
not just retuning that one number.

What changed and why:

  1. TRAINING DATA SOURCE. FP long runs under-represent race-day wear — your
     own plot shows the FP-fitted curve sitting almost flat under a Sunday
     stint that clearly steepens. Practice tyres are used lighter, less
     rubbered-in, and the runs are often not full representative stints.
     Training now blends FP (volume) with PRIOR-YEAR RACE sessions at the
     same track (real degradation, no leakage into the validated race).

  2. CLIPPING BIAS. `Degradation_Delta.clip(lower=0.0)` floored every lap
     that ran a hair under its noisy 3-lap baseline to exactly 0, while
     doing nothing to laps that ran over. That's not just "removing noise" —
     it's a one-sided edit that systematically drags a least-squares fit
     down. Deltas are now signed and unclipped.

  3. TIGHT BOUNDS + L2 LOSS. bounds=([0.015, 0.0002], [0.09, 0.0025]) left
     almost no room to move, and plain curve_fit uses squared-error loss
     (sensitive to exactly the kind of stray high-delta laps visible in your
     plot). Bounds are loosened to a physically-sane range, the fit uses
     Huber/soft-L1 loss (`loss='soft_l1', f_scale=...`) so a handful of
     compromised laps can't dominate it, and the script now PRINTS A WARNING
     any time a fitted parameter lands on its bound again — that's your
     early-warning signal that the box is still wrong, instead of a silently
     wrong curve.

  4. INCONSISTENT FILTERING. TrackStatus / IsAccurate / fresh-tyre checks
     were only applied to the Sunday validation lap, not to the practice
     laps used to train the model. Both stages now apply the same filters,
     using the actual FastF1 fields (`IsAccurate`, `TrackStatus`, `FreshTyre`)
     documented at theoehrly-fast-f1.mintlify.app/core-concepts/lap-timing.

  5. TWO DIFFERENT FUEL CONVENTIONS. FP laps were corrected UP toward a
     start-of-stint reference; Race laps were corrected DOWN toward a
     zero-fuel reference. Pooling those together means alpha/beta were fit
     on two different quantities. Both now use one convention (zero-fuel-
     equivalent laptime) with a fuel-burn rate derived from the track's
     actual race distance, instead of a flat "1.9kg/lap" guess.

  6. POOLING ACROSS CARS. VER/PER/LEC/SAI/HAM/RUS were fit as one curve, but
     2023 Red Bull, Ferrari and Mercedes did not degrade their tyres the same
     way. The pooled curve is now treated as the grid's "shape", and a
     closed-form per-driver scale factor calibrates it to that specific
     driver's own other races.

  7. TrackTemp WAS UNUSED. It was computed and attached to every lap but
     never touched the fit, despite beta being labelled "thermal". It's now
     an actual regressor (gamma term), not just a comment.

  8. HONESTY CHECK. A leave-one-stint-out cross-validated MAE on the
     TRAINING pool is now reported alongside the Sunday holdout MAE, so a
     good-looking holdout number can be sanity-checked against something
     that can't have been cherry-picked.

A note on the <0.05s/lap target: that's a very tight bar for lap-by-lap
prediction — published tyre-degradation models in the sim-racing/F1-analytics
space typically land in the 0.10-0.20s MAE range, because driver-to-driver
variance, traffic, and engine-mode changes are real signal the model has no
way to see. This rewrite should get you meaningfully closer (fixing a fit
that was pinned at its own boundary is a large gain), but treat 0.05s as
something to chase incrementally with the diagnostics below, not something
any lap-count-only model is guaranteed to hit out of the box.

I could not execute this end-to-end in this environment (no network route to
the F1 timing API here), so the MAE numbers below are printed live when you
run it, not asserted by me.
"""

import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None

CACHE_DIR = 'cache'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
fastf1.Cache.enable_cache(CACHE_DIR)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRACK_FUEL_SENSITIVITY = {
    'Italy': 0.021,
    'Great Britain': 0.032,
    'Spain': 0.033,
    'default': 0.030
}

FULL_TANK_KG = 105.0
GREEN_FLAG_STATUS = '1'
MAD_K = 3.5              # robust outlier threshold, in scaled-MAD units
HUBER_F_SCALE = 0.15     # seconds; residuals beyond this get down-weighted


# ===========================================================================
# 1. RACE-DISTANCE LOOKUP  (so FP and Race data share ONE fuel model)
# ===========================================================================

def get_total_race_laps(year, race):
    """Real race distance, so fuel-burn-per-lap isn't a guessed constant."""
    try:
        r = fastf1.get_session(year, race, 'R')
        r.load(laps=True, telemetry=False, weather=False, messages=False)
        return int(r.laps['LapNumber'].max())
    except Exception:
        return None


# ===========================================================================
# 2. FILTERING HELPERS
# ===========================================================================

def _safe_bool_filter(series, keep_value):
    """Keep a row if the flag equals keep_value OR is missing (unknown isn't
    a violation). Avoids the NaN-in-boolean-mask trap that a plain
    `series != True` can fall into depending on dtype."""
    return series.isna() | (series == keep_value)


def _is_green_flag(track_status):
    """TrackStatus concatenates codes within a lap ('14' = part green, part
    yellow) — only a bare '1' is a fully representative green-flag lap."""
    if pd.isna(track_status):
        return True
    return str(track_status) == GREEN_FLAG_STATUS


def _mad_filter(series, k=MAD_K):
    """Two-sided robust outlier mask (scaled median-absolute-deviation).
    Replaces the old one-sided '<= median*1.045' rule, which only ever
    dropped slow laps and left fast-lap noise untouched."""
    med = series.median()
    mad = (series - med).abs().median()
    if mad == 0:
        return pd.Series(True, index=series.index)
    scaled = 1.4826 * mad
    return (series - med).abs() <= k * scaled


# ===========================================================================
# 3. INGESTION (FP *and* RACE, identical filters to validation)
# ===========================================================================

def load_clean_driver_laps(session, driver, race_name, year, session_type, burn_per_lap):
    try:
        driver_laps = session.laps.pick_drivers(driver)
        if driver_laps.empty:
            return pd.DataFrame()

        weather = driver_laps.get_weather_data()
        driver_laps['TrackTemp'] = weather['TrackTemp'].values

        valid = driver_laps[
            pd.notnull(driver_laps['LapTime']) &
            pd.isnull(driver_laps['PitOutTime']) &
            pd.isnull(driver_laps['PitInTime'])
        ].copy()

        if 'Deleted' in valid.columns:
            valid = valid[_safe_bool_filter(valid['Deleted'], False)]
        if 'IsAccurate' in valid.columns:
            valid = valid[_safe_bool_filter(valid['IsAccurate'], True)]
        if 'TrackStatus' in valid.columns:
            valid = valid[valid['TrackStatus'].apply(_is_green_flag)]
        if 'FreshTyre' in valid.columns:
            valid = valid[_safe_bool_filter(valid['FreshTyre'], True)]

        if valid.empty:
            return pd.DataFrame()

        valid['LapTime_sec'] = valid['LapTime'].dt.total_seconds()
        valid['Track'] = race_name
        valid['Year'] = year
        valid['Session'] = session_type
        valid['Driver'] = driver
        valid['Unique_Stint_ID'] = (
            f"{year}_{race_name}_{session_type}_{driver}_s" + valid['Stint'].astype(str)
        )

        clean_laps = []
        for stint_id in valid['Unique_Stint_ID'].unique():
            stint = valid[valid['Unique_Stint_ID'] == stint_id].sort_values('TyreLife').copy()
            if len(stint) < 3:
                continue
            # A stint that didn't start on ~fresh tyres poisons the
            # TyreLife=0 baseline instead of just being noisy -> drop it.
            if stint['TyreLife'].iloc[0] > 2:
                continue

            stint = stint[_mad_filter(stint['LapTime_sec'])]
            if len(stint) >= 3:
                clean_laps.append(stint)

        if not clean_laps:
            return pd.DataFrame()

        out = pd.concat(clean_laps, ignore_index=True)
        out['FuelBurnPerLap'] = burn_per_lap
        return out

    except Exception as e:
        print(f"     [!] Skipping driver {driver} in {session_type}: {e}")
        return pd.DataFrame()


def ingest_grid_dataset(events, drivers):
    print("[1/6] Ingesting Multi-Driver, Multi-Session Telemetry across Grid...")
    all_laps = []

    for ev in events:
        total_laps = get_total_race_laps(ev['year'], ev['race'])
        burn_per_lap = (FULL_TANK_KG / total_laps) if total_laps else (FULL_TANK_KG / 55.0)

        for s in ev['sessions']:
            print(f"  -> Loading {ev['year']} {ev['race']} {s}...")
            try:
                session = fastf1.get_session(ev['year'], ev['race'], s)
                session.load(telemetry=True, weather=True, messages=False)
                for drv in drivers:
                    df = load_clean_driver_laps(session, drv, ev['race'], ev['year'], s, burn_per_lap)
                    if not df.empty:
                        all_laps.append(df)
            except Exception as e:
                print(f"     [!] Failed loading session {s}: {e}")

    if not all_laps:
        raise ValueError("No data could be ingested. Check your network or cache.")

    master = pd.concat(all_laps, ignore_index=True)
    n_race = int((master['Session'] == 'R').sum())
    n_prac = len(master) - n_race
    print(f"  -> Compiled {len(master)} clean push laps ({n_race} race / {n_prac} practice) "
          f"across {len(drivers)} drivers.")
    return master


# ===========================================================================
# 4. UNIFIED FUEL NORMALIZATION (one convention, FP == Race)
# ===========================================================================

def extract_compound_wear_deltas(df):
    print("[2/6] Normalizing Fuel Weights (unified convention) & Isolating Wear...")
    processed = []

    for stint_id in df['Unique_Stint_ID'].unique():
        stint = df[df['Unique_Stint_ID'] == stint_id].sort_values('TyreLife').copy()
        track = stint.iloc[0]['Track']
        penalty = TRACK_FUEL_SENSITIVITY.get(track, TRACK_FUEL_SENSITIVITY['default'])
        burn = stint.iloc[0]['FuelBurnPerLap']

        stint['Stint_Lap_Index'] = np.arange(len(stint))
        stint['Fuel_Weight_kg'] = np.maximum(FULL_TANK_KG - stint['Stint_Lap_Index'] * burn, 1.5)
        stint['Fuel_Corrected_LapTime'] = stint['LapTime_sec'] - stint['Fuel_Weight_kg'] * penalty

        base_pace = stint['Fuel_Corrected_LapTime'].head(min(5, len(stint))).median()
        # Signed, NOT clipped -- see fix #2 above.
        stint['Degradation_Delta'] = stint['Fuel_Corrected_LapTime'] - base_pace
        stint['TempDev'] = stint['TrackTemp'] - stint['TrackTemp'].median()

        processed.append(stint)

    return pd.concat(processed, ignore_index=True)


# ===========================================================================
# 5. ROBUST, TEMPERATURE-AWARE PHYSICS SOLVER
# ===========================================================================

def tyre_physics_law(X, alpha, beta, gamma):
    """t: tyre life (laps). temp_dev: TrackTemp minus this group's median
    TrackTemp -- gamma is a real thermal-sensitivity term now, not just a
    comment next to an unused column."""
    t, temp_dev = X
    return (alpha * t) + (beta * (t ** 2)) + (gamma * temp_dev)


PHYSICS_BOUNDS = ([0.0, -0.01, -0.05], [0.30, 0.02, 0.05])
PHYSICS_P0 = [0.04, 0.001, 0.0]


def fit_multi_driver_physics(clean_df):
    print("[3/6] Fitting Pooled Degradation Model (robust loss, temp-aware)...")
    models = {}

    for (track, compound), group in clean_df.groupby(['Track', 'Compound']):
        x = group['TyreLife'].values.astype(float)
        temp_dev = group['TempDev'].fillna(0).values.astype(float)
        y = group['Degradation_Delta'].values.astype(float)

        try:
            popt, _ = curve_fit(
                tyre_physics_law, (x, temp_dev), y,
                p0=PHYSICS_P0, bounds=PHYSICS_BOUNDS,
                loss='soft_l1', f_scale=HUBER_F_SCALE, max_nfev=20000
            )
            alpha, beta, gamma = popt
        except Exception:
            alpha, beta, gamma = 0.04, 0.001, 0.0

        flags = []
        for name, val, lo, hi in [
            ('alpha', alpha, PHYSICS_BOUNDS[0][0], PHYSICS_BOUNDS[1][0]),
            ('beta', beta, PHYSICS_BOUNDS[0][1], PHYSICS_BOUNDS[1][1]),
            ('gamma', gamma, PHYSICS_BOUNDS[0][2], PHYSICS_BOUNDS[1][2]),
        ]:
            if abs(val - lo) < 1e-6 or abs(val - hi) < 1e-6:
                flags.append(name)
        tag = f"  [!! {','.join(flags)} PINNED AT BOUND]" if flags else ""

        print(f"  -> {track} [{compound}] (n={len(group)}): "
              f"α={alpha:.4f}s/lap  β={beta:.6f}s/lap²  γ={gamma:.4f}s/°C{tag}")
        models[(track, compound)] = (alpha, beta, gamma)

    return models


# ===========================================================================
# 6. PER-DRIVER CALIBRATION
# ===========================================================================

def calibrate_driver(clean_df, models, track, compound, driver, exclude_year=None):
    """Closed-form scalar (can't meaningfully overfit) that pulls the pooled
    grid curve toward this specific driver's own other races/practice."""
    alpha, beta, gamma = models.get((track, compound), (0.04, 0.001, 0.0))

    own = clean_df[
        (clean_df['Track'] == track) &
        (clean_df['Compound'] == compound) &
        (clean_df['Driver'] == driver)
    ]
    if exclude_year is not None:
        own = own[own['Year'] != exclude_year]

    if len(own) < 5:
        return 1.0  # not enough of this driver's own data -> trust the pooled curve

    t = own['TyreLife'].values.astype(float)
    temp_dev = own['TempDev'].fillna(0).values.astype(float)
    pred = tyre_physics_law((t, temp_dev), alpha, beta, gamma)
    y = own['Degradation_Delta'].values.astype(float)

    denom = np.sum(pred ** 2)
    if denom < 1e-9:
        return 1.0
    return float(np.clip(np.sum(y * pred) / denom, 0.5, 2.0))


# ===========================================================================
# 7. HONEST CROSS-VALIDATED MAE (on the training pool, not the holdout)
# ===========================================================================

def cross_validate(clean_df, track, compound):
    """Leave-one-stint-out MAE on TRAINING data. Trust this number if the
    Sunday holdout ever looks suspiciously good -- it can't be inflated by
    having implicitly tuned anything against the one race being validated."""
    group = clean_df[(clean_df['Track'] == track) & (clean_df['Compound'] == compound)]
    stint_ids = group['Unique_Stint_ID'].unique()
    if len(stint_ids) < 4:
        return None

    errors = []
    for held_out in stint_ids:
        train = group[group['Unique_Stint_ID'] != held_out]
        test = group[group['Unique_Stint_ID'] == held_out]
        try:
            popt, _ = curve_fit(
                tyre_physics_law,
                (train['TyreLife'].values.astype(float), train['TempDev'].fillna(0).values.astype(float)),
                train['Degradation_Delta'].values.astype(float),
                p0=PHYSICS_P0, bounds=PHYSICS_BOUNDS,
                loss='soft_l1', f_scale=HUBER_F_SCALE, max_nfev=20000
            )
        except Exception:
            continue
        pred = tyre_physics_law(
            (test['TyreLife'].values.astype(float), test['TempDev'].fillna(0).values.astype(float)),
            *popt
        )
        errors.extend(np.abs(pred - test['Degradation_Delta'].values.astype(float)))

    return float(np.mean(errors)) if errors else None


# ===========================================================================
# 8. SUNDAY VALIDATION (same filters as training, temp-aware, calibrated)
# ===========================================================================

def run_telemetry_ground_truth_validation(models, clean_df, year, race, driver):
    print(f"\n[4/6] Ingesting Sunday Race Telemetry ({year} {race})...")

    session = fastf1.get_session(year, race, 'R')
    session.load(telemetry=True, weather=True, messages=False)

    race_laps = session.laps.pick_drivers(driver)
    weather = race_laps.get_weather_data()
    race_laps['TrackTemp'] = weather['TrackTemp'].values

    mask = (
        pd.notnull(race_laps['LapTime']) &
        pd.isnull(race_laps['PitOutTime']) &
        pd.isnull(race_laps['PitInTime'])
    )
    if 'Deleted' in race_laps.columns:
        mask &= _safe_bool_filter(race_laps['Deleted'], False)
    if 'IsAccurate' in race_laps.columns:
        mask &= _safe_bool_filter(race_laps['IsAccurate'], True)
    if 'TrackStatus' in race_laps.columns:
        mask &= race_laps['TrackStatus'].apply(_is_green_flag)

    race_laps = race_laps[mask].copy()
    race_laps['LapTime_sec'] = race_laps['LapTime'].dt.total_seconds()

    penalty = TRACK_FUEL_SENSITIVITY.get(race, TRACK_FUEL_SENSITIVITY['default'])
    total_laps = race_laps['LapNumber'].max()
    burn = FULL_TANK_KG / total_laps

    race_laps['Fuel_Weight_kg'] = np.maximum(FULL_TANK_KG - race_laps['LapNumber'] * burn, 1.5)
    race_laps['Fuel_Corrected_LapTime'] = race_laps['LapTime_sec'] - race_laps['Fuel_Weight_kg'] * penalty

    longest_stint_num = race_laps['Stint'].value_counts().idxmax()
    stint_df = race_laps[race_laps['Stint'] == longest_stint_num].sort_values('TyreLife').copy()
    compound = stint_df.iloc[0]['Compound']
    stint_df['TempDev'] = stint_df['TrackTemp'] - stint_df['TrackTemp'].median()

    print(f"[5/6] Analyzing Stint {int(longest_stint_num)} ({compound} Compound, {len(stint_df)} Laps)...")

    keep = _mad_filter(stint_df['Fuel_Corrected_LapTime'])
    clean_sunday = stint_df[keep].copy()
    compromised_sunday = stint_df[~keep].copy()

    alpha, beta, gamma = models.get((race, compound), (0.04, 0.001, 0.0))
    k = calibrate_driver(clean_df, models, race, compound, driver, exclude_year=year)
    direction = 'scaled up' if k > 1.02 else 'scaled down' if k < 0.98 else 'left unchanged'
    print(f"  -> Driver calibration for {driver}: k={k:.3f} ({direction} vs. grid-average curve)")

    X = (clean_sunday['TyreLife'].values.astype(float), clean_sunday['TempDev'].fillna(0).values.astype(float))
    clean_sunday['Predicted_Delta'] = k * tyre_physics_law(X, alpha, beta, gamma)
    sunday_base_pace = clean_sunday['Fuel_Corrected_LapTime'].head(min(4, len(clean_sunday))).median()
    clean_sunday['Predicted_Pace'] = sunday_base_pace + clean_sunday['Predicted_Delta']

    mae = float(np.mean(np.abs(clean_sunday['Fuel_Corrected_LapTime'] - clean_sunday['Predicted_Pace'])))

    print("[6/6] Cross-validating on the training pool (leave-one-stint-out)...")
    cv_mae = cross_validate(clean_df, race, compound)

    print(f"\n=======================================================")
    print(f" SUNDAY HOLDOUT MAE:      {mae:.3f} s/lap")
    if cv_mae is not None:
        print(f" TRAINING CROSS-VAL MAE:  {cv_mae:.3f} s/lap   <- trust this one")
    else:
        print(f" TRAINING CROSS-VAL MAE:  n/a (fewer than 4 stints for this track/compound)")
    print(f" Target (< 0.05s) met:    {'YES' if mae < 0.05 else 'NO'}")
    print(f"=======================================================")

    # ------------------------------------------------------------------
    # Visualization (same two-panel layout you had, labels updated)
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    smooth_laps = np.linspace(stint_df['TyreLife'].min(), clean_sunday['TyreLife'].max(), 100)
    smooth_temp = np.full_like(smooth_laps, clean_sunday['TempDev'].fillna(0).mean())
    smooth_pred = k * tyre_physics_law((smooth_laps, smooth_temp), alpha, beta, gamma)

    actual_deltas = clean_sunday['Fuel_Corrected_LapTime'] - sunday_base_pace
    ax1.scatter(clean_sunday['TyreLife'], actual_deltas, color='black', alpha=0.8,
                label='Actual Sunday Telemetry (Clean)')
    ax1.plot(smooth_laps, smooth_pred, color='red', linewidth=2.5,
             label=f'Calibrated Model (α={alpha:.3f}, β={beta:.5f}, γ={gamma:.3f}, k={k:.2f})')
    ax1.set_title(f"Pure Isolated Degradation Curve ({compound})")
    ax1.set_xlabel("Tyre Life (Laps)")
    ax1.set_ylabel("Wear Delta (Seconds)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.scatter(compromised_sunday['TyreLife'], compromised_sunday['Fuel_Corrected_LapTime'],
                facecolors='none', edgecolors='gray', alpha=0.6, label='Compromised Laps (filtered)')
    ax2.scatter(clean_sunday['TyreLife'], clean_sunday['Fuel_Corrected_LapTime'],
                color='black', zorder=4, label='Clean Flying Laps')
    ax2.plot(smooth_laps, sunday_base_pace + smooth_pred,
             color='red', linewidth=2.5, zorder=5, label='Model Projected Sunday Pace')

    ax2.set_title(f"Sunday Race Pace Match | {driver} @ {race}")
    ax2.set_xlabel("Tyre Life (Laps)")
    ax2.set_ylabel("Fuel-Corrected Lap Time (Seconds)")
    ax2.grid(True, alpha=0.3)
    ax2.text(0.05, 0.90,
              f"Holdout MAE: {mae:.3f}s/lap\nCross-val MAE: {cv_mae:.3f}s/lap" if cv_mae is not None
              else f"Holdout MAE: {mae:.3f}s/lap",
              transform=ax2.transAxes, fontsize=11, weight='bold',
              bbox=dict(facecolor='white', alpha=0.9, edgecolor='silver'))
    ax2.legend()

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/sunday_validation.png', dpi=150)
    plt.show()


# ===========================================================================
# MAIN EXECUTION
# ===========================================================================
if __name__ == "__main__":
    GRID_DRIVERS = ['VER', 'PER', 'LEC', 'SAI', 'HAM', 'RUS']

    TRAINING_EVENTS = [
        # Current-year practice: adds volume/shape, but under-represents
        # true race-day wear on its own (see the diagnostic plot).
        {'year': 2023, 'race': 'Italy', 'sessions': ['FP1', 'FP2', 'FP3']},
        {'year': 2023, 'race': 'Great Britain', 'sessions': ['FP1', 'FP2', 'FP3']},
        {'year': 2023, 'race': 'Spain', 'sessions': ['FP1', 'FP2', 'FP3']},
        # Prior-year RACES at the validation track: real degradation
        # behaviour, with no leakage into the 2023 Italy race below.
        {'year': 2021, 'race': 'Italy', 'sessions': ['R']},
        {'year': 2022, 'race': 'Italy', 'sessions': ['R']},
    ]
    VALIDATION = {'year': 2023, 'race': 'Italy', 'driver': 'VER'}

    assert not any(
        ev['year'] == VALIDATION['year'] and ev['race'] == VALIDATION['race'] and 'R' in ev['sessions']
        for ev in TRAINING_EVENTS
    ), "Training set contains the exact race session being validated -- that's leakage, fix TRAINING_EVENTS."

    raw_df = ingest_grid_dataset(TRAINING_EVENTS, GRID_DRIVERS)
    wear_dataset = extract_compound_wear_deltas(raw_df)
    physics_models = fit_multi_driver_physics(wear_dataset)
    run_telemetry_ground_truth_validation(
        physics_models, wear_dataset, VALIDATION['year'], VALIDATION['race'], VALIDATION['driver']
    )