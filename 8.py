import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import VotingRegressor
import os
import gc
import warnings

# Suppress warnings for clean terminal output
warnings.simplefilter(action='ignore')
pd.options.mode.chained_assignment = None

CACHE_DIR = 'cache'
PARQUET_STORE = 'ultimate_ensemble_f1_degradation.parquet'

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
fastf1.Cache.enable_cache(CACHE_DIR)

# =====================================================================
# 1. THE PHYSICS KNOWLEDGE BASE
# =====================================================================

PIRELLI_ALLOCATIONS = {
    'Bahrain': {'HARD': 1, 'MEDIUM': 2, 'SOFT': 3},
    'Saudi Arabia': {'HARD': 2, 'MEDIUM': 3, 'SOFT': 4},
    'Australia': {'HARD': 2, 'MEDIUM': 3, 'SOFT': 4},
    'Azerbaijan': {'HARD': 3, 'MEDIUM': 4, 'SOFT': 5},
    'Miami': {'HARD': 2, 'MEDIUM': 3, 'SOFT': 4},
    'Spain': {'HARD': 1, 'MEDIUM': 2, 'SOFT': 3},
    'Canada': {'HARD': 3, 'MEDIUM': 4, 'SOFT': 5},
    'Austria': {'HARD': 3, 'MEDIUM': 4, 'SOFT': 5},
    'Great Britain': {'HARD': 1, 'MEDIUM': 2, 'SOFT': 3},
    'Hungary': {'HARD': 3, 'MEDIUM': 4, 'SOFT': 5},
    'Belgium': {'HARD': 2, 'MEDIUM': 3, 'SOFT': 4},
    'Netherlands': {'HARD': 1, 'MEDIUM': 2, 'SOFT': 3},
    'Italy': {'HARD': 3, 'MEDIUM': 4, 'SOFT': 5},
    'Japan': {'HARD': 1, 'MEDIUM': 2, 'SOFT': 3}
}

TRACK_ENERGY_INDEX = {
    'Bahrain': 1.70, 'Saudi Arabia': 0.90, 'Australia': 1.10,
    'Azerbaijan': 1.05, 'Miami': 1.15, 'Spain': 1.45,
    'Canada': 0.95, 'Austria': 1.10, 'Great Britain': 1.55,
    'Hungary': 1.15, 'Belgium': 1.25, 'Netherlands': 1.30,
    'Italy': 0.85, 'Japan': 1.50, 'default': 1.00
}

TRACK_FUEL_SENSITIVITY = {
    'Bahrain': 0.033, 'Saudi Arabia': 0.031, 'Australia': 0.032,
    'Azerbaijan': 0.031, 'Miami': 0.032, 'Spain': 0.035,
    'Canada': 0.027, 'Austria': 0.025, 'Great Britain': 0.034,
    'Hungary': 0.032, 'Belgium': 0.038, 'Netherlands': 0.033,
    'Italy': 0.018, 'Japan': 0.035, 'default': 0.030
}

# =====================================================================
# 2. MASS INGESTION & ABSOLUTE ANCHOR BASELINING
# =====================================================================

def fetch_sunday_race(year, race, drivers):
    """Fetches lightweight timing data and anchors to absolute peak pace."""
    try:
        session = fastf1.get_session(year, race, 'R')
        session.load(telemetry=False, weather=True, messages=False)
        
        laps = session.laps[session.laps['Driver'].isin(drivers)].copy()
        
        valid = laps[
            pd.notnull(laps['LapTime']) & 
            pd.isnull(laps['PitOutTime']) & pd.isnull(laps['PitInTime']) &
            (laps['TrackStatus'] == '1')
        ].copy()
        
        if valid.empty: return pd.DataFrame()
        
        valid['LapTime_sec'] = valid['LapTime'].dt.total_seconds()
        valid['TrackTemp'] = valid.get_weather_data()['TrackTemp'].values
        
        allocs = PIRELLI_ALLOCATIONS.get(race, {})
        valid['Compound_C_Rating'] = valid['Compound'].map(allocs)
        valid = valid[pd.notnull(valid['Compound_C_Rating'])]
        
        valid['Track'] = race
        valid['Energy_Index'] = TRACK_ENERGY_INDEX.get(race, 1.0)
        valid['Fuel_Penalty'] = TRACK_FUEL_SENSITIVITY.get(race, 0.030)
        
        total_laps = valid['LapNumber'].max()
        valid['Fuel_Weight_kg'] = 105.0 - (valid['LapNumber'] * (105.0 / total_laps))
        valid['Fuel_Corrected_LapTime'] = valid['LapTime_sec'] - (valid['Fuel_Weight_kg'] * valid['Fuel_Penalty'])
        valid['Stint_ID'] = valid['Driver'] + "_" + valid['Stint'].astype(str)
        
        clean_stints = []
        for sid in valid['Stint_ID'].unique():
            stint = valid[valid['Stint_ID'] == sid].sort_values('TyreLife').copy()
            if len(stint) < 8: continue
            
            # Filter traffic anomalies (> 2.5% off median)
            med_pace = stint['Fuel_Corrected_LapTime'].median()
            stint = stint[stint['Fuel_Corrected_LapTime'] <= (med_pace * 1.025)]
            if len(stint) < 8: continue
            
            # THE ABSOLUTE ANCHOR FIX: Average of the 3 absolute fastest fuel-corrected laps
            stint_base_pace = stint['Fuel_Corrected_LapTime'].nsmallest(3).mean()
            stint['Degradation_Delta'] = (stint['Fuel_Corrected_LapTime'] - stint_base_pace).clip(lower=0.0)
            
            clean_stints.append(stint)
            
        del session, laps
        gc.collect()
        
        return pd.concat(clean_stints, ignore_index=True) if clean_stints else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def build_season_dataset(races, drivers):
    if os.path.exists(PARQUET_STORE):
        print(f"[1/3] Loading cached season dataset: {PARQUET_STORE}")
        return pd.read_parquet(PARQUET_STORE)
        
    print("[1/3] Downloading Season-Wide Telemetry Database...")
    data = []
    for race in races:
        df = fetch_sunday_race(2023, race, drivers)
        if not df.empty: data.append(df)
            
    master = pd.concat(data, ignore_index=True)
    master['Cumulative_Thermal_Energy'] = master['TyreLife'] * master['Energy_Index'] * (master['TrackTemp'] / 35.0)
    master.to_parquet(PARQUET_STORE, index=False)
    return master

# =====================================================================
# 3. ENSEMBLE AI MODEL (XGBOOST + POLYNOMIAL RIDGE)
# =====================================================================

def train_ensemble_ai(df):
    """Blends physical monotonic bounds with smooth polynomial tracking."""
    print("\n[2/3] Training Voting Ensemble AI (XGBoost + Polynomial)...")
    
    features = ['TyreLife', 'Cumulative_Thermal_Energy', 'Compound_C_Rating', 'Energy_Index', 'TrackTemp']
    X = df[features]
    y = df['Degradation_Delta']
    
    # Model 1: XGBoost (Strictly bounded by physics)
    # 1 = positive correlation (deg increases as tyre life, energy, and softness increase)
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.02,
        max_depth=4,
        subsample=0.8,
        monotone_constraints=(1, 1, 1, 1, 0),
        reg_lambda=15.0, # High regularisation for smoothness
        random_state=42,
        n_jobs=-1
    )
    
    # Model 2: Polynomial Ridge (Smooth underlying curve, fixes intercepts)
    poly_model = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=3, include_bias=False)),
        ('ridge', Ridge(alpha=10.0, random_state=42))
    ])
    
    # The Ensemble combines both strengths
    ensemble = VotingRegressor(
        estimators=[('xgb', xgb_model), ('poly', poly_model)],
        weights=[0.65, 0.35] # Favor XGBoost for the cliff, Poly for the smoothing
    )
    
    ensemble.fit(X, y)
    print("  -> Ensemble architecture locked in.")
    return ensemble, features

# =====================================================================
# 4. TERMINAL LOGGING & VALIDATION DASHBOARD
# =====================================================================

def run_multi_track_validation(model, features, year, target_races, driver):
    """Executes prediction, prints detailed terminal logs, and generates a 2x3 plot grid."""
    print("\n=====================================================================")
    print(f" [3/3] EXECUTING GROUND TRUTH VALIDATION SUITE: {driver} ({year})")
    print("=====================================================================")
    
    # Setup 2x3 Plot Grid for 6 races
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    overall_maes = []
    
    for idx, target_race in enumerate(target_races):
        raw_val = fetch_sunday_race(year, target_race, [driver])
        if raw_val.empty: continue
        
        longest_sid = raw_val['Stint_ID'].value_counts().idxmax()
        clean_stint = raw_val[raw_val['Stint_ID'] == longest_sid].sort_values('TyreLife').copy()
        
        comp_nom = clean_stint.iloc[0]['Compound']
        comp_c = clean_stint.iloc[0]['Compound_C_Rating']
        
        clean_stint['Cumulative_Thermal_Energy'] = clean_stint['TyreLife'] * clean_stint['Energy_Index'] * (clean_stint['TrackTemp'] / 35.0)
        
        # Smooth plotting data
        smooth_laps = np.linspace(clean_stint['TyreLife'].min(), clean_stint['TyreLife'].max(), 100)
        smooth_X = pd.DataFrame({
            'TyreLife': smooth_laps,
            'Cumulative_Thermal_Energy': smooth_laps * clean_stint.iloc[0]['Energy_Index'] * (clean_stint['TrackTemp'].mean() / 35.0),
            'Compound_C_Rating': [comp_c] * 100,
            'Energy_Index': [clean_stint.iloc[0]['Energy_Index']] * 100,
            'TrackTemp': [clean_stint['TrackTemp'].mean()] * 100
        })
        
        smooth_pred_delta = model.predict(smooth_X)
        clean_stint['Predicted_Delta'] = model.predict(clean_stint[features])
        
        # Exact Stint Anchor (Mean of fastest 3 fuel-corrected laps)
        stint_base_pace = clean_stint['Fuel_Corrected_LapTime'].nsmallest(3).mean()
        clean_stint['Predicted_Pace'] = stint_base_pace + clean_stint['Predicted_Delta']
        
        # Metrics
        errors = np.abs(clean_stint['Fuel_Corrected_LapTime'] - clean_stint['Predicted_Pace'])
        mae = np.mean(errors)
        max_err = np.max(errors)
        overall_maes.append(mae)
        
        # ==========================================
        # TERMINAL LOGGING (Requested Feature)
        # ==========================================
        status = "SUCCESS" if mae < 0.150 else ("ACCEPTABLE" if mae < 0.250 else "REVIEW")
        print(f" ► {target_race.upper()} - {comp_nom} (C{int(comp_c)})")
        print(f"   Laps Evaluated : {len(clean_stint)}")
        print(f"   Anchor Pace    : {stint_base_pace:.3f}s")
        print(f"   Mean Error     : {mae:.3f}s / lap")
        print(f"   Max Error Peak : {max_err:.3f}s")
        print(f"   Status         : {status}")
        print("---------------------------------------------------------------------")
        
        # ==========================================
        # MATPLOTLIB PLOTTING
        # ==========================================
        ax = axes[idx]
        ax.scatter(clean_stint['TyreLife'], clean_stint['Fuel_Corrected_LapTime'], 
                   color='black', s=45, zorder=4, label='Actual Telemetry')
                   
        ax.plot(smooth_laps, stint_base_pace + smooth_pred_delta, 
                color='red', linewidth=3, zorder=5, label='Ensemble AI')
                
        ax.set_title(f"{target_race} ({comp_nom}) | C{int(comp_c)}", fontsize=12, weight='bold')
        ax.set_xlabel("Tyre Life", fontsize=10)
        ax.set_ylabel("Lap Time (s)", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        box_color = 'lightgreen' if mae <= 0.15 else ('lightgoldenrodyellow' if mae <= 0.25 else 'mistyrose')
        ax.text(0.05, 0.90, f"MAE: {mae:.3f}s", transform=ax.transAxes,
                fontsize=11, weight='bold', bbox=dict(facecolor=box_color, alpha=0.9, edgecolor='gray'))
        if idx == 0: ax.legend(loc='lower right', fontsize=9)
        
    print(f" >>> OVERALL GRID MAE: {np.mean(overall_maes):.3f}s / lap")
    print("=====================================================================\n")
    
    plt.tight_layout()
    plt.show()

# =====================================================================
# EXECUTION
# =====================================================================
if __name__ == "__main__":
    DRIVERS = ['VER', 'PER', 'HAM', 'RUS', 'LEC', 'SAI', 'ALO', 'NOR']
    
    CALENDAR = [
        'Bahrain', 'Saudi Arabia', 'Australia', 'Azerbaijan', 'Miami', 
        'Spain', 'Canada', 'Austria', 'Great Britain', 'Hungary', 
        'Belgium', 'Netherlands', 'Italy', 'Japan'
    ]
    
    # Massive 6-Race Validation Matrix covering every track style and compound
    TEST_RACES = ['Italy', 'Great Britain', 'Spain', 'Bahrain', 'Hungary', 'Japan']
    
    df_season = build_season_dataset(CALENDAR, DRIVERS)
    ensemble_model, feature_list = train_ensemble_ai(df_season)
    run_multi_track_validation(ensemble_model, feature_list, 2023, TEST_RACES, 'VER')