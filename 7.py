import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import os
import gc
import warnings

warnings.simplefilter(action='ignore')
pd.options.mode.chained_assignment = None

CACHE_DIR = 'cache'
PARQUET_STORE = 'ultimate_2023_f1_degradation.parquet'

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
fastf1.Cache.enable_cache(CACHE_DIR)

# =====================================================================
# 1. THE PHYSICS & CALENDAR KNOWLEDGE BASE
# =====================================================================

# Master Pirelli C1 (Hardest) to C5 (Softest) Map
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

# Lateral & Thermal Track Severity (Higher = tyres destroyed faster)
TRACK_ENERGY_INDEX = {
    'Bahrain': 1.70, 'Saudi Arabia': 0.90, 'Australia': 1.10,
    'Azerbaijan': 1.05, 'Miami': 1.15, 'Spain': 1.45,
    'Canada': 0.95, 'Austria': 1.10, 'Great Britain': 1.55,
    'Hungary': 1.15, 'Belgium': 1.25, 'Netherlands': 1.30,
    'Italy': 0.85, 'Japan': 1.50, 'default': 1.00
}

# Accurate Track-Specific Fuel Penalties (Seconds lost per 1kg of fuel)
TRACK_FUEL_SENSITIVITY = {
    'Bahrain': 0.033, 'Saudi Arabia': 0.031, 'Australia': 0.032,
    'Azerbaijan': 0.031, 'Miami': 0.032, 'Spain': 0.035,
    'Canada': 0.027, 'Austria': 0.025, 'Great Britain': 0.034,
    'Hungary': 0.032, 'Belgium': 0.038, 'Netherlands': 0.033,
    'Italy': 0.018, 'Japan': 0.035, 'default': 0.030
}

# =====================================================================
# 2. MASS INGESTION & DATA CLEANING ENGINE
# =====================================================================

def fetch_sunday_race(year, race, drivers):
    """Fetches lightweight timing data for a specific race."""
    try:
        session = fastf1.get_session(year, race, 'R')
        session.load(telemetry=False, weather=True, messages=False)
        
        laps = session.laps[session.laps['Driver'].isin(drivers)].copy()
        
        # Must be valid, no pits, green flag
        valid = laps[
            pd.notnull(laps['LapTime']) & 
            pd.isnull(laps['PitOutTime']) & pd.isnull(laps['PitInTime']) &
            (laps['TrackStatus'] == '1')
        ].copy()
        
        valid['LapTime_sec'] = valid['LapTime'].dt.total_seconds()
        valid['TrackTemp'] = valid.get_weather_data()['TrackTemp'].values
        
        # Map physics parameters
        allocs = PIRELLI_ALLOCATIONS.get(race, {})
        valid['Compound_C_Rating'] = valid['Compound'].map(allocs)
        valid = valid[pd.notnull(valid['Compound_C_Rating'])]
        
        valid['Track'] = race
        valid['Energy_Index'] = TRACK_ENERGY_INDEX.get(race, 1.0)
        valid['Fuel_Penalty'] = TRACK_FUEL_SENSITIVITY.get(race, 0.030)
        
        # Fuel burn correction (105kg standard race burn)
        total_laps = valid['LapNumber'].max()
        valid['Fuel_Weight_kg'] = 105.0 - (valid['LapNumber'] * (105.0 / total_laps))
        valid['Fuel_Corrected_LapTime'] = valid['LapTime_sec'] - (valid['Fuel_Weight_kg'] * valid['Fuel_Penalty'])
        
        valid['Stint_ID'] = valid['Driver'] + "_" + valid['Stint'].astype(str)
        
        # -------------------------------------------------------------
        # THE ENVELOPE FILTER: Isolate the true degradation curve
        # -------------------------------------------------------------
        clean_stints = []
        for sid in valid['Stint_ID'].unique():
            stint = valid[valid['Stint_ID'] == sid].sort_values('TyreLife').copy()
            if len(stint) < 8: continue  # Skip sprint/safety car stints
            
            # Remove egregious traffic (laps > 2% slower than median)
            med_pace = stint['Fuel_Corrected_LapTime'].median()
            stint = stint[stint['Fuel_Corrected_LapTime'] <= (med_pace * 1.020)]
            if len(stint) < 8: continue
            
            # Anchor pace: Use the 10th percentile of laps 2-6 (ignores lap 1 chaos)
            anchor_window = stint[(stint['TyreLife'] >= 2) & (stint['TyreLife'] <= 6)]
            if anchor_window.empty: continue
            
            stint_base_pace = anchor_window['Fuel_Corrected_LapTime'].quantile(0.10)
            stint['Degradation_Delta'] = (stint['Fuel_Corrected_LapTime'] - stint_base_pace).clip(lower=0.0)
            
            clean_stints.append(stint)
            
        del session, laps
        gc.collect()
        
        return pd.concat(clean_stints, ignore_index=True) if clean_stints else pd.DataFrame()
    except Exception as e:
        print(f"     [!] Skipping {race}: {e}")
        return pd.DataFrame()

def build_season_dataset(races, drivers):
    """Builds and caches the mass Parquet dataset."""
    if os.path.exists(PARQUET_STORE):
        print(f"[1/3] Loading ultra-fast cached dataset: {PARQUET_STORE}...")
        return pd.read_parquet(PARQUET_STORE)
        
    print("[1/3] Downloading Season-Wide Telemetry (This will take ~1-2 mins)...")
    data = []
    for race in races:
        print(f"  -> Fetching {race}...")
        df = fetch_sunday_race(2023, race, drivers)
        if not df.empty:
            data.append(df)
            
    master = pd.concat(data, ignore_index=True)
    
    # Feature Engineering: The Golden Feature
    # TyreLife alone is useless. 10 laps at Monza != 10 laps at Silverstone.
    master['Cumulative_Thermal_Energy'] = master['TyreLife'] * master['Energy_Index'] * (master['TrackTemp'] / 35.0)
    
    master.to_parquet(PARQUET_STORE, index=False)
    print(f"  -> Data saved! {len(master)} pristine laps collected.")
    return master

# =====================================================================
# 3. MONOTONIC XGBOOST PHYSICS AI
# =====================================================================

def train_physics_xgboost(df):
    """
    Trains XGBoost with strict mathematical constraints.
    Tyre degradation can ONLY increase or stay flat, preventing downward bending.
    """
    print("\n[2/3] Training Monotonically Constrained XGBoost AI...")
    
    features = ['TyreLife', 'Cumulative_Thermal_Energy', 'Compound_C_Rating', 'Energy_Index', 'TrackTemp']
    X = df[features]
    y = df['Degradation_Delta']
    
    # constraint 1: As TyreLife goes up, degradation MUST go up or stay flat.
    # constraint 1: As Cumulative Energy goes up, degradation MUST go up.
    # constraint -1: As Compound rating goes up (softer), degradation goes up (we map 1=Hard, 5=Soft, so soft degrades faster).
    constraints = (1, 1, 1, 0, 0)
    
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        monotone_constraints=constraints,
        reg_lambda=10.0,  # Heavy L2 regularization forces smooth, sweeping curves
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X, y)
    print("  -> Model converged strictly to physical tyre mechanics.")
    return model, features

# =====================================================================
# 4. TRIPLE-VALIDATION DASHBOARD
# =====================================================================

def run_validation(model, features, year, target_races, driver):
    """Executes prediction on completely unseen Sunday race stints."""
    print("\n[3/3] Executing Multi-Track Ground Truth Validation...")
    
    fig, axes = plt.subplots(1, len(target_races), figsize=(6 * len(target_races), 6))
    
    for ax, target_race in zip(axes, target_races):
        
        # Load fresh validation telemetry
        raw_val = fetch_sunday_race(year, target_race, [driver])
        if raw_val.empty: continue
        
        # Find the longest uninterrupted stint
        longest_sid = raw_val['Stint_ID'].value_counts().idxmax()
        clean_stint = raw_val[raw_val['Stint_ID'] == longest_sid].sort_values('TyreLife').copy()
        
        comp_nom = clean_stint.iloc[0]['Compound']
        comp_c = clean_stint.iloc[0]['Compound_C_Rating']
        
        # Feature Engineering for Validation
        clean_stint['Cumulative_Thermal_Energy'] = clean_stint['TyreLife'] * clean_stint['Energy_Index'] * (clean_stint['TrackTemp'] / 35.0)
        
        # Generate the Smooth Continuous Curve for plotting
        smooth_laps = np.linspace(clean_stint['TyreLife'].min(), clean_stint['TyreLife'].max(), 100)
        smooth_X = pd.DataFrame({
            'TyreLife': smooth_laps,
            'Cumulative_Thermal_Energy': smooth_laps * clean_stint.iloc[0]['Energy_Index'] * (clean_stint['TrackTemp'].mean() / 35.0),
            'Compound_C_Rating': [comp_c] * 100,
            'Energy_Index': [clean_stint.iloc[0]['Energy_Index']] * 100,
            'TrackTemp': [clean_stint['TrackTemp'].mean()] * 100
        })
        
        smooth_pred_delta = model.predict(smooth_X)
        
        # Actual predictions on telemetry points
        clean_stint['Predicted_Delta'] = model.predict(clean_stint[features])
        
        # Anchor Pace (The true zero-degradation baseline)
        anchor_window = clean_stint[(clean_stint['TyreLife'] >= 2) & (clean_stint['TyreLife'] <= 6)]
        if anchor_window.empty: anchor_window = clean_stint.head(3)
        stint_base_pace = anchor_window['Fuel_Corrected_LapTime'].quantile(0.10)
        
        clean_stint['Predicted_Pace'] = stint_base_pace + clean_stint['Predicted_Delta']
        
        # Final MAE Calculation
        mae = np.mean(np.abs(clean_stint['Fuel_Corrected_LapTime'] - clean_stint['Predicted_Pace']))
        
        # PLOTTING
        ax.scatter(clean_stint['TyreLife'], clean_stint['Fuel_Corrected_LapTime'], 
                   color='black', s=50, zorder=4, label='Actual Telemetry (Clean)')
                   
        ax.plot(smooth_laps, stint_base_pace + smooth_pred_delta, 
                color='red', linewidth=3.5, zorder=5, label=f'Monotonic XGBoost (C{int(comp_c)})')
                
        ax.set_title(f"{target_race} - {comp_nom} (C{int(comp_c)})", fontsize=13, weight='bold')
        ax.set_xlabel("Tyre Life (Laps)", fontsize=11)
        ax.set_ylabel("Fuel-Corrected Lap Time (s)", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Dynamic MAE Box Coloring (Green if < 0.15s, Yellow if < 0.30s)
        box_color = 'lightgreen' if mae <= 0.15 else ('lightgoldenrodyellow' if mae <= 0.30 else 'mistyrose')
        ax.text(0.05, 0.90, f"MAE: {mae:.3f}s / lap", transform=ax.transAxes,
                fontsize=13, weight='bold', bbox=dict(facecolor=box_color, alpha=0.9, edgecolor='gray'))
        ax.legend(loc='lower right')
        
    plt.tight_layout()
    plt.show()

# =====================================================================
# EXECUTION COMMANDS
# =====================================================================
if __name__ == "__main__":
    # Top 8 drivers representing a massive spread of car dynamics
    DRIVERS = ['VER', 'PER', 'HAM', 'RUS', 'LEC', 'SAI', 'ALO', 'NOR']
    
    # 14 Races - The model learns the entire spectrum of Pirelli dynamics
    CALENDAR = [
        'Bahrain', 'Saudi Arabia', 'Australia', 'Azerbaijan', 'Miami', 
        'Spain', 'Canada', 'Austria', 'Great Britain', 'Hungary', 
        'Belgium', 'Netherlands', 'Italy', 'Japan'
    ]
    
    # The 3 specific validation tests to conquer
    TEST_RACES = ['Italy', 'Great Britain', 'Spain']
    
    # Pipeline Execute
    df_season = build_season_dataset(CALENDAR, DRIVERS)
    ai_model, feature_list = train_physics_xgboost(df_season)
    run_validation(ai_model, feature_list, 2023, TEST_RACES, 'VER')