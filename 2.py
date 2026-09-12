import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import os
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None

if not os.path.exists('cache'):
    os.makedirs('cache')
fastf1.Cache.enable_cache('cache')

# =====================================================================
# 1. ROBUST DATA INGESTION & PRACTICE LONG-RUN ISOLATION
# =====================================================================

def load_and_isolate_practice_laps(year, race, session_type, driver):
    """Loads session, extracts only true long-run race simulations, and filters cool-downs."""
    print(f"  -> Ingesting {year} {race} {session_type} for {driver}...")
    try:
        session = fastf1.get_session(year, race, session_type)
        session.load(telemetry=True, weather=True, messages=False)
        
        driver_laps = session.laps.pick_drivers(driver)
        if driver_laps.empty:
            return pd.DataFrame()
            
        # 1. Drop in/out laps and track limit deletions
        valid = driver_laps[
            pd.notnull(driver_laps['LapTime']) & 
            pd.isnull(driver_laps['PitOutTime']) & 
            pd.isnull(driver_laps['PitInTime'])
        ].copy()
        
        if 'Deleted' in valid.columns:
            valid = valid[valid['Deleted'].fillna(False) != True]
            
        if valid.empty:
            return pd.DataFrame()
            
        valid['LapTime_sec'] = valid['LapTime'].dt.total_seconds()
        valid['TrackTemp'] = valid.get_weather_data()['TrackTemp'].values
        valid['Track'] = race
        valid['Year'] = year
        valid['Session'] = session_type
        valid['Unique_Stint_ID'] = f"{year}_{race}_{session_type}_" + valid['Stint'].astype(str)
        
        # 2. Extract ONLY Long Runs (>= 5 timed laps in a single stint)
        long_run_stints = valid['Unique_Stint_ID'].value_counts()
        valid_stints = long_run_stints[long_run_stints >= 5].index
        valid = valid[valid['Unique_Stint_ID'].isin(valid_stints)].copy()
        
        # 3. Filter out cool-down / recharge laps: Keep only laps within 106% of stint median
        clean_stints = []
        for stint_id in valid['Unique_Stint_ID'].unique():
            stint = valid[valid['Unique_Stint_ID'] == stint_id].copy()
            p50 = stint['LapTime_sec'].median()
            stint = stint[stint['LapTime_sec'] <= (p50 * 1.06)]
            if len(stint) >= 4:
                clean_stints.append(stint)
                
        if not clean_stints:
            return pd.DataFrame()
            
        result = pd.concat(clean_stints, ignore_index=True)
        print(f"     Retained {len(result)} long-run push laps across {len(clean_stints)} stint(s).")
        return result
        
    except Exception as e:
        print(f"     [!] Skip {race} {session_type}: {e}")
        return pd.DataFrame()

# =====================================================================
# 2. FUEL NORMALIZATION & RELATIVE DEGRADATION EXTRACTION
# =====================================================================

def extract_relative_degradation(laps_df, fuel_sec_per_lap=0.055):
    """
    Normalizes fuel burn and establishes baseline stint pace.
    Calculates pure degradation delta (Δt) relative to stint start.
    """
    processed = []
    
    for stint_id in laps_df['Unique_Stint_ID'].unique():
        stint = laps_df[laps_df['Unique_Stint_ID'] == stint_id].sort_values('TyreLife').copy()
        
        # Incremental fuel correction within the practice stint
        stint['Stint_Lap_Index'] = np.arange(len(stint))
        stint['Fuel_Corrected_LapTime'] = stint['LapTime_sec'] + (stint['Stint_Lap_Index'] * fuel_sec_per_lap)
        
        # Baseline pace: 15th percentile of first 3 push laps (avoids lap 1 traffic noise)
        base_pace = stint['Fuel_Corrected_LapTime'].head(3).quantile(0.15)
        
        # Relative Degradation Delta: starts at 0.00s and climbs upward
        stint['Degradation_Delta'] = stint['Fuel_Corrected_LapTime'] - base_pace
        
        # Drop anomalous initial laps that had negative degradation (gaining rubber on lap 2)
        stint['Degradation_Delta'] = stint['Degradation_Delta'].clip(lower=0.0)
        
        processed.append(stint)
        
    return pd.concat(processed, ignore_index=True)

from scipy.optimize import curve_fit

# =====================================================================
# 3. CONTINUOUS PHYSICS ENGINE (Quadratic Wear Model)
# =====================================================================

def tyre_physics_model(tyre_life, alpha, beta):
    """
    Continuous tyre wear law:
    alpha: linear abrasion rate (seconds lost per lap)
    beta:  exponential thermal breakdown / core fatigue
    """
    return (alpha * tyre_life) + (beta * (tyre_life ** 2))

def fit_physics_degradation_model(clean_df):
    """
    Extracts physically sound, continuous wear coefficients (alpha, beta)
    for each track, compound, and temperature regime.
    """
    print("\n[3/5] Fitting Continuous Physics Degradation Engine...")
    
    models = {}
    
    # Fit physics parameters per Track + Compound group
    for (track, compound), group in clean_df.groupby(['Track', 'Compound']):
        x_data = group['TyreLife'].values
        y_data = group['Degradation_Delta'].values
        
        # We enforce physical boundaries:
        # alpha >= 0 (wear can't give free lap time)
        # beta >= 0  (wear accelerates, doesn't decelerate)
        try:
            popt, _ = curve_fit(
                tyre_physics_model, 
                x_data, 
                y_data, 
                p0=[0.03, 0.001], 
                bounds=([0.0, 0.0], [0.25, 0.02])
            )
            alpha, beta = popt
        except Exception:
            # Safe physical default for modern Pirelli compounds if data is noisy
            alpha, beta = 0.035, 0.0015
            
        print(f"  -> Model for {track} [{compound}]: Linear Wear (α) = +{alpha:.4f}s/lap | Thermal Cliff (β) = +{beta:.5f}")
        models[(track, compound)] = (alpha, beta)
        
    return models

# =====================================================================
# 4. SUNDAY RACE PACE VALIDATION SUITE
# =====================================================================

def run_sunday_validation(physics_models, year, race, driver):
    """Validates predictions against actual Sunday pace with continuous curves."""
    print(f"\n[4/5] Ingesting Sunday Race Telemetry ({year} {race})...")
    
    session = fastf1.get_session(year, race, 'R')
    session.load(telemetry=True, weather=True, messages=False)
    
    race_laps = session.laps.pick_drivers(driver)
    race_laps = race_laps[
        pd.notnull(race_laps['LapTime']) & 
        pd.isnull(race_laps['PitOutTime']) & 
        pd.isnull(race_laps['PitInTime']) &
        (race_laps['TrackStatus'] == '1')
    ].copy()
    
    race_laps['LapTime_sec'] = race_laps['LapTime'].dt.total_seconds()
    race_laps['TrackTemp'] = race_laps.get_weather_data()['TrackTemp'].values
    
    # Full Sunday Fuel Correction: 110kg burned across race distance
    total_laps = race_laps['LapNumber'].max()
    fuel_burn_rate = 110.0 / total_laps
    race_laps['Fuel_Weight_kg'] = 110.0 - (race_laps['LapNumber'] * fuel_burn_rate)
    race_laps['Fuel_Corrected_LapTime'] = race_laps['LapTime_sec'] - (race_laps['Fuel_Weight_kg'] * 0.033)
    
    # Isolate longest race stint
    longest_stint_num = race_laps['Stint'].value_counts().idxmax()
    stint_df = race_laps[race_laps['Stint'] == longest_stint_num].sort_values('TyreLife').copy()
    compound_name = stint_df.iloc[0]['Compound']
    
    print(f"[5/5] Validating Stint {int(longest_stint_num)}: {compound_name} Compound ({len(stint_df)} Laps)")
    
    # Filter traffic laps (> 103.5% of median stint pace)
    median_pace = stint_df['Fuel_Corrected_LapTime'].median()
    clean_sunday_laps = stint_df[stint_df['Fuel_Corrected_LapTime'] <= (median_pace * 1.035)].copy()
    
    # Retrieve physics parameters for this Track & Compound
    alpha, beta = physics_models.get((race, compound_name), (0.04, 0.0015))
    
    # Generate continuous smooth curve points
    smooth_laps = np.linspace(clean_sunday_laps['TyreLife'].min(), clean_sunday_laps['TyreLife'].max(), 100)
    smooth_predicted_delta = tyre_physics_model(smooth_laps, alpha, beta)
    
    # Calculate point-by-point predictions for clean laps
    clean_sunday_laps['Predicted_Delta'] = tyre_physics_model(clean_sunday_laps['TyreLife'], alpha, beta)
    
    # Reconstruct Sunday Absolute Pace
    sunday_base_pace = clean_sunday_laps['Fuel_Corrected_LapTime'].head(3).quantile(0.15)
    clean_sunday_laps['Predicted_Pace'] = sunday_base_pace + clean_sunday_laps['Predicted_Delta']
    
    # Error Metrics
    mae = np.mean(np.abs(clean_sunday_laps['Fuel_Corrected_LapTime'] - clean_sunday_laps['Predicted_Pace']))
    print(f"\n=======================================================")
    print(f" VALIDATION MAE: {mae:.3f} seconds / lap")
    print(f"=======================================================")
    
    # =================================================================
    # 5. CONTINUOUS DASHBOARD VISUALIZATION
    # =================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel 1: Pure Isolated Degradation Curve
    actual_deltas = clean_sunday_laps['Fuel_Corrected_LapTime'] - sunday_base_pace
    ax1.scatter(clean_sunday_laps['TyreLife'], actual_deltas, color='black', alpha=0.7, label='Actual Sunday Wear (Δt)')
    ax1.plot(smooth_laps, smooth_predicted_delta, color='red', linewidth=2.5, label=f'Physics AI Model (α={alpha:.3f}, β={beta:.4f})')
    ax1.set_title(f"Pure Isolated Degradation Curve ({compound_name})")
    ax1.set_xlabel("Tyre Life (Laps)")
    ax1.set_ylabel("Pace Degradation Delta (Seconds)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Panel 2: Reconstructed Sunday Race Pace
    traffic_laps = stint_df[~stint_df.index.isin(clean_sunday_laps.index)]
    ax2.scatter(traffic_laps['TyreLife'], traffic_laps['Fuel_Corrected_LapTime'], 
                facecolors='none', edgecolors='gray', alpha=0.5, label='Compromised Laps (Traffic/Yellow)')
    
    ax2.scatter(clean_sunday_laps['TyreLife'], clean_sunday_laps['Fuel_Corrected_LapTime'], 
                color='black', zorder=4, label='Clean Flying Laps')
    ax2.plot(smooth_laps, sunday_base_pace + smooth_predicted_delta, 
             color='red', linewidth=2.5, zorder=5, label='AI Projected Sunday Pace')
    
    ax2.set_title(f"Race Pace Reconciliation | {driver} @ {race}")
    ax2.set_xlabel("Tyre Life (Laps)")
    ax2.set_ylabel("Fuel-Corrected Lap Time (Seconds)")
    ax2.grid(True, alpha=0.3)
    ax2.text(0.05, 0.90, f"Average Error: {mae:.3f}s / lap", transform=ax2.transAxes,
             fontsize=11, weight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='silver'))
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# =====================================================================
# MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    DRIVER = 'VER'
    
    TRAINING_EVENTS = [
        {'year': 2023, 'race': 'Italy', 'sessions': ['FP1', 'FP2', 'FP3']},
        {'year': 2023, 'race': 'Great Britain', 'sessions': ['FP1', 'FP2', 'FP3']},
        {'year': 2023, 'race': 'Spain', 'sessions': ['FP1', 'FP2', 'FP3']}
    ]
    VALIDATION = {'year': 2023, 'race': 'Italy'}
    
    print("[1/5] Extracting Practice Long-Run Runs...")
    batch_laps = []
    for event in TRAINING_EVENTS:
        for s in event['sessions']:
            data = load_and_isolate_practice_laps(event['year'], event['race'], s, DRIVER)
            if not data.empty:
                batch_laps.append(data)
                
    if not batch_laps:
        print("No training data found.")
        exit()
        
    combined_practice = pd.concat(batch_laps, ignore_index=True)
    
    print("[2/5] Normalizing Fuel & Computing Compound Degradation Deltas...")
    deg_dataset = extract_relative_degradation(combined_practice)
    
    # Fit continuous physics degradation laws
    physics_models = fit_physics_degradation_model(deg_dataset)
    
    # Validate against Sunday
    run_sunday_validation(physics_models, VALIDATION['year'], VALIDATION['race'], DRIVER)