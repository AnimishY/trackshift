import fastf1
import pandas as pd
import numpy as np
from matplotlib.pyplot import subplots
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

# ==========================================
# PHASE 1: DATA ISOLATION & BATCH PROCESSING
# ==========================================

def load_and_clean_data(year, race, session_type, driver):
    """Loads session, extracts valid laps, and merges track temperature."""
    print(f"  -> Loading {year} {race} {session_type}...")
    try:
        session = fastf1.get_session(year, race, session_type)
        session.load(telemetry=True, weather=True, messages=False)
        
        driver_laps = session.laps.pick_drivers(driver)
        print(f"     Raw laps found for {driver}: {len(driver_laps)}")
        
        if driver_laps.empty:
            return pd.DataFrame(), session
            
        # 1. Must have a recorded LapTime
        valid = driver_laps[pd.notnull(driver_laps['LapTime'])].copy()
        
        # 2. Exclude in-laps and out-laps
        valid = valid[pd.isnull(valid['PitOutTime']) & pd.isnull(valid['PitInTime'])].copy()
        
        # 3. FIX: Handle NaN in 'Deleted' safely
        if 'Deleted' in valid.columns:
            valid = valid[valid['Deleted'].fillna(False) != True]
            
        # 4. Convert LapTime to numeric seconds
        valid['LapTime_sec'] = valid['LapTime'].dt.total_seconds()
        
        # 5. Pace Cutoff: Drop cool-down laps (> 120% of fastest lap)
        if not valid.empty:
            min_lap = valid['LapTime_sec'].min()
            valid = valid[valid['LapTime_sec'] <= (min_lap * 1.20)].copy()
            
        print(f"     Clean flying laps retained: {len(valid)}")
        
        if valid.empty:
            print(f"     [!] No valid flying laps found for {driver}.")
            return pd.DataFrame(), session
            
        # 6. Map Weather (TrackTemp)
        weather_data = valid.get_weather_data()
        valid['TrackTemp'] = weather_data['TrackTemp'].values
        
        # Metadata
        valid['Track'] = race
        valid['Year'] = year
        valid['Session'] = session_type
        valid['Unique_Stint_ID'] = (
            valid['Year'].astype(str) + "_" + 
            valid['Track'] + "_" + 
            valid['Session'] + "_" + 
            valid['Stint'].astype(str)
        )
        
        return valid, session
        
    except Exception as e:
        print(f"     [!] FAILED to load {race} {session_type}. Reason: {e}")
        return pd.DataFrame(), None

def filter_traffic_and_management(laps_df, threshold_pct=0.92):
    """Drops laps where throttle telemetry indicates traffic or management."""
    print("[2/5] Running Telemetry Traffic Filter across all sessions...")
    clean_laps_indices = []
    
    # NEW: Group by the globally unique Stint ID, not just 'Stint'
    for stint_id in laps_df['Unique_Stint_ID'].unique():
        stint_laps = laps_df[laps_df['Unique_Stint_ID'] == stint_id]
        if len(stint_laps) < 3:
            clean_laps_indices.extend(stint_laps.index)
            continue
            
        try:
            fastest_tel = stint_laps.pick_fastest().get_telemetry()
            base_ft_ratio = (fastest_tel['Throttle'] >= 99).sum() / len(fastest_tel)
            
            for idx, lap in stint_laps.iterrows():
                lap_ft_ratio = (lap.get_telemetry()['Throttle'] >= 99).sum() / len(lap.get_telemetry())
                if lap_ft_ratio >= (base_ft_ratio * threshold_pct):
                    clean_laps_indices.append(idx)
        except Exception:
            # Fallback if telemetry is corrupted for a specific lap
            clean_laps_indices.extend(stint_laps.index)

    return laps_df.loc[clean_laps_indices].copy()

def apply_fp_fuel_correction(laps_df, time_gain_per_lap=0.045):
    """Simulates a constant fuel weight for Practice data."""
    # NEW: Increment stint laps based on Unique ID
    laps_df['Stint_Lap'] = laps_df.groupby('Unique_Stint_ID').cumcount() + 1
    laps_df['Fuel_Corrected_LapTime'] = laps_df['LapTime_sec'] + (laps_df['Stint_Lap'] * time_gain_per_lap)
    return laps_df

# ==========================================
# PHASE 2: MULTI-TRACK MACHINE LEARNING
# ==========================================

def train_xgboost_model(laps_df):
    """Engineers features and trains the degradation model across tracks."""
    print("[3/5] Engineering ML Features and Training XGBoost...")
    df = laps_df[['TyreLife', 'TrackTemp', 'Compound', 'Track', 'Fuel_Corrected_LapTime']].dropna().copy()
    
    df['TyreLife_Squared'] = df['TyreLife'] ** 2
    
    # Encode Compounds
    le_compound = LabelEncoder()
    df['Compound_Encoded'] = le_compound.fit_transform(df['Compound'])
    
    # NEW: Encode Tracks so the model learns track-specific abrasiveness
    le_track = LabelEncoder()
    df['Track_Encoded'] = le_track.fit_transform(df['Track'])
    
    X = df[['TyreLife', 'TyreLife_Squared', 'TrackTemp', 'Compound_Encoded', 'Track_Encoded']]
    y = df['Fuel_Corrected_LapTime']
    
    model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X, y)
    
    return model, le_compound, le_track

# ==========================================
# PHASE 3: SUNDAY RACE VALIDATION
# ==========================================

def run_sunday_validation(model, le_compound, le_track, year, race, driver):
    """Validates predictions against actual Sunday pace."""
    print(f"\n[4/5] Loading Sunday Race Data for Validation ({race})...")
    
    race_laps, session = load_and_clean_data(year, race, 'R', driver)
    if race_laps.empty:
        return
        
    # Full Race Fuel Correction
    total_race_laps = session.laps['LapNumber'].max()
    race_laps['Fuel_Weight_Penalty'] = (110.0 - (race_laps['LapNumber'] * (110.0 / total_race_laps))) * 0.03
    race_laps['Fuel_Corrected_LapTime'] = race_laps['LapTime_sec'] - race_laps['Fuel_Weight_Penalty']
    
    longest_stint_id = race_laps['Unique_Stint_ID'].value_counts().idxmax()
    validation_stint = race_laps[race_laps['Unique_Stint_ID'] == longest_stint_id].copy()
    
    compound_name = validation_stint.iloc[0]['Compound']
    print(f"[5/5] Validating {longest_stint_id} ({compound_name} Tyre, {len(validation_stint)} laps)")
    
    try:
        encoded_compound = le_compound.transform([compound_name])[0]
        encoded_track = le_track.transform([race])[0]
    except ValueError as e:
        print(f"Cannot validate: {e}")
        return

    validation_stint['TyreLife_Squared'] = validation_stint['TyreLife'] ** 2
    validation_stint['Compound_Encoded'] = encoded_compound
    validation_stint['Track_Encoded'] = encoded_track
    
    X_val = validation_stint[['TyreLife', 'TyreLife_Squared', 'TrackTemp', 'Compound_Encoded', 'Track_Encoded']]
    validation_stint['Predicted_Pace'] = model.predict(X_val)
    
    avg_variance = (validation_stint['Fuel_Corrected_LapTime'] - validation_stint['Predicted_Pace']).abs().mean()
    
    # Plotting
    plt.figure(figsize=(12, 7))
    plt.scatter(validation_stint['TyreLife'], validation_stint['Fuel_Corrected_LapTime'], 
                color='black', label='Actual Sunday Pace (Fuel Corrected)', zorder=5)
    plt.plot(validation_stint['TyreLife'], validation_stint['Predicted_Pace'], 
             color='red', linewidth=3, label='AI Predicted Pace (Multi-Session Trained)')
    
    for _, row in validation_stint.iterrows():
        plt.plot([row['TyreLife'], row['TyreLife']], 
                 [row['Fuel_Corrected_LapTime'], row['Predicted_Pace']], 
                 color='gray', linestyle=':', alpha=0.7)

    plt.title(f"Sunday Validation: Actual vs. AI Predicted | {driver} @ {race} ({compound_name})")
    plt.xlabel("Tyre Life (Laps)")
    plt.ylabel("Fuel-Corrected Lap Time (Seconds)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.text(0.05, 0.90, f"Average Error: {avg_variance:.3f}s per lap", 
             transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    plt.show()

# ==========================================
# EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    DRIVER = 'VER'
    
    # 1. Use official country names and standard non-sprint weekends to ensure FP1/FP2/FP3 exist
    TRAINING_EVENTS = [
        {'year': 2023, 'race': 'Italy', 'sessions': ['FP1', 'FP2', 'FP3']},           # Monza
        {'year': 2023, 'race': 'Great Britain', 'sessions': ['FP1', 'FP2', 'FP3']},   # Silverstone
        {'year': 2023, 'race': 'Spain', 'sessions': ['FP1', 'FP2', 'FP3']}            # Barcelona
    ]
    
    VALIDATION_EVENT = {'year': 2023, 'race': 'Italy'}
    
    print("[1/5] Fetching and Batch Processing Multi-Event Data...")
    all_raw_laps = []
    
    for event in TRAINING_EVENTS:
        for session_type in event['sessions']:
            laps, _ = load_and_clean_data(event['year'], event['race'], session_type, DRIVER)
            # Safety check to ensure we only append successful downloads
            if laps is not None and not laps.empty:
                all_raw_laps.append(laps)
                
    # 2. Safety Net: Catch if EVERYTHING failed before crashing pd.concat
    if len(all_raw_laps) == 0:
        print("\n❌ CRITICAL ERROR: Could not load any F1 data.")
        print("Check your internet connection, or try clearing the '/cache' folder in your directory.")
        exit()
                
    combined_raw_data = pd.concat(all_raw_laps, ignore_index=True)
    
    fp_clean = filter_traffic_and_management(combined_raw_data)
    fp_corrected = apply_fp_fuel_correction(fp_clean)
    
    xgb_model, le_compound, le_track = train_xgboost_model(fp_corrected)
    
    run_sunday_validation(xgb_model, le_compound, le_track, 
                          VALIDATION_EVENT['year'], VALIDATION_EVENT['race'], DRIVER)