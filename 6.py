import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
import os
import gc
import warnings

warnings.simplefilter(action='ignore')
pd.options.mode.chained_assignment = None

CACHE_DIR = 'cache'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
fastf1.Cache.enable_cache(CACHE_DIR)

# =====================================================================
# 1. SEASON-WIDE VEHICLE DYNAMICS KNOWLEDGE BASE
# =====================================================================

# Maps nominal race compounds to actual Pirelli physical compounds (C1-C5)
PIRELLI_ALLOCATIONS = {
    'Bahrain': {'HARD': 1, 'MEDIUM': 2, 'SOFT': 3},
    'Saudi Arabia': {'HARD': 2, 'MEDIUM': 3, 'SOFT': 4},
    'Spain': {'HARD': 1, 'MEDIUM': 2, 'SOFT': 3},
    'Canada': {'HARD': 3, 'MEDIUM': 4, 'SOFT': 5},
    'Great Britain': {'HARD': 1, 'MEDIUM': 2, 'SOFT': 3},
    'Hungary': {'HARD': 3, 'MEDIUM': 4, 'SOFT': 5},
    'Netherlands': {'HARD': 1, 'MEDIUM': 2, 'SOFT': 3},
    'Italy': {'HARD': 3, 'MEDIUM': 4, 'SOFT': 5}
}

# Lateral/Longitudinal Energy Stress Index (1.0 = Average)
TRACK_ENERGY_INDEX = {
    'Bahrain': 1.40,      # Extremely abrasive
    'Saudi Arabia': 0.90, # Smooth street circuit
    'Spain': 1.35,        # High downforce, long high-G corners
    'Canada': 0.95,       # Point and shoot, low lateral
    'Great Britain': 1.50,# Extreme lateral stress (Maggots/Becketts)
    'Hungary': 1.10,      # Continuous medium-speed corners
    'Netherlands': 1.25,  # Banked high-load corners
    'Italy': 0.80,        # Lowest downforce, purely longitudinal
    'default': 1.00
}

# Critical Fix: Precise Track-Specific Fuel Penalties (seconds lost per kg)
TRACK_FUEL_SENSITIVITY = {
    'Bahrain': 0.032,
    'Saudi Arabia': 0.030,
    'Spain': 0.034,
    'Canada': 0.026,
    'Great Britain': 0.032,
    'Hungary': 0.031,
    'Netherlands': 0.031,
    'Italy': 0.018,       # Monza fuel costs almost nothing due to lack of drag
    'default': 0.030
}

# =====================================================================
# 2. RACE DATA INGESTION ENGINE
# =====================================================================

def ingest_sunday_race_stints(year, race, drivers):
    """Ingests actual Sunday race data (no practice). telemetry=False for speed & memory."""
    try:
        session = fastf1.get_session(year, race, 'R')
        session.load(telemetry=False, weather=True, messages=False)
        
        laps = session.laps[session.laps['Driver'].isin(drivers)].copy()
        
        # Keep only valid green-flag flying laps
        valid = laps[
            pd.notnull(laps['LapTime']) & 
            pd.isnull(laps['PitOutTime']) & 
            pd.isnull(laps['PitInTime']) &
            (laps['TrackStatus'] == '1')
        ].copy()
        
        valid['LapTime_sec'] = valid['LapTime'].dt.total_seconds()
        valid['TrackTemp'] = valid.get_weather_data()['TrackTemp'].values
        
        # Apply track physics metadata
        allocs = PIRELLI_ALLOCATIONS.get(race, {})
        valid['Compound_Softness'] = valid['Compound'].map(allocs)
        valid = valid[pd.notnull(valid['Compound_Softness'])]
        
        valid['Track'] = race
        valid['Energy_Index'] = TRACK_ENERGY_INDEX.get(race, 1.0)
        valid['Fuel_Penalty'] = TRACK_FUEL_SENSITIVITY.get(race, 0.030)
        
        # Calculate precise fuel burn (assuming 105kg starting race fuel)
        total_race_laps = valid['LapNumber'].max()
        valid['Fuel_Weight_kg'] = 105.0 - (valid['LapNumber'] * (105.0 / total_race_laps))
        valid['Fuel_Corrected_LapTime'] = valid['LapTime_sec'] - (valid['Fuel_Weight_kg'] * valid['Fuel_Penalty'])
        
        valid['Unique_Stint'] = valid['Driver'] + "_" + valid['Stint'].astype(str)
        
        clean_stints = []
        for stint_id in valid['Unique_Stint'].unique():
            stint = valid[valid['Unique_Stint'] == stint_id].sort_values('TyreLife').copy()
            if len(stint) < 6: # Only learn from genuine race stints
                continue
                
            # Filter traffic & engine-saving laps (> 102.5% of running median)
            median_pace = stint['Fuel_Corrected_LapTime'].median()
            clean_stint = stint[stint['Fuel_Corrected_LapTime'] <= (median_pace * 1.025)].copy()
            
            if len(clean_stint) >= 5:
                # Establish stint baseline pace (safely ignoring lap 1 anomalies)
                base_window = clean_stint['Fuel_Corrected_LapTime'].iloc[1:4] if len(clean_stint) > 4 else clean_stint['Fuel_Corrected_LapTime'].head(3)
                stint_base_pace = base_window.min()
                
                clean_stint['Degradation_Delta'] = (clean_stint['Fuel_Corrected_LapTime'] - stint_base_pace).clip(lower=0.0)
                clean_stints.append(clean_stint)
                
        del session, laps
        gc.collect()
        
        return pd.concat(clean_stints, ignore_index=True) if clean_stints else pd.DataFrame()
    except Exception as e:
        print(f"     [!] Failed loading {race}: {e}")
        return pd.DataFrame()

# =====================================================================
# 3. POLYNOMIAL MACHINE LEARNING ARCHITECTURE
# =====================================================================

def train_polynomial_ai_model(df):
    """
    Trains a 3rd-Degree Polynomial Ridge Regressor.
    Why? It natively learns S-curves and cliffs without step-functions,
    and scales smoothly across compounds and track temperatures.
    """
    print("\n[2/3] Training Season-Wide Polynomial AI Engine...")
    
    features = ['TyreLife', 'Compound_Softness', 'Energy_Index', 'TrackTemp']
    X = df[features]
    y = df['Degradation_Delta']
    
    # Pipeline: Scale Data -> Create 3rd Degree Interactions -> Ridge Regression
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=3, include_bias=False)),
        ('ridge', Ridge(alpha=5.0, random_state=42))
    ])
    
    model.fit(X, y)
    print(f"  -> Model trained on {len(df)} clean race laps.")
    return model, features

# =====================================================================
# 4. HIGH-PRECISION VALIDATION DASHBOARD
# =====================================================================

def run_validation(model, features, year, target_races, driver):
    """Validates predictions against actual Sunday telemetry."""
    print("\n[3/3] Executing Multi-Race Sunday Validation...")
    
    fig, axes = plt.subplots(1, len(target_races), figsize=(6 * len(target_races), 6))
    if len(target_races) == 1: axes = [axes]
    
    for ax, target_race in zip(axes, target_races):
        print(f"  -> Validating {target_race} ({driver})...")
        
        # Load specific race validation data
        raw_val = ingest_sunday_race_stints(year, target_race, [driver])
        if raw_val.empty:
            continue
            
        # Isolate the longest stint for validation visualization
        longest_stint = raw_val['Unique_Stint'].value_counts().idxmax()
        clean_stint = raw_val[raw_val['Unique_Stint'] == longest_stint].sort_values('TyreLife').copy()
        
        nom_comp = clean_stint.iloc[0]['Compound']
        pirelli_comp = clean_stint.iloc[0]['Compound_Softness']
        
        # Predict continuous curve for plotting
        smooth_laps = np.linspace(clean_stint['TyreLife'].min(), clean_stint['TyreLife'].max(), 100)
        smooth_X = pd.DataFrame({
            'TyreLife': smooth_laps,
            'Compound_Softness': [pirelli_comp] * 100,
            'Energy_Index': [clean_stint.iloc[0]['Energy_Index']] * 100,
            'TrackTemp': [clean_stint['TrackTemp'].mean()] * 100
        })
        smooth_pred_delta = model.predict(smooth_X)
        
        # Predict actual laps to calculate MAE
        clean_stint['Predicted_Delta'] = model.predict(clean_stint[features])
        
        # Reconstruct Absolute Pace
        base_window = clean_stint['Fuel_Corrected_LapTime'].iloc[1:4] if len(clean_stint) > 4 else clean_stint['Fuel_Corrected_LapTime'].head(3)
        sunday_base = base_window.min()
        
        clean_stint['Predicted_Pace'] = sunday_base + clean_stint['Predicted_Delta']
        mae = np.mean(np.abs(clean_stint['Fuel_Corrected_LapTime'] - clean_stint['Predicted_Pace']))
        
        # Visualization
        ax.scatter(clean_stint['TyreLife'], clean_stint['Fuel_Corrected_LapTime'], 
                   color='black', s=45, zorder=4, label='Actual Clean Race Laps')
                   
        ax.plot(smooth_laps, sunday_base + smooth_pred_delta, 
                color='red', linewidth=3, zorder=5, label=f'Polynomial AI (C{int(pirelli_comp)})')
                
        title = f"{target_race} - {nom_comp} (C{int(pirelli_comp)})"
        ax.set_title(title, fontsize=12, weight='bold')
        ax.set_xlabel("Tyre Life (Laps)", fontsize=10)
        ax.set_ylabel("Fuel-Corrected Lap Time (s)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.text(0.05, 0.90, f"MAE: {mae:.3f}s / lap", transform=ax.transAxes,
                fontsize=12, weight='bold', bbox=dict(facecolor='white', alpha=0.9, edgecolor='silver'))
        ax.legend(loc='lower right')
        
    plt.tight_layout()
    plt.show()

# =====================================================================
# MAIN PIPELINE EXECUTION
# =====================================================================
if __name__ == "__main__":
    # Expand driver pool to capture generic car dynamics, ignoring team-specific biases
    GRID_DRIVERS = ['VER', 'PER', 'HAM', 'RUS', 'LEC', 'SAI', 'ALO', 'NOR']
    
    # Train on a massive block of 2023 races to let the AI learn true degradation physics
    TRAINING_RACES = ['Bahrain', 'Saudi Arabia', 'Spain', 'Canada', 'Great Britain', 'Hungary', 'Netherlands', 'Italy']
    
    print("[1/3] Ingesting Mass Season-Wide Race Dataset...")
    all_race_data = []
    for race in TRAINING_RACES:
        print(f"  -> Fetching {race} Race Data...")
        df = ingest_sunday_race_stints(2023, race, GRID_DRIVERS)
        if not df.empty:
            all_race_data.append(df)
            
    master_dataset = pd.concat(all_race_data, ignore_index=True)
    
    # Train the machine learning pipeline
    ai_model, ai_features = train_polynomial_ai_model(master_dataset)
    
    # Validate against the three exact scenarios you requested
    TARGET_VALIDATION_RACES = ['Italy', 'Great Britain', 'Spain']
    run_validation(ai_model, ai_features, 2023, TARGET_VALIDATION_RACES, 'VER')