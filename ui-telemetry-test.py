import os
import fastf1
import fastf1.plotting
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors

# 1. Setup cache
cache_dir = 'fastf1_cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
fastf1.Cache.enable_cache(cache_dir)

# 2. Load Session and Lap Data
session = fastf1.get_session(2025, 'Monza', 'R')
session.load()

lap = session.laps.pick_drivers('VER').pick_fastest()
tel = lap.get_telemetry().add_distance()

# 3. Convert units and prepare time intervals
v_ms = tel['Speed'] / 3.6  # km/h to m/s
time_sec = tel['Time'].dt.total_seconds()
dt = time_sec.diff().bfill()

# 4. Smooth GPS coordinates (removes satellite step-jitter)
x_smooth = savgol_filter(tel['X'], window_length=15, polyorder=3)
y_smooth = savgol_filter(tel['Y'], window_length=15, polyorder=3)

# 5. Derive Heading and Yaw Rate
dx = np.gradient(x_smooth)
dy = np.gradient(y_smooth)
heading = np.unwrap(np.arctan2(dy, dx))
yaw_rate = np.gradient(heading, time_sec)

# 6. Synthesize Lateral and Longitudinal G-Forces
lat_g = savgol_filter((v_ms * yaw_rate) / 9.81, 15, 3)
lon_g = savgol_filter(np.gradient(v_ms, time_sec) / 9.81, 15, 3)

# 7. Calculate Friction Circle & Tyre Thermal/Mechanical Dissipation
combined_g = np.sqrt(lat_g**2 + lon_g**2)
tyre_stress_power = combined_g * v_ms

tel['Stress_Power'] = tyre_stress_power

# 8. Render Continuous LineCollection Map
fig, ax = plt.subplots(figsize=(11, 7), facecolor='#0e1117')
ax.set_facecolor('#0e1117')

# Prepare coordinate segments
points = np.array([tel['X'], tel['Y']]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Normalization across 5th-95th percentile with square-root curve (gamma=0.5)
vmin = np.percentile(tel['Stress_Power'], 5)
vmax = np.percentile(tel['Stress_Power'], 95)
norm = mcolors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)

# Underlay dark base track
ax.plot(tel['X'], tel['Y'], color='#2a2e39', linewidth=6, zorder=1)

# Overlay stress ribbon
lc = LineCollection(segments, cmap='turbo', norm=norm, linewidth=3.5, zorder=2)
lc.set_array(tel['Stress_Power'][:-1])
line = ax.add_collection(lc)

ax.set_title("Monza: Tyre Degradation & Stress Heatmap (Verstappen)", 
             color='white', fontsize=14, pad=15, fontweight='bold')
ax.axis('equal')
ax.axis('off')

# Colorbar setup
cbar = fig.colorbar(line, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Tyre Mechanical & Thermal Stress Index', color='white', fontsize=10, labelpad=10)
cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
cbar.outline.set_edgecolor('#444444')

plt.tight_layout()
plt.show()