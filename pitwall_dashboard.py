"""Streamlit pit-wall view for live latent tyre-state inference.

Run with ``streamlit run pitwall_dashboard.py``.  Upload a cleaned stint CSV
containing ``TyreLife`` and ``Degradation_Delta`` (the pipeline output is
accepted directly); no team-private sensor feed is required for the demo.
"""

from __future__ import annotations

import pickle
from pathlib import Path
import os

try:
    import streamlit as st
except ModuleNotFoundError as exc:
    raise SystemExit("Install the optional dashboard dependency with: pip install streamlit") from exc

import plotly.graph_objects as go
import numpy as np
import pandas as pd
import fastf1
import fastf1.plotting
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors

from latent_tyre_model import LatentTyreModel, first_cliff_forecast
from virtual_sensors import default_sensor_state, wear_load_breakdown


MONO = "'JetBrains Mono', monospace"

# Enable FastF1 Cache
cache_dir = 'fastf1_cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
fastf1.Cache.enable_cache(cache_dir)
fastf1.plotting.setup_mpl(mpl_timedelta_support=False, misc_mpl_mods=False)

# Official FastF1 event names, used directly as get_session() queries so a
# track choice always resolves to the correct circuit (short country-style
# names like "Great Britain" can resolve to the wrong event).
DEMO_TRACK_CHOICES = [
    "Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix",
    "Japanese Grand Prix", "Chinese Grand Prix", "Miami Grand Prix",
    "Emilia Romagna Grand Prix", "Monaco Grand Prix", "Canadian Grand Prix",
    "Spanish Grand Prix", "Austrian Grand Prix", "British Grand Prix",
    "Hungarian Grand Prix", "Belgian Grand Prix", "Dutch Grand Prix",
    "Italian Grand Prix", "Azerbaijan Grand Prix", "Singapore Grand Prix",
    "United States Grand Prix", "Mexico City Grand Prix", "São Paulo Grand Prix",
    "Las Vegas Grand Prix", "Qatar Grand Prix", "Abu Dhabi Grand Prix",
]
KNOWN_2025_DRIVERS = [
    "VER", "NOR", "PIA", "LEC", "HAM", "RUS", "ANT", "ALO", "STR", "GAS",
    "OCO", "HUL", "SAI", "TSU", "LAW", "ALB", "COL", "BOR", "HAD", "BEA", "DOO",
]

# Accent colour per tyre-health stage, used as a left-border "status light"
# on the Status tile so severity reads at a glance without an emoji.
STAGE_ACCENTS = {
    "Optimal": "#2ED573",
    "Thermal": "#FFC107",
    "Nearing Cliff": "#FFA502",
    "Cliff Reached": "#FF4757",
}


@st.cache_data(show_spinner=False)
def load_sim_laps(year=2024, track="Monza", driver="VER"):
    """Fetch a driver's race laps plus per-lap weather (FastF1 telemetry + weather API)."""
    try:
        session = fastf1.get_session(year, track, 'R')
        session.load(weather=True)
    except Exception:
        session = fastf1.get_session(2024, track, 'R')
        session.load(weather=True)

    # Keep `laps` as FastF1's own Laps object (untouched) - get_telemetry()
    # needs its live session reference. Weather is matched up separately, by
    # LapNumber, rather than merged into `laps` itself (a plain-DataFrame
    # merge result loses that session link and breaks telemetry lookups).
    laps = session.laps.pick_drivers(driver)

    weather = (
        session.weather_data[["Time", "AirTemp", "TrackTemp", "Humidity", "WindSpeed", "Rainfall"]]
        .dropna(subset=["Time"])
        .sort_values("Time")
    )
    lap_weather = pd.merge_asof(
        laps[["LapNumber", "Time"]].sort_values("Time"), weather, on="Time", direction="backward"
    ).set_index("LapNumber")

    try:
        team_color = fastf1.plotting.get_driver_color(driver, session=session)
    except Exception:
        team_color = '#3671C6'

    return laps, lap_weather, team_color


def get_lap_telemetry(laps, lap_number):
    lap = laps[laps['LapNumber'] == lap_number].iloc[0]
    telemetry = lap.get_telemetry().add_distance()

    # Calculate Stress Power
    v_ms = telemetry['Speed'] / 3.6
    time_sec = telemetry['Time'].dt.total_seconds()
    x_smooth = savgol_filter(telemetry['X'], window_length=15, polyorder=3)
    y_smooth = savgol_filter(telemetry['Y'], window_length=15, polyorder=3)
    dx = np.gradient(x_smooth)
    dy = np.gradient(y_smooth)
    heading = np.unwrap(np.arctan2(dy, dx))
    yaw_rate = np.gradient(heading, time_sec)
    lat_g = savgol_filter((v_ms * yaw_rate) / 9.81, 15, 3)
    lon_g = savgol_filter(np.gradient(v_ms, time_sec) / 9.81, 15, 3)
    combined_g = np.sqrt(lat_g**2 + lon_g**2)
    telemetry['Stress_Power'] = combined_g * v_ms

    return telemetry


def _safe_float(row, column: str, default: float) -> float:
    value = row.get(column) if hasattr(row, "get") else None
    return float(value) if value is not None and pd.notna(value) else default


def _tile(label: str, value: str, tooltip: str, accent: str = "#4A90E2") -> str:
    """One compact stat tile for a CSS grid; the tooltip carries detail that
    would otherwise clutter the label.

    Rendered as a single line with no blank line between tiles: Streamlit's
    markdown parser treats a blank (or whitespace-only) line as the end of a
    raw-HTML block, which would make every tile after the first render as
    literal escaped text instead of HTML.
    """
    return (
        f'<div title="{tooltip}" style="min-width: 0; overflow: hidden; box-sizing: border-box; '
        f'background: rgba(255,255,255,0.03); padding: 7px 10px; border: 1px solid rgba(255,255,255,0.08); '
        f'border-left: 3px solid {accent};">'
        f'<div style="font-family: {MONO}; font-size: 9px; letter-spacing: 0.08em; text-transform: uppercase; '
        f'color: #888; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{label}</div>'
        f'<div style="font-family: {MONO}; font-size: 14px; font-weight: 700; color: #eee; '
        f'white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{value}</div>'
        f'</div>'
    )


def render_sim_frame(placeholder, telemetry, current_idx: int, ctx: dict) -> None:
    current_data = telemetry.iloc[current_idx]
    tel_slice = telemetry.iloc[:current_idx + 1]

    with placeholder.container():
        st.markdown(
            f"<div style='font-family:{MONO}; font-size:0.95rem; font-weight:700; letter-spacing:0.03em; "
            f"margin:0 0 0.3rem 0; color:#ddd;'>"
            f"{ctx['race'].upper()} <span style='color:#555;'>/</span> {ctx['driver']} "
            f"<span style='color:#555;'>/</span> LAP {ctx['race_lap']:02d}"
            f"<span style='color:#555;'>/</span>{ctx['max_lap']:02d}"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Status gets its own full-width strip - it carries the longest text
        # (stage + action) and is the single most important readout, so it
        # should never compete for space or get ellipsis-truncated.
        st.markdown(
            f'<div title="Current tyre health stage and recommended action" style="background: rgba(255,255,255,0.04); '
            f'padding: 8px 12px; border: 1px solid rgba(255,255,255,0.08); border-left: 4px solid {ctx["accent"]}; '
            f'margin-bottom: 6px; box-sizing: border-box;">'
            f'<div style="font-family: {MONO}; font-size: 9px; letter-spacing: 0.08em; text-transform: uppercase; color: #888;">STATUS</div>'
            f'<div style="font-family: {MONO}; font-size: 16px; font-weight: 700; color: #fff;">{ctx["status"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # The rest sit in a CSS grid (not flexbox): auto-fit + minmax guarantees
        # every tile keeps at least 130px before the grid wraps to a new row,
        # so tiles can never be squeezed thinner than that - the flexbox
        # version could shrink indefinitely and made text overlap/collide.
        tiles = "".join([
            _tile("Compound", ctx["compound"], "Tyre compound currently fitted", accent="#4A90E2"),
            _tile("Weather", ctx["weather_value"], ctx["weather_tip"], accent="#17A2B8"),
            _tile("Cliff", ctx["cliff"], "Tyre age at which modelled cliff risk reaches 80%", accent="#FF4757"),
            _tile("Pit Window", ctx["pit_window"], "Recommended lap range to box, based on the cliff age", accent="#FFA502"),
            _tile("Degradation", ctx["degradation"], "Latent pace-loss growth rate per lap, with uncertainty", accent="#4A90E2"),
            _tile("Stress", ctx["stress"], "Combined wear multiplier from sensors and driving mode", accent="#9B59B6"),
        ])
        st.markdown(
            f'<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); '
            f'gap: 6px; margin-bottom: 0.35rem;">{tiles}</div>',
            unsafe_allow_html=True,
        )

        telemetry_line = (
            f"SPEED {int(current_data['Speed'])} KM/H &nbsp;&middot;&nbsp; "
            f"GEAR {int(current_data['nGear'])} &nbsp;&middot;&nbsp; "
            f"RPM {int(current_data['RPM'])} &nbsp;&middot;&nbsp; "
            f"THROTTLE {int(current_data['Throttle'])}%"
        )
        st.markdown(
            f'<div style="font-family:{MONO}; font-size:12px; letter-spacing:0.03em; color:#8fd3ff; '
            f'margin-bottom:0.3rem;">{telemetry_line}</div>',
            unsafe_allow_html=True,
        )

        # Merge Map and Telemetry into a single Figure for buttery smooth rendering (dpi=70 speeds up streaming)
        fig = plt.figure(figsize=(11, 5.4), dpi=70)
        fig.patch.set_facecolor('#0e1117')

        # Track Map (Left) - Rotated 90 degrees for wider fit
        ax_map = plt.subplot2grid((4, 7), (0, 0), rowspan=4, colspan=3)
        ax_map.set_facecolor('#0e1117')

        # Rotate coordinates: X_new = Y, Y_new = -X
        tel_x = telemetry['Y']
        tel_y = -telemetry['X']
        curr_x = current_data['Y']
        curr_y = -current_data['X']

        points = np.array([tel_x, tel_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Global cumulative stress scale to prevent flashing
        vmin, vmax = 0.0, 250.0
        norm = mcolors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)

        ax_map.plot(tel_x, tel_y, color='#2a2e39', linewidth=6, zorder=1)
        lc = LineCollection(segments, cmap='turbo', norm=norm, linewidth=3.5, zorder=2)
        lc.set_array(telemetry['Stress_Power'][:-1])
        ax_map.add_collection(lc)

        ax_map.scatter(curr_x, curr_y, color='white', s=120, edgecolors='#ff4757', linewidth=1.5, zorder=5)
        ax_map.axis('equal')
        ax_map.axis('off')

        # Telemetry Channels (Right)
        dist_current = current_data['Distance']
        max_dist = telemetry['Distance'].max()
        channels = [('Speed (km/h)', 'Speed', '#1e90ff'), ('RPM', 'RPM', '#ff4757'), ('Throttle (%)', 'Throttle', '#ffa502'), ('Gear', 'nGear', '#2ed573')]

        axs = [plt.subplot2grid((4, 7), (i, 3), colspan=4) for i in range(4)]

        for i, (label, col_name, color) in enumerate(channels):
            ax = axs[i]
            ax.set_facecolor('#151922')
            # No background full trace, just the live drawn trace!
            ax.plot(tel_slice['Distance'], tel_slice[col_name], color=color, linewidth=2)
            ax.axvline(dist_current, color='white', linestyle=':', linewidth=1.5, alpha=0.8)
            ax.set_ylabel(label, color='white', fontsize=9)
            ax.tick_params(colors='white', labelsize=8)
            ax.grid(True, linestyle='--', alpha=0.2)
            ax.set_xlim(0, max_dist)

            if col_name == 'nGear':
                ax.set_yticks(range(1, 9))
                ax.set_ylim(0, 9)
            elif col_name == 'Speed':
                ax.set_ylim(0, 350)
            elif col_name == 'RPM':
                ax.set_ylim(0, 13000)
            elif col_name == 'Throttle':
                ax.set_ylim(0, 105)

            if i < 3: ax.set_xticklabels([])

        axs[-1].set_xlabel('Distance (meters)', color='white')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def load_model(path: Path = Path("models/tyre_degradation_2025.pkl")) -> LatentTyreModel:
    """Load the state-space bundle, falling back to explicit neutral priors."""
    if path.exists():
        with path.open("rb") as handle:
            bundle = pickle.load(handle)
        if isinstance(bundle, dict) and isinstance(bundle.get("model"), LatentTyreModel):
            return bundle["model"]
    return LatentTyreModel()


def demo_stint(track: str = "Demo Track", compound: str = "SOFT") -> pd.DataFrame:
    """A clearly labelled deterministic demo series when a CSV is not supplied."""
    life = np.arange(1, 26, dtype=float)
    # Steeper degradation curve to ensure the cliff is reached around lap 18
    degradation = np.maximum(0.0, 0.04 * (life - 2) + 0.012 * np.maximum(life - 12, 0) ** 2)
    return pd.DataFrame({
        "TyreLife": life,
        "Degradation_Delta": degradation,
        "Source": "Illustrative demo",
        "Race": track,
        "Driver": "DEMO",
        "Compound": compound,
        "Stint_ID": 1
    })


def fan_chart(observed: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    """Build the pace-loss fan chart for the compact side panel (no in-figure
    title or legend - both would overlap a narrow column; colours and the
    cliff line are enough to read once the reader knows the app)."""
    fig = go.Figure()
    if not observed.empty:
        if "Latent_Degradation_Mean" in observed:
            fig.add_trace(go.Scatter(
                x=observed["TyreLife"], y=observed["Latent_Degradation_Mean"],
                mode='lines', name="Filtered latent state", line=dict(color="#4A90E2", width=2)
            ))
        fig.add_trace(go.Scatter(
            x=observed["TyreLife"], y=observed["Degradation_Delta"],
            mode='markers', name="Observed pace loss", marker=dict(color="#E0E0E0", size=6, opacity=0.8)
        ))

    # 90% interval
    fig.add_trace(go.Scatter(
        x=forecast["TyreLife"], y=forecast["p95_s"],
        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=forecast["TyreLife"], y=forecast["p05_s"],
        mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(114, 183, 178, 0.15)', name='90% interval', hoverinfo='skip'
    ))

    # 70% interval
    fig.add_trace(go.Scatter(
        x=forecast["TyreLife"], y=forecast["p85_s"],
        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=forecast["TyreLife"], y=forecast["p15_s"],
        mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(114, 183, 178, 0.3)', name='70% interval', hoverinfo='skip'
    ))

    # Mean forecast
    fig.add_trace(go.Scatter(
        x=forecast["TyreLife"], y=forecast["mean_s"],
        mode='lines', name="Forecast mean", line=dict(color="#FF4B4B", width=3)
    ))

    fig.add_hline(
        y=1.5, line_dash="dash", line_color="#FF4B4B", opacity=0.7,
        annotation_text="1.5s CLIFF RISK", annotation_position="top left",
        annotation_font=dict(color="#FF4B4B", size=10, family=MONO),
    )

    max_y = 2.5
    if not observed.empty:
        max_y = max(max_y, observed["Degradation_Delta"].max() * 1.1)
    if not forecast.empty:
        max_y = max(max_y, forecast["p95_s"].max() * 1.1)

    fig.add_hrect(y0=0.0, y1=0.5, fillcolor="rgba(0, 255, 0, 0.05)", line_width=0, layer="below")
    fig.add_hrect(y0=0.5, y1=1.0, fillcolor="rgba(255, 255, 0, 0.05)", line_width=0, layer="below")
    fig.add_hrect(y0=1.0, y1=1.5, fillcolor="rgba(255, 165, 0, 0.05)", line_width=0, layer="below")
    fig.add_hrect(y0=1.5, y1=max_y, fillcolor="rgba(255, 0, 0, 0.05)", line_width=0, layer="below")

    fig.update_layout(
        title="",
        xaxis_title="TYRE LIFE (LAPS)",
        yaxis_title="PACE LOSS (S)",
        yaxis=dict(range=[0, max_y]),
        hovermode="x unified",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=500,
        margin=dict(l=45, r=15, t=15, b=40),
        font=dict(family=MONO, size=11),
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="TrackShift — Tyre Telemetry", layout="wide")

    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}
        .block-container {{
            padding-top: 3.5rem;
            padding-bottom: 0.5rem;
            max-width: 100%;
        }}
        [data-testid="stVerticalBlock"] {{
            gap: 0.4rem;
        }}
        [data-testid="stMetric"], .stSlider, .stToggle, .stSelectSlider {{
            min-width: 0;
        }}
        [data-testid="column"] {{
            padding: 0 0.15rem;
            min-width: 0;
        }}
        hr {{
            margin: 0.3rem 0 !important;
        }}
        [data-testid="stCaptionContainer"] p {{
            margin-bottom: 0 !important;
        }}
        [data-testid="stSidebar"] {{
            background-image: linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px),
                               linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px);
            background-size: 18px 18px;
        }}
        [data-testid="stSidebarHeader"] {{ padding-bottom: 0; }}
        .stTabs {{ display: none; }}
        </style>
    """, unsafe_allow_html=True)

    st.markdown(
        f"<div style='display:flex; align-items:center; gap:0.5rem; padding-bottom:0.4rem; "
        f"border-bottom:1px solid rgba(255,255,255,0.08); margin-bottom:0.5rem;'>"
        f"<span style='width:8px; height:8px; border-radius:50%; background:#FF4757; display:inline-block; "
        f"box-shadow:0 0 6px #FF4757;'></span>"
        f"<span style='font-family:{MONO}; font-size:1.15rem; font-weight:700; letter-spacing:0.06em; "
        f"text-transform:uppercase; color:#fff;'>TrackShift</span>"
        f"<span style='font-family:{MONO}; font-size:0.75rem; color:#666; letter-spacing:0.04em;'>"
        f"// LATENT TYRE-STATE TELEMETRY</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.sidebar.header("Race Setup")
    import glob
    local_files = glob.glob("outputs/latent_stint_observations_*.csv") + glob.glob("latent_stint_observations_*.csv")
    upload = local_files[-1] if local_files else None

    uploaded_data = pd.DataFrame()
    selected_race = selected_driver = None
    if upload is None:
        st.sidebar.caption("No stint data found locally — showing demo mode.")
    else:
        try:
            uploaded_data = pd.read_csv(upload)
            inference_cols = ["Latent_Degradation_Mean", "Latent_Degradation_Std", "Prior_Degradation_Mean", "Prior_Degradation_Std", "Latent_Degradation_Rate"]
            uploaded_data = uploaded_data.drop(columns=[c for c in inference_cols if c in uploaded_data.columns])
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            return

        races = uploaded_data["Race"].unique().tolist() if "Race" in uploaded_data.columns else ["Unknown"]
        selected_race = st.sidebar.selectbox("Select Race", races)
        race_data = uploaded_data[uploaded_data["Race"] == selected_race] if "Race" in uploaded_data.columns else uploaded_data

        drivers = race_data["Driver"].unique().tolist() if "Driver" in race_data.columns else ["Unknown"]
        selected_driver = st.sidebar.selectbox("Select Driver", drivers)

    with st.sidebar.expander("Virtual Sensor Setup", expanded=False):
        use_live_weather = st.checkbox(
            "Use live weather for track temp", value=True,
            help="On: track temperature comes from FastF1's per-lap weather data. Off: use the slider below instead.",
        )
        track_temp_override = st.slider("Track temperature (°C)", 15, 60, 35, disabled=use_live_weather)
        severity = st.slider("Track tyre-stress score", 1.0, 5.0, 3.0, 0.1)
        profile = {"traction": 3.0, "tyre_stress": severity, "lateral": 3.0, "braking": 3.0, "downforce": 3.0}
        seed_sensor = default_sensor_state(profile, 35.0)
        wheel_slip_override = st.slider("Wheel slip (%)", 2.0, 12.0, float(seed_sensor["wheel_slip_pct"]), .1)
        carcass_temp_override = st.slider("Carcass temperature (°C)", 70.0, 125.0, float(seed_sensor["carcass_temp_c"]), .5)

    st.sidebar.header("Driving Situation")
    situation = st.sidebar.radio("Current Mode", ["Standard", "Aggressive / Pushing", "Traffic / Dirty Air", "Conserving"])
    situation_multiplier = 1.0
    if situation == "Aggressive / Pushing":
        situation_multiplier = 1.20
    elif situation == "Traffic / Dirty Air":
        situation_multiplier = 1.10
    elif situation == "Conserving":
        situation_multiplier = 0.85

    model = load_model()

    if upload is not None and not uploaded_data.empty:
        sim_race, sim_driver = selected_race, selected_driver
    else:
        demo_col1, demo_col2 = st.columns(2)
        sim_race = demo_col1.selectbox("Track / Circuit", DEMO_TRACK_CHOICES, index=DEMO_TRACK_CHOICES.index("Italian Grand Prix"))
        sim_driver = demo_col2.selectbox("Driver", KNOWN_2025_DRIVERS, index=KNOWN_2025_DRIVERS.index("VER"))

    with st.spinner("Loading race + weather data..."):
        sim_laps, lap_weather, team_color = load_sim_laps(2025, sim_race, sim_driver)

    ver_data = pd.DataFrame()
    if upload is not None and not uploaded_data.empty and {"Driver", "Race"}.issubset(uploaded_data.columns):
        ver_data = uploaded_data[(uploaded_data["Driver"] == sim_driver) & (uploaded_data["Race"] == sim_race)]

    if ver_data.empty:
        st.caption("Demo data — upload a latent_stint_observations CSV for real predictions.")

    max_lap = int(sim_laps['LapNumber'].max())

    ctrl1, ctrl2, ctrl3 = st.columns([1, 1.6, 1])
    play_animation = ctrl1.toggle("Continuous Replay", value=False)
    start_lap = ctrl2.slider("Race Lap", 1, max_lap, 1)
    playback_speed = ctrl3.select_slider("Playback Speed", options=["1x", "2x", "4x", "8x", "16x", "32x"], value="4x")

    speed_multiplier = int(playback_speed.replace("x", ""))
    step_size = max(1, speed_multiplier)

    col_main, col_fan = st.columns([1.8, 1])
    with col_main:
        main_placeholder = st.empty()
    with col_fan:
        fan_placeholder = st.empty()

    def run_lap(race_lap, is_animating):
        try:
            lap_info = sim_laps.loc[sim_laps['LapNumber'] == race_lap].iloc[0]
            sim_tel = get_lap_telemetry(sim_laps, race_lap)
        except Exception:
            return

        current_stint = lap_info['Stint']
        current_tyre_life = lap_info['TyreLife']
        active_compound = lap_info['Compound']

        # Weather as a live predictive input: real per-lap track temperature
        # (not a manual slider) drives the virtual-sensor thermal/pressure
        # state, which in turn scales the degradation forecast below.
        try:
            weather_row = lap_weather.loc[race_lap]
        except KeyError:
            weather_row = pd.Series(dtype=float)
        live_track_temp = _safe_float(weather_row, "TrackTemp", 35.0)
        live_air_temp = _safe_float(weather_row, "AirTemp", 22.0)
        live_humidity = _safe_float(weather_row, "Humidity", 50.0)
        live_wind = _safe_float(weather_row, "WindSpeed", 0.0)
        live_rain = bool(weather_row["Rainfall"]) if "Rainfall" in weather_row and pd.notna(weather_row["Rainfall"]) else False

        effective_track_temp = live_track_temp if use_live_weather else float(track_temp_override)
        sensor = default_sensor_state(profile, effective_track_temp)
        sensor["wheel_slip_pct"] = wheel_slip_override
        sensor["carcass_temp_c"] = carcass_temp_override
        load = wear_load_breakdown(sensor)

        weather_value = f"{live_track_temp:.0f}°C · {'WET' if live_rain else 'DRY'}"
        weather_tip = (
            f"Track {live_track_temp:.1f}°C · Air {live_air_temp:.1f}°C · "
            f"Humidity {live_humidity:.0f}% · Wind {live_wind:.1f} km/h · "
            f"{'Rain — model trained on dry stints only, treat as indicative' if live_rain else 'Dry'}"
            + ("" if use_live_weather else f" · Model is using the manual {track_temp_override:.0f}°C override, not this reading")
        )

        stint_data = ver_data[ver_data["Stint_ID"] == current_stint] if "Stint_ID" in ver_data.columns else pd.DataFrame()
        if stint_data.empty:
            stint_data = demo_stint(sim_race, str(active_compound))

        sim_observed = stint_data[stint_data["TyreLife"] <= current_tyre_life].copy()
        if sim_observed.empty:
            sim_observed = stint_data.iloc[[0]].copy()

        current_lap_stress = 1.0
        if not sim_observed.empty and "Degradation_Delta" in sim_observed.columns:
            current_lap_stress = 1.0 + (float(sim_observed.iloc[-1]["Degradation_Delta"]) * 0.1)

        sim_posterior = model.infer_stint(sim_observed, wear_multiplier=load["multiplier"])
        sim_state_row = sim_posterior.iloc[-1]
        sim_state = model.initial_state(float(sim_state_row["TyreLife"]), load["multiplier"])
        sim_state.mean_s, sim_state.rate_s_per_lap = float(sim_state_row["Latent_Degradation_Mean"]), float(sim_state_row["Latent_Degradation_Rate"])
        sim_state.covariance[0, 0] = float(sim_state_row["Latent_Degradation_Std"]) ** 2
        sim_state.rate_s_per_lap = sim_state.rate_s_per_lap * situation_multiplier * load["multiplier"] * current_lap_stress

        sim_horizon = np.arange(float(sim_state.tyre_life), float(sim_state.tyre_life) + 16)
        sim_forecast = model.forecast(sim_state, sim_horizon)
        sim_cliff_lap = first_cliff_forecast(sim_forecast)

        if sim_state.mean_s < 0.5: sim_tyre_stage, sim_action = "Optimal", "Maintain Pace"
        elif sim_state.mean_s < 1.0: sim_tyre_stage, sim_action = "Thermal", "Monitor"
        elif sim_state.mean_s < 1.5: sim_tyre_stage, sim_action = "Nearing Cliff", "Prepare To Pit"
        else: sim_tyre_stage, sim_action = "Cliff Reached", "Pit Immediately"

        deg_rate_str = f"{sim_state.rate_s_per_lap:.2f}±{np.sqrt(sim_state.covariance[0,0]):.2f}s"
        stress_value = load['multiplier'] * situation_multiplier * current_lap_stress

        ctx = {
            "race": sim_race, "driver": sim_driver, "race_lap": race_lap, "max_lap": max_lap,
            "status": f"{sim_tyre_stage} — {sim_action}", "accent": STAGE_ACCENTS.get(sim_tyre_stage, "#4A90E2"),
            "compound": str(active_compound),
            "weather_value": weather_value, "weather_tip": weather_tip,
            "cliff": f"Lap {int(sim_cliff_lap)}" if sim_cliff_lap else "Stable",
            "pit_window": f"Lap {int(sim_cliff_lap) - 2}-{int(sim_cliff_lap)}" if sim_cliff_lap else "Stable (>15L)",
            "degradation": deg_rate_str,
            "stress": f"{stress_value:.2f}x",
        }

        with fan_placeholder.container():
            st.markdown(
                f"<div style='font-family:{MONO}; font-size:0.95rem; font-weight:700; letter-spacing:0.03em; "
                f"margin:0 0 0.3rem 0; color:#ddd;'>STRATEGY FORECAST</div>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(fan_chart(sim_posterior, sim_forecast), use_container_width=True, key=f"fan_{race_lap}")

        total_points = len(sim_tel)
        if is_animating:
            for idx in range(0, total_points, step_size):
                render_sim_frame(main_placeholder, sim_tel, idx, ctx)
        else:
            render_sim_frame(main_placeholder, sim_tel, total_points - 1, ctx)

    if play_animation:
        for r_lap in range(start_lap, max_lap + 1):
            run_lap(r_lap, True)
    else:
        run_lap(start_lap, False)


if __name__ == "__main__":
    main()
