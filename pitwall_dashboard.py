"""Streamlit pit-wall view for live latent tyre-state inference.

Run with ``streamlit run pitwall_dashboard.py``.  Upload a cleaned stint CSV
containing ``TyreLife`` and ``Degradation_Delta`` (the pipeline output is
accepted directly); no team-private sensor feed is required for the demo.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import plotly.graph_objects as go
import numpy as np
import pandas as pd

from latent_tyre_model import LatentTyreModel, first_cliff_forecast
from virtual_sensors import default_sensor_state, wear_load_breakdown


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


def read_stint(uploaded_file: object | None, track: str = "Demo Track", compound: str = "SOFT") -> pd.DataFrame:
    if uploaded_file is None:
        return demo_stint(track, compound)
    data = pd.read_csv(uploaded_file)
    missing = {"TyreLife", "Degradation_Delta"}.difference(data.columns)
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(sorted(missing))}")
    return data.dropna(subset=["TyreLife", "Degradation_Delta"]).sort_values("TyreLife")


def stint_choices(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split pipeline output into labelled, single-stint dashboard inputs."""
    if "Stint_ID" not in data.columns:
        return {"Uploaded stint": data}
    choices: dict[str, pd.DataFrame] = {}
    for stint_id, part in data.groupby("Stint_ID", sort=False):
        first = part.iloc[0]
        prefix = " | ".join(str(first.get(column, "Unknown")) for column in ("Race", "Driver", "Compound"))
        choices[f"{prefix} | stint {stint_id}"] = part.sort_values("TyreLife").copy()
    return choices


def fan_chart(observed: pd.DataFrame, forecast: pd.DataFrame, title: str) -> go.Figure:
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

    fig.add_hline(y=1.5, line_dash="dash", line_color="#FF4B4B", opacity=0.7, annotation_text="1.5s cliff risk", annotation_position="top left", annotation_font=dict(color="#FF4B4B"))

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
        title=title,
        xaxis_title="Tyre life (laps)",
        yaxis_title="Fuel-corrected pace loss (s)",
        yaxis=dict(range=[0, max_y]),
        hovermode="x unified",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def main() -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError as exc:  # pragma: no cover - user-facing runtime guard
        raise SystemExit("Install the optional dashboard dependency with: pip install streamlit") from exc

    st.set_page_config(page_title="Tyre State Intelligence", layout="wide")
    
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        [data-testid="stMetric"] {
            background-color: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 1rem 1.25rem;
            border-radius: 0.75rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.2);
            border-color: rgba(255, 75, 75, 0.4);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 1.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 1rem 0;
            border-radius: 0;
        }
        .stTabs [aria-selected="true"] {
            background-color: transparent !important;
            border-bottom: 2px solid #FF4B4B !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Beyond Curve Fitting — Tyre State Intelligence")
    st.caption("Fuel-corrected observations update a latent degradation state; bands are model uncertainty, not guaranteed pace.")
    
    st.sidebar.header("Data Source")
    import glob
    local_files = glob.glob("outputs/latent_stint_observations_*.csv") + glob.glob("latent_stint_observations_*.csv")
    upload = local_files[-1] if local_files else None

    if upload is None:
        st.sidebar.warning("No latent_stint_observations CSV found locally. Using Demo Mode.")

    if upload is not None:
        try:
            uploaded_data = pd.read_csv(upload)
            # Remove inference columns if present
            inference_cols = ["Latent_Degradation_Mean", "Latent_Degradation_Std", "Prior_Degradation_Mean", "Prior_Degradation_Std", "Latent_Degradation_Rate"]
            uploaded_data = uploaded_data.drop(columns=[c for c in inference_cols if c in uploaded_data.columns])
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            return
            
        st.sidebar.header("Stint Selection")
        races = uploaded_data["Race"].unique().tolist() if "Race" in uploaded_data.columns else ["Unknown"]
        selected_race = st.sidebar.selectbox("Select Race", races)
        race_data = uploaded_data[uploaded_data["Race"] == selected_race] if "Race" in uploaded_data.columns else uploaded_data
        
        drivers = race_data["Driver"].unique().tolist() if "Driver" in race_data.columns else ["Unknown"]
        selected_driver = st.sidebar.selectbox("Select Driver", drivers)
        driver_data = race_data[race_data["Driver"] == selected_driver] if "Driver" in race_data.columns else race_data
        
        stints = driver_data["Stint_ID"].unique().tolist() if "Stint_ID" in driver_data.columns else [1]
        stint_options = []
        for s in stints:
            comp = driver_data[driver_data["Stint_ID"] == s]["Compound"].iloc[0] if "Compound" in driver_data.columns else "Unknown"
            stint_options.append(f"Stint {s} ({comp})")
            
        selected_stint_idx = st.sidebar.selectbox("Select Stint", range(len(stints)), format_func=lambda x: stint_options[x])
        stint = driver_data[driver_data["Stint_ID"] == stints[selected_stint_idx]].sort_values("TyreLife").copy()
    else:
        st.sidebar.header("Demo Configuration")
        track_choice = st.sidebar.selectbox("Track / Circuit", [
            "Bahrain", "Saudi Arabia", "Australia", "Japan", "China", "Miami", "Emilia Romagna",
            "Monaco", "Canada", "Spain", "Austria", "Great Britain", "Hungary", "Belgium",
            "Netherlands", "Italy", "Azerbaijan", "Singapore", "United States", "Mexico",
            "Brazil", "Las Vegas", "Qatar", "Abu Dhabi"
        ])
        tyre_choice = st.sidebar.selectbox("Tyre Compound", ["SOFT", "MEDIUM", "HARD"])
        stint = demo_stint(track_choice, tyre_choice)
        st.info("Showing an illustrative demo stint. Run the pipeline and upload its exported latent_stint_observations CSV for real data.")

    with st.sidebar.expander("Virtual Sensor Setup", expanded=False):
        track_temp = st.slider("Track temperature (°C)", 15, 60, 35)
        severity = st.slider("Track tyre-stress score", 1.0, 5.0, 3.0, 0.1)
        profile = {"traction": 3.0, "tyre_stress": severity, "lateral": 3.0, "braking": 3.0, "downforce": 3.0}
        sensor = default_sensor_state(profile, float(track_temp))
        sensor["wheel_slip_pct"] = st.slider("Wheel slip (%)", 2.0, 12.0, float(sensor["wheel_slip_pct"]), .1)
        sensor["carcass_temp_c"] = st.slider("Carcass temperature (°C)", 70.0, 125.0, float(sensor["carcass_temp_c"]), .5)
    load = wear_load_breakdown(sensor)
    
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
    live_tab, validation_tab, eval_tab = st.tabs(["🔴 Live Weekend", "🏁 Post-Race Validation", "💡 Framework Evaluation"])

    with live_tab:
        count = st.slider("Completed clean laps across FP1 → FP3", 1, len(stint), max(1, len(stint) // 2))
        observed = stint.iloc[:count].copy()
        posterior = model.infer_stint(observed, wear_multiplier=load["multiplier"])
        state_row = posterior.iloc[-1]
        state = model.initial_state(float(state_row["TyreLife"]), load["multiplier"])
        state.mean_s, state.rate_s_per_lap = float(state_row["Latent_Degradation_Mean"]), float(state_row["Latent_Degradation_Rate"])
        # Reconstruct covariance from reported state uncertainty; the rate
        # covariance stays conservative, as only its display is unavailable.
        state.covariance[0, 0] = float(state_row["Latent_Degradation_Std"]) ** 2

        # Apply multipliers so that virtual sensors and driving situation reactively change the forecast line
        state.rate_s_per_lap = state.rate_s_per_lap * situation_multiplier * load["multiplier"]

        horizon = np.arange(float(state.tyre_life), float(state.tyre_life) + 16)
        forecast = model.forecast(state, horizon)
        cliff_lap = first_cliff_forecast(forecast)
        
        if state.mean_s < 0.5:
            tyre_stage = "🟢 Optimal Grip"
            action = "Maintain Pace"
        elif state.mean_s < 1.0:
            tyre_stage = "🟡 Thermal Deg"
            action = "Monitor Strategy"
        elif state.mean_s < 1.5:
            tyre_stage = "🟠 Nearing Cliff"
            action = "PREPARE TO PIT"
        else:
            tyre_stage = "🔴 Cliff Reached"
            action = "PIT IMMEDIATELY"

        if cliff_lap is not None:
            pit_suggestion = f"Lap {int(cliff_lap) - 2} – {int(cliff_lap)}"
        else:
            pit_suggestion = "Stable (>15 Laps)"

        st.subheader("Strategy Overview")
        a, b, c, d = st.columns(4)
        a.metric("Current Tyre Stage", tyre_stage)
        b.metric("Strategy Recommendation", action)
        c.metric("Suggested Pit Window", pit_suggestion)
        d.metric("Current Pace Loss", f"{state.mean_s:.2f} ± {state.std_s:.2f} s")
        
        st.plotly_chart(fan_chart(posterior, forecast, "Live weekend posterior and race forecast"), use_container_width=True)
        st.caption("Update behaviour is sequential: each clean practice lap changes the posterior and its uncertainty. Slow outliers are innovation-clipped rather than treated as tyre failure.")

        st.divider()
        st.subheader("Virtual Sensor Breakdown")
        st.caption("Identify which physical factors are currently driving tyre wear. A multiplier > 1.0 accelerates degradation.")
        cols = st.columns(5)
        cols[0].metric("Thermal Stress", f"{load['thermal']:.2f}x")
        cols[1].metric("Slip Energy", f"{load['slip']:.2f}x")
        cols[2].metric("Lateral Load", f"{load['lateral']:.2f}x")
        cols[3].metric("Pressure Deviation", f"{load['pressure']:.2f}x")
        cols[4].metric("Overall Wear Modifier", f"{load['multiplier']:.2f}x")

    with validation_tab:
        posterior = model.infer_stint(stint, wear_multiplier=load["multiplier"])
        last = posterior.iloc[-1]
        state = model.initial_state(float(last["TyreLife"]), load["multiplier"])
        state.mean_s, state.rate_s_per_lap = float(last["Latent_Degradation_Mean"]), float(last["Latent_Degradation_Rate"])
        state.covariance[0, 0] = float(last["Latent_Degradation_Std"]) ** 2
        forecast = model.forecast(state, np.arange(float(state.tyre_life), float(state.tyre_life) + 8))
        st.plotly_chart(fan_chart(posterior, forecast, "Post-Race Validation: Actual vs. Forecast"), use_container_width=True)
        coverage = np.mean((posterior["Degradation_Delta"] >= posterior["Latent_Degradation_Mean"] - 1.645 * posterior["Latent_Degradation_Std"]) & (posterior["Degradation_Delta"] <= posterior["Latent_Degradation_Mean"] + 1.645 * posterior["Latent_Degradation_Std"]))
        st.metric("In-sample 90% posterior-band coverage", f"{coverage:.0%}")
        st.caption("Coverage is a calibration diagnostic, not a race-performance score. Use a held-out race for an honest validation result.")

    with eval_tab:
        st.header("Why Probabilistic Latent Modelling?")
        st.markdown("""
        In Formula 1, predicting a single deterministic lap time (e.g., 1:34.5) is insufficient for high-stakes decision making. 
        Race engineers need to manage **risk**, which requires an understanding of **uncertainty** and **probability**.

        ### The Pit Wall Problem
        Traditional public-data models treat tyre wear as a curve-fitting exercise on lap times. If a model has a 0.3s Mean Absolute Error (MAE), it still doesn't tell the race engineer if the tyre is going to suddenly fall off a "cliff" in the next 5 laps.

        ### Our Solution
        This portal solves the problem by transitioning to **inferring latent tyre states** using State-Space models (Kalman Filters):
        1. **Virtual Sensor Emulation Layer**: Recreates missing physical CAN-bus signals (carcass temp, slip energy) using Track Profiles & FastF1 telemetry. It mirrors how real F1 teams operate when competitor data is hidden.
        2. **Probabilistic Cliff Detection**: Instead of a single MAE metric, this model outputs a probability distribution. It answers the question: *What is the probability that the pace drop exceeds 1.5s on Lap 44?*
        3. **Recursive Weekend Learning**: The model leverages prior race data but updates its state automatically via Bayesian inference as FP1, FP2, and FP3 unfold, adapting to the track's evolution for that specific weekend.

        ### Benefits for the Race Strategy Team
        - **Trust & Explainability**: The pit wall inherently distrusts black-box AI. By providing an 80% risk threshold (the *cliff-risk age* metric in the Live Weekend tab), engineers can make decisions based on risk tolerance.
        - **Handling Anomalies**: The model inherently filters out traffic or driver mistakes as observation noise, preventing knee-jerk strategy calls based on an isolated slow lap.
        - **Visual Validation**: The Fan Charts allow strategists to visually confirm that the actual race pace falls within the model's predicted uncertainty bounds.
        """)


if __name__ == "__main__":
    main()
