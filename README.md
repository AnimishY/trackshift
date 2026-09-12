# Beyond Curve Fitting: Virtual Sensors and Probabilistic Tyre Intelligence

## Pitch

Formula 1 strategy is a risk-management problem, not a lap-time leaderboard. A
pit wall does not need a black-box estimate that a lap will be 94.5 seconds; it
needs an auditable answer to a decision question: *what is the chance that this
stint has reached a tyre cliff before the next pit window?* This project turns
public FastF1 timing, weather and track-profile data into that answer.

The first layer is a virtual-sensor system. FastF1 cannot expose a team's
private tyre CAN channels, so the project reconstructs reproducible proxies
for thermal stress, pressure deviation, slip energy, lateral load, braking
thermal load and aero load. The proxies are deliberately transparent: a user
can see the track and setup inputs, inspect the formula, and adjust a
simulated sensor state. The initial emphasis on slip, temperature and lateral
load follows established tyre-dynamics thinking; detailed physical tyre
models explain why load, camber and slip alter the contact patch and heat
generation.

The second layer treats degradation as a hidden state. Fuel-corrected lap-time
loss is a noisy observation, not the state itself. A two-state Kalman filter
estimates latent pace loss and its growth rate and carries their covariance
forward. Its robust innovation clipping stops an isolated traffic lap or
driver mistake from being declared a tyre failure. Every future lap therefore
has a mean forecast, a 70% and 90% uncertainty interval, and a normal-CDF
probability of exceeding the configurable 1.5-second cliff threshold.

Finally, the filter learns online. Clean FP1, FP2 and FP3 observations update
the posterior before the race forecast. The dashboard makes this visible as a
fan chart and reports the first tyre age with at least 80% cliff risk. This is
not a claim to recover secret competitor telemetry: it is a transparent,
public-data decision-support prototype whose assumptions, uncertainty and
failure modes are explicit.

## Architecture

```text
FastF1 race laps + weather + track-factor CSV
                  |
                  v
          data_pipeline.py
  green-flag filtering + fuel correction
                  |
                  v
           virtual_sensors.py
  thermal / slip / lateral load proxies
                  |
                  +----> observable sensor controls in dashboard
                  v
          latent_tyre_model.py
  state = [latent degradation, loss-per-lap]
  predict -> robust update -> covariance
                  |
                  v
   cliff probability + 70/90% fan chart
                  |
                  v
          pitwall_dashboard.py
    Live Weekend | Post-Race Validation
```

## Quick start

Use Python 3.10+ and install the core requirements:

```bash
pip install fastf1 numpy pandas scikit-learn matplotlib streamlit
```

Train on a track-factor CSV containing `race`, `tyre_compounds`, and the
columns listed in `tyre_features.PROFILE_COLUMNS`:

```bash
python 14_2025_tyre_degradation_intelligence.py --track-factors-csv track_factors.csv --year 2025
streamlit run pitwall_dashboard.py
```

The first command downloads FastF1 race timing and creates these outputs:

```text
outputs/latent_stint_observations_2025.csv  <- upload this to the dashboard
outputs/tyre_cliff_report_2025.csv          <- per-stint race summary
outputs/race_mae_2025.csv                   <- held-out/analysis error summary
```

In the dashboard, upload `latent_stint_observations_2025.csv`, then choose a
single `Race | Driver | Compound | Stint` in the sidebar. The Live Weekend tab
uses the first N clean laps as FP observations and forecasts the next 15 tyre
ages. The Post-Race tab shows the complete selected stint. It displays a
labelled illustrative series only when no file is uploaded. A model bundle
created before this pivot is intentionally rejected: retrain to create the
latent-state bundle.

## Evidence and modelling boundaries

- Pacejka's *Tire and Vehicle Dynamics* is the physical reference for the
  slip/load/camber hierarchy. [Elsevier book record](https://shop.elsevier.com/books/tire-and-vehicle-dynamics/pacejka/978-0-08-097016-5)
- Farroni and collaborators' TRICK work is a relevant tyre/road thermal and
  wear modelling reference; cite the exact edition/DOI used by the team in a
  final submission rather than relying on an unchecked secondary reference.
- Kalman filtering is used here as an engineering state estimator, not as a
  Bayesian claim stronger than the public data support. The robust update is a
  practical approximation to heavy-tailed observation noise, not a fitted
  skewed-t likelihood.
- The listed wear weights are normalised physical priors, not measured F1
  coefficients. `update_wear_weight_priors` permits bounded weekend evidence
  updates and preserves their sum.

See `Methodology_and_Weights.pdf` for the concise judges' brief and
`bayesian_deg_model.ipynb` for the equations and executable walkthrough.
