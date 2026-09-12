"""Public-data virtual tyre sensors.

This is the presentation-facing name for :mod:`tyre_features`; importing from
either module remains supported for backwards compatibility.
"""

from tyre_features import (  # noqa: F401
    ENGINEERING_FEATURE_COLUMNS,
    PROFILE_COLUMNS,
    WEAR_WEIGHTS,
    WEAR_WEIGHT_PRIOR_STD,
    default_sensor_state,
    derive_race_sensor_proxies,
    wear_load_breakdown,
    update_wear_weight_priors,
)

__all__ = [
    "ENGINEERING_FEATURE_COLUMNS", "PROFILE_COLUMNS", "WEAR_WEIGHTS", "WEAR_WEIGHT_PRIOR_STD",
    "default_sensor_state", "derive_race_sensor_proxies", "wear_load_breakdown", "update_wear_weight_priors",
]
