"""Parameter bounds and signed clamp inputs for the learning models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ParameterSpec:
    """Real-space bounded parameter specification."""

    name: str
    lower: float
    upper: float
    init: float


def build_trial_clamp(dat):
    """Build a signed clamp-angle series, zeroed during non-clamped trials."""
    if "trial_clamp" in dat.columns:
        signed_clamp = pd.to_numeric(dat["trial_clamp"], errors="coerce").fillna(0.0)
    elif "clamp_angle_relative" in dat.columns:
        signed_clamp = pd.to_numeric(dat["clamp_angle_relative"], errors="coerce").fillna(0.0)
    elif {"target_clamp_sign", "clamp_magnitude"}.issubset(dat.columns):
        signed_clamp = pd.to_numeric(dat["target_clamp_sign"], errors="coerce").fillna(
            0.0
        ) * pd.to_numeric(dat["clamp_magnitude"], errors="coerce").fillna(0.0)
    elif "clamp_magnitude" in dat.columns:
        signed_clamp = pd.to_numeric(dat["clamp_magnitude"], errors="coerce").fillna(0.0)
    else:
        raise ValueError("Could not infer signed clamp angles from dataframe columns.")

    if "clamped" not in dat.columns:
        return signed_clamp.to_numpy(dtype=float)

    clamped = dat["clamped"].to_numpy(dtype=bool)
    return np.where(clamped, signed_clamp.to_numpy(dtype=float), 0.0)
