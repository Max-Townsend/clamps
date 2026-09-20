from __future__ import annotations

import numpy as np


def wrap360(angle_deg):
    return np.mod(np.asarray(angle_deg, dtype=float), 360.0)
