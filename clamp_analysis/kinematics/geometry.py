"""Clamp magnitudes and rotation into the target-centered coordinate frame."""

from __future__ import annotations

import numpy as np

CLAMPS = [
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    10.0,
    15.0,
    20.0,
    25.0,
    30.0,
    35.0,
    40.0,
    45.0,
    50.0,
    55.0,
    60.0,
    65.0,
    70.0,
    75.0,
    80.0,
    85.0,
    90.0,
    95.0,
    100.0,
    105.0,
    110.0,
    115.0,
    120.0,
    125.0,
    130.0,
    135.0,
    140.0,
    145.0,
    150.0,
    155.0,
    160.0,
    165.0,
    170.0,
]


def rotate_to_target(
    x: np.ndarray, y: np.ndarray, target_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(float(target_deg))
    x_rot = x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)
    return x_rot, y_rot
