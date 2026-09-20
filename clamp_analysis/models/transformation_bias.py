"""Spatial transformation of movement start and endpoint coordinates."""

from __future__ import annotations

import numpy as np

from clamp_analysis.models.biomechanical import wrap360
from clamp_analysis.models.numerics import wrap_angle


def _bias_vector_from_mag_angle(bias_mag, bias_angle_deg):
    bias_mag = float(bias_mag)
    bias_angle_rad = np.deg2rad(float(bias_angle_deg))
    return np.asarray(
        [
            bias_mag * np.cos(bias_angle_rad),
            bias_mag * np.sin(bias_angle_rad),
        ],
        dtype=float,
    )


def transformation_policy_mean(
    plan_abs_deg,
    ref_x,
    ref_y,
    bias_mag,
    bias_angle_deg,
    power_exp=1.0,
    target_radius=1.0,
):
    """Return absolute movement angles after a distance-scaled spatial bias.

    Angles are in degrees. Reference coordinates and target_radius share units;
    bias scales with distance from (ref_x, ref_y) raised to power_exp.
    """

    plan_abs_deg = wrap360(plan_abs_deg)
    plan_abs_rad = np.deg2rad(plan_abs_deg)

    target_xy = target_radius * np.stack(
        [
            np.cos(plan_abs_rad),
            np.sin(plan_abs_rad),
        ],
        axis=0,
    )

    ref_point = np.asarray([ref_x, ref_y], dtype=float)
    bias_vec = _bias_vector_from_mag_angle(bias_mag, bias_angle_deg)
    power_exp = float(power_exp)
    expand_shape = (2,) + (1,) * plan_abs_deg.ndim
    ref_broadcast = ref_point.reshape(expand_shape)
    bias_broadcast = bias_vec.reshape(expand_shape)

    start_bias = np.linalg.norm(ref_point) ** power_exp * bias_vec
    distance_to_ref = np.linalg.norm(target_xy - ref_broadcast, axis=0) ** power_exp
    endpoint_bias = bias_broadcast * distance_to_ref[None, ...]

    planned_endpoint = target_xy + endpoint_bias
    # Both endpoints are displaced; their difference determines movement direction.
    planned_movement = planned_endpoint - start_bias.reshape(expand_shape)
    exec_abs_deg = np.rad2deg(np.arctan2(planned_movement[1], planned_movement[0]))
    return wrap360(exec_abs_deg)


def transformed_relative_action_mean(exec_abs_deg, target_angles_deg, reference_rel_deg):
    """Choose the equivalent target-relative angle nearest reference_rel_deg."""
    exec_abs_deg = np.asarray(exec_abs_deg, dtype=float)
    target_angles_deg = np.asarray(target_angles_deg, dtype=float)
    reference_rel_deg = np.asarray(reference_rel_deg, dtype=float)
    wrapped_rel = wrap_angle(exec_abs_deg - target_angles_deg)
    return wrapped_rel + 360.0 * np.round((reference_rel_deg - wrapped_rel) / 360.0)
