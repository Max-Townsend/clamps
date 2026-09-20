from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from clamp_analysis.models.transformation_bias import (
    transformation_policy_mean,
    transformed_relative_action_mean,
)


def fit_transformation_bias_patterns(summary_df):
    fit_summary_records = []
    fit_point_records = []

    for pointer, ps in summary_df.groupby("pointer", observed=True):
        ps = ps.sort_values("target_angles_degrees")
        targets = ps["target_angles_degrees"].to_numpy(dtype=float)
        obs = ps["mean"].to_numpy(dtype=float)
        bounds = [
            (-15.0, 15.0),
            (-15.0, 15.0),
            (0.0, float(np.sqrt(2.0) * 0.20)),
            (-180.0, 180.0),
            (0.5, 2.0),
        ]

        def objective(x):
            exec_abs = transformation_policy_mean(targets, x[0], x[1], x[2], x[3], x[4])
            pred = transformed_relative_action_mean(exec_abs, targets, np.zeros_like(targets))
            pred = ((pred + 180.0) % 360.0) - 180.0
            return float(np.sum((pred - obs) ** 2))

        fit = differential_evolution(objective, bounds=bounds, seed=123, maxiter=300, polish=True)
        x = fit.x
        exec_abs = transformation_policy_mean(targets, x[0], x[1], x[2], x[3], x[4])
        pred = transformed_relative_action_mean(exec_abs, targets, np.zeros_like(targets))
        pred = ((pred + 180.0) % 360.0) - 180.0

        fit_summary_records.append(
            {
                "pointer": pointer,
                "trRefX": float(x[0]),
                "trRefY": float(x[1]),
                "trBiasMag": float(x[2]),
                "trBiasAngle": float(x[3]),
                "trPowerExp": float(x[4]),
                "rmse_to_mean_pattern": float(np.sqrt(np.mean((pred - obs) ** 2))),
            }
        )
        for target, human_mean, model_pred in zip(targets, obs, pred):
            fit_point_records.append(
                {
                    "pointer": pointer,
                    "target_angles_degrees": int(target),
                    "human_mean_bias": float(human_mean),
                    "model_pred_bias": float(model_pred),
                }
            )

    fit_summary_df = (
        pd.DataFrame.from_records(fit_summary_records).sort_values("pointer").reset_index(drop=True)
    )
    fit_point_df = (
        pd.DataFrame.from_records(fit_point_records)
        .sort_values(["pointer", "target_angles_degrees"])
        .reset_index(drop=True)
    )
    return fit_summary_df, fit_point_df
