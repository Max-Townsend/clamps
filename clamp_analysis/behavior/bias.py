"""No-feedback spatial biases and their alignment with compensation."""

import numpy as np
import pandas as pd

BIAS_DIRECTION_ORDER = ["Incorrect", "Near-zero", "Correct"]


def build_bias_tables(trials, late_subject, save_table):
    BASELINE_BIAS_CONDITION = "NoFeedbackBaseline"
    baseline_trials = trials[trials["condition"].eq(BASELINE_BIAS_CONDITION)]

    BASELINE_BIAS_ABS_LIMIT = 40.0

    # Task alignment flips the hand angle so compensation is positive for either clamp sign.
    raw_signed_spatial_bias = (
        baseline_trials.dropna(subset=["target_angles_degrees"])
        .groupby(["ppid", "target_angles_degrees"], as_index=False, dropna=False)
        .agg(bias=("unflipped_hand_angle", "mean"))
    )
    ptr_map = trials[["ppid", "pointer"]].drop_duplicates(subset="ppid").dropna(subset=["pointer"])
    raw_signed_spatial_bias = raw_signed_spatial_bias.merge(ptr_map, on="ppid", how="left")
    raw_signed_spatial_bias = raw_signed_spatial_bias.loc[
        raw_signed_spatial_bias["bias"].abs().le(BASELINE_BIAS_ABS_LIMIT)
    ].copy()
    raw_signed_spatial_bias["target_angles_degrees"] = raw_signed_spatial_bias[
        "target_angles_degrees"
    ].astype(int)

    task_aligned_spatial_bias = (
        baseline_trials.dropna(subset=["target_angles_degrees"])
        .groupby(["ppid", "target_angles_degrees"], as_index=False, dropna=False)
        .agg(bias=("hand_angle", "mean"))
    )
    task_aligned_spatial_bias = task_aligned_spatial_bias.merge(ptr_map, on="ppid", how="left")
    task_aligned_spatial_bias = task_aligned_spatial_bias.loc[
        task_aligned_spatial_bias["bias"].abs().le(BASELINE_BIAS_ABS_LIMIT)
    ].copy()
    task_aligned_spatial_bias["target_angles_degrees"] = task_aligned_spatial_bias[
        "target_angles_degrees"
    ].astype(int)

    spatial_bias = raw_signed_spatial_bias

    baseline_bias = (
        baseline_trials.dropna(subset=["clamp_magnitude"])
        .groupby(["ppid", "clamp_magnitude"], as_index=False, dropna=False)
        .agg(baseline_bias=("hand_angle", "mean"))
    )
    baseline_bias["clamp_magnitude"] = baseline_bias["clamp_magnitude"].astype(int)
    baseline_bias = baseline_bias.merge(ptr_map, on="ppid", how="left")
    baseline_bias = baseline_bias.loc[
        baseline_bias["baseline_bias"].abs().le(BASELINE_BIAS_ABS_LIMIT)
    ].copy()

    raw_signed_baseline_bias = (
        baseline_trials.dropna(subset=["clamp_magnitude"])
        .groupby(["ppid", "clamp_magnitude"], as_index=False, dropna=False)
        .agg(
            raw_signed_baseline_bias=("unflipped_hand_angle", "mean"),
            signed_clamp_angle=("clamp_angle_relative", "mean"),
            target_clamp_sign=("target_clamp_sign", "first"),
        )
    )
    raw_signed_baseline_bias["clamp_magnitude"] = raw_signed_baseline_bias[
        "clamp_magnitude"
    ].astype(int)
    raw_signed_baseline_bias["target_clamp_sign"] = raw_signed_baseline_bias[
        "target_clamp_sign"
    ].astype(int)
    raw_signed_baseline_bias = raw_signed_baseline_bias.merge(ptr_map, on="ppid", how="left")
    raw_signed_baseline_bias = raw_signed_baseline_bias.loc[
        raw_signed_baseline_bias["raw_signed_baseline_bias"].abs().le(BASELINE_BIAS_ABS_LIMIT)
    ].copy()

    bias_late = baseline_bias.merge(
        late_subject[["ppid", "clamp_magnitude", "value"]].rename(columns={"value": "late_adapt"}),
        on=["ppid", "clamp_magnitude"],
        how="inner",
    ).dropna(subset=["baseline_bias", "late_adapt", "pointer"])

    save_table(spatial_bias, "baseline_bias_by_target.csv")
    save_table(raw_signed_spatial_bias, "baseline_bias_raw_signed_by_target.csv")
    save_table(task_aligned_spatial_bias, "baseline_bias_task_aligned_by_target.csv")
    save_table(baseline_bias, "baseline_bias_by_ppid_clamp.csv")
    save_table(raw_signed_baseline_bias, "baseline_bias_raw_signed_by_ppid_clamp.csv")
    save_table(bias_late, "baseline_bias_vs_late_adaptation.csv")

    BIAS_DIRECTION_THRESHOLD = 3.0
    bias_direction_pairings = bias_late.copy()
    bias_direction_pairings["bias_direction"] = np.select(
        [
            bias_direction_pairings["baseline_bias"] > BIAS_DIRECTION_THRESHOLD,
            bias_direction_pairings["baseline_bias"] < -BIAS_DIRECTION_THRESHOLD,
        ],
        ["Correct", "Incorrect"],
        default="Near-zero",
    )
    bias_direction_pairings["bias_direction"] = pd.Categorical(
        bias_direction_pairings["bias_direction"],
        categories=BIAS_DIRECTION_ORDER,
        ordered=True,
    )

    bias_direction_summary_records = []
    for (bias_direction, clamp_magnitude), g in bias_direction_pairings.groupby(
        ["bias_direction", "clamp_magnitude"],
        observed=True,
    ):
        late_vals = pd.Series(g["late_adapt"], dtype=float).dropna()
        row = {
            "n": int(len(late_vals)),
            "median": float(late_vals.median()) if len(late_vals) else np.nan,
            "q25": float(late_vals.quantile(0.25)) if len(late_vals) else np.nan,
            "q75": float(late_vals.quantile(0.75)) if len(late_vals) else np.nan,
        }
        row.update(
            {
                "bias_direction": bias_direction,
                "clamp_magnitude": int(clamp_magnitude),
            }
        )
        bias_direction_summary_records.append(row)

    bias_direction_summary_df = pd.DataFrame.from_records(bias_direction_summary_records)
    bias_direction_summary_df["bias_direction"] = pd.Categorical(
        bias_direction_summary_df["bias_direction"],
        categories=BIAS_DIRECTION_ORDER,
        ordered=True,
    )
    bias_direction_summary_df = bias_direction_summary_df.sort_values(
        ["bias_direction", "clamp_magnitude"]
    ).reset_index(drop=True)

    bias_direction_counts_df = (
        bias_direction_pairings.groupby(["bias_direction"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    bias_direction_counts_df["bias_direction"] = pd.Categorical(
        bias_direction_counts_df["bias_direction"],
        categories=BIAS_DIRECTION_ORDER,
        ordered=True,
    )
    bias_direction_counts_df = bias_direction_counts_df.sort_values(["bias_direction"]).reset_index(
        drop=True
    )

    save_table(bias_direction_pairings, "late_adaptation_by_signed_bias_direction_pairings.csv")
    save_table(bias_direction_summary_df, "late_adaptation_by_signed_bias_direction_summary.csv")
    save_table(bias_direction_counts_df, "late_adaptation_by_signed_bias_direction_counts.csv")
