"""Thesis analysis windows, target-specific updates, and participant summaries."""

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

from clamp_analysis import paths


@dataclass(frozen=True)
class WindowSpec:
    condition: str
    trial_range: Optional[Tuple[int, int]] = None
    cycle_range: Optional[Tuple[int, int]] = None


@dataclass(frozen=True)
class AnalysisConfig:
    baseline_condition: str = "FeedbackBaseline"
    baseline_reference: WindowSpec = WindowSpec("FeedbackBaseline", trial_range=(21, 40))
    late_adaptation: WindowSpec = WindowSpec("Clamp", trial_range=(881, 960))
    stl_early: WindowSpec = WindowSpec("Clamp", cycle_range=(0, 9))
    stl_washout: WindowSpec = WindowSpec("FeedbackWashout", cycle_range=(240, 249))
    high_error_clamps: Tuple[int, ...] = (135, 140, 145, 150, 155, 160, 165, 170)
    peak_distribution_clamps: Tuple[int, ...] = (15, 20, 25, 30, 35, 40, 45, 50)


CONFIG = AnalysisConfig()


def window_mask(df, window):
    mask = df["condition"].eq(window.condition)

    if window.trial_range is not None:
        start, end = window.trial_range
        mask &= df["trial_num_in_block"].between(start, end)

    if window.cycle_range is not None:
        start, end = window.cycle_range
        mask &= df["cycle_wrt_clamp"].between(start, end)

    return mask


def prepare_data(path, baseline_condition):
    df = pd.read_csv(path)
    df["clamp_magnitude"] = pd.to_numeric(df["clamp_magnitude"], errors="coerce")
    df["group"] = pd.to_numeric(df["group"], errors="coerce")
    df["cycle_wrt_clamp"] = df["cycle_num"] - 21
    df = df.sort_values(["ppid", "trial_num"]).reset_index(drop=True)

    df["hand_angle"] = df["recentred_hand_angle"]

    details_path = paths.DETAILS_PATH
    if details_path.exists():
        details = pd.read_csv(details_path, usecols=["ppid_session_dataname", "pointer"])
        details["ppid"] = details["ppid_session_dataname"].str.split("_").str[0]
        details = details[["ppid", "pointer"]].drop_duplicates(subset="ppid")
        df["ppid"] = df["ppid"].astype(str)
        details["ppid"] = details["ppid"].astype(str)
        df = df.merge(details, on="ppid", how="left")

    return df


def add_same_target_delta(df, value_col):
    """Assign each same-target change to its later trial, including phase transitions."""
    delta_df = df.sort_values(["ppid", "target_angles_degrees", "trial_num"]).copy()
    delta_df["same_target_delta"] = delta_df.groupby(["ppid", "target_angles_degrees"])[
        value_col
    ].diff()
    return delta_df


def stl_consistency_check(df, baseline_condition, value_col, delta_col):
    clamp_df = df[df["condition"].eq("Clamp")].copy()
    summed_delta = (
        clamp_df.groupby(["ppid", "clamp_magnitude", "target_angles_degrees"], as_index=False)[
            delta_col
        ]
        .sum(min_count=1)
        .rename(columns={delta_col: "summed_stl"})
    )

    baseline_last = (
        df[df["condition"].eq(baseline_condition)]
        .groupby(["ppid", "target_angles_degrees"], as_index=False)[value_col]
        .last()
        .rename(columns={value_col: "baseline_last"})
    )

    clamp_last = (
        clamp_df.groupby(["ppid", "clamp_magnitude", "target_angles_degrees"], as_index=False)[
            value_col
        ]
        .last()
        .rename(columns={value_col: "clamp_last"})
    )

    check = summed_delta.merge(baseline_last, on=["ppid", "target_angles_degrees"], how="inner")
    check = check.merge(
        clamp_last, on=["ppid", "clamp_magnitude", "target_angles_degrees"], how="inner"
    )
    check["expected_total_change"] = check["clamp_last"] - check["baseline_last"]
    check["error"] = check["summed_stl"] - check["expected_total_change"]
    return check


def participant_window_summary(df, window, value_col):
    has_pointer = "pointer" in df.columns
    group_keys = ["ppid", "clamp_magnitude", "group"]
    columns = group_keys + [value_col]
    if has_pointer:
        columns.append("pointer")
    summary = (
        df.loc[window_mask(df, window), columns]
        .dropna(subset=["clamp_magnitude", "group"])
        .groupby(group_keys + (["pointer"] if has_pointer else []), as_index=False, dropna=False)[
            value_col
        ]
        .mean()
        .rename(columns={value_col: "value"})
        .dropna(subset=["value"])
        .sort_values(["group", "clamp_magnitude", "ppid"])
        .reset_index(drop=True)
    )
    summary["group"] = summary["group"].astype(int)
    summary["clamp_magnitude"] = summary["clamp_magnitude"].astype(int)
    return summary


def make_phase_comparison_df(baseline_subject, late_subject, clamp_values):
    baseline_df = baseline_subject[baseline_subject["clamp_magnitude"].isin(clamp_values)].copy()
    baseline_df["phase"] = "Baseline"

    late_df = late_subject[late_subject["clamp_magnitude"].isin(clamp_values)].copy()
    late_df["phase"] = "Late adaptation"

    return (
        pd.concat([baseline_df, late_df], ignore_index=True)
        .sort_values(["clamp_magnitude", "phase", "ppid"])
        .reset_index(drop=True)
    )


def participant_window_variability(df, window, value_col, min_n=3):
    """Sample SD per participant and clamp, excluding groups with fewer than min_n trials."""
    has_pointer = "pointer" in df.columns
    group_keys = ["ppid", "clamp_magnitude", "group"]
    columns = group_keys + [value_col]
    if has_pointer:
        columns.append("pointer")

    raw = df.loc[window_mask(df, window), columns].dropna(subset=["clamp_magnitude", "group"])
    groupers = group_keys + (["pointer"] if has_pointer else [])
    agg = (
        raw.groupby(groupers, as_index=False, dropna=False)[value_col]
        .agg(["std", "count"])
        .reset_index()
        .rename(columns={"std": "value", "count": "n_trials"})
    )
    agg = agg.loc[agg["n_trials"] >= min_n].dropna(subset=["value"])
    agg["group"] = agg["group"].astype(int)
    agg["clamp_magnitude"] = agg["clamp_magnitude"].astype(int)
    return agg.sort_values(["group", "clamp_magnitude", "ppid"]).reset_index(drop=True)
