from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from clamp_analysis import paths
from clamp_analysis.kinematics.geometry import CLAMPS, rotate_to_target

TRACKERS_CSV = paths.RAW_DIR / "trackers.csv"
TRIALS_CSV = paths.DATA_PATH
CACHE_DIR = paths.KINEMATICS_DIR / "cache_full_cohort"
WINDOW = 7  # Seven updates connect the first eight target exposures.

BALLISTIC_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]

TRACKER_CHUNK_SIZE = 2048

MIN_BALLISTIC_DURATION = 0.04

MAX_BALLISTIC_DURATION = 0.75

MIN_BALLISTIC_ARC = 1.0

MAX_BALLISTIC_ARC = 50.0

CACHE_PATH = CACHE_DIR / "ballistic_first7_update_components.csv"

TRIAL_QC_PATH = CACHE_DIR / "ballistic_first7_trial_qc.csv"


def load_metadata() -> pd.DataFrame:
    usecols = [
        "type",
        "condition",
        "ppid",
        "trial_num",
        "trial_num_in_block",
        "target_angles_degrees",
        "clamp_angle_relative",
        "clamp_magnitude",
        "start_moved_time",
        "target_hit_time",
    ]
    trials = pd.read_csv(TRIALS_CSV, usecols=usecols)
    trials["ppid"] = trials["ppid"].astype(str)
    trials = trials.loc[
        trials["type"].eq("main")
        & trials["condition"].eq("Clamp")
        & trials["clamp_angle_relative"].notna()
        & trials["start_moved_time"].notna()
        & trials["target_hit_time"].notna()
    ].copy()
    trials["clamp_magnitude"] = pd.to_numeric(trials["clamp_magnitude"], errors="coerce").fillna(
        pd.to_numeric(trials["clamp_angle_relative"], errors="coerce").abs()
    )
    trials = trials.loc[trials["clamp_magnitude"].isin(CLAMPS)].copy()
    trials["align_sign"] = np.where(
        pd.to_numeric(trials["clamp_angle_relative"], errors="coerce") > 0, -1.0, 1.0
    )
    trials = trials.sort_values(
        ["ppid", "target_angles_degrees", "trial_num_in_block", "trial_num"]
    )
    trials["target_exposure_index"] = (
        trials.groupby(["ppid", "target_angles_degrees"]).cumcount() + 1
    )
    trials = trials.loc[trials["target_exposure_index"].le(WINDOW + 1)].copy()
    return trials[
        [
            "ppid",
            "trial_num",
            "target_angles_degrees",
            "align_sign",
            "target_exposure_index",
            "clamp_magnitude",
            "start_moved_time",
            "target_hit_time",
        ]
    ].copy()


def unique_time_arrays(
    time: np.ndarray,
    moving: np.ndarray,
    x_user: np.ndarray,
    y_user: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isfinite(time) & np.isfinite(x_user) & np.isfinite(y_user)
    time = time[mask]
    moving = moving[mask]
    x_user = x_user[mask]
    y_user = y_user[mask]
    order = np.argsort(time)
    time = time[order]
    moving = moving[order]
    x_user = x_user[order]
    y_user = y_user[order]
    unique_time, unique_idx = np.unique(time, return_index=True)
    return unique_time, moving[unique_idx], x_user[unique_idx], y_user[unique_idx]


def first_movement_bout_end(
    time: np.ndarray, moving: np.ndarray, start_time: float
) -> float | None:
    after = np.flatnonzero((time >= start_time) & moving)
    if after.size == 0:
        return None
    end_idx = int(after[0])
    while end_idx + 1 < len(time) and bool(moving[end_idx + 1]):
        end_idx += 1
    end_time = float(time[end_idx])
    if not np.isfinite(end_time) or end_time <= start_time:
        return None
    return end_time


def ballistic_samples_for_trial(
    row,
) -> tuple[list[dict[str, float | str]], dict[str, float | str] | None]:
    time = np.asarray(json.loads(row.time), dtype=float)
    moving = np.asarray(json.loads(row.moving), dtype=bool)
    x_user = np.asarray(json.loads(row.pos_x_user), dtype=float)
    y_user = np.asarray(json.loads(row.pos_y_user), dtype=float)
    if not (len(time) == len(moving) == len(x_user) == len(y_user) and len(time) >= 2):
        return [], None

    time, moving, x_user, y_user = unique_time_arrays(time, moving, x_user, y_user)
    if len(time) < 2:
        return [], None

    start_time = float(row.start_moved_time)
    hit_time = float(row.target_hit_time)
    end_time = first_movement_bout_end(time, moving, start_time)
    if end_time is None:
        return [], None
    if start_time < time[0] or end_time > time[-1]:
        return [], None

    interval_mask = (time >= start_time) & (time <= end_time)
    segment_time = np.concatenate([[start_time], time[interval_mask], [end_time]])
    segment_x = np.interp(segment_time, time, x_user)
    segment_y = np.interp(segment_time, time, y_user)
    segment_time, unique_idx = np.unique(segment_time, return_index=True)
    segment_x = segment_x[unique_idx]
    segment_y = segment_y[unique_idx]
    if len(segment_time) < 2:
        return [], None

    x_rot, y_rot = rotate_to_target(segment_x, segment_y, float(row.target_angles_degrees))
    y_rot = y_rot * float(row.align_sign)
    steps = np.sqrt(np.diff(x_rot) ** 2 + np.diff(y_rot) ** 2)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    total_arc = float(arc[-1])
    if not np.isfinite(total_arc) or total_arc <= 1e-6:
        return [], None
    unique_arc, arc_idx = np.unique(arc, return_index=True)
    if len(unique_arc) < 2:
        return [], None
    x_by_arc = x_rot[arc_idx]
    y_by_arc = y_rot[arc_idx]

    hit_time_fraction = (hit_time - start_time) / (end_time - start_time)
    if hit_time <= end_time:
        hit_arc = float(np.interp(hit_time, segment_time, arc))
        hit_arc_fraction = hit_arc / total_arc
    else:
        hit_arc_fraction = np.nan

    qc = {
        "ppid": row.ppid,
        "trial_num": int(row.trial_num),
        "target_angles_degrees": float(row.target_angles_degrees),
        "target_exposure_index": int(row.target_exposure_index),
        "clamp_magnitude": float(row.clamp_magnitude),
        "ballistic_duration": end_time - start_time,
        "target_hit_time_fraction_of_bout": hit_time_fraction,
        "target_hit_arc_fraction_of_bout": hit_arc_fraction,
        "target_hit_within_bout": bool(hit_time <= end_time),
        "ballistic_arc_length": total_arc,
    }

    rows: list[dict[str, float | str]] = []
    # Fractions measure distance along the bout, not elapsed time.
    for frac in BALLISTIC_FRACTIONS:
        target_arc = float(frac) * total_arc
        rows.append(
            {
                "ppid": row.ppid,
                "trial_num": int(row.trial_num),
                "target_angles_degrees": float(row.target_angles_degrees),
                "target_exposure_index": int(row.target_exposure_index),
                "clamp_magnitude": float(row.clamp_magnitude),
                "ballistic_fraction": float(frac),
                "x_hand": float(np.interp(target_arc, unique_arc, x_by_arc)),
                "y_hand": float(np.interp(target_arc, unique_arc, y_by_arc)),
            }
        )
    return rows, qc


def build_ballistic_cache() -> tuple[pd.DataFrame, pd.DataFrame]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_metadata()
    eligible_ppids = set(meta["ppid"].unique())
    tracker_cols = [
        "ppid_session_dataname",
        "trial_num",
        "time",
        "moving",
        "pos_x_user",
        "pos_y_user",
    ]
    sample_rows: list[dict[str, float | str]] = []
    qc_rows: list[dict[str, float | str]] = []
    matched_trials = 0

    for chunk_idx, chunk in enumerate(
        pd.read_csv(TRACKERS_CSV, usecols=tracker_cols, chunksize=TRACKER_CHUNK_SIZE), start=1
    ):
        chunk["ppid"] = chunk["ppid_session_dataname"].astype(str).str.split("_").str[0]
        chunk = chunk.loc[chunk["ppid"].isin(eligible_ppids)].copy()
        if chunk.empty:
            continue
        matched = chunk.merge(meta, on=["ppid", "trial_num"], how="inner")
        if matched.empty:
            continue
        matched_trials += len(matched)

        for row in matched.itertuples(index=False):
            rows, qc = ballistic_samples_for_trial(row)
            sample_rows.extend(rows)
            if qc is not None:
                qc_rows.append(qc)

        if chunk_idx % 25 == 0:
            print(
                f"Processed {chunk_idx} chunks | {matched_trials:,} matched trials | "
                f"{len(sample_rows):,} ballistic sample rows"
            )

    samples = pd.DataFrame.from_records(sample_rows)
    qc = pd.DataFrame.from_records(qc_rows)
    if samples.empty:
        raise RuntimeError("No ballistic samples were extracted.")
    samples = samples.sort_values(
        ["ppid", "target_angles_degrees", "ballistic_fraction", "target_exposure_index"]
    ).reset_index(drop=True)
    grouped = samples.groupby(["ppid", "target_angles_degrees", "ballistic_fraction"], sort=False)
    samples["next_target_exposure_index"] = grouped["target_exposure_index"].shift(-1)
    samples["next_x_hand"] = grouped["x_hand"].shift(-1)
    samples["next_y_hand"] = grouped["y_hand"].shift(-1)
    # Missing bouts must not turn a multi-exposure gap into a single update.
    consecutive = samples["next_target_exposure_index"].eq(samples["target_exposure_index"] + 1)
    samples["update_radial"] = np.where(
        consecutive, samples["next_x_hand"] - samples["x_hand"], np.nan
    )
    samples["update_tangential"] = np.where(
        consecutive, samples["next_y_hand"] - samples["y_hand"], np.nan
    )
    samples.loc[
        samples["target_exposure_index"].gt(WINDOW), ["update_radial", "update_tangential"]
    ] = np.nan
    samples.to_csv(CACHE_PATH, index=False)
    qc.to_csv(TRIAL_QC_PATH, index=False)
    return samples, qc


def mark_qc_pass(qc: pd.DataFrame) -> pd.DataFrame:
    qc = qc.copy()
    qc["target_hit_within_bout"] = qc["target_hit_within_bout"].astype(bool)
    qc["ballistic_duration"] = pd.to_numeric(qc["ballistic_duration"], errors="coerce")
    qc["ballistic_arc_length"] = pd.to_numeric(qc["ballistic_arc_length"], errors="coerce")
    qc["ballistic_qc_pass"] = (
        qc["target_hit_within_bout"]
        & qc["ballistic_duration"].between(MIN_BALLISTIC_DURATION, MAX_BALLISTIC_DURATION)
        & qc["ballistic_arc_length"].between(MIN_BALLISTIC_ARC, MAX_BALLISTIC_ARC)
    )
    return qc


def load_ballistic_updates(force_rebuild: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not force_rebuild and CACHE_PATH.exists() and TRIAL_QC_PATH.exists():
        samples = pd.read_csv(CACHE_PATH)
        qc = pd.read_csv(TRIAL_QC_PATH)
    else:
        samples, qc = build_ballistic_cache()
    samples["ppid"] = samples["ppid"].astype(str)
    qc["ppid"] = qc["ppid"].astype(str)
    qc = mark_qc_pass(qc)
    keep = qc.loc[qc["ballistic_qc_pass"], ["ppid", "trial_num"]].copy()
    samples = samples.merge(keep, on=["ppid", "trial_num"], how="inner", validate="many_to_one")
    numeric_cols = [
        "target_exposure_index",
        "clamp_magnitude",
        "ballistic_fraction",
        "update_radial",
        "update_tangential",
    ]
    for col in numeric_cols:
        samples[col] = pd.to_numeric(samples[col], errors="coerce")
    return samples.loc[
        samples["ballistic_fraction"].isin(BALLISTIC_FRACTIONS)
        & samples["target_exposure_index"].between(1, WINDOW)
        & samples["clamp_magnitude"].isin(CLAMPS)
        & samples["update_radial"].notna()
        & samples["update_tangential"].notna()
    ].copy(), qc
