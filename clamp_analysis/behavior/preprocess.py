"""Rebuild formattedData.csv from raw trials and the original participant list.

The >=1,000-main-trial completeness screen precedes the thesis analyses.
No performance-based participant or trial exclusions are added here.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from clamp_analysis import paths

CLAMP_GROUPS = (
    (2, 4, 6, 8),
    (3, 5, 7, 10),
    (15, 25, 35, 45),
    (20, 30, 40, 50),
    (55, 65, 75, 85),
    (60, 70, 80, 90),
    (95, 105, 115, 125),
    (100, 110, 120, 130),
    (135, 145, 155, 165),
    (140, 150, 160, 170),
)

TRIAL_COLUMNS = [
    "type",
    "clamp_angle_relative",
    "clamped",
    "condition",
    "outcome",
    "ppid",
    "target_angles_degrees",
    "trial_num",
    "trial_num_in_block",
    "user_hand_angle",
    "start_time",
    "end_time",
    "start_moved_time",
    "target_hit_time",
    "target_shown_time",
]


def wrap_degrees(values):
    """Wrap angles to (-180, 180], assigning the boundary to +180 degrees."""
    wrapped = (values + 180) % 360 - 180
    return wrapped.where(wrapped != -180, 180)


def prepare_trials(trials, participant_ids):
    trials = trials.loc[trials["type"].eq("main")].copy()
    trials["ppid"] = trials.ppid.astype(str)
    trials = trials.loc[trials.ppid.isin(set(map(str, participant_ids)))].copy()
    counts = trials.ppid.value_counts()
    trials = trials.loc[trials.ppid.isin(counts[counts >= 1000].index)].copy()
    trials["recentred_hand_angle"] = wrap_degrees(
        trials.user_hand_angle - trials.target_angles_degrees
    )
    trials["cycle_num"] = (trials.trial_num - 1) // 4
    trials["circular_target_angle"] = wrap_degrees(trials.target_angles_degrees)
    trials["unflipped_hand_angle"] = trials.recentred_hand_angle

    # Baseline and washout inherit the clamp assigned to the same participant and target.
    keys = ["ppid", "target_angles_degrees"]
    clamp_lookup = (
        trials.loc[trials.condition.eq("Clamp")]
        .drop_duplicates(keys)
        .set_index(keys)
        .clamp_angle_relative
    )
    needs_clamp = trials.condition.str.contains("Washout|Baseline", na=False)
    lookup_keys = pd.MultiIndex.from_frame(trials.loc[needs_clamp, keys])
    angles = clamp_lookup.reindex(lookup_keys).to_numpy()
    if pd.isna(angles).any():
        raise ValueError("A baseline/washout target has no corresponding clamp trial.")
    trials.loc[needs_clamp, "clamp_angle_relative"] = angles
    trials.loc[trials.clamp_angle_relative > 0, "recentred_hand_angle"] *= -1
    trials["target_clamp_sign"] = np.sign(trials.clamp_angle_relative).astype(int)
    trials["clamp_magnitude"] = pd.to_numeric(trials.clamp_angle_relative.abs()).astype("Int64")
    trials["clamp_mag_float"] = trials.clamp_angle_relative.abs()
    group_lookup = {
        clamp: group for group, clamps in enumerate(CLAMP_GROUPS, start=1) for clamp in clamps
    }
    trials["group"] = trials.clamp_magnitude.map(group_lookup)
    if trials.group.isna().any():
        raise ValueError("Unexpected clamp magnitude; check the experimental design.")
    return trials


def run(output_path=None):
    output_path = paths.DATA_PATH if output_path is None else Path(output_path)
    raw = pd.read_csv(paths.RAW_DIR / "trialsV2.csv", usecols=TRIAL_COLUMNS, low_memory=False)
    participant_ids = pd.read_csv(paths.RAW_DIR / "ppidsV2.csv").ppid
    formatted = prepare_trials(raw, participant_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    formatted.to_csv(output_path)  # Keep raw-trial row IDs for traceability.
    print(
        f"Prepared {len(formatted):,} trials from {formatted.ppid.nunique()} participants: {output_path}"
    )
    return formatted
