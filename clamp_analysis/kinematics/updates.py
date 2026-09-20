from __future__ import annotations

import matplotlib

from clamp_analysis import paths

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CACHE_DIR = paths.KINEMATICS_DIR / "cache_full_cohort"

OUT_DIR = paths.KINEMATICS_DIR / "early_2d_learning"

UPDATE_CACHE = CACHE_DIR / "ballistic_first7_update_components.csv"

TRIAL_QC_PATH = CACHE_DIR / "ballistic_first7_trial_qc.csv"

EARLY_WINDOW = 7

SAMPLE_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]

TARGET_DISTANCE_UNITS = 4.0

POSITION_TO_TARGET_PERCENT = 100.0 / TARGET_DISTANCE_UNITS

MIN_BALLISTIC_DURATION = 0.04

MAX_BALLISTIC_DURATION = 0.75

MIN_BALLISTIC_ARC = 1.0

MAX_BALLISTIC_ARC = 50.0

CLAMP_BINS = [
    (2.0, 20.0, "2-20 deg", "#4C78A8"),
    (25.0, 70.0, "25-70 deg", "#F58518"),
    (75.0, 120.0, "75-120 deg", "#54A24B"),
    (125.0, 170.0, "125-170 deg", "#B279A2"),
]

plt.rcParams.update(
    {
        "font.size": 8.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "legend.title_fontsize": 8.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def ci95(values: pd.Series) -> float:
    valid = pd.Series(values, dtype=float).dropna()
    if len(valid) <= 1:
        return np.nan
    return 1.96 * float(valid.std(ddof=1)) / np.sqrt(len(valid))


def mark_qc_pass(qc: pd.DataFrame) -> pd.DataFrame:
    qc = qc.copy()
    qc["target_hit_within_bout"] = qc["target_hit_within_bout"].astype(bool)
    for col in ["ballistic_duration", "ballistic_arc_length"]:
        qc[col] = pd.to_numeric(qc[col], errors="coerce")
    qc["ballistic_qc_pass"] = (
        qc["target_hit_within_bout"]
        & qc["ballistic_duration"].between(MIN_BALLISTIC_DURATION, MAX_BALLISTIC_DURATION)
        & qc["ballistic_arc_length"].between(MIN_BALLISTIC_ARC, MAX_BALLISTIC_ARC)
    )
    return qc


def load_updates() -> pd.DataFrame:
    usecols = [
        "ppid",
        "trial_num",
        "target_angles_degrees",
        "target_exposure_index",
        "clamp_magnitude",
        "ballistic_fraction",
        "update_radial",
        "update_tangential",
    ]
    updates = pd.read_csv(UPDATE_CACHE, usecols=usecols)
    updates["ppid"] = updates["ppid"].astype(str)
    for col in usecols:
        if col != "ppid":
            updates[col] = pd.to_numeric(updates[col], errors="coerce")

    qc = pd.read_csv(
        TRIAL_QC_PATH,
        usecols=[
            "ppid",
            "trial_num",
            "target_hit_within_bout",
            "ballistic_duration",
            "ballistic_arc_length",
        ],
    )
    qc["ppid"] = qc["ppid"].astype(str)
    qc["trial_num"] = pd.to_numeric(qc["trial_num"], errors="coerce")
    keep = mark_qc_pass(qc).loc[lambda frame: frame["ballistic_qc_pass"], ["ppid", "trial_num"]]

    updates = updates.merge(keep, on=["ppid", "trial_num"], how="inner", validate="many_to_one")
    updates = updates.loc[
        updates["target_exposure_index"].between(1, EARLY_WINDOW)
        & updates["ballistic_fraction"].isin(SAMPLE_FRACTIONS)
        & updates["update_radial"].notna()
        & updates["update_tangential"].notna()
    ].copy()

    conditions = [(updates["clamp_magnitude"].between(low, high)) for low, high, _, _ in CLAMP_BINS]
    labels = [label for _, _, label, _ in CLAMP_BINS]
    updates["clamp_bin"] = np.select(conditions, labels, default="")
    updates = updates.loc[updates["clamp_bin"].ne("")].copy()
    return updates
