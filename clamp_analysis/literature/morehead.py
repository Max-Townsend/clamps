from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from clamp_analysis import paths

SOURCE_DIR = paths.LITERATURE_DIR / "morehead_raw"

OUTPUT_DIR = paths.LITERATURE_DIR

RYAN_OUTPUT_DIR = OUTPUT_DIR / "ryan_morehead_multiclamp"

SERIES_ID = "morehead_multiclamp_local"

SERIES_LABEL = "Morehead multi-clamp"

SOURCE_PAPER = "Ryan Morehead multi-size clamp data"

SOURCE_EXPERIMENT = "Local multi-size clamp MATLAB files"

SOURCE_DATASET = "MultiClampRyanMorehead/*.mat"

SOURCE_NOTES = (
    "Recomputed from local MATLAB files using the exact late-learning logic in "
    "MultiSizeClamp_allsizes_plotting.m. Includes local multi-size clamp data and "
    "additional small-sample conditions not present in the published Morehead 2017 summary. "
    "For the 1.75/2.5/3.5/7.5 block, subject 1 is excluded entirely to avoid relying on the "
    "manual duplicated-condition repair used in the original MATLAB plotting script."
)


@dataclass(frozen=True)
class DatasetSpec:
    file_name: str
    variable_name: str
    cw_subjects: tuple[int, ...]
    ccw_subjects: tuple[int, ...]
    clamp_sizes_deg: tuple[float, ...]
    block_id: str
    is_first_block: bool = False


DATASET_SPECS = (
    DatasetSpec(
        file_name="data_MultiSizeClamp_1237.mat",
        variable_name="data_MultiSizeClamp_1237",
        cw_subjects=(2, 4, 6, 8, 10, 12, 14, 16, 18, 19, 20),
        ccw_subjects=(1, 3, 5, 7, 9, 11, 13, 15, 17),
        clamp_sizes_deg=(1.75, 2.5, 3.5, 7.5),
        block_id="1237",
        is_first_block=True,
    ),
    DatasetSpec(
        file_name="data_MultiSizeClamp_15304560.mat",
        variable_name="data_MultiSizeClamp_15304560",
        cw_subjects=(2, 4, 6, 8, 9, 11, 13, 15, 17, 19, 21),
        ccw_subjects=(1, 3, 5, 7, 10, 12, 14, 16, 18, 20),
        clamp_sizes_deg=(15.0, 30.0, 45.0, 60.0),
        block_id="15304560",
    ),
    DatasetSpec(
        file_name="data_MultiSizeClamp_7080100110.mat",
        variable_name="data_MultiSizeClamp_7080100110",
        cw_subjects=(2, 4, 6, 8, 10, 12, 14, 16, 18),
        ccw_subjects=(1, 3, 5, 7, 9, 11, 13, 15, 17),
        clamp_sizes_deg=(70.0, 80.0, 100.0, 110.0),
        block_id="7080100110",
    ),
    DatasetSpec(
        file_name="data_MultiSizeClamp_125135145165.mat",
        variable_name="data_MultiSizeClamp_125135145165",
        cw_subjects=(2, 4),
        ccw_subjects=(1, 3),
        clamp_sizes_deg=(125.0, 135.0, 145.0, 165.0),
        block_id="125135145165",
    ),
)


def load_mat_struct(path: Path, variable_name: str):
    return loadmat(path, squeeze_me=False, struct_as_record=False)[variable_name][0, 0]


def compute_aligned_trials(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    """Replicate MultiSizeClamp_allsizes_plotting.m for one block."""
    signflip = -1
    data = load_mat_struct(SOURCE_DIR / spec.file_name, spec.variable_name)

    n_subjects = len(spec.cw_subjects) + len(spec.ccw_subjects)
    n_trials = data.rpospeakth.shape[1]

    app = np.full((n_subjects, n_trials), np.nan, dtype=float)
    irot = np.full((n_subjects, n_trials), np.nan, dtype=float)

    cw_idx = np.asarray(spec.cw_subjects, dtype=int) - 1
    ccw_idx = np.asarray(spec.ccw_subjects, dtype=int) - 1

    app[cw_idx, :] = np.asarray(data.rpospeakth[cw_idx, :], dtype=float)
    app[ccw_idx, :] = np.asarray(data.rpospeakth[ccw_idx, :], dtype=float) * signflip
    irot[cw_idx, :] = np.asarray(data.rotation[cw_idx, :], dtype=float) * signflip
    irot[ccw_idx, :] = np.asarray(data.rotation[ccw_idx, :], dtype=float)
    target_angle = np.asarray(data.targetangle, dtype=float)

    app[(app > 90) | (app < -90)] = np.nan

    clamp_trials = slice(320, 1840)  # MATLAB 321:1840
    rotations = np.unique(irot[:, clamp_trials])
    rrindex = irot[:, 320:324]
    ttindex = target_angle[:, 320:324]

    target_picker = np.full((n_subjects, len(rotations)), np.nan, dtype=float)
    for subject_index in range(n_subjects):
        for rotation_index, rotation_value in enumerate(rotations):
            target_matches = np.where(rrindex[subject_index, :] == rotation_value)[0]
            if len(target_matches):
                target_picker[subject_index, rotation_index] = ttindex[
                    subject_index, target_matches[0]
                ]

    aligned = np.full((n_subjects, 540, len(rotations)), np.nan, dtype=float)
    for subject_index in range(n_subjects):
        for rotation_index in range(len(rotations)):
            target_matches = np.where(
                target_angle[subject_index, :] == target_picker[subject_index, rotation_index]
            )[0]
            aligned[subject_index, : len(target_matches), rotation_index] = app[
                subject_index, target_matches
            ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        baseline_mean = np.nanmean(aligned[:, 30:40, :], axis=1, keepdims=True)  # MATLAB 31:40
    aligned = aligned - baseline_mean

    if spec.is_first_block:
        # Subject 1 in the smallest-clamp block has a duplicated 2.5 deg assignment and no
        # valid 3.5 deg condition in the raw data. Exclude the participant entirely rather
        # than reproducing the manual repair used in the MATLAB plotting script.
        aligned[0, :, :] = np.nan

    if not np.allclose(rotations, np.asarray(spec.clamp_sizes_deg, dtype=float)):
        raise ValueError(
            f"Rotation mismatch for {spec.file_name}: {rotations} vs {spec.clamp_sizes_deg}"
        )

    return aligned, rotations


def make_trial_level_rows(
    spec: DatasetSpec, aligned: np.ndarray, clamp_sizes: np.ndarray
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n_subjects = aligned.shape[0]

    for subject_index in range(n_subjects):
        subject_id = f"msc_{spec.block_id}_s{subject_index + 1:02d}"
        for clamp_index, clamp_size in enumerate(clamp_sizes):
            for aligned_trial_num, hand_angle_deg in enumerate(
                aligned[subject_index, :, clamp_index], start=1
            ):
                if np.isnan(hand_angle_deg):
                    continue
                rows.append(
                    {
                        "series_id": SERIES_ID,
                        "series_label": SERIES_LABEL,
                        "source_paper": SOURCE_PAPER,
                        "source_year": "",
                        "source_doi": "",
                        "source_dataset": SOURCE_DATASET,
                        "source_dataset_doi": "",
                        "source_dataset_url": "",
                        "source_experiment": SOURCE_EXPERIMENT,
                        "group_id": spec.block_id,
                        "source_file": spec.file_name,
                        "subject_id": subject_id,
                        "perturbation_size_deg": float(clamp_size),
                        "phase": "aligned_clamp_sequence",
                        "aligned_trial_num": aligned_trial_num,
                        "hand_angle_deg": float(hand_angle_deg),
                    }
                )

    return pd.DataFrame(rows)


def make_subject_summary_rows(
    spec: DatasetSpec, aligned: np.ndarray, clamp_sizes: np.ndarray
) -> pd.DataFrame:
    late_window = aligned[:, 300:500, :]  # MATLAB 301:500
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        subject_means = np.nanmean(late_window, axis=1)

    rows: list[dict[str, object]] = []
    for subject_index in range(subject_means.shape[0]):
        subject_id = f"msc_{spec.block_id}_s{subject_index + 1:02d}"
        for clamp_index, clamp_size in enumerate(clamp_sizes):
            rows.append(
                {
                    "series_id": SERIES_ID,
                    "series_label": SERIES_LABEL,
                    "source_paper": SOURCE_PAPER,
                    "source_year": "",
                    "source_doi": "",
                    "source_experiment": SOURCE_EXPERIMENT,
                    "group_id": spec.block_id,
                    "source_file": spec.file_name,
                    "subject_id": subject_id,
                    "perturbation_size_deg": float(clamp_size),
                    "extent_window": (
                        "mean aligned trials 301-500 after baseline correction "
                        "using aligned trials 31-40"
                    ),
                    "subject_mean_extent_deg": float(subject_means[subject_index, clamp_index]),
                }
            )

    return pd.DataFrame(rows)


def make_panel_summary(subject_summary: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        subject_summary.groupby("perturbation_size_deg", as_index=False)
        .agg(
            adaptation_extent_deg=("subject_mean_extent_deg", "mean"),
            sem_deg=("subject_mean_extent_deg", lambda x: x.std(ddof=1) / np.sqrt(x.count())),
            n_subjects=("subject_mean_extent_deg", "count"),
        )
        .sort_values("perturbation_size_deg")
        .reset_index(drop=True)
    )

    grouped.insert(0, "series_order", 5)
    grouped.insert(1, "series_id", SERIES_ID)
    grouped.insert(2, "series_label", SERIES_LABEL)
    grouped.insert(3, "source_paper", SOURCE_PAPER)
    grouped.insert(4, "source_year", "")
    grouped.insert(5, "source_doi", "")
    grouped.insert(6, "source_dataset", SOURCE_DATASET)
    grouped.insert(7, "source_dataset_doi", "")
    grouped.insert(8, "source_dataset_url", "")
    grouped.insert(9, "source_experiment", SOURCE_EXPERIMENT)
    grouped.insert(10, "data_origin", "recomputed_from_local_matlab_files")
    grouped["extent_window"] = (
        "mean aligned trials 301-500 after baseline correction using aligned trials 31-40"
    )
    grouped["panel_target"] = "Panel-B-style perturbation-size summary"
    grouped["notes"] = SOURCE_NOTES
    return grouped


def write_provenance() -> None:
    provenance_text = "\n".join(
        [
            "Ryan Morehead multi-clamp local dataset provenance",
            "",
            f"Source directory: {SOURCE_DIR.relative_to(paths.PROJECT_ROOT).as_posix()}",
            "MATLAB script reproduced: MultiSizeClamp_allsizes_plotting.m",
            "",
            "Reproduction details:",
            "- signflip fixed to -1, matching the script's intended collapsing of CW/CCW onto positive clamp magnitudes.",
            "- Target-specific sequences aligned exactly as in the MATLAB plotting script.",
            "- Baseline correction uses aligned trials 31-40.",
            "- Late learning summary uses mean aligned trials 301-500 ('last 200 cycles' in the MATLAB comments).",
            "- Subject 1 from the first 1.75/2.5/3.5/7.5 block is excluded entirely because the raw block contains a duplicated 2.5 deg assignment and no valid 3.5 deg assignment for that participant.",
            "",
            "Outputs:",
            "- ryan_morehead_multiclamp_trial_level_standardized.csv",
            "- ryan_morehead_multiclamp_subject_mean_extent_late200.csv",
            "- ryan_morehead_multiclamp_panel_summary.csv",
            "- panelB_unified_summary_with_exp2_and_morehead_multiclamp.csv",
            "",
            f"Notes: {SOURCE_NOTES}",
        ]
    )
    (RYAN_OUTPUT_DIR / "README_provenance.txt").write_text(provenance_text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    RYAN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trial_frames = []
    summary_frames = []
    for spec in DATASET_SPECS:
        aligned, clamp_sizes = compute_aligned_trials(spec)
        trial_frames.append(make_trial_level_rows(spec, aligned, clamp_sizes))
        summary_frames.append(make_subject_summary_rows(spec, aligned, clamp_sizes))

    trial_level = pd.concat(trial_frames, ignore_index=True).sort_values(
        ["perturbation_size_deg", "subject_id", "aligned_trial_num"]
    )
    subject_summary = pd.concat(summary_frames, ignore_index=True).sort_values(
        ["perturbation_size_deg", "subject_id"]
    )
    panel_summary = make_panel_summary(subject_summary)

    trial_level_path = RYAN_OUTPUT_DIR / "ryan_morehead_multiclamp_trial_level_standardized.csv"
    subject_summary_path = (
        RYAN_OUTPUT_DIR / "ryan_morehead_multiclamp_subject_mean_extent_late200.csv"
    )
    panel_summary_path = RYAN_OUTPUT_DIR / "ryan_morehead_multiclamp_panel_summary.csv"

    trial_level.to_csv(trial_level_path, index=False)
    subject_summary.to_csv(subject_summary_path, index=False)
    panel_summary.to_csv(panel_summary_path, index=False)

    combined_source_path = OUTPUT_DIR / "panelB_unified_summary_with_exp2.csv"
    combined_output_path = (
        OUTPUT_DIR / "panelB_unified_summary_with_exp2_and_morehead_multiclamp.csv"
    )
    if combined_source_path.exists():
        combined = pd.concat([pd.read_csv(combined_source_path), panel_summary], ignore_index=True)
        combined = combined.sort_values(["series_order", "perturbation_size_deg"]).reset_index(
            drop=True
        )
        combined.to_csv(combined_output_path, index=False)

    write_provenance()

    print(f"Saved: {trial_level_path}")
    print(f"Saved: {subject_summary_path}")
    print(f"Saved: {panel_summary_path}")
    if combined_source_path.exists():
        print(f"Saved: {combined_output_path}")
