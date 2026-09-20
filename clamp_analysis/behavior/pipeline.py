"""Build participant summaries used by the Chapter 4 figures and tests."""

import json

import pandas as pd

from clamp_analysis import paths

from .bias import build_bias_tables
from .summaries import (
    CONFIG,
    add_same_target_delta,
    make_phase_comparison_df,
    participant_window_summary,
    participant_window_variability,
    prepare_data,
    stl_consistency_check,
)


def save_table(frame, filename):
    paths.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(paths.TABLE_DIR / filename, index=False)


def run():
    paths.ensure_output_dirs()
    trials = prepare_data(paths.DATA_PATH, CONFIG.baseline_condition)
    delta = add_same_target_delta(trials, "hand_angle")
    late = participant_window_summary(trials, CONFIG.late_adaptation, "hand_angle")
    baseline = participant_window_summary(trials, CONFIG.baseline_reference, "hand_angle")
    save_table(late, "late_adaptation_subject_means.csv")
    save_table(baseline, "baseline_reference_subject_means.csv")

    for name, window in [("early", CONFIG.stl_early), ("washout", CONFIG.stl_washout)]:
        summary = participant_window_summary(delta, window, "same_target_delta").drop(
            columns="pointer", errors="ignore"
        )
        # Numeric participant order makes seeded bootstrap samples reproducible.
        summary = (
            summary.assign(_participant_order=pd.to_numeric(summary.ppid))
            .sort_values(["group", "clamp_magnitude", "_participant_order"])
            .drop(columns="_participant_order")
            .reset_index(drop=True)
        )
        save_table(summary, f"stl_{name}_subject_means.csv")

    late_variability = participant_window_variability(trials, CONFIG.late_adaptation, "hand_angle")
    save_table(late_variability, "late_variability_subject.csv")
    baseline_variability = (
        trials.loc[trials.condition.eq("FeedbackBaseline")]
        .groupby(["ppid", "clamp_magnitude"], as_index=False)
        .agg(base_sd=("hand_angle", "std"), n_baseline=("hand_angle", "count"))
        .merge(trials[["ppid", "pointer"]].drop_duplicates("ppid"), on="ppid", how="left")
    )
    save_table(baseline_variability, "baseline_variability_by_ppid_clamp.csv")

    for name, clamps in [
        ("peak", CONFIG.peak_distribution_clamps),
        ("high_error", CONFIG.high_error_clamps),
    ]:
        save_table(
            make_phase_comparison_df(baseline, late, clamps), f"{name}_phase_subject_means.csv"
        )
    build_bias_tables(trials, late, save_table)

    literature = pd.read_csv(
        paths.LITERATURE_DIR / "panelB_unified_summary_with_exp2_and_morehead_multiclamp.csv"
    )
    literature = literature.sort_values(["series_order", "perturbation_size_deg"]).reset_index(
        drop=True
    )
    literature = literature.loc[
        ~(
            (literature.series_label == "Morehead multi-clamp")
            & (literature.perturbation_size_deg >= 125)
        )
    ].reset_index(drop=True)
    save_table(literature, "panelB_literature_plus_exp2_summary.csv")

    check = stl_consistency_check(
        delta, CONFIG.baseline_condition, "hand_angle", "same_target_delta"
    )
    summary = {
        "participants": int(trials.ppid.nunique()),
        "rows": len(trials),
        "clamp_magnitudes": int(trials.clamp_magnitude.nunique()),
        "stl_max_absolute_error": float(check.error.abs().max()),
        "baseline_corrected": False,
    }
    (paths.TABLE_DIR / "cohort.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary
