"""Numbered main figures: same data, windows, plots and fitted predictions as the thesis."""

import matplotlib.pyplot as plt
import pandas as pd

from .panels import (
    COMBINED_MODEL_DATASET_SLUG,
    DEVICE_EDGE_COLORS,
    LATE_ADAPTATION_CYCLE_RANGE,
    MAGENTA_DARK,
    MEAN_TIMESERIES_MODEL_COLORS,
    MEAN_TIMESERIES_MODEL_LABELS,
    MEAN_TIMESERIES_MODEL_ORDER,
    MODEL_DASH,
    PAPER_FONT,
    POINTER_ORDER,
    REPRESENTATIVE_CLAMPS,
    STL_EARLY_CYCLE_RANGE,
    STL_WASHOUT_CYCLE_RANGE,
    TABLE_DIR,
    TEAL_DARK,
    _load_subject_table,
    add_panel_label,
    build_early_stl_variability_subject,
    build_filtered_spatial_bias_tables,
    compute_fit_metrics,
    format_fit_metrics_line,
    load_mean_timeseries_prediction_df,
    padded_ylim,
    plot_bias_direction_summary,
    plot_first_bout_radial_bridge_panels,
    plot_high_error_bias_sign_trajectories,
    plot_literature_panel,
    plot_phase_distribution,
    plot_pointer_summary,
    plot_pooled_summary,
    plot_representative_learning_curves,
    plot_spatial_bias_with_tr_markers,
    summarize_by_clamp,
    summarize_by_pointer,
    summarize_model_fit_window,
)


def build_chapter_figures() -> dict[str, plt.Figure]:
    plt.rcParams.update(
        {
            "font.size": PAPER_FONT["tick"],
            "axes.titlesize": PAPER_FONT["title"],
            "axes.labelsize": PAPER_FONT["axis_label"],
            "xtick.labelsize": PAPER_FONT["tick"],
            "ytick.labelsize": PAPER_FONT["tick"],
            "legend.fontsize": PAPER_FONT["legend"],
        }
    )

    fig41, ax_literature = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    fig42, axes42 = plt.subplots(3, 2, figsize=(17, 18), constrained_layout=True)
    ax_extent, ax_curves, ax_early_stl, ax_washout_stl, ax_2d_updates, ax_radial_updates = (
        axes42.ravel()
    )
    fig43, axes43 = plt.subplots(2, 2, figsize=(17, 12), constrained_layout=True)
    ax_peak_distribution, ax_large_distribution, ax_bias_groups, ax_bias_curves = axes43.ravel()
    fig44, axes44 = plt.subplots(2, 2, figsize=(17, 12), constrained_layout=True)
    ax_device_extent, ax_early_variability, ax_late_variability, ax_spatial_bias = axes44.ravel()

    literature_df = pd.read_csv(TABLE_DIR / "panelB_literature_plus_exp2_summary.csv")
    late_subject = _load_subject_table("late_adaptation_subject_means.csv")
    late_summary = summarize_by_clamp(late_subject, stat="mean")
    late_pointer_summary = summarize_by_pointer(late_subject, stat="mean")
    pooled_model_late = {
        model_name: summarize_model_fit_window(
            model_name,
            COMBINED_MODEL_DATASET_SLUG,
            condition="Clamp",
            cycle_range=LATE_ADAPTATION_CYCLE_RANGE,
        )
        for model_name in MEAN_TIMESERIES_MODEL_ORDER
    }
    device_vector_late = {
        "Mouse": summarize_model_fit_window(
            "MeanTimeseriesVectorBCC",
            "mouse",
            condition="Clamp",
            cycle_range=LATE_ADAPTATION_CYCLE_RANGE,
        ),
        "Trackpad": summarize_model_fit_window(
            "MeanTimeseriesVectorBCC",
            "trackpad",
            condition="Clamp",
            cycle_range=LATE_ADAPTATION_CYCLE_RANGE,
        ),
    }
    pooled_vector_stl_early = summarize_model_fit_window(
        "MeanTimeseriesVectorBCC",
        COMBINED_MODEL_DATASET_SLUG,
        condition="Clamp",
        cycle_range=STL_EARLY_CYCLE_RANGE,
        use_same_target_delta=True,
    )
    pooled_vector_stl_washout = summarize_model_fit_window(
        "MeanTimeseriesVectorBCC",
        COMBINED_MODEL_DATASET_SLUG,
        condition="FeedbackWashout",
        cycle_range=STL_WASHOUT_CYCLE_RANGE,
        use_same_target_delta=True,
    )
    pooled_vector_rep_curves = load_mean_timeseries_prediction_df(
        COMBINED_MODEL_DATASET_SLUG, "MeanTimeseriesVectorBCC"
    )
    pooled_vector_rep_curves = pooled_vector_rep_curves.loc[
        pooled_vector_rep_curves["clamp_magnitude"].isin(REPRESENTATIVE_CLAMPS)
        & pooled_vector_rep_curves["condition"].isin(
            ["FeedbackBaseline", "Clamp", "FeedbackWashout"]
        )
    ].copy()
    late_metrics_lines = [
        format_fit_metrics_line(
            MEAN_TIMESERIES_MODEL_LABELS[model_name],
            compute_fit_metrics(late_summary, pooled_model_late[model_name]),
        )
        for model_name in MEAN_TIMESERIES_MODEL_ORDER
    ]
    device_metrics_lines = [
        format_fit_metrics_line(
            pointer,
            compute_fit_metrics(
                late_pointer_summary.loc[late_pointer_summary["pointer"].eq(pointer)].copy(),
                device_vector_late[pointer],
            ),
        )
        for pointer in POINTER_ORDER
    ]
    top_row_ylim = padded_ylim(
        late_summary["ci_low"],
        late_summary["ci_high"],
        *(summary_df["predicted_value"] for summary_df in pooled_model_late.values()),
        include_zero=True,
        min_pad=2.0,
    )
    late_device_ylim = padded_ylim(
        late_pointer_summary["ci_low"],
        late_pointer_summary["ci_high"],
        device_vector_late["Mouse"]["predicted_value"],
        device_vector_late["Trackpad"]["predicted_value"],
        include_zero=True,
        min_pad=2.0,
    )

    early_stl_subject = _load_subject_table("stl_early_subject_means.csv")
    washout_stl_subject = _load_subject_table("stl_washout_subject_means.csv")
    early_stl_summary = summarize_by_clamp(early_stl_subject, stat="mean")
    washout_stl_summary = summarize_by_clamp(washout_stl_subject, stat="mean")
    stl_ylim = padded_ylim(
        early_stl_summary["ci_low"],
        early_stl_summary["ci_high"],
        washout_stl_summary["ci_low"],
        washout_stl_summary["ci_high"],
        pooled_vector_stl_early["predicted_value"],
        pooled_vector_stl_washout["predicted_value"],
        include_zero=True,
        min_pad=0.75,
    )
    early_metrics_lines = [
        format_fit_metrics_line(
            "Vector BCC",
            compute_fit_metrics(early_stl_summary, pooled_vector_stl_early),
        )
    ]
    washout_metrics_lines = [
        format_fit_metrics_line(
            "Vector BCC",
            compute_fit_metrics(washout_stl_summary, pooled_vector_stl_washout),
        )
    ]

    peak_subject = _load_subject_table("peak_phase_subject_means.csv")
    high_error_subject = _load_subject_table("high_error_phase_subject_means.csv")
    distribution_ylim = padded_ylim(
        peak_subject["value"],
        high_error_subject["value"],
        include_zero=True,
        min_pad=3.0,
    )

    bias_direction_summary = pd.read_csv(
        TABLE_DIR / "late_adaptation_by_signed_bias_direction_summary.csv"
    )
    bias_direction_counts = pd.read_csv(
        TABLE_DIR / "late_adaptation_by_signed_bias_direction_counts.csv"
    )
    bias_direction_ylim = padded_ylim(
        bias_direction_summary["q25"],
        bias_direction_summary["q75"],
        bias_direction_summary["median"],
        include_zero=True,
        min_pad=1.5,
    )

    spatial_raw, spatial_summary, spatial_fit_points = build_filtered_spatial_bias_tables()
    spatial_bias_ylim = padded_ylim(
        spatial_raw["bias"],
        spatial_summary["ci_low"],
        spatial_summary["ci_high"],
        spatial_fit_points["model_pred_bias"],
        include_zero=True,
        min_pad=1.5,
    )

    plot_literature_panel(ax_literature, df=literature_df, ylim=top_row_ylim)
    plot_pooled_summary(
        ax_extent,
        late_subject,
        title="Adaptation extent",
        ylabel="Hand angle ($^\\circ$)",
        color=MAGENTA_DARK,
        ylim=top_row_ylim,
        stat="mean",
        overlay_lines=[
            {
                "summary_df": pooled_model_late[model_name],
                "color": MEAN_TIMESERIES_MODEL_COLORS[model_name],
                "label": MEAN_TIMESERIES_MODEL_LABELS[model_name],
                "linewidth": 2.6 if model_name != "MeanTimeseriesVectorBCC" else 2.8,
                "linestyle": MODEL_DASH,
                "alpha": 0.70 if model_name != "MeanTimeseriesVectorBCC" else 0.95,
            }
            for model_name in MEAN_TIMESERIES_MODEL_ORDER
        ],
        legend_loc="upper right",
        metrics_lines=late_metrics_lines,
        metrics_loc="lower left",
    )

    plot_representative_learning_curves(ax_curves, model_df=pooled_vector_rep_curves)
    plot_pointer_summary(
        ax_device_extent,
        late_subject,
        title="Adaptation extent by input device",
        ylabel="Adaptation extent ($^\\circ$)",
        stat="mean",
        ylim=late_device_ylim,
        overlay_lines_by_pointer={
            pointer: {
                "summary_df": summary_df,
                "color": DEVICE_EDGE_COLORS[pointer],
                "linewidth": 2.8,
                "linestyle": MODEL_DASH,
            }
            for pointer, summary_df in device_vector_late.items()
        },
        show_style_legend=True,
        metrics_lines=device_metrics_lines,
        metrics_loc="upper left",
    )

    plot_pooled_summary(
        ax_early_stl,
        early_stl_subject,
        title="Early single-trial learning",
        ylabel="Same-target delta ($^\\circ$)",
        color=MAGENTA_DARK,
        ylim=stl_ylim,
        stat="mean",
        overlay_lines=[
            {
                "summary_df": pooled_vector_stl_early,
                "color": MEAN_TIMESERIES_MODEL_COLORS["MeanTimeseriesVectorBCC"],
                "label": "Vector BCC",
                "linewidth": 2.8,
                "linestyle": MODEL_DASH,
            }
        ],
        human_legend_label="Human mean",
        legend_loc="lower left",
        metrics_lines=early_metrics_lines,
        metrics_loc="upper right",
    )
    plot_pooled_summary(
        ax_washout_stl,
        washout_stl_subject,
        title="Washout single-trial learning",
        ylabel="Same-target delta ($^\\circ$)",
        color=TEAL_DARK,
        ylim=stl_ylim,
        stat="mean",
        overlay_lines=[
            {
                "summary_df": pooled_vector_stl_washout,
                "color": MEAN_TIMESERIES_MODEL_COLORS["MeanTimeseriesVectorBCC"],
                "linewidth": 2.8,
                "linestyle": MODEL_DASH,
            }
        ],
        metrics_lines=washout_metrics_lines,
        metrics_loc="upper left",
    )

    plot_phase_distribution(
        ax_peak_distribution,
        peak_subject,
        clamp_order=[15, 20, 25, 30, 35, 40, 45, 50],
        title="Peak-clamp hand-angle distributions",
        ylim=distribution_ylim,
    )
    plot_phase_distribution(
        ax_large_distribution,
        high_error_subject,
        clamp_order=[135, 140, 145, 150, 155, 160, 165, 170],
        title="Large-clamp hand-angle distributions",
        ylim=distribution_ylim,
    )

    plot_spatial_bias_with_tr_markers(
        ax_spatial_bias,
        raw=spatial_raw,
        summary=spatial_summary,
        fit_points=spatial_fit_points,
        ylim=spatial_bias_ylim,
    )
    plot_bias_direction_summary(
        ax_bias_groups,
        summary=bias_direction_summary,
        counts=bias_direction_counts,
        ylim=bias_direction_ylim,
    )

    plot_high_error_bias_sign_trajectories(ax_bias_curves)
    early_var = build_early_stl_variability_subject()
    late_var_subject = _load_subject_table("late_variability_subject.csv")
    early_var_summary = summarize_by_pointer(early_var, value_col="value", stat="median")
    late_var_summary = summarize_by_pointer(late_var_subject, stat="median")
    variability_ylim = padded_ylim(
        early_var_summary["ci_low"],
        early_var_summary["ci_high"],
        late_var_summary["ci_low"],
        late_var_summary["ci_high"],
        min_pad=1.0,
    )
    plot_pointer_summary(
        ax_early_variability,
        early_var,
        title="Early reach variability by input device",
        ylabel="Median trial-to-trial SD ($^\\circ$)",
        stat="median",
        value_col="value",
        ylim=variability_ylim,
        zero_line=False,
    )

    plot_pointer_summary(
        ax_late_variability,
        late_var_subject,
        title="Late reach variability by input device",
        ylabel="Median trial-to-trial SD ($^\\circ$)",
        stat="median",
        ylim=variability_ylim,
        zero_line=False,
    )
    plot_first_bout_radial_bridge_panels(ax_2d_updates, ax_radial_updates)
    # Keep the long kinematic labels clear of the panel letters.
    ax_2d_updates.set_ylabel("Cumulative azimuthal change\n(% target distance)")
    ax_radial_updates.set_ylabel("Cumulative radial change\n(% target distance)")

    for axes in (axes42, axes43, axes44):
        for label, ax in zip("ABCDEF", axes.ravel()):
            add_panel_label(ax, label)
    return {"figure_4_1": fig41, "figure_4_2": fig42, "figure_4_3": fig43, "figure_4_4": fig44}
