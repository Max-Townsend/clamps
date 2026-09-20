"""Figure S4.2: model comparison, recovery, and Vector-BCC parameters."""

import matplotlib.pyplot as plt
import pandas as pd

from clamp_analysis import paths

from . import panels, recovery


def build_figure():
    recovery.set_paper_rcparams()
    fig = plt.figure(figsize=(17, 18), constrained_layout=True)
    grid = fig.add_gridspec(3, 6, height_ratios=(1.15, 1, 1))
    bic_ax = fig.add_subplot(grid[0, :3])
    recovery_ax = fig.add_subplot(grid[0, 3:])
    panels.plot_model_delta_bic_panel(bic_ax, panels.load_mean_timeseries_dataset_bic_table())
    for annotation in bic_ax.texts:
        if annotation.get_text().startswith("Relative to the best model"):
            annotation.set_position((0.02, 0.97))
            annotation.set_ha("left")
    panels.add_panel_label(bic_ax, "A")

    winners = pd.read_csv(paths.RECOVERY_COLLECTION / "model_recovery_winners.csv")
    order = recovery.MODEL_ORDER
    counts = pd.crosstab(winners.generatingModel, winners.selectedModel).reindex(
        index=order, columns=order, fill_value=0
    )
    proportions = counts.div(counts.sum(axis=1), axis=0)
    # True model on x, recovered model on y, as labelled in the thesis.
    image = recovery_ax.imshow(proportions.to_numpy().T, vmin=0, vmax=1, cmap=recovery.HEATMAP_CMAP)
    labels = [panels.MEAN_TIMESERIES_MODEL_LABELS[name] for name in order]
    recovery_ax.set_xticks(range(3), labels, rotation=28, ha="right")
    recovery_ax.set_yticks(range(3), labels)
    recovery_ax.set(xlabel="True", ylabel="Recovered", title="Model recovery")
    for i in range(3):
        for j in range(3):
            value = proportions.iloc[i, j]
            recovery_ax.text(
                i,
                j,
                f"{value:.2f}\n({counts.iloc[i, j]})",
                ha="center",
                va="center",
                fontsize=14,
                color="white" if value >= 0.58 else "#111111",
            )
    fig.colorbar(image, ax=recovery_ax, fraction=0.048, pad=0.035, label="Selection proportion")
    panels.add_panel_label(recovery_ax, "B")

    parameters = recovery.parameter_plot_values(
        pd.read_csv(paths.RECOVERY_COLLECTION / "parameter_recovery_long.csv")
    )
    parameters = parameters.loc[parameters.model.eq("MeanTimeseriesVectorBCC")]
    for i, parameter in enumerate(
        recovery.ordered_parameters("MeanTimeseriesVectorBCC", parameters)
    ):
        ax = fig.add_subplot(grid[1 + i // 3, (i % 3) * 2 : (i % 3) * 2 + 2])
        sample = parameters.loc[parameters.parameter.eq(parameter)]
        true, recovered = sample.truePlotValue, sample.recoveredPlotValue
        ax.scatter(
            true,
            recovered,
            s=52,
            alpha=0.76,
            color=recovery.parameter_color("MeanTimeseriesVectorBCC", parameter),
            edgecolor=recovery.parameter_edge_color("MeanTimeseriesVectorBCC", parameter),
            linewidth=0.45,
        )
        lo, hi = recovery.square_limits(recovery.finite_array(true, recovered), parameter)
        ax.plot([lo, hi], [lo, hi], color="#303030", linewidth=1.25, linestyle=(0, (4.5, 3)))
        recovery.apply_square_axis_geometry(ax, lo, hi)
        ax.xaxis.set_major_formatter(recovery.format_axis_number)
        ax.yaxis.set_major_formatter(recovery.format_axis_number)
        ax.set(
            title=recovery.PARAMETER_LABELS.get(parameter, parameter),
            xlabel="True",
            ylabel="Recovered" if i % 3 == 0 else "",
        )
        ax.grid(color="0.86", linewidth=0.75)
        recovery.annotate_recovery_r(ax, true, recovered)
        panels.style_axes(ax)
        if i == 0:
            panels.add_panel_label(ax, "C")
    return fig
