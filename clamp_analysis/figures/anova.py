"""Figure S4.1: group-by-device means with two-sided 95% Student's t intervals.

Each participant contributes the mean across their four assigned clamps.
Group means are unconnected because the groups contain interleaved clamp sizes.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from clamp_analysis.statistics.anova import (
    CI_LEVEL,
    DEVICE_ORDER,
    MEASURES,
    cell_descriptives,
    clamp_labels,
    load_measure,
    participant_group_means,
)

# Match the Mouse/Trackpad palette used in the thesis figures.
DEVICE_FILL = {"Mouse": "#7FB8F1", "Trackpad": "#F5B16D"}

DEVICE_ERROR = {"Mouse": "#4F8ED8", "Trackpad": "#E58B25"}

DEVICE_MARKER = {"Mouse": "o", "Trackpad": "o"}

DEVICE_X_OFFSET = {"Mouse": -0.11, "Trackpad": 0.11}

THESIS_MARKER_AREA = 70.0

THESIS_MARKER_SIZE = THESIS_MARKER_AREA**0.5

THESIS_MARKER_EDGE_WIDTH = 1.0

THESIS_ERRORBAR_WIDTH = 1.3

THESIS_ERRORBAR_CAPSIZE = 3.0

PANEL_CONFIG = {
    "Azimuthal adaptation extent": {
        "label": "A",
        "title": "Azimuthal adaptation extent",
        "ylabel": "Adaptation extent (°)",
        "ylim": (-20.0, 55.0),
        "yticks": (-5.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0),
        "major_tick": 10.0,
    },
    "Early STL": {
        "label": "B",
        "title": "Early single-trial learning",
        "ylabel": "Same-target delta (°)",
        "ylim": (-2.5, 5.0),
        "major_tick": 1.0,
    },
    "Washout STL": {
        "label": "C",
        "title": "Washout single-trial learning",
        "ylabel": "Same-target delta (°)",
        "ylim": (-4.0, 2.75),
        "major_tick": 1.0,
    },
}


def configure_style() -> None:
    """Apply a restrained print style consistent with the thesis figures."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 9.0,
            "axes.titlesize": 10.5,
            "axes.titleweight": "normal",
            "axes.labelsize": 10.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def build_plot_data() -> pd.DataFrame:
    """Calculate participant-averaged cell means and confidence intervals."""
    results: list[pd.DataFrame] = []

    if CI_LEVEL != 0.95:
        raise AssertionError(f"This figure requires 95% CIs, found {CI_LEVEL:.3f}")

    for measure, specification in MEASURES.items():
        raw, _ = load_measure(
            specification["filename"],
            add_pointer=specification["add_pointer"],
        )
        participant_means = participant_group_means(raw)

        descriptive = cell_descriptives(
            participant_means,
            measure=measure,
            group_clamps=clamp_labels(raw),
        )
        if len(descriptive) != 20:
            raise AssertionError(f"{measure}: expected 20 group x device cells")
        results.append(descriptive)

    combined = pd.concat(results, ignore_index=True)
    combined["pointer"] = combined["pointer"].astype(str)

    # Panel A's sample-size strip applies to all three outcomes.
    n_by_measure = combined.pivot(
        index=["group", "pointer"],
        columns="measure",
        values="n",
    )
    if not n_by_measure.nunique(axis=1).eq(1).all():
        raise AssertionError("Groupwise sample sizes differ between outcomes")
    return combined


def add_sample_size_strip(ax: plt.Axes, panel_data: pd.DataFrame) -> None:
    """Add a compact groupwise-n band inside the bottom of Panel A."""
    band_top = -7.0
    row_y = {"Mouse": -13.0, "Trackpad": -17.0}

    ax.axhspan(-20.0, band_top, color="#FAFAFA", zorder=0)
    ax.axhline(band_top, color="#D5D5D5", linewidth=0.7, zorder=1)
    ax.text(
        0.72,
        -9.4,
        r"Group $n$",
        ha="left",
        va="center",
        fontsize=7.5,
        color="#555555",
        clip_on=True,
    )

    for device in DEVICE_ORDER:
        device_data = panel_data.loc[panel_data["pointer"].eq(device)].sort_values("group")
        ax.text(
            0.76,
            row_y[device],
            device[0],
            ha="center",
            va="center",
            fontsize=7.7,
            color=DEVICE_FILL[device],
            fontweight="bold",
            clip_on=True,
        )
        for row in device_data.itertuples(index=False):
            ax.text(
                float(row.group),
                row_y[device],
                str(int(row.n)),
                ha="center",
                va="center",
                fontsize=7.7,
                color=DEVICE_FILL[device],
                clip_on=True,
            )


def style_axis(ax: plt.Axes) -> None:
    """Style one panel without unnecessary chart furniture."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(colors="#333333")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#B0B0B0", linewidth=0.8, alpha=0.16)
    ax.axhline(0.0, color="#595959", linewidth=1.0, linestyle="--", zorder=1)
    ax.set_xlim(0.65, 10.35)
    ax.set_xticks(range(1, 11))
    ax.tick_params(axis="x", labelbottom=True)
    ax.set_xlabel("Assigned clamp group", labelpad=6)


def plot_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    measure: str,
) -> None:
    """Draw unconnected cell means with two-sided 95% CI error bars."""
    config = PANEL_CONFIG[measure]
    panel = data.loc[data["measure"].eq(measure)].copy()

    style_axis(ax)

    for device in DEVICE_ORDER:
        subset = panel.loc[panel["pointer"].eq(device)].sort_values("group")
        x = subset["group"].to_numpy(dtype=float) + DEVICE_X_OFFSET[device]
        mean = subset["mean"].to_numpy(dtype=float)
        low = subset["ci_low"].to_numpy(dtype=float)
        high = subset["ci_high"].to_numpy(dtype=float)

        ax.errorbar(
            x,
            mean,
            yerr=[mean - low, high - mean],
            fmt="none",
            ecolor=DEVICE_ERROR[device],
            elinewidth=THESIS_ERRORBAR_WIDTH,
            capsize=THESIS_ERRORBAR_CAPSIZE,
            capthick=THESIS_MARKER_EDGE_WIDTH,
            zorder=2,
        )
        ax.scatter(
            x,
            mean,
            s=THESIS_MARKER_AREA,
            marker=DEVICE_MARKER[device],
            color=DEVICE_FILL[device],
            edgecolor="white",
            linewidth=THESIS_MARKER_EDGE_WIDTH,
            zorder=3,
        )

    ax.set_title(config["title"], loc="center", pad=7)
    ax.set_ylabel(config["ylabel"], labelpad=7)
    ax.set_ylim(*config["ylim"])
    if "yticks" in config:
        ax.set_yticks(config["yticks"])
    else:
        ax.yaxis.set_major_locator(MultipleLocator(config["major_tick"]))
    ax.text(
        -0.105,
        1.055,
        config["label"],
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13.0,
        fontweight="bold",
        color="#111111",
    )


def make_figure(plot_data: pd.DataFrame) -> plt.Figure:
    """Assemble a portrait three-panel figure suitable for a thesis page."""
    figure, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(7.15, 9.50),
        sharex=True,
        gridspec_kw={"hspace": 0.44, "height_ratios": [1.18, 1.0, 1.0]},
    )

    measures = list(PANEL_CONFIG)
    for axis, measure in zip(axes, measures, strict=True):
        plot_panel(
            axis,
            plot_data,
            measure=measure,
        )

    panel_a_data = plot_data.loc[plot_data["measure"].eq("Azimuthal adaptation extent")].copy()
    add_sample_size_strip(axes[0], panel_a_data)

    total_n = {
        device: int(panel_a_data.loc[panel_a_data["pointer"].eq(device), "n"].sum())
        for device in DEVICE_ORDER
    }

    handles = [
        Line2D(
            [0],
            [0],
            color=DEVICE_FILL[device],
            linewidth=0.0,
            linestyle="none",
            marker=DEVICE_MARKER[device],
            markersize=THESIS_MARKER_SIZE,
            markerfacecolor=DEVICE_FILL[device],
            markeredgecolor="white",
            markeredgewidth=THESIS_MARKER_EDGE_WIDTH,
            label=rf"{device} ($n$ = {total_n[device]})",
        )
        for device in DEVICE_ORDER
    ]
    axes[0].legend(
        handles=handles,
        title="Input device",
        frameon=False,
        loc="upper right",
        ncol=2,
        columnspacing=1.0,
        handlelength=0.8,
        borderaxespad=0.3,
    )

    figure.subplots_adjust(left=0.145, right=0.975, top=0.975, bottom=0.075)
    return figure
