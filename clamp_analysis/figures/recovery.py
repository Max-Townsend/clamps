from __future__ import annotations

import math
from pathlib import Path

import matplotlib

from clamp_analysis import paths

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.ticker import MaxNLocator

from clamp_analysis.figures.panels import (
    DEVICE_COLORS,
    DEVICE_EDGE_COLORS,
    MEAN_TIMESERIES_MODEL_COLORS,
    MEAN_TIMESERIES_MODEL_LABELS,
    MEAN_TIMESERIES_MODEL_ORDER,
    PAPER_FONT,
    TEAL_DARK,
    TEAL_LIGHT,
    style_axes,
)

ROOT = paths.PROJECT_ROOT

MODEL_ORDER = list(MEAN_TIMESERIES_MODEL_ORDER)

PARAMETER_ORDER_BY_MODEL = {
    "MeanTimeseriesBaseBCC": [
        "propVar_mouse",
        "propVar_trackpad",
        "visVarSlope",
        "retention",
        "lr",
    ],
    "MeanTimeseriesCausalInf": [
        "senseVar_mouse",
        "senseVar_trackpad",
        "c",
        "retention",
        "lr",
    ],
    "MeanTimeseriesVectorBCC": [
        "propVar_mouse",
        "propVar_trackpad",
        "visVarSlope",
        "retention",
        "lr",
    ],
}

PARAMETER_LABELS = {
    "propVar_mouse": r"$\sigma_{p,\mathrm{mouse}}$",
    "propVar_trackpad": r"$\sigma_{p,\mathrm{trackpad}}$",
    "senseVar_mouse": r"$\sigma_{s,\mathrm{mouse}}$",
    "senseVar_trackpad": r"$\sigma_{s,\mathrm{trackpad}}$",
    "visVarSlope": r"$\sigma_{v\beta}$",
    "retention": r"$A$",
    "lr": r"$B$",
    "c": r"$c$",
}

DEVICE_PARAMETER_PREFIXES = {
    "propVar_mouse": "Mouse",
    "propVar_trackpad": "Trackpad",
    "senseVar_mouse": "Mouse",
    "senseVar_trackpad": "Trackpad",
}

NONNEGATIVE_PARAMETERS = {
    "propVar_mouse",
    "propVar_trackpad",
    "senseVar_mouse",
    "senseVar_trackpad",
    "c",
    "visVarSlope",
    "retention",
    "lr",
}

SD_DISPLAY_PARAMETERS = {
    "propVar_mouse",
    "propVar_trackpad",
    "senseVar_mouse",
    "senseVar_trackpad",
}

HEATMAP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "paper_recovery_proportion",
    ["#F7F7F7", TEAL_LIGHT, TEAL_DARK, "#111111"],
)


def set_paper_rcparams() -> None:
    plt.rcParams.update(
        {
            "font.size": PAPER_FONT["tick"],
            "axes.titlesize": PAPER_FONT["title"],
            "axes.labelsize": PAPER_FONT["axis_label"],
            "xtick.labelsize": PAPER_FONT["tick"],
            "ytick.labelsize": PAPER_FONT["tick"],
            "legend.fontsize": PAPER_FONT["legend"],
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv_first_existing(output_dir: Path, names: list[str]) -> pd.DataFrame:
    for name in names:
        path = output_dir / name
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(
        f"None of these input CSV files exist in {output_dir}: {', '.join(names)}"
    )


def pearson_r(x: pd.Series, y: pd.Series) -> float:
    x_values = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    if int(mask.sum()) < 3:
        return float("nan")
    if np.std(x_values[mask]) <= 0 or np.std(y_values[mask]) <= 0:
        return float("nan")
    return float(np.corrcoef(x_values[mask], y_values[mask])[0, 1])


def pearson_r_text(x: pd.Series, y: pd.Series) -> str:
    value = pearson_r(x, y)
    return "r = n/a" if not np.isfinite(value) else f"r = {value:.2f}"


def annotate_recovery_r(ax: plt.Axes, x: pd.Series, y: pd.Series) -> None:
    ax.text(
        0.055,
        0.94,
        pearson_r_text(x, y),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PAPER_FONT["legend"] * 0.78,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "0.78",
            "alpha": 0.9,
            "linewidth": 0.6,
        },
    )


def ensure_plot_subset_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "plotSubsetSlug" not in df.columns:
        df["plotSubsetSlug"] = df.get("subsetSlug", pd.Series(dtype=str)).astype(str)
    if "plotSubsetLabel" not in df.columns:
        source = (
            df["subsetLabel"] if "subsetLabel" in df.columns else df["plotSubsetSlug"].astype(str)
        )
        df["plotSubsetLabel"] = source
    return df


def parameter_plot_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    true_values = pd.to_numeric(df["trueValue"], errors="coerce").to_numpy(dtype=float)
    recovered_values = pd.to_numeric(df["recoveredValue"], errors="coerce").to_numpy(dtype=float)
    parameter = df["parameter"].astype(str)
    sd_mask = parameter.isin(SD_DISPLAY_PARAMETERS).to_numpy(dtype=bool)

    true_plot = true_values.copy()
    recovered_plot = recovered_values.copy()
    valid_true_sd = sd_mask & np.isfinite(true_plot) & (true_plot >= 0.0)
    valid_recovered_sd = sd_mask & np.isfinite(recovered_plot) & (recovered_plot >= 0.0)
    true_plot[sd_mask] = np.nan
    recovered_plot[sd_mask] = np.nan
    true_plot[valid_true_sd] = np.sqrt(true_values[valid_true_sd])
    recovered_plot[valid_recovered_sd] = np.sqrt(recovered_values[valid_recovered_sd])

    df["displayParameter"] = parameter.map(lambda name: PARAMETER_LABELS.get(name, name))
    df["displayScale"] = np.where(sd_mask, "sd", "raw")
    df["truePlotValue"] = true_plot
    df["recoveredPlotValue"] = recovered_plot
    df["plotError"] = df["recoveredPlotValue"] - df["truePlotValue"]
    df["plotAbsError"] = np.abs(df["plotError"])
    return df


def finite_array(*values: pd.Series | np.ndarray) -> np.ndarray:
    arrays = []
    for value in values:
        array = np.asarray(value, dtype=float).ravel()
        array = array[np.isfinite(array)]
        if array.size:
            arrays.append(array)
    if not arrays:
        return np.asarray([], dtype=float)
    return np.concatenate(arrays)


def square_limits(values: np.ndarray, parameter: str | None = None) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0

    lo = float(np.min(values))
    hi = float(np.max(values))
    if math.isclose(lo, hi):
        pad = max(abs(lo) * 0.08, 1e-6)
    else:
        pad = max((hi - lo) * 0.08, 1e-6)
    lo -= pad
    hi += pad

    # Clip near-zero padding so nonnegative parameters do not acquire negative axes.
    data_min = float(np.min(values))
    if parameter in NONNEGATIVE_PARAMETERS and data_min >= 0.0 and lo < 0.0:
        lo = 0.0
    if math.isclose(lo, hi):
        hi = lo + 1.0
    return lo, hi


def parameter_color(model_name: str, parameter: str) -> str:
    device = DEVICE_PARAMETER_PREFIXES.get(parameter)
    if device is not None:
        return DEVICE_COLORS[device]
    return MEAN_TIMESERIES_MODEL_COLORS.get(model_name, "#111111")


def parameter_edge_color(model_name: str, parameter: str) -> str:
    device = DEVICE_PARAMETER_PREFIXES.get(parameter)
    if device is not None:
        return DEVICE_EDGE_COLORS[device]
    return MEAN_TIMESERIES_MODEL_COLORS.get(model_name, "#111111")


def format_axis_number(value: float, _pos: int | None = None) -> str:
    if not np.isfinite(value):
        return ""
    abs_value = abs(float(value))
    if abs_value >= 100:
        return f"{value:.0f}"
    if abs_value >= 10:
        return f"{value:.1f}"
    if abs_value >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def square_ticks(lo: float, hi: float, *, max_ticks: int = 5) -> np.ndarray:
    locator = MaxNLocator(nbins=max_ticks, steps=[1, 2, 2.5, 5, 10])
    ticks = np.asarray(locator.tick_values(lo, hi), dtype=float)
    ticks = ticks[np.isfinite(ticks)]
    ticks = ticks[(ticks >= lo - 1e-12) & (ticks <= hi + 1e-12)]
    if ticks.size < 2:
        ticks = np.linspace(lo, hi, 3)
    return ticks


def apply_square_axis_geometry(ax: plt.Axes, lo: float, hi: float) -> None:
    ticks = square_ticks(lo, hi)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_aspect("equal", adjustable="box")


def save_figure(
    fig: plt.Figure,
    png_path: Path,
    *,
    figure_key: str,
    figure_kind: str,
    primary_paths: list[Path],
    all_paths: list[dict[str, object]],
) -> None:
    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.12, "facecolor": "white"}
    for fmt in ("png", "svg", "pdf"):
        path = png_path.with_suffix(f".{fmt}")
        if fmt == "png":
            fig.savefig(path, dpi=220, **save_kwargs)
        else:
            fig.savefig(path, **save_kwargs)
        all_paths.append(
            {
                "figureKey": figure_key,
                "figureKind": figure_kind,
                "format": fmt,
                "path": relative_path(path),
            }
        )
    primary_paths.append(png_path)


def summarize_recovery(gdf: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, part in gdf.groupby(group_cols, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        true_col = "truePlotValue" if "truePlotValue" in part.columns else "trueValue"
        recovered_col = (
            "recoveredPlotValue" if "recoveredPlotValue" in part.columns else "recoveredValue"
        )
        error_col = "plotError" if "plotError" in part.columns else "error"
        abs_error_col = "plotAbsError" if "plotAbsError" in part.columns else "absError"
        true_values = pd.to_numeric(part[true_col], errors="coerce")
        recovered_values = pd.to_numeric(part[recovered_col], errors="coerce")
        error = pd.to_numeric(part[error_col], errors="coerce")
        abs_error = pd.to_numeric(part[abs_error_col], errors="coerce")
        row = dict(zip(group_cols, key))
        if "displayParameter" in part.columns:
            row["displayParameter"] = str(part["displayParameter"].iloc[0])
        if "displayScale" in part.columns:
            row["displayScale"] = str(part["displayScale"].iloc[0])
        row.update(
            {
                "n": int(len(part)),
                "correlation": pearson_r(true_values, recovered_values),
                "bias": float(np.nanmean(error)),
                "mae": float(np.nanmean(abs_error)),
                "rmse": float(np.sqrt(np.nanmean(np.square(error)))),
            }
        )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def plot_motorvar_calibration(
    output_dir: Path,
    primary_paths: list[Path],
    all_paths: list[dict[str, object]],
) -> None:
    path = output_dir / "baseline_motorvar_calibration.csv"
    if not path.exists():
        return

    df = pd.read_csv(path)
    if df.empty or "fixedMotorVar" not in df.columns:
        return

    plot_df = df.copy()
    plot_df["subsetLabel"] = plot_df.get("subsetLabel", plot_df["subsetSlug"]).astype(str)
    plot_df["pointerDevice"] = plot_df.get("pointerDevice", plot_df["subsetLabel"]).astype(str)
    plot_df["fixedMotorVar"] = pd.to_numeric(plot_df["fixedMotorVar"], errors="coerce")
    plot_df = plot_df.dropna(subset=["fixedMotorVar"])
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7.1, 5.4), constrained_layout=True)
    x = np.arange(len(plot_df))
    colors = [
        DEVICE_COLORS.get(device, "#9A9A9A") for device in plot_df["pointerDevice"].astype(str)
    ]
    edge_colors = [
        DEVICE_EDGE_COLORS.get(device, "#4A4A4A") for device in plot_df["pointerDevice"].astype(str)
    ]
    bars = ax.bar(
        x,
        plot_df["fixedMotorVar"].to_numpy(dtype=float),
        color=colors,
        edgecolor=edge_colors,
        linewidth=1.0,
        width=0.64,
    )

    for bar, value in zip(bars, plot_df["fixedMotorVar"].to_numpy(dtype=float)):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=PAPER_FONT["legend"] * 0.75,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["subsetLabel"].astype(str))
    ax.set_ylabel("Fixed motor variance")
    ax.set_title("Motor variance calibration")
    ax.set_ylim(0.0, float(plot_df["fixedMotorVar"].max()) * 1.25)
    style_axes(ax)
    save_figure(
        fig,
        output_dir / "motorvar_calibration_by_subset.png",
        figure_key="motorvar_calibration_by_subset",
        figure_kind="calibration",
        primary_paths=primary_paths,
        all_paths=all_paths,
    )
    plt.close(fig)


def plot_model_recovery(
    output_dir: Path,
    winners_df: pd.DataFrame,
    primary_paths: list[Path],
    all_paths: list[dict[str, object]],
) -> None:
    winners_df = ensure_plot_subset_columns(winners_df)
    winners_df.to_csv(output_dir / "plot_model_recovery_winners.csv", index=False)

    confusion_rows = []
    for (subset_slug, subset_label), subset_df in winners_df.groupby(
        ["plotSubsetSlug", "plotSubsetLabel"], sort=False
    ):
        counts = pd.crosstab(
            subset_df["generatingModel"],
            subset_df["selectedModel"],
            dropna=False,
        ).reindex(index=MODEL_ORDER, columns=MODEL_ORDER, fill_value=0)
        props = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

        for generating_model in MODEL_ORDER:
            for selected_model in MODEL_ORDER:
                confusion_rows.append(
                    {
                        "plotSubsetSlug": subset_slug,
                        "plotSubsetLabel": subset_label,
                        "generatingModel": generating_model,
                        "selectedModel": selected_model,
                        "count": int(counts.loc[generating_model, selected_model]),
                        "selectionProportion": float(props.loc[generating_model, selected_model]),
                    }
                )

        fig, ax = plt.subplots(figsize=(8.2, 7.1), constrained_layout=True)
        image = ax.imshow(
            props.to_numpy(dtype=float),
            vmin=0.0,
            vmax=1.0,
            cmap=HEATMAP_CMAP,
            interpolation="nearest",
        )
        tick_labels = [MEAN_TIMESERIES_MODEL_LABELS.get(model, model) for model in MODEL_ORDER]
        ax.set_xticks(np.arange(len(MODEL_ORDER)))
        ax.set_yticks(np.arange(len(MODEL_ORDER)))
        ax.set_xticklabels(tick_labels, rotation=28, ha="right", rotation_mode="anchor")
        ax.set_yticklabels(tick_labels)
        ax.set_xlabel("Selected model")
        ax.set_ylabel("Generating model")
        ax.set_title(f"Model recovery: {subset_label}")

        for i, generating_model in enumerate(MODEL_ORDER):
            for j, selected_model in enumerate(MODEL_ORDER):
                value = float(props.loc[generating_model, selected_model])
                count = int(counts.loc[generating_model, selected_model])
                ax.text(
                    j,
                    i,
                    f"{value:.2f}\n({count})",
                    ha="center",
                    va="center",
                    color="white" if value >= 0.58 else "#111111",
                    fontsize=PAPER_FONT["legend"] * 0.76,
                    linespacing=1.05,
                )

        cbar = fig.colorbar(image, ax=ax, fraction=0.048, pad=0.035)
        cbar.set_label("Selection proportion", fontsize=PAPER_FONT["axis_label"])
        cbar.ax.tick_params(labelsize=PAPER_FONT["tick"], length=4.0, width=1.0)
        style_axes(ax)
        save_figure(
            fig,
            output_dir / f"{subset_slug}_model_recovery_confusion.png",
            figure_key=f"{subset_slug}_model_recovery_confusion",
            figure_kind="model_recovery",
            primary_paths=primary_paths,
            all_paths=all_paths,
        )
        plt.close(fig)

    pd.DataFrame.from_records(confusion_rows).to_csv(
        output_dir / "plot_model_recovery_confusion.csv", index=False
    )


def ordered_parameters(model_name: str, gdf: pd.DataFrame) -> list[str]:
    present = set(gdf["parameter"].astype(str))
    ordered = [
        parameter
        for parameter in PARAMETER_ORDER_BY_MODEL.get(model_name, [])
        if parameter in present
    ]
    remainder = [
        parameter
        for parameter in gdf["parameter"].astype(str).drop_duplicates()
        if parameter not in set(ordered)
    ]
    return ordered + remainder


def plot_parameter_recovery(
    output_dir: Path,
    parameter_df: pd.DataFrame,
    primary_paths: list[Path],
    all_paths: list[dict[str, object]],
) -> None:
    parameter_df = ensure_plot_subset_columns(parameter_df)
    parameter_df = parameter_plot_values(parameter_df)
    parameter_df.to_csv(output_dir / "plot_parameter_recovery_long.csv", index=False)

    summary_df = summarize_recovery(
        parameter_df,
        ["plotSubsetSlug", "plotSubsetLabel", "model", "parameter"],
    )
    summary_df.to_csv(output_dir / "plot_parameter_recovery_summary.csv", index=False)

    for (subset_slug, subset_label, model_name), gdf in parameter_df.groupby(
        ["plotSubsetSlug", "plotSubsetLabel", "model"], sort=False
    ):
        params = ordered_parameters(str(model_name), gdf)
        if not params:
            continue

        n_params = len(params)
        ncols = min(3, n_params)
        nrows = int(math.ceil(n_params / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5.35 * ncols, 4.95 * nrows),
            constrained_layout=True,
            squeeze=False,
        )
        axes_flat = axes.ravel()

        for plot_idx, (ax, parameter) in enumerate(zip(axes_flat, params)):
            pdf = gdf.loc[gdf["parameter"].astype(str).eq(parameter)].copy()
            true_values = pd.to_numeric(pdf["truePlotValue"], errors="coerce")
            recovered_values = pd.to_numeric(pdf["recoveredPlotValue"], errors="coerce")
            color = parameter_color(str(model_name), parameter)
            edge_color = parameter_edge_color(str(model_name), parameter)
            ax.scatter(
                true_values,
                recovered_values,
                s=52,
                alpha=0.76,
                color=color,
                edgecolor=edge_color,
                linewidth=0.45,
            )

            lo, hi = square_limits(finite_array(true_values, recovered_values), parameter)
            ax.plot(
                [lo, hi],
                [lo, hi],
                color="#303030",
                linewidth=1.25,
                linestyle=(0, (4.5, 3.0)),
                zorder=0,
            )
            apply_square_axis_geometry(ax, lo, hi)
            ax.xaxis.set_major_formatter(format_axis_number)
            ax.yaxis.set_major_formatter(format_axis_number)
            ax.set_title(PARAMETER_LABELS.get(parameter, parameter))
            row_idx = plot_idx // ncols
            col_idx = plot_idx % ncols
            ax.set_xlabel("True" if row_idx == nrows - 1 else "")
            ax.set_ylabel("Recovered" if col_idx == 0 else "")
            ax.grid(color="0.86", linewidth=0.75)
            annotate_recovery_r(ax, true_values, recovered_values)
            style_axes(ax)

        for ax in axes_flat[n_params:]:
            ax.axis("off")

        save_figure(
            fig,
            output_dir / f"{subset_slug}_{model_name}_parameter_recovery.png",
            figure_key=f"{subset_slug}_{model_name}_parameter_recovery",
            figure_kind="parameter_recovery",
            primary_paths=primary_paths,
            all_paths=all_paths,
        )
        plt.close(fig)


def load_or_build_alpha_beta_average(output_dir: Path) -> pd.DataFrame:
    average_path = output_dir / "plot_vector_bcc_effective_alpha_beta_average_recovery.csv"
    if average_path.exists():
        return pd.read_csv(average_path)

    source_path = output_dir / "vector_bcc_effective_alpha_beta_recovery_long.csv"
    if not source_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(source_path)
    if df.empty:
        return pd.DataFrame()

    average_df = df.groupby(["subsetSlug", "syntheticId", "metric"], as_index=False).agg(
        trueValue=("trueValue", "mean"), recoveredValue=("recoveredValue", "mean")
    )
    average_df["error"] = average_df["recoveredValue"] - average_df["trueValue"]
    average_df["absError"] = np.abs(average_df["error"])
    average_df.to_csv(
        output_dir / "vector_bcc_effective_alpha_beta_average_recovery.csv",
        index=False,
    )
    return ensure_plot_subset_columns(average_df)


def plot_alpha_beta_average_recovery(
    output_dir: Path,
    primary_paths: list[Path],
    all_paths: list[dict[str, object]],
) -> None:
    alpha_beta_df = load_or_build_alpha_beta_average(output_dir)
    if alpha_beta_df.empty:
        return

    alpha_beta_df = ensure_plot_subset_columns(alpha_beta_df)
    alpha_beta_df.to_csv(
        output_dir / "plot_vector_bcc_effective_alpha_beta_average_recovery.csv",
        index=False,
    )
    summarize_recovery(
        alpha_beta_df,
        ["plotSubsetSlug", "plotSubsetLabel", "metric"],
    ).to_csv(
        output_dir / "plot_vector_bcc_effective_alpha_beta_average_recovery_summary.csv",
        index=False,
    )

    metric_labels = {
        "alpha": r"Mean $\alpha$",
        "beta": r"Mean $\beta$",
    }
    metric_colors = {
        "alpha": MEAN_TIMESERIES_MODEL_COLORS["MeanTimeseriesVectorBCC"],
        "beta": TEAL_DARK,
    }
    limits_by_metric = {
        metric: square_limits(
            finite_array(gdf["trueValue"], gdf["recoveredValue"]),
            str(metric),
        )
        for metric, gdf in alpha_beta_df.groupby("metric", sort=False)
    }

    for (subset_slug, subset_label), gdf in alpha_beta_df.groupby(
        ["plotSubsetSlug", "plotSubsetLabel"], sort=False
    ):
        metrics = [
            metric for metric in ["alpha", "beta"] if metric in set(gdf["metric"].astype(str))
        ]
        if not metrics:
            continue

        fig, axes = plt.subplots(
            1,
            len(metrics),
            figsize=(5.35 * len(metrics), 5.1),
            constrained_layout=True,
            squeeze=False,
        )
        for plot_idx, (ax, metric) in enumerate(zip(axes.ravel(), metrics)):
            pdf = gdf.loc[gdf["metric"].astype(str).eq(metric)].copy()
            true_values = pd.to_numeric(pdf["trueValue"], errors="coerce")
            recovered_values = pd.to_numeric(pdf["recoveredValue"], errors="coerce")
            color = metric_colors.get(metric, TEAL_DARK)
            ax.scatter(
                true_values,
                recovered_values,
                s=58,
                alpha=0.78,
                color=color,
                edgecolor="#111111",
                linewidth=0.42,
            )
            lo, hi = limits_by_metric[metric]
            ax.plot(
                [lo, hi],
                [lo, hi],
                color="#303030",
                linewidth=1.25,
                linestyle=(0, (4.5, 3.0)),
                zorder=0,
            )
            apply_square_axis_geometry(ax, lo, hi)
            ax.xaxis.set_major_formatter(format_axis_number)
            ax.yaxis.set_major_formatter(format_axis_number)
            ax.set_title(metric_labels.get(metric, metric))
            ax.set_xlabel("True")
            ax.set_ylabel("Recovered" if plot_idx == 0 else "")
            ax.grid(color="0.86", linewidth=0.75)
            annotate_recovery_r(ax, true_values, recovered_values)
            style_axes(ax)

        save_figure(
            fig,
            output_dir
            / f"{subset_slug}_MeanTimeseriesVectorBCC_effective_alpha_beta_average_recovery.png",
            figure_key=(
                f"{subset_slug}_MeanTimeseriesVectorBCC_effective_alpha_beta_average_recovery"
            ),
            figure_kind="effective_alpha_beta_recovery",
            primary_paths=primary_paths,
            all_paths=all_paths,
        )
        plt.close(fig)


def regenerate_recovery_figures(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_paper_rcparams()

    primary_paths: list[Path] = []
    all_paths: list[dict[str, object]] = []

    plot_motorvar_calibration(output_dir, primary_paths, all_paths)

    winners_df = read_csv_first_existing(
        output_dir,
        ["plot_model_recovery_winners.csv", "model_recovery_winners.csv"],
    )
    plot_model_recovery(output_dir, winners_df, primary_paths, all_paths)

    parameter_df = read_csv_first_existing(
        output_dir,
        ["plot_parameter_recovery_long.csv", "parameter_recovery_long.csv"],
    )
    plot_parameter_recovery(output_dir, parameter_df, primary_paths, all_paths)

    plot_alpha_beta_average_recovery(output_dir, primary_paths, all_paths)

    pd.DataFrame({"path": [relative_path(path) for path in primary_paths]}).to_csv(
        output_dir / "recovery_figure_paths.csv",
        index=False,
    )
    pd.DataFrame.from_records(all_paths).to_csv(
        output_dir / "recovery_manuscript_figure_paths.csv",
        index=False,
    )
