from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

from clamp_analysis.kinematics.calibration import (
    CALIBRATED_PATHS_PATH,
    CALIBRATED_SUMMARY_PATH,
    add_gradient_line_by_clamp,
    add_gradient_path,
    evaluate_calibrated_first_bout,
    format_p,
    symmetric_limit,
)
from clamp_analysis.kinematics.colors import (
    CLAMP_CMAP,
    CLAMP_NORM,
    ONE_D_CMAP,
    TWO_D_CMAP,
)

SELECTED_FRACTION = 1.00

plt.rcParams.update(
    {
        "font.size": 8.5,
        "axes.titlesize": 10.0,
        "axes.labelsize": 8.8,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.3,
        "legend.title_fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_calibrated_tables(force_rebuild: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not force_rebuild and CALIBRATED_SUMMARY_PATH.exists() and CALIBRATED_PATHS_PATH.exists():
        return pd.read_csv(CALIBRATED_SUMMARY_PATH), pd.read_csv(CALIBRATED_PATHS_PATH)
    summary, paths = evaluate_calibrated_first_bout()
    summary.to_csv(CALIBRATED_SUMMARY_PATH, index=False)
    paths.to_csv(CALIBRATED_PATHS_PATH, index=False)
    return summary, paths


def ols_fit_with_mean_ci(
    x: np.ndarray, y: np.ndarray, x_values: np.ndarray
) -> dict[str, float | np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_values = np.asarray(x_values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 3:
        nan_line = np.full_like(x_values, np.nan, dtype=float)
        return {
            "intercept_pct_target": np.nan,
            "slope_pct_target_per_deg": np.nan,
            "slope_pct_target_per_100deg": np.nan,
            "slope_ci_low_pct_target_per_100deg": np.nan,
            "slope_ci_high_pct_target_per_100deg": np.nan,
            "p_two_sided": np.nan,
            "line": nan_line,
            "ci_low": nan_line,
            "ci_high": nan_line,
        }

    design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ beta
    dof = len(x) - 2
    rss = float(np.sum((y - fitted) ** 2))
    mse = rss / max(dof, 1)
    xtx_inv = np.linalg.inv(design.T @ design)
    pred_design = np.column_stack([np.ones(len(x_values)), x_values])
    line = pred_design @ beta
    leverage = np.sum((pred_design @ xtx_inv) * pred_design, axis=1)
    tcrit = float(stats.t.ppf(0.975, dof)) if dof > 0 else np.nan
    mean_se = np.sqrt(np.maximum(mse * leverage, 0.0))
    slope_se = float(np.sqrt(max(mse * xtx_inv[1, 1], 0.0)))
    if dof > 0 and slope_se > 0.0:
        slope_t = float(beta[1] / slope_se)
        p_two_sided = float(2.0 * stats.t.sf(abs(slope_t), dof))
    else:
        p_two_sided = np.nan

    slope_ci = (
        np.asarray([beta[1] - tcrit * slope_se, beta[1] + tcrit * slope_se], dtype=float) * 100.0
    )
    return {
        "intercept_pct_target": float(beta[0]),
        "slope_pct_target_per_deg": float(beta[1]),
        "slope_pct_target_per_100deg": float(100.0 * beta[1]),
        "slope_ci_low_pct_target_per_100deg": float(slope_ci[0]),
        "slope_ci_high_pct_target_per_100deg": float(slope_ci[1]),
        "p_two_sided": p_two_sided,
        "line": line,
        "ci_low": line - tcrit * mean_se,
        "ci_high": line + tcrit * mean_se,
    }


def model_line_metrics(human_y: np.ndarray, model_y: np.ndarray) -> dict[str, float]:
    human_y = np.asarray(human_y, dtype=float)
    model_y = np.asarray(model_y, dtype=float)
    valid = np.isfinite(human_y) & np.isfinite(model_y)
    human_y = human_y[valid]
    model_y = model_y[valid]
    if len(human_y) == 0:
        return {"rmse": np.nan, "centered_r2": np.nan}
    residual = human_y - model_y
    rmse = float(np.sqrt(np.mean(residual**2)))
    denom = float(np.sum((human_y - np.mean(human_y)) ** 2))
    centered_r2 = np.nan if denom <= 1e-12 else 1.0 - float(np.sum(residual**2)) / denom
    return {"rmse": rmse, "centered_r2": centered_r2}


def ols_point_metrics(
    fit: dict[str, float | np.ndarray], x: np.ndarray, y: np.ndarray
) -> dict[str, float]:
    yhat = float(fit["intercept_pct_target"]) + float(fit["slope_pct_target_per_deg"]) * np.asarray(
        x, dtype=float
    )
    return model_line_metrics(y, yhat)


def focus_paths(paths: pd.DataFrame) -> pd.DataFrame:
    return paths.loc[np.isclose(paths["ballistic_fraction"], SELECTED_FRACTION)].copy()


def scheme_color(cmap, level: float = 0.76):
    return cmap(float(level))


MODEL_RADIAL_FIT_ALPHA = 0.62

MODEL_RADIAL_FIT_LINESTYLE = "--"


def radial_dose_legend_handles(human_mean_label: str = "human mean") -> list:
    human_color = scheme_color(CLAMP_CMAP)
    one_d_color = scheme_color(ONE_D_CMAP)
    two_d_color = scheme_color(TWO_D_CMAP)
    return [
        Patch(facecolor=human_color, alpha=0.16, edgecolor="none", label="human OLS fit 95% CI"),
        Line2D([0], [0], color=human_color, linewidth=2.1, linestyle="-", label="human OLS fit"),
        Line2D(
            [0],
            [0],
            color=one_d_color,
            linewidth=2.0,
            linestyle=MODEL_RADIAL_FIT_LINESTYLE,
            alpha=MODEL_RADIAL_FIT_ALPHA,
            label="calibrated 1D prediction",
        ),
        Line2D(
            [0],
            [0],
            color=two_d_color,
            linewidth=2.1,
            linestyle=MODEL_RADIAL_FIT_LINESTYLE,
            alpha=MODEL_RADIAL_FIT_ALPHA,
            label="calibrated 2D prediction",
        ),
        Line2D(
            [0],
            [0],
            color=human_color,
            marker="o",
            markerfacecolor="white",
            markeredgecolor=human_color,
            linewidth=0,
            label=human_mean_label,
        ),
    ]


def plot_model_geometry(ax: plt.Axes, paths: pd.DataFrame, summary: pd.DataFrame) -> None:
    focus = focus_paths(paths)
    human_points = focus.loc[focus["panel"].eq("Human") & focus["path_type"].eq("condition_mean")]
    human_path = focus.loc[focus["panel"].eq("Human") & focus["path_type"].eq("smoothed_path")]
    one_d_path = focus.loc[
        focus["panel"].eq("Calibrated 1D") & focus["path_type"].eq("smoothed_path")
    ]
    two_d_path = focus.loc[
        focus["panel"].eq("Calibrated 2D") & focus["path_type"].eq("smoothed_path")
    ]
    row = summary.loc[np.isclose(summary["ballistic_fraction"], SELECTED_FRACTION)].iloc[0]

    ax.scatter(
        human_points["forward"],
        human_points["sideways"],
        c=human_points["clamp_magnitude"],
        cmap=CLAMP_CMAP,
        norm=CLAMP_NORM,
        s=29,
        alpha=0.52,
        edgecolors="none",
        zorder=1,
    )
    add_gradient_path(ax, human_path, linewidth=4.2, alpha=0.82, linestyle="solid", zorder=3)
    add_gradient_path(
        ax,
        one_d_path,
        linewidth=2.6,
        alpha=0.96,
        linestyle="-",
        zorder=4,
        cmap=ONE_D_CMAP,
    )
    add_gradient_path(
        ax,
        two_d_path,
        linewidth=2.8,
        alpha=0.96,
        linestyle="-",
        zorder=5,
        cmap=TWO_D_CMAP,
    )
    values = focus[["forward", "sideways"]].to_numpy(dtype=float)
    limit = symmetric_limit(values, pad=0.16, floor=8.0)
    ax.axhline(0.0, color="0.84", linewidth=0.9)
    ax.axvline(0.0, color="0.84", linewidth=0.9)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"A. Cumulative calibrated geometry at {int(round(100 * SELECTED_FRACTION))}% arc")
    ax.set_xlabel("Cumulative radial change (% target distance)")
    ax.set_ylabel("Cumulative tangential change (% target distance)")
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=scheme_color(CLAMP_CMAP),
                linewidth=4.0,
                alpha=0.7,
                label="human trend",
            ),
            Line2D(
                [0],
                [0],
                color=scheme_color(ONE_D_CMAP),
                linewidth=2.6,
                linestyle="-",
                label="calibrated 1D",
            ),
            Line2D(
                [0],
                [0],
                color=scheme_color(TWO_D_CMAP),
                linewidth=2.8,
                linestyle="-",
                label="calibrated 2D",
            ),
        ],
        frameon=False,
        loc="best",
    )
    ax.text(
        0.04,
        0.05,
        (
            f"2D R2={float(row['two_d_trendline_r2_from_zero']):.2f}, "
            f"RMSE={float(row['two_d_trendline_rmse']):.1f}\n"
            f"1D R2={float(row['one_d_trendline_r2_from_zero']):.2f}, "
            f"RMSE={float(row['one_d_trendline_rmse']):.1f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.0,
    )


def plot_model_dose_response(
    ax: plt.Axes, paths: pd.DataFrame, summary: pd.DataFrame, axis: str
) -> None:
    focus = focus_paths(paths)
    human = focus.loc[
        focus["panel"].eq("Human") & focus["path_type"].eq("condition_mean")
    ].sort_values("clamp_magnitude")
    one_d_points = focus.loc[
        focus["panel"].eq("Calibrated 1D") & focus["path_type"].eq("condition_mean")
    ].sort_values("clamp_magnitude")
    two_d_points = focus.loc[
        focus["panel"].eq("Calibrated 2D") & focus["path_type"].eq("condition_mean")
    ].sort_values("clamp_magnitude")
    one_d = focus.loc[
        focus["panel"].eq("Calibrated 1D") & focus["path_type"].eq("smoothed_path")
    ].sort_values("clamp_magnitude")
    two_d = focus.loc[
        focus["panel"].eq("Calibrated 2D") & focus["path_type"].eq("smoothed_path")
    ].sort_values("clamp_magnitude")

    if axis == "radial":
        col = "forward"
        ci_col = "forward_ci95"
        title = "B. Cumulative radial change separates 1D from 2D"
        ylabel = "Cumulative radial change (% target distance)"
    else:
        col = "sideways"
        ci_col = "sideways_ci95"
        title = "C. Tangential update is not the discriminator"
        ylabel = "Cumulative tangential change (% target distance)"

    x = human["clamp_magnitude"].to_numpy(dtype=float)
    y = human[col].to_numpy(dtype=float)
    ci = human[ci_col].to_numpy(dtype=float)
    human_color = scheme_color(CLAMP_CMAP)
    for _, row in human.iterrows():
        ax.errorbar(
            float(row["clamp_magnitude"]),
            float(row[col]),
            yerr=float(row[ci_col]),
            fmt="o",
            color=human_color,
            markerfacecolor="white",
            markeredgecolor=human_color,
            markeredgewidth=1.0,
            markersize=4.2,
            elinewidth=0.9,
            capsize=2.0,
            alpha=0.92,
            zorder=5,
        )
    if axis == "radial":
        x_band = np.linspace(0.0, 175.0, 220)
        human_fit = ols_fit_with_mean_ci(x, y, x_band)
        ax.fill_between(
            x_band,
            human_fit["ci_low"],
            human_fit["ci_high"],
            color=human_color,
            alpha=0.16,
            linewidth=0.0,
            label="human OLS fit 95% CI",
            zorder=1,
        )
        ax.plot(
            x_band,
            human_fit["line"],
            color=human_color,
            linestyle="-",
            linewidth=2.1,
            label="human OLS fit",
            zorder=4,
        )
        ax.plot(
            one_d_points["clamp_magnitude"].to_numpy(dtype=float),
            one_d_points[col].to_numpy(dtype=float),
            color=scheme_color(ONE_D_CMAP),
            linestyle=MODEL_RADIAL_FIT_LINESTYLE,
            linewidth=2.0,
            alpha=MODEL_RADIAL_FIT_ALPHA,
            label="calibrated 1D prediction",
            zorder=2,
        )
        ax.plot(
            two_d_points["clamp_magnitude"].to_numpy(dtype=float),
            two_d_points[col].to_numpy(dtype=float),
            color=scheme_color(TWO_D_CMAP),
            linestyle=MODEL_RADIAL_FIT_LINESTYLE,
            linewidth=2.1,
            alpha=MODEL_RADIAL_FIT_ALPHA,
            label="calibrated 2D prediction",
            zorder=3,
        )
        human_ols_metrics = ols_point_metrics(human_fit, x, y)
        one_d_metrics = model_line_metrics(y, one_d_points[col].to_numpy(dtype=float))
        two_d_metrics = model_line_metrics(y, two_d_points[col].to_numpy(dtype=float))
        stat_text = (
            f"OLS human fit\n"
            f"Human b={float(human_fit['slope_pct_target_per_100deg']):.1f}%/100 deg\n"
            f"95% CI [{float(human_fit['slope_ci_low_pct_target_per_100deg']):.1f}, "
            f"{float(human_fit['slope_ci_high_pct_target_per_100deg']):.1f}]\n"
            f"{format_p(float(human_fit['p_two_sided']))}\n\n"
            f"Radial prediction fit\n"
            f"OLS R2c={float(human_ols_metrics['centered_r2']):.2f}, RMSE={float(human_ols_metrics['rmse']):.1f}\n"
            f"2D R2c={float(two_d_metrics['centered_r2']):.2f}, RMSE={float(two_d_metrics['rmse']):.1f}\n"
            f"1D R2c={float(one_d_metrics['centered_r2']):.2f}, RMSE={float(one_d_metrics['rmse']):.1f}"
        )
    else:
        add_gradient_line_by_clamp(
            ax,
            one_d["clamp_magnitude"].to_numpy(dtype=float),
            one_d[col].to_numpy(dtype=float),
            cmap=ONE_D_CMAP,
            linestyle="-",
            linewidth=2.2,
            label="calibrated 1D trend",
            zorder=3,
        )
        add_gradient_line_by_clamp(
            ax,
            two_d["clamp_magnitude"].to_numpy(dtype=float),
            two_d[col].to_numpy(dtype=float),
            cmap=TWO_D_CMAP,
            linestyle="-",
            linewidth=2.4,
            label="calibrated 2D trend",
            zorder=4,
        )
    ax.axhline(0.0, color="0.84", linewidth=0.9)
    ax.set_xlim(0, 175)
    ax.set_xticks([0, 50, 100, 150])
    y_limit = symmetric_limit(
        np.concatenate(
            [
                y - ci,
                y + ci,
                one_d_points[col].to_numpy(dtype=float),
                two_d_points[col].to_numpy(dtype=float),
            ]
        ),
        pad=0.12,
        floor=10.0,
    )
    ax.set_ylim(-y_limit, y_limit)
    ax.set_title(title)
    ax.set_xlabel("Clamp magnitude (deg)")
    ax.set_ylabel(ylabel)
    if axis == "radial":
        ax.legend(handles=radial_dose_legend_handles(), frameon=False, loc="best")
    else:
        ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color=scheme_color(CLAMP_CMAP),
                    marker="o",
                    markerfacecolor="white",
                    linewidth=0,
                    label="human mean",
                ),
                Line2D(
                    [0],
                    [0],
                    color=scheme_color(ONE_D_CMAP),
                    linewidth=2.2,
                    linestyle="-",
                    label="calibrated 1D",
                ),
                Line2D(
                    [0],
                    [0],
                    color=scheme_color(TWO_D_CMAP),
                    linewidth=2.4,
                    linestyle="-",
                    label="calibrated 2D",
                ),
            ],
            frameon=False,
            loc="best",
        )
    if axis == "radial":
        ax.text(0.04, 0.95, stat_text, transform=ax.transAxes, ha="left", va="top", fontsize=7.0)
