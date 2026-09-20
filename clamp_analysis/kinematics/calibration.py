from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy import stats

from clamp_analysis.kinematics.colors import (
    CLAMP_CMAP,
    CLAMP_NORM,
)
from clamp_analysis.kinematics.geometry import CLAMPS
from clamp_analysis.kinematics.percepts import simulate_integrated_percept_predictors
from clamp_analysis.kinematics.updates import (
    EARLY_WINDOW,
    OUT_DIR,
    POSITION_TO_TARGET_PERCENT,
    SAMPLE_FRACTIONS,
    ci95,
    load_updates,
)

SMOOTH_BANDWIDTH_DEG = 16.0

CLAMP_GRID = np.linspace(min(CLAMPS), max(CLAMPS), 240)

CALIBRATED_SUMMARY_PATH = OUT_DIR / "best_first_bout_calibrated_model_comparison_summary.csv"

CALIBRATED_PATHS_PATH = OUT_DIR / "best_first_bout_calibrated_model_comparison_paths.csv"

plt.rcParams.update(
    {
        "font.size": 8.5,
        "axes.titlesize": 10.0,
        "axes.labelsize": 8.8,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.4,
        "legend.title_fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def format_p(value: float) -> str:
    if value < 0.001:
        return "p<.001"
    return f"p={value:.3f}".replace("0.", ".")


def symmetric_limit(values: np.ndarray, pad: float = 0.14, floor: float = 1.0) -> float:
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if len(valid) == 0:
        return floor
    return max(floor, float(np.nanmax(np.abs(valid))) * (1.0 + pad))


def calibrated_shared_gain_bias_readout(
    actual: pd.DataFrame,
    predictors: pd.DataFrame,
    forward_predictor: str | None,
    sideways_predictor: str | None,
    panel: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fit one model scale plus one offset for each plotted dimension.

    The simulator outputs are in model-state units, so a single scale factor is
    still needed to express predictions in percent-target-distance units. The
    radial and tangential offsets then translate the model path without allowing
    each dimension to choose its own slope.
    """
    merged = actual.merge(predictors, on="clamp_magnitude", how="inner", validate="one_to_one")
    raw_forward = (
        np.zeros(len(merged), dtype=float)
        if forward_predictor is None
        else merged[forward_predictor].to_numpy(dtype=float)
    )
    raw_sideways = (
        np.zeros(len(merged), dtype=float)
        if sideways_predictor is None
        else merged[sideways_predictor].to_numpy(dtype=float)
    )

    design_forward = np.column_stack([np.ones(len(merged)), np.zeros(len(merged)), raw_forward])
    design_sideways = np.column_stack([np.zeros(len(merged)), np.ones(len(merged)), raw_sideways])
    design = np.vstack([design_forward, design_sideways])
    y = np.concatenate(
        [
            merged["forward"].to_numpy(dtype=float),
            merged["sideways"].to_numpy(dtype=float),
        ]
    )
    forward_bias, sideways_bias, shared_gain = np.linalg.lstsq(design, y, rcond=None)[0]

    out = merged[["clamp_magnitude"]].copy()
    out["forward"] = float(forward_bias) + float(shared_gain) * raw_forward
    out["sideways"] = float(sideways_bias) + float(shared_gain) * raw_sideways
    out["ballistic_fraction"] = float(actual["ballistic_fraction"].iloc[0])
    out["sample_percent"] = 100.0 * float(actual["ballistic_fraction"].iloc[0])
    out["panel"] = panel
    coeffs = {
        "forward_bias": float(forward_bias),
        "sideways_bias": float(sideways_bias),
        "shared_gain": float(shared_gain),
    }
    return out, coeffs


def smooth_clamp_path(points: pd.DataFrame) -> pd.DataFrame:
    ordered = points.sort_values("clamp_magnitude").reset_index(drop=True)
    clamps = ordered["clamp_magnitude"].to_numpy(dtype=float)
    xy = ordered[["forward", "sideways"]].to_numpy(dtype=float)
    rows = []
    for clamp in CLAMP_GRID:
        weights = np.exp(-0.5 * ((clamps - clamp) / SMOOTH_BANDWIDTH_DEG) ** 2)
        weights /= max(float(weights.sum()), 1e-12)
        point = weights @ xy
        rows.append(
            {
                "clamp_magnitude": float(clamp),
                "forward": float(point[0]),
                "sideways": float(point[1]),
            }
        )
    return pd.DataFrame.from_records(rows)


def trend_metrics(human_path: pd.DataFrame, pred_path: pd.DataFrame) -> dict[str, float]:
    human = human_path[["forward", "sideways"]].to_numpy(dtype=float)
    pred = pred_path[["forward", "sideways"]].to_numpy(dtype=float)
    sse = float(np.sum((human - pred) ** 2))
    zero_denom = float(np.sum(human**2))
    centered_denom = float(np.sum((human - human.mean(axis=0)) ** 2))
    return {
        "trendline_rmse": float(np.sqrt(np.mean(np.sum((human - pred) ** 2, axis=1)))),
        "trendline_r2_from_zero": np.nan if zero_denom <= 1e-12 else 1.0 - sse / zero_denom,
        "trendline_centered_r2": np.nan if centered_denom <= 1e-12 else 1.0 - sse / centered_denom,
    }


def condition_centered_r2(actual: pd.DataFrame, pred: pd.DataFrame) -> float:
    merged = actual.merge(
        pred, on="clamp_magnitude", suffixes=("_human", "_pred"), validate="one_to_one"
    )
    human = merged[["forward_human", "sideways_human"]].to_numpy(dtype=float)
    yhat = merged[["forward_pred", "sideways_pred"]].to_numpy(dtype=float)
    denom = float(np.sum((human - human.mean(axis=0)) ** 2))
    if denom <= 1e-12:
        return np.nan
    return 1.0 - float(np.sum((human - yhat) ** 2)) / denom


def radial_slope(points: pd.DataFrame) -> dict[str, float]:
    fit = stats.linregress(
        points["clamp_magnitude"].to_numpy(dtype=float),
        points["forward"].to_numpy(dtype=float),
    )
    return {
        "radial_intercept_pct_target": float(fit.intercept),
        "radial_slope_pct_target_per_deg": float(fit.slope),
        "radial_slope_pct_target_per_100deg": float(100.0 * fit.slope),
        "radial_r": float(fit.rvalue),
        "radial_p": float(fit.pvalue),
    }


def actual_condition_means(
    updates: pd.DataFrame, fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = updates.loc[np.isclose(updates["ballistic_fraction"], float(fraction))].copy()
    cycle_means = (
        sub.groupby(
            ["ppid", "clamp_magnitude", "target_angles_degrees", "target_exposure_index"],
            as_index=False,
        )[["update_radial", "update_tangential"]]
        .mean(numeric_only=True)
        .reset_index(drop=True)
    )
    target_totals = (
        cycle_means.groupby(["ppid", "clamp_magnitude", "target_angles_degrees"], as_index=False)
        .agg(
            update_radial=("update_radial", "sum"),
            update_tangential=("update_tangential", "sum"),
            n_observed_transitions=("target_exposure_index", "nunique"),
        )
        .reset_index(drop=True)
    )
    per_participant = (
        target_totals.groupby(["ppid", "clamp_magnitude"], as_index=False)
        .agg(
            update_radial=("update_radial", "mean"),
            update_tangential=("update_tangential", "mean"),
            n_target_directions=("target_angles_degrees", "nunique"),
            mean_observed_transitions=("n_observed_transitions", "mean"),
            min_observed_transitions=("n_observed_transitions", "min"),
        )
        .reset_index(drop=True)
    )
    per_participant["forward"] = per_participant["update_radial"] * POSITION_TO_TARGET_PERCENT
    per_participant["sideways"] = per_participant["update_tangential"] * POSITION_TO_TARGET_PERCENT
    means = (
        per_participant.groupby("clamp_magnitude", as_index=False)
        .agg(
            forward=("forward", "mean"),
            forward_ci95=("forward", ci95),
            sideways=("sideways", "mean"),
            sideways_ci95=("sideways", ci95),
            n_participants=("ppid", "nunique"),
        )
        .sort_values("clamp_magnitude")
        .reset_index(drop=True)
    )
    means["ballistic_fraction"] = float(fraction)
    means["sample_percent"] = 100.0 * float(fraction)
    means["panel"] = "Human"
    return means, per_participant


def model_predictors(window: int = EARLY_WINDOW) -> pd.DataFrame:
    predictors = simulate_integrated_percept_predictors(window)
    return (
        predictors.loc[predictors["target_exposure_index"].between(1, window)]
        .groupby("clamp_magnitude", as_index=False)[
            ["scalar_update_tangential", "vector_update_radial", "vector_update_tangential"]
        ]
        .sum(numeric_only=True)
        .sort_values("clamp_magnitude")
        .reset_index(drop=True)
    )


def evaluate_calibrated_first_bout() -> tuple[pd.DataFrame, pd.DataFrame]:
    updates = load_updates()
    predictors = model_predictors(EARLY_WINDOW)
    summary_rows = []
    path_frames = []
    for fraction in SAMPLE_FRACTIONS:
        actual, _ = actual_condition_means(updates, float(fraction))
        one_d, one_d_coeffs = calibrated_shared_gain_bias_readout(
            actual,
            predictors,
            forward_predictor=None,
            sideways_predictor="scalar_update_tangential",
            panel="Calibrated 1D",
        )
        two_d, two_d_coeffs = calibrated_shared_gain_bias_readout(
            actual,
            predictors,
            forward_predictor="vector_update_radial",
            sideways_predictor="vector_update_tangential",
            panel="Calibrated 2D",
        )
        human_path = smooth_clamp_path(actual)
        one_d_path = smooth_clamp_path(one_d)
        two_d_path = smooth_clamp_path(two_d)

        for panel, frame in [
            ("Human", actual),
            ("Calibrated 1D", one_d),
            ("Calibrated 2D", two_d),
        ]:
            out = frame.copy()
            out["path_type"] = "condition_mean"
            out["panel"] = panel
            path_frames.append(out)
        for panel, frame in [
            ("Human", human_path),
            ("Calibrated 1D", one_d_path),
            ("Calibrated 2D", two_d_path),
        ]:
            out = frame.copy()
            out["ballistic_fraction"] = float(fraction)
            out["sample_percent"] = 100.0 * float(fraction)
            out["path_type"] = "smoothed_path"
            out["panel"] = panel
            path_frames.append(out)

        one_d_metrics = trend_metrics(human_path, one_d_path)
        two_d_metrics = trend_metrics(human_path, two_d_path)
        human_slope = radial_slope(actual)
        one_d_slope = radial_slope(one_d)
        two_d_slope = radial_slope(two_d)
        summary_rows.append(
            {
                "ballistic_fraction": float(fraction),
                "sample_percent": 100.0 * float(fraction),
                "mean_n_participants": float(actual["n_participants"].mean()),
                "one_d_forward_intercept": float(one_d_coeffs["forward_bias"]),
                "one_d_sideways_intercept": float(one_d_coeffs["sideways_bias"]),
                "one_d_sideways_slope": float(one_d_coeffs["shared_gain"]),
                "one_d_shared_gain": float(one_d_coeffs["shared_gain"]),
                "two_d_forward_intercept": float(two_d_coeffs["forward_bias"]),
                "two_d_forward_slope": float(two_d_coeffs["shared_gain"]),
                "two_d_sideways_intercept": float(two_d_coeffs["sideways_bias"]),
                "two_d_sideways_slope": float(two_d_coeffs["shared_gain"]),
                "two_d_shared_gain": float(two_d_coeffs["shared_gain"]),
                "one_d_condition_centered_r2": condition_centered_r2(actual, one_d),
                "two_d_condition_centered_r2": condition_centered_r2(actual, two_d),
                **{f"one_d_{key}": value for key, value in one_d_metrics.items()},
                **{f"two_d_{key}": value for key, value in two_d_metrics.items()},
                **{f"human_{key}": value for key, value in human_slope.items()},
                **{f"one_d_{key}": value for key, value in one_d_slope.items()},
                **{f"two_d_{key}": value for key, value in two_d_slope.items()},
            }
        )
    summary = pd.DataFrame.from_records(summary_rows)
    summary["trendline_rmse_improvement"] = (
        summary["one_d_trendline_rmse"] - summary["two_d_trendline_rmse"]
    )
    summary["trendline_r2_advantage"] = (
        summary["two_d_trendline_r2_from_zero"] - summary["one_d_trendline_r2_from_zero"]
    )
    summary["condition_r2_advantage"] = (
        summary["two_d_condition_centered_r2"] - summary["one_d_condition_centered_r2"]
    )
    paths = pd.concat(path_frames, axis=0, ignore_index=True)
    return summary, paths


def add_gradient_path(
    ax: plt.Axes,
    path: pd.DataFrame,
    linewidth: float,
    alpha: float,
    linestyle: str,
    zorder: int,
    cmap=CLAMP_CMAP,
    norm=CLAMP_NORM,
    label: str | None = None,
) -> LineCollection | None:
    xy = path[["forward", "sideways"]].to_numpy(dtype=float)
    if len(xy) < 2:
        return None
    segments = np.stack([xy[:-1], xy[1:]], axis=1)
    clamps = path["clamp_magnitude"].to_numpy(dtype=float)
    segment_clamps = 0.5 * (clamps[:-1] + clamps[1:])
    collection = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidths=linewidth,
        alpha=alpha,
        linestyles=linestyle,
        zorder=zorder,
        label=label,
    )
    collection.set_array(segment_clamps)
    collection.set_capstyle("round")
    collection.set_joinstyle("round")
    ax.add_collection(collection)
    return collection


def add_gradient_line_by_clamp(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    cmap,
    linewidth: float,
    alpha: float = 1.0,
    linestyle: str = "solid",
    zorder: int = 4,
    label: str | None = None,
    norm=CLAMP_NORM,
) -> LineCollection | None:
    xy = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    if len(xy) < 2:
        return None
    segments = np.stack([xy[:-1], xy[1:]], axis=1)
    segment_clamps = 0.5 * (xy[:-1, 0] + xy[1:, 0])
    collection = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidths=linewidth,
        alpha=alpha,
        linestyles=linestyle,
        zorder=zorder,
        label=label,
    )
    collection.set_array(segment_clamps)
    collection.set_capstyle("round")
    collection.set_joinstyle("round")
    ax.add_collection(collection)
    return collection
