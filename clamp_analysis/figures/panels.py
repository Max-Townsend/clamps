from __future__ import annotations

import zlib
from functools import lru_cache

import matplotlib

from clamp_analysis import paths

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
from scipy import stats

from clamp_analysis.figures.device_bias import fit_transformation_bias_patterns
from clamp_analysis.kinematics.radial import (
    focus_paths as focus_first_bout_radial_paths,
)
from clamp_analysis.kinematics.radial import (
    format_p as format_first_bout_p,
)
from clamp_analysis.kinematics.radial import (
    load_calibrated_tables as load_first_bout_radial_calibrated_tables,
)
from clamp_analysis.kinematics.radial import (
    ols_fit_with_mean_ci as first_bout_ols_fit_with_mean_ci,
)
from clamp_analysis.kinematics.radial import (
    plot_model_dose_response as plot_first_bout_radial_dose_response,
)
from clamp_analysis.kinematics.radial import (
    plot_model_geometry as plot_first_bout_calibrated_geometry,
)

TABLE_DIR = paths.RESULTS_DIR / "tables"

DATA_PATH = paths.DATA_PATH

DETAILS_PATH = paths.DETAILS_PATH

MEAN_TIMESERIES_FIT_DIR = paths.FITS_DIR / "mean_timeseries_shared_point_joint_device_motorvar"

POINTER_ORDER = ["Mouse", "Trackpad"]

DEVICE_COLORS = {
    "Mouse": "#7FB8F1",
    "Trackpad": "#F5B16D",
}

DEVICE_EDGE_COLORS = {
    "Mouse": "#4F8ED8",
    "Trackpad": "#E58B25",
}

TEAL_DARK = "#2AA79D"

TEAL_LIGHT = "#C8F0EB"

MAGENTA_DARK = "#B857A8"

MAGENTA_LIGHT = "#F3D1EC"

BIAS_ORDER = ["Incorrect", "Near-zero", "Correct"]

BIAS_COLORS = {
    "Incorrect": MAGENTA_DARK,
    "Near-zero": "#9A9A9A",
    "Correct": TEAL_DARK,
}

REPRESENTATIVE_COLORS = {
    4: TEAL_DARK,
    40: "#6BCB9A",
    125: "#D9821B",
    170: MAGENTA_DARK,
}

REPRESENTATIVE_LABELS = {
    4: "Small (4$^\\circ$)",
    40: "Peak (40$^\\circ$)",
    125: "Large-descending (125$^\\circ$)",
    170: "Large-bottomed (170$^\\circ$)",
}

LITERATURE_COLORS = {
    "Kim, 2018 exp1": "#C78CC3",
    "Kim, 2018 exp2": "#79D6CA",
    "Morehead, 2017": "#DDB85A",
    "Zhang et al., 2024 exp2": "#E47B7B",
    "Morehead multi-clamp": "#5A84B4",
}

MEAN_TIMESERIES_MODEL_ORDER = [
    "MeanTimeseriesBaseBCC",
    "MeanTimeseriesCausalInf",
    "MeanTimeseriesVectorBCC",
]

MEAN_TIMESERIES_MODEL_LABELS = {
    "MeanTimeseriesBaseBCC": "Base BCC",
    "MeanTimeseriesCausalInf": "Causal inference",
    "MeanTimeseriesVectorBCC": "Vector BCC",
}

MEAN_TIMESERIES_MODEL_COLORS = {
    "MeanTimeseriesBaseBCC": "#4C78A8",
    "MeanTimeseriesCausalInf": "#54A24B",
    "MeanTimeseriesVectorBCC": "#111111",
}

DATASET_LABELS = {
    "mouse": "Mouse-only",
    "trackpad": "Trackpad-only",
}

MODEL_FIT_DATASET_LABELS = {
    "joint_devices": "Mouse + Trackpad",
}

COMBINED_MODEL_DATASET_SLUG = "devices_combined"

COMBINED_MODEL_SOURCE_DATASETS = ("mouse", "trackpad")

MODEL_DASH = (0, (6, 3))

FIRST_BOUT_HUMAN_COLOR = MAGENTA_DARK

FIRST_BOUT_ONE_D_COLOR = DEVICE_EDGE_COLORS["Trackpad"]

FIRST_BOUT_TWO_D_COLOR = DEVICE_EDGE_COLORS["Mouse"]

FIRST_BOUT_CLAMP_NORM = mcolors.PowerNorm(gamma=0.55, vmin=0.0, vmax=170.0, clip=True)

FIRST_BOUT_HUMAN_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "paper_first_bout_human",
    ["#2A1824", FIRST_BOUT_HUMAN_COLOR, "#F7DDED"],
)

FIRST_BOUT_ONE_D_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "paper_first_bout_1d",
    ["#402000", FIRST_BOUT_ONE_D_COLOR, "#FFE3C2"],
)

FIRST_BOUT_TWO_D_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "paper_first_bout_2d",
    ["#142B45", FIRST_BOUT_TWO_D_COLOR, "#DCEBFF"],
)

FIRST_BOUT_DOT_MARKERSIZE = 5.4

FIRST_BOUT_DOT_EDGEWIDTH = 0.45

RNG_SEED = 20260421

BOOTSTRAP_DRAWS = 2000

BASE_PAPER_FONT = {
    "title": 16.0,
    "axis_label": 15.0,
    "tick": 13.0,
    "legend": 13.0,
    "panel_label": 16.0,
}

FONT_SCALE = 1.2

PAPER_FONT = {key: value * FONT_SCALE for key, value in BASE_PAPER_FONT.items()}

SUMMARY_MARKER_SIZE = 78

DEVICE_MARKER_SIZE = 70

TR_MARKER_SIZE = 64

RAW_POINT_SIZE = 10

INDIVIDUAL_POINT_SIZE = 14

NO_FEEDBACK_SPAN = (-20.5, -10.5)

NO_FEEDBACK_SHADE = "0.92"

CLAMP_XLIM = (0.0, 170.0)

CLAMP_XTICKS = np.arange(0, 171, 20)

REPRESENTATIVE_CLAMPS = (4, 40, 125, 170)

LATE_ADAPTATION_CYCLE_RANGE = (220.0, 239.0)

STL_EARLY_CYCLE_RANGE = (0.0, 9.0)

STL_WASHOUT_CYCLE_RANGE = (240.0, 249.0)


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(length=4.0, width=1.0)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=PAPER_FONT["panel_label"],
        fontweight="bold",
    )


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts).encode("utf-8")
    return RNG_SEED + zlib.crc32(text)


def mean_ci(values: pd.Series | np.ndarray, ci: float = 0.95) -> dict[str, float]:
    values = pd.Series(values, dtype=float).dropna()
    n = len(values)
    if n == 0:
        return {"n": 0, "center": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    center = float(values.mean())
    if n == 1:
        return {"n": 1, "center": center, "ci_low": center, "ci_high": center}
    sem = float(stats.sem(values, nan_policy="omit"))
    half = float(stats.t.ppf((1.0 + ci) / 2.0, df=n - 1) * sem)
    return {"n": n, "center": center, "ci_low": center - half, "ci_high": center + half}


def median_bootstrap_ci(
    values: pd.Series | np.ndarray,
    *,
    n_boot: int = BOOTSTRAP_DRAWS,
    ci: float = 0.95,
    seed: int,
) -> dict[str, float]:
    values = pd.Series(values, dtype=float).dropna().to_numpy(dtype=float)
    n = len(values)
    if n == 0:
        return {"n": 0, "center": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    center = float(np.median(values))
    if n == 1:
        return {"n": 1, "center": center, "ci_low": center, "ci_high": center}
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot[idx] = float(np.median(sample))
    alpha = (1.0 - ci) / 2.0
    return {
        "n": n,
        "center": center,
        "ci_low": float(np.quantile(boot, alpha)),
        "ci_high": float(np.quantile(boot, 1.0 - alpha)),
    }


def finite_values(*arrays: pd.Series | np.ndarray | list[float] | None) -> np.ndarray:
    collected: list[np.ndarray] = []
    for values in arrays:
        if values is None:
            continue
        array = np.asarray(values, dtype=float).ravel()
        array = array[np.isfinite(array)]
        if array.size:
            collected.append(array)
    if not collected:
        return np.array([], dtype=float)
    return np.concatenate(collected)


def padded_ylim(
    *arrays: pd.Series | np.ndarray | list[float] | None,
    include_zero: bool = False,
    pad_frac: float = 0.08,
    min_pad: float = 1.0,
) -> tuple[float, float] | None:
    values = finite_values(*arrays)
    if include_zero:
        zero = np.array([0.0], dtype=float)
        values = zero if values.size == 0 else np.concatenate([values, zero])
    if values.size == 0:
        return None

    lower = float(values.min())
    upper = float(values.max())
    span = upper - lower
    if span <= 0:
        pad = max(min_pad, 0.08 * max(abs(lower), 1.0))
    else:
        pad = max(min_pad, pad_frac * span)
    return lower - pad, upper + pad


def apply_y_limits(ax: plt.Axes, ylim: tuple[float, float] | None) -> None:
    if ylim is None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        return
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))


def darken_color(color: str, factor: float = 0.68) -> tuple[float, float, float]:
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    return tuple(np.clip(rgb * factor, 0.0, 1.0))


def shade_no_feedback_region(ax: plt.Axes) -> None:
    ax.axvspan(NO_FEEDBACK_SPAN[0], NO_FEEDBACK_SPAN[1], color=NO_FEEDBACK_SHADE, zorder=0)


def compute_fit_metrics(
    human_summary_df: pd.DataFrame,
    model_summary_df: pd.DataFrame,
    *,
    key_cols: tuple[str, ...] = ("clamp_magnitude",),
    human_col: str = "center",
    model_col: str = "predicted_value",
) -> dict[str, float]:
    merge_cols = list(key_cols)
    human = human_summary_df.loc[:, merge_cols + [human_col]].copy()
    model = model_summary_df.loc[:, merge_cols + [model_col]].copy()
    merged = human.merge(model, on=merge_cols, how="inner").dropna()
    if merged.empty:
        return {"n": 0.0, "r2": np.nan, "rmse": np.nan}

    observed = pd.to_numeric(merged[human_col], errors="coerce").to_numpy(dtype=float)
    predicted = pd.to_numeric(merged[model_col], errors="coerce").to_numpy(dtype=float)
    finite_mask = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[finite_mask]
    predicted = predicted[finite_mask]
    if observed.size == 0:
        return {"n": 0.0, "r2": np.nan, "rmse": np.nan}

    rmse = float(np.sqrt(np.mean((predicted - observed) ** 2)))
    ss_res = float(np.sum((observed - predicted) ** 2))
    ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
    r2 = np.nan if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)
    return {"n": float(observed.size), "r2": r2, "rmse": rmse}


def format_fit_metrics_line(label: str, metrics: dict[str, float]) -> str:
    r2 = metrics.get("r2", np.nan)
    rmse = metrics.get("rmse", np.nan)
    r2_text = "nan" if not np.isfinite(r2) else f"{r2:.2f}"
    rmse_text = "nan" if not np.isfinite(rmse) else f"{rmse:.2f}"
    return f"{label}: $R^2$={r2_text}, RMSE={rmse_text}"


def add_metrics_box(
    ax: plt.Axes,
    lines: list[str] | None,
    *,
    loc: str = "upper left",
    fontsize: float | None = None,
) -> None:
    if not lines:
        return
    x = 0.02 if "left" in loc else 0.98
    y = 0.98 if "upper" in loc else 0.02
    ha = "left" if "left" in loc else "right"
    va = "top" if "upper" in loc else "bottom"
    ax.text(
        x,
        y,
        "\n".join(lines),
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=PAPER_FONT["legend"] * 0.70 if fontsize is None else fontsize,
        bbox={
            "facecolor": "white",
            "alpha": 0.84,
            "edgecolor": "0.82",
            "boxstyle": "round,pad=0.25",
        },
        zorder=10,
    )


def summarize_by_clamp(
    df: pd.DataFrame,
    *,
    value_col: str = "value",
    stat: str = "mean",
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for clamp_magnitude, g in df.groupby("clamp_magnitude", observed=True):
        if stat == "median":
            row = median_bootstrap_ci(
                g[value_col], seed=stable_seed(value_col, "pooled", clamp_magnitude)
            )
        else:
            row = mean_ci(g[value_col])
        row["clamp_magnitude"] = int(clamp_magnitude)
        rows.append(row)
    return pd.DataFrame.from_records(rows).sort_values("clamp_magnitude").reset_index(drop=True)


def summarize_by_pointer(
    df: pd.DataFrame,
    *,
    value_col: str = "value",
    stat: str = "mean",
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (pointer, clamp_magnitude), g in df.groupby(["pointer", "clamp_magnitude"], observed=True):
        if stat == "median":
            row = median_bootstrap_ci(
                g[value_col], seed=stable_seed(value_col, pointer, clamp_magnitude)
            )
        else:
            row = mean_ci(g[value_col])
        row["pointer"] = pointer
        row["clamp_magnitude"] = int(clamp_magnitude)
        rows.append(row)
    return (
        pd.DataFrame.from_records(rows)
        .sort_values(["pointer", "clamp_magnitude"])
        .reset_index(drop=True)
    )


def add_zero_anchor(
    df: pd.DataFrame,
    *,
    x_col: str = "clamp_magnitude",
    y_col: str = "predicted_value",
    x_value: float = 0.0,
    y_value: float = 0.0,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame([{x_col: x_value, y_col: y_value}])
    anchored = df.copy()
    if np.isclose(pd.to_numeric(anchored[x_col], errors="coerce"), x_value).any():
        return anchored.sort_values(x_col).reset_index(drop=True)
    anchor = pd.DataFrame([{x_col: x_value, y_col: y_value}])
    return (
        pd.concat([anchor, anchored], axis=0, ignore_index=True)
        .sort_values(x_col)
        .reset_index(drop=True)
    )


def _coerce_mean_timeseries_prediction_df(df: pd.DataFrame) -> pd.DataFrame:
    coerced = df.copy()
    numeric_cols = [
        "clamp_magnitude",
        "cycle_wrt_clamp",
        "sequence_trial",
        "trial_num_in_block",
        "predicted_late_mean",
    ]
    for col in numeric_cols:
        if col in coerced.columns:
            coerced[col] = pd.to_numeric(coerced[col], errors="coerce")
    if "condition" in coerced.columns:
        coerced["condition"] = coerced["condition"].astype(str)
    return coerced


@lru_cache(maxsize=1)
def _mean_timeseries_device_weights() -> dict[str, float]:
    overview_path = MEAN_TIMESERIES_FIT_DIR / "dataset_overview.csv"
    if overview_path.exists():
        overview = pd.read_csv(overview_path)
        if {"subsetSlug", "n_participants"}.issubset(overview.columns):
            weights = {
                str(row["subsetSlug"]): float(row["n_participants"])
                for _, row in overview.iterrows()
                if str(row["subsetSlug"]) in COMBINED_MODEL_SOURCE_DATASETS
                and np.isfinite(float(row["n_participants"]))
                and float(row["n_participants"]) > 0.0
            }
            if weights:
                return weights

    weights = {}
    for dataset_slug in COMBINED_MODEL_SOURCE_DATASETS:
        summary_path = (
            MEAN_TIMESERIES_FIT_DIR
            / "data_summaries"
            / f"{dataset_slug}_mean_timeseries_summary.csv"
        )
        if not summary_path.exists():
            continue
        summary = pd.read_csv(summary_path, usecols=lambda col: col in {"n_obs"})
        if "n_obs" in summary.columns:
            weight = float(
                pd.to_numeric(summary["n_obs"], errors="coerce").fillna(0.0).clip(lower=0.0).sum()
            )
            if weight > 0.0:
                weights[dataset_slug] = weight
    if weights:
        return weights
    return {dataset_slug: 1.0 for dataset_slug in COMBINED_MODEL_SOURCE_DATASETS}


def _combine_mean_timeseries_prediction_dfs(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    key_cols = [
        col
        for col in [
            "clamp_magnitude",
            "condition",
            "clamped",
            "cycle_wrt_clamp",
            "sequence_trial",
            "trial_num_in_block",
        ]
        if col in combined.columns
    ]
    numeric_mean_cols = [
        col
        for col in [
            "predicted_late_mean",
            "model_late_mean",
            "late_mean",
            "mean_hand_angle",
            "mean_sd",
            "mean_sem",
            "obs_var",
        ]
        if col in combined.columns
    ]

    working = combined[key_cols].copy()
    source_col = None
    for candidate in ["_source_dataset_slug", "subsetSlug", "dataset_slug"]:
        if candidate in combined.columns:
            values = set(combined[candidate].dropna().astype(str))
            if values.intersection(COMBINED_MODEL_SOURCE_DATASETS):
                source_col = candidate
                break
    if source_col is not None:
        device_weights = _mean_timeseries_device_weights()
        working["_weight"] = (
            combined[source_col]
            .astype(str)
            .map(device_weights)
            .fillna(1.0)
            .astype(float)
            .clip(lower=0.0)
        )
    elif "n_obs" in combined.columns:
        working["_weight"] = (
            pd.to_numeric(combined["n_obs"], errors="coerce").fillna(0.0).clip(lower=0.0)
        )
    else:
        working["_weight"] = 1.0
    for col in numeric_mean_cols:
        values = pd.to_numeric(combined[col], errors="coerce")
        weights = working["_weight"].where(values.notna(), 0.0)
        working[f"_weighted_{col}"] = values.fillna(0.0) * weights
        working[f"_weight_{col}"] = weights

    grouped = (
        working.groupby(key_cols, dropna=False, observed=True, sort=False)
        .sum(numeric_only=True)
        .reset_index()
    )
    out = grouped[key_cols].copy()
    out["n_obs"] = grouped["_weight"].round().astype(int)
    for col in numeric_mean_cols:
        denom = grouped[f"_weight_{col}"].replace(0.0, np.nan)
        out[col] = grouped[f"_weighted_{col}"] / denom

    out["ppid"] = "device_combined_model_mean"
    out["group"] = 0
    out["target_angles_degrees"] = 0.0
    out["target_clamp_sign"] = 1.0
    out["clamp_sign"] = 1.0
    out["rotation_model"] = np.where(
        out.get("clamped", False).astype(bool),
        -pd.to_numeric(out["clamp_magnitude"], errors="coerce"),
        0.0,
    )

    sort_cols = [col for col in ["clamp_magnitude", "sequence_trial"] if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols)
    return _coerce_mean_timeseries_prediction_df(out.reset_index(drop=True))


@lru_cache(maxsize=16)
def _cached_mean_timeseries_prediction_df(dataset_slug: str, model_name: str) -> pd.DataFrame:
    candidate_paths = [
        MEAN_TIMESERIES_FIT_DIR
        / dataset_slug
        / model_name
        / f"{model_name}_posterior_predictions.csv",
    ]
    for path in candidate_paths:
        if path.exists():
            return _coerce_mean_timeseries_prediction_df(pd.read_csv(path))
    if dataset_slug in {COMBINED_MODEL_DATASET_SLUG, "pooled"}:
        frames = []
        for source_slug in COMBINED_MODEL_SOURCE_DATASETS:
            frame = _cached_mean_timeseries_prediction_df(source_slug, model_name).copy()
            frame["_source_dataset_slug"] = source_slug
            frames.append(frame)
        return _combine_mean_timeseries_prediction_dfs(frames)
    path_text = "\n".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(
        f"Missing posterior predictions for {dataset_slug}/{model_name}.\nTried:\n{path_text}"
    )


def load_mean_timeseries_prediction_df(dataset_slug: str, model_name: str) -> pd.DataFrame:
    return _cached_mean_timeseries_prediction_df(dataset_slug, model_name).copy()


def summarize_model_fit_window(
    model_name: str,
    dataset_slug: str,
    *,
    condition: str,
    cycle_range: tuple[float, float],
    use_same_target_delta: bool = False,
) -> pd.DataFrame:
    df = load_mean_timeseries_prediction_df(dataset_slug, model_name)
    if use_same_target_delta:
        df = df.sort_values(["clamp_magnitude", "sequence_trial"]).reset_index(drop=True)
        df["predicted_value"] = df.groupby("clamp_magnitude", observed=True)[
            "predicted_late_mean"
        ].diff()
    else:
        df["predicted_value"] = pd.to_numeric(df["predicted_late_mean"], errors="coerce")

    subset = df.loc[
        df["condition"].eq(condition)
        & df["cycle_wrt_clamp"].between(cycle_range[0], cycle_range[1])
        & df["predicted_value"].notna(),
        ["clamp_magnitude", "predicted_value"],
    ].copy()

    rows: list[dict[str, float | int]] = []
    for clamp_magnitude, g in subset.groupby("clamp_magnitude", observed=True):
        rows.append(
            {
                "clamp_magnitude": int(round(float(clamp_magnitude))),
                "predicted_value": float(g["predicted_value"].mean()),
            }
        )
    out = pd.DataFrame.from_records(rows).sort_values("clamp_magnitude").reset_index(drop=True)
    return add_zero_anchor(out)


@lru_cache(maxsize=1)
def _cached_representative_learning_curve_summary() -> pd.DataFrame:
    trials = pd.read_csv(
        DATA_PATH,
        usecols=["ppid", "condition", "cycle_num", "clamp_magnitude", "recentred_hand_angle"],
        dtype={"ppid": str},
    )
    trials["clamp_magnitude"] = pd.to_numeric(trials["clamp_magnitude"], errors="coerce")
    trials["cycle_num"] = pd.to_numeric(trials["cycle_num"], errors="coerce")
    trials["recentred_hand_angle"] = pd.to_numeric(trials["recentred_hand_angle"], errors="coerce")
    trials["cycle_wrt_clamp"] = trials["cycle_num"] - 21
    trials = trials.loc[
        trials["condition"].isin(["FeedbackBaseline", "Clamp", "FeedbackWashout"])
        & trials["clamp_magnitude"].isin(REPRESENTATIVE_CLAMPS)
    ].copy()

    subject_curves = (
        trials.groupby(["ppid", "clamp_magnitude", "cycle_wrt_clamp"], as_index=False)[
            "recentred_hand_angle"
        ]
        .mean()
        .dropna(subset=["recentred_hand_angle"])
    )

    rows: list[dict[str, float | int]] = []
    for (clamp_magnitude, cycle_wrt_clamp), g in subject_curves.groupby(
        ["clamp_magnitude", "cycle_wrt_clamp"],
        observed=True,
    ):
        row = mean_ci(g["recentred_hand_angle"])
        row["clamp_magnitude"] = int(round(float(clamp_magnitude)))
        row["cycle_wrt_clamp"] = float(cycle_wrt_clamp)
        row["mean"] = row.pop("center")
        rows.append(row)

    return (
        pd.DataFrame.from_records(rows)
        .sort_values(["clamp_magnitude", "cycle_wrt_clamp"])
        .reset_index(drop=True)
    )


def load_representative_learning_curve_summary() -> pd.DataFrame:
    return _cached_representative_learning_curve_summary().copy()


@lru_cache(maxsize=1)
def _cached_mean_timeseries_fit_summary_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_slug in MODEL_FIT_DATASET_LABELS:
        for model_name in MEAN_TIMESERIES_MODEL_ORDER:
            path = (
                MEAN_TIMESERIES_FIT_DIR
                / dataset_slug
                / model_name
                / f"{model_name}_fit_summary.csv"
            )
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if df.empty:
                continue
            row = df.iloc[0].to_dict()
            row["dataset_slug"] = dataset_slug
            row["dataset_label"] = MODEL_FIT_DATASET_LABELS[dataset_slug]
            row["model_name"] = model_name
            row["model_label"] = MEAN_TIMESERIES_MODEL_LABELS[model_name]
            rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=["dataset_slug", "dataset_label", "model_name", "model_label", "BIC"]
        )
    return pd.DataFrame.from_records(rows)


def load_mean_timeseries_fit_summary_table() -> pd.DataFrame:
    return _cached_mean_timeseries_fit_summary_table().copy()


@lru_cache(maxsize=1)
def _cached_mean_timeseries_dataset_bic_table() -> pd.DataFrame:
    fit_summary_df = load_mean_timeseries_fit_summary_table()
    if fit_summary_df.empty or "BIC" not in fit_summary_df.columns:
        return pd.DataFrame()

    out = fit_summary_df.loc[
        :,
        ["dataset_slug", "dataset_label", "model_name", "model_label", "BIC"],
    ].copy()
    out["bic"] = pd.to_numeric(out["BIC"], errors="coerce")
    out = out.dropna(subset=["bic"]).reset_index(drop=True)
    out["delta_bic"] = out["bic"] - out.groupby("dataset_slug", observed=True)["bic"].transform(
        "min"
    )
    out["delta_bic_thousands"] = out["delta_bic"] / 1000.0
    out["is_best"] = pd.to_numeric(out["delta_bic"], errors="coerce").abs() < 1e-9
    return out.sort_values(["dataset_slug", "model_name"]).reset_index(drop=True)


def load_mean_timeseries_dataset_bic_table() -> pd.DataFrame:
    return _cached_mean_timeseries_dataset_bic_table().copy()


@lru_cache(maxsize=1)
def _cached_mean_timeseries_parameter_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_slug in MODEL_FIT_DATASET_LABELS:
        for model_name in MEAN_TIMESERIES_MODEL_ORDER:
            path = (
                MEAN_TIMESERIES_FIT_DIR
                / dataset_slug
                / model_name
                / f"{model_name}_parameter_summary.csv"
            )
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if df.empty:
                continue
            working = df.copy()
            working["dataset_slug"] = dataset_slug
            working["dataset_label"] = MODEL_FIT_DATASET_LABELS[dataset_slug]
            working["model_name"] = model_name
            working["model_label"] = MEAN_TIMESERIES_MODEL_LABELS[model_name]
            rows.append(working)
    if not rows:
        return pd.DataFrame(
            columns=[
                "scope",
                "parameter",
                "mean",
                "sd",
                "fixed",
                "datasetSlug",
                "datasetLabel",
                "dataset_slug",
                "dataset_label",
                "model_name",
                "model_label",
            ]
        )
    return pd.concat(rows, axis=0, ignore_index=True)


def load_mean_timeseries_parameter_table() -> pd.DataFrame:
    return _cached_mean_timeseries_parameter_table().copy()


def _load_subject_table(name: str) -> pd.DataFrame:
    path = TABLE_DIR / name
    df = pd.read_csv(path)
    if "ppid" in df.columns:
        df["ppid"] = df["ppid"].astype(str)
    if "pointer" in df.columns:
        df = df.loc[df["pointer"].isin(POINTER_ORDER)].copy()
    if "clamp_magnitude" in df.columns:
        df["clamp_magnitude"] = pd.to_numeric(df["clamp_magnitude"], errors="coerce").astype(int)
    return df


def build_early_stl_variability_subject() -> pd.DataFrame:
    trials = pd.read_csv(
        DATA_PATH,
        usecols=[
            "ppid",
            "condition",
            "trial_num",
            "cycle_num",
            "target_angles_degrees",
            "recentred_hand_angle",
            "clamp_magnitude",
            "group",
        ],
        dtype={"ppid": str},
    )
    trials["clamp_magnitude"] = pd.to_numeric(trials["clamp_magnitude"], errors="coerce")
    trials["group"] = pd.to_numeric(trials["group"], errors="coerce")
    trials["cycle_num"] = pd.to_numeric(trials["cycle_num"], errors="coerce")
    trials["trial_num"] = pd.to_numeric(trials["trial_num"], errors="coerce")
    trials["target_angles_degrees"] = pd.to_numeric(
        trials["target_angles_degrees"], errors="coerce"
    )
    # recentred_hand_angle is task-aligned in this dataset; it is not
    # participant-baseline-corrected.
    trials["hand_angle"] = pd.to_numeric(trials["recentred_hand_angle"], errors="coerce")
    trials["cycle_wrt_clamp"] = trials["cycle_num"] - 21
    trials = trials.sort_values(["ppid", "trial_num"]).reset_index(drop=True)

    details = pd.read_csv(DETAILS_PATH, usecols=["pointer", "ppid_session_dataname"])
    details["ppid"] = (
        details["ppid_session_dataname"]
        .astype(str)
        .str.replace(r"_s\d+_participant_details$", "", regex=True)
    )
    details = details[["ppid", "pointer"]].drop_duplicates()
    trials = trials.merge(details, on="ppid", how="left")
    trials = trials.loc[trials["pointer"].isin(POINTER_ORDER)].copy()

    delta_df = trials.sort_values(["ppid", "target_angles_degrees", "trial_num"]).copy()
    delta_df["same_target_delta"] = delta_df.groupby(
        ["ppid", "target_angles_degrees"], observed=True
    )["hand_angle"].diff()

    raw = delta_df.loc[
        delta_df["condition"].eq("Clamp")
        & delta_df["cycle_wrt_clamp"].between(0, 9)
        & delta_df["clamp_magnitude"].notna()
        & delta_df["group"].notna(),
        ["ppid", "clamp_magnitude", "group", "pointer", "same_target_delta"],
    ].dropna(subset=["same_target_delta"])

    subject = (
        raw.groupby(["ppid", "clamp_magnitude", "group", "pointer"], as_index=False, observed=True)[
            "same_target_delta"
        ]
        .agg(["std", "count"])
        .reset_index()
        .rename(columns={"std": "value", "count": "n_trials"})
    )
    subject = subject.loc[subject["n_trials"] >= 3].dropna(subset=["value"]).copy()
    subject["clamp_magnitude"] = subject["clamp_magnitude"].astype(int)
    subject["group"] = subject["group"].astype(int)
    return subject.sort_values(["pointer", "clamp_magnitude", "ppid"]).reset_index(drop=True)


def plot_literature_panel(
    ax: plt.Axes,
    *,
    df: pd.DataFrame | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    if df is None:
        df = pd.read_csv(TABLE_DIR / "panelB_literature_plus_exp2_summary.csv")
    df["perturbation_size_deg"] = pd.to_numeric(df["perturbation_size_deg"], errors="coerce")
    df["adaptation_extent_deg"] = pd.to_numeric(df["adaptation_extent_deg"], errors="coerce")
    df["series_order"] = pd.to_numeric(df["series_order"], errors="coerce")
    df = df.sort_values(["series_order", "perturbation_size_deg"]).reset_index(drop=True)

    for series_label, sdf in df.groupby("series_label", sort=False):
        color = LITERATURE_COLORS.get(series_label, "#666666")
        x = sdf["perturbation_size_deg"].to_numpy(dtype=float)
        y = sdf["adaptation_extent_deg"].to_numpy(dtype=float)
        ax.plot(x, y, color=color, linewidth=2.3, alpha=0.90, zorder=2)
        ax.scatter(
            x,
            y,
            s=170,
            color=color,
            alpha=0.56,
            edgecolor=color,
            linewidth=2.2,
            zorder=3,
            label=series_label,
        )

    ax.set_xlim(0, 112)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    apply_y_limits(ax, ylim)
    ax.set_title("", pad=8)
    ax.set_xlabel("Clamp magnitude ($^\\circ$)")
    ax.set_ylabel("Adaptation extent ($^\\circ$)")
    ax.grid(axis="y", alpha=0.16)
    ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=PAPER_FONT["legend"],
        handlelength=2.4,
        labelspacing=0.3,
    )
    style_axes(ax)


def plot_pooled_summary(
    ax: plt.Axes,
    subject_df: pd.DataFrame,
    *,
    title: str,
    ylabel: str,
    color: str,
    ylim: tuple[float, float] | None = None,
    yticks: np.ndarray | None = None,
    stat: str = "mean",
    overlay_lines: list[dict[str, object]] | None = None,
    human_legend_label: str | None = None,
    legend_loc: str = "upper right",
    metrics_lines: list[str] | None = None,
    metrics_loc: str = "upper left",
) -> None:
    summary = summarize_by_clamp(subject_df, stat=stat)
    x = summary["clamp_magnitude"].to_numpy(dtype=float)
    y = summary["center"].to_numpy(dtype=float)
    lo = summary["ci_low"].to_numpy(dtype=float)
    hi = summary["ci_high"].to_numpy(dtype=float)

    ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0, zorder=1)

    legend_handles: list[Line2D] = []
    if human_legend_label is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=2.3,
                marker="o",
                markersize=6.5,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=1.0,
                label=human_legend_label,
            )
        )
    if overlay_lines:
        for overlay in overlay_lines:
            summary_df = add_zero_anchor(pd.DataFrame(overlay["summary_df"]).copy())
            line_color = str(overlay.get("color", "#111111"))
            line_width = float(overlay.get("linewidth", 2.6))
            line_alpha = float(overlay.get("alpha", 0.95))
            line_style = overlay.get("linestyle", MODEL_DASH)
            ax.plot(
                pd.to_numeric(summary_df["clamp_magnitude"], errors="coerce"),
                pd.to_numeric(summary_df["predicted_value"], errors="coerce"),
                color=line_color,
                linewidth=line_width,
                alpha=line_alpha,
                linestyle=line_style,
                zorder=float(overlay.get("zorder", 2.0)),
            )
            label = overlay.get("label")
            if label is not None:
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=line_color,
                        linewidth=line_width,
                        alpha=line_alpha,
                        linestyle=line_style,
                        label=str(label),
                    )
                )

    ax.plot(x, y, color=color, linewidth=2.3, zorder=3)
    ax.scatter(x, y, s=SUMMARY_MARKER_SIZE, color=color, edgecolor="white", linewidth=1.0, zorder=4)

    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    ax.set_xlim(*CLAMP_XLIM)
    ax.set_xticks(CLAMP_XTICKS)
    if ylim is not None:
        apply_y_limits(ax, ylim)
    else:
        apply_y_limits(ax, None)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.set_title(title, pad=8)
    ax.set_xlabel("Clamp magnitude ($^\\circ$)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.16)
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            frameon=False,
            loc=legend_loc,
            fontsize=PAPER_FONT["legend"] * 0.78,
            handlelength=2.8,
            labelspacing=0.3,
        )
    add_metrics_box(ax, metrics_lines, loc=metrics_loc)
    style_axes(ax)


def plot_representative_learning_curves(
    ax: plt.Axes,
    *,
    model_df: pd.DataFrame | None = None,
    model_label: str = "Vector BCC",
) -> None:
    df = load_representative_learning_curve_summary()
    df["clamp_magnitude"] = pd.to_numeric(df["clamp_magnitude"], errors="coerce").astype(int)
    df["cycle_wrt_clamp"] = pd.to_numeric(df["cycle_wrt_clamp"], errors="coerce")
    for col in ["mean", "ci_low", "ci_high"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    model_curves = None
    if model_df is not None:
        model_curves = model_df.copy()
        model_curves["clamp_magnitude"] = pd.to_numeric(
            model_curves["clamp_magnitude"], errors="coerce"
        ).astype(int)
        model_curves["cycle_wrt_clamp"] = pd.to_numeric(
            model_curves["cycle_wrt_clamp"], errors="coerce"
        )
        model_curves["predicted_late_mean"] = pd.to_numeric(
            model_curves["predicted_late_mean"], errors="coerce"
        )
        model_curves = model_curves.loc[
            model_curves["clamp_magnitude"].isin(REPRESENTATIVE_CLAMPS)
        ].copy()

    y_min = np.inf
    y_max = -np.inf
    for clamp_magnitude, sdf in df.groupby("clamp_magnitude", observed=True):
        sdf = sdf.sort_values("cycle_wrt_clamp")
        color = REPRESENTATIVE_COLORS.get(int(clamp_magnitude), TEAL_DARK)
        model_color = darken_color(color, factor=0.62)
        human_line_alpha = 0.82 if int(clamp_magnitude) == 125 else 0.68
        human_line_width = 2.5 if int(clamp_magnitude) == 125 else 2.3
        ax.fill_between(
            sdf["cycle_wrt_clamp"].to_numpy(dtype=float),
            sdf["ci_low"].to_numpy(dtype=float),
            sdf["ci_high"].to_numpy(dtype=float),
            color=color,
            alpha=0.22,
            linewidth=0,
            zorder=1,
        )
        y_min = min(y_min, float(sdf["ci_low"].min()))
        y_max = max(y_max, float(sdf["ci_high"].max()))

        if model_curves is not None:
            mdf = model_curves.loc[
                model_curves["clamp_magnitude"].eq(int(clamp_magnitude))
            ].sort_values("cycle_wrt_clamp")
            if not mdf.empty:
                ax.plot(
                    mdf["cycle_wrt_clamp"].to_numpy(dtype=float),
                    mdf["predicted_late_mean"].to_numpy(dtype=float),
                    color=model_color,
                    linewidth=2.8,
                    alpha=0.98,
                    linestyle=MODEL_DASH,
                    zorder=2.5,
                )
                y_min = min(y_min, float(mdf["predicted_late_mean"].min()))
                y_max = max(y_max, float(mdf["predicted_late_mean"].max()))

        ax.plot(
            sdf["cycle_wrt_clamp"].to_numpy(dtype=float),
            sdf["mean"].to_numpy(dtype=float),
            color=color,
            linewidth=human_line_width,
            alpha=human_line_alpha,
            label=REPRESENTATIVE_LABELS.get(int(clamp_magnitude), f"{int(clamp_magnitude)} deg"),
            zorder=3.0,
        )
    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    shade_no_feedback_region(ax)
    ax.axvline(0.0, color="0.45", linestyle=":", linewidth=1.0)
    ax.axvline(240.0, color="0.45", linestyle=":", linewidth=1.0)
    ax.set_xlim(-20, 249)
    ax.set_xticks([-20, 0, 60, 120, 180, 240])
    if np.isfinite(y_min) and np.isfinite(y_max):
        apply_y_limits(ax, padded_ylim([y_min, y_max], include_zero=True, min_pad=2.0))
    else:
        apply_y_limits(ax, None)
    ax.set_title("Representative learning curves", pad=8)
    ax.set_xlabel("Cycle relative to clamp onset")
    ax.set_ylabel("Hand angle ($^\\circ$)")
    ax.grid(axis="y", alpha=0.16)
    clamp_legend = ax.legend(frameon=False, loc="upper left", fontsize=PAPER_FONT["legend"], ncol=2)
    if model_curves is not None:
        style_legend = ax.legend(
            handles=[
                Line2D([0], [0], color="0.35", linewidth=2.3, alpha=0.68, label="Human mean"),
                Line2D(
                    [0], [0], color="0.15", linewidth=2.8, linestyle=MODEL_DASH, label=model_label
                ),
            ],
            frameon=False,
            loc="lower right",
            fontsize=PAPER_FONT["legend"] * 0.76,
        )
        ax.add_artist(clamp_legend)
        ax.add_artist(style_legend)
    style_axes(ax)


def plot_pointer_summary(
    ax: plt.Axes,
    subject_df: pd.DataFrame,
    *,
    title: str,
    ylabel: str,
    stat: str = "mean",
    value_col: str = "value",
    ylim: tuple[float, float] | None = None,
    zero_line: bool = True,
    overlay_lines_by_pointer: dict[str, dict[str, object]] | None = None,
    show_style_legend: bool = False,
    metrics_lines: list[str] | None = None,
    metrics_loc: str = "upper left",
) -> None:
    summary = summarize_by_pointer(subject_df, value_col=value_col, stat=stat)
    for pointer in POINTER_ORDER:
        sdf = summary.loc[summary["pointer"].eq(pointer)].sort_values("clamp_magnitude")
        if sdf.empty:
            continue
        color = DEVICE_COLORS[pointer]
        x = sdf["clamp_magnitude"].to_numpy(dtype=float)
        lo = sdf["ci_low"].to_numpy(dtype=float)
        hi = sdf["ci_high"].to_numpy(dtype=float)
        ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0, zorder=1)

    if overlay_lines_by_pointer:
        for pointer in POINTER_ORDER:
            overlay = overlay_lines_by_pointer.get(pointer)
            if overlay is None:
                continue
            summary_df = add_zero_anchor(pd.DataFrame(overlay["summary_df"]).copy())
            ax.plot(
                pd.to_numeric(summary_df["clamp_magnitude"], errors="coerce"),
                pd.to_numeric(summary_df["predicted_value"], errors="coerce"),
                color=str(overlay.get("color", DEVICE_EDGE_COLORS.get(pointer, "#111111"))),
                linewidth=float(overlay.get("linewidth", 2.8)),
                alpha=float(overlay.get("alpha", 0.98)),
                linestyle=overlay.get("linestyle", MODEL_DASH),
                zorder=float(overlay.get("zorder", 2.0)),
            )

    for pointer in POINTER_ORDER:
        sdf = summary.loc[summary["pointer"].eq(pointer)].sort_values("clamp_magnitude")
        if sdf.empty:
            continue
        color = DEVICE_COLORS[pointer]
        x = sdf["clamp_magnitude"].to_numpy(dtype=float)
        y = sdf["center"].to_numpy(dtype=float)
        ax.plot(x, y, color=color, linewidth=2.3, zorder=3)
        ax.scatter(
            x,
            y,
            s=DEVICE_MARKER_SIZE,
            color=color,
            edgecolor="white",
            linewidth=1.0,
            zorder=4,
            label=pointer,
        )

    if zero_line:
        ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    ax.set_xlim(*CLAMP_XLIM)
    ax.set_xticks(CLAMP_XTICKS)
    if ylim is not None:
        apply_y_limits(ax, ylim)
    else:
        apply_y_limits(ax, None)
    ax.set_title(title, pad=8)
    ax.set_xlabel("Clamp magnitude ($^\\circ$)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.16)
    device_legend = ax.legend(
        frameon=False,
        loc="upper right",
        title="Input device",
        fontsize=PAPER_FONT["legend"],
        title_fontsize=PAPER_FONT["legend"],
    )
    if show_style_legend:
        style_legend = ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color="0.35",
                    linewidth=2.0,
                    marker="o",
                    markersize=6.5,
                    markerfacecolor="0.35",
                    markeredgecolor="white",
                    markeredgewidth=1.0,
                    label="Human mean",
                ),
                Line2D(
                    [0], [0], color="0.15", linewidth=2.8, linestyle=MODEL_DASH, label="Vector BCC"
                ),
            ],
            frameon=False,
            loc="lower right",
            fontsize=PAPER_FONT["legend"] * 0.76,
        )
        ax.add_artist(device_legend)
        ax.add_artist(style_legend)
    add_metrics_box(ax, metrics_lines, loc=metrics_loc)
    style_axes(ax)


def _draw_box_series(
    ax: plt.Axes,
    values_by_clamp: dict[int, np.ndarray],
    positions: np.ndarray,
    *,
    edgecolor: str,
    facecolor: str,
    scatter_color: str,
    rng_seed: int,
) -> None:
    ordered_clamps = list(values_by_clamp)
    data = [values_by_clamp[clamp] for clamp in ordered_clamps]
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )
    for patch in bp["boxes"]:
        patch.set(facecolor=facecolor, edgecolor=edgecolor, linewidth=1.0, alpha=0.88)
    for element in ("whiskers", "caps", "medians"):
        for artist in bp[element]:
            artist.set(color=edgecolor, linewidth=1.0)

    rng = np.random.default_rng(rng_seed)
    for pos, values in zip(positions, data, strict=True):
        if len(values) == 0:
            continue
        jitter = rng.uniform(-0.05, 0.05, size=len(values))
        ax.scatter(
            np.full(len(values), pos) + jitter,
            values,
            s=INDIVIDUAL_POINT_SIZE,
            color=scatter_color,
            alpha=0.30,
            linewidths=0,
            zorder=2,
        )


def plot_phase_distribution(
    ax: plt.Axes,
    subject_df: pd.DataFrame,
    clamp_order: list[int],
    *,
    title: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    base_vals = {
        clamp: (
            subject_df.loc[
                subject_df["phase"].eq("Baseline") & subject_df["clamp_magnitude"].eq(clamp),
                "value",
            ]
            .dropna()
            .to_numpy(dtype=float)
        )
        for clamp in clamp_order
    }
    late_vals = {
        clamp: (
            subject_df.loc[
                subject_df["phase"].eq("Late adaptation") & subject_df["clamp_magnitude"].eq(clamp),
                "value",
            ]
            .dropna()
            .to_numpy(dtype=float)
        )
        for clamp in clamp_order
    }

    x = np.arange(len(clamp_order), dtype=float)
    _draw_box_series(
        ax,
        base_vals,
        x - 0.16,
        edgecolor=TEAL_DARK,
        facecolor=TEAL_LIGHT,
        scatter_color=TEAL_DARK,
        rng_seed=stable_seed(title, "baseline"),
    )
    _draw_box_series(
        ax,
        late_vals,
        x + 0.16,
        edgecolor=MAGENTA_DARK,
        facecolor=MAGENTA_LIGHT,
        scatter_color=MAGENTA_DARK,
        rng_seed=stable_seed(title, "late"),
    )

    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(clamp) for clamp in clamp_order])
    if ylim is None:
        ylim = padded_ylim(
            *(list(base_vals.values()) + list(late_vals.values())),
            include_zero=True,
            min_pad=3.0,
        )
    apply_y_limits(ax, ylim)
    ax.set_title(title, pad=8)
    ax.set_xlabel("Clamp magnitude ($^\\circ$)")
    ax.set_ylabel("Hand angle ($^\\circ$)")
    ax.grid(axis="y", alpha=0.16)
    handles = [
        Patch(facecolor=TEAL_LIGHT, edgecolor=TEAL_DARK, label="Baseline"),
        Patch(facecolor=MAGENTA_LIGHT, edgecolor=MAGENTA_DARK, label="Late adaptation"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=PAPER_FONT["legend"])
    style_axes(ax)


def plot_bias_direction_summary(
    ax: plt.Axes,
    *,
    summary: pd.DataFrame | None = None,
    counts: pd.DataFrame | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    if summary is None:
        summary = pd.read_csv(TABLE_DIR / "late_adaptation_by_signed_bias_direction_summary.csv")
    if counts is None:
        counts = pd.read_csv(TABLE_DIR / "late_adaptation_by_signed_bias_direction_counts.csv")
    summary["clamp_magnitude"] = pd.to_numeric(summary["clamp_magnitude"], errors="coerce").astype(
        int
    )
    for col in ["median", "q25", "q75"]:
        summary[col] = pd.to_numeric(summary[col], errors="coerce")

    count_lookup = dict(zip(counts["bias_direction"], counts["n"], strict=True))
    for label in BIAS_ORDER:
        sdf = summary.loc[summary["bias_direction"].eq(label)].sort_values("clamp_magnitude")
        if sdf.empty:
            continue
        color = BIAS_COLORS[label]
        x = sdf["clamp_magnitude"].to_numpy(dtype=float)
        ax.fill_between(
            x,
            sdf["q25"].to_numpy(dtype=float),
            sdf["q75"].to_numpy(dtype=float),
            color=color,
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )
        ax.plot(x, sdf["median"], color=color, linewidth=2.3, zorder=2)
        ax.scatter(
            x,
            sdf["median"],
            s=DEVICE_MARKER_SIZE,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
            label=f"{label} (n={int(count_lookup.get(label, 0))})",
        )

    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    ax.set_xlim(*CLAMP_XLIM)
    ax.set_xticks(CLAMP_XTICKS)
    if ylim is None:
        ylim = padded_ylim(
            summary["q25"], summary["q75"], summary["median"], include_zero=True, min_pad=1.5
        )
    apply_y_limits(ax, ylim)
    ax.set_title("Late adaptation by baseline-bias sign", pad=8)
    ax.set_xlabel("Clamp magnitude ($^\\circ$)")
    ax.set_ylabel("Adaptation extent ($^\\circ$)")
    ax.grid(axis="y", alpha=0.16)
    ax.legend(
        frameon=False,
        loc="upper right",
        title="Baseline bias sign",
        fontsize=PAPER_FONT["legend"],
        title_fontsize=PAPER_FONT["legend"],
    )
    style_axes(ax)


def build_filtered_spatial_bias_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(TABLE_DIR / "baseline_bias_by_target.csv")
    raw["pointer"] = raw["pointer"].astype(str)
    raw["target_angles_degrees"] = pd.to_numeric(raw["target_angles_degrees"], errors="coerce")
    raw["bias"] = pd.to_numeric(raw["bias"], errors="coerce")
    raw = raw.loc[raw["pointer"].isin(POINTER_ORDER) & raw["bias"].abs().le(40.0)].copy()

    rows: list[dict[str, float | str | int]] = []
    for (pointer, target_angle), g in raw.groupby(
        ["pointer", "target_angles_degrees"], observed=True
    ):
        row = mean_ci(g["bias"])
        row["pointer"] = pointer
        row["target_angles_degrees"] = int(target_angle)
        row["mean"] = row.pop("center")
        rows.append(row)
    summary = (
        pd.DataFrame.from_records(rows)
        .sort_values(["pointer", "target_angles_degrees"])
        .reset_index(drop=True)
    )

    _, fit_points = fit_transformation_bias_patterns(
        summary[["pointer", "target_angles_degrees", "mean"]].copy()
    )
    return raw, summary, fit_points


def plot_spatial_bias_with_tr_markers(
    ax: plt.Axes,
    *,
    raw: pd.DataFrame | None = None,
    summary: pd.DataFrame | None = None,
    fit_points: pd.DataFrame | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    if raw is None or summary is None or fit_points is None:
        raw, summary, fit_points = build_filtered_spatial_bias_tables()
    scatter_offsets = {"Mouse": -4.0, "Trackpad": 4.0}
    model_offsets = {"Mouse": -10.0, "Trackpad": 10.0}
    rng = np.random.default_rng(RNG_SEED)

    for pointer in POINTER_ORDER:
        color = DEVICE_COLORS[pointer]
        edge = DEVICE_EDGE_COLORS[pointer]
        sub = raw.loc[raw["pointer"].eq(pointer)].copy()
        x_center = sub["target_angles_degrees"].to_numpy(dtype=float) + scatter_offsets[pointer]
        x_jitter = x_center + rng.uniform(-1.4, 1.4, size=len(sub))
        ax.scatter(
            x_jitter,
            sub["bias"].to_numpy(dtype=float),
            color=color,
            alpha=0.18,
            s=RAW_POINT_SIZE,
            linewidths=0,
            zorder=1,
        )

        ss = summary.loc[summary["pointer"].eq(pointer)].sort_values("target_angles_degrees")
        x = ss["target_angles_degrees"].to_numpy(dtype=float) + scatter_offsets[pointer]
        ax.errorbar(
            x,
            ss["mean"].to_numpy(dtype=float),
            yerr=[
                (ss["mean"] - ss["ci_low"]).to_numpy(dtype=float),
                (ss["ci_high"] - ss["mean"]).to_numpy(dtype=float),
            ],
            fmt="none",
            ecolor=edge,
            elinewidth=1.3,
            capsize=3.0,
            zorder=2,
        )
        ax.scatter(
            x,
            ss["mean"].to_numpy(dtype=float),
            s=DEVICE_MARKER_SIZE,
            color=color,
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
            label=f"{pointer} human",
        )

        fp = fit_points.loc[fit_points["pointer"].eq(pointer)].sort_values("target_angles_degrees")
        pred_x = fp["target_angles_degrees"].to_numpy(dtype=float) + model_offsets[pointer]
        pred_y = fp["model_pred_bias"].to_numpy(dtype=float)
        ax.scatter(
            pred_x,
            pred_y,
            s=TR_MARKER_SIZE,
            marker="D",
            facecolors="white",
            edgecolors=edge,
            linewidths=1.4,
            zorder=4,
            label=f"{pointer} TR",
        )

    ax.axhline(0.0, color="0.35", linestyle=":", linewidth=1.0)
    ax.set_xlim(20, 340)
    ax.set_xticks([45, 135, 225, 315])
    if ylim is None:
        ylim = padded_ylim(
            raw["bias"],
            summary["ci_low"],
            summary["ci_high"],
            fit_points["model_pred_bias"],
            include_zero=True,
            min_pad=1.5,
        )
    apply_y_limits(ax, ylim)
    ax.set_title("Baseline bias by input device", pad=8)
    ax.set_xlabel("Target angle ($^\\circ$)")
    ax.set_ylabel("No-FB baseline bias ($^\\circ$)")
    ax.grid(axis="y", alpha=0.16)
    ax.legend(frameon=False, loc="upper left", fontsize=PAPER_FONT["legend"], ncol=2)
    style_axes(ax)


def build_high_error_bias_sign_summary() -> tuple[pd.DataFrame, dict[str, int]]:
    membership = pd.read_csv(
        TABLE_DIR / "late_adaptation_by_signed_bias_direction_pairings.csv", dtype={"ppid": str}
    )
    membership["clamp_magnitude"] = pd.to_numeric(
        membership["clamp_magnitude"], errors="coerce"
    ).astype(int)
    membership = membership.loc[membership["clamp_magnitude"].between(135, 170)].copy()
    membership = membership[["ppid", "clamp_magnitude", "bias_direction"]].drop_duplicates()

    trials = pd.read_csv(
        DATA_PATH,
        usecols=["ppid", "condition", "cycle_num", "clamp_magnitude", "recentred_hand_angle"],
        dtype={"ppid": str},
    )
    trials["clamp_magnitude"] = pd.to_numeric(trials["clamp_magnitude"], errors="coerce")
    trials["cycle_num"] = pd.to_numeric(trials["cycle_num"], errors="coerce")
    # recentred_hand_angle is task-aligned in this dataset; it is not
    # participant-baseline-corrected.
    trials["recentred_hand_angle"] = pd.to_numeric(trials["recentred_hand_angle"], errors="coerce")
    trials["task_cycle"] = trials["cycle_num"] - 21
    trials = trials.loc[
        trials["condition"].isin(
            ["NoFeedbackBaseline", "FeedbackBaseline", "Clamp", "FeedbackWashout"]
        )
        & trials["clamp_magnitude"].between(135, 170)
    ].copy()
    trials["clamp_magnitude"] = trials["clamp_magnitude"].astype(int)
    trials = trials.merge(membership, on=["ppid", "clamp_magnitude"], how="inner")

    participant_clamp_curve = (
        trials.groupby(["ppid", "clamp_magnitude", "bias_direction", "task_cycle"], as_index=False)[
            "recentred_hand_angle"
        ]
        .mean()
        .rename(columns={"recentred_hand_angle": "value"})
    )
    participant_curve = (
        participant_clamp_curve.groupby(["ppid", "bias_direction", "task_cycle"], as_index=False)[
            "value"
        ]
        .mean()
        .sort_values(["bias_direction", "ppid", "task_cycle"])
        .reset_index(drop=True)
    )

    counts = participant_curve.groupby("bias_direction")["ppid"].nunique().to_dict()
    rows: list[dict[str, float | int | str]] = []
    for (bias_direction, task_cycle), g in participant_curve.groupby(
        ["bias_direction", "task_cycle"], observed=True
    ):
        row = mean_ci(g["value"])
        row["bias_direction"] = bias_direction
        row["task_cycle"] = int(task_cycle)
        row["n_participants"] = int(counts.get(bias_direction, 0))
        rows.append(row)
    summary = (
        pd.DataFrame.from_records(rows)
        .sort_values(["bias_direction", "task_cycle"])
        .reset_index(drop=True)
    )
    return summary, {str(key): int(value) for key, value in counts.items()}


def plot_high_error_bias_sign_trajectories(ax: plt.Axes) -> None:
    summary, counts = build_high_error_bias_sign_summary()
    y_min = np.inf
    y_max = -np.inf
    for label in BIAS_ORDER:
        sdf = summary.loc[summary["bias_direction"].eq(label)].sort_values("task_cycle")
        if sdf.empty:
            continue
        color = BIAS_COLORS[label]
        x = sdf["task_cycle"].to_numpy(dtype=float)
        y = sdf["center"].to_numpy(dtype=float)
        lo = sdf["ci_low"].to_numpy(dtype=float)
        hi = sdf["ci_high"].to_numpy(dtype=float)
        ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0, zorder=1)
        ax.plot(
            x,
            y,
            color=color,
            linewidth=2.3,
            zorder=2,
            label=f"{label} (n={counts.get(label, 0)})",
        )
        y_min = min(y_min, float(np.nanmin(lo)))
        y_max = max(y_max, float(np.nanmax(hi)))

    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    shade_no_feedback_region(ax)
    ax.axvline(0.0, color="0.45", linestyle=":", linewidth=1.0)
    ax.axvline(240.0, color="0.45", linestyle=":", linewidth=1.0)
    ax.set_xlim(-20, 249)
    ax.set_xticks([-20, 0, 60, 120, 180, 240])
    if np.isfinite(y_min) and np.isfinite(y_max):
        apply_y_limits(ax, padded_ylim([y_min, y_max], include_zero=True, min_pad=1.5))
    else:
        apply_y_limits(ax, None)
    ax.set_title("Large-clamp behaviour by baseline bias", pad=8)
    ax.set_xlabel("Cycle relative to clamp onset")
    ax.set_ylabel("Hand angle ($^\\circ$)")
    ax.grid(axis="y", alpha=0.16)
    ax.legend(
        frameon=False,
        loc="upper right",
        title="Baseline bias sign",
        fontsize=PAPER_FONT["legend"],
        title_fontsize=PAPER_FONT["legend"],
    )
    style_axes(ax)


def plot_model_delta_bic_panel(ax: plt.Axes, dataset_bic_df: pd.DataFrame) -> None:
    if dataset_bic_df.empty:
        ax.axis("off")
        return

    available_datasets = list(dict.fromkeys(dataset_bic_df["dataset_slug"].astype(str)))
    dataset_order = [
        dataset_slug
        for dataset_slug in MODEL_FIT_DATASET_LABELS
        if dataset_slug in available_datasets
    ]
    dataset_order.extend(
        dataset_slug for dataset_slug in available_datasets if dataset_slug not in dataset_order
    )
    x_positions = np.arange(len(dataset_order), dtype=float)
    x_map = {
        dataset_slug: x_value
        for dataset_slug, x_value in zip(dataset_order, x_positions, strict=True)
    }
    model_offsets = {
        "MeanTimeseriesBaseBCC": -0.18,
        "MeanTimeseriesCausalInf": 0.0,
        "MeanTimeseriesVectorBCC": 0.18,
    }
    max_delta_bic = float(pd.to_numeric(dataset_bic_df["delta_bic"], errors="coerce").max())
    max_delta_bic = max(max_delta_bic, 1.0)
    y_top = max_delta_bic * 1.12 + 400.0
    label_offset = max(120.0, max_delta_bic * 0.022)

    for model_name in MEAN_TIMESERIES_MODEL_ORDER:
        color = MEAN_TIMESERIES_MODEL_COLORS[model_name]
        mdf = (
            dataset_bic_df.loc[dataset_bic_df["model_name"].eq(model_name)]
            .set_index("dataset_slug")
            .reindex(dataset_order)
            .reset_index()
        )
        x_values = np.array(
            [x_map[dataset_slug] + model_offsets[model_name] for dataset_slug in dataset_order],
            dtype=float,
        )
        y_values = pd.to_numeric(mdf["delta_bic"], errors="coerce").to_numpy(dtype=float)

        ax.scatter(
            x_values,
            y_values,
            s=260,
            color=color,
            alpha=0.14,
            linewidths=0,
            zorder=3,
        )
        ax.scatter(
            x_values,
            y_values,
            s=88 if model_name != "MeanTimeseriesVectorBCC" else 96,
            color=color,
            edgecolors="white",
            linewidths=1.2,
            zorder=4,
            label=MEAN_TIMESERIES_MODEL_LABELS[model_name],
        )

        for x_value, y_value, delta_bic, is_best in zip(
            x_values,
            y_values,
            pd.to_numeric(mdf["delta_bic"], errors="coerce").to_numpy(dtype=float),
            mdf["is_best"].to_numpy(dtype=bool),
            strict=True,
        ):
            label = f"{delta_bic:,.2f}"
            ax.text(
                x_value,
                y_value + label_offset,
                label,
                ha="center",
                va="bottom",
                fontsize=PAPER_FONT["legend"] * 0.76,
                color=color,
                fontweight="bold" if is_best else None,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.88,
                },
                zorder=5,
            )

    ax.axhline(0.0, color="0.78", linewidth=1.1, zorder=1)
    ax.set_xlim(-0.5, len(dataset_order) - 0.5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [
            MODEL_FIT_DATASET_LABELS.get(
                dataset_slug, DATASET_LABELS.get(dataset_slug, dataset_slug)
            )
            for dataset_slug in dataset_order
        ]
    )
    ax.set_xlabel("")
    ax.set_ylim(-220.0, y_top)
    ax.set_ylabel("$\\Delta$BIC")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", alpha=0.12, zorder=1)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_title("BIC model comparison", pad=8)
    ax.text(
        0.995,
        0.985,
        "Relative to the best model within each dataset",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=PAPER_FONT["legend"] * 0.68,
        color="0.35",
    )
    ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=PAPER_FONT["legend"] * 0.74,
        ncol=1,
        columnspacing=1.0,
        handlelength=1.0,
    )
    style_axes(ax)


def _harmonize_imported_axis_fonts(
    ax: plt.Axes,
    *,
    annotation_scale: float = 0.68,
    legend_scale: float = 1.0,
) -> None:
    ax.title.set_fontsize(PAPER_FONT["title"])
    ax.xaxis.label.set_fontsize(PAPER_FONT["axis_label"])
    ax.yaxis.label.set_fontsize(PAPER_FONT["axis_label"])
    ax.tick_params(labelsize=PAPER_FONT["tick"])
    for text in ax.texts:
        if text.get_fontsize() < PAPER_FONT["legend"] * 0.75:
            text.set_fontsize(PAPER_FONT["legend"] * annotation_scale)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(PAPER_FONT["legend"] * legend_scale)
        title = legend.get_title()
        if title is not None:
            title.set_fontsize(PAPER_FONT["legend"] * legend_scale)


def _add_first_bout_clamp_colorbar(ax: plt.Axes) -> None:
    cax = ax.inset_axes([1.015, 0.28, 0.026, 0.48], transform=ax.transAxes)
    cbar = ax.figure.colorbar(
        ScalarMappable(norm=FIRST_BOUT_CLAMP_NORM, cmap=FIRST_BOUT_HUMAN_CMAP),
        cax=cax,
    )
    cbar.set_label(r"Clamp ($^\circ$)", fontsize=PAPER_FONT["axis_label"] * 0.62)
    cbar.ax.tick_params(labelsize=PAPER_FONT["tick"] * 0.58, length=2.4, width=0.8)


def _set_line_artist_color(
    line: Line2D,
    color: str,
    *,
    marker_face: str | None = None,
    alpha: float | None = None,
) -> None:
    line.set_color(color)
    line.set_markeredgecolor(color)
    if marker_face is not None:
        line.set_markerfacecolor(marker_face)
    if alpha is not None:
        line.set_alpha(alpha)


def _apply_first_bout_geometry_palette(ax: plt.Axes) -> None:
    point_collections = [
        collection for collection in ax.collections if isinstance(collection, PathCollection)
    ]
    if point_collections:
        point_collections[0].set_cmap(FIRST_BOUT_HUMAN_CMAP)
        point_collections[0].set_norm(FIRST_BOUT_CLAMP_NORM)
        point_collections[0].set_edgecolors(mcolors.to_rgba(FIRST_BOUT_HUMAN_COLOR, 0.86))
        point_collections[0].set_linewidths(0.45)
        point_collections[0].set_alpha(0.76)

    trend_cmaps = [FIRST_BOUT_HUMAN_CMAP, FIRST_BOUT_ONE_D_CMAP, FIRST_BOUT_TWO_D_CMAP]
    line_collections = [
        collection for collection in ax.collections if isinstance(collection, LineCollection)
    ]
    for collection, cmap in zip(line_collections, trend_cmaps, strict=False):
        collection.set_cmap(cmap)
        collection.set_norm(FIRST_BOUT_CLAMP_NORM)
        collection.set_alpha(0.96)


def _apply_first_bout_radial_palette(ax: plt.Axes) -> None:
    for line in ax.lines:
        label = line.get_label()
        marker = line.get_marker()
        if label == "human OLS fit":
            _set_line_artist_color(line, FIRST_BOUT_HUMAN_COLOR, alpha=1.0)
        elif label == "calibrated 1D prediction":
            _set_line_artist_color(line, FIRST_BOUT_ONE_D_COLOR)
        elif label == "calibrated 2D prediction":
            _set_line_artist_color(line, FIRST_BOUT_TWO_D_COLOR)
        elif marker in {"o", "_"}:
            _set_line_artist_color(
                line,
                FIRST_BOUT_HUMAN_COLOR,
                marker_face=FIRST_BOUT_HUMAN_COLOR,
                alpha=1.0,
            )
            if marker == "o":
                line.set_markersize(FIRST_BOUT_DOT_MARKERSIZE)
                line.set_markeredgewidth(FIRST_BOUT_DOT_EDGEWIDTH)

    for collection in ax.collections:
        label = collection.get_label()
        if label == "human OLS fit 95% CI":
            rgba = mcolors.to_rgba(FIRST_BOUT_HUMAN_COLOR, 0.12)
            collection.set_facecolor(rgba)
            collection.set_edgecolor(rgba)
        elif isinstance(collection, LineCollection):
            collection.set_color(FIRST_BOUT_HUMAN_COLOR)
            collection.set_alpha(1.0)


def _set_first_bout_geometry_legend(ax: plt.Axes) -> None:
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    ax.legend(
        handles=[
            Line2D([0], [0], color=FIRST_BOUT_HUMAN_COLOR, linewidth=2.2, label="human trend"),
            Line2D(
                [0], [0], color=FIRST_BOUT_ONE_D_COLOR, linewidth=2.2, label="calibrated Base BCC"
            ),
            Line2D(
                [0], [0], color=FIRST_BOUT_TWO_D_COLOR, linewidth=2.2, label="calibrated Vector BCC"
            ),
        ],
        frameon=False,
        loc="upper right",
        fontsize=PAPER_FONT["legend"] * 0.70,
        handlelength=1.7,
        labelspacing=0.35,
        borderaxespad=0.25,
    )


def _set_first_bout_radial_legend(ax: plt.Axes) -> None:
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    ax.legend(
        handles=[
            Patch(
                facecolor=mcolors.to_rgba(FIRST_BOUT_HUMAN_COLOR, 0.12),
                edgecolor="none",
                label="human OLS fit 95% CI",
            ),
            Line2D([0], [0], color=FIRST_BOUT_HUMAN_COLOR, linewidth=2.0, label="human OLS fit"),
            Line2D(
                [0],
                [0],
                color=FIRST_BOUT_ONE_D_COLOR,
                linewidth=2.0,
                linestyle=MODEL_DASH,
                label="calibrated Base BCC prediction",
            ),
            Line2D(
                [0],
                [0],
                color=FIRST_BOUT_TWO_D_COLOR,
                linewidth=2.0,
                linestyle=MODEL_DASH,
                label="calibrated Vector BCC prediction",
            ),
            Line2D(
                [0],
                [0],
                color=FIRST_BOUT_HUMAN_COLOR,
                marker="o",
                markersize=FIRST_BOUT_DOT_MARKERSIZE,
                markerfacecolor=FIRST_BOUT_HUMAN_COLOR,
                markeredgecolor=FIRST_BOUT_HUMAN_COLOR,
                markeredgewidth=FIRST_BOUT_DOT_EDGEWIDTH,
                linewidth=0,
                label="human mean",
            ),
        ],
        frameon=False,
        loc="lower right",
        fontsize=PAPER_FONT["legend"] * 0.58,
        handlelength=1.6,
        labelspacing=0.26,
        borderaxespad=0.20,
    )


def _standard_r2(y: np.ndarray, yhat: np.ndarray) -> float:
    valid = np.isfinite(y) & np.isfinite(yhat)
    y = y[valid]
    yhat = yhat[valid]
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom <= 1e-12:
        return np.nan
    return 1.0 - float(np.sum((y - yhat) ** 2)) / denom


def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    valid = np.isfinite(y) & np.isfinite(yhat)
    if not np.any(valid):
        return np.nan
    return float(np.sqrt(np.mean((y[valid] - yhat[valid]) ** 2)))


def _replace_first_bout_radial_metrics_text(ax: plt.Axes, paths: pd.DataFrame) -> None:
    focus = focus_first_bout_radial_paths(paths)
    human = focus.loc[
        focus["panel"].eq("Human") & focus["path_type"].eq("condition_mean")
    ].sort_values("clamp_magnitude")
    one_d = focus.loc[
        focus["panel"].eq("Calibrated 1D") & focus["path_type"].eq("condition_mean")
    ].sort_values("clamp_magnitude")
    two_d = focus.loc[
        focus["panel"].eq("Calibrated 2D") & focus["path_type"].eq("condition_mean")
    ].sort_values("clamp_magnitude")

    x = human["clamp_magnitude"].to_numpy(dtype=float)
    y = human["forward"].to_numpy(dtype=float)
    human_fit = first_bout_ols_fit_with_mean_ci(x, y, x)
    ols_yhat = (
        float(human_fit["intercept_pct_target"]) + float(human_fit["slope_pct_target_per_deg"]) * x
    )
    one_d_yhat = one_d["forward"].to_numpy(dtype=float)
    two_d_yhat = two_d["forward"].to_numpy(dtype=float)

    stat_text = (
        f"OLS human fit\n"
        f"Human b={float(human_fit['slope_pct_target_per_100deg']):.1f}%/100°\n"
        f"95% CI [{float(human_fit['slope_ci_low_pct_target_per_100deg']):.1f}, "
        f"{float(human_fit['slope_ci_high_pct_target_per_100deg']):.1f}]\n"
        f"{format_first_bout_p(float(human_fit['p_two_sided']))}\n\n"
        f"Radial prediction fit\n"
        f"OLS $R^2$={_standard_r2(y, ols_yhat):.2f}, RMSE={_rmse(y, ols_yhat):.1f}\n"
        f"Vector BCC $R^2$={_standard_r2(y, two_d_yhat):.2f}, RMSE={_rmse(y, two_d_yhat):.1f}\n"
        f"Base BCC $R^2$={_standard_r2(y, one_d_yhat):.2f}, RMSE={_rmse(y, one_d_yhat):.1f}"
    )
    for text in ax.texts:
        if "Radial prediction fit" in text.get_text():
            text.set_text(stat_text)
            return


def _clean_first_bout_text(ax: plt.Axes) -> None:
    replacements = {
        "calibrated 1D": "calibrated Base BCC",
        "calibrated 2D": "calibrated Vector BCC",
        "2D R2c=": "Vector BCC $R^2_c$=",
        "1D R2c=": "Base BCC $R^2_c$=",
        "OLS R2c=": "OLS $R^2_c$=",
        "2D R2=": "Vector BCC $R^2$=",
        "1D R2=": "Base BCC $R^2$=",
        "OLS R2=": "OLS $R^2$=",
        "/100 deg": "/100$^\\circ$",
    }
    for text in ax.texts:
        updated = text.get_text()
        for old, new in replacements.items():
            updated = updated.replace(old, new)
        text.set_text(updated)


def _send_zero_guides_to_back(ax: plt.Axes) -> None:
    for line in ax.lines:
        color = mcolors.to_rgba(line.get_color())
        is_reference_gray = np.allclose(color[:3], mcolors.to_rgba("0.84")[:3], atol=0.015)
        if is_reference_gray:
            line.set_zorder(0.1)
            line.set_alpha(1.0)
    ax.set_axisbelow(True)


def plot_first_bout_radial_bridge_panels(ax_geometry: plt.Axes, ax_radial: plt.Axes) -> None:
    summary, paths = load_first_bout_radial_calibrated_tables(force_rebuild=False)
    plot_first_bout_calibrated_geometry(ax_geometry, paths, summary)
    plot_first_bout_radial_dose_response(ax_radial, paths, summary, "radial")

    ax_geometry.set_title("Early two-dimensional learning")
    ax_radial.set_title("Early radial learning")
    ax_geometry.set_xlim(-80.0, 100.0)
    ax_geometry.set_ylim(-25.0, 100.0)
    ax_geometry.set_xticks([-80, -40, 0, 40, 80])
    ax_geometry.set_yticks([-25, 0, 25, 50, 75, 100])
    ax_geometry.set_aspect("auto")
    ax_radial.set_ylim(-150.0, 200.0)
    ax_radial.set_yticks([-150, -100, -50, 0, 50, 100, 150, 200])
    ax_geometry.set_ylabel("Cumulative azimuthal change (% target distance)")
    ax_radial.set_xlabel(r"Clamp magnitude ($^\circ$)")
    _apply_first_bout_geometry_palette(ax_geometry)
    _apply_first_bout_radial_palette(ax_radial)
    _replace_first_bout_radial_metrics_text(ax_radial, paths)
    _clean_first_bout_text(ax_geometry)
    _clean_first_bout_text(ax_radial)
    _set_first_bout_geometry_legend(ax_geometry)
    _set_first_bout_radial_legend(ax_radial)
    _send_zero_guides_to_back(ax_geometry)
    _send_zero_guides_to_back(ax_radial)
    _add_first_bout_clamp_colorbar(ax_geometry)
    style_axes(ax_geometry)
    style_axes(ax_radial)
    _harmonize_imported_axis_fonts(ax_geometry, annotation_scale=0.58, legend_scale=0.70)
    _harmonize_imported_axis_fonts(ax_radial, annotation_scale=0.48, legend_scale=0.58)
