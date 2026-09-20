from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from clamp_analysis import paths
from clamp_analysis.kinematics.geometry import CLAMPS

MEAN_TIMESERIES_COLLECTION = paths.FITS_DIR / "mean_timeseries_shared_point_joint_device_motorvar"

JOINT_DATASET_SLUG = "joint_devices"

MODEL_DATASET_SLUGS = ("mouse", "trackpad")

MODEL_NAMES = {
    "scalar": "MeanTimeseriesBaseBCC",
    "vector": "MeanTimeseriesVectorBCC",
}

VIS_VAR_INT = 1.853

BASELINE = np.asarray([1.0, 0.0], dtype=float)


def _model_dir(dataset_slug: str, model_name: str) -> Path:
    return MEAN_TIMESERIES_COLLECTION / dataset_slug / model_name


def _model_param_dirs(dataset_slug: str, model_name: str) -> list[Path]:
    return [
        _model_dir(dataset_slug, model_name),
        _model_dir(JOINT_DATASET_SLUG, model_name),
    ]


def _summary_param_value(
    table: pd.DataFrame,
    parameter: str,
    *,
    scope: str | None = None,
    dataset_slug: str | None = None,
) -> float | None:
    rows = table.loc[table["parameter"].astype(str).eq(parameter)].copy()
    if scope is not None and "scope" in rows.columns:
        rows = rows.loc[rows["scope"].astype(str).eq(scope)].copy()
    if dataset_slug is not None:
        if "datasetSlug" in rows.columns:
            rows = rows.loc[rows["datasetSlug"].fillna("").astype(str).eq(dataset_slug)].copy()
        elif "dataset_slug" in rows.columns:
            rows = rows.loc[rows["dataset_slug"].fillna("").astype(str).eq(dataset_slug)].copy()
    elif "datasetSlug" in rows.columns:
        rows = rows.loc[rows["datasetSlug"].fillna("").astype(str).eq("")].copy()
    if rows.empty:
        return None
    return float(pd.to_numeric(pd.Series([rows.iloc[0]["mean"]]), errors="coerce").iloc[0])


def _params_from_joint_summary(
    table: pd.DataFrame, model_name: str, dataset_slug: str
) -> dict[str, float]:
    params: dict[str, float] = {}
    if model_name in {"MeanTimeseriesBaseBCC", "MeanTimeseriesVectorBCC"}:
        params["propVar"] = _summary_param_value(
            table, "propVar", scope="device_specific", dataset_slug=dataset_slug
        )
        params["visVarSlope"] = _summary_param_value(table, "visVarSlope", scope="shared")
        params["motorVar"] = _summary_param_value(
            table, "motorVar", scope="fixed_by_subset", dataset_slug=dataset_slug
        )
        params["retention"] = _summary_param_value(table, "retention", scope="shared")
        params["lr"] = _summary_param_value(table, "lr", scope="shared")
    elif model_name == "MeanTimeseriesCausalInf":
        params["senseVar"] = _summary_param_value(
            table, "senseVar", scope="device_specific", dataset_slug=dataset_slug
        )
        params["c"] = _summary_param_value(table, "c", scope="shared")
        params["retention"] = _summary_param_value(table, "retention", scope="shared")
        params["lr"] = _summary_param_value(table, "lr", scope="shared")
    params = {
        key: value for key, value in params.items() if value is not None and np.isfinite(value)
    }
    if model_name in {"MeanTimeseriesBaseBCC", "MeanTimeseriesVectorBCC"}:
        required = {"propVar", "visVarSlope", "motorVar", "retention", "lr"}
    else:
        required = {"senseVar", "c", "retention", "lr"}
    if required.issubset(params):
        return params
    return {}


def _params_from_shared_table(
    table: pd.DataFrame, model_name: str, dataset_slug: str
) -> dict[str, float]:
    values = table.set_index("parameter")["value"].astype(float).to_dict()
    if model_name in {"MeanTimeseriesBaseBCC", "MeanTimeseriesVectorBCC"}:
        params = {
            "propVar": values.get(f"propVar_{dataset_slug}", values.get("propVar")),
            "visVarSlope": values.get("visVarSlope"),
            "retention": values.get("retention"),
            "lr": values.get("lr"),
        }
        fixed_motor_path_candidates = [
            directory / f"{model_name}_fixed_motorvar_by_subset.csv"
            for directory in _model_param_dirs(dataset_slug, model_name)
        ]
        for fixed_motor_path in fixed_motor_path_candidates:
            if fixed_motor_path.exists():
                fixed_df = pd.read_csv(fixed_motor_path)
                match = fixed_df.loc[fixed_df["subsetSlug"].astype(str).eq(dataset_slug)]
                if not match.empty:
                    params["motorVar"] = float(match["fixedMotorVar"].iloc[0])
                    break
    elif model_name == "MeanTimeseriesCausalInf":
        params = {
            "senseVar": values.get(f"senseVar_{dataset_slug}", values.get("senseVar")),
            "c": values.get("c"),
            "retention": values.get("retention"),
            "lr": values.get("lr"),
        }
    else:
        params = {}
    return {
        key: float(value)
        for key, value in params.items()
        if value is not None and np.isfinite(value)
    }


def load_params(model_name: str, dataset_slug: str) -> dict[str, float]:
    for directory in _model_param_dirs(dataset_slug, model_name):
        summary_path = directory / f"{model_name}_parameter_summary.csv"
        if summary_path.exists():
            table = pd.read_csv(summary_path)
            if "scope" in table.columns and not table["scope"].astype(str).eq("population").all():
                params = _params_from_joint_summary(table, model_name, dataset_slug)
                if params:
                    return params
            if "scope" in table.columns:
                population = table.loc[table["scope"].astype(str).eq("population")].copy()
            else:
                population = table.copy()
            if not population.empty:
                return {
                    str(row["parameter"]): float(row["mean"]) for _, row in population.iterrows()
                }

        shared_path = directory / f"{model_name}_shared_params.csv"
        if shared_path.exists():
            params = _params_from_shared_table(pd.read_csv(shared_path), model_name, dataset_slug)
            if params:
                return params

    raise FileNotFoundError(
        f"Missing fitted parameters for {dataset_slug}/{model_name} in {MEAN_TIMESERIES_COLLECTION}."
    )


@lru_cache(maxsize=1)
def _constant_device_weights() -> dict[str, float]:
    overview_path = MEAN_TIMESERIES_COLLECTION / "dataset_overview.csv"
    if overview_path.exists():
        overview = pd.read_csv(overview_path)
        if {"subsetSlug", "n_participants"}.issubset(overview.columns):
            weights = {
                str(row["subsetSlug"]): float(row["n_participants"])
                for _, row in overview.iterrows()
                if str(row["subsetSlug"]) in MODEL_DATASET_SLUGS
                and np.isfinite(float(row["n_participants"]))
                and float(row["n_participants"]) > 0.0
            }
            if weights:
                return weights
    return {dataset_slug: 1.0 for dataset_slug in MODEL_DATASET_SLUGS}


def _device_weight_table() -> pd.DataFrame:
    constant_weights = _constant_device_weights()
    rows = []
    for dataset_slug in MODEL_DATASET_SLUGS:
        summary_path = (
            MEAN_TIMESERIES_COLLECTION
            / "data_summaries"
            / f"{dataset_slug}_mean_timeseries_summary.csv"
        )
        if not summary_path.exists():
            for clamp in CLAMPS:
                rows.append(
                    {
                        "dataset_slug": dataset_slug,
                        "clamp_magnitude": float(clamp),
                        "weight": float(constant_weights.get(dataset_slug, 1.0)),
                    }
                )
            continue

        summary = pd.read_csv(summary_path, usecols=["clamp_magnitude"])
        summary["clamp_magnitude"] = pd.to_numeric(summary["clamp_magnitude"], errors="coerce")
        clamp_summary = (
            summary.dropna(subset=["clamp_magnitude"])
            .loc[:, ["clamp_magnitude"]]
            .drop_duplicates()
            .sort_values("clamp_magnitude")
            .reset_index(drop=True)
        )
        clamp_summary["weight"] = float(constant_weights.get(dataset_slug, 1.0))
        clamp_summary["dataset_slug"] = dataset_slug
        rows.extend(clamp_summary.to_dict(orient="records"))
    return pd.DataFrame.from_records(rows)


def _combine_device_predictions(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    weights = _device_weight_table()
    combined = combined.merge(weights, on=["dataset_slug", "clamp_magnitude"], how="left")
    combined["weight"] = (
        pd.to_numeric(combined["weight"], errors="coerce").fillna(1.0).clip(lower=0.0)
    )

    key_cols = ["clamp_magnitude", "target_exposure_index"]
    numeric_cols = [
        col
        for col in combined.columns
        if col not in {*key_cols, "dataset_slug", "weight"}
        and pd.api.types.is_numeric_dtype(combined[col])
    ]

    work = combined[key_cols + ["weight"]].copy()
    for col in numeric_cols:
        values = pd.to_numeric(combined[col], errors="coerce")
        col_weights = work["weight"].where(values.notna(), 0.0)
        work[f"_weighted_{col}"] = values.fillna(0.0) * col_weights
        work[f"_weight_{col}"] = col_weights

    grouped = work.groupby(key_cols, as_index=False, sort=True).sum(numeric_only=True)
    out = grouped[key_cols].copy()
    for col in numeric_cols:
        denom = grouped[f"_weight_{col}"].replace(0.0, np.nan)
        out[col] = grouped[f"_weighted_{col}"] / denom
    return out.sort_values(key_cols).reset_index(drop=True)


def angle_to_unit_vector(angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(float(angle_deg))
    return np.asarray([np.cos(angle_rad), np.sin(angle_rad)], dtype=float)


def cartesian_posterior_mean(
    cue_means: tuple[np.ndarray, ...], cue_variances: tuple[float, ...]
) -> np.ndarray:
    total_precision = 0.0
    rhs = np.zeros(2, dtype=float)
    for mean_vec, variance in zip(cue_means, cue_variances):
        variance = max(float(variance), 1e-9)
        precision = 1.0 / variance
        total_precision += precision
        rhs += precision * np.asarray(mean_vec, dtype=float)
    return rhs / max(total_precision, 1e-12)


def simulate_vector_bcc_percepts(
    max_window: int, params: dict[str, float] | None = None
) -> pd.DataFrame:
    if params is None:
        frames = []
        for dataset_slug in MODEL_DATASET_SLUGS:
            device_params = load_params(MODEL_NAMES["vector"], dataset_slug)
            device_df = simulate_vector_bcc_percepts(max_window, device_params)
            device_df["dataset_slug"] = dataset_slug
            frames.append(device_df)
        return _combine_device_predictions(frames)

    prop_var = max(float(params["propVar"]), 1e-9)
    vis_var_slope = max(float(params["visVarSlope"]), 0.0)
    motor_var = max(float(params["motorVar"]), 1e-9)
    retention = float(params["retention"])
    lr = float(params["lr"])

    rows = []
    for clamp in CLAMPS:
        state = BASELINE.copy()
        rotation_model = -float(clamp)
        visual_cue = angle_to_unit_vector(rotation_model)
        vis_var = max((VIS_VAR_INT + vis_var_slope * abs(rotation_model)) ** 2, 1e-9)

        for exposure in range(1, max_window + 1):
            proprioceptive_state = state.copy()
            combined_percept = cartesian_posterior_mean(
                cue_means=(visual_cue, proprioceptive_state, BASELINE),
                cue_variances=(vis_var, prop_var, motor_var),
            )
            teaching_signal = BASELINE - combined_percept
            next_state = BASELINE + retention * (state - BASELINE) + lr * teaching_signal
            model_update = next_state - state

            rows.append(
                {
                    "clamp_magnitude": float(clamp),
                    "target_exposure_index": int(exposure),
                    "rotation_model": rotation_model,
                    "vector_visual_cue_x": float(visual_cue[0]),
                    "vector_visual_cue_y": float(visual_cue[1]),
                    "vector_proprioceptive_state_x": float(proprioceptive_state[0]),
                    "vector_proprioceptive_state_y": float(proprioceptive_state[1]),
                    "vector_combined_percept_x": float(combined_percept[0]),
                    "vector_combined_percept_y": float(combined_percept[1]),
                    "vector_combined_percept_radial": float(combined_percept[0] - BASELINE[0]),
                    "vector_combined_percept_tangential": float(combined_percept[1] - BASELINE[1]),
                    "vector_teaching_radial": float(teaching_signal[0]),
                    "vector_teaching_tangential": float(teaching_signal[1]),
                    "vector_update_radial": float(model_update[0]),
                    "vector_update_tangential": float(model_update[1]),
                }
            )
            state = next_state

    return pd.DataFrame.from_records(rows)


def simulate_scalar_bcc_percepts(
    max_window: int, params: dict[str, float] | None = None
) -> pd.DataFrame:
    if params is None:
        frames = []
        for dataset_slug in MODEL_DATASET_SLUGS:
            device_params = load_params(MODEL_NAMES["scalar"], dataset_slug)
            device_df = simulate_scalar_bcc_percepts(max_window, device_params)
            device_df["dataset_slug"] = dataset_slug
            frames.append(device_df)
        return _combine_device_predictions(frames)

    prop_var = max(float(params["propVar"]), 1e-9)
    vis_var_slope = max(float(params["visVarSlope"]), 0.0)
    motor_var = max(float(params["motorVar"]), 1e-9)
    retention = float(params["retention"])
    lr = float(params["lr"])

    rows = []
    for clamp in CLAMPS:
        state = 0.0
        rotation_model = -float(clamp)
        vis_var = max((VIS_VAR_INT + vis_var_slope * abs(rotation_model)) ** 2, 1e-9)

        for exposure in range(1, max_window + 1):
            total_precision = 1.0 / vis_var + 1.0 / prop_var + 1.0 / motor_var
            combined_percept = (rotation_model / vis_var + state / prop_var) / total_precision
            teaching_signal = -combined_percept
            next_state = retention * state + lr * teaching_signal
            model_update = next_state - state

            rows.append(
                {
                    "clamp_magnitude": float(clamp),
                    "target_exposure_index": int(exposure),
                    "scalar_combined_percept": float(combined_percept),
                    "scalar_teaching_tangential": float(teaching_signal),
                    "scalar_update_radial": 0.0,
                    "scalar_update_tangential": float(model_update),
                }
            )
            state = next_state

    return pd.DataFrame.from_records(rows)


def simulate_integrated_percept_predictors(max_window: int) -> pd.DataFrame:
    scalar = simulate_scalar_bcc_percepts(max_window)
    vector = simulate_vector_bcc_percepts(max_window)
    return scalar.merge(
        vector, on=["clamp_magnitude", "target_exposure_index"], how="inner", validate="one_to_one"
    )
