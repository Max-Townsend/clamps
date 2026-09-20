"""Simulate and refit datasets to assess model and parameter recovery."""

import json
import math
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from clamp_analysis import paths
from clamp_analysis.figures.recovery import regenerate_recovery_figures
from clamp_analysis.models.joint_device import (
    active_params_to_device_raw,
    make_joint_device_model_classes,
)
from clamp_analysis.models.state_space import (
    MeanTimeseriesBaseBCC,
    MeanTimeseriesCausalInf,
    MeanTimeseriesVectorBCC,
)

RAW_MODEL_CLASSES = [
    MeanTimeseriesBaseBCC,
    MeanTimeseriesCausalInf,
    MeanTimeseriesVectorBCC,
]

RAW_MODEL_BY_NAME = {model_cls.model_name: model_cls for model_cls in RAW_MODEL_CLASSES}

SEED = 42

PARAMETERIZATION = "joint_device_fixed_motorVar"

DATA_PATH = paths.DATA_PATH

DETAILS_PATH = paths.DETAILS_PATH

SOURCE_FIT_COLLECTION = paths.FITS_DIR / "mean_timeseries_shared_point_joint_device_motorvar"

RECOVERY_COLLECTION_NAME = "mean_timeseries_shared_point_recovery_joint_device_motorvar"

JOINT_SUBSET_SLUG = "joint_devices"

JOINT_SUBSET_LABEL = "Mouse + Trackpad joint"

SUBSET_SLUGS = [JOINT_SUBSET_SLUG]

DEVICE_SUBSET_SLUGS = ["mouse", "trackpad"]

SUBSET_LABEL_BY_SLUG = {
    JOINT_SUBSET_SLUG: JOINT_SUBSET_LABEL,
    "mouse": "Mouse",
    "trackpad": "Trackpad",
}

POINTER_DEVICE_BY_SUBSET = {
    "mouse": "Mouse",
    "trackpad": "Trackpad",
}

PLOT_POOL_DEVICE_SUBSETS = False

PLOT_POOL_SOURCE_SUBSETS = []

PLOT_POOL_SUBSET_SLUG = JOINT_SUBSET_SLUG

PLOT_POOL_SUBSET_LABEL = JOINT_SUBSET_LABEL

USE_BASELINE_MOTORVAR_CALIBRATION = True

MOTORVAR_CALIBRATION_CONDITION = "NoFeedbackBaseline"

MOTORVAR_CALIBRATION_VALUE_COL = "recentred_hand_angle"

MOTORVAR_CALIBRATION_ABS_ANGLE_CLIP_DEG = 60.0

MOTORVAR_CALIBRATION_GROUP_COLS = ["ppid", "target_angles_degrees", "clamp_magnitude"]

MOTORVAR_CALIBRATION_SCOPE = "subset"  # Device-level calibration: one robust motorVar per device.

MOTORVAR_CALIBRATION_AGGREGATION = (
    "median_group_variance"  # robust to rare extreme no-feedback reaches.
)

VECTOR_RETENTION_LOWER_BOUND = 0.5

PARAMETER_RANGE_MODE = "local"

LOG_PARAMETER_NAMES = {
    "propVar_mouse",
    "propVar_trackpad",
    "senseVar_mouse",
    "senseVar_trackpad",
    "c",
}

LOG_SCALE_FACTOR = 4.0

LINEAR_HALF_WIDTH_FRACTION = 0.20

RETENTION_HALF_WIDTH = 0.03

ADD_GAUSSIAN_NOISE = True

SAVE_SYNTHETIC_DATASETS = False

RUN_PRESETS = {
    "smoke": {
        "n_simulations_per_generating_model": 1,
        "fit_config": {
            "nRestarts": 10,
            "nPolish": 0,
            "maxfevals": 700,
        },
    },
    "working": {
        "n_simulations_per_generating_model": 10,
        "fit_config": {
            "nRestarts": 8,
            "nPolish": 2,
            "maxfevals": 3000,
        },
    },
    "publication": {
        "n_simulations_per_generating_model": 20,
        "fit_config": {
            "nRestarts": 20,
            "nPolish": 2,
            "maxfevals": 6000,
        },
    },
}


def run(*, output_dir=None, cores=1, preset="publication", restarts=None, max_evals=None):
    output_dir = Path(output_dir) if output_dir is not None else paths.RECOVERY_COLLECTION

    matplotlib.use("Agg", force=True)

    N_CORES = cores

    OUTPUT_DIR = output_dir

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    RUN_PRESET = preset

    N_SIMULATIONS_PER_GENERATING_MODEL = RUN_PRESETS[RUN_PRESET][
        "n_simulations_per_generating_model"
    ]

    FIT_CONFIG = dict(RUN_PRESETS[RUN_PRESET]["fit_config"])

    if restarts is not None:
        FIT_CONFIG["nRestarts"] = restarts

    if max_evals is not None:
        FIT_CONFIG["maxfevals"] = max_evals

    FIT_CONFIG["num_cores"] = 1

    FIT_CONFIG["restart_num_cores"] = int(N_CORES)

    MODEL_ORDER = [
        "MeanTimeseriesBaseBCC",
        "MeanTimeseriesCausalInf",
        "MeanTimeseriesVectorBCC",
    ]

    RUN_METADATA = {
        "seed": SEED,
        "nCores": int(N_CORES),
        "parallelism": "restart searches only; objective evaluations use one core",
        "parameterization": PARAMETERIZATION,
        "useBaselineMotorVarCalibration": bool(USE_BASELINE_MOTORVAR_CALIBRATION),
        "motorVarCalibrationCondition": MOTORVAR_CALIBRATION_CONDITION,
        "motorVarCalibrationValueCol": MOTORVAR_CALIBRATION_VALUE_COL,
        "motorVarCalibrationAbsAngleClipDeg": MOTORVAR_CALIBRATION_ABS_ANGLE_CLIP_DEG,
        "motorVarCalibrationGroupCols": MOTORVAR_CALIBRATION_GROUP_COLS,
        "motorVarCalibrationScope": MOTORVAR_CALIBRATION_SCOPE,
        "motorVarCalibrationAggregation": MOTORVAR_CALIBRATION_AGGREGATION,
        "vectorRetentionLowerBound": VECTOR_RETENTION_LOWER_BOUND,
        "sourceFitCollection": str(SOURCE_FIT_COLLECTION),
        "recoveryCollectionName": RECOVERY_COLLECTION_NAME,
        "subsetSlugs": SUBSET_SLUGS,
        "deviceSubsetSlugs": DEVICE_SUBSET_SLUGS,
        "plotPoolDeviceSubsets": bool(PLOT_POOL_DEVICE_SUBSETS),
        "plotPoolSourceSubsets": PLOT_POOL_SOURCE_SUBSETS,
        "plotPoolSubsetSlug": PLOT_POOL_SUBSET_SLUG,
        "plotPoolSubsetLabel": PLOT_POOL_SUBSET_LABEL,
        "parameterRangeMode": PARAMETER_RANGE_MODE,
        "logScaleFactor": LOG_SCALE_FACTOR,
        "linearHalfWidthFraction": LINEAR_HALF_WIDTH_FRACTION,
        "retentionHalfWidth": RETENTION_HALF_WIDTH,
        "addGaussianNoise": ADD_GAUSSIAN_NOISE,
        "runPreset": RUN_PRESET,
        "nSimulationsPerGeneratingModel": N_SIMULATIONS_PER_GENERATING_MODEL,
        "fitConfig": FIT_CONFIG,
    }

    with open(OUTPUT_DIR / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(RUN_METADATA, f, indent=2)

    design_by_subset = {slug: load_design_summary(slug) for slug in SUBSET_SLUGS}

    motorvar_calibration_df = load_baseline_motorvar_calibration_from_source_fit()

    motorvar_calibration_df.to_csv(OUTPUT_DIR / "baseline_motorvar_calibration.csv", index=False)

    model_classes_by_subset = {
        slug: build_model_classes_for_subset(slug, motorvar_calibration_df) for slug in SUBSET_SLUGS
    }

    saved_params_by_subset = {
        slug: {
            model_cls.model_name: load_saved_params(model_cls, slug)
            for model_cls in model_classes_by_subset[slug]
        }
        for slug in SUBSET_SLUGS
    }

    fit_param_rows = []

    for subset_slug, model_param_map in saved_params_by_subset.items():
        for model_cls in model_classes_by_subset[subset_slug]:
            model_name = model_cls.model_name
            active_series = model_param_map[model_name]
            for parameter, value in active_series.items():
                device_slug = active_parameter_device_slug(parameter)
                fit_param_rows.append(
                    {
                        "subsetSlug": subset_slug,
                        "subsetLabel": SUBSET_LABEL_BY_SLUG.get(subset_slug, subset_slug),
                        "model": model_name,
                        "sourceModel": source_model_name(model_cls),
                        "parameterization": getattr(
                            model_cls, "parameterization", PARAMETERIZATION
                        ),
                        "parameter": parameter,
                        "baseParameter": active_parameter_base_name(parameter),
                        "deviceSubsetSlug": device_slug,
                        "deviceSubsetLabel": SUBSET_LABEL_BY_SLUG.get(device_slug, "Shared")
                        if device_slug
                        else "Shared",
                        "fittedValue": float(value),
                        "fixed": False,
                    }
                )
            for parameter, value in getattr(model_cls, "fixed_parameter_values", {}).items():
                fit_param_rows.append(
                    {
                        "subsetSlug": subset_slug,
                        "subsetLabel": SUBSET_LABEL_BY_SLUG.get(subset_slug, subset_slug),
                        "model": model_name,
                        "sourceModel": source_model_name(model_cls),
                        "parameterization": "fixed",
                        "parameter": parameter,
                        "baseParameter": parameter,
                        "deviceSubsetSlug": "",
                        "deviceSubsetLabel": "Shared",
                        "fittedValue": float(value),
                        "fixed": True,
                    }
                )
            for device_slug, fixed_value in sorted(
                getattr(model_cls, "fixed_motor_var_by_subset", {}).items()
            ):
                fit_param_rows.append(
                    {
                        "subsetSlug": subset_slug,
                        "subsetLabel": SUBSET_LABEL_BY_SLUG.get(subset_slug, subset_slug),
                        "model": model_name,
                        "sourceModel": source_model_name(model_cls),
                        "parameterization": "fixed_by_device",
                        "parameter": f"motorVar_{device_slug}",
                        "baseParameter": "motorVar",
                        "deviceSubsetSlug": device_slug,
                        "deviceSubsetLabel": SUBSET_LABEL_BY_SLUG.get(device_slug, device_slug),
                        "fittedValue": float(fixed_value),
                        "fixed": True,
                    }
                )

    fitted_parameter_df = pd.DataFrame(fit_param_rows)

    fitted_parameter_df.to_csv(OUTPUT_DIR / "fitted_parameter_centers.csv", index=False)

    RUN_METADATA["motorVarCalibration"] = motorvar_calibration_df.to_dict(orient="records")

    RUN_METADATA["activeModelParameterCounts"] = {
        subset_slug: {
            model_cls.model_name: len(model_cls.param_specs)
            for model_cls in model_classes_by_subset[subset_slug]
        }
        for subset_slug in SUBSET_SLUGS
    }

    with open(OUTPUT_DIR / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(RUN_METADATA, f, indent=2)

    range_tables = []

    for subset_slug, model_param_map in saved_params_by_subset.items():
        for model_cls in model_classes_by_subset[subset_slug]:
            ranges = parameter_ranges_for_model(
                model_cls,
                model_param_map[model_cls.model_name],
                range_mode=PARAMETER_RANGE_MODE,
            )
            ranges.insert(0, "subsetSlug", subset_slug)
            ranges.insert(1, "subsetLabel", SUBSET_LABEL_BY_SLUG.get(subset_slug, subset_slug))
            range_tables.append(ranges)

    parameter_range_df = pd.concat(range_tables, ignore_index=True)

    parameter_range_df.to_csv(OUTPUT_DIR / "parameter_sampling_ranges.csv", index=False)

    rng = np.random.default_rng(SEED)

    fit_rows = []

    true_param_rows = []

    synthetic_output_dir = OUTPUT_DIR / "synthetic_datasets"

    if SAVE_SYNTHETIC_DATASETS:
        synthetic_output_dir.mkdir(parents=True, exist_ok=True)

    for subset_slug in SUBSET_SLUGS:
        subset_label = SUBSET_LABEL_BY_SLUG.get(subset_slug, subset_slug)
        design_df = design_by_subset[subset_slug]
        model_param_map = saved_params_by_subset[subset_slug]
        subset_model_classes = model_classes_by_subset[subset_slug]

        for generating_model_cls in subset_model_classes:
            true_param_df, _ = draw_true_parameter_table(
                generating_model_cls,
                model_param_map[generating_model_cls.model_name],
                n_draws=N_SIMULATIONS_PER_GENERATING_MODEL,
                rng=rng,
                range_mode=PARAMETER_RANGE_MODE,
            )

            for _, true_row in true_param_df.iterrows():
                draw_index = int(true_row["drawIndex"])
                synthetic_id = f"{subset_slug}_{generating_model_cls.model_name}_{draw_index:04d}"
                true_params = np.asarray(
                    [float(true_row[spec.name]) for spec in generating_model_cls.param_specs],
                    dtype=float,
                )
                synthetic_df = simulate_synthetic_summary(
                    generating_model_cls,
                    true_params,
                    design_df,
                    rng=rng,
                    synthetic_id=synthetic_id,
                    add_noise=ADD_GAUSSIAN_NOISE,
                )
                if SAVE_SYNTHETIC_DATASETS:
                    synthetic_df.to_csv(
                        synthetic_output_dir / f"{synthetic_id}.csv.gz",
                        index=False,
                        compression="gzip",
                    )

                true_record = {
                    "subsetSlug": subset_slug,
                    "subsetLabel": subset_label,
                    "syntheticId": synthetic_id,
                    "drawIndex": draw_index,
                    "generatingModel": generating_model_cls.model_name,
                }
                for spec, value in zip(generating_model_cls.param_specs, true_params):
                    true_record[f"true_{spec.name}"] = float(value)
                true_raw_series = active_params_to_raw_series(generating_model_cls, true_params)
                for parameter, value in true_raw_series.items():
                    value = float(value)
                    true_record[f"true_raw_{parameter}"] = value
                    true_record.setdefault(f"true_{parameter}", value)
                true_param_rows.append(true_record)

                print(
                    f"Synthetic dataset {synthetic_id}: generated by {generating_model_cls.model_name}",
                    flush=True,
                )
                for candidate_model_cls in subset_model_classes:
                    print(f"  fitting {candidate_model_cls.model_name}", flush=True)
                    fit_seed = int(rng.integers(1, np.iinfo(np.int32).max))
                    fit_row = fit_candidate_model(
                        candidate_model_cls,
                        synthetic_df,
                        fitted_init_params=model_param_map[candidate_model_cls.model_name],
                        seed=fit_seed,
                        fit_config=FIT_CONFIG,
                    )
                    fit_row.update(true_record)
                    fit_row["fitSeed"] = fit_seed
                    for spec, value in zip(generating_model_cls.param_specs, true_params):
                        fit_row[f"true_{spec.name}"] = float(value)
                    for parameter, value in true_raw_series.items():
                        value = float(value)
                        fit_row[f"true_raw_{parameter}"] = value
                        fit_row.setdefault(f"true_{parameter}", value)
                    fit_rows.append(fit_row)

    fit_results_df = pd.DataFrame.from_records(fit_rows)

    true_parameters_df = pd.DataFrame.from_records(true_param_rows)

    fit_results_path = OUTPUT_DIR / "recovery_fit_results.csv"

    true_parameters_path = OUTPUT_DIR / "recovery_true_parameters.csv"

    fit_results_df.to_csv(fit_results_path, index=False)

    true_parameters_df.to_csv(true_parameters_path, index=False)

    write_recovery_reports(
        fit_results_df, OUTPUT_DIR, model_classes_by_subset, design_by_subset, MODEL_ORDER
    )


def summary_path_for_subset(subset_slug):
    return SOURCE_FIT_COLLECTION / "data_summaries" / f"{subset_slug}_mean_timeseries_summary.csv"


def source_model_name(model_cls):
    return getattr(model_cls, "source_model_name", model_cls.model_name)


def shared_param_path(model_cls, subset_slug):
    model_name = source_model_name(model_cls)
    return SOURCE_FIT_COLLECTION / subset_slug / model_name / f"{model_name}_shared_params.csv"


def load_design_summary(subset_slug):
    path = summary_path_for_subset(subset_slug)
    if not path.exists():
        raise FileNotFoundError(f"Could not find saved mean time-series summary: {path}")
    df = pd.read_csv(path)
    df["ppid"] = df["ppid"].astype(str)
    if "subsetSlug" not in df.columns:
        raise ValueError(
            f"{path} is missing subsetSlug. Rebuild joint-device fits with the fit command; see README.md."
        )
    missing_devices = sorted(
        set(DEVICE_SUBSET_SLUGS).difference(set(df["subsetSlug"].dropna().astype(str)))
    )
    if missing_devices:
        raise ValueError(f"Joint recovery design is missing device subset rows: {missing_devices}")
    return df


def load_baseline_motorvar_calibration_from_source_fit():
    if not USE_BASELINE_MOTORVAR_CALIBRATION:
        return pd.DataFrame(
            columns=[
                "subsetSlug",
                "subsetLabel",
                "pointerDevice",
                "fixedMotorVar",
                "nBaselineRows",
                "nParticipants",
                "nVarianceGroups",
            ]
        )

    path = SOURCE_FIT_COLLECTION / "baseline_motorvar_calibration.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing saved motorVar calibration: {path}. Run the fit command first; see README.md."
        )
    df = pd.read_csv(path)
    required_cols = {"subsetSlug", "fixedMotorVar"}
    missing_cols = sorted(required_cols.difference(df.columns))
    if missing_cols:
        raise ValueError(f"Saved motorVar calibration is missing columns {missing_cols}: {path}")

    df = df.loc[df["subsetSlug"].astype(str).isin(DEVICE_SUBSET_SLUGS)].copy()
    if df.empty:
        raise ValueError(
            f"Saved motorVar calibration has no rows for {DEVICE_SUBSET_SLUGS}: {path}"
        )
    missing_devices = sorted(set(DEVICE_SUBSET_SLUGS).difference(set(df["subsetSlug"].astype(str))))
    if missing_devices:
        raise ValueError(
            f"Saved motorVar calibration is missing device subset(s): {missing_devices}"
        )
    if "calibrationScope" in df.columns:
        bad_scope = sorted(
            set(df["calibrationScope"].dropna().astype(str)).difference(
                {MOTORVAR_CALIBRATION_SCOPE}
            )
        )
        if bad_scope:
            raise ValueError(
                f"Saved motorVar calibration scope does not match the recovery configuration: {bad_scope} vs {MOTORVAR_CALIBRATION_SCOPE}."
            )
    if "aggregation" in df.columns:
        bad_aggregation = sorted(
            set(df["aggregation"].dropna().astype(str)).difference(
                {MOTORVAR_CALIBRATION_AGGREGATION}
            )
        )
        if bad_aggregation:
            raise ValueError(
                "Saved motorVar calibration aggregation does not match the recovery configuration: "
                f"{bad_aggregation} vs {MOTORVAR_CALIBRATION_AGGREGATION}."
            )
    df["fixedMotorVar"] = pd.to_numeric(df["fixedMotorVar"], errors="coerce")
    if df["fixedMotorVar"].isna().any():
        raise ValueError(
            f"Saved motorVar calibration contains nonnumeric fixedMotorVar values: {path}"
        )
    return df.reset_index(drop=True)


def build_model_classes_for_subset(subset_slug, motorvar_calibration_df):
    if subset_slug != JOINT_SUBSET_SLUG:
        raise ValueError(
            f"Joint recovery only supports subset '{JOINT_SUBSET_SLUG}', got '{subset_slug}'."
        )
    fixed_motor_var_by_subset = {
        str(row["subsetSlug"]): float(row["fixedMotorVar"])
        for _, row in motorvar_calibration_df.iterrows()
        if str(row["subsetSlug"]) in set(DEVICE_SUBSET_SLUGS)
    }
    return make_joint_device_model_classes(
        fixed_motor_var_by_subset,
        device_slugs=tuple(DEVICE_SUBSET_SLUGS),
        vector_retention_lower=VECTOR_RETENTION_LOWER_BOUND,
    )


def model_class_for(subset_slug, model_name, model_classes_by_subset):
    for model_cls in model_classes_by_subset[subset_slug]:
        if model_cls.model_name == model_name:
            return model_cls
    raise KeyError(f"No model class named '{model_name}' for subset '{subset_slug}'.")


def default_param_series(model_cls):
    return pd.Series(
        {spec.name: float(spec.init) for spec in model_cls.param_specs},
        name=model_cls.model_name,
        dtype=float,
    )


def load_saved_params(model_cls, subset_slug):
    path = shared_param_path(model_cls, subset_slug)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing fitted params at {path}. Run the fit command before recovery; see README.md."
        )
    df = pd.read_csv(path)
    if "parameter" not in df.columns or "value" not in df.columns:
        raise ValueError(f"Saved parameter file has unexpected columns: {path}")
    values = df.set_index("parameter")["value"].astype(float)
    out = default_param_series(model_cls)
    missing = []
    for spec in model_cls.param_specs:
        if spec.name in values.index:
            out.loc[spec.name] = float(values.loc[spec.name])
        else:
            missing.append(spec.name)
    if missing:
        raise ValueError(
            f"Saved parameter file is missing active joint parameter(s) {missing}: {path}"
        )
    return out


def params_array_from_series(model_cls, params):
    return np.asarray([float(params[spec.name]) for spec in model_cls.param_specs], dtype=float)


def active_parameter_device_slug(parameter):
    parameter = str(parameter)
    for slug in DEVICE_SUBSET_SLUGS:
        suffix = f"_{slug}"
        if parameter.endswith(suffix):
            base = parameter[: -len(suffix)]
            if base in {"propVar", "senseVar"}:
                return slug
    return ""


def active_parameter_base_name(parameter):
    parameter = str(parameter)
    slug = active_parameter_device_slug(parameter)
    if not slug:
        return parameter
    return parameter[: -(len(slug) + 1)]


def active_params_to_raw_series(model_cls, params):
    params = (
        params_array_from_series(model_cls, params)
        if isinstance(params, pd.Series)
        else np.asarray(params, dtype=float)
    )
    values = {spec.name: float(value) for spec, value in zip(model_cls.param_specs, params)}
    for parameter, value in getattr(model_cls, "fixed_parameter_values", {}).items():
        values[str(parameter)] = float(value)
    for device_slug, value in sorted(getattr(model_cls, "fixed_motor_var_by_subset", {}).items()):
        values[f"motorVar_{device_slug}"] = float(value)
    return pd.Series(values, name=model_cls.model_name, dtype=float)


def is_log_parameter(parameter_name, spec):
    return parameter_name in LOG_PARAMETER_NAMES and float(spec.lower) > 0.0


def clipped_interval(lo, hi, spec):
    lo = max(float(spec.lower), float(lo))
    hi = min(float(spec.upper), float(hi))
    if hi > lo:
        return lo, hi

    # Fallback for boundary solutions: make the widest small interval possible around the center.
    span = float(spec.upper - spec.lower)
    center = float(np.clip(0.5 * (lo + hi), spec.lower, spec.upper))
    fallback_width = max(0.02 * span, 1e-8)
    lo = max(float(spec.lower), center - fallback_width)
    hi = min(float(spec.upper), center + fallback_width)
    if hi <= lo:
        lo, hi = float(spec.lower), float(spec.upper)
    return float(lo), float(hi)


def parameter_ranges_for_model(model_cls, fitted_params, range_mode=PARAMETER_RANGE_MODE):
    rows = []
    for spec in model_cls.param_specs:
        center = float(fitted_params.get(spec.name, spec.init))
        center = float(np.clip(center, spec.lower, spec.upper))

        if range_mode == "full":
            lo, hi = float(spec.lower), float(spec.upper)
            sample_space = "log" if is_log_parameter(spec.name, spec) else "linear"
        elif range_mode == "local":
            if is_log_parameter(spec.name, spec):
                safe_center = max(center, float(spec.lower) * 1.01, 1e-12)
                lo, hi = clipped_interval(
                    safe_center / LOG_SCALE_FACTOR, safe_center * LOG_SCALE_FACTOR, spec
                )
                sample_space = "log" if lo > 0.0 else "linear"
            else:
                span = float(spec.upper - spec.lower)
                if spec.name == "retention":
                    half_width = RETENTION_HALF_WIDTH
                else:
                    half_width = LINEAR_HALF_WIDTH_FRACTION * span
                lo, hi = clipped_interval(center - half_width, center + half_width, spec)
                sample_space = "linear"
        else:
            raise ValueError("PARAMETER_RANGE_MODE must be 'local' or 'full'.")

        rows.append(
            {
                "model": model_cls.model_name,
                "parameter": spec.name,
                "lowerBound": float(spec.lower),
                "upperBound": float(spec.upper),
                "fittedCenter": center,
                "sampleLower": float(lo),
                "sampleUpper": float(hi),
                "sampleSpace": sample_space,
                "rangeMode": range_mode,
            }
        )
    return pd.DataFrame(rows)


def draw_true_parameter_table(
    model_cls, fitted_params, n_draws, rng, range_mode=PARAMETER_RANGE_MODE
):
    range_df = parameter_ranges_for_model(model_cls, fitted_params, range_mode=range_mode)
    records = []
    for draw_index in range(int(n_draws)):
        row = {"drawIndex": int(draw_index), "generatingModel": model_cls.model_name}
        for _, range_row in range_df.iterrows():
            lo = float(range_row["sampleLower"])
            hi = float(range_row["sampleUpper"])
            if str(range_row["sampleSpace"]) == "log" and lo > 0.0:
                value = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            else:
                value = float(rng.uniform(lo, hi))
            row[str(range_row["parameter"])] = value
        records.append(row)
    return pd.DataFrame.from_records(records), range_df


def simulate_synthetic_summary(
    generating_model_cls,
    true_params,
    design_df,
    rng,
    synthetic_id,
    add_noise=ADD_GAUSSIAN_NOISE,
):
    generator = generating_model_cls(design_df)
    synthetic_df = design_df.copy().reset_index(drop=True)
    true_params = np.asarray(true_params, dtype=float)
    predicted = np.asarray(generator.predict_rows(true_params, synthetic_df), dtype=float)

    obs_sd = np.sqrt(
        np.clip(
            pd.to_numeric(synthetic_df["obs_var"], errors="coerce").to_numpy(dtype=float),
            1e-9,
            None,
        )
    )
    if add_noise:
        observed = predicted + rng.normal(loc=0.0, scale=obs_sd, size=len(predicted))
    else:
        observed = predicted.copy()

    synthetic_df["ppid"] = str(synthetic_id)
    synthetic_df["group"] = 0
    synthetic_df["late_mean"] = observed
    synthetic_df["mean_hand_angle"] = observed
    synthetic_df["true_late_mean"] = predicted
    synthetic_df["generating_model"] = generating_model_cls.model_name
    synthetic_df["synthetic_id"] = str(synthetic_id)
    return synthetic_df


def information_criteria(neg_log_lik, n_rows, n_params):
    neg_log_lik = float(neg_log_lik)
    n_rows = max(int(n_rows), 1)
    n_params = int(n_params)
    return {
        "AIC": 2.0 * n_params + 2.0 * neg_log_lik,
        "BIC": math.log(n_rows) * n_params + 2.0 * neg_log_lik,
    }


def fit_candidate_model(candidate_model_cls, synthetic_df, fitted_init_params, seed, fit_config):
    model = candidate_model_cls(synthetic_df)
    init_params = params_array_from_series(candidate_model_cls, fitted_init_params)
    kwargs = dict(fit_config)
    kwargs["seed"] = int(seed)

    start_time = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            fit_mode="point",
            objective_name="negll",
            population_init_params=init_params,
            **kwargs,
        )
    elapsed = time.perf_counter() - start_time

    neg_log_lik = float(model.negLogLik)
    criteria = information_criteria(
        neg_log_lik=neg_log_lik,
        n_rows=len(synthetic_df),
        n_params=len(candidate_model_cls.param_specs),
    )
    recovered = model.sharedParams.set_index("parameter")["value"].astype(float).to_dict()
    recovered_active = pd.Series(
        {spec.name: float(recovered[spec.name]) for spec in candidate_model_cls.param_specs},
        dtype=float,
    )
    recovered_raw = active_params_to_raw_series(candidate_model_cls, recovered_active)
    row = {
        "candidateModel": candidate_model_cls.model_name,
        "parameterization": getattr(candidate_model_cls, "parameterization", "raw"),
        "negLogLik": neg_log_lik,
        "rmse": float(model.rmse),
        "elapsedSeconds": float(elapsed),
        "nRows": int(len(synthetic_df)),
        "nParams": int(len(candidate_model_cls.param_specs)),
        **criteria,
    }
    for spec in candidate_model_cls.param_specs:
        row[f"recovered_{spec.name}"] = float(recovered[spec.name])
    for parameter, value in recovered_raw.items():
        value = float(value)
        row[f"recovered_raw_{parameter}"] = value
        row.setdefault(f"recovered_{parameter}", value)
    return row


def _vector_bcc_raw_params_from_row(model_cls, row, prefix, device_slug):
    values = []
    for spec in model_cls.param_specs:
        col = f"{prefix}_{spec.name}"
        if col not in row.index:
            return None
        values.append(float(row[col]))
    active = np.asarray(values, dtype=float)
    raw = active_params_to_device_raw(model_cls, active, device_slug)
    return np.asarray(
        [raw[name] for name in ["propVar", "visVarSlope", "motorVar", "retention", "lr"]],
        dtype=float,
    )


def vector_bcc_effective_terms(raw_params, clamp_grid):
    prop_var, vis_var_slope, motor_var, retention, lr = np.asarray(raw_params, dtype=float)
    vis_var_int = float(MeanTimeseriesVectorBCC.fixed_parameter_values["visVarInt"])
    clamp_grid = np.asarray(clamp_grid, dtype=float)
    vis_var = (vis_var_int + vis_var_slope * np.abs(clamp_grid)) ** 2
    visual_precision = 1.0 / np.clip(vis_var, 1e-12, None)
    proprioceptive_precision = 1.0 / max(float(prop_var), 1e-12)
    motor_prior_precision = 1.0 / max(float(motor_var), 1e-12)
    total_precision = visual_precision + proprioceptive_precision + motor_prior_precision
    w_visual = visual_precision / total_precision
    w_proprioceptive = proprioceptive_precision / total_precision
    w_motor_prior = motor_prior_precision / total_precision
    return pd.DataFrame(
        {
            "clamp": clamp_grid,
            "w_visual": w_visual,
            "w_proprioceptive": w_proprioceptive,
            "w_motor_prior": w_motor_prior,
            "beta_lr_times_w_visual": lr * w_visual,
            "alpha_retention_minus_lr_times_w_proprioceptive": retention - lr * w_proprioceptive,
        }
    )


def write_recovery_reports(
    fit_results_df, OUTPUT_DIR, model_classes_by_subset, design_by_subset, MODEL_ORDER
):
    """Export model/parameter recovery and Vector-BCC identifiability checks."""
    if fit_results_df.empty:
        raise RuntimeError("No recovery fits were run. Run the previous cell first.")

    winner_idx = fit_results_df.groupby(["subsetSlug", "syntheticId"], sort=False)["BIC"].idxmin()

    winners_df = fit_results_df.loc[winner_idx].copy().reset_index(drop=True)

    winners_df = winners_df.rename(columns={"candidateModel": "selectedModel"})

    winners_df.to_csv(OUTPUT_DIR / "model_recovery_winners.csv", index=False)

    confusion_tables = []

    for subset_slug, subset_winners in winners_df.groupby("subsetSlug", sort=False):
        counts = pd.crosstab(
            subset_winners["generatingModel"],
            subset_winners["selectedModel"],
            dropna=False,
        ).reindex(index=MODEL_ORDER, columns=MODEL_ORDER, fill_value=0)
        props = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
        counts.insert(0, "subsetSlug", subset_slug)
        props.insert(0, "subsetSlug", subset_slug)
        counts.insert(1, "stat", "count")
        props.insert(1, "stat", "proportion")
        confusion_tables.extend(
            [
                counts.reset_index(names="generatingModel"),
                props.reset_index(names="generatingModel"),
            ]
        )

    model_recovery_confusion_df = pd.concat(confusion_tables, ignore_index=True)

    model_recovery_confusion_df.to_csv(OUTPUT_DIR / "model_recovery_confusion.csv", index=False)

    param_recovery_rows = []

    same_model_fits = fit_results_df.loc[
        fit_results_df["candidateModel"].eq(fit_results_df["generatingModel"])
    ].copy()

    for _, row in same_model_fits.iterrows():
        model_cls = model_class_for(
            str(row["subsetSlug"]), str(row["generatingModel"]), model_classes_by_subset
        )
        for spec in model_cls.param_specs:
            true_col = f"true_{spec.name}"
            recovered_col = f"recovered_{spec.name}"
            if (
                true_col not in same_model_fits.columns
                or recovered_col not in same_model_fits.columns
            ):
                continue
            true_value = row.get(true_col, np.nan)
            recovered_value = row.get(recovered_col, np.nan)
            if pd.isna(true_value) or pd.isna(recovered_value):
                continue
            param_recovery_rows.append(
                {
                    "subsetSlug": row["subsetSlug"],
                    "subsetLabel": row["subsetLabel"],
                    "syntheticId": row["syntheticId"],
                    "model": row["generatingModel"],
                    "parameter": spec.name,
                    "trueValue": float(true_value),
                    "recoveredValue": float(recovered_value),
                    "error": float(recovered_value - true_value),
                    "absError": float(abs(recovered_value - true_value)),
                }
            )

    parameter_recovery_long_df = pd.DataFrame.from_records(param_recovery_rows)

    parameter_recovery_long_df.to_csv(OUTPUT_DIR / "parameter_recovery_long.csv", index=False)

    summary_rows = []

    for key, gdf in parameter_recovery_long_df.groupby(
        ["subsetSlug", "model", "parameter"], sort=False
    ):
        subset_slug, model_name, parameter = key
        true_values = gdf["trueValue"].to_numpy(dtype=float)
        recovered_values = gdf["recoveredValue"].to_numpy(dtype=float)
        if len(gdf) >= 3 and np.std(true_values) > 0 and np.std(recovered_values) > 0:
            corr = float(np.corrcoef(true_values, recovered_values)[0, 1])
        else:
            corr = np.nan
        summary_rows.append(
            {
                "subsetSlug": subset_slug,
                "subsetLabel": SUBSET_LABEL_BY_SLUG.get(subset_slug, subset_slug),
                "model": model_name,
                "parameter": parameter,
                "n": int(len(gdf)),
                "correlation": corr,
                "bias": float(np.mean(gdf["error"])),
                "mae": float(np.mean(gdf["absError"])),
                "rmse": float(np.sqrt(np.mean(np.square(gdf["error"])))),
            }
        )

    parameter_recovery_summary_df = pd.DataFrame.from_records(summary_rows)

    parameter_recovery_summary_df.to_csv(OUTPUT_DIR / "parameter_recovery_summary.csv", index=False)

    vector_identifiability_rows = []

    vector_effective_rows = []

    vector_alpha_beta_rows = []

    vector_fit_rows = fit_results_df.loc[
        fit_results_df["generatingModel"].eq("MeanTimeseriesVectorBCC")
        & fit_results_df["candidateModel"].eq("MeanTimeseriesVectorBCC")
    ].copy()

    if vector_fit_rows.empty:
        print("No same-model VectorBCC recovery rows available.")
    else:
        clamp_grid = np.asarray([2, 5, 10, 20, 50, 100, 135, 170], dtype=float)

        for _, row in vector_fit_rows.iterrows():
            subset_slug = row["subsetSlug"]
            vector_model_cls = model_class_for(
                str(subset_slug), "MeanTimeseriesVectorBCC", model_classes_by_subset
            )
            design_df = design_by_subset[subset_slug]
            model = vector_model_cls(design_df)
            true_params = np.asarray(
                [float(row[f"true_{spec.name}"]) for spec in vector_model_cls.param_specs],
                dtype=float,
            )
            recovered_params = np.asarray(
                [float(row[f"recovered_{spec.name}"]) for spec in vector_model_cls.param_specs],
                dtype=float,
            )
            true_pred = np.asarray(model.predict_rows(true_params, design_df), dtype=float)
            recovered_pred = np.asarray(
                model.predict_rows(recovered_params, design_df), dtype=float
            )
            pred_diff = recovered_pred - true_pred

            regions = {
                "all": np.ones(len(design_df), dtype=bool),
                "high_clamped": design_df["clamp_magnitude"].gt(100).to_numpy(dtype=bool)
                & design_df["clamped"].to_numpy(dtype=bool),
                "high_late_clamped": (
                    design_df["clamp_magnitude"].gt(100).to_numpy(dtype=bool)
                    & design_df["clamped"].to_numpy(dtype=bool)
                    & design_df["cycle_wrt_clamp"].ge(100).to_numpy(dtype=bool)
                ),
            }
            for region, mask in regions.items():
                values = pred_diff[mask]
                vector_identifiability_rows.append(
                    {
                        "subsetSlug": subset_slug,
                        "syntheticId": row["syntheticId"],
                        "region": region,
                        "nRows": int(mask.sum()),
                        "fitRmseToNoisySyntheticData": float(row["rmse"]),
                        "rmsTrueRecoveredPredictionDiff": float(np.sqrt(np.mean(values**2))),
                        "meanAbsTrueRecoveredPredictionDiff": float(np.mean(np.abs(values))),
                        "maxAbsTrueRecoveredPredictionDiff": float(np.max(np.abs(values))),
                    }
                )

            for device_slug in DEVICE_SUBSET_SLUGS:
                true_raw = _vector_bcc_raw_params_from_row(
                    vector_model_cls, row, "true", device_slug
                )
                recovered_raw = _vector_bcc_raw_params_from_row(
                    vector_model_cls, row, "recovered", device_slug
                )
                if true_raw is None or recovered_raw is None:
                    continue

                true_eff = vector_bcc_effective_terms(true_raw, clamp_grid).set_index("clamp")
                recovered_eff = vector_bcc_effective_terms(recovered_raw, clamp_grid).set_index(
                    "clamp"
                )
                for metric in true_eff.columns:
                    diff = recovered_eff[metric] - true_eff[metric]
                    vector_effective_rows.append(
                        {
                            "subsetSlug": subset_slug,
                            "deviceSubsetSlug": device_slug,
                            "deviceSubsetLabel": SUBSET_LABEL_BY_SLUG.get(device_slug, device_slug),
                            "syntheticId": row["syntheticId"],
                            "metric": metric,
                            "rmsDiffAcrossClampGrid": float(
                                np.sqrt(np.mean(diff.to_numpy(dtype=float) ** 2))
                            ),
                            "meanAbsDiffAcrossClampGrid": float(
                                np.mean(np.abs(diff.to_numpy(dtype=float)))
                            ),
                            "maxAbsDiffAcrossClampGrid": float(
                                np.max(np.abs(diff.to_numpy(dtype=float)))
                            ),
                        }
                    )

                for clamp in clamp_grid:
                    metric_map = {
                        "alpha": "alpha_retention_minus_lr_times_w_proprioceptive",
                        "beta": "beta_lr_times_w_visual",
                    }
                    for short_name, column_name in metric_map.items():
                        true_value = float(true_eff.loc[clamp, column_name])
                        recovered_value = float(recovered_eff.loc[clamp, column_name])
                        vector_alpha_beta_rows.append(
                            {
                                "subsetSlug": subset_slug,
                                "deviceSubsetSlug": device_slug,
                                "deviceSubsetLabel": SUBSET_LABEL_BY_SLUG.get(
                                    device_slug, device_slug
                                ),
                                "syntheticId": row["syntheticId"],
                                "clamp": float(clamp),
                                "metric": short_name,
                                "trueValue": true_value,
                                "recoveredValue": recovered_value,
                                "error": recovered_value - true_value,
                                "absError": abs(recovered_value - true_value),
                            }
                        )

    vector_prediction_equivalence_df = pd.DataFrame.from_records(vector_identifiability_rows)

    vector_effective_equivalence_df = pd.DataFrame.from_records(vector_effective_rows)

    vector_alpha_beta_recovery_df = pd.DataFrame.from_records(vector_alpha_beta_rows)

    if not vector_prediction_equivalence_df.empty:
        vector_prediction_equivalence_df.to_csv(
            OUTPUT_DIR / "vector_bcc_true_recovered_prediction_equivalence.csv", index=False
        )
        vector_prediction_equivalence_summary_df = vector_prediction_equivalence_df.groupby(
            ["subsetSlug", "region"], as_index=False
        ).agg(
            nFits=("syntheticId", "nunique"),
            medianFitRmse=("fitRmseToNoisySyntheticData", "median"),
            medianRmsPredictionDiff=("rmsTrueRecoveredPredictionDiff", "median"),
            maxRmsPredictionDiff=("rmsTrueRecoveredPredictionDiff", "max"),
            medianMaxAbsPredictionDiff=("maxAbsTrueRecoveredPredictionDiff", "median"),
        )
        vector_prediction_equivalence_summary_df.to_csv(
            OUTPUT_DIR / "vector_bcc_true_recovered_prediction_equivalence_summary.csv",
            index=False,
        )

    if not vector_effective_equivalence_df.empty:
        vector_effective_equivalence_df.to_csv(
            OUTPUT_DIR / "vector_bcc_effective_term_equivalence.csv", index=False
        )
        vector_effective_equivalence_summary_df = vector_effective_equivalence_df.groupby(
            ["subsetSlug", "metric"], as_index=False
        ).agg(
            nFits=("syntheticId", "nunique"),
            medianRmsDiff=("rmsDiffAcrossClampGrid", "median"),
            maxRmsDiff=("rmsDiffAcrossClampGrid", "max"),
            medianMaxAbsDiff=("maxAbsDiffAcrossClampGrid", "median"),
        )
        vector_effective_equivalence_summary_df.to_csv(
            OUTPUT_DIR / "vector_bcc_effective_term_equivalence_summary.csv",
            index=False,
        )

    if not vector_alpha_beta_recovery_df.empty:
        vector_alpha_beta_recovery_df.to_csv(
            OUTPUT_DIR / "vector_bcc_effective_alpha_beta_recovery_long.csv", index=False
        )
        alpha_beta_summary_rows = []
        for (subset_slug, metric, clamp), gdf in vector_alpha_beta_recovery_df.groupby(
            ["subsetSlug", "metric", "clamp"], sort=True
        ):
            true_values = gdf["trueValue"].to_numpy(dtype=float)
            recovered_values = gdf["recoveredValue"].to_numpy(dtype=float)
            corr = (
                float(np.corrcoef(true_values, recovered_values)[0, 1])
                if len(gdf) >= 3 and np.std(true_values) > 0 and np.std(recovered_values) > 0
                else np.nan
            )
            alpha_beta_summary_rows.append(
                {
                    "subsetSlug": subset_slug,
                    "subsetLabel": SUBSET_LABEL_BY_SLUG.get(subset_slug, subset_slug),
                    "metric": metric,
                    "clamp": float(clamp),
                    "n": int(len(gdf)),
                    "correlation": corr,
                    "bias": float(np.mean(gdf["error"])),
                    "mae": float(np.mean(gdf["absError"])),
                    "rmse": float(np.sqrt(np.mean(np.square(gdf["error"])))),
                    "trueMin": float(np.min(true_values)),
                    "trueMax": float(np.max(true_values)),
                    "recoveredMin": float(np.min(recovered_values)),
                    "recoveredMax": float(np.max(recovered_values)),
                }
            )
        vector_alpha_beta_recovery_summary_df = pd.DataFrame.from_records(alpha_beta_summary_rows)
        vector_alpha_beta_recovery_summary_df.to_csv(
            OUTPUT_DIR / "vector_bcc_effective_alpha_beta_recovery_summary.csv",
            index=False,
        )

    regenerate_recovery_figures(OUTPUT_DIR)
