"""Fit the three joint-device models and export their parameters and predictions."""

import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clamp_analysis import paths
from clamp_analysis.models.fitting import gaussian_summary_negloglik, weighted_rmse
from clamp_analysis.models.joint_device import make_joint_device_model_classes
from clamp_analysis.models.summaries import (
    attach_pointer_device,
    plot_mean_timeseries_fit,
    prepare_mean_timeseries_data,
    summarize_mean_timeseries,
)

SEED = 42

DATA_PATH = paths.DATA_PATH

DETAILS_PATH = paths.DETAILS_PATH

HUMAN_VALUE_COL = "recentred_hand_angle"

EXCLUDED_CONDITIONS = ("NoFeedbackBaseline",)

FIT_COLLECTION_NAME = "mean_timeseries_shared_point_joint_device_motorvar"

JOINT_SUBSET_SLUG = "joint_devices"

JOINT_SUBSET_LABEL = "Mouse + Trackpad joint"

DEVICE_DATASET_SPECS = [
    {"label": "Mouse", "pointer_device": "Mouse", "slug": "mouse"},
    {"label": "Trackpad", "pointer_device": "Trackpad", "slug": "trackpad"},
]

DEVICE_SLUGS = tuple(spec["slug"] for spec in DEVICE_DATASET_SPECS)

FIXED_PARAMETER_SETTINGS = {
    "MeanTimeseriesBaseBCC": {
        "visVarInt": 1.853,
        "motorVar": "calibrated_no_feedback_by_device",
        "shared": ["visVarSlope", "retention", "lr"],
        "deviceSpecific": ["propVar"],
    },
    "MeanTimeseriesCausalInf": {
        "shared": ["c", "retention", "lr"],
        "deviceSpecific": ["senseVar"],
    },
    "MeanTimeseriesVectorBCC": {
        "visVarInt": 1.853,
        "motorVar": "calibrated_no_feedback_by_device",
        "shared": ["visVarSlope", "retention", "lr"],
        "deviceSpecific": ["propVar"],
    },
}

USE_BASELINE_MOTORVAR_CALIBRATION = True

MOTORVAR_CALIBRATION_CONDITION = "NoFeedbackBaseline"

MOTORVAR_CALIBRATION_VALUE_COL = "recentred_hand_angle"

MOTORVAR_CALIBRATION_ABS_ANGLE_CLIP_DEG = 60.0

MOTORVAR_CALIBRATION_GROUP_COLS = ["ppid", "target_angles_degrees", "clamp_magnitude"]

MOTORVAR_CALIBRATION_SCOPE = "subset"

MOTORVAR_CALIBRATION_AGGREGATION = "median_group_variance"

MODEL_LABELS = {
    "MeanTimeseriesBaseBCC": "BaseBCC (joint device, shared dynamics, calibrated motorVar)",
    "MeanTimeseriesCausalInf": "Causal inference (joint device, shared dynamics)",
    "MeanTimeseriesVectorBCC": "Vector BCC (joint device, shared dynamics, calibrated motorVar)",
}


def run(*, output_dir=None, cores=1, restarts=20, polish=10, max_evals=6000):
    output_dir = Path(output_dir) if output_dir is not None else paths.FIT_COLLECTION

    matplotlib.use("Agg")

    N_CORES = cores

    OUTPUT_DIR = output_dir

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    SHARED_POINT_FIT_CONFIG = {
        "fit_mode": "point",
        "objective_name": "negll",
        "nRestarts": restarts,
        "nPolish": polish,
        "maxfevals": max_evals,
        "num_cores": 1,
        "restart_num_cores": int(N_CORES),
        "seed": SEED,
    }

    raw = prepare_mean_timeseries_data(DATA_PATH, seed=SEED)

    raw_with_device = attach_pointer_device(raw, details_path=DETAILS_PATH)

    motorvar_calibration_df = estimate_baseline_motorvar_by_device(raw_with_device)

    motorvar_calibration_df.to_csv(OUTPUT_DIR / "baseline_motorvar_calibration.csv", index=False)

    fixed_motor_var_by_subset = dict(
        zip(
            motorvar_calibration_df["subsetSlug"],
            motorvar_calibration_df["fixedMotorVar"],
            strict=True,
        )
    )

    joint_model_classes = make_joint_device_model_classes(
        fixed_motor_var_by_subset, device_slugs=DEVICE_SLUGS
    )

    dataset_summaries = {}

    dataset_overview_rows = []

    summary_dir = OUTPUT_DIR / "data_summaries"

    summary_dir.mkdir(parents=True, exist_ok=True)

    for spec in DEVICE_DATASET_SPECS:
        subset_label = spec["label"]
        pointer_device = spec["pointer_device"]
        subset_slug = spec["slug"]

        subset_raw = raw_with_device.loc[
            raw_with_device["pointer_device"].eq(pointer_device)
        ].copy()
        summary_df = summarize_mean_timeseries(subset_raw, value_col=HUMAN_VALUE_COL)
        summary_df["subsetSlug"] = subset_slug
        summary_df["subsetLabel"] = subset_label
        dataset_summaries[subset_slug] = summary_df
        summary_df.to_csv(summary_dir / f"{subset_slug}_mean_timeseries_summary.csv", index=False)

        dataset_overview_rows.append(
            {
                "subsetLabel": subset_label,
                "subsetSlug": subset_slug,
                "pointer_device": pointer_device,
                "n_participants": int(subset_raw["ppid"].nunique()),
                "n_summary_rows": int(len(summary_df)),
                "n_clamps": int(summary_df["clamp_magnitude"].nunique()),
                "min_cycle_wrt_clamp": int(summary_df["cycle_wrt_clamp"].min()),
                "max_cycle_wrt_clamp": int(summary_df["cycle_wrt_clamp"].max()),
            }
        )

    joint_summary_df = pd.concat(
        [dataset_summaries[slug] for slug in DEVICE_SLUGS], axis=0, ignore_index=True, sort=False
    )

    joint_summary_df["ppid"] = JOINT_SUBSET_SLUG

    joint_summary_df["group"] = 0

    joint_summary_df.to_csv(
        summary_dir / f"{JOINT_SUBSET_SLUG}_mean_timeseries_summary.csv", index=False
    )

    dataset_overview_df = pd.DataFrame.from_records(dataset_overview_rows)

    dataset_overview_df.to_csv(OUTPUT_DIR / "dataset_overview.csv", index=False)

    RUN_METADATA = {
        "seed": SEED,
        "nCores": int(N_CORES),
        "parallelism": "restart searches only; objective evaluations use one core",
        "fitCollectionName": FIT_COLLECTION_NAME,
        "dataPath": str(DATA_PATH),
        "detailsPath": str(DETAILS_PATH),
        "humanValueCol": HUMAN_VALUE_COL,
        "excludedConditions": EXCLUDED_CONDITIONS,
        "fixedParameterSettings": FIXED_PARAMETER_SETTINGS,
        "useBaselineMotorVarCalibration": bool(USE_BASELINE_MOTORVAR_CALIBRATION),
        "motorVarCalibrationCondition": MOTORVAR_CALIBRATION_CONDITION,
        "motorVarCalibrationValueCol": MOTORVAR_CALIBRATION_VALUE_COL,
        "motorVarCalibrationAbsAngleClipDeg": MOTORVAR_CALIBRATION_ABS_ANGLE_CLIP_DEG,
        "motorVarCalibrationGroupCols": MOTORVAR_CALIBRATION_GROUP_COLS,
        "motorVarCalibrationScope": MOTORVAR_CALIBRATION_SCOPE,
        "motorVarCalibrationAggregation": MOTORVAR_CALIBRATION_AGGREGATION,
        "jointSubsetSlug": JOINT_SUBSET_SLUG,
        "deviceSlugs": list(DEVICE_SLUGS),
        "fitConfig": SHARED_POINT_FIT_CONFIG,
    }

    with open(OUTPUT_DIR / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(RUN_METADATA, f, indent=2)

    fit_results = {}

    comparison_rows = []

    parameter_tables = []

    joint_results = {}

    for model_cls in joint_model_classes:
        fit_info = run_joint_mean_timeseries_model(
            model_cls,
            joint_summary_df=joint_summary_df,
            fit_config=SHARED_POINT_FIT_CONFIG,
            output_dir=OUTPUT_DIR,
            raw_with_device=raw_with_device,
            dataset_overview_df=dataset_overview_df,
        )
        joint_results[model_cls.model_name] = fit_info
        comparison_rows.append(fit_info["summary"])
        table = fit_info["parameter_summary_df"].copy()
        table["subsetLabel"] = JOINT_SUBSET_LABEL
        table["subsetSlug"] = JOINT_SUBSET_SLUG
        table["model"] = fit_info["summary"]["runLabel"]
        table["model_name"] = model_cls.model_name
        parameter_tables.append(table)

    fit_results[JOINT_SUBSET_LABEL] = joint_results

    comparison_df = (
        pd.DataFrame(comparison_rows).sort_values(["negLogLik", "BIC"]).reset_index(drop=True)
    )

    comparison_df.to_csv(OUTPUT_DIR / "model_comparison_joint_device.csv", index=False)

    comparison_df.to_csv(OUTPUT_DIR / "model_comparison_by_subset.csv", index=False)

    all_parameter_tables = pd.concat(parameter_tables, ignore_index=True)

    all_parameter_tables.to_csv(OUTPUT_DIR / "parameter_tables_joint_device.csv", index=False)

    all_parameter_tables.to_csv(OUTPUT_DIR / "parameter_tables_by_subset.csv", index=False)


def estimate_baseline_motorvar_by_device(raw_df):
    if not USE_BASELINE_MOTORVAR_CALIBRATION:
        return pd.DataFrame(
            columns=[
                "subsetSlug",
                "subsetLabel",
                "pointerDevice",
                "calibrationScope",
                "clampMagnitude",
                "fixedMotorVar",
            ]
        )

    working = raw_df.copy()
    working["ppid"] = working["ppid"].astype(str)
    working["_motor_calibration_value"] = pd.to_numeric(
        working[MOTORVAR_CALIBRATION_VALUE_COL], errors="coerce"
    )
    baseline = (
        working.loc[working["condition"].eq(MOTORVAR_CALIBRATION_CONDITION)]
        .dropna(subset=["_motor_calibration_value"])
        .copy()
    )
    if MOTORVAR_CALIBRATION_ABS_ANGLE_CLIP_DEG is not None:
        clip_deg = float(MOTORVAR_CALIBRATION_ABS_ANGLE_CLIP_DEG)
        baseline = baseline.loc[baseline["_motor_calibration_value"].abs().le(clip_deg)].copy()

    missing_group_cols = [
        col for col in MOTORVAR_CALIBRATION_GROUP_COLS if col not in baseline.columns
    ]
    if missing_group_cols:
        raise ValueError(f"Missing motor calibration grouping columns: {missing_group_cols}")
    required_group_cols = {"ppid", "target_angles_degrees", "clamp_magnitude"}
    missing_required = sorted(required_group_cols.difference(MOTORVAR_CALIBRATION_GROUP_COLS))
    if missing_required:
        raise ValueError(
            "MOTORVAR_CALIBRATION_GROUP_COLS must keep participant-target-clamp streams separate; "
            f"missing {missing_required}."
        )

    rows = []
    for spec in DEVICE_DATASET_SPECS:
        subset_slug = spec["slug"]
        subset_label = spec["label"]
        pointer_device = spec["pointer_device"]
        subset = baseline.loc[baseline["pointer_device"].eq(pointer_device)].copy()
        if subset.empty:
            raise ValueError(f"No no-feedback baseline rows available for subset '{subset_slug}'.")

        group_var = (
            subset.groupby(MOTORVAR_CALIBRATION_GROUP_COLS, dropna=False)[
                "_motor_calibration_value"
            ]
            .agg(n="count", variance=lambda x: float(np.var(x, ddof=1)) if len(x) > 1 else np.nan)
            .reset_index()
        )
        valid_var = group_var.loc[group_var["n"].ge(2) & np.isfinite(group_var["variance"])].copy()
        if valid_var.empty:
            raise ValueError(f"No valid within-stream variance groups for subset '{subset_slug}'.")

        variances = valid_var["variance"].to_numpy(dtype=float)
        if MOTORVAR_CALIBRATION_AGGREGATION == "median_group_variance":
            fixed_motor_var = float(np.median(variances))
        elif MOTORVAR_CALIBRATION_AGGREGATION == "mean_group_variance":
            fixed_motor_var = float(np.mean(variances))
        else:
            raise ValueError("Unsupported MOTORVAR_CALIBRATION_AGGREGATION.")

        rows.append(
            {
                "subsetSlug": subset_slug,
                "subsetLabel": subset_label,
                "pointerDevice": str(pointer_device),
                "calibrationScope": MOTORVAR_CALIBRATION_SCOPE,
                "clampMagnitude": np.nan,
                "fixedMotorVar": fixed_motor_var,
                "nBaselineRows": int(len(subset)),
                "nParticipants": int(subset["ppid"].nunique()),
                "nVarianceGroups": int(len(valid_var)),
                "medianGroupN": float(np.median(valid_var["n"])),
                "meanGroupVariance": float(np.mean(variances)),
                "medianGroupVariance": float(np.median(variances)),
                "q25GroupVariance": float(np.percentile(variances, 25)),
                "q75GroupVariance": float(np.percentile(variances, 75)),
                "absAngleClipDeg": None
                if MOTORVAR_CALIBRATION_ABS_ANGLE_CLIP_DEG is None
                else float(MOTORVAR_CALIBRATION_ABS_ANGLE_CLIP_DEG),
                "groupCols": ",".join(MOTORVAR_CALIBRATION_GROUP_COLS),
                "valueCol": MOTORVAR_CALIBRATION_VALUE_COL,
                "condition": MOTORVAR_CALIBRATION_CONDITION,
                "aggregation": MOTORVAR_CALIBRATION_AGGREGATION,
            }
        )

    return pd.DataFrame.from_records(rows)


def add_information_criteria(summary_dict):
    summary = dict(summary_dict)
    n = max(int(summary["n_summary_rows"]), 1)
    k = int(summary.get("n_population_params", summary["n_params_total"]))
    summary["AIC"] = 2.0 * k + 2.0 * float(summary["negLogLik"])
    summary["BIC"] = math.log(n) * k + 2.0 * float(summary["negLogLik"])
    return summary


def fixed_parameters_for_model(model):
    fixed_parameters = dict(FIXED_PARAMETER_SETTINGS.get(model.model_name, {}))
    if getattr(model, "fixed_motor_var_by_subset", None):
        fixed_parameters["motorVarBySubset"] = {
            str(slug): float(value)
            for slug, value in sorted(model.fixed_motor_var_by_subset.items())
        }
    return fixed_parameters


def prediction_fit_summary(model, prediction_df, run_title, subset_label, subset_slug):
    pred = prediction_df.copy()
    observed = pd.to_numeric(pred["late_mean"], errors="coerce").to_numpy(dtype=float)
    predicted = pd.to_numeric(pred["predicted_late_mean"], errors="coerce").to_numpy(dtype=float)
    obs_var = pd.to_numeric(pred["obs_var"], errors="coerce").to_numpy(dtype=float)
    n_obs = pd.to_numeric(pred["n_obs"], errors="coerce").to_numpy(dtype=float)
    neg_log_lik = gaussian_summary_negloglik(observed, predicted, obs_var)
    summary = {
        "model": model.model_name,
        "objectiveName": getattr(model, "population_objective_name", "negll"),
        "fitObjective": float(neg_log_lik),
        "negLogLik": float(neg_log_lik),
        "rmse": float(weighted_rmse(observed, predicted, n_obs)),
        "n_summary_rows": int(len(pred)),
        "n_params_total": int(len(model.param_specs)),
        "n_population_params": int(len(model.param_specs)),
        "fitMode": "shared_point_joint_device",
        "fixedParamNames": [spec.name for spec in model.param_specs],
        "runLabel": run_title,
        "subsetLabel": subset_label,
        "subsetSlug": subset_slug,
        "jointDeviceFit": True,
        "jointSubsetSlug": JOINT_SUBSET_SLUG,
        "deviceSlugs": list(DEVICE_SLUGS),
        "fixedParameters": json.dumps(fixed_parameters_for_model(model), sort_keys=True),
    }
    if getattr(model, "fixed_parameter_values", None):
        summary["fixedParameterValues"] = {
            str(name): float(value) for name, value in model.fixed_parameter_values.items()
        }
    if getattr(model, "fixed_motor_var_by_subset", None):
        summary["fixedMotorVarBySubset"] = {
            str(slug): float(value)
            for slug, value in sorted(model.fixed_motor_var_by_subset.items())
        }
    return add_information_criteria(summary)


def _weighted_mean(values, weights):
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0)
    mask = values.notna() & weights.gt(0.0)
    if mask.any():
        return float(
            np.average(
                values.loc[mask].to_numpy(dtype=float),
                weights=weights.loc[mask].to_numpy(dtype=float),
            )
        )
    return float(values.mean())


def mean_timeseries_plot_frame(df):
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
        if col in df.columns
    ]
    if not key_cols or not df.duplicated(key_cols, keep=False).any():
        return df.copy()

    numeric_mean_cols = [
        col
        for col in [
            "late_mean",
            "mean_hand_angle",
            "predicted_late_mean",
            "model_late_mean",
            "mean_sd",
            "mean_sem",
            "obs_var",
            "rotation_model",
            "target_angles_degrees",
            "target_clamp_sign",
            "clamp_sign",
            "raw_trial_num_in_block",
        ]
        if col in df.columns
    ]
    passthrough_cols = [
        col for col in df.columns if col not in set(key_cols).union(numeric_mean_cols, {"n_obs"})
    ]

    rows = []
    for key, gdf in df.groupby(key_cols, dropna=False, sort=True, observed=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(key_cols, key, strict=True))
        weights = pd.to_numeric(
            gdf.get("n_obs", pd.Series(1.0, index=gdf.index)), errors="coerce"
        ).fillna(1.0)
        row["n_obs"] = int(round(float(weights.sum())))
        for col in numeric_mean_cols:
            row[col] = _weighted_mean(gdf[col], weights)
        for col in passthrough_cols:
            row[col] = gdf[col].iloc[0]
        rows.append(row)

    sort_cols = [col for col in ["clamp_magnitude", "sequence_trial"] if col in key_cols]
    out = pd.DataFrame.from_records(rows)
    if sort_cols:
        out = out.sort_values(sort_cols)
    return out.reset_index(drop=True)


def save_mean_timeseries_fit_outputs(
    model, output_dir, run_title, prediction_df=None, summary_override=None
):
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = output_dir / f"{model.model_name}_posterior_predictions.csv"
    shared_param_path = output_dir / f"{model.model_name}_shared_params.csv"
    param_summary_path = output_dir / f"{model.model_name}_parameter_summary.csv"
    summary_csv_path = output_dir / f"{model.model_name}_fit_summary.csv"
    summary_json_path = output_dir / f"{model.model_name}_fit_summary.json"
    figure_path = output_dir / f"{model.model_name}_mean_timeseries_fit.png"
    fixed_motorvar_path = output_dir / f"{model.model_name}_fixed_motorvar_by_subset.csv"

    prediction_df = (
        model.posterior_prediction_df.copy() if prediction_df is None else prediction_df.copy()
    )
    prediction_df.to_csv(prediction_path, index=False)
    model.sharedParams.to_csv(shared_param_path, index=False)
    parameter_summary_df = model.parameter_summary_table()
    parameter_summary_df.to_csv(param_summary_path, index=False)

    if getattr(model, "fixed_motor_var_by_subset", None):
        pd.DataFrame.from_records(
            [
                {"subsetSlug": str(slug), "fixedMotorVar": float(value)}
                for slug, value in sorted(model.fixed_motor_var_by_subset.items())
            ]
        ).to_csv(fixed_motorvar_path, index=False)

    summary = (
        add_information_criteria(model.fit_summary())
        if summary_override is None
        else dict(summary_override)
    )
    summary["runLabel"] = run_title
    summary["fixedParameters"] = json.dumps(fixed_parameters_for_model(model), sort_keys=True)
    summary["outputDir"] = str(output_dir)
    pd.DataFrame([summary]).to_csv(summary_csv_path, index=False)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_df = mean_timeseries_plot_frame(prediction_df)
    fig = plot_mean_timeseries_fit(plot_df, plot_df, run_title)
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return (
        summary,
        parameter_summary_df,
        {
            "prediction_path": prediction_path,
            "shared_param_path": shared_param_path,
            "param_summary_path": param_summary_path,
            "summary_csv_path": summary_csv_path,
            "summary_json_path": summary_json_path,
            "figure_path": figure_path,
            "fixed_motorvar_path": fixed_motorvar_path,
        },
    )


def run_joint_mean_timeseries_model(
    model_cls, joint_summary_df, fit_config, *, output_dir, raw_with_device, dataset_overview_df
):
    model = model_cls(joint_summary_df)
    model_label = MODEL_LABELS.get(model.model_name, model.model_name)
    run_title = f"{JOINT_SUBSET_LABEL}: {model_label}"
    fit_output_dir = output_dir / JOINT_SUBSET_SLUG / model.model_name

    print()
    print(f"=== Fitting {run_title} ===")
    model.fit(**fit_config)

    joint_summary, parameter_summary_df, joint_paths = save_mean_timeseries_fit_outputs(
        model,
        fit_output_dir,
        run_title,
    )
    joint_summary["subsetLabel"] = JOINT_SUBSET_LABEL
    joint_summary["subsetSlug"] = JOINT_SUBSET_SLUG
    joint_summary["nParticipantsSubset"] = int(raw_with_device["ppid"].nunique())

    split_summaries = []
    split_paths = {}
    for spec in DEVICE_DATASET_SPECS:
        subset_slug = spec["slug"]
        subset_label = spec["label"]
        pred = model.posterior_prediction_df.loc[
            model.posterior_prediction_df["subsetSlug"].astype(str).eq(subset_slug)
        ].copy()
        split_title = f"{subset_label}: {model_label} (joint device fit)"
        split_summary = prediction_fit_summary(model, pred, split_title, subset_label, subset_slug)
        split_summary["nParticipantsSubset"] = int(
            dataset_overview_df.loc[
                dataset_overview_df["subsetSlug"].eq(subset_slug), "n_participants"
            ].iloc[0]
        )
        _, _, paths = save_mean_timeseries_fit_outputs(
            model,
            output_dir / subset_slug / model.model_name,
            split_title,
            prediction_df=pred,
            summary_override=split_summary,
        )
        split_summaries.append(split_summary)
        split_paths[subset_slug] = paths

    return {
        "model": model,
        "summary": joint_summary,
        "split_summaries": split_summaries,
        "summary_df": pd.DataFrame([joint_summary]),
        "parameter_summary_df": parameter_summary_df,
        "paths": joint_paths,
        "split_paths": split_paths,
    }
