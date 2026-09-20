from __future__ import annotations

import numpy as np
import pandas as pd

from clamp_analysis.models.parameters import ParameterSpec
from clamp_analysis.models.state_space import (
    MeanTimeseriesBaseBCC,
    MeanTimeseriesCausalInf,
    MeanTimeseriesVectorBCC,
)

DEVICE_SLUGS = ("mouse", "trackpad")

DEVICE_LABELS = {
    "mouse": "Mouse",
    "trackpad": "Trackpad",
}


def _device_param_name(parameter: str, slug: str) -> str:
    return f"{parameter}_{slug}"


def _spec_by_name(model_cls) -> dict[str, ParameterSpec]:
    return {spec.name: spec for spec in model_cls.param_specs}


def _clone_spec(spec: ParameterSpec, name: str) -> ParameterSpec:
    return ParameterSpec(name, float(spec.lower), float(spec.upper), float(spec.init))


def _subset_slug_from_series(series_df: pd.DataFrame, device_slugs=DEVICE_SLUGS) -> str:
    if "subsetSlug" in series_df.columns:
        values = series_df["subsetSlug"].dropna().astype(str).unique()
    elif "dataset_slug" in series_df.columns:
        values = series_df["dataset_slug"].dropna().astype(str).unique()
    else:
        values = np.asarray([], dtype=str)

    if len(values) != 1:
        raise ValueError(
            "Joint device models require exactly one subsetSlug/dataset_slug per simulated series; "
            f"got {list(values)}."
        )
    slug = str(values[0])
    if slug not in set(device_slugs):
        raise KeyError(f"Unknown device subset '{slug}'. Expected one of {tuple(device_slugs)}.")
    return slug


class _JointDeviceMixin:
    parameterization = "joint_device_fixed_motorVar"
    series_key_cols = ("subsetSlug", "clamp_magnitude")
    device_slugs = DEVICE_SLUGS
    fixed_motor_var_by_subset: dict[str, float] = {}
    device_specific_parameter = ""
    shared_parameter_names: tuple[str, ...] = ()

    @classmethod
    def _params_by_name(cls, params) -> dict[str, float]:
        return {
            spec.name: float(value)
            for spec, value in zip(cls.param_specs, np.asarray(params, dtype=float))
        }

    @classmethod
    def _device_value(
        cls, params_by_name: dict[str, float], parameter: str, subset_slug: str
    ) -> float:
        name = _device_param_name(parameter, subset_slug)
        if name not in params_by_name:
            raise KeyError(f"Missing device-specific parameter '{name}'.")
        return float(params_by_name[name])

    @classmethod
    def fixed_motor_var_for_subset(cls, subset_slug: str) -> float:
        if subset_slug not in cls.fixed_motor_var_by_subset:
            raise KeyError(f"No calibrated motorVar available for subset '{subset_slug}'.")
        return float(cls.fixed_motor_var_by_subset[subset_slug])

    def parameter_summary_table(self):
        if (
            not hasattr(self, "sharedParams")
            or self.sharedParams is None
            or self.sharedParams.empty
        ):
            return super().parameter_summary_table()

        values = self.sharedParams.set_index("parameter")["value"].astype(float).to_dict()
        sd_values = self.sharedParams.set_index("parameter")["sd"].astype(float).to_dict()
        rows: list[dict[str, object]] = []

        for slug in self.device_slugs:
            name = _device_param_name(self.device_specific_parameter, slug)
            if name in values:
                rows.append(
                    {
                        "scope": "device_specific",
                        "parameter": self.device_specific_parameter,
                        "mean": float(values[name]),
                        "sd": float(sd_values.get(name, 0.0)),
                        "fixed": False,
                        "datasetSlug": slug,
                        "datasetLabel": DEVICE_LABELS.get(slug, slug),
                    }
                )

        for parameter in self.shared_parameter_names:
            if parameter in values:
                rows.append(
                    {
                        "scope": "shared",
                        "parameter": parameter,
                        "mean": float(values[parameter]),
                        "sd": float(sd_values.get(parameter, 0.0)),
                        "fixed": False,
                        "datasetSlug": "",
                        "datasetLabel": "Shared",
                    }
                )

        for parameter, value in getattr(self, "fixed_parameter_values", {}).items():
            rows.append(
                {
                    "scope": "fixed",
                    "parameter": str(parameter),
                    "mean": float(value),
                    "sd": 0.0,
                    "fixed": True,
                    "datasetSlug": "",
                    "datasetLabel": "Shared",
                }
            )

        for slug, value in sorted(self.fixed_motor_var_by_subset.items()):
            rows.append(
                {
                    "scope": "fixed_by_subset",
                    "parameter": "motorVar",
                    "mean": float(value),
                    "sd": 0.0,
                    "fixed": True,
                    "datasetSlug": slug,
                    "datasetLabel": DEVICE_LABELS.get(slug, slug),
                }
            )

        return pd.DataFrame.from_records(rows)

    def fit_summary(self):
        summary = super().fit_summary()
        if self.fixed_motor_var_by_subset:
            summary["fixedMotorVarBySubset"] = {
                str(slug): float(value)
                for slug, value in sorted(self.fixed_motor_var_by_subset.items())
            }
        summary["jointDeviceFit"] = True
        summary["deviceSlugs"] = list(self.device_slugs)
        return summary


def make_joint_device_model_classes(
    fixed_motor_var_by_subset: dict[str, float],
    device_slugs: tuple[str, ...] = DEVICE_SLUGS,
    *,
    vector_retention_lower: float | None = None,
):
    fmv_map = {str(slug): float(value) for slug, value in fixed_motor_var_by_subset.items()}
    device_slugs_local = tuple(str(slug) for slug in device_slugs)
    missing_motor = sorted(set(device_slugs_local).difference(fmv_map))
    if missing_motor:
        raise ValueError(f"Missing calibrated motorVar for device subset(s): {missing_motor}.")

    base_specs = _spec_by_name(MeanTimeseriesBaseBCC)
    vector_specs = _spec_by_name(MeanTimeseriesVectorBCC)
    causal_specs = _spec_by_name(MeanTimeseriesCausalInf)

    vector_retention_spec = vector_specs["retention"]
    if vector_retention_lower is not None:
        vector_retention_spec = ParameterSpec(
            "retention",
            float(vector_retention_lower),
            float(vector_retention_spec.upper),
            float(vector_retention_spec.init),
        )

    class MeanTimeseriesJointDeviceBaseBCC(_JointDeviceMixin, MeanTimeseriesBaseBCC):
        model_name = MeanTimeseriesBaseBCC.model_name
        source_model_name = MeanTimeseriesBaseBCC.model_name
        fixed_parameter_values = dict(MeanTimeseriesBaseBCC.fixed_parameter_values)
        fixed_motor_var_by_subset = fmv_map
        device_slugs = device_slugs_local
        device_specific_parameter = "propVar"
        shared_parameter_names = ("visVarSlope", "retention", "lr")
        param_specs = (
            *tuple(
                _clone_spec(base_specs["propVar"], _device_param_name("propVar", slug))
                for slug in device_slugs_local
            ),
            base_specs["visVarSlope"],
            base_specs["retention"],
            base_specs["lr"],
        )

        def _simulate_series(self, params, series_df):
            subset_slug = _subset_slug_from_series(series_df, self.device_slugs)
            by_name = self._params_by_name(params)
            raw_params = np.asarray(
                [
                    self._device_value(by_name, "propVar", subset_slug),
                    by_name["visVarSlope"],
                    self.fixed_motor_var_for_subset(subset_slug),
                    by_name["retention"],
                    by_name["lr"],
                ],
                dtype=float,
            )
            return MeanTimeseriesBaseBCC._simulate_series(self, raw_params, series_df)

    class MeanTimeseriesJointDeviceCausalInf(_JointDeviceMixin, MeanTimeseriesCausalInf):
        model_name = MeanTimeseriesCausalInf.model_name
        source_model_name = MeanTimeseriesCausalInf.model_name
        fixed_parameter_values = {}
        fixed_motor_var_by_subset = {}
        device_slugs = device_slugs_local
        device_specific_parameter = "senseVar"
        shared_parameter_names = ("c", "retention", "lr")
        param_specs = (
            *tuple(
                _clone_spec(causal_specs["senseVar"], _device_param_name("senseVar", slug))
                for slug in device_slugs_local
            ),
            causal_specs["c"],
            causal_specs["retention"],
            causal_specs["lr"],
        )

        def _simulate_series(self, params, series_df):
            subset_slug = _subset_slug_from_series(series_df, self.device_slugs)
            by_name = self._params_by_name(params)
            raw_params = np.asarray(
                [
                    self._device_value(by_name, "senseVar", subset_slug),
                    by_name["c"],
                    by_name["retention"],
                    by_name["lr"],
                ],
                dtype=float,
            )
            return MeanTimeseriesCausalInf._simulate_series(self, raw_params, series_df)

    class MeanTimeseriesJointDeviceVectorBCC(_JointDeviceMixin, MeanTimeseriesVectorBCC):
        model_name = MeanTimeseriesVectorBCC.model_name
        source_model_name = MeanTimeseriesVectorBCC.model_name
        fixed_parameter_values = dict(MeanTimeseriesVectorBCC.fixed_parameter_values)
        fixed_motor_var_by_subset = fmv_map
        device_slugs = device_slugs_local
        device_specific_parameter = "propVar"
        shared_parameter_names = ("visVarSlope", "retention", "lr")
        param_specs = (
            *tuple(
                _clone_spec(vector_specs["propVar"], _device_param_name("propVar", slug))
                for slug in device_slugs_local
            ),
            vector_specs["visVarSlope"],
            vector_retention_spec,
            vector_specs["lr"],
        )

        def _simulate_series(self, params, series_df):
            subset_slug = _subset_slug_from_series(series_df, self.device_slugs)
            by_name = self._params_by_name(params)
            raw_params = np.asarray(
                [
                    self._device_value(by_name, "propVar", subset_slug),
                    by_name["visVarSlope"],
                    self.fixed_motor_var_for_subset(subset_slug),
                    by_name["retention"],
                    by_name["lr"],
                ],
                dtype=float,
            )
            return MeanTimeseriesVectorBCC._simulate_series(self, raw_params, series_df)

    for cls in (
        MeanTimeseriesJointDeviceBaseBCC,
        MeanTimeseriesJointDeviceCausalInf,
        MeanTimeseriesJointDeviceVectorBCC,
    ):
        cls.__name__ = f"{cls.model_name}JointDevice"
        cls.__qualname__ = cls.__name__

    return [
        MeanTimeseriesJointDeviceBaseBCC,
        MeanTimeseriesJointDeviceCausalInf,
        MeanTimeseriesJointDeviceVectorBCC,
    ]


def active_params_to_device_raw(model_cls, params, subset_slug: str) -> pd.Series:
    by_name = {
        spec.name: float(value)
        for spec, value in zip(model_cls.param_specs, np.asarray(params, dtype=float))
    }
    model_name = getattr(model_cls, "source_model_name", getattr(model_cls, "model_name", ""))
    if model_name in {MeanTimeseriesBaseBCC.model_name, MeanTimeseriesVectorBCC.model_name}:
        return pd.Series(
            {
                "propVar": by_name[_device_param_name("propVar", subset_slug)],
                "visVarSlope": by_name["visVarSlope"],
                "motorVar": float(model_cls.fixed_motor_var_by_subset[subset_slug]),
                "retention": by_name["retention"],
                "lr": by_name["lr"],
            },
            dtype=float,
        )
    if model_name == MeanTimeseriesCausalInf.model_name:
        return pd.Series(
            {
                "senseVar": by_name[_device_param_name("senseVar", subset_slug)],
                "c": by_name["c"],
                "retention": by_name["retention"],
                "lr": by_name["lr"],
            },
            dtype=float,
        )
    raise KeyError(f"Unsupported joint device model: {model_name}")
