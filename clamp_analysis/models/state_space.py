from __future__ import annotations

import numpy as np
import pandas as pd

from clamp_analysis.models.fitting import LateAdaptationMCEMBetaModelBase
from clamp_analysis.models.numerics import wrap_angle
from clamp_analysis.models.parameters import ParameterSpec


class MeanTimeseriesModelBase(LateAdaptationMCEMBetaModelBase):
    """
    Shared-point fit to participant-averaged mean time series per clamp cycle.

    The summary dataframe contains one pseudo-participant whose rows are the
    across-participant mean hand angles for each clamp magnitude and cycle
    relative to clamp onset, excluding the no-feedback baseline. The inherited
    late-summary fitter is reused rowwise: each row has an observed mean
    (late_mean) and an observation variance (obs_var), while
    predict_rows returns the model mean for the corresponding rows.
    """

    sequence_col = "sequence_trial"
    series_key_cols = ("clamp_magnitude",)
    fixed_parameter_values: dict[str, float] = {}

    def fit(
        self,
        fit_mode="point",
        objective_name="negll",
        population_init_params=None,
        **kwargs,
    ):
        fit_mode = str(fit_mode).lower()
        if fit_mode == "population":
            return super().fit(
                population_objective=objective_name,
                population_init_params=population_init_params,
                **kwargs,
            )
        return self.fit_shared_point(
            objective_name=objective_name,
            init_params=population_init_params,
            **kwargs,
        )

    def _simulate_series(self, params, series_df):
        raise NotImplementedError

    def fit_summary(self):
        summary = super().fit_summary()
        if self.fixed_parameter_values:
            summary["fixedParameterValues"] = {
                str(name): float(value) for name, value in self.fixed_parameter_values.items()
            }
        return summary

    def parameter_summary_table(self):
        table = super().parameter_summary_table()
        if table.empty:
            table = pd.DataFrame(columns=["scope", "parameter", "mean", "sd"])
        table = table.copy()
        table["fixed"] = False
        if not self.fixed_parameter_values:
            return table

        fixed_rows = pd.DataFrame.from_records(
            [
                {
                    "scope": "population",
                    "parameter": str(name),
                    "mean": float(value),
                    "sd": 0.0,
                    "fixed": True,
                }
                for name, value in self.fixed_parameter_values.items()
            ]
        )
        return pd.concat([table, fixed_rows], axis=0, ignore_index=True)

    def _predict_batched_rows(self, params, df):
        if df.empty:
            return np.asarray([], dtype=float)

        working = df.copy().reset_index(drop=True)
        # Simulate each series chronologically, then restore the caller's row order.
        working["_row_index"] = np.arange(len(working), dtype=int)
        working[self.sequence_col] = pd.to_numeric(working[self.sequence_col], errors="coerce")

        predicted_blocks = []
        group_cols = list(self.series_key_cols)
        for _, sdf in working.groupby(group_cols, sort=True, observed=False):
            ordered = sdf.sort_values(self.sequence_col).reset_index(drop=True)
            pred = np.asarray(self._simulate_series(params, ordered), dtype=float)
            if pred.shape != (len(ordered),):
                raise ValueError(
                    f"{self.model_name}._simulate_series must return shape ({len(ordered)},), "
                    f"got {pred.shape}."
                )
            block = ordered[["_row_index"]].copy()
            block["predicted_late_mean"] = pred
            predicted_blocks.append(block)

        merged = pd.concat(predicted_blocks, axis=0, ignore_index=True)
        merged = merged.sort_values("_row_index").reset_index(drop=True)
        return merged["predicted_late_mean"].to_numpy(dtype=float)

    def predict_rows(self, params, df):
        return self._predict_batched_rows(params, df)


class MeanTimeseriesBaseBCC(MeanTimeseriesModelBase):
    model_name = "MeanTimeseriesBaseBCC"
    fixed_parameter_values = {
        "visVarInt": 1.853,
    }
    param_specs = (
        ParameterSpec("propVar", 1e-6, 1000.0, 100.0),
        ParameterSpec("visVarSlope", 0.0, 1.0, 0.5),
        ParameterSpec("motorVar", 1e-6, 1000.0, 25.0),
        ParameterSpec("retention", 0.5, 1.0, 0.95),
        ParameterSpec("lr", 1e-6, 1.0, 0.25),
    )

    def _simulate_series(self, params, series_df):
        propVar, visVarSlope, motorVar, retention, lr = np.asarray(params, dtype=float)
        propVar = max(float(propVar), 1e-9)
        visVarInt = max(float(self.fixed_parameter_values["visVarInt"]), 0.1)
        visVarSlope = max(float(visVarSlope), 0.0)
        motorVar = max(float(motorVar), 1e-9)
        retention = float(retention)
        lr = float(lr)

        rotation = pd.to_numeric(series_df["rotation_model"], errors="coerce").to_numpy(dtype=float)
        is_clamped = series_df["clamped"].to_numpy(dtype=bool)
        n_trials = len(series_df)

        prop_pos = 0.0
        predicted = np.zeros(n_trials, dtype=float)

        for trial in range(n_trials):
            predicted[trial] = wrap_angle(prop_pos)
            if trial == n_trials - 1:
                break

            vis_pos = rotation[trial] if is_clamped[trial] else prop_pos
            vis_var = (visVarInt + visVarSlope * abs(vis_pos)) ** 2
            vis_var = max(float(vis_var), 1e-9)
            total_precision = 1.0 / vis_var + 1.0 / propVar + 1.0 / motorVar
            combined_mean = (vis_pos / vis_var + prop_pos / propVar) / total_precision
            teaching_signal = wrap_angle(-combined_mean)
            prop_pos = retention * prop_pos + lr * teaching_signal

        return predicted


class MeanTimeseriesCausalInf(MeanTimeseriesModelBase):
    model_name = "MeanTimeseriesCausalInf"
    param_specs = (
        ParameterSpec("senseVar", 1e-6, 10000.0, 100.0),
        ParameterSpec("c", 1e-6, 100.0, 1.0),
        ParameterSpec("retention", 0.0, 1.0, 0.99),
        ParameterSpec("lr", 0.0, 1.0, 0.05),
    )

    def _simulate_series(self, params, series_df):
        senseVar, c, retention, lr = np.asarray(params, dtype=float)
        senseVar = max(float(senseVar), 1e-9)
        c = max(float(c), 1e-9)
        retention = float(retention)
        lr = float(lr)

        rotation = pd.to_numeric(series_df["rotation_model"], errors="coerce").to_numpy(dtype=float)
        is_clamped = series_df["clamped"].to_numpy(dtype=bool)
        n_trials = len(series_df)

        adapted_state = 0.0
        predicted = np.zeros(n_trials, dtype=float)

        for trial in range(n_trials):
            predicted[trial] = wrap_angle(adapted_state)
            if trial == n_trials - 1:
                break

            vis_pos = rotation[trial] if is_clamped[trial] else adapted_state
            lik_sense = np.exp(-(vis_pos**2) / (2.0 * senseVar)) / np.sqrt(2.0 * np.pi * senseVar)
            vis_coeff = lik_sense / (lik_sense + c)
            adapted_state = adapted_state * retention + (-vis_pos * vis_coeff) * lr

        return predicted


class MeanTimeseriesVectorBCC(MeanTimeseriesModelBase):
    """
    2D latent-state BCC with isotropic Cartesian Bayesian cue integration.

    Visual, proprioceptive, and motor-prior cues are represented as 2D mean
    vectors with isotropic covariance. Under this isotropic assumption, the
    Cartesian posterior mean is the closed-form Bayesian combination of those
    cue means. The latent state is updated with a standard target-centred
    state-space rule in Cartesian coordinates, while the behavioural prediction
    remains angular via atan2 readout of the 2D state.
    """

    model_name = "MeanTimeseriesVectorBCC"
    _IDENTITY_2D = np.eye(2, dtype=float)
    fixed_parameter_values = {
        "visVarInt": 1.853,
    }
    param_specs = (
        ParameterSpec("propVar", 1e-6, 1000.0, 100.0),
        ParameterSpec("visVarSlope", 0.0, 1.0, 0.5),
        ParameterSpec("motorVar", 1e-6, 1000.0, 25.0),
        ParameterSpec("retention", 0.9, 1.0, 0.95),
        ParameterSpec("lr", 1e-6, 1.0, 0.25),
    )

    @staticmethod
    def _angle_to_unit_components(angle_deg):
        angle_rad = np.deg2rad(np.asarray(angle_deg, dtype=float))
        return np.cos(angle_rad), np.sin(angle_rad)

    @staticmethod
    def _state_angle_deg(state_vec):
        return float(np.rad2deg(np.arctan2(float(state_vec[1]), float(state_vec[0]))))

    @classmethod
    def _cartesian_posterior_mean_isotropic(cls, cue_means, cue_variances):
        total_precision = 0.0
        rhs = np.zeros(2, dtype=float)

        for mean_vec, variance in zip(cue_means, cue_variances):
            variance = max(float(variance), 1e-9)
            precision = 1.0 / variance
            total_precision += precision
            rhs += precision * np.asarray(mean_vec, dtype=float)

        posterior_precision = total_precision * cls._IDENTITY_2D
        return np.linalg.solve(posterior_precision, rhs)

    def _simulate_series(self, params, series_df):
        propVar, visVarSlope, motorVar, retention, lr = np.asarray(params, dtype=float)
        propVar = max(float(propVar), 1e-9)
        visVarInt = max(float(self.fixed_parameter_values["visVarInt"]), 0.1)
        visVarSlope = max(float(visVarSlope), 0.0)
        motorVar = max(float(motorVar), 1e-9)
        retention = float(retention)
        lr = float(lr)

        rotation = pd.to_numeric(series_df["rotation_model"], errors="coerce").to_numpy(dtype=float)
        is_clamped = series_df["clamped"].to_numpy(dtype=bool)
        n_trials = len(series_df)

        baseline = np.asarray([1.0, 0.0], dtype=float)
        state = baseline.copy()
        predicted = np.zeros(n_trials, dtype=float)

        for trial in range(n_trials):
            current_angle = self._state_angle_deg(state)
            predicted[trial] = wrap_angle(current_angle)
            if trial == n_trials - 1:
                break

            vis_pos = rotation[trial] if is_clamped[trial] else current_angle
            vis_var = (visVarInt + visVarSlope * abs(vis_pos)) ** 2
            vis_var = max(float(vis_var), 1e-9)

            perceived_hand = self._cartesian_posterior_mean_isotropic(
                cue_means=(
                    np.asarray(self._angle_to_unit_components(vis_pos), dtype=float),
                    state,
                    baseline,
                ),
                cue_variances=(vis_var, propVar, motorVar),
            )

            state = baseline + retention * (state - baseline) - lr * (perceived_hand - baseline)

        return predicted
