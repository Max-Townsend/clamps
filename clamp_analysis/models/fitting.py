from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.optimize import minimize
from scipy.special import betaln, logsumexp
from scipy.stats.qmc import LatinHypercube

from clamp_analysis.models.parameters import ParameterSpec, build_trial_clamp

SUMMARY_SEM_FLOOR_DEG = 1.0


def _format_eta(seconds):
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _evaluate_late_population_samples_worker(payload):
    job_index = payload["job_index"]
    model_cls = payload["model_cls"]
    participant = payload["participant"]
    params_matrix = payload["params_matrix"]
    objective_name = str(payload.get("objective_name", "negll")).lower()

    model = model_cls(pd.DataFrame({"ppid": []}))
    scores = np.empty(len(params_matrix), dtype=float)
    for idx, params in enumerate(params_matrix):
        result = model.evaluate_participant(np.asarray(params, dtype=float), participant)
        if objective_name == "negll":
            scores[idx] = -float(result["neg_log_lik"])
        elif objective_name == "rmse":
            scores[idx] = -float(result["weighted_sse"])
        else:
            raise ValueError(f"Unsupported late population objective '{objective_name}'.")
    return job_index, scores


def _chunk_bounds(n_items, n_chunks):
    n_items = int(n_items)
    n_chunks = max(1, min(int(n_chunks), n_items))
    edges = np.linspace(0, n_items, n_chunks + 1, dtype=int)
    return [
        (int(edges[idx]), int(edges[idx + 1]))
        for idx in range(n_chunks)
        if edges[idx] < edges[idx + 1]
    ]


def _evaluate_late_population_samples_chunk_worker(payload):
    model_cls = payload["model_cls"]
    participants = payload["participants"]
    params_matrix = payload["params_matrix"]
    objective_name = str(payload.get("objective_name", "negll")).lower()

    model = model_cls(pd.DataFrame({"ppid": []}))
    rows = []
    for job_index, participant in participants:
        scores = np.empty(len(params_matrix), dtype=float)
        for idx, params in enumerate(params_matrix):
            result = model.evaluate_participant(np.asarray(params, dtype=float), participant)
            if objective_name == "negll":
                scores[idx] = -float(result["neg_log_lik"])
            elif objective_name == "rmse":
                scores[idx] = -float(result["weighted_sse"])
            else:
                raise ValueError(f"Unsupported late population objective '{objective_name}'.")
        rows.append((job_index, scores))
    return rows


def _evaluate_late_point_participant_chunk_worker(payload):
    model_cls = payload["model_cls"]
    participants = payload["participants"]
    params = payload["params"]

    model = model_cls(pd.DataFrame({"ppid": []}))
    rows = []
    for job_index, participant in participants:
        result = model.evaluate_participant(np.asarray(params, dtype=float), participant)
        rows.append((job_index, result))
    return rows


def _late_shared_point_polish_worker(payload):
    model_cls = payload["model_cls"]
    participants = payload["participants"]
    start_params = np.asarray(payload["start_params"], dtype=float)
    bounds = payload["bounds"]
    objective_name = str(payload.get("objective_name", "negll")).lower()
    objective_num_cores = max(1, int(payload.get("objective_num_cores", 1)))
    maxiter = int(payload["maxiter"])
    maxfun = int(payload["maxfun"])
    label = str(payload.get("label", "restart"))

    lower = np.asarray([float(b[0]) for b in bounds], dtype=float)
    upper = np.asarray([float(b[1]) for b in bounds], dtype=float)

    model = model_cls(pd.DataFrame({"ppid": []}))
    model.participants = list(participants)

    def objective(params):
        return model._point_objective(
            np.asarray(params, dtype=float),
            objective_name=objective_name,
            num_cores=objective_num_cores,
        )

    start_params = np.clip(np.asarray(start_params, dtype=float), lower, upper)
    result = minimize(
        objective,
        start_params,
        method="L-BFGS-B",
        bounds=bounds,
        options=model.shared_point_optimizer_options(maxiter=maxiter, maxfun=maxfun),
    )
    params_hat = np.asarray(
        result.x if np.all(np.isfinite(result.x)) else start_params, dtype=float
    )
    params_hat = np.clip(params_hat, lower, upper)
    value_hat = float(objective(params_hat))
    history_row = {
        "stage": label,
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(getattr(result, "nfev", -1)),
        "objective_value": value_hat,
    }
    return {
        "label": label,
        "params_hat": params_hat,
        "value_hat": value_hat,
        "history_row": history_row,
    }


def prepare_late_adaptation_data(
    data_path: str | Path = "formattedData.csv",
    subset_groups: list[int] | tuple[int, ...] | None = None,
    participants_per_group: int | None = None,
    seed: int = 42,
):
    raw = pd.read_csv(data_path)
    raw["ppid"] = raw["ppid"].astype(str)

    if subset_groups:
        subset_groups = tuple(int(g) for g in subset_groups)
        raw = raw[raw["group"].isin(subset_groups)].copy()

    raw["trial_clamp"] = build_trial_clamp(raw)
    raw["rotation_model"] = -pd.to_numeric(raw["trial_clamp"], errors="coerce").fillna(0.0)
    raw["group"] = pd.to_numeric(raw["group"], errors="coerce")
    raw["clamp_magnitude"] = pd.to_numeric(raw["clamp_magnitude"], errors="coerce")
    raw["target_angles_degrees"] = pd.to_numeric(raw["target_angles_degrees"], errors="coerce")
    raw["target_clamp_sign"] = pd.to_numeric(raw["target_clamp_sign"], errors="coerce")
    raw["trial_num_in_block"] = pd.to_numeric(raw["trial_num_in_block"], errors="coerce")
    raw["cycle_num"] = pd.to_numeric(raw["cycle_num"], errors="coerce")
    raw["recentred_hand_angle"] = pd.to_numeric(raw["recentred_hand_angle"], errors="coerce")
    raw["cycle_wrt_clamp"] = raw["cycle_num"] - 21

    if participants_per_group is not None:
        rng = np.random.default_rng(seed)
        sampled_ppids = (
            raw.groupby("group")["ppid"]
            .apply(
                lambda x: rng.choice(
                    x.unique(),
                    size=min(int(participants_per_group), x.nunique()),
                    replace=False,
                )
            )
            .explode()
            .astype(str)
        )
        raw = raw[raw["ppid"].isin(sampled_ppids.to_numpy())].copy()

    raw = raw.sort_values(["ppid", "trial_num"]).reset_index(drop=True)
    return raw


def gaussian_summary_negloglik(observed, predicted, obs_var):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    obs_var = np.clip(np.asarray(obs_var, dtype=float), 1e-9, None)
    error = observed - predicted
    return float(np.sum(0.5 * np.log(2.0 * np.pi * obs_var) + error**2 / (2.0 * obs_var)))


def weighted_rmse(observed, predicted, weights=None):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if weights is None:
        return float(np.sqrt(np.mean((observed - predicted) ** 2)))
    weights = np.asarray(weights, dtype=float)
    weights = np.clip(weights, 1e-9, None)
    return float(np.sqrt(np.sum(weights * (observed - predicted) ** 2) / np.sum(weights)))


def weighted_sse(observed, predicted, weights=None):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error_sq = (observed - predicted) ** 2
    if weights is None:
        return float(np.sum(error_sq))
    weights = np.asarray(weights, dtype=float)
    weights = np.clip(weights, 1e-9, None)
    return float(np.sum(weights * error_sq))


def build_late_predictive_design(summary_df, clamp_grid):
    clamp_grid = np.asarray(clamp_grid, dtype=float)
    if clamp_grid.ndim != 1 or len(clamp_grid) == 0:
        raise ValueError("clamp_grid must be a non-empty 1D array.")

    design_cols = ["target_angles_degrees", "target_clamp_sign"]
    if summary_df.empty or any(col not in summary_df.columns for col in design_cols):
        raise ValueError("summary_df must contain target_angles_degrees and target_clamp_sign.")

    base_design = (
        summary_df[design_cols].drop_duplicates().sort_values(design_cols).reset_index(drop=True)
    )
    if base_design.empty:
        raise ValueError("No design rows were available to build the predictive curve.")

    base_design["target_angles_degrees"] = pd.to_numeric(
        base_design["target_angles_degrees"], errors="coerce"
    )
    base_design["target_clamp_sign"] = pd.to_numeric(
        base_design["target_clamp_sign"], errors="coerce"
    )
    base_design["clamp_sign"] = base_design["target_clamp_sign"]
    rows = []
    for clamp_mag in clamp_grid:
        block = base_design.copy()
        block["clamp_magnitude"] = float(clamp_mag)
        block["rotation_model"] = -float(clamp_mag) * block["target_clamp_sign"]
        rows.append(block)
    return pd.concat(rows, axis=0, ignore_index=True)


class LateAdaptationMCEMBetaModelBase:
    model_name = "LateAdaptationMCEMBetaModelBase"
    param_specs: tuple[ParameterSpec, ...] = ()

    def __init__(self, summary_df):
        self.summary_df = summary_df.copy()
        if "ppid" in self.summary_df.columns:
            self.summary_df["ppid"] = self.summary_df["ppid"].astype(str)
        self.participants = self._prepare_participants(self.summary_df)

    def predict_rows(self, params, df):
        raise NotImplementedError

    @staticmethod
    def _prepare_participants(summary_df):
        if summary_df.empty:
            return []
        participants = []
        for ppid, pdf in summary_df.groupby("ppid", sort=True):
            participants.append(
                {
                    "ppid": str(ppid),
                    "group": int(pdf["group"].iloc[0]) if "group" in pdf.columns else np.nan,
                    "df": pdf.copy().reset_index(drop=True),
                    "late_mean": pdf["late_mean"].to_numpy(dtype=float),
                    "obs_var": pdf["obs_var"].to_numpy(dtype=float),
                    "n_obs": pdf["n_obs"].to_numpy(dtype=float),
                }
            )
        return participants

    def params_to_unit(self, params):
        params = np.asarray(params, dtype=float)
        unit = []
        for value, spec in zip(params, self.param_specs):
            scale = float(spec.upper - spec.lower)
            if scale <= 0:
                unit.append(0.5)
                continue
            u = (float(value) - float(spec.lower)) / scale
            unit.append(float(np.clip(u, 1e-8, 1.0 - 1e-8)))
        return np.asarray(unit, dtype=float)

    def unit_to_params(self, unit):
        unit = np.asarray(unit, dtype=float)
        params = []
        for ui, spec in zip(unit, self.param_specs):
            ui = float(np.clip(ui, 1e-8, 1.0 - 1e-8))
            params.append(float(spec.lower + (spec.upper - spec.lower) * ui))
        return np.asarray(params, dtype=float)

    def _beta_init_from_params(
        self,
        init_params=None,
        init_concentration=24.0,
        enforce_unimodal=False,
    ):
        if init_params is None:
            init_params = [spec.init for spec in self.param_specs]
        init_params = np.asarray(init_params, dtype=float)
        if init_params.shape != (len(self.param_specs),):
            raise ValueError(
                f"population_init_params must have shape ({len(self.param_specs)},), "
                f"got {init_params.shape}."
            )

        unit = self.params_to_unit(init_params)
        concentration = max(float(init_concentration), 1e-3)
        if enforce_unimodal:
            extra = max(concentration - 2.0, 1e-3)
            alpha = 1.0 + unit * extra
            beta = 1.0 + (1.0 - unit) * extra
        else:
            alpha = np.clip(unit * concentration, 1e-3, None)
            beta = np.clip((1.0 - unit) * concentration, 1e-3, None)
        return np.asarray(alpha, dtype=float), np.asarray(beta, dtype=float)

    def _curve_from_params(self, params, clamp_grid=None, curve_source="deterministic_params"):
        observed = (
            pd.to_numeric(self.summary_df["clamp_magnitude"], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        if len(observed) == 0:
            raise ValueError("No clamp magnitudes were available to build the predictive curve.")
        if clamp_grid is None:
            lo = float(np.min(observed))
            hi = float(np.max(observed))
            n_grid = max(300, int(np.ceil(2.0 * max(1.0, hi - lo))) + 1)
            clamp_grid = np.linspace(lo, hi, n_grid)
        else:
            clamp_grid = np.asarray(clamp_grid, dtype=float)
            if clamp_grid.ndim != 1 or len(clamp_grid) == 0:
                raise ValueError("clamp_grid must be a non-empty 1D array.")

        design_df = build_late_predictive_design(self.summary_df, clamp_grid)
        n_design = len(design_df) // len(clamp_grid)
        predicted = np.asarray(
            self.predict_rows(np.asarray(params, dtype=float), design_df), dtype=float
        )
        mean_curve = predicted.reshape(len(clamp_grid), n_design).mean(axis=1)
        zeros = np.zeros(len(clamp_grid), dtype=float)
        return pd.DataFrame(
            {
                "clamp_magnitude": clamp_grid,
                "mean": mean_curve,
                "sem": zeros,
                "ci_low": mean_curve,
                "ci_high": mean_curve,
                "n_draws": np.ones(len(clamp_grid), dtype=int),
                "curve_source": curve_source,
            }
        )

    def evaluate_participant(self, params, participant):
        predicted = self.predict_rows(params, participant["df"])
        neg_log_lik = gaussian_summary_negloglik(
            participant["late_mean"],
            predicted,
            participant["obs_var"],
        )
        weighted_sse_value = weighted_sse(
            participant["late_mean"],
            predicted,
            participant["n_obs"],
        )
        rmse = weighted_rmse(
            participant["late_mean"],
            predicted,
            participant["n_obs"],
        )
        return {
            "neg_log_lik": float(neg_log_lik),
            "weighted_sse": float(weighted_sse_value),
            "rmse": float(rmse),
            "predicted_late_mean": np.asarray(predicted, dtype=float),
        }

    def _evaluate_params_across_participants(self, params, participants=None, num_cores=1):
        params = np.asarray(params, dtype=float)
        participants = self.participants if participants is None else list(participants)
        if int(num_cores) == 1:
            return [self.evaluate_participant(params, participant) for participant in participants]
        n_jobs = max(1, min(int(num_cores), len(participants)))
        indexed_participants = list(enumerate(participants))
        payloads = []
        for start, end in _chunk_bounds(len(indexed_participants), n_jobs):
            payloads.append(
                {
                    "model_cls": self.__class__,
                    "participants": indexed_participants[start:end],
                    "params": params,
                }
            )
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_evaluate_late_point_participant_chunk_worker)(payload) for payload in payloads
        )
        flat_results = [item for chunk in results for item in chunk]
        return [result for _, result in sorted(flat_results, key=lambda item: item[0])]

    def _point_objective(self, params, objective_name="negll", num_cores=1):
        objective_name = str(objective_name).lower()
        total = 0.0
        eval_results = self._evaluate_params_across_participants(
            params, participants=self.participants, num_cores=num_cores
        )
        for participant, eval_result in zip(self.participants, eval_results):
            if objective_name == "negll":
                total += float(eval_result["neg_log_lik"])
            elif objective_name == "rmse":
                total += float(
                    np.sum((participant["late_mean"] - eval_result["predicted_late_mean"]) ** 2)
                )
            else:
                raise ValueError(f"Unsupported point-fit objective '{objective_name}'.")
        return float(total)

    def _build_point_prediction_frames(self, params, num_cores=1):
        prediction_rows = []
        participant_rows = []
        point_param_rows = []
        eval_results = self._evaluate_params_across_participants(
            params, participants=self.participants, num_cores=num_cores
        )
        for participant, eval_result in zip(self.participants, eval_results):
            participant_rows.append(
                {
                    "ppid": participant["ppid"],
                    "group": participant["group"],
                    "negLogLik": float(eval_result["neg_log_lik"]),
                    "rmse": float(eval_result["rmse"]),
                }
            )
            pdf = participant["df"].copy()
            pdf["predicted_late_mean"] = np.asarray(eval_result["predicted_late_mean"], dtype=float)
            pdf["model_late_mean"] = pdf["predicted_late_mean"]
            prediction_rows.append(pdf)

            row = {"ppid": participant["ppid"], "group": participant["group"]}
            for spec, value in zip(self.param_specs, np.asarray(params, dtype=float)):
                row[spec.name] = float(value)
            point_param_rows.append(row)

        prediction_df = (
            pd.concat(prediction_rows, axis=0, ignore_index=True)
            if prediction_rows
            else pd.DataFrame()
        )
        participant_df = pd.DataFrame.from_records(participant_rows)
        param_df = pd.DataFrame.from_records(point_param_rows)
        return prediction_df, participant_df, param_df

    def shared_point_optimizer_options(self, maxiter, maxfun):
        return {"maxiter": int(maxiter), "maxfun": int(maxfun), "ftol": 1e-9}

    def fit_shared_point(
        self,
        objective_name="negll",
        init_params=None,
        max_participants=None,
        nRestarts=5,
        nPolish=5,
        maxfevals=8000,
        num_cores=1,
        restart_num_cores=1,
        seed=42,
        **unused_kwargs,
    ):
        objective_name = str(objective_name).lower()
        if objective_name not in {"negll", "rmse"}:
            raise ValueError(
                "Late shared-point fitter only supports objective_name in {'negll', 'rmse'}."
            )

        self.participants = self._prepare_participants(self.summary_df)
        if max_participants is not None:
            self.participants = self.participants[: int(max_participants)]
        if not self.participants:
            raise ValueError("No participants available for late shared-point fitting.")
        print(
            f"Fitting {self.model_name} shared point on {len(self.participants)} participants "
            f"using {max(1, int(num_cores))} core(s)...",
            flush=True,
        )

        bounds = [(float(spec.lower), float(spec.upper)) for spec in self.param_specs]
        init_params = np.asarray(
            init_params if init_params is not None else [spec.init for spec in self.param_specs],
            dtype=float,
        )
        if init_params.shape != (len(self.param_specs),):
            raise ValueError(
                f"init_params must have shape ({len(self.param_specs)},), got {init_params.shape}."
            )

        def objective(params):
            return self._point_objective(
                np.asarray(params, dtype=float),
                objective_name=objective_name,
                num_cores=num_cores,
            )

        history_rows = []
        lower = np.asarray([b[0] for b in bounds], dtype=float)
        upper = np.asarray([b[1] for b in bounds], dtype=float)
        span = upper - lower
        maxfun = max(200, int(maxfevals))
        maxiter = max(100, min(int(maxfevals), 1000))
        restart_num_cores = max(1, int(restart_num_cores))

        def polish(start_params, label):
            start_params = np.clip(
                np.asarray(start_params, dtype=float),
                [b[0] for b in bounds],
                [b[1] for b in bounds],
            )
            result = minimize(
                objective,
                start_params,
                method="L-BFGS-B",
                bounds=bounds,
                options=self.shared_point_optimizer_options(maxiter=maxiter, maxfun=maxfun),
            )
            params_hat = np.asarray(
                result.x if np.all(np.isfinite(result.x)) else start_params, dtype=float
            )
            params_hat = np.clip(params_hat, [b[0] for b in bounds], [b[1] for b in bounds])
            value_hat = float(objective(params_hat))
            history_rows.append(
                {
                    "stage": label,
                    "success": bool(result.success),
                    "message": str(result.message),
                    "nfev": int(getattr(result, "nfev", -1)),
                    "objective_value": value_hat,
                }
            )
            return params_hat, value_hat

        best_params, best_value = polish(init_params, "init")
        n_restarts = max(0, int(nRestarts))
        if n_restarts > 0:
            sampler = LatinHypercube(d=len(bounds), seed=int(seed))
            starts_unit = sampler.random(n=n_restarts)
            starts_real = lower[None, :] + starts_unit * span[None, :]
            start_time = time.perf_counter()
            if restart_num_cores > 1 and n_restarts > 1:
                restart_jobs = min(restart_num_cores, n_restarts)
                print(
                    f"  Parallelizing restart searches across {restart_jobs} worker(s); "
                    f"each restart uses 1 core for objective evaluations.",
                    flush=True,
                )
                for batch_start in range(0, n_restarts, restart_jobs):
                    batch_pairs = list(
                        enumerate(
                            starts_real[batch_start : batch_start + restart_jobs],
                            start=batch_start + 1,
                        )
                    )
                    payloads = []
                    for restart_idx, start in batch_pairs:
                        payloads.append(
                            {
                                "model_cls": self.__class__,
                                "participants": self.participants,
                                "start_params": np.asarray(start, dtype=float),
                                "bounds": bounds,
                                "objective_name": objective_name,
                                "objective_num_cores": 1,
                                "maxiter": maxiter,
                                "maxfun": maxfun,
                                "label": f"restart_{restart_idx:02d}",
                            }
                        )
                    batch_results = Parallel(
                        n_jobs=min(restart_jobs, len(payloads)), backend="loky"
                    )(delayed(_late_shared_point_polish_worker)(payload) for payload in payloads)
                    for result in batch_results:
                        history_rows.append(result["history_row"])
                        if result["value_hat"] < best_value:
                            best_params = np.asarray(result["params_hat"], dtype=float)
                            best_value = float(result["value_hat"])
                    completed = batch_start + len(batch_results)
                    elapsed = max(0.0, time.perf_counter() - start_time)
                    avg_time = elapsed / completed
                    eta = avg_time * max(0, n_restarts - completed)
                    print(
                        f"  {self.model_name} late shared L-BFGS-B {completed}/{n_restarts}: "
                        f"best={best_value:.2f}  elapsed={_format_eta(elapsed)}  ETA={_format_eta(eta)}",
                        flush=True,
                    )
            else:
                for restart_idx, start in enumerate(starts_real, start=1):
                    params_hat, value_hat = polish(start, f"restart_{restart_idx:02d}")
                    if value_hat < best_value:
                        best_params, best_value = params_hat, value_hat
                    if restart_idx % 5 == 0 or restart_idx == n_restarts:
                        elapsed = max(0.0, time.perf_counter() - start_time)
                        avg_time = elapsed / restart_idx
                        eta = avg_time * max(0, n_restarts - restart_idx)
                        print(
                            f"  {self.model_name} late shared L-BFGS-B {restart_idx}/{n_restarts}: "
                            f"best={best_value:.2f}  elapsed={_format_eta(elapsed)}  ETA={_format_eta(eta)}",
                            flush=True,
                        )

        self.population_distribution_family = "shared_point"
        self.population_fit_mode = "shared_point"
        self.population_objective_name = objective_name
        self.population_mean_params = np.asarray(best_params, dtype=float)
        self.population_mode_params = self.population_mean_params.copy()
        self.population_sd_params = np.zeros(len(self.param_specs), dtype=float)
        self.population_beta_alpha = None
        self.population_beta_beta = None
        self.population_beta_unimodal = False
        self.population_beta_param_table = None
        self.population_fixed_param_names = [spec.name for spec in self.param_specs]
        self.population_fixed_unit = self.params_to_unit(self.population_mean_params)

        prediction_df, participant_df, param_df = self._build_point_prediction_frames(
            self.population_mean_params,
            num_cores=num_cores,
        )
        self.posterior_prediction_df = prediction_df
        self.posterior_participantParams = param_df
        self.participantParams = participant_df
        self.prediction_df = prediction_df.copy()
        self.history = pd.DataFrame.from_records(history_rows)
        self.fitObjective = float(best_value)
        self.negLogLik = float(
            self._point_objective(
                self.population_mean_params,
                objective_name="negll",
                num_cores=num_cores,
            )
        )
        self.dataNegLogLik = float(self.negLogLik)

        all_observed = []
        all_predicted = []
        eval_results = self._evaluate_params_across_participants(
            self.population_mean_params,
            participants=self.participants,
            num_cores=num_cores,
        )
        for participant, eval_result in zip(self.participants, eval_results):
            all_observed.append(np.asarray(participant["late_mean"], dtype=float))
            all_predicted.append(np.asarray(eval_result["predicted_late_mean"], dtype=float))
        all_observed = np.concatenate(all_observed) if all_observed else np.asarray([], dtype=float)
        all_predicted = (
            np.concatenate(all_predicted) if all_predicted else np.asarray([], dtype=float)
        )
        self.rmse = (
            float(np.sqrt(np.mean((all_observed - all_predicted) ** 2)))
            if len(all_observed)
            else np.nan
        )

        self.n_population_params = int(len(self.param_specs))
        self.n_params_total = int(len(self.param_specs))
        self.sharedParams = pd.DataFrame(
            {
                "parameter": [spec.name for spec in self.param_specs],
                "value": self.population_mean_params,
                "sd": self.population_sd_params,
            }
        )
        return self

    def _sample_beta_population_params(self, alpha, beta, n_samples, seed, fixed_unit=None):
        rng = np.random.default_rng(seed)
        alpha = np.asarray(alpha, dtype=float)
        beta = np.asarray(beta, dtype=float)
        unit = rng.beta(alpha[None, :], beta[None, :], size=(int(n_samples), len(alpha)))
        unit = np.clip(unit, 1e-8, 1.0 - 1e-8)
        if fixed_unit is not None:
            fixed_unit = np.asarray(fixed_unit, dtype=float)
            fixed_mask = np.isfinite(fixed_unit)
            if fixed_unit.shape != (len(alpha),):
                raise ValueError(
                    f"fixed_unit must have shape ({len(alpha)},), got {fixed_unit.shape}."
                )
            unit[:, fixed_mask] = fixed_unit[fixed_mask]
        params = np.vstack([self.unit_to_params(row) for row in unit])
        return unit, params

    def _weighted_beta_fit(
        self,
        unit_values,
        weights,
        init_alpha=1.0,
        init_beta=1.0,
        enforce_unimodal=False,
    ):
        unit_values = np.clip(np.asarray(unit_values, dtype=float), 1e-8, 1.0 - 1e-8)
        weights = np.asarray(weights, dtype=float)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0 or not np.isfinite(weight_sum):
            if enforce_unimodal:
                return max(float(init_alpha), 1.0), max(float(init_beta), 1.0)
            return max(float(init_alpha), 1e-3), max(float(init_beta), 1e-3)

        weights = weights / weight_sum
        log_u = np.log(unit_values)
        log_one_minus_u = np.log1p(-unit_values)

        def objective(log_ab):
            alpha = np.exp(log_ab[0])
            beta = np.exp(log_ab[1])
            log_pdf = (alpha - 1.0) * log_u + (beta - 1.0) * log_one_minus_u - betaln(alpha, beta)
            return float(-np.sum(weights * log_pdf))

        if enforce_unimodal:
            init_alpha = max(float(init_alpha), 1.0)
            init_beta = max(float(init_beta), 1.0)
            bounds = [(0.0, 8.0), (0.0, 8.0)]
        else:
            init_alpha = max(float(init_alpha), 1e-3)
            init_beta = max(float(init_beta), 1e-3)
            bounds = [(-6.0, 8.0), (-6.0, 8.0)]

        res = minimize(
            objective,
            x0=np.log([init_alpha, init_beta]),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-9},
        )
        if not res.success or not np.all(np.isfinite(res.x)):
            return init_alpha, init_beta
        alpha_hat, beta_hat = np.exp(res.x)
        return float(alpha_hat), float(beta_hat)

    def _population_beta_summary(self, alpha, beta):
        alpha = np.asarray(alpha, dtype=float)
        beta = np.asarray(beta, dtype=float)
        mean_u = alpha / (alpha + beta)
        var_u = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1.0))
        lower = np.asarray([spec.lower for spec in self.param_specs], dtype=float)
        scale = np.asarray([spec.upper - spec.lower for spec in self.param_specs], dtype=float)
        mean_params = lower + scale * mean_u
        sd_params = scale * np.sqrt(np.clip(var_u, 0.0, None))
        return mean_params, sd_params

    def _evaluate_population_loglik_matrix(
        self, params_matrix, participants, num_cores=1, objective_name="negll"
    ):
        params_matrix = np.asarray(params_matrix, dtype=float)
        if int(num_cores) == 1:
            payloads = [
                {
                    "job_index": idx,
                    "model_cls": self.__class__,
                    "participant": participant,
                    "params_matrix": params_matrix,
                    "objective_name": objective_name,
                }
                for idx, participant in enumerate(participants)
            ]
            results = [_evaluate_late_population_samples_worker(payload) for payload in payloads]
        else:
            n_jobs = max(1, min(int(num_cores), len(participants)))
            indexed_participants = list(enumerate(participants))
            payloads = []
            for start, end in _chunk_bounds(len(indexed_participants), n_jobs):
                payloads.append(
                    {
                        "model_cls": self.__class__,
                        "participants": indexed_participants[start:end],
                        "params_matrix": params_matrix,
                        "objective_name": objective_name,
                    }
                )
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_evaluate_late_population_samples_chunk_worker)(payload)
                for payload in payloads
            )
            results = [item for chunk in results for item in chunk]
        results = [scores for _, scores in sorted(results, key=lambda item: item[0])]
        return np.vstack(results)

    def fit(
        self,
        population_iterations=10,
        population_samples=1024,
        population_eval_samples=2048,
        population_beta_unimodal=False,
        population_init_params=None,
        population_init_concentration=24.0,
        population_update="mcem_beta",
        population_objective="negll",
        fixed_param_names=None,
        covariance_type="diag",
        max_participants=None,
        num_cores=1,
        seed=42,
        **unused_kwargs,
    ):
        if str(population_update).lower() != "mcem_beta":
            raise ValueError(
                "Late adaptation MCEM fitter only supports population_update='mcem_beta'."
            )
        if str(covariance_type).lower() != "diag":
            raise ValueError("Late adaptation MCEM fitter only supports covariance_type='diag'.")
        population_objective = str(population_objective).lower()
        if population_objective not in {"negll", "rmse"}:
            raise ValueError(
                "Late adaptation MCEM fitter only supports population_objective in "
                "{'negll', 'rmse'}."
            )
        self.participants = self._prepare_participants(self.summary_df)
        if max_participants is not None:
            self.participants = self.participants[: int(max_participants)]

        n_subjects = len(self.participants)
        n_params = len(self.param_specs)
        print(
            f"Fitting {self.model_name} population distribution on {n_subjects} participants "
            f"using {max(1, int(num_cores))} core(s)...",
            flush=True,
        )

        alpha, beta = self._beta_init_from_params(
            init_params=population_init_params,
            init_concentration=population_init_concentration,
            enforce_unimodal=bool(population_beta_unimodal),
        )
        fixed_mask = np.zeros(len(self.param_specs), dtype=bool)
        fixed_unit = None
        if fixed_param_names:
            fixed_names = {str(name) for name in fixed_param_names}
            fixed_mask = np.asarray(
                [spec.name in fixed_names for spec in self.param_specs], dtype=bool
            )
            init_params = np.asarray(
                population_init_params
                if population_init_params is not None
                else [spec.init for spec in self.param_specs],
                dtype=float,
            )
            fixed_unit = np.full(len(self.param_specs), np.nan, dtype=float)
            fixed_unit[fixed_mask] = self.params_to_unit(init_params)[fixed_mask]
        history = []
        fit_start_time = time.perf_counter()

        for iteration in range(int(population_iterations)):
            iter_start_time = time.perf_counter()
            sample_seed = int(seed) + 10_000 * iteration
            unit_matrix, params_matrix = self._sample_beta_population_params(
                alpha,
                beta,
                n_samples=int(population_samples),
                seed=sample_seed,
                fixed_unit=fixed_unit,
            )
            score_matrix = self._evaluate_population_loglik_matrix(
                params_matrix,
                self.participants,
                num_cores=num_cores,
                objective_name=population_objective,
            )
            log_norm = logsumexp(score_matrix, axis=1, keepdims=True)
            weights = np.exp(score_matrix - log_norm)
            aggregate_weights = np.sum(weights, axis=0)

            new_alpha = np.empty_like(alpha)
            new_beta = np.empty_like(beta)
            for param_idx in range(n_params):
                if fixed_mask[param_idx]:
                    new_alpha[param_idx] = alpha[param_idx]
                    new_beta[param_idx] = beta[param_idx]
                else:
                    new_alpha[param_idx], new_beta[param_idx] = self._weighted_beta_fit(
                        unit_matrix[:, param_idx],
                        aggregate_weights,
                        init_alpha=alpha[param_idx],
                        init_beta=beta[param_idx],
                        enforce_unimodal=bool(population_beta_unimodal),
                    )

            marginal_objective = float(-np.sum(log_norm[:, 0] - np.log(score_matrix.shape[1])))
            alpha_shift = float(np.linalg.norm(new_alpha - alpha))
            beta_shift = float(np.linalg.norm(new_beta - beta))
            alpha, beta = new_alpha, new_beta
            iter_elapsed = max(0.0, time.perf_counter() - iter_start_time)
            total_elapsed = max(0.0, time.perf_counter() - fit_start_time)
            completed = iteration + 1
            avg_iter_time = total_elapsed / completed
            remaining_iters = max(0, int(population_iterations) - completed)
            eta_seconds = avg_iter_time * remaining_iters
            history.append(
                {
                    "iteration": completed,
                    "marginal_objective": marginal_objective,
                    "objective_name": population_objective,
                    "alpha_shift": alpha_shift,
                    "beta_shift": beta_shift,
                    "iter_elapsed_sec": iter_elapsed,
                    "elapsed_sec": total_elapsed,
                    "eta_sec": eta_seconds,
                }
            )
            print(
                f"  {self.model_name} MCEM iter {completed}/{population_iterations}: "
                f"marginal_{population_objective}={marginal_objective:.2f}  alpha_shift={alpha_shift:.4f}  "
                f"beta_shift={beta_shift:.4f}  iter={_format_eta(iter_elapsed)}  "
                f"elapsed={_format_eta(total_elapsed)}  ETA={_format_eta(eta_seconds)}",
                flush=True,
            )

        final_draws = max(int(population_eval_samples), int(population_samples))
        final_unit_matrix, final_params_matrix = self._sample_beta_population_params(
            alpha,
            beta,
            n_samples=final_draws,
            seed=int(seed) + 999_999,
            fixed_unit=fixed_unit,
        )
        final_score_matrix = self._evaluate_population_loglik_matrix(
            final_params_matrix,
            self.participants,
            num_cores=num_cores,
            objective_name=population_objective,
        )
        final_log_norm = logsumexp(final_score_matrix, axis=1, keepdims=True)
        marginal_objective_per_subject = -(
            final_log_norm[:, 0] - np.log(final_score_matrix.shape[1])
        )
        if population_objective == "negll":
            marginal_nll_per_subject = marginal_objective_per_subject.copy()
        else:
            final_negll_matrix = self._evaluate_population_loglik_matrix(
                final_params_matrix,
                self.participants,
                num_cores=num_cores,
                objective_name="negll",
            )
            final_negll_log_norm = logsumexp(final_negll_matrix, axis=1, keepdims=True)
            marginal_nll_per_subject = -(
                final_negll_log_norm[:, 0] - np.log(final_negll_matrix.shape[1])
            )

        participant_rows = []
        for participant_idx, participant in enumerate(self.participants):
            participant_rows.append(
                {
                    "ppid": participant["ppid"],
                    "group": participant["group"],
                    "marginalObjective": float(marginal_objective_per_subject[participant_idx]),
                    "objective_name": population_objective,
                }
            )

        self.population_distribution_family = "beta_factorized_real"
        self.population_beta_alpha = np.asarray(alpha, dtype=float)
        self.population_beta_beta = np.asarray(beta, dtype=float)
        self.population_beta_unimodal = bool(population_beta_unimodal)
        self.population_fixed_param_names = [
            spec.name for spec, is_fixed in zip(self.param_specs, fixed_mask) if is_fixed
        ]
        self.population_fixed_unit = (
            None if fixed_unit is None else np.asarray(fixed_unit, dtype=float)
        )
        self.population_beta_param_table = pd.DataFrame(
            {
                "parameter": [spec.name for spec in self.param_specs],
                "alpha": self.population_beta_alpha,
                "beta": self.population_beta_beta,
            }
        )
        self.population_mean_params, self.population_sd_params = self._population_beta_summary(
            alpha, beta
        )
        if fixed_unit is not None and np.any(fixed_mask):
            fixed_params = self.unit_to_params(np.where(np.isfinite(fixed_unit), fixed_unit, 0.5))
            self.population_mean_params[fixed_mask] = fixed_params[fixed_mask]
            self.population_sd_params[fixed_mask] = 0.0
        self.population_mode_params = self.population_mean_params.copy()
        self.subject_fits = []
        self.posterior_participantParams = None
        self.posterior_prediction_df = None
        self.participantParams = pd.DataFrame.from_records(participant_rows)
        self.prediction_df = pd.DataFrame()
        self.history = pd.DataFrame.from_records(history)
        self.population_objective_name = population_objective
        self.fitObjective = float(np.sum(marginal_objective_per_subject))
        self.dataNegLogLik = float(np.sum(marginal_nll_per_subject))
        self.negLogLik = float(self.dataNegLogLik)
        self.rmse = np.nan
        self.n_population_params = int(2 * len(self.param_specs))
        self.n_params_total = int(self.n_population_params)
        self.sharedParams = pd.DataFrame(
            {
                "parameter": [spec.name for spec in self.param_specs],
                "value": self.population_mean_params,
                "sd": self.population_sd_params,
            }
        )
        return self

    def sample_population_params(self, n_samples=4096, seed=42):
        if getattr(self, "population_distribution_family", None) != "beta_factorized_real":
            raise RuntimeError("Call fit() before sampling population parameters.")
        _, params = self._sample_beta_population_params(
            self.population_beta_alpha,
            self.population_beta_beta,
            n_samples=int(n_samples),
            seed=seed,
            fixed_unit=getattr(self, "population_fixed_unit", None),
        )
        return pd.DataFrame(params, columns=[spec.name for spec in self.param_specs])

    @staticmethod
    def _beta_logpdf_units(unit_matrix, alpha, beta):
        units = np.clip(np.asarray(unit_matrix, dtype=float), 1e-8, 1.0 - 1e-8)
        alpha = np.asarray(alpha, dtype=float)
        beta = np.asarray(beta, dtype=float)
        return np.sum(
            (alpha - 1.0) * np.log(units) + (beta - 1.0) * np.log1p(-units) - betaln(alpha, beta),
            axis=1,
        )

    def _fit_participant_posterior_mode(
        self,
        participant,
        sample_units,
        sample_params,
        sample_loglik,
        alpha,
        beta,
        top_k=3,
        maxiter=80,
    ):
        sample_units = np.asarray(sample_units, dtype=float)
        sample_params = np.asarray(sample_params, dtype=float)
        sample_loglik = np.asarray(sample_loglik, dtype=float)
        sample_logprior = self._beta_logpdf_units(sample_units, alpha, beta)
        sample_logposterior = sample_loglik + sample_logprior
        order = np.argsort(sample_logposterior)[::-1]
        n_candidates = max(1, min(int(top_k), len(order)))
        starts = [np.clip(sample_units[idx], 1e-8, 1.0 - 1e-8) for idx in order[:n_candidates]]
        if len(sample_units):
            weights = np.exp(sample_logposterior - logsumexp(sample_logposterior))
            starts.append(np.clip(weights @ sample_units, 1e-8, 1.0 - 1e-8))

        bounds = [(1e-8, 1.0 - 1e-8)] * len(self.param_specs)
        best = None

        def objective(unit):
            unit = np.clip(np.asarray(unit, dtype=float), 1e-8, 1.0 - 1e-8)
            params = self.unit_to_params(unit)
            eval_result = self.evaluate_participant(params, participant)
            neg_log_prior = -float(self._beta_logpdf_units(unit[None, :], alpha, beta)[0])
            return float(eval_result["neg_log_lik"] + neg_log_prior)

        for start in starts:
            res = minimize(
                objective,
                np.asarray(start, dtype=float),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": int(maxiter), "ftol": 1e-9},
            )
            unit_hat = np.clip(np.asarray(res.x, dtype=float), 1e-8, 1.0 - 1e-8)
            params_hat = self.unit_to_params(unit_hat)
            eval_result = self.evaluate_participant(params_hat, participant)
            neg_log_prior = -float(self._beta_logpdf_units(unit_hat[None, :], alpha, beta)[0])
            candidate = {
                "params": np.asarray(params_hat, dtype=float),
                "objective": float(eval_result["neg_log_lik"] + neg_log_prior),
                "neg_log_lik": float(eval_result["neg_log_lik"]),
                "rmse": float(eval_result["rmse"]),
                "predicted_late_mean": np.asarray(eval_result["predicted_late_mean"], dtype=float),
            }
            if best is None or candidate["objective"] < best["objective"]:
                best = candidate

        if best is None:
            raise RuntimeError("Could not refine participant posterior mode.")
        return best

    def sample_population_joint_params(self, n_samples=4096, seed=42):
        return self.sample_population_params(n_samples=int(n_samples), seed=seed)

    def sample_population_predictive_summary(self, n_synthetic_participants=4096, seed=42):
        if getattr(self, "population_distribution_family", None) == "shared_point":
            if self.posterior_prediction_df is None or self.posterior_prediction_df.empty:
                return pd.DataFrame(
                    columns=[
                        "ppid",
                        "clamp_magnitude",
                        "late_mean",
                        "model_late_mean",
                        "obs_var",
                        "n_obs",
                    ]
                )
            out = self.posterior_prediction_df.copy()
            out["late_mean"] = pd.to_numeric(out["predicted_late_mean"], errors="coerce")
            out["model_late_mean"] = out["late_mean"]
            return out
        if self.summary_df.empty:
            return pd.DataFrame(
                columns=[
                    "ppid",
                    "clamp_magnitude",
                    "late_mean",
                    "model_late_mean",
                    "obs_var",
                    "n_obs",
                ]
            )
        clamp_grid = np.sort(
            pd.to_numeric(self.summary_df["clamp_magnitude"], errors="coerce")
            .dropna()
            .unique()
            .astype(float)
        )
        design_df = build_late_predictive_design(self.summary_df, clamp_grid)
        n_design = len(design_df) // len(clamp_grid)
        params_df = self.sample_population_joint_params(
            n_samples=int(n_synthetic_participants), seed=seed
        )
        params_matrix = params_df.to_numpy(dtype=float)
        rows = []
        sem_floor = float(SUMMARY_SEM_FLOOR_DEG)
        for idx, params in enumerate(params_matrix):
            predicted = np.asarray(
                self.predict_rows(np.asarray(params, dtype=float), design_df), dtype=float
            )
            clamp_means = predicted.reshape(len(clamp_grid), n_design).mean(axis=1)
            sdf = pd.DataFrame(
                {
                    "ppid": [f"synthetic_{idx + 1:05d}"] * len(clamp_grid),
                    "group": np.nan,
                    "clamp_magnitude": clamp_grid,
                    "target_angles_degrees": np.nan,
                    "target_clamp_sign": np.nan,
                    "clamp_sign": np.nan,
                    "rotation_model": np.nan,
                    "late_mean": clamp_means,
                    "late_sd": np.zeros(len(clamp_grid), dtype=float),
                    "late_sem": np.full(len(clamp_grid), sem_floor, dtype=float),
                    "obs_var": np.full(len(clamp_grid), sem_floor**2, dtype=float),
                    "n_obs": np.ones(len(clamp_grid), dtype=int),
                    "model_late_mean": clamp_means,
                }
            )
            rows.append(sdf)
        return pd.concat(rows, axis=0, ignore_index=True)

    def population_predictive_curve(
        self, clamp_grid=None, n_param_draws=4096, seed=42, ci_level=0.95
    ):
        if getattr(self, "population_distribution_family", None) == "shared_point":
            return self._curve_from_params(
                self.population_mean_params,
                clamp_grid=clamp_grid,
                curve_source="shared_point_params",
            )
        observed = (
            pd.to_numeric(self.summary_df["clamp_magnitude"], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        if len(observed) == 0:
            raise ValueError("No clamp magnitudes were available to build the predictive curve.")
        if clamp_grid is None:
            lo = float(np.min(observed))
            hi = float(np.max(observed))
            n_grid = max(300, int(np.ceil(2.0 * max(1.0, hi - lo))) + 1)
            clamp_grid = np.linspace(lo, hi, n_grid)
        else:
            clamp_grid = np.asarray(clamp_grid, dtype=float)
            if clamp_grid.ndim != 1 or len(clamp_grid) == 0:
                raise ValueError("clamp_grid must be a non-empty 1D array.")
        design_df = build_late_predictive_design(self.summary_df, clamp_grid)
        n_design = len(design_df) // len(clamp_grid)
        params_df = self.sample_population_joint_params(n_samples=int(n_param_draws), seed=seed)
        params_matrix = params_df.to_numpy(dtype=float)
        curve_draws = np.empty((len(params_matrix), len(clamp_grid)), dtype=float)

        for idx, params in enumerate(params_matrix):
            predicted = self.predict_rows(np.asarray(params, dtype=float), design_df)
            curve_draws[idx] = (
                np.asarray(predicted, dtype=float).reshape(len(clamp_grid), n_design).mean(axis=1)
            )

        mean_curve = np.mean(curve_draws, axis=0)
        if len(curve_draws) > 1:
            sem_curve = np.std(curve_draws, axis=0, ddof=1) / np.sqrt(float(len(curve_draws)))
            t_crit = float(stats.t.ppf((1.0 + float(ci_level)) / 2.0, df=len(curve_draws) - 1))
            half_width = t_crit * sem_curve
        else:
            sem_curve = np.zeros(len(clamp_grid), dtype=float)
            half_width = np.zeros(len(clamp_grid), dtype=float)
        return pd.DataFrame(
            {
                "clamp_magnitude": clamp_grid,
                "mean": mean_curve,
                "sem": sem_curve,
                "ci_low": mean_curve - half_width,
                "ci_high": mean_curve + half_width,
                "n_draws": int(len(params_matrix)),
            }
        )

    def population_mean_param_curve(self, clamp_grid=None):
        return self._curve_from_params(
            self.population_mean_params,
            clamp_grid=clamp_grid,
            curve_source="population_mean_params",
        )

    def prediction_dataframe(self):
        return self.prediction_df.copy()

    def parameter_summary_table(self):
        rows = []
        for _, row in self.sharedParams.iterrows():
            rows.append(
                {
                    "scope": "population",
                    "parameter": row["parameter"],
                    "mean": float(row["value"]),
                    "sd": float(row["sd"]),
                }
            )
        return pd.DataFrame.from_records(rows)

    def fit_summary(self):
        summary = {
            "model": self.model_name,
            "objectiveName": getattr(self, "population_objective_name", "negll"),
            "fitObjective": float(getattr(self, "fitObjective", np.nan)),
            "negLogLik": float(self.negLogLik),
            "rmse": float(self.rmse),
            "n_summary_rows": int(len(self.summary_df)),
            "n_params_total": int(self.n_params_total),
            "n_population_params": int(self.n_population_params),
        }
        if hasattr(self, "population_fit_mode"):
            summary["fitMode"] = getattr(self, "population_fit_mode")
        if hasattr(self, "population_fixed_param_names"):
            summary["fixedParamNames"] = list(getattr(self, "population_fixed_param_names"))
        if hasattr(self, "population_variable_param_names"):
            summary["variableParamNames"] = list(getattr(self, "population_variable_param_names"))
        return summary
