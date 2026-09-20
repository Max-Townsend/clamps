from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

from clamp_analysis import paths
from clamp_analysis.statistics.clamp_tests import (
    CI_LEVEL,
    POINTER_ORDER,
    TABLE_DIR,
    _add_anova_effect_sizes,
    _anova_effect_ci_summary,
    _attach_anova_effect_cis,
    _bootstrap_ci_1d,
    cluster_robust_term_table,
    cohens_d_one_sample,
    format_effect_ci,
    grouped_descriptives,
    load_bias_direction_pairings,
    load_pointer_lookup,
    load_table,
    p_text,
    planned_zero_tests_by_clamp,
    series_summary,
    stable_seed,
)

OUT_PREFIX = "paper_draft_results"

SUMMARY_MARKDOWN_PATH = TABLE_DIR / f"{OUT_PREFIX}_summary.md"

MANUSCRIPT_MARKDOWN_PATH = TABLE_DIR / f"{OUT_PREFIX}_manuscript_ready.md"

DATA_PATH = paths.DATA_PATH

BASELINE_BIAS_CONDITION = "NoFeedbackBaseline"

BASELINE_BIAS_ABS_LIMIT = 40.0


def save_table(df: pd.DataFrame, filename: str, outputs: dict[str, Path]) -> Path:
    path = TABLE_DIR / filename
    df.to_csv(path, index=False)
    outputs[filename] = path
    return path


def _safe_float(value) -> float:
    array = np.asarray(value)
    if array.size == 0:
        return np.nan
    return float(array.squeeze())


def sample_sd(values: pd.Series | np.ndarray) -> float:
    array = pd.Series(values, dtype=float).dropna().to_numpy(dtype=float)
    if len(array) <= 1:
        return 0.0
    return float(np.std(array, ddof=1))


def _bootstrap_ci_independent(
    a: pd.Series | np.ndarray,
    b: pd.Series | np.ndarray,
    stat_fn,
    *,
    n_boot: int = 2000,
    ci: float = CI_LEVEL,
    seed: int,
) -> tuple[float, float]:
    a_array = pd.Series(a, dtype=float).dropna().to_numpy(dtype=float)
    b_array = pd.Series(b, dtype=float).dropna().to_numpy(dtype=float)
    if len(a_array) == 0 or len(b_array) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        a_boot = rng.choice(a_array, size=len(a_array), replace=True)
        b_boot = rng.choice(b_array, size=len(b_array), replace=True)
        draws[idx] = float(stat_fn(a_boot, b_boot))
    alpha = (1.0 - ci) / 2.0
    return (
        float(np.quantile(draws, alpha)),
        float(np.quantile(draws, 1.0 - alpha)),
    )


def hedges_g_independent(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
    a_array = pd.Series(a, dtype=float).dropna().to_numpy(dtype=float)
    b_array = pd.Series(b, dtype=float).dropna().to_numpy(dtype=float)
    n_a = len(a_array)
    n_b = len(b_array)
    if n_a < 2 or n_b < 2:
        return np.nan
    var_a = np.var(a_array, ddof=1)
    var_b = np.var(b_array, ddof=1)
    pooled_df = n_a + n_b - 2
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / pooled_df
    pooled_sd = float(np.sqrt(pooled_var))
    if pooled_sd == 0.0:
        return 0.0
    cohen_d = float((np.mean(a_array) - np.mean(b_array)) / pooled_sd)
    correction = 1.0 - (3.0 / (4.0 * pooled_df - 1.0))
    return correction * cohen_d


def rank_biserial_independent(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
    a_array = pd.Series(a, dtype=float).dropna().to_numpy(dtype=float)
    b_array = pd.Series(b, dtype=float).dropna().to_numpy(dtype=float)
    n_a = len(a_array)
    n_b = len(b_array)
    if n_a == 0 or n_b == 0:
        return np.nan
    u_stat = float(stats.mannwhitneyu(a_array, b_array, alternative="two-sided").statistic)
    return (2.0 * u_stat / (n_a * n_b)) - 1.0


def _cluster_pointer_row(
    df: pd.DataFrame,
    *,
    formula: str,
    measure: str,
    cluster_col: str = "ppid",
) -> dict[str, float | int | str]:
    model = smf.ols(formula, data=df).fit()
    used_rows = pd.Index(model.model.data.row_labels)
    clean_df = df.loc[used_rows].copy()
    clean_df = clean_df.loc[clean_df[cluster_col].notna()].copy()
    robust_model = smf.ols(formula, data=clean_df).fit(
        cov_type="cluster",
        cov_kwds={"groups": clean_df[cluster_col].to_numpy()},
    )
    term = next(
        name for name in robust_model.params.index if "pointer" in name and "[T.Mouse]" in name
    )
    ci = robust_model.conf_int().loc[term]
    return {
        "measure": measure,
        "cluster_formula": formula,
        "cluster_estimate_mouse_minus_trackpad": float(robust_model.params[term]),
        "cluster_se": float(robust_model.bse[term]),
        "cluster_ci_low": float(ci.iloc[0]),
        "cluster_ci_high": float(ci.iloc[1]),
        "cluster_p": float(robust_model.pvalues[term]),
        "cluster_n_rows": int(len(clean_df)),
        "cluster_n_participants": int(clean_df[cluster_col].nunique()),
    }


def _cluster_pointer_log_row(
    df: pd.DataFrame,
    *,
    formula: str,
    measure: str,
    cluster_col: str = "ppid",
) -> dict[str, float | int | str]:
    if (df["value"] <= 0).any():
        return {
            "measure": measure,
            "log_cluster_formula": "",
            "log_cluster_estimate_log_mouse_minus_trackpad": np.nan,
            "log_cluster_ci_low": np.nan,
            "log_cluster_ci_high": np.nan,
            "log_cluster_p": np.nan,
            "log_cluster_ratio_mouse_over_trackpad": np.nan,
            "log_cluster_ratio_ci_low": np.nan,
            "log_cluster_ratio_ci_high": np.nan,
        }

    rhs = formula.split("~", 1)[1].strip()
    log_formula = f"log_value ~ {rhs}"
    log_df = df.assign(log_value=np.log(df["value"]))
    model = smf.ols(log_formula, data=log_df).fit()
    used_rows = pd.Index(model.model.data.row_labels)
    clean_df = log_df.loc[used_rows].copy()
    clean_df = clean_df.loc[clean_df[cluster_col].notna()].copy()
    robust_model = smf.ols(log_formula, data=clean_df).fit(
        cov_type="cluster",
        cov_kwds={"groups": clean_df[cluster_col].to_numpy()},
    )
    term = next(
        name for name in robust_model.params.index if "pointer" in name and "[T.Mouse]" in name
    )
    ci = robust_model.conf_int().loc[term]
    coef = float(robust_model.params[term])
    ci_low = float(ci.iloc[0])
    ci_high = float(ci.iloc[1])
    return {
        "measure": measure,
        "log_cluster_formula": log_formula,
        "log_cluster_estimate_log_mouse_minus_trackpad": coef,
        "log_cluster_ci_low": ci_low,
        "log_cluster_ci_high": ci_high,
        "log_cluster_p": float(robust_model.pvalues[term]),
        "log_cluster_ratio_mouse_over_trackpad": float(np.exp(coef)),
        "log_cluster_ratio_ci_low": float(np.exp(ci_low)),
        "log_cluster_ratio_ci_high": float(np.exp(ci_high)),
    }


def pointer_contrast_row(
    df: pd.DataFrame,
    *,
    measure: str,
    value_col: str = "value",
    cluster_formula: str | None = None,
) -> dict[str, float | int | str]:
    participant_df = (
        df.groupby(["ppid", "pointer"], observed=True, as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "value"})
    )
    mouse = participant_df.loc[participant_df["pointer"].eq("Mouse"), "value"]
    trackpad = participant_df.loc[participant_df["pointer"].eq("Trackpad"), "value"]

    mouse_summary = series_summary(mouse)
    trackpad_summary = series_summary(trackpad)
    mouse_median = float(np.median(mouse)) if len(mouse) else np.nan
    trackpad_median = float(np.median(trackpad)) if len(trackpad) else np.nan
    mouse_q25 = float(np.quantile(mouse, 0.25)) if len(mouse) else np.nan
    mouse_q75 = float(np.quantile(mouse, 0.75)) if len(mouse) else np.nan
    trackpad_q25 = float(np.quantile(trackpad, 0.25)) if len(trackpad) else np.nan
    trackpad_q75 = float(np.quantile(trackpad, 0.75)) if len(trackpad) else np.nan
    mouse_median_low, mouse_median_high = _bootstrap_ci_1d(
        mouse,
        np.median,
        seed=stable_seed(OUT_PREFIX, measure, "mouse_median"),
    )
    trackpad_median_low, trackpad_median_high = _bootstrap_ci_1d(
        trackpad,
        np.median,
        seed=stable_seed(OUT_PREFIX, measure, "trackpad_median"),
    )

    welch = stats.ttest_ind(mouse, trackpad, equal_var=False)
    hedges_g = hedges_g_independent(mouse, trackpad)
    hedges_low, hedges_high = _bootstrap_ci_independent(
        mouse,
        trackpad,
        hedges_g_independent,
        seed=stable_seed(OUT_PREFIX, measure, "hedges_g"),
    )
    mw = stats.mannwhitneyu(mouse, trackpad, alternative="two-sided")
    rank_biserial = rank_biserial_independent(mouse, trackpad)
    rbc_low, rbc_high = _bootstrap_ci_independent(
        mouse,
        trackpad,
        rank_biserial_independent,
        seed=stable_seed(OUT_PREFIX, measure, "rank_biserial"),
    )

    row: dict[str, float | int | str] = {
        "measure": measure,
        "n_mouse": int(mouse_summary["n"]),
        "mouse_mean": mouse_summary["mean"],
        "mouse_sd": mouse_summary["sd"],
        "mouse_ci_low": mouse_summary["ci_low"],
        "mouse_ci_high": mouse_summary["ci_high"],
        "mouse_median": mouse_median,
        "mouse_q25": mouse_q25,
        "mouse_q75": mouse_q75,
        "mouse_median_ci_low": mouse_median_low,
        "mouse_median_ci_high": mouse_median_high,
        "n_trackpad": int(trackpad_summary["n"]),
        "trackpad_mean": trackpad_summary["mean"],
        "trackpad_sd": trackpad_summary["sd"],
        "trackpad_ci_low": trackpad_summary["ci_low"],
        "trackpad_ci_high": trackpad_summary["ci_high"],
        "trackpad_median": trackpad_median,
        "trackpad_q25": trackpad_q25,
        "trackpad_q75": trackpad_q75,
        "trackpad_median_ci_low": trackpad_median_low,
        "trackpad_median_ci_high": trackpad_median_high,
        "welch_t": _safe_float(welch.statistic),
        "welch_df": _safe_float(getattr(welch, "df", np.nan)),
        "welch_p": _safe_float(welch.pvalue),
        "hedges_g_mouse_minus_trackpad": hedges_g,
        "hedges_g_ci_low": hedges_low,
        "hedges_g_ci_high": hedges_high,
        "mann_whitney_u": float(mw.statistic),
        "mann_whitney_p": float(mw.pvalue),
        "rank_biserial_mouse_minus_trackpad": rank_biserial,
        "rank_biserial_ci_low": rbc_low,
        "rank_biserial_ci_high": rbc_high,
        "cluster_formula": "",
        "cluster_estimate_mouse_minus_trackpad": np.nan,
        "cluster_se": np.nan,
        "cluster_ci_low": np.nan,
        "cluster_ci_high": np.nan,
        "cluster_p": np.nan,
        "cluster_n_rows": np.nan,
        "cluster_n_participants": np.nan,
        "log_cluster_formula": "",
        "log_cluster_estimate_log_mouse_minus_trackpad": np.nan,
        "log_cluster_ci_low": np.nan,
        "log_cluster_ci_high": np.nan,
        "log_cluster_p": np.nan,
        "log_cluster_ratio_mouse_over_trackpad": np.nan,
        "log_cluster_ratio_ci_low": np.nan,
        "log_cluster_ratio_ci_high": np.nan,
    }
    if cluster_formula is not None:
        clean_df = df.rename(columns={value_col: "value"}) if value_col != "value" else df.copy()
        row.update(
            _cluster_pointer_row(
                clean_df,
                formula=cluster_formula,
                measure=measure,
            )
        )
        row.update(
            _cluster_pointer_log_row(
                clean_df,
                formula=cluster_formula,
                measure=measure,
            )
        )
    return row


def zero_tests_by_pointer(
    df: pd.DataFrame,
    *,
    measure: str,
    value_col: str = "value",
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    work = df.copy()
    if value_col != "value":
        work = work.rename(columns={value_col: "value"})
    for pointer, group_df in work.groupby("pointer", observed=True):
        tests = planned_zero_tests_by_clamp(group_df, value_col="value")
        tests["measure"] = measure
        tests["pointer"] = pointer
        rows.append(tests)
    return pd.concat(rows, ignore_index=True)


def _load_baseline_bias_trials() -> pd.DataFrame:
    trials = pd.read_csv(
        DATA_PATH,
        usecols=[
            "ppid",
            "condition",
            "target_angles_degrees",
            "clamp_magnitude",
            "clamp_angle_relative",
            "target_clamp_sign",
            "recentred_hand_angle",
            "unflipped_hand_angle",
        ],
        dtype={"ppid": str},
    )
    trials = trials.loc[trials["condition"].eq(BASELINE_BIAS_CONDITION)].copy()
    for col in [
        "target_angles_degrees",
        "clamp_magnitude",
        "clamp_angle_relative",
        "target_clamp_sign",
        "recentred_hand_angle",
        "unflipped_hand_angle",
    ]:
        trials[col] = pd.to_numeric(trials[col], errors="coerce")
    trials = trials.merge(load_pointer_lookup(), on="ppid", how="left")
    trials = trials.loc[trials["pointer"].isin(POINTER_ORDER)].copy()
    return trials.reset_index(drop=True)


def load_or_build_raw_signed_baseline_bias() -> pd.DataFrame:
    filename = "baseline_bias_raw_signed_by_ppid_clamp.csv"
    path = TABLE_DIR / filename
    if path.exists():
        return load_table(filename)

    trials = _load_baseline_bias_trials()
    raw_bias = (
        trials.dropna(subset=["clamp_magnitude"])
        .groupby(["ppid", "clamp_magnitude"], as_index=False, dropna=False)
        .agg(
            raw_signed_baseline_bias=("unflipped_hand_angle", "mean"),
            signed_clamp_angle=("clamp_angle_relative", "mean"),
            target_clamp_sign=("target_clamp_sign", "first"),
        )
    )
    raw_bias["clamp_magnitude"] = raw_bias["clamp_magnitude"].astype(int)
    raw_bias["target_clamp_sign"] = raw_bias["target_clamp_sign"].astype(int)
    raw_bias = raw_bias.merge(load_pointer_lookup(), on="ppid", how="left")
    raw_bias = raw_bias.loc[
        raw_bias["pointer"].isin(POINTER_ORDER)
        & raw_bias["raw_signed_baseline_bias"].abs().le(BASELINE_BIAS_ABS_LIMIT)
    ].copy()
    raw_bias.to_csv(path, index=False)
    return load_table(filename)


def _spatial_bias_table_from_trials(trials: pd.DataFrame, value_col: str) -> pd.DataFrame:
    spatial = (
        trials.dropna(subset=["target_angles_degrees"])
        .groupby(["ppid", "target_angles_degrees"], as_index=False, dropna=False)
        .agg(bias=(value_col, "mean"))
    )
    spatial = spatial.merge(load_pointer_lookup(), on="ppid", how="left")
    spatial = spatial.loc[
        spatial["pointer"].isin(POINTER_ORDER) & spatial["bias"].abs().le(BASELINE_BIAS_ABS_LIMIT)
    ].copy()
    spatial["target_angles_degrees"] = spatial["target_angles_degrees"].astype(int)
    return spatial.reset_index(drop=True)


def load_or_build_bias_frame_spatial_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    task_filename = "baseline_bias_task_aligned_by_target.csv"
    raw_filename = "baseline_bias_raw_signed_by_target.csv"
    task_path = TABLE_DIR / task_filename
    raw_path = TABLE_DIR / raw_filename
    if task_path.exists() and raw_path.exists():
        return load_table(task_filename), load_table(raw_filename)

    trials = _load_baseline_bias_trials()
    task_aligned_spatial = _spatial_bias_table_from_trials(trials, "recentred_hand_angle")
    raw_signed_spatial = _spatial_bias_table_from_trials(trials, "unflipped_hand_angle")
    task_aligned_spatial.to_csv(task_path, index=False)
    raw_signed_spatial.to_csv(raw_path, index=False)
    return load_table(task_filename), load_table(raw_filename)


def bias_frame_spatial_omnibus_table(
    task_aligned_spatial: pd.DataFrame,
    raw_signed_spatial: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for label, df in [
        ("Task-aligned", task_aligned_spatial),
        ("Raw signed", raw_signed_spatial),
    ]:
        table = cluster_robust_term_table(
            df,
            "bias ~ C(pointer) * C(target_angles_degrees)",
        )
        table.insert(0, "bias_frame", label)
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def paired_tests_by_clamp(
    baseline_df: pd.DataFrame,
    late_df: pd.DataFrame,
    *,
    baseline_value_col: str,
    late_value_col: str,
    measure: str,
) -> pd.DataFrame:
    merged = (
        baseline_df[["ppid", "clamp_magnitude", baseline_value_col]]
        .rename(columns={baseline_value_col: "baseline_value"})
        .merge(
            late_df[["ppid", "clamp_magnitude", late_value_col]].rename(
                columns={late_value_col: "late_value"}
            ),
            on=["ppid", "clamp_magnitude"],
            how="inner",
        )
    )

    records: list[dict[str, float | int | str]] = []
    for clamp, group_df in merged.groupby("clamp_magnitude", observed=True):
        baseline_summary = series_summary(group_df["baseline_value"])
        late_summary = series_summary(group_df["late_value"])
        diff = group_df["late_value"] - group_df["baseline_value"]
        diff_summary = series_summary(diff)
        t_stat, p_value = stats.ttest_1samp(diff, 0.0)
        dz = cohens_d_one_sample(diff)
        dz_low, dz_high = _bootstrap_ci_1d(
            diff,
            cohens_d_one_sample,
            seed=stable_seed(OUT_PREFIX, measure, clamp, "cohens_dz"),
        )
        records.append(
            {
                "measure": measure,
                "clamp_magnitude": int(clamp),
                "n_participants": int(len(group_df)),
                "baseline_mean": baseline_summary["mean"],
                "baseline_sd": baseline_summary["sd"],
                "baseline_ci_low": baseline_summary["ci_low"],
                "baseline_ci_high": baseline_summary["ci_high"],
                "late_mean": late_summary["mean"],
                "late_sd": late_summary["sd"],
                "late_ci_low": late_summary["ci_low"],
                "late_ci_high": late_summary["ci_high"],
                "mean_diff_late_minus_baseline": diff_summary["mean"],
                "sd_diff": diff_summary["sd"],
                "diff_ci_low": diff_summary["ci_low"],
                "diff_ci_high": diff_summary["ci_high"],
                "t": float(t_stat),
                "p": float(p_value),
                "cohens_dz": dz,
                "cohens_dz_ci_low": dz_low,
                "cohens_dz_ci_high": dz_high,
            }
        )

    out = pd.DataFrame.from_records(records).sort_values("clamp_magnitude").reset_index(drop=True)
    reject, p_holm, _, _ = multipletests(out["p"], method="holm")
    out["p_holm"] = p_holm
    out["reject_holm"] = reject
    return out


def phase_descriptives(
    baseline_df: pd.DataFrame,
    late_df: pd.DataFrame,
    *,
    baseline_value_col: str,
    late_value_col: str,
    measure: str,
) -> pd.DataFrame:
    combined = pd.concat(
        [
            baseline_df[["ppid", "clamp_magnitude", baseline_value_col]]
            .rename(columns={baseline_value_col: "value"})
            .assign(phase="Baseline"),
            late_df[["ppid", "clamp_magnitude", late_value_col]]
            .rename(columns={late_value_col: "value"})
            .assign(phase="Late adaptation"),
        ],
        ignore_index=True,
    )
    descriptives = grouped_descriptives(
        combined, group_cols=["phase", "clamp_magnitude"], value_col="value"
    )
    descriptives["measure"] = measure
    return descriptives


def phase_spread_dataset(
    baseline_df: pd.DataFrame,
    late_df: pd.DataFrame,
    *,
    baseline_value_col: str,
    late_value_col: str,
) -> pd.DataFrame:
    combined = pd.concat(
        [
            baseline_df[["ppid", "clamp_magnitude", "pointer", baseline_value_col]]
            .rename(columns={baseline_value_col: "value"})
            .assign(phase="Baseline"),
            late_df[["ppid", "clamp_magnitude", "pointer", late_value_col]]
            .rename(columns={late_value_col: "value"})
            .assign(phase="Late adaptation"),
        ],
        ignore_index=True,
    )
    centers = (
        combined.groupby(["phase", "clamp_magnitude"], observed=True)["value"]
        .median()
        .rename("phase_center_median")
        .reset_index()
    )
    combined = combined.merge(centers, on=["phase", "clamp_magnitude"], how="left")
    combined["abs_dev"] = (combined["value"] - combined["phase_center_median"]).abs()
    return combined


def phase_spread_descriptives(
    baseline_df: pd.DataFrame,
    late_df: pd.DataFrame,
    *,
    baseline_value_col: str,
    late_value_col: str,
    measure: str,
) -> pd.DataFrame:
    combined = phase_spread_dataset(
        baseline_df,
        late_df,
        baseline_value_col=baseline_value_col,
        late_value_col=late_value_col,
    )
    records: list[dict[str, float | int | str]] = []
    for (phase, clamp), group_df in combined.groupby(["phase", "clamp_magnitude"], observed=True):
        summary = series_summary(group_df["value"])
        sd_low, sd_high = _bootstrap_ci_1d(
            group_df["value"],
            sample_sd,
            seed=stable_seed(OUT_PREFIX, measure, phase, clamp, "sd"),
        )
        abs_dev_summary = series_summary(group_df["abs_dev"])
        records.append(
            {
                "measure": measure,
                "phase": phase,
                "clamp_magnitude": int(clamp),
                "n_participants": int(len(group_df)),
                "phase_center_median": float(group_df["phase_center_median"].iloc[0]),
                "mean": summary["mean"],
                "mean_ci_low": summary["ci_low"],
                "mean_ci_high": summary["ci_high"],
                "sd": summary["sd"],
                "sd_ci_low": sd_low,
                "sd_ci_high": sd_high,
                "median": summary["median"],
                "q25": summary["q25"],
                "q75": summary["q75"],
                "mean_abs_dev": abs_dev_summary["mean"],
                "mean_abs_dev_ci_low": abs_dev_summary["ci_low"],
                "mean_abs_dev_ci_high": abs_dev_summary["ci_high"],
                "median_abs_dev": abs_dev_summary["median"],
            }
        )
    return (
        pd.DataFrame.from_records(records)
        .sort_values(["clamp_magnitude", "phase"])
        .reset_index(drop=True)
    )


def paired_spread_tests_by_clamp(
    baseline_df: pd.DataFrame,
    late_df: pd.DataFrame,
    *,
    baseline_value_col: str,
    late_value_col: str,
    measure: str,
) -> pd.DataFrame:
    merged = (
        baseline_df[["ppid", "clamp_magnitude", baseline_value_col]]
        .rename(columns={baseline_value_col: "baseline_value"})
        .merge(
            late_df[["ppid", "clamp_magnitude", late_value_col]].rename(
                columns={late_value_col: "late_value"}
            ),
            on=["ppid", "clamp_magnitude"],
            how="inner",
        )
    )

    records: list[dict[str, float | int | str]] = []
    for clamp, group_df in merged.groupby("clamp_magnitude", observed=True):
        baseline_center = float(group_df["baseline_value"].median())
        late_center = float(group_df["late_value"].median())
        baseline_abs_dev = (group_df["baseline_value"] - baseline_center).abs()
        late_abs_dev = (group_df["late_value"] - late_center).abs()
        baseline_summary = series_summary(baseline_abs_dev)
        late_summary = series_summary(late_abs_dev)
        diff = late_abs_dev - baseline_abs_dev
        diff_summary = series_summary(diff)
        t_stat, p_value = stats.ttest_1samp(diff, 0.0)
        dz = cohens_d_one_sample(diff)
        dz_low, dz_high = _bootstrap_ci_1d(
            diff,
            cohens_d_one_sample,
            seed=stable_seed(OUT_PREFIX, measure, clamp, "cohens_dz"),
        )
        records.append(
            {
                "measure": measure,
                "clamp_magnitude": int(clamp),
                "n_participants": int(len(group_df)),
                "baseline_center_median": baseline_center,
                "late_center_median": late_center,
                "baseline_mean_abs_dev": baseline_summary["mean"],
                "baseline_mean_abs_dev_ci_low": baseline_summary["ci_low"],
                "baseline_mean_abs_dev_ci_high": baseline_summary["ci_high"],
                "baseline_median_abs_dev": baseline_summary["median"],
                "late_mean_abs_dev": late_summary["mean"],
                "late_mean_abs_dev_ci_low": late_summary["ci_low"],
                "late_mean_abs_dev_ci_high": late_summary["ci_high"],
                "late_median_abs_dev": late_summary["median"],
                "mean_diff_late_minus_baseline_abs_dev": diff_summary["mean"],
                "diff_ci_low": diff_summary["ci_low"],
                "diff_ci_high": diff_summary["ci_high"],
                "t": float(t_stat),
                "p": float(p_value),
                "cohens_dz": dz,
                "cohens_dz_ci_low": dz_low,
                "cohens_dz_ci_high": dz_high,
            }
        )

    out = pd.DataFrame.from_records(records).sort_values("clamp_magnitude").reset_index(drop=True)
    reject, p_holm, _, _ = multipletests(out["p"], method="holm")
    out["p_holm"] = p_holm
    out["reject_holm"] = reject
    return out


def participant_group_means(df: pd.DataFrame, *, value_col: str = "value") -> pd.DataFrame:
    return (
        df.groupby(["ppid", "group", "pointer"], observed=True, as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "value"})
    )


def phase_group_dataset(
    baseline_df: pd.DataFrame,
    late_df: pd.DataFrame,
    *,
    baseline_value_col: str,
    late_value_col: str,
) -> pd.DataFrame:
    return pd.concat(
        [
            participant_group_means(baseline_df, value_col=baseline_value_col).assign(
                phase="Baseline"
            ),
            participant_group_means(late_df, value_col=late_value_col).assign(
                phase="Late adaptation"
            ),
        ],
        ignore_index=True,
    )


def phase_group_spread_dataset(
    baseline_df: pd.DataFrame,
    late_df: pd.DataFrame,
    *,
    baseline_value_col: str,
    late_value_col: str,
) -> pd.DataFrame:
    combined = phase_group_dataset(
        baseline_df,
        late_df,
        baseline_value_col=baseline_value_col,
        late_value_col=late_value_col,
    )
    centers = (
        combined.groupby(["phase", "group"], observed=True)["value"]
        .median()
        .rename("phase_group_center_median")
        .reset_index()
    )
    combined = combined.merge(centers, on=["phase", "group"], how="left")
    combined["abs_dev"] = (combined["value"] - combined["phase_group_center_median"]).abs()
    return combined


def between_subject_anova_table(df: pd.DataFrame, formula: str) -> pd.DataFrame:
    model = smf.ols(formula, data=df).fit()
    table = (
        anova_lm(model, typ=2)
        .reset_index()
        .rename(columns={"index": "term", "df": "df_num", "F": "f_value", "PR(>F)": "p_value"})
    )
    table = _add_anova_effect_sizes(table.rename(columns={"df_num": "df"})).rename(
        columns={"df": "df_num"}
    )
    table = _attach_anova_effect_cis(
        table,
        _anova_effect_ci_summary(
            df,
            formula,
            cluster_col=None,
            seed=stable_seed("between_subject_anova_table", formula),
        ),
    )
    residual_mask = table["term"].eq("Residual")
    df_den = float(table.loc[residual_mask, "df_num"].iloc[0]) if residual_mask.any() else np.nan
    table["df_den"] = df_den
    table["formula"] = formula
    table["n_rows"] = int(len(df))
    table["n_participants"] = int(df["ppid"].nunique()) if "ppid" in df.columns else int(len(df))
    return table


def mixed_anova_table(df: pd.DataFrame, formula: str, *, subject_col: str = "ppid") -> pd.DataFrame:
    model = smf.ols(formula, data=df).fit()
    table = (
        anova_lm(model, typ=2)
        .reset_index()
        .rename(columns={"index": "term", "df": "df_num", "F": "f_value", "PR(>F)": "p_value"})
    )
    table = _add_anova_effect_sizes(table.rename(columns={"df_num": "df"})).rename(
        columns={"df": "df_num"}
    )
    table = _attach_anova_effect_cis(
        table,
        _anova_effect_ci_summary(
            df,
            formula,
            cluster_col=subject_col,
            seed=stable_seed("mixed_anova_table", formula, subject_col),
        ),
    )
    residual_mask = table["term"].eq("Residual")
    df_den = float(table.loc[residual_mask, "df_num"].iloc[0]) if residual_mask.any() else np.nan
    table["df_den"] = df_den
    table["formula"] = formula
    table["n_rows"] = int(len(df))
    table["n_participants"] = int(df[subject_col].nunique())
    subject_term = f"C({subject_col})"
    return table.loc[~table["term"].eq(subject_term)].reset_index(drop=True)


def zero_tests_by_group(
    df: pd.DataFrame,
    *,
    group_col: str = "group",
    value_col: str = "value",
) -> pd.DataFrame:
    tests = planned_zero_tests_by_clamp(
        df.rename(columns={group_col: "clamp_magnitude", value_col: "value"})[
            ["clamp_magnitude", "value"]
        ],
        value_col="value",
    )
    return tests.rename(columns={"clamp_magnitude": group_col})


def sentence_map() -> pd.DataFrame:
    rows = [
        {
            "draft_location": "Paragraph 12",
            "draft_claim": "Primary late-adaptation omnibus by assigned clamp group",
            "primary_table": f"{OUT_PREFIX}_late_group_omnibus.csv",
            "secondary_table": f"{OUT_PREFIX}_late_group_zero_tests.csv",
            "note": "This is the cleaner between-subject group x pointer ANOVA on each participant's mean across the four assigned clamp magnitudes; groups 1-8 survive Holm, groups 9-10 do not.",
        },
        {
            "draft_location": "Paragraph 12",
            "draft_claim": "Secondary late-adaptation zero tests by individual clamp and input device",
            "primary_table": f"{OUT_PREFIX}_zero_tests_by_pointer.csv",
            "secondary_table": "paper_requested_seven_row_late_adaptation_pointer_descriptives.csv",
            "note": "These are secondary descriptive follow-ups on the 40 individual clamp magnitudes, not the primary omnibus.",
        },
        {
            "draft_location": "Paragraph 13",
            "draft_claim": "Primary early STL and washout omnibus tests by assigned clamp group",
            "primary_table": f"{OUT_PREFIX}_early_group_omnibus.csv / {OUT_PREFIX}_washout_group_omnibus.csv",
            "secondary_table": "paper_requested_seven_row_early_stl_zero_tests.csv / paper_requested_seven_row_washout_stl_zero_tests.csv",
            "note": "Use the cleaner group x pointer ANOVAs for the paragraph openers; retain the 40-clamp zero tests as secondary follow-ups.",
        },
        {
            "draft_location": "Paragraph 17",
            "draft_claim": "Baseline vs late adaptation mean shift",
            "primary_table": f"{OUT_PREFIX}_phase_mean_omnibus.csv",
            "secondary_table": f"{OUT_PREFIX}_phase_mean_by_clamp_paired_tests.csv",
            "note": "Use the mixed ANOVA on participant-level group averages for the paragraph opener, then the by-clamp paired table for the 40-clamp follow-up.",
        },
        {
            "draft_location": "Paragraph 17",
            "draft_claim": "Baseline vs late between-participant spread of mean hand angle",
            "primary_table": f"{OUT_PREFIX}_phase_spread_omnibus.csv",
            "secondary_table": f"{OUT_PREFIX}_phase_spread_by_clamp_paired_tests.csv",
            "note": "The opener now uses a mixed ANOVA on group-level spread via absolute deviation from the phase-specific group median; the by-clamp follow-up still uses clamp-specific spread tests.",
        },
        {
            "draft_location": "Paragraph 17",
            "draft_claim": "High-error bias-sign differences",
            "primary_table": "paper_requested_seven_row_bias_direction_omnibus.csv",
            "secondary_table": "paper_requested_seven_row_bias_direction_pairwise.csv",
            "note": "Use the cluster-robust bias-direction term test plus pairwise contrast CIs; overall descriptive means are in the new high-error summary table.",
        },
        {
            "draft_location": "Paragraph 19",
            "draft_claim": "Mouse vs Trackpad late adaptation",
            "primary_table": f"{OUT_PREFIX}_pointer_contrasts.csv",
            "secondary_table": "paper_requested_seven_row_late_adaptation_omnibus.csv",
            "note": "The clean simple effect for the 'medium-size clamps' sentence is the peak-range row in the pointer-contrast table.",
        },
        {
            "draft_location": "Paragraph 19",
            "draft_claim": "Mouse vs Trackpad early/late variability",
            "primary_table": f"{OUT_PREFIX}_pointer_contrasts.csv",
            "secondary_table": "paper_requested_seven_row_early_stl_variability_omnibus.csv / paper_requested_seven_row_late_variability_omnibus.csv",
            "note": "For reach variability, prefer the rank-based and log-scale device contrasts because the SD distributions are strongly right-skewed.",
        },
        {
            "draft_location": "Paragraph 19",
            "draft_claim": "No difference in mean task-aligned baseline bias",
            "primary_table": f"{OUT_PREFIX}_pointer_contrasts.csv",
            "secondary_table": "",
            "note": "Use the Task-aligned baseline bias row for the clamp-sign-aligned analysis; the Raw signed baseline bias row is the same comparison without task alignment.",
        },
        {
            "draft_location": "Paragraph 19",
            "draft_claim": "Raw signed baseline bias control",
            "primary_table": f"{OUT_PREFIX}_pointer_contrasts.csv",
            "secondary_table": f"{OUT_PREFIX}_bias_frame_spatial_omnibus.csv",
            "note": "Use the Raw signed rows when describing unflipped baseline biases; the printout reports these next to the task-aligned rows.",
        },
        {
            "draft_location": "Paragraph 19",
            "draft_claim": "Different spatial bias pattern by input device",
            "primary_table": f"{OUT_PREFIX}_bias_frame_spatial_omnibus.csv",
            "secondary_table": "paper_requested_seven_row_spatial_bias_descriptives.csv",
            "note": "Use the Task-aligned or Raw signed pointer x target interaction according to the bias frame being reported.",
        },
    ]
    return pd.DataFrame.from_records(rows)


def fmt_p_bound(value: float) -> str:
    if value < 0.001:
        return ".001"
    return f"{value:.3f}".lstrip("0")


def p_exact_text(value: float, *, label: str = "p") -> str:
    if value < 0.001:
        return f"{label} < .001"
    return f"{label} = {value:.3f}".replace("0.", ".")


def p_series_text(values: pd.Series | np.ndarray, *, label: str = "p") -> str:
    series = pd.Series(values, dtype=float).dropna()
    if series.empty:
        return ""
    low = float(series.min())
    high = float(series.max())
    if np.isclose(low, high):
        if len(series) > 1:
            if low < 0.001:
                return f"all {label}s < .001"
            return f"all {label}s = {fmt_p_bound(low)}"
        return p_exact_text(low, label=label)
    if high <= 0.05:
        return f"all {label}s <= {fmt_p_bound(high)}"
    if low >= 0.05:
        return f"all {label}s >= {fmt_p_bound(low)}"
    return f"{label}s = {fmt_p_bound(low)}-{fmt_p_bound(high)}"


def wald_text(row: pd.Series, *, n_label: str = "participants") -> str:
    n_value = (
        int(row["n_participants"])
        if pd.notna(row.get("n_participants", np.nan))
        else int(row["n_rows"])
    )
    return (
        f"Wald chi2({int(row['df_constraint'])}, N = {n_value} {n_label}) = {row['wald_chi2']:.2f}"
    )


def build_summary_markdown(
    *,
    late_group_omnibus: pd.DataFrame,
    late_group_zero: pd.DataFrame,
    early_group_omnibus: pd.DataFrame,
    washout_group_omnibus: pd.DataFrame,
    pointer_contrasts: pd.DataFrame,
    zero_tests_by_pointer_df: pd.DataFrame,
    phase_mean_omnibus: pd.DataFrame,
    phase_mean_paired: pd.DataFrame,
    phase_spread_omnibus: pd.DataFrame,
    phase_spread_paired: pd.DataFrame,
    high_error_bias_overall: pd.DataFrame,
    bias_frame_spatial_omnibus: pd.DataFrame,
) -> str:
    late_zero = pd.read_csv(TABLE_DIR / "paper_requested_seven_row_late_adaptation_zero_tests.csv")
    early_zero = pd.read_csv(TABLE_DIR / "paper_requested_seven_row_early_stl_zero_tests.csv")
    washout_zero = pd.read_csv(TABLE_DIR / "paper_requested_seven_row_washout_stl_zero_tests.csv")
    peak_phase = pd.read_csv(
        TABLE_DIR / "paper_requested_seven_row_peak_phase_paired_test.csv"
    ).iloc[0]
    high_error_phase = pd.read_csv(
        TABLE_DIR / "paper_requested_seven_row_high_error_phase_paired_test.csv"
    ).iloc[0]
    bias_omnibus = pd.read_csv(TABLE_DIR / "paper_requested_seven_row_bias_direction_omnibus.csv")
    bias_pairwise = pd.read_csv(TABLE_DIR / "paper_requested_seven_row_bias_direction_pairwise.csv")
    late_var_omnibus = pd.read_csv(
        TABLE_DIR / "paper_requested_seven_row_late_variability_omnibus.csv"
    )

    def term_row(df: pd.DataFrame, term: str) -> pd.Series:
        return df.loc[df["term"].eq(term)].iloc[0]

    group_term = term_row(late_group_omnibus, "C(group)")
    pointer_term = term_row(late_group_omnibus, "C(pointer)")
    group_pointer_term = term_row(late_group_omnibus, "C(group):C(pointer)")
    early_group_term = term_row(early_group_omnibus, "C(group)")
    early_pointer_term = term_row(early_group_omnibus, "C(pointer)")
    early_group_pointer_term = term_row(early_group_omnibus, "C(group):C(pointer)")
    washout_group_term = term_row(washout_group_omnibus, "C(group)")
    washout_pointer_term = term_row(washout_group_omnibus, "C(pointer)")
    washout_group_pointer_term = term_row(washout_group_omnibus, "C(group):C(pointer)")
    mean_phase = term_row(phase_mean_omnibus, "C(phase)")
    mean_phase_group = term_row(phase_mean_omnibus, "C(phase):C(group)")
    mean_phase_pointer = term_row(phase_mean_omnibus, "C(phase):C(pointer)")
    mean_phase_group_pointer = term_row(phase_mean_omnibus, "C(phase):C(group):C(pointer)")
    spread_phase = term_row(phase_spread_omnibus, "C(phase)")
    spread_phase_group = term_row(phase_spread_omnibus, "C(phase):C(group)")
    spread_phase_pointer = term_row(phase_spread_omnibus, "C(phase):C(pointer)")
    spread_phase_group_pointer = term_row(phase_spread_omnibus, "C(phase):C(group):C(pointer)")
    bias_direction_term = term_row(bias_omnibus, "C(bias_direction)")
    task_spatial_interaction = bias_frame_spatial_omnibus.loc[
        bias_frame_spatial_omnibus["bias_frame"].eq("Task-aligned")
        & bias_frame_spatial_omnibus["term"].eq("C(pointer):C(target_angles_degrees)")
    ].iloc[0]
    raw_spatial_interaction = bias_frame_spatial_omnibus.loc[
        bias_frame_spatial_omnibus["bias_frame"].eq("Raw signed")
        & bias_frame_spatial_omnibus["term"].eq("C(pointer):C(target_angles_degrees)")
    ].iloc[0]
    late_var_pointer = term_row(late_var_omnibus, "C(pointer)")

    late_group_sig = (
        late_group_zero.loc[late_group_zero["reject_holm"], "group"].astype(int).tolist()
    )
    late_group_nonsig = (
        late_group_zero.loc[~late_group_zero["reject_holm"], "group"].astype(int).tolist()
    )
    early_sig = early_zero.loc[early_zero["reject_holm"], "clamp_magnitude"].astype(int).tolist()

    pointer_zero_summary = (
        zero_tests_by_pointer_df.groupby(["measure", "pointer"], observed=True)["reject_holm"]
        .sum()
        .reset_index()
        .rename(columns={"reject_holm": "n_sig"})
    )
    late_mouse_sig = int(
        pointer_zero_summary.loc[
            pointer_zero_summary["measure"].eq("Late adaptation")
            & pointer_zero_summary["pointer"].eq("Mouse"),
            "n_sig",
        ].iloc[0]
    )
    late_trackpad_sig = int(
        pointer_zero_summary.loc[
            pointer_zero_summary["measure"].eq("Late adaptation")
            & pointer_zero_summary["pointer"].eq("Trackpad"),
            "n_sig",
        ].iloc[0]
    )

    late_sig_p = p_series_text(late_zero.loc[late_zero["reject_holm"], "p_holm"], label="p_Holm")
    late_large_zero = late_zero.loc[late_zero["clamp_magnitude"].between(135, 170)].copy()
    late_large_p = p_series_text(late_large_zero["p_holm"], label="p_Holm")
    early_large_zero = early_zero.loc[early_zero["clamp_magnitude"].between(135, 170)].copy()
    late_mouse_zero = zero_tests_by_pointer_df.loc[
        zero_tests_by_pointer_df["measure"].eq("Late adaptation")
        & zero_tests_by_pointer_df["pointer"].eq("Mouse")
    ].copy()
    late_trackpad_zero = zero_tests_by_pointer_df.loc[
        zero_tests_by_pointer_df["measure"].eq("Late adaptation")
        & zero_tests_by_pointer_df["pointer"].eq("Trackpad")
    ].copy()
    late_mouse_large_zero = late_mouse_zero.loc[
        late_mouse_zero["clamp_magnitude"].between(135, 170)
    ].copy()
    late_trackpad_large_zero = late_trackpad_zero.loc[
        late_trackpad_zero["clamp_magnitude"].between(135, 170)
    ].copy()
    late_mouse_sig_p = p_series_text(
        late_mouse_zero.loc[late_mouse_zero["reject_holm"], "p_holm"], label="p_Holm"
    )
    late_trackpad_sig_p = p_series_text(
        late_trackpad_zero.loc[late_trackpad_zero["reject_holm"], "p_holm"], label="p_Holm"
    )
    late_mouse_large_p = p_series_text(late_mouse_large_zero["p_holm"], label="p_Holm")
    late_trackpad_large_p = p_series_text(late_trackpad_large_zero["p_holm"], label="p_Holm")

    mean_sig = (
        phase_mean_paired.loc[phase_mean_paired["reject_holm"], "clamp_magnitude"]
        .astype(int)
        .tolist()
    )
    spread_sig = (
        phase_spread_paired.loc[phase_spread_paired["reject_holm"], "clamp_magnitude"]
        .astype(int)
        .tolist()
    )
    mean_high = phase_mean_paired.loc[phase_mean_paired["clamp_magnitude"].between(135, 170)].copy()
    spread_high = phase_spread_paired.loc[
        phase_spread_paired["clamp_magnitude"].between(135, 170)
    ].copy()

    peak_device = pointer_contrasts.loc[
        pointer_contrasts["measure"].eq("Late adaptation peak range (15-50 deg)")
    ].iloc[0]
    late_device = pointer_contrasts.loc[
        pointer_contrasts["measure"].eq("Late adaptation overall")
    ].iloc[0]
    early_var_device = pointer_contrasts.loc[
        pointer_contrasts["measure"].eq("Early rotation variability overall")
    ].iloc[0]
    late_var_device = pointer_contrasts.loc[
        pointer_contrasts["measure"].eq("Late adaptation variability overall")
    ].iloc[0]
    task_bias_device = pointer_contrasts.loc[
        pointer_contrasts["measure"].eq("Task-aligned baseline bias overall")
    ].iloc[0]
    raw_bias_device = pointer_contrasts.loc[
        pointer_contrasts["measure"].eq("Raw signed baseline bias overall")
    ].iloc[0]

    incorrect_vs_zero = bias_pairwise.loc[
        bias_pairwise["contrast"].eq("Incorrect vs Near-zero")
    ].iloc[0]
    correct_vs_zero = bias_pairwise.loc[bias_pairwise["contrast"].eq("Correct vs Near-zero")].iloc[
        0
    ]

    lines = [
        f"# {OUT_PREFIX} manuscript companion stats",
        "",
        "This summary lists the descriptive and inferential results from the chapter-level analyses.",
        "",
        "## Figure 2",
        (
            "- A 10 (assigned clamp group) x 2 (input device) between-subject ANOVA on participant-averaged late adaptation showed a main effect of group "
            f"(F({int(group_term['df_num'])}, {int(group_term['df_den'])}) = {group_term['f_value']:.2f}, "
            f"{p_text(float(group_term['p_value']))}, "
            f"{format_effect_ci('eta_p^2', group_term['partial_eta_sq'], group_term['partial_eta_sq_ci_low'], group_term['partial_eta_sq_ci_high'], digits=3)}), "
            "a main effect of input device "
            f"(F({int(pointer_term['df_num'])}, {int(pointer_term['df_den'])}) = {pointer_term['f_value']:.2f}, "
            f"{p_text(float(pointer_term['p_value']))}, "
            f"{format_effect_ci('eta_p^2', pointer_term['partial_eta_sq'], pointer_term['partial_eta_sq_ci_low'], pointer_term['partial_eta_sq_ci_high'], digits=3)}), "
            "and a group x device interaction "
            f"(F({int(group_pointer_term['df_num'])}, {int(group_pointer_term['df_den'])}) = {group_pointer_term['f_value']:.2f}, "
            f"{p_text(float(group_pointer_term['p_value']))}, "
            f"{format_effect_ci('eta_p^2', group_pointer_term['partial_eta_sq'], group_pointer_term['partial_eta_sq_ci_low'], group_pointer_term['partial_eta_sq_ci_high'], digits=3)})."
        ),
        (
            f"- Holm-corrected one-sample t-tests on participant means averaged across the four assigned clamp magnitudes were significant in groups {', '.join(str(v) for v in late_group_sig)} "
            f"({p_series_text(late_group_zero.loc[late_group_zero['reject_holm'], 'p_holm'], label='p_Holm')}), "
            f"but not in groups {', '.join(str(v) for v in late_group_nonsig)} ({p_series_text(late_group_zero.loc[~late_group_zero['reject_holm'], 'p_holm'], label='p_Holm')})."
        ),
        (
            f"- In the 40-clamp follow-up, pooled Holm-corrected one-sample t-tests against zero were significant at 29 of 40 clamp magnitudes ({late_sig_p}), but not at any clamp in the 135-170 deg range ({late_large_p}). "
            f"Split by device, Mouse showed significant late adaptation at {late_mouse_sig} of 40 clamp magnitudes ({late_mouse_sig_p}) and Trackpad at {late_trackpad_sig} of 40 ({late_trackpad_sig_p}); no 135-170 deg clamp was significant for Mouse ({late_mouse_large_p}) or Trackpad ({late_trackpad_large_p})."
        ),
        (
            "- A 10 (assigned clamp group) x 2 (input device) between-subject ANOVA on participant-averaged early STL showed a main effect of group "
            f"(F({int(early_group_term['df_num'])}, {int(early_group_term['df_den'])}) = {early_group_term['f_value']:.2f}, "
            f"{p_text(float(early_group_term['p_value']))}, "
            f"{format_effect_ci('eta_p^2', early_group_term['partial_eta_sq'], early_group_term['partial_eta_sq_ci_low'], early_group_term['partial_eta_sq_ci_high'], digits=3)}), "
            "but no main effect of input device "
            f"(F({int(early_pointer_term['df_num'])}, {int(early_pointer_term['df_den'])}) = {early_pointer_term['f_value']:.2f}, "
            f"{p_text(float(early_pointer_term['p_value']))}, "
            f"{format_effect_ci('eta_p^2', early_pointer_term['partial_eta_sq'], early_pointer_term['partial_eta_sq_ci_low'], early_pointer_term['partial_eta_sq_ci_high'], digits=3)}) "
            "and no group x device interaction "
            f"(F({int(early_group_pointer_term['df_num'])}, {int(early_group_pointer_term['df_den'])}) = {early_group_pointer_term['f_value']:.2f}, "
            f"{p_text(float(early_group_pointer_term['p_value']))}, "
            f"{format_effect_ci('eta_p^2', early_group_pointer_term['partial_eta_sq'], early_group_pointer_term['partial_eta_sq_ci_low'], early_group_pointer_term['partial_eta_sq_ci_high'], digits=3)}). "
            f"Holm-corrected one-sample t-tests against zero were significant only at the {early_sig[0]} deg clamp ({p_series_text(early_zero.loc[early_zero['reject_holm'], 'p_holm'], label='p_Holm')}), whereas none of the 135-170 deg clamps survived correction ({p_series_text(early_large_zero['p_holm'], label='p_Holm')})."
        ),
        (
            "- A corresponding 10 (assigned clamp group) x 2 (input device) between-subject ANOVA on participant-averaged washout STL showed a main effect of group "
            f"(F({int(washout_group_term['df_num'])}, {int(washout_group_term['df_den'])}) = {washout_group_term['f_value']:.2f}, "
            f"{p_text(float(washout_group_term['p_value']))}, "
            f"{format_effect_ci('eta_p^2', washout_group_term['partial_eta_sq'], washout_group_term['partial_eta_sq_ci_low'], washout_group_term['partial_eta_sq_ci_high'], digits=3)}), "
            "a main effect of input device "
            f"(F({int(washout_pointer_term['df_num'])}, {int(washout_pointer_term['df_den'])}) = {washout_pointer_term['f_value']:.2f}, "
            f"{p_text(float(washout_pointer_term['p_value']))}, "
            f"{format_effect_ci('eta_p^2', washout_pointer_term['partial_eta_sq'], washout_pointer_term['partial_eta_sq_ci_low'], washout_pointer_term['partial_eta_sq_ci_high'], digits=3)}), "
            "but no group x device interaction "
            f"(F({int(washout_group_pointer_term['df_num'])}, {int(washout_group_pointer_term['df_den'])}) = {washout_group_pointer_term['f_value']:.2f}, "
            f"{p_text(float(washout_group_pointer_term['p_value']))}, "
            f"{format_effect_ci('eta_p^2', washout_group_pointer_term['partial_eta_sq'], washout_group_pointer_term['partial_eta_sq_ci_low'], washout_group_pointer_term['partial_eta_sq_ci_high'], digits=3)}), "
            f"but no individual washout clamp survived Holm correction ({p_series_text(washout_zero['p_holm'], label='p_Holm')})."
        ),
        "",
        "## Figure 3",
        (
            "- A 2 (phase) x 10 (assigned clamp group) x 2 (input device) mixed ANOVA on participant-averaged mean hand angle showed a main effect of phase "
            f"(F({int(mean_phase['df_num'])}, {int(mean_phase['df_den'])}) = {mean_phase['f_value']:.2f}, "
            f"{p_text(float(mean_phase['p_value']))}, "
            f"{format_effect_ci('eta_p^2', mean_phase['partial_eta_sq'], mean_phase['partial_eta_sq_ci_low'], mean_phase['partial_eta_sq_ci_high'], digits=3)}), "
            "a phase x group interaction "
            f"(F({int(mean_phase_group['df_num'])}, {int(mean_phase_group['df_den'])}) = {mean_phase_group['f_value']:.2f}, "
            f"{p_text(float(mean_phase_group['p_value']))}, "
            f"{format_effect_ci('eta_p^2', mean_phase_group['partial_eta_sq'], mean_phase_group['partial_eta_sq_ci_low'], mean_phase_group['partial_eta_sq_ci_high'], digits=3)}), "
            "a phase x device interaction "
            f"(F({int(mean_phase_pointer['df_num'])}, {int(mean_phase_pointer['df_den'])}) = {mean_phase_pointer['f_value']:.2f}, "
            f"{p_text(float(mean_phase_pointer['p_value']))}, "
            f"{format_effect_ci('eta_p^2', mean_phase_pointer['partial_eta_sq'], mean_phase_pointer['partial_eta_sq_ci_low'], mean_phase_pointer['partial_eta_sq_ci_high'], digits=3)}), "
            "but no phase x group x device interaction "
            f"(F({int(mean_phase_group_pointer['df_num'])}, {int(mean_phase_group_pointer['df_den'])}) = {mean_phase_group_pointer['f_value']:.2f}, "
            f"{p_text(float(mean_phase_group_pointer['p_value']))}, "
            f"{format_effect_ci('eta_p^2', mean_phase_group_pointer['partial_eta_sq'], mean_phase_group_pointer['partial_eta_sq_ci_low'], mean_phase_group_pointer['partial_eta_sq_ci_high'], digits=3)}). "
            f"Holm-corrected paired t-tests were significant at {len(mean_sig)}/40 clamps ({p_series_text(phase_mean_paired.loc[phase_mean_paired['reject_holm'], 'p_holm'], label='p_Holm')}) and at 0/8 of the 135-170 deg clamps ({p_series_text(mean_high['p_holm'], label='p_Holm')})."
        ),
        (
            f"- In the peak range, late adaptation exceeded baseline by {peak_phase['mean_diff_late_minus_baseline']:.2f} deg "
            f"(t({int(peak_phase['n_participants']) - 1}) = {peak_phase['t']:.2f}, {p_text(float(peak_phase['p']))}, "
            f"{format_effect_ci('cohens_dz', peak_phase['cohens_dz'], peak_phase['cohens_dz_ci_low'], peak_phase['cohens_dz_ci_high'])}). "
            f"In the 135-170 deg range, the phase difference was negligible "
            f"(t({int(high_error_phase['n_participants']) - 1}) = {high_error_phase['t']:.2f}, {p_text(float(high_error_phase['p']))}, "
            f"{format_effect_ci('cohens_dz', high_error_phase['cohens_dz'], high_error_phase['cohens_dz_ci_low'], high_error_phase['cohens_dz_ci_high'])})."
        ),
        (
            "- A corresponding mixed ANOVA on between-participant spread of mean hand angle, quantified as absolute deviation from the phase-specific group median, showed a main effect of phase "
            f"(F({int(spread_phase['df_num'])}, {int(spread_phase['df_den'])}) = {spread_phase['f_value']:.2f}, "
            f"{p_text(float(spread_phase['p_value']))}, "
            f"{format_effect_ci('eta_p^2', spread_phase['partial_eta_sq'], spread_phase['partial_eta_sq_ci_low'], spread_phase['partial_eta_sq_ci_high'], digits=3)}), "
            "but no phase x group interaction "
            f"(F({int(spread_phase_group['df_num'])}, {int(spread_phase_group['df_den'])}) = {spread_phase_group['f_value']:.2f}, "
            f"{p_text(float(spread_phase_group['p_value']))}, "
            f"{format_effect_ci('eta_p^2', spread_phase_group['partial_eta_sq'], spread_phase_group['partial_eta_sq_ci_low'], spread_phase_group['partial_eta_sq_ci_high'], digits=3)}), "
            "no phase x device interaction "
            f"(F({int(spread_phase_pointer['df_num'])}, {int(spread_phase_pointer['df_den'])}) = {spread_phase_pointer['f_value']:.2f}, "
            f"{p_text(float(spread_phase_pointer['p_value']))}, "
            f"{format_effect_ci('eta_p^2', spread_phase_pointer['partial_eta_sq'], spread_phase_pointer['partial_eta_sq_ci_low'], spread_phase_pointer['partial_eta_sq_ci_high'], digits=3)}), "
            "and no phase x group x device interaction "
            f"(F({int(spread_phase_group_pointer['df_num'])}, {int(spread_phase_group_pointer['df_den'])}) = {spread_phase_group_pointer['f_value']:.2f}, "
            f"{p_text(float(spread_phase_group_pointer['p_value']))}, "
            f"{format_effect_ci('eta_p^2', spread_phase_group_pointer['partial_eta_sq'], spread_phase_group_pointer['partial_eta_sq_ci_low'], spread_phase_group_pointer['partial_eta_sq_ci_high'], digits=3)}). "
            f"Holm-corrected paired t-tests on spread were significant at {len(spread_sig)}/40 clamps ({p_series_text(phase_spread_paired.loc[phase_spread_paired['reject_holm'], 'p_holm'], label='p_Holm')}) and at "
            f"{int(phase_spread_paired.loc[phase_spread_paired['clamp_magnitude'].between(135, 170), 'reject_holm'].sum())}/8 of the 135-170 deg clamps (significant tests: {p_series_text(spread_high.loc[spread_high['reject_holm'], 'p_holm'], label='p_Holm')}; remaining test: {p_series_text(spread_high.loc[~spread_high['reject_holm'], 'p_holm'], label='p_Holm')})."
        ),
        (
            "- In the 135-170 deg clamps, bias direction predicted late adaptation "
            f"({wald_text(bias_direction_term)}, "
            f"{p_text(float(bias_direction_term['p_value_cluster']))}, "
            f"{format_effect_ci('eta_p^2', bias_direction_term['partial_eta_sq'], bias_direction_term['partial_eta_sq_ci_low'], bias_direction_term['partial_eta_sq_ci_high'], digits=3)}). "
            f"Relative to the near-zero group, Incorrect was {incorrect_vs_zero['estimate']:.2f} deg lower "
            f"[{incorrect_vs_zero['ci_low']:.2f}, {incorrect_vs_zero['ci_high']:.2f}] ({p_exact_text(float(incorrect_vs_zero['p_holm']), label='p_Holm')}) and Correct was {correct_vs_zero['estimate']:.2f} deg higher "
            f"[{correct_vs_zero['ci_low']:.2f}, {correct_vs_zero['ci_high']:.2f}] ({p_exact_text(float(correct_vs_zero['p_holm']), label='p_Holm')})."
        ),
        "",
        "## Figure 4",
        (
            f"- For late adaptation averaged across each participant's assigned clamps, Mouse participants were higher than Trackpad "
            f"(Mouse {late_device['mouse_mean']:.2f} [{late_device['mouse_ci_low']:.2f}, {late_device['mouse_ci_high']:.2f}], "
            f"Trackpad {late_device['trackpad_mean']:.2f} [{late_device['trackpad_ci_low']:.2f}, {late_device['trackpad_ci_high']:.2f}], "
            f"Welch t({late_device['welch_df']:.2f}) = {late_device['welch_t']:.2f}, {p_text(float(late_device['welch_p']))}, "
            f"{format_effect_ci('g', late_device['hedges_g_mouse_minus_trackpad'], late_device['hedges_g_ci_low'], late_device['hedges_g_ci_high'])}). "
            f"A clamp-adjusted model gave Mouse - Trackpad = {late_device['cluster_estimate_mouse_minus_trackpad']:.2f} deg "
            f"[{late_device['cluster_ci_low']:.2f}, {late_device['cluster_ci_high']:.2f}], {p_text(float(late_device['cluster_p']))}."
        ),
        (
            f"- The cleaner simple effect for the 'medium-size clamps' sentence is the peak-range contrast: Mouse {peak_device['mouse_mean']:.2f} "
            f"[{peak_device['mouse_ci_low']:.2f}, {peak_device['mouse_ci_high']:.2f}] versus Trackpad {peak_device['trackpad_mean']:.2f} "
            f"[{peak_device['trackpad_ci_low']:.2f}, {peak_device['trackpad_ci_high']:.2f}], Welch t({peak_device['welch_df']:.2f}) = {peak_device['welch_t']:.2f}, "
            f"{p_text(float(peak_device['welch_p']))}, {format_effect_ci('g', peak_device['hedges_g_mouse_minus_trackpad'], peak_device['hedges_g_ci_low'], peak_device['hedges_g_ci_high'])}; "
            f"clamp-adjusted Mouse - Trackpad = {peak_device['cluster_estimate_mouse_minus_trackpad']:.2f} deg "
            f"[{peak_device['cluster_ci_low']:.2f}, {peak_device['cluster_ci_high']:.2f}], {p_text(float(peak_device['cluster_p']))}."
        ),
        (
            f"- Task-aligned baseline bias showed no device difference "
            f"(Mouse {task_bias_device['mouse_mean']:.2f} [{task_bias_device['mouse_ci_low']:.2f}, {task_bias_device['mouse_ci_high']:.2f}], "
            f"Trackpad {task_bias_device['trackpad_mean']:.2f} [{task_bias_device['trackpad_ci_low']:.2f}, {task_bias_device['trackpad_ci_high']:.2f}], "
            f"Welch t({task_bias_device['welch_df']:.2f}) = {task_bias_device['welch_t']:.2f}, {p_text(float(task_bias_device['welch_p']))}, "
            f"{format_effect_ci('g', task_bias_device['hedges_g_mouse_minus_trackpad'], task_bias_device['hedges_g_ci_low'], task_bias_device['hedges_g_ci_high'])})."
        ),
        (
            f"- Raw signed baseline bias is the same device comparison without clamp-sign task alignment "
            f"(Mouse {raw_bias_device['mouse_mean']:.2f} [{raw_bias_device['mouse_ci_low']:.2f}, {raw_bias_device['mouse_ci_high']:.2f}], "
            f"Trackpad {raw_bias_device['trackpad_mean']:.2f} [{raw_bias_device['trackpad_ci_low']:.2f}, {raw_bias_device['trackpad_ci_high']:.2f}], "
            f"Welch t({raw_bias_device['welch_df']:.2f}) = {raw_bias_device['welch_t']:.2f}, {p_text(float(raw_bias_device['welch_p']))}, "
            f"{format_effect_ci('g', raw_bias_device['hedges_g_mouse_minus_trackpad'], raw_bias_device['hedges_g_ci_low'], raw_bias_device['hedges_g_ci_high'])})."
        ),
        (
            f"- Spatial bias pattern tests are separated by bias frame: Task-aligned "
            f"({wald_text(task_spatial_interaction)}, "
            f"{p_text(float(task_spatial_interaction['p_value_cluster']))}, "
            f"{format_effect_ci('eta_p^2', task_spatial_interaction['partial_eta_sq'], task_spatial_interaction['partial_eta_sq_ci_low'], task_spatial_interaction['partial_eta_sq_ci_high'], digits=3)}) "
            f"and raw signed "
            f"({wald_text(raw_spatial_interaction)}, "
            f"{p_text(float(raw_spatial_interaction['p_value_cluster']))}, "
            f"{format_effect_ci('eta_p^2', raw_spatial_interaction['partial_eta_sq'], raw_spatial_interaction['partial_eta_sq_ci_low'], raw_spatial_interaction['partial_eta_sq_ci_high'], digits=3)})."
        ),
        (
            "- For reach variability, the rank-based and log-scale device contrasts match the plotted medians better than mean-based Welch tests because the SD distributions are strongly right-skewed. "
            f"Early variability was higher for Mouse (participant-average median {early_var_device['mouse_median']:.2f} "
            f"[{early_var_device['mouse_median_ci_low']:.2f}, {early_var_device['mouse_median_ci_high']:.2f}] versus "
            f"{early_var_device['trackpad_median']:.2f} [{early_var_device['trackpad_median_ci_low']:.2f}, {early_var_device['trackpad_median_ci_high']:.2f}], "
            f"Mann-Whitney U = {early_var_device['mann_whitney_u']:.0f}, {p_text(float(early_var_device['mann_whitney_p']))}, "
            f"{format_effect_ci('rank-biserial', early_var_device['rank_biserial_mouse_minus_trackpad'], early_var_device['rank_biserial_ci_low'], early_var_device['rank_biserial_ci_high'], digits=3)}; "
            f"clamp-adjusted log-SD ratio = {early_var_device['log_cluster_ratio_mouse_over_trackpad']:.2f} "
            f"[{early_var_device['log_cluster_ratio_ci_low']:.2f}, {early_var_device['log_cluster_ratio_ci_high']:.2f}], {p_text(float(early_var_device['log_cluster_p']))}). "
            f"Late variability showed the same direction (participant-average median {late_var_device['mouse_median']:.2f} "
            f"[{late_var_device['mouse_median_ci_low']:.2f}, {late_var_device['mouse_median_ci_high']:.2f}] versus "
            f"{late_var_device['trackpad_median']:.2f} [{late_var_device['trackpad_median_ci_low']:.2f}, {late_var_device['trackpad_median_ci_high']:.2f}], "
            f"Mann-Whitney U = {late_var_device['mann_whitney_u']:.0f}, {p_text(float(late_var_device['mann_whitney_p']))}, "
            f"{format_effect_ci('rank-biserial', late_var_device['rank_biserial_mouse_minus_trackpad'], late_var_device['rank_biserial_ci_low'], late_var_device['rank_biserial_ci_high'], digits=3)}; "
            f"clamp-adjusted log-SD ratio = {late_var_device['log_cluster_ratio_mouse_over_trackpad']:.2f} "
            f"[{late_var_device['log_cluster_ratio_ci_low']:.2f}, {late_var_device['log_cluster_ratio_ci_high']:.2f}], {p_text(float(late_var_device['log_cluster_p']))}). "
            f"The raw-value late-variability term table still shows a pointer effect ({wald_text(late_var_pointer)}, "
            f"{p_text(float(late_var_pointer['p_value_cluster']))}, {format_effect_ci('eta_p^2', late_var_pointer['partial_eta_sq'], late_var_pointer['partial_eta_sq_ci_low'], late_var_pointer['partial_eta_sq_ci_high'], digits=3)})."
        ),
    ]
    return "\n".join(lines)


def build_manuscript_markdown(
    *,
    late_group_omnibus: pd.DataFrame,
    late_group_zero: pd.DataFrame,
    early_group_omnibus: pd.DataFrame,
    washout_group_omnibus: pd.DataFrame,
    pointer_contrasts: pd.DataFrame,
    zero_tests_by_pointer_df: pd.DataFrame,
    phase_mean_omnibus: pd.DataFrame,
    phase_mean_paired: pd.DataFrame,
    phase_spread_omnibus: pd.DataFrame,
    phase_spread_paired: pd.DataFrame,
    bias_frame_spatial_omnibus: pd.DataFrame,
) -> str:
    early_zero = pd.read_csv(TABLE_DIR / "paper_requested_seven_row_early_stl_zero_tests.csv")
    washout_zero = pd.read_csv(TABLE_DIR / "paper_requested_seven_row_washout_stl_zero_tests.csv")
    peak_phase = pd.read_csv(
        TABLE_DIR / "paper_requested_seven_row_peak_phase_paired_test.csv"
    ).iloc[0]
    high_error_phase = pd.read_csv(
        TABLE_DIR / "paper_requested_seven_row_high_error_phase_paired_test.csv"
    ).iloc[0]
    bias_omnibus = pd.read_csv(TABLE_DIR / "paper_requested_seven_row_bias_direction_omnibus.csv")
    bias_pairwise = pd.read_csv(TABLE_DIR / "paper_requested_seven_row_bias_direction_pairwise.csv")

    def term_row(df: pd.DataFrame, term: str) -> pd.Series:
        return df.loc[df["term"].eq(term)].iloc[0]

    def dec(value: float, digits: int = 3) -> str:
        text = f"{value:.{digits}f}"
        if text.startswith("0"):
            return text[1:]
        if text.startswith("-0"):
            return "-" + text[2:]
        return text

    def effect_text(
        label: str, value: float, ci_low: float, ci_high: float, *, digits: int = 3
    ) -> str:
        return f"{label} = {dec(value, digits)}, 95% CI [{dec(ci_low, digits)}, {dec(ci_high, digits)}]"

    group_term = term_row(late_group_omnibus, "C(group)")
    pointer_term = term_row(late_group_omnibus, "C(pointer)")
    group_pointer_term = term_row(late_group_omnibus, "C(group):C(pointer)")
    early_group_term = term_row(early_group_omnibus, "C(group)")
    early_pointer_term = term_row(early_group_omnibus, "C(pointer)")
    early_group_pointer_term = term_row(early_group_omnibus, "C(group):C(pointer)")
    washout_group_term = term_row(washout_group_omnibus, "C(group)")
    washout_pointer_term = term_row(washout_group_omnibus, "C(pointer)")
    washout_group_pointer_term = term_row(washout_group_omnibus, "C(group):C(pointer)")
    mean_phase = term_row(phase_mean_omnibus, "C(phase)")
    mean_phase_group = term_row(phase_mean_omnibus, "C(phase):C(group)")
    mean_phase_pointer = term_row(phase_mean_omnibus, "C(phase):C(pointer)")
    mean_phase_group_pointer = term_row(phase_mean_omnibus, "C(phase):C(group):C(pointer)")
    spread_phase = term_row(phase_spread_omnibus, "C(phase)")
    spread_phase_group = term_row(phase_spread_omnibus, "C(phase):C(group)")
    spread_phase_pointer = term_row(phase_spread_omnibus, "C(phase):C(pointer)")
    spread_phase_group_pointer = term_row(phase_spread_omnibus, "C(phase):C(group):C(pointer)")
    bias_direction_term = term_row(bias_omnibus, "C(bias_direction)")
    task_spatial_interaction = bias_frame_spatial_omnibus.loc[
        bias_frame_spatial_omnibus["bias_frame"].eq("Task-aligned")
        & bias_frame_spatial_omnibus["term"].eq("C(pointer):C(target_angles_degrees)")
    ].iloc[0]
    raw_spatial_interaction = bias_frame_spatial_omnibus.loc[
        bias_frame_spatial_omnibus["bias_frame"].eq("Raw signed")
        & bias_frame_spatial_omnibus["term"].eq("C(pointer):C(target_angles_degrees)")
    ].iloc[0]

    early_sig = early_zero.loc[early_zero["reject_holm"], "clamp_magnitude"].astype(int).tolist()
    mean_sig = (
        phase_mean_paired.loc[phase_mean_paired["reject_holm"], "clamp_magnitude"]
        .astype(int)
        .tolist()
    )
    spread_sig = (
        phase_spread_paired.loc[phase_spread_paired["reject_holm"], "clamp_magnitude"]
        .astype(int)
        .tolist()
    )

    early_high = early_zero.loc[early_zero["clamp_magnitude"].between(135, 170)].copy()
    washout_high = washout_zero.loc[washout_zero["clamp_magnitude"].between(135, 170)].copy()
    mean_high = phase_mean_paired.loc[phase_mean_paired["clamp_magnitude"].between(135, 170)].copy()
    spread_high = phase_spread_paired.loc[
        phase_spread_paired["clamp_magnitude"].between(135, 170)
    ].copy()

    late_group_sig = (
        late_group_zero.loc[late_group_zero["reject_holm"], "group"].astype(int).tolist()
    )
    late_group_nonsig = (
        late_group_zero.loc[~late_group_zero["reject_holm"], "group"].astype(int).tolist()
    )
    late_group_n = int(late_group_zero["n"].sum())
    late_group_sig_p = p_series_text(
        late_group_zero.loc[late_group_zero["reject_holm"], "p_holm"], label="p_Holm"
    )
    late_group_nonsig_p = p_series_text(
        late_group_zero.loc[~late_group_zero["reject_holm"], "p_holm"], label="p_Holm"
    )
    early_sig = early_zero.loc[early_zero["reject_holm"], "clamp_magnitude"].astype(int).tolist()

    late_device = pointer_contrasts.loc[
        pointer_contrasts["measure"].eq("Late adaptation overall")
    ].iloc[0]
    peak_device = pointer_contrasts.loc[
        pointer_contrasts["measure"].eq("Late adaptation peak range (15-50 deg)")
    ].iloc[0]
    early_var_device = pointer_contrasts.loc[
        pointer_contrasts["measure"].eq("Early rotation variability overall")
    ].iloc[0]
    late_var_device = pointer_contrasts.loc[
        pointer_contrasts["measure"].eq("Late adaptation variability overall")
    ].iloc[0]
    task_bias_device = pointer_contrasts.loc[
        pointer_contrasts["measure"].eq("Task-aligned baseline bias overall")
    ].iloc[0]
    raw_bias_device = pointer_contrasts.loc[
        pointer_contrasts["measure"].eq("Raw signed baseline bias overall")
    ].iloc[0]

    incorrect_vs_zero = bias_pairwise.loc[
        bias_pairwise["contrast"].eq("Incorrect vs Near-zero")
    ].iloc[0]
    correct_vs_zero = bias_pairwise.loc[bias_pairwise["contrast"].eq("Correct vs Near-zero")].iloc[
        0
    ]

    lines = [
        "# Manuscript-Ready Results Text",
        "",
        "These paragraphs summarize the statistical tables generated by the analysis workflows.",
        "",
        "## Figure 2",
        (
            f"A 10 (assigned clamp group) x 2 (input device) between-subject ANOVA on participant-averaged late adaptation (N = {late_group_n}) showed a main effect of group, "
            f"F({int(group_term['df_num'])}, {int(group_term['df_den'])}) = {group_term['f_value']:.2f}, {p_exact_text(float(group_term['p_value']))}, "
            f"{effect_text('eta_p^2', group_term['partial_eta_sq'], group_term['partial_eta_sq_ci_low'], group_term['partial_eta_sq_ci_high'])}, "
            "a main effect of input device, "
            f"F({int(pointer_term['df_num'])}, {int(pointer_term['df_den'])}) = {pointer_term['f_value']:.2f}, {p_exact_text(float(pointer_term['p_value']))}, "
            f"{effect_text('eta_p^2', pointer_term['partial_eta_sq'], pointer_term['partial_eta_sq_ci_low'], pointer_term['partial_eta_sq_ci_high'])}, "
            "and a group x device interaction, "
            f"F({int(group_pointer_term['df_num'])}, {int(group_pointer_term['df_den'])}) = {group_pointer_term['f_value']:.2f}, {p_exact_text(float(group_pointer_term['p_value']))}, "
            f"{effect_text('eta_p^2', group_pointer_term['partial_eta_sq'], group_pointer_term['partial_eta_sq_ci_low'], group_pointer_term['partial_eta_sq_ci_high'])}."
        ),
        "",
        (
            f"Holm-corrected one-sample t-tests on participant means averaged across the four clamp magnitudes within each assigned group showed significant late adaptation in groups {', '.join(str(v) for v in late_group_sig)} ({late_group_sig_p}), but not in groups {', '.join(str(v) for v in late_group_nonsig)} ({late_group_nonsig_p})."
        ),
        "",
        (
            f"A 10 (assigned clamp group) x 2 (input device) between-subject ANOVA on participant-averaged early STL (N = {int(early_group_omnibus['n_participants'].iloc[0])}) showed a main effect of group, "
            f"F({int(early_group_term['df_num'])}, {int(early_group_term['df_den'])}) = {early_group_term['f_value']:.2f}, {p_exact_text(float(early_group_term['p_value']))}, "
            f"{effect_text('eta_p^2', early_group_term['partial_eta_sq'], early_group_term['partial_eta_sq_ci_low'], early_group_term['partial_eta_sq_ci_high'])}, but no main effect of input device, "
            f"F({int(early_pointer_term['df_num'])}, {int(early_pointer_term['df_den'])}) = {early_pointer_term['f_value']:.2f}, {p_exact_text(float(early_pointer_term['p_value']))}, "
            f"{effect_text('eta_p^2', early_pointer_term['partial_eta_sq'], early_pointer_term['partial_eta_sq_ci_low'], early_pointer_term['partial_eta_sq_ci_high'])}, and no group x device interaction, "
            f"F({int(early_group_pointer_term['df_num'])}, {int(early_group_pointer_term['df_den'])}) = {early_group_pointer_term['f_value']:.2f}, {p_exact_text(float(early_group_pointer_term['p_value']))}, "
            f"{effect_text('eta_p^2', early_group_pointer_term['partial_eta_sq'], early_group_pointer_term['partial_eta_sq_ci_low'], early_group_pointer_term['partial_eta_sq_ci_high'])}. "
            f"Holm-corrected one-sample t-tests against zero were significant only at the {early_sig[0]} deg clamp ({p_series_text(early_zero.loc[early_zero['reject_holm'], 'p_holm'], label='p_Holm')}), whereas none of the 135-170 deg clamps survived correction ({p_series_text(early_high['p_holm'], label='p_Holm')}). "
            f"A corresponding 10 (assigned clamp group) x 2 (input device) between-subject ANOVA on participant-averaged washout STL (N = {int(washout_group_omnibus['n_participants'].iloc[0])}) showed a main effect of group, "
            f"F({int(washout_group_term['df_num'])}, {int(washout_group_term['df_den'])}) = {washout_group_term['f_value']:.2f}, {p_exact_text(float(washout_group_term['p_value']))}, "
            f"{effect_text('eta_p^2', washout_group_term['partial_eta_sq'], washout_group_term['partial_eta_sq_ci_low'], washout_group_term['partial_eta_sq_ci_high'])}, a main effect of input device, "
            f"F({int(washout_pointer_term['df_num'])}, {int(washout_pointer_term['df_den'])}) = {washout_pointer_term['f_value']:.2f}, {p_exact_text(float(washout_pointer_term['p_value']))}, "
            f"{effect_text('eta_p^2', washout_pointer_term['partial_eta_sq'], washout_pointer_term['partial_eta_sq_ci_low'], washout_pointer_term['partial_eta_sq_ci_high'])}, but no group x device interaction, "
            f"F({int(washout_group_pointer_term['df_num'])}, {int(washout_group_pointer_term['df_den'])}) = {washout_group_pointer_term['f_value']:.2f}, {p_exact_text(float(washout_group_pointer_term['p_value']))}, "
            f"{effect_text('eta_p^2', washout_group_pointer_term['partial_eta_sq'], washout_group_pointer_term['partial_eta_sq_ci_low'], washout_group_pointer_term['partial_eta_sq_ci_high'])}. "
            f"No individual washout clamp survived Holm correction ({p_series_text(washout_zero['p_holm'], label='p_Holm')}), including the 135-170 deg range ({p_series_text(washout_high['p_holm'], label='p_Holm')})."
        ),
        "",
        "## Figure 3",
        (
            f"A 2 (phase: baseline vs late adaptation) x 10 (assigned clamp group) x 2 (input device) mixed ANOVA on participant-averaged mean hand angle (N = {int(mean_phase['n_participants'])}) showed a main effect of phase, "
            f"F({int(mean_phase['df_num'])}, {int(mean_phase['df_den'])}) = {mean_phase['f_value']:.2f}, {p_exact_text(float(mean_phase['p_value']))}, "
            f"{effect_text('eta_p^2', mean_phase['partial_eta_sq'], mean_phase['partial_eta_sq_ci_low'], mean_phase['partial_eta_sq_ci_high'])}, a phase x group interaction, "
            f"F({int(mean_phase_group['df_num'])}, {int(mean_phase_group['df_den'])}) = {mean_phase_group['f_value']:.2f}, {p_exact_text(float(mean_phase_group['p_value']))}, "
            f"{effect_text('eta_p^2', mean_phase_group['partial_eta_sq'], mean_phase_group['partial_eta_sq_ci_low'], mean_phase_group['partial_eta_sq_ci_high'])}, and a phase x device interaction, "
            f"F({int(mean_phase_pointer['df_num'])}, {int(mean_phase_pointer['df_den'])}) = {mean_phase_pointer['f_value']:.2f}, {p_exact_text(float(mean_phase_pointer['p_value']))}, "
            f"{effect_text('eta_p^2', mean_phase_pointer['partial_eta_sq'], mean_phase_pointer['partial_eta_sq_ci_low'], mean_phase_pointer['partial_eta_sq_ci_high'])}, but no phase x group x device interaction, "
            f"F({int(mean_phase_group_pointer['df_num'])}, {int(mean_phase_group_pointer['df_den'])}) = {mean_phase_group_pointer['f_value']:.2f}, {p_exact_text(float(mean_phase_group_pointer['p_value']))}, "
            f"{effect_text('eta_p^2', mean_phase_group_pointer['partial_eta_sq'], mean_phase_group_pointer['partial_eta_sq_ci_low'], mean_phase_group_pointer['partial_eta_sq_ci_high'])}. "
            f"Holm-corrected paired t-tests were significant at {len(mean_sig)} of 40 clamp magnitudes ({p_series_text(phase_mean_paired.loc[phase_mean_paired['reject_holm'], 'p_holm'], label='p_Holm')}), but not at any of the 8 magnitudes in the 135-170 deg range ({p_series_text(mean_high['p_holm'], label='p_Holm')}). "
            f"In the peak range, late adaptation exceeded baseline by {peak_phase['mean_diff_late_minus_baseline']:.2f} deg, t({int(peak_phase['n_participants']) - 1}) = {peak_phase['t']:.2f}, {p_exact_text(float(peak_phase['p']))}, "
            f"{effect_text('cohens_dz', peak_phase['cohens_dz'], peak_phase['cohens_dz_ci_low'], peak_phase['cohens_dz_ci_high'])}. "
            f"By contrast, the phase difference was negligible in the 135-170 deg range, t({int(high_error_phase['n_participants']) - 1}) = {high_error_phase['t']:.2f}, {p_exact_text(float(high_error_phase['p']))}, "
            f"{effect_text('cohens_dz', high_error_phase['cohens_dz'], high_error_phase['cohens_dz_ci_low'], high_error_phase['cohens_dz_ci_high'])}."
        ),
        "",
        (
            f"A corresponding mixed ANOVA on between-participant spread of mean hand angle, quantified as absolute deviation from the phase-specific group median (N = {int(spread_phase['n_participants'])}), showed a main effect of phase, "
            f"F({int(spread_phase['df_num'])}, {int(spread_phase['df_den'])}) = {spread_phase['f_value']:.2f}, {p_exact_text(float(spread_phase['p_value']))}, "
            f"{effect_text('eta_p^2', spread_phase['partial_eta_sq'], spread_phase['partial_eta_sq_ci_low'], spread_phase['partial_eta_sq_ci_high'])}, but no phase x group interaction, "
            f"F({int(spread_phase_group['df_num'])}, {int(spread_phase_group['df_den'])}) = {spread_phase_group['f_value']:.2f}, {p_exact_text(float(spread_phase_group['p_value']))}, "
            f"{effect_text('eta_p^2', spread_phase_group['partial_eta_sq'], spread_phase_group['partial_eta_sq_ci_low'], spread_phase_group['partial_eta_sq_ci_high'])}, no phase x device interaction, "
            f"F({int(spread_phase_pointer['df_num'])}, {int(spread_phase_pointer['df_den'])}) = {spread_phase_pointer['f_value']:.2f}, {p_exact_text(float(spread_phase_pointer['p_value']))}, "
            f"{effect_text('eta_p^2', spread_phase_pointer['partial_eta_sq'], spread_phase_pointer['partial_eta_sq_ci_low'], spread_phase_pointer['partial_eta_sq_ci_high'])}, and no phase x group x device interaction, "
            f"F({int(spread_phase_group_pointer['df_num'])}, {int(spread_phase_group_pointer['df_den'])}) = {spread_phase_group_pointer['f_value']:.2f}, {p_exact_text(float(spread_phase_group_pointer['p_value']))}, "
            f"{effect_text('eta_p^2', spread_phase_group_pointer['partial_eta_sq'], spread_phase_group_pointer['partial_eta_sq_ci_low'], spread_phase_group_pointer['partial_eta_sq_ci_high'])}. "
            f"Holm-corrected paired tests on spread were significant at {len(spread_sig)} of 40 clamp magnitudes ({p_series_text(phase_spread_paired.loc[phase_spread_paired['reject_holm'], 'p_holm'], label='p_Holm')}), including "
            f"{int(spread_high['reject_holm'].sum())} of the 8 magnitudes in the 135-170 deg range (significant tests: {p_series_text(spread_high.loc[spread_high['reject_holm'], 'p_holm'], label='p_Holm')}; remaining test: {p_series_text(spread_high.loc[~spread_high['reject_holm'], 'p_holm'], label='p_Holm')})."
        ),
        "",
        (
            "Within the 135-170 deg clamp range, bias direction predicted late adaptation, "
            f"{wald_text(bias_direction_term)}, {p_exact_text(float(bias_direction_term['p_value_cluster']))}, "
            f"{effect_text('eta_p^2', bias_direction_term['partial_eta_sq'], bias_direction_term['partial_eta_sq_ci_low'], bias_direction_term['partial_eta_sq_ci_high'])}. "
            f"Relative to the near-zero group, the Incorrect group showed less late adaptation ({incorrect_vs_zero['estimate']:.2f} deg, 95% CI [{incorrect_vs_zero['ci_low']:.2f}, {incorrect_vs_zero['ci_high']:.2f}], {p_exact_text(float(incorrect_vs_zero['p_holm']), label='p_Holm')}), whereas the Correct group showed more late adaptation ({correct_vs_zero['estimate']:.2f} deg, 95% CI [{correct_vs_zero['ci_low']:.2f}, {correct_vs_zero['ci_high']:.2f}], {p_exact_text(float(correct_vs_zero['p_holm']), label='p_Holm')})."
        ),
        "",
        "## Figure 4",
        (
            "Across each participant's assigned clamp magnitudes, late adaptation was higher for Mouse than Trackpad (Mouse mean = "
            f"{late_device['mouse_mean']:.2f} deg, 95% CI [{late_device['mouse_ci_low']:.2f}, {late_device['mouse_ci_high']:.2f}]; Trackpad mean = {late_device['trackpad_mean']:.2f} deg, 95% CI [{late_device['trackpad_ci_low']:.2f}, {late_device['trackpad_ci_high']:.2f}]), "
            f"Welch t({late_device['welch_df']:.2f}) = {late_device['welch_t']:.2f}, {p_exact_text(float(late_device['welch_p']))}, "
            f"{effect_text('g', late_device['hedges_g_mouse_minus_trackpad'], late_device['hedges_g_ci_low'], late_device['hedges_g_ci_high'])}. "
            f"A clamp-adjusted clustered model gave a Mouse-Trackpad difference of {late_device['cluster_estimate_mouse_minus_trackpad']:.2f} deg, 95% CI [{late_device['cluster_ci_low']:.2f}, {late_device['cluster_ci_high']:.2f}], {p_exact_text(float(late_device['cluster_p']))}. "
            "This device difference was particularly clear in the peak range (15-50 deg), where Mouse participants showed greater adaptation than Trackpad participants "
            f"(Mouse mean = {peak_device['mouse_mean']:.2f} deg, 95% CI [{peak_device['mouse_ci_low']:.2f}, {peak_device['mouse_ci_high']:.2f}]; Trackpad mean = {peak_device['trackpad_mean']:.2f} deg, 95% CI [{peak_device['trackpad_ci_low']:.2f}, {peak_device['trackpad_ci_high']:.2f}]), "
            f"Welch t({peak_device['welch_df']:.2f}) = {peak_device['welch_t']:.2f}, {p_exact_text(float(peak_device['welch_p']))}, "
            f"{effect_text('g', peak_device['hedges_g_mouse_minus_trackpad'], peak_device['hedges_g_ci_low'], peak_device['hedges_g_ci_high'])}; the corresponding clamp-adjusted Mouse-Trackpad difference was {peak_device['cluster_estimate_mouse_minus_trackpad']:.2f} deg, 95% CI [{peak_device['cluster_ci_low']:.2f}, {peak_device['cluster_ci_high']:.2f}], {p_exact_text(float(peak_device['cluster_p']))}."
        ),
        "",
        (
            "Task-aligned baseline bias did not differ by device (Mouse mean = "
            f"{task_bias_device['mouse_mean']:.2f} deg, 95% CI [{task_bias_device['mouse_ci_low']:.2f}, {task_bias_device['mouse_ci_high']:.2f}]; Trackpad mean = {task_bias_device['trackpad_mean']:.2f} deg, 95% CI [{task_bias_device['trackpad_ci_low']:.2f}, {task_bias_device['trackpad_ci_high']:.2f}]), "
            f"Welch t({task_bias_device['welch_df']:.2f}) = {task_bias_device['welch_t']:.2f}, {p_exact_text(float(task_bias_device['welch_p']))}, "
            f"{effect_text('g', task_bias_device['hedges_g_mouse_minus_trackpad'], task_bias_device['hedges_g_ci_low'], task_bias_device['hedges_g_ci_high'])}. "
            "The raw signed counterpart, computed from unflipped baseline hand angles without clamp-sign task alignment, was "
            f"Mouse mean = {raw_bias_device['mouse_mean']:.2f} deg, 95% CI [{raw_bias_device['mouse_ci_low']:.2f}, {raw_bias_device['mouse_ci_high']:.2f}], and Trackpad mean = {raw_bias_device['trackpad_mean']:.2f} deg, 95% CI [{raw_bias_device['trackpad_ci_low']:.2f}, {raw_bias_device['trackpad_ci_high']:.2f}], "
            f"Welch t({raw_bias_device['welch_df']:.2f}) = {raw_bias_device['welch_t']:.2f}, {p_exact_text(float(raw_bias_device['welch_p']))}, "
            f"{effect_text('g', raw_bias_device['hedges_g_mouse_minus_trackpad'], raw_bias_device['hedges_g_ci_low'], raw_bias_device['hedges_g_ci_high'])}. "
            f"Spatial bias patterns differed by device across targets when tested in the task-aligned frame, {wald_text(task_spatial_interaction)}, {p_exact_text(float(task_spatial_interaction['p_value_cluster']))}, {effect_text('eta_p^2', task_spatial_interaction['partial_eta_sq'], task_spatial_interaction['partial_eta_sq_ci_low'], task_spatial_interaction['partial_eta_sq_ci_high'])}, and in the raw signed frame, {wald_text(raw_spatial_interaction)}, {p_exact_text(float(raw_spatial_interaction['p_value_cluster']))}, {effect_text('eta_p^2', raw_spatial_interaction['partial_eta_sq'], raw_spatial_interaction['partial_eta_sq_ci_low'], raw_spatial_interaction['partial_eta_sq_ci_high'])}."
        ),
        "",
        (
            "Because participant-level reach-variability estimates were strictly positive and strongly right-skewed, device differences in variability were summarised with medians and IQRs and tested with clamp-adjusted log-scale models, with rank-based tests reported as robustness checks. "
            f"Early reach variability was higher for Mouse than Trackpad (Mouse median = {early_var_device['mouse_median']:.2f} deg, IQR = {early_var_device['mouse_q25']:.2f}-{early_var_device['mouse_q75']:.2f}, 95% CI [{early_var_device['mouse_median_ci_low']:.2f}, {early_var_device['mouse_median_ci_high']:.2f}]; Trackpad median = {early_var_device['trackpad_median']:.2f} deg, IQR = {early_var_device['trackpad_q25']:.2f}-{early_var_device['trackpad_q75']:.2f}, 95% CI [{early_var_device['trackpad_median_ci_low']:.2f}, {early_var_device['trackpad_median_ci_high']:.2f}]), "
            f"Mann-Whitney U = {early_var_device['mann_whitney_u']:.0f}, {p_exact_text(float(early_var_device['mann_whitney_p']))}, {effect_text('rank-biserial', early_var_device['rank_biserial_mouse_minus_trackpad'], early_var_device['rank_biserial_ci_low'], early_var_device['rank_biserial_ci_high'])}. "
            f"In the clamp-adjusted log-scale model, Mouse variability was {early_var_device['log_cluster_ratio_mouse_over_trackpad']:.2f} times higher than Trackpad, 95% CI [{early_var_device['log_cluster_ratio_ci_low']:.2f}, {early_var_device['log_cluster_ratio_ci_high']:.2f}], {p_exact_text(float(early_var_device['log_cluster_p']))}. "
            f"Late reach variability showed the same direction (Mouse median = {late_var_device['mouse_median']:.2f} deg, IQR = {late_var_device['mouse_q25']:.2f}-{late_var_device['mouse_q75']:.2f}, 95% CI [{late_var_device['mouse_median_ci_low']:.2f}, {late_var_device['mouse_median_ci_high']:.2f}]; Trackpad median = {late_var_device['trackpad_median']:.2f} deg, IQR = {late_var_device['trackpad_q25']:.2f}-{late_var_device['trackpad_q75']:.2f}, 95% CI [{late_var_device['trackpad_median_ci_low']:.2f}, {late_var_device['trackpad_median_ci_high']:.2f}]), "
            f"Mann-Whitney U = {late_var_device['mann_whitney_u']:.0f}, {p_exact_text(float(late_var_device['mann_whitney_p']))}, {effect_text('rank-biserial', late_var_device['rank_biserial_mouse_minus_trackpad'], late_var_device['rank_biserial_ci_low'], late_var_device['rank_biserial_ci_high'])}. "
            f"In the clamp-adjusted log-scale model, Mouse variability was {late_var_device['log_cluster_ratio_mouse_over_trackpad']:.2f} times higher than Trackpad, 95% CI [{late_var_device['log_cluster_ratio_ci_low']:.2f}, {late_var_device['log_cluster_ratio_ci_high']:.2f}], {p_exact_text(float(late_var_device['log_cluster_p']))}."
        ),
        "",
        "## Methods Wording For Reach Variability",
        (
            "For device comparisons in reach variability, participant-level SD measures were strictly positive and strongly right-skewed, so raw-scale mean comparisons were sensitive to a small number of high-variability observations. Device effects were therefore tested with clamp-adjusted linear models on log-transformed variability values using participant-clustered robust standard errors, and are reported as Mouse/Trackpad ratios with 95% confidence intervals. Medians, IQRs, and bootstrap 95% confidence intervals are reported descriptively, with Mann-Whitney tests and rank-biserial effect sizes provided as robustness checks."
        ),
    ]
    return "\n".join(lines)


def run_all_analyses() -> dict[str, object]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}

    late_subject = load_table("late_adaptation_subject_means.csv")
    early_stl = load_table("stl_early_subject_means.csv", add_pointer=True)
    washout_stl = load_table("stl_washout_subject_means.csv", add_pointer=True)
    baseline_reference = load_table("baseline_reference_subject_means.csv")
    late_group_subject = participant_group_means(late_subject, value_col="value")
    early_group_subject = participant_group_means(early_stl, value_col="value")
    washout_group_subject = participant_group_means(washout_stl, value_col="value")

    late_reach_variability = load_table("late_variability_subject.csv")
    early_variability = load_table("paper_requested_seven_row_early_stl_variability_subject.csv")

    zero_tests_by_pointer_df = pd.concat(
        [
            zero_tests_by_pointer(late_subject, measure="Late adaptation"),
            zero_tests_by_pointer(early_stl, measure="Early STL"),
            zero_tests_by_pointer(washout_stl, measure="Washout STL"),
        ],
        ignore_index=True,
    )
    save_table(zero_tests_by_pointer_df, f"{OUT_PREFIX}_zero_tests_by_pointer.csv", outputs)

    late_group_omnibus = between_subject_anova_table(
        late_group_subject,
        "value ~ C(group) * C(pointer)",
    )
    save_table(late_group_omnibus, f"{OUT_PREFIX}_late_group_omnibus.csv", outputs)

    late_group_zero = zero_tests_by_group(late_group_subject, group_col="group", value_col="value")
    save_table(late_group_zero, f"{OUT_PREFIX}_late_group_zero_tests.csv", outputs)

    early_group_omnibus = between_subject_anova_table(
        early_group_subject,
        "value ~ C(group) * C(pointer)",
    )
    save_table(early_group_omnibus, f"{OUT_PREFIX}_early_group_omnibus.csv", outputs)

    washout_group_omnibus = between_subject_anova_table(
        washout_group_subject,
        "value ~ C(group) * C(pointer)",
    )
    save_table(washout_group_omnibus, f"{OUT_PREFIX}_washout_group_omnibus.csv", outputs)

    phase_mean_group_data = phase_group_dataset(
        baseline_reference,
        late_subject,
        baseline_value_col="value",
        late_value_col="value",
    )
    phase_mean_omnibus = mixed_anova_table(
        phase_mean_group_data,
        "value ~ C(ppid) + C(phase) * C(group) * C(pointer)",
    )
    save_table(phase_mean_omnibus, f"{OUT_PREFIX}_phase_mean_omnibus.csv", outputs)

    phase_mean_descriptives = phase_descriptives(
        baseline_reference,
        late_subject,
        baseline_value_col="value",
        late_value_col="value",
        measure="Late adaptation mean",
    )
    save_table(phase_mean_descriptives, f"{OUT_PREFIX}_phase_mean_descriptives.csv", outputs)

    phase_mean_paired = paired_tests_by_clamp(
        baseline_reference,
        late_subject,
        baseline_value_col="value",
        late_value_col="value",
        measure="Late adaptation mean",
    )
    save_table(phase_mean_paired, f"{OUT_PREFIX}_phase_mean_by_clamp_paired_tests.csv", outputs)

    phase_spread_group_data = phase_group_spread_dataset(
        baseline_reference,
        late_subject,
        baseline_value_col="value",
        late_value_col="value",
    )
    phase_spread_omnibus = mixed_anova_table(
        phase_spread_group_data,
        "abs_dev ~ C(ppid) + C(phase) * C(group) * C(pointer)",
    )
    save_table(phase_spread_omnibus, f"{OUT_PREFIX}_phase_spread_omnibus.csv", outputs)

    spread_descriptives = phase_spread_descriptives(
        baseline_reference,
        late_subject,
        baseline_value_col="value",
        late_value_col="value",
        measure="Between-participant spread of mean hand angle",
    )
    save_table(spread_descriptives, f"{OUT_PREFIX}_phase_spread_descriptives.csv", outputs)

    phase_spread_paired = paired_spread_tests_by_clamp(
        baseline_reference,
        late_subject,
        baseline_value_col="value",
        late_value_col="value",
        measure="Between-participant spread of mean hand angle",
    )
    save_table(phase_spread_paired, f"{OUT_PREFIX}_phase_spread_by_clamp_paired_tests.csv", outputs)

    task_aligned_bias = load_table("baseline_bias_by_ppid_clamp.csv")
    raw_signed_bias = load_or_build_raw_signed_baseline_bias()
    task_aligned_spatial_bias, raw_signed_spatial_bias = load_or_build_bias_frame_spatial_tables()
    bias_frame_spatial_omnibus = bias_frame_spatial_omnibus_table(
        task_aligned_spatial_bias,
        raw_signed_spatial_bias,
    )
    save_table(bias_frame_spatial_omnibus, f"{OUT_PREFIX}_bias_frame_spatial_omnibus.csv", outputs)
    pointer_contrasts = pd.DataFrame.from_records(
        [
            pointer_contrast_row(
                late_subject,
                measure="Late adaptation overall",
                value_col="value",
                cluster_formula="value ~ C(pointer, Treatment(reference='Trackpad')) + C(clamp_magnitude)",
            ),
            pointer_contrast_row(
                late_subject.loc[late_subject["clamp_magnitude"].between(15, 50)].copy(),
                measure="Late adaptation peak range (15-50 deg)",
                value_col="value",
                cluster_formula="value ~ C(pointer, Treatment(reference='Trackpad')) + C(clamp_magnitude)",
            ),
            pointer_contrast_row(
                early_variability,
                measure="Early rotation variability overall",
                value_col="value",
                cluster_formula="value ~ C(pointer, Treatment(reference='Trackpad')) + C(clamp_magnitude)",
            ),
            pointer_contrast_row(
                late_reach_variability,
                measure="Late adaptation variability overall",
                value_col="value",
                cluster_formula="value ~ C(pointer, Treatment(reference='Trackpad')) + C(clamp_magnitude)",
            ),
            pointer_contrast_row(
                task_aligned_bias,
                measure="Task-aligned baseline bias overall",
                value_col="baseline_bias",
                cluster_formula="value ~ C(pointer, Treatment(reference='Trackpad')) + C(clamp_magnitude)",
            ),
            pointer_contrast_row(
                raw_signed_bias,
                measure="Raw signed baseline bias overall",
                value_col="raw_signed_baseline_bias",
                cluster_formula=(
                    "value ~ C(pointer, Treatment(reference='Trackpad')) "
                    "+ C(clamp_magnitude) + C(target_clamp_sign)"
                ),
            ),
        ]
    )
    save_table(pointer_contrasts, f"{OUT_PREFIX}_pointer_contrasts.csv", outputs)

    high_error_bias_overall = grouped_descriptives(
        load_bias_direction_pairings(),
        group_cols=["bias_direction"],
        value_col="late_adapt",
    )
    save_table(
        high_error_bias_overall, f"{OUT_PREFIX}_high_error_bias_direction_overall.csv", outputs
    )

    sentence_map_df = sentence_map()
    save_table(sentence_map_df, f"{OUT_PREFIX}_sentence_map.csv", outputs)

    summary_markdown = build_summary_markdown(
        late_group_omnibus=late_group_omnibus,
        late_group_zero=late_group_zero,
        early_group_omnibus=early_group_omnibus,
        washout_group_omnibus=washout_group_omnibus,
        pointer_contrasts=pointer_contrasts,
        zero_tests_by_pointer_df=zero_tests_by_pointer_df,
        phase_mean_omnibus=phase_mean_omnibus,
        phase_mean_paired=phase_mean_paired,
        phase_spread_omnibus=phase_spread_omnibus,
        phase_spread_paired=phase_spread_paired,
        high_error_bias_overall=high_error_bias_overall,
        bias_frame_spatial_omnibus=bias_frame_spatial_omnibus,
    )
    SUMMARY_MARKDOWN_PATH.write_text(summary_markdown, encoding="utf-8")

    manuscript_markdown = build_manuscript_markdown(
        late_group_omnibus=late_group_omnibus,
        late_group_zero=late_group_zero,
        early_group_omnibus=early_group_omnibus,
        washout_group_omnibus=washout_group_omnibus,
        pointer_contrasts=pointer_contrasts,
        zero_tests_by_pointer_df=zero_tests_by_pointer_df,
        phase_mean_omnibus=phase_mean_omnibus,
        phase_mean_paired=phase_mean_paired,
        phase_spread_omnibus=phase_spread_omnibus,
        phase_spread_paired=phase_spread_paired,
        bias_frame_spatial_omnibus=bias_frame_spatial_omnibus,
    )
    MANUSCRIPT_MARKDOWN_PATH.write_text(manuscript_markdown, encoding="utf-8")

    return {
        "summary_markdown": SUMMARY_MARKDOWN_PATH,
        "manuscript_markdown": MANUSCRIPT_MARKDOWN_PATH,
        "tables": outputs,
    }
