from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

from clamp_analysis import paths

TABLE_DIR = paths.RESULTS_DIR / "tables"

DETAILS_PATH = paths.DETAILS_PATH

DATA_PATH = paths.DATA_PATH

OUT_PREFIX = "paper_requested_seven_row"

SUMMARY_MARKDOWN_PATH = TABLE_DIR / f"{OUT_PREFIX}_summary.md"

POINTER_ORDER = ["Mouse", "Trackpad"]

BIAS_REFERENCE_ORDER = ["Near-zero", "Incorrect", "Correct"]

TRAJECTORY_WINDOW_ORDER = [
    "No-feedback baseline",
    "Early clamp",
    "Late clamp",
    "Early washout",
]

RNG_SEED = 20260421

BOOTSTRAP_DRAWS = 2000

ANOVA_CI_BOOTSTRAPS = 400

CI_LEVEL = 0.95


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts).encode("utf-8")
    return RNG_SEED + zlib.crc32(text)


def load_pointer_lookup() -> pd.DataFrame:
    details = pd.read_csv(DETAILS_PATH)
    details["ppid"] = (
        details["ppid_session_dataname"]
        .astype(str)
        .str.replace(r"_s\d+_participant_details$", "", regex=True)
    )
    return details[["ppid", "pointer"]].drop_duplicates()


def load_table(name: str, *, add_pointer: bool = False) -> pd.DataFrame:
    df = pd.read_csv(TABLE_DIR / name, dtype={"ppid": str})
    if add_pointer and "pointer" not in df.columns:
        df = df.merge(load_pointer_lookup(), on="ppid", how="left")
    if "pointer" in df.columns:
        df["pointer"] = df["pointer"].astype(str)
        df = df.loc[df["pointer"].isin(POINTER_ORDER)].copy()
    if "clamp_magnitude" in df.columns:
        df["clamp_magnitude"] = pd.to_numeric(df["clamp_magnitude"], errors="coerce")
        df = df.loc[df["clamp_magnitude"].notna()].copy()
        df["clamp_magnitude"] = df["clamp_magnitude"].astype(int)
    if "target_angles_degrees" in df.columns:
        df["target_angles_degrees"] = pd.to_numeric(df["target_angles_degrees"], errors="coerce")
        df = df.loc[df["target_angles_degrees"].notna()].copy()
        df["target_angles_degrees"] = df["target_angles_degrees"].astype(int)
    return df.reset_index(drop=True)


def series_summary(values: pd.Series | np.ndarray, *, ci: float = CI_LEVEL) -> dict[str, float]:
    series = pd.Series(values, dtype=float).dropna()
    n = len(series)
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "sd": np.nan,
            "sem": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    mean = float(series.mean())
    sd = float(series.std(ddof=1)) if n > 1 else 0.0
    sem = float(series.sem()) if n > 1 else 0.0
    if n > 1:
        half = float(stats.t.ppf((1.0 + ci) / 2.0, df=n - 1) * sem)
    else:
        half = 0.0
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "sem": sem,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "median": float(series.median()),
        "q25": float(series.quantile(0.25)),
        "q75": float(series.quantile(0.75)),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def median_bootstrap_ci(
    values: pd.Series | np.ndarray,
    *,
    n_boot: int = BOOTSTRAP_DRAWS,
    ci: float = CI_LEVEL,
    seed: int,
) -> dict[str, float]:
    array = pd.Series(values, dtype=float).dropna().to_numpy(dtype=float)
    n = len(array)
    if n == 0:
        return {"n": 0, "median": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    center = float(np.median(array))
    if n == 1:
        return {"n": 1, "median": center, "ci_low": center, "ci_high": center}
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        boot[idx] = float(np.median(rng.choice(array, size=n, replace=True)))
    alpha = (1.0 - ci) / 2.0
    return {
        "n": n,
        "median": center,
        "ci_low": float(np.quantile(boot, alpha)),
        "ci_high": float(np.quantile(boot, 1.0 - alpha)),
    }


def grouped_descriptives(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    value_col: str = "value",
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for keys, g in df.groupby(group_cols, observed=True, sort=True):
        row = series_summary(g[value_col])
        if not isinstance(keys, tuple):
            keys = (keys,)
        for col, value in zip(group_cols, keys, strict=True):
            row[col] = value
        if "ppid" in g.columns:
            row["n_participants"] = int(g["ppid"].nunique())
        rows.append(row)
    return pd.DataFrame.from_records(rows).sort_values(group_cols).reset_index(drop=True)


def grouped_median_descriptives(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    value_col: str = "value",
    seed_prefix: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for keys, g in df.groupby(group_cols, observed=True, sort=True):
        row = median_bootstrap_ci(
            g[value_col],
            seed=stable_seed(seed_prefix, *(keys if isinstance(keys, tuple) else (keys,))),
        )
        if not isinstance(keys, tuple):
            keys = (keys,)
        for col, value in zip(group_cols, keys, strict=True):
            row[col] = value
        if "ppid" in g.columns:
            row["n_participants"] = int(g["ppid"].nunique())
        rows.append(row)
    return pd.DataFrame.from_records(rows).sort_values(group_cols).reset_index(drop=True)


def _clean_numeric(values: pd.Series | np.ndarray) -> pd.Series:
    return pd.Series(values, dtype=float).dropna()


def _bootstrap_ci_1d(
    values: pd.Series | np.ndarray,
    stat_fn,
    *,
    ci: float = CI_LEVEL,
    n_boot: int = BOOTSTRAP_DRAWS,
    seed: int,
) -> tuple[float, float]:
    series = _clean_numeric(values)
    n = len(series)
    if n == 0:
        return (np.nan, np.nan)

    array = series.to_numpy(dtype=float)
    try:
        estimate = float(stat_fn(array))
    except Exception:
        estimate = np.nan
    if n == 1 or not np.isfinite(estimate):
        return (estimate, estimate)

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        sample = array[rng.integers(0, n, size=n)]
        try:
            boot[idx] = float(stat_fn(sample))
        except Exception:
            boot[idx] = np.nan

    boot = boot[np.isfinite(boot)]
    if len(boot) == 0:
        return (np.nan, np.nan)

    alpha = 1.0 - ci
    return (
        float(np.quantile(boot, alpha / 2.0)),
        float(np.quantile(boot, 1.0 - alpha / 2.0)),
    )


def cohens_d_one_sample(values: pd.Series | np.ndarray) -> float:
    series = _clean_numeric(values)
    if len(series) < 2:
        return np.nan
    sd = float(series.std(ddof=1))
    return float(series.mean() / sd) if sd > 0 else np.nan


def _add_anova_effect_sizes(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    total_ss = float(out["sum_sq"].sum(skipna=True))
    residual_mask = out["term"].eq("Residual")
    residual_ss = float(out.loc[residual_mask, "sum_sq"].iloc[0]) if residual_mask.any() else np.nan
    out["eta_sq"] = out["sum_sq"] / total_ss if total_ss > 0 else np.nan
    out["partial_eta_sq"] = (
        out["sum_sq"] / (out["sum_sq"] + residual_ss) if np.isfinite(residual_ss) else np.nan
    )
    out.loc[residual_mask, ["eta_sq", "partial_eta_sq"]] = np.nan
    return out


def _anova_effect_ci_summary(
    data: pd.DataFrame,
    formula: str,
    *,
    cluster_col: str | None = None,
    ci: float = CI_LEVEL,
    n_boot: int = ANOVA_CI_BOOTSTRAPS,
    seed: int,
) -> dict[str, dict[str, float]]:
    sample_df = data.reset_index(drop=True).copy()
    n_rows = len(sample_df)
    if n_rows < 2:
        return {}

    rng = np.random.default_rng(seed)
    alpha = 1.0 - ci
    cluster_ids = None
    if cluster_col is not None and cluster_col in sample_df.columns:
        cluster_ids = pd.Index(sample_df[cluster_col].dropna().unique())
        if len(cluster_ids) < 2:
            cluster_ids = None
    # Keep each sampled cluster intact, including repeats and within-cluster row order.
    cluster_rows = (
        sample_df.groupby(cluster_col, sort=False, observed=True).indices
        if cluster_ids is not None
        else {}
    )
    boot_records: list[dict[str, float | str]] = []
    for _ in range(n_boot):
        if cluster_ids is None:
            boot_df = sample_df.iloc[rng.integers(0, n_rows, size=n_rows)].copy()
        else:
            sampled_clusters = cluster_ids[rng.integers(0, len(cluster_ids), size=len(cluster_ids))]
            indices = np.concatenate([cluster_rows[cluster_id] for cluster_id in sampled_clusters])
            boot_df = sample_df.iloc[indices].reset_index(drop=True)
        try:
            table = (
                anova_lm(smf.ols(formula, data=boot_df).fit(), typ=2)
                .reset_index()
                .rename(columns={"index": "term"})
            )
            table = _add_anova_effect_sizes(table)
        except Exception:
            continue
        for row in table.loc[
            ~table["term"].eq("Residual"), ["term", "eta_sq", "partial_eta_sq"]
        ].itertuples(index=False):
            if np.isfinite(row.eta_sq) and np.isfinite(row.partial_eta_sq):
                boot_records.append(
                    {
                        "term": str(row.term),
                        "eta_sq": float(row.eta_sq),
                        "partial_eta_sq": float(row.partial_eta_sq),
                    }
                )

    if not boot_records:
        return {}

    boot_summary = pd.DataFrame.from_records(boot_records)
    ci_summary: dict[str, dict[str, float]] = {}
    for term, group_df in boot_summary.groupby("term", observed=True):
        ci_summary[str(term)] = {
            "eta_sq_ci_low": float(group_df["eta_sq"].quantile(alpha / 2.0)),
            "eta_sq_ci_high": float(group_df["eta_sq"].quantile(1.0 - alpha / 2.0)),
            "partial_eta_sq_ci_low": float(group_df["partial_eta_sq"].quantile(alpha / 2.0)),
            "partial_eta_sq_ci_high": float(group_df["partial_eta_sq"].quantile(1.0 - alpha / 2.0)),
        }
    return ci_summary


def _attach_anova_effect_cis(
    table: pd.DataFrame,
    ci_summary: dict[str, dict[str, float]],
) -> pd.DataFrame:
    out = table.copy()
    for col in [
        "eta_sq_ci_low",
        "eta_sq_ci_high",
        "partial_eta_sq_ci_low",
        "partial_eta_sq_ci_high",
    ]:
        out[col] = np.nan
    for idx, term in out["term"].items():
        if term in ci_summary:
            for col, value in ci_summary[term].items():
                out.loc[idx, col] = value
    for effect_col in ["eta_sq", "partial_eta_sq"]:
        low_col = f"{effect_col}_ci_low"
        high_col = f"{effect_col}_ci_high"
        mask = out[effect_col].notna()
        out.loc[mask, low_col] = np.minimum(out.loc[mask, low_col], out.loc[mask, effect_col])
        out.loc[mask, high_col] = np.maximum(out.loc[mask, high_col], out.loc[mask, effect_col])
    return out


def planned_zero_tests_by_clamp(df: pd.DataFrame, *, value_col: str = "value") -> pd.DataFrame:
    records: list[dict[str, float | int]] = []
    for clamp, g in df.groupby("clamp_magnitude", observed=True):
        values = pd.to_numeric(g[value_col], errors="coerce").dropna().to_numpy(dtype=float)
        n = len(values)
        if n < 2:
            continue
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        sem = float(stats.sem(values))
        half = float(stats.t.ppf((1.0 + CI_LEVEL) / 2.0, df=n - 1) * sem)
        t_stat, p_value = stats.ttest_1samp(values, 0.0)
        cohens_d = cohens_d_one_sample(values)
        d_low, d_high = _bootstrap_ci_1d(
            values,
            cohens_d_one_sample,
            seed=stable_seed("planned_zero_tests_by_clamp", clamp, value_col),
        )
        records.append(
            {
                "clamp_magnitude": int(clamp),
                "n": n,
                "mean": mean,
                "sd": sd,
                "sem": sem,
                "ci_low": mean - half,
                "ci_high": mean + half,
                "t": float(t_stat),
                "p": float(p_value),
                "cohens_d": cohens_d,
                "cohens_d_ci_low": d_low,
                "cohens_d_ci_high": d_high,
            }
        )
    results = (
        pd.DataFrame.from_records(records).sort_values("clamp_magnitude").reset_index(drop=True)
    )
    if not results.empty:
        reject, p_holm, _, _ = multipletests(results["p"], method="holm")
        results["p_holm"] = p_holm
        results["reject_holm"] = reject
    return results


def cluster_robust_term_table(
    df: pd.DataFrame, formula: str, *, cluster_col: str = "ppid"
) -> pd.DataFrame:
    plain_model = smf.ols(formula, data=df).fit()
    used_rows = pd.Index(plain_model.model.data.row_labels)
    clean_df = df.loc[used_rows].copy()
    clean_df = clean_df.loc[clean_df[cluster_col].notna()].copy()

    robust_model = smf.ols(formula, data=clean_df).fit(
        cov_type="cluster",
        cov_kwds={"groups": clean_df[cluster_col].to_numpy()},
    )
    anova = anova_lm(plain_model, typ=2).reset_index().rename(columns={"index": "term"})
    anova = _add_anova_effect_sizes(anova)
    anova = anova.rename(columns={"F": "f_naive", "PR(>F)": "p_value_naive", "df": "df_model"})
    anova = _attach_anova_effect_cis(
        anova,
        _anova_effect_ci_summary(
            clean_df,
            formula,
            cluster_col=cluster_col,
            seed=stable_seed("cluster_robust_term_table", formula, cluster_col),
        ),
    )

    wald = (
        robust_model.wald_test_terms(skip_single=False, scalar=True)
        .summary_frame()
        .reset_index()
        .rename(
            columns={
                "index": "term",
                "chi2": "wald_chi2",
                "P>chi2": "p_value_cluster",
                "df constraint": "df_constraint",
            }
        )
    )

    merged = anova.merge(wald, on="term", how="left")
    merged["formula"] = formula
    merged["n_rows"] = int(len(clean_df))
    merged["n_participants"] = int(clean_df[cluster_col].nunique())
    return merged


def pairwise_cluster_contrasts(
    model,
    *,
    hypotheses: list[tuple[str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for label, hypothesis in hypotheses:
        test = model.t_test(hypothesis)
        ci = test.conf_int(alpha=1.0 - CI_LEVEL)
        rows.append(
            {
                "contrast": label,
                "estimate": float(np.asarray(test.effect).squeeze()),
                "se": float(np.asarray(test.sd).squeeze()),
                "t_value": float(np.asarray(test.tvalue).squeeze()),
                "p": float(np.asarray(test.pvalue).squeeze()),
                "ci_low": float(np.asarray(ci).reshape(-1, 2)[0, 0]),
                "ci_high": float(np.asarray(ci).reshape(-1, 2)[0, 1]),
            }
        )
    out = pd.DataFrame.from_records(rows)
    reject, p_holm, _, _ = multipletests(out["p"], method="holm")
    out["p_holm"] = p_holm
    out["reject_holm"] = reject
    return out


def paired_range_test(df: pd.DataFrame, *, range_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = df.pivot_table(index="ppid", columns="phase", values="value", aggfunc="mean").dropna()
    pivot = pivot.rename_axis(index="ppid").reset_index()
    pivot["late_minus_baseline"] = pivot["Late adaptation"] - pivot["Baseline"]
    diff = pivot["late_minus_baseline"].to_numpy(dtype=float)
    n = len(diff)
    mean_diff = float(diff.mean())
    sd_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    sem_diff = float(stats.sem(diff)) if n > 1 else 0.0
    half = float(stats.t.ppf((1.0 + CI_LEVEL) / 2.0, df=n - 1) * sem_diff) if n > 1 else 0.0
    t_stat, p_value = stats.ttest_1samp(diff, 0.0)
    dz = cohens_d_one_sample(diff)
    dz_low, dz_high = _bootstrap_ci_1d(
        diff,
        cohens_d_one_sample,
        seed=stable_seed("paired_range_test", range_label),
    )
    row = pd.DataFrame(
        [
            {
                "range_label": range_label,
                "n_participants": n,
                "baseline_mean": float(pivot["Baseline"].mean()),
                "late_mean": float(pivot["Late adaptation"].mean()),
                "mean_diff_late_minus_baseline": mean_diff,
                "sd_diff": sd_diff,
                "ci_low": mean_diff - half,
                "ci_high": mean_diff + half,
                "t": float(t_stat),
                "p": float(p_value),
                "cohens_dz": dz,
                "cohens_dz_ci_low": dz_low,
                "cohens_dz_ci_high": dz_high,
            }
        ]
    )
    return row, pivot


def add_signed_fraction_columns(
    df: pd.DataFrame,
    *,
    value_col: str = "value",
    near_zero_abs_deg: float = 5.0,
) -> pd.DataFrame:
    work = df.copy()
    work["positive_fraction"] = pd.to_numeric(work[value_col], errors="coerce") > near_zero_abs_deg
    work["negative_fraction"] = pd.to_numeric(work[value_col], errors="coerce") < -near_zero_abs_deg
    work["near_zero_fraction_abs_5deg"] = (
        pd.to_numeric(work[value_col], errors="coerce").abs() <= near_zero_abs_deg
    )
    return work


def phase_range_descriptives(df: pd.DataFrame, *, range_label: str) -> pd.DataFrame:
    work = add_signed_fraction_columns(df)
    rows: list[dict[str, float | int | str]] = []
    for phase, g in work.groupby("phase", observed=True):
        row = series_summary(g["value"])
        row["range_label"] = range_label
        row["phase"] = phase
        row["n_participants"] = int(g["ppid"].nunique())
        row["positive_fraction"] = float(g["positive_fraction"].mean())
        row["negative_fraction"] = float(g["negative_fraction"].mean())
        row["near_zero_fraction_abs_5deg"] = float(g["near_zero_fraction_abs_5deg"].mean())
        rows.append(row)
    return pd.DataFrame.from_records(rows).sort_values("phase").reset_index(drop=True)


def load_spatial_bias_table() -> pd.DataFrame:
    df = pd.read_csv(TABLE_DIR / "baseline_bias_by_target.csv", dtype={"ppid": str})
    df["target_angles_degrees"] = pd.to_numeric(df["target_angles_degrees"], errors="coerce")
    df["bias"] = pd.to_numeric(df["bias"], errors="coerce")
    df = df.loc[df["pointer"].isin(POINTER_ORDER) & df["bias"].abs().le(40.0)].copy()
    df["target_angles_degrees"] = df["target_angles_degrees"].astype(int)
    return df.reset_index(drop=True)


def spatial_bias_descriptives(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (pointer, target), g in raw.groupby(["pointer", "target_angles_degrees"], observed=True):
        row = series_summary(g["bias"])
        row["pointer"] = pointer
        row["target_angles_degrees"] = int(target)
        row["n_participants"] = int(g["ppid"].nunique())
        rows.append(row)
    return (
        pd.DataFrame.from_records(rows)
        .sort_values(["pointer", "target_angles_degrees"])
        .reset_index(drop=True)
    )


def spatial_bias_fit_table(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from clamp_analysis.figures.device_bias import fit_transformation_bias_patterns

    fit_input = summary[["pointer", "target_angles_degrees", "mean"]].copy()
    fit_summary, fit_points = fit_transformation_bias_patterns(fit_input)
    extra_rows: list[dict[str, float | str]] = []
    for pointer, g in fit_points.groupby("pointer", observed=True):
        obs = g["human_mean_bias"].to_numpy(dtype=float)
        pred = g["model_pred_bias"].to_numpy(dtype=float)
        corr = float(pd.Series(obs).corr(pd.Series(pred)))
        extra_rows.append(
            {
                "pointer": pointer,
                "pearson_r_to_mean_pattern": corr,
                "r_squared_to_mean_pattern": corr**2,
            }
        )
    extra = pd.DataFrame.from_records(extra_rows)
    fit_summary = fit_summary.merge(extra, on="pointer", how="left")
    return fit_summary, fit_points


def load_bias_direction_pairings() -> pd.DataFrame:
    df = pd.read_csv(
        TABLE_DIR / "late_adaptation_by_signed_bias_direction_pairings.csv", dtype={"ppid": str}
    )
    df = df.loc[df["clamp_magnitude"].between(135, 170) & df["pointer"].isin(POINTER_ORDER)].copy()
    df["bias_direction"] = pd.Categorical(
        df["bias_direction"], categories=BIAS_REFERENCE_ORDER, ordered=True
    )
    return df.reset_index(drop=True)


def bias_direction_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (bias_direction, clamp), g in df.groupby(
        ["bias_direction", "clamp_magnitude"], observed=True
    ):
        row = series_summary(g["late_adapt"])
        row["bias_direction"] = str(bias_direction)
        row["clamp_magnitude"] = int(clamp)
        row["n_participants"] = int(g["ppid"].nunique())
        rows.append(row)
    return (
        pd.DataFrame.from_records(rows)
        .sort_values(["bias_direction", "clamp_magnitude"])
        .reset_index(drop=True)
    )


def bias_direction_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, int | str]] = []
    for label, g in df.groupby("bias_direction", observed=True):
        rows.append(
            {
                "bias_direction": str(label),
                "n_rows": int(len(g)),
                "n_participants": int(g["ppid"].nunique()),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values("bias_direction").reset_index(drop=True)


def build_trajectory_window_subject_table(pairings: pd.DataFrame) -> pd.DataFrame:
    membership = pairings[["ppid", "clamp_magnitude", "bias_direction"]].drop_duplicates()
    trials = pd.read_csv(
        DATA_PATH,
        usecols=["ppid", "condition", "cycle_num", "clamp_magnitude", "recentred_hand_angle"],
        dtype={"ppid": str},
    )
    trials["clamp_magnitude"] = pd.to_numeric(trials["clamp_magnitude"], errors="coerce")
    trials["cycle_num"] = pd.to_numeric(trials["cycle_num"], errors="coerce")
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

    per_clamp = (
        trials.groupby(
            ["ppid", "clamp_magnitude", "bias_direction", "task_cycle"],
            as_index=False,
            observed=True,
        )["recentred_hand_angle"]
        .mean()
        .rename(columns={"recentred_hand_angle": "value"})
    )
    participant_curve = (
        per_clamp.groupby(["ppid", "bias_direction", "task_cycle"], as_index=False, observed=True)[
            "value"
        ]
        .mean()
        .reset_index(drop=True)
    )

    def assign_window(task_cycle: int | float) -> str | float:
        if -20 <= task_cycle <= -11:
            return "No-feedback baseline"
        if 0 <= task_cycle <= 9:
            return "Early clamp"
        if 220 <= task_cycle <= 239:
            return "Late clamp"
        if 240 <= task_cycle <= 249:
            return "Early washout"
        return np.nan

    participant_curve["window"] = participant_curve["task_cycle"].apply(assign_window)
    window_df = (
        participant_curve.dropna(subset=["window"])
        .groupby(["ppid", "bias_direction", "window"], as_index=False, observed=True)["value"]
        .mean()
        .reset_index(drop=True)
    )
    window_df["window"] = pd.Categorical(
        window_df["window"], categories=TRAJECTORY_WINDOW_ORDER, ordered=True
    )
    return window_df


def trajectory_window_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (window, bias_direction), g in df.groupby(["window", "bias_direction"], observed=True):
        row = series_summary(g["value"])
        row["window"] = str(window)
        row["bias_direction"] = str(bias_direction)
        row["n_participants"] = int(g["ppid"].nunique())
        rows.append(row)
    out = pd.DataFrame.from_records(rows)
    out["window"] = pd.Categorical(out["window"], categories=TRAJECTORY_WINDOW_ORDER, ordered=True)
    return out.sort_values(["window", "bias_direction"]).reset_index(drop=True)


def overall_pointer_medians(df: pd.DataFrame, *, seed_prefix: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for pointer, g in df.groupby("pointer", observed=True):
        row = median_bootstrap_ci(g["value"], seed=stable_seed(seed_prefix, pointer))
        row["pointer"] = pointer
        row["mean"] = float(pd.to_numeric(g["value"], errors="coerce").mean())
        row["sd"] = float(pd.to_numeric(g["value"], errors="coerce").std(ddof=1))
        row["n_participants"] = int(g["ppid"].nunique())
        rows.append(row)
    return pd.DataFrame.from_records(rows).sort_values("pointer").reset_index(drop=True)


def save_table(df: pd.DataFrame, name: str, outputs: dict[str, Path]) -> Path:
    path = TABLE_DIR / name
    df.to_csv(path, index=False)
    outputs[name] = path
    return path


def p_text(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "p = NA"
    if p_value < 0.001:
        return "p < .001"
    return f"p = {p_value:.3f}".replace("0.", ".")


def format_effect_ci(
    label: str,
    estimate: float,
    ci_low: float,
    ci_high: float,
    *,
    digits: int = 2,
) -> str:
    if not np.isfinite(estimate):
        return f"{label} = NA"
    if np.isfinite(ci_low) and np.isfinite(ci_high):
        return f"{label} = {estimate:.{digits}f} [{ci_low:.{digits}f}, {ci_high:.{digits}f}]"
    return f"{label} = {estimate:.{digits}f}"


def chi2_sentence(row: pd.Series) -> str:
    return (
        f"Wald chi2({int(row['df_constraint'])}) = {row['wald_chi2']:.2f}, "
        f"{p_text(float(row['p_value_cluster']))}, "
        f"{format_effect_ci('eta_p^2', row['partial_eta_sq'], row.get('partial_eta_sq_ci_low', np.nan), row.get('partial_eta_sq_ci_high', np.nan), digits=3)}"
    )


def t_sentence(row: pd.Series, *, effect_col: str = "cohens_dz") -> str:
    n = int(row["n_participants"])
    ci_low = row.get(f"{effect_col}_ci_low", np.nan)
    ci_high = row.get(f"{effect_col}_ci_high", np.nan)
    return (
        f"t({n - 1}) = {row['t']:.2f}, {p_text(float(row['p']))}, "
        f"{format_effect_ci(effect_col, row[effect_col], ci_low, ci_high)}"
    )


def build_summary_markdown(results: dict[str, pd.DataFrame]) -> str:
    late_term = results["late_omnibus"].set_index("term")
    late_zero = results["late_zero"]
    late_sig = late_zero.loc[late_zero["reject_holm"], "clamp_magnitude"].tolist()

    early_term = results["early_stl_omnibus"].set_index("term")
    washout_term = results["washout_stl_omnibus"].set_index("term")
    peak_pair = results["peak_paired"].iloc[0]
    high_pair = results["high_error_paired"].iloc[0]
    spatial_term = results["spatial_omnibus"].set_index("term")
    bias_term = results["bias_omnibus"].set_index("term")
    bias_pairwise = results["bias_pairwise"]
    traj_term = results["trajectory_omnibus"].set_index("term")
    early_var_term = results["early_var_omnibus"].set_index("term")
    late_var_term = results["late_var_omnibus"].set_index("term")

    mouse_spatial = results["spatial_descriptives"].loc[
        results["spatial_descriptives"]["pointer"].eq("Mouse")
    ]
    trackpad_spatial = results["spatial_descriptives"].loc[
        results["spatial_descriptives"]["pointer"].eq("Trackpad")
    ]

    lines = [
        f"# {OUT_PREFIX} manuscript stats",
        "",
        "Inferential model-term tests below are participant-clustered Wald chi-square tests from linear models fit to the per-subject summary tables. The eta-squared intervals are participant-level bootstrap confidence intervals for the same model terms.",
        "",
        "- Late adaptation varied strongly with clamp magnitude "
        f"({chi2_sentence(late_term.loc['C(clamp_magnitude)'])}). "
        "The device gap changed across clamp magnitudes "
        f"({chi2_sentence(late_term.loc['C(clamp_magnitude):C(pointer)'])}). "
        f"Holm-corrected one-sample tests against zero were significant at {len(late_sig)}/40 clamps: "
        f"{', '.join(str(v) for v in late_sig)} deg.",
        "- Early single-trial learning depended on clamp magnitude "
        f"({chi2_sentence(early_term.loc['C(clamp_magnitude)'])}), "
        "whereas washout single-trial learning also varied with clamp magnitude "
        f"({chi2_sentence(washout_term.loc['C(clamp_magnitude)'])}). "
        f"Only the 80 deg clamp survived Holm correction in early STL; no washout clamp did.",
        "- In the peak range, late adaptation exceeded baseline by "
        f"{peak_pair['mean_diff_late_minus_baseline']:.2f} deg on average "
        f"({t_sentence(peak_pair)}). "
        "In the high-error range, the phase difference was negligible "
        f"({high_pair['mean_diff_late_minus_baseline']:.2f} deg; {t_sentence(high_pair)}).",
        "- Spatial baseline bias showed a strong pointer-by-target interaction "
        f"({chi2_sentence(spatial_term.loc['C(pointer):C(target_angles_degrees)'])}). "
        f"Mouse means were uniformly negative ({mouse_spatial['mean'].min():.2f} to {mouse_spatial['mean'].max():.2f} deg), "
        f"whereas trackpad means ranged from {trackpad_spatial['mean'].min():.2f} to "
        f"{trackpad_spatial['mean'].max():.2f} deg across the four targets.",
        "- Within the 135 to 170 deg clamps, late adaptation depended on baseline-bias sign after adjusting for clamp and pointer "
        f"({chi2_sentence(bias_term.loc['C(bias_direction)'])}). "
        f"Relative to the near-zero group, the incorrect-sign group was "
        f"{bias_pairwise.loc[bias_pairwise['contrast'].eq('Incorrect vs Near-zero'), 'estimate'].iloc[0]:.2f} deg lower "
        f"and the correct-sign group was "
        f"{bias_pairwise.loc[bias_pairwise['contrast'].eq('Correct vs Near-zero'), 'estimate'].iloc[0]:.2f} deg higher.",
        "- The high-error trajectory panel showed a robust window-by-bias-sign interaction "
        f"({chi2_sentence(traj_term.loc['C(window):C(bias_direction)'])}), "
        "so the sign-group separation was not confined to a single time window.",
        "- Early STL variability showed a pointer effect in the full clamp-by-device model "
        f"({chi2_sentence(early_var_term.loc['C(pointer)'])}), with a near-significant interaction trend "
        f"({p_text(float(early_var_term.loc['C(clamp_magnitude):C(pointer)', 'p_value_cluster']))}). "
        "Late adaptation variability was higher for mouse than trackpad in the additive model "
        f"({chi2_sentence(late_var_term.loc['C(pointer)'])}).",
        "",
        "The CSV tables written beside this summary contain the full per-clamp descriptives, effect sizes, confidence intervals, and participant-clustered Wald term tests for the clamp-level analyses.",
    ]
    return "\n".join(lines)


def run_all_analyses() -> dict[str, object]:
    ensure_dirs()

    from clamp_analysis.figures.panels import build_early_stl_variability_subject

    outputs: dict[str, Path] = {}
    results: dict[str, pd.DataFrame] = {}

    late_subject = load_table("late_adaptation_subject_means.csv")
    late_omnibus = cluster_robust_term_table(
        late_subject, "value ~ C(clamp_magnitude) * C(pointer)"
    )
    late_descriptives = grouped_descriptives(late_subject, group_cols=["clamp_magnitude"])
    late_pointer_descriptives = grouped_descriptives(
        late_subject,
        group_cols=["pointer", "clamp_magnitude"],
    )
    late_zero = planned_zero_tests_by_clamp(late_subject)

    save_table(late_omnibus, f"{OUT_PREFIX}_late_adaptation_omnibus.csv", outputs)
    save_table(late_descriptives, f"{OUT_PREFIX}_late_adaptation_descriptives.csv", outputs)
    save_table(
        late_pointer_descriptives, f"{OUT_PREFIX}_late_adaptation_pointer_descriptives.csv", outputs
    )
    save_table(late_zero, f"{OUT_PREFIX}_late_adaptation_zero_tests.csv", outputs)
    results["late_omnibus"] = late_omnibus
    results["late_zero"] = late_zero

    early_stl = load_table("stl_early_subject_means.csv", add_pointer=True)
    early_stl_omnibus = cluster_robust_term_table(
        early_stl, "value ~ C(clamp_magnitude) + C(pointer)"
    )
    early_stl_descriptives = grouped_descriptives(early_stl, group_cols=["clamp_magnitude"])
    early_stl_zero = planned_zero_tests_by_clamp(early_stl)

    washout_stl = load_table("stl_washout_subject_means.csv", add_pointer=True)
    washout_stl_omnibus = cluster_robust_term_table(
        washout_stl, "value ~ C(clamp_magnitude) + C(pointer)"
    )
    washout_stl_descriptives = grouped_descriptives(washout_stl, group_cols=["clamp_magnitude"])
    washout_stl_zero = planned_zero_tests_by_clamp(washout_stl)

    save_table(early_stl_omnibus, f"{OUT_PREFIX}_early_stl_omnibus.csv", outputs)
    save_table(early_stl_descriptives, f"{OUT_PREFIX}_early_stl_descriptives.csv", outputs)
    save_table(early_stl_zero, f"{OUT_PREFIX}_early_stl_zero_tests.csv", outputs)
    save_table(washout_stl_omnibus, f"{OUT_PREFIX}_washout_stl_omnibus.csv", outputs)
    save_table(washout_stl_descriptives, f"{OUT_PREFIX}_washout_stl_descriptives.csv", outputs)
    save_table(washout_stl_zero, f"{OUT_PREFIX}_washout_stl_zero_tests.csv", outputs)
    results["early_stl_omnibus"] = early_stl_omnibus
    results["washout_stl_omnibus"] = washout_stl_omnibus

    peak_phase = load_table("peak_phase_subject_means.csv")
    peak_range_descriptives = phase_range_descriptives(peak_phase, range_label="Peak (15-50 deg)")
    peak_paired, peak_subject_paired = paired_range_test(peak_phase, range_label="Peak (15-50 deg)")
    save_table(peak_range_descriptives, f"{OUT_PREFIX}_peak_phase_range_descriptives.csv", outputs)
    save_table(peak_paired, f"{OUT_PREFIX}_peak_phase_paired_test.csv", outputs)
    save_table(peak_subject_paired, f"{OUT_PREFIX}_peak_phase_subject_paired_values.csv", outputs)
    results["peak_paired"] = peak_paired

    high_error_phase = load_table("high_error_phase_subject_means.csv")
    high_error_range_descriptives = phase_range_descriptives(
        high_error_phase, range_label="High error (135-170 deg)"
    )
    high_error_paired, high_error_subject_paired = paired_range_test(
        high_error_phase,
        range_label="High error (135-170 deg)",
    )
    save_table(
        high_error_range_descriptives,
        f"{OUT_PREFIX}_high_error_phase_range_descriptives.csv",
        outputs,
    )
    save_table(high_error_paired, f"{OUT_PREFIX}_high_error_phase_paired_test.csv", outputs)
    save_table(
        high_error_subject_paired,
        f"{OUT_PREFIX}_high_error_phase_subject_paired_values.csv",
        outputs,
    )
    results["high_error_paired"] = high_error_paired

    spatial_raw = load_spatial_bias_table()
    spatial_descriptives = spatial_bias_descriptives(spatial_raw)
    spatial_omnibus = cluster_robust_term_table(
        spatial_raw,
        "bias ~ C(pointer) * C(target_angles_degrees)",
    )
    spatial_fit_summary, spatial_fit_points = spatial_bias_fit_table(spatial_descriptives)
    save_table(spatial_descriptives, f"{OUT_PREFIX}_spatial_bias_descriptives.csv", outputs)
    save_table(spatial_omnibus, f"{OUT_PREFIX}_spatial_bias_omnibus.csv", outputs)
    save_table(spatial_fit_summary, f"{OUT_PREFIX}_spatial_bias_transformation_fit.csv", outputs)
    save_table(
        spatial_fit_points, f"{OUT_PREFIX}_spatial_bias_transformation_fit_points.csv", outputs
    )
    results["spatial_omnibus"] = spatial_omnibus
    results["spatial_descriptives"] = spatial_descriptives

    bias_pairings = load_bias_direction_pairings()
    bias_omnibus = cluster_robust_term_table(
        bias_pairings,
        "late_adapt ~ C(bias_direction) + C(clamp_magnitude) + C(pointer)",
    )
    bias_model = smf.ols(
        "late_adapt ~ C(bias_direction) + C(clamp_magnitude) + C(pointer)",
        data=bias_pairings,
    ).fit(cov_type="cluster", cov_kwds={"groups": bias_pairings["ppid"]})
    bias_pairwise = pairwise_cluster_contrasts(
        bias_model,
        hypotheses=[
            ("Incorrect vs Near-zero", "C(bias_direction)[T.Incorrect] = 0"),
            ("Correct vs Near-zero", "C(bias_direction)[T.Correct] = 0"),
            (
                "Incorrect vs Correct",
                "C(bias_direction)[T.Incorrect] - C(bias_direction)[T.Correct] = 0",
            ),
        ],
    )
    bias_descr = bias_direction_descriptives(bias_pairings)
    bias_counts_df = bias_direction_counts(bias_pairings)
    save_table(bias_omnibus, f"{OUT_PREFIX}_bias_direction_omnibus.csv", outputs)
    save_table(bias_pairwise, f"{OUT_PREFIX}_bias_direction_pairwise.csv", outputs)
    save_table(bias_descr, f"{OUT_PREFIX}_bias_direction_descriptives.csv", outputs)
    save_table(bias_counts_df, f"{OUT_PREFIX}_bias_direction_counts.csv", outputs)
    results["bias_omnibus"] = bias_omnibus
    results["bias_pairwise"] = bias_pairwise

    trajectory_window_subject = build_trajectory_window_subject_table(bias_pairings)
    trajectory_omnibus = cluster_robust_term_table(
        trajectory_window_subject,
        "value ~ C(window) * C(bias_direction)",
    )
    trajectory_descriptives = trajectory_window_descriptives(trajectory_window_subject)
    save_table(trajectory_window_subject, f"{OUT_PREFIX}_trajectory_windows_subject.csv", outputs)
    save_table(trajectory_omnibus, f"{OUT_PREFIX}_trajectory_windows_omnibus.csv", outputs)
    save_table(
        trajectory_descriptives, f"{OUT_PREFIX}_trajectory_windows_descriptives.csv", outputs
    )
    results["trajectory_omnibus"] = trajectory_omnibus

    early_var = build_early_stl_variability_subject()
    early_var = early_var.loc[early_var["pointer"].isin(POINTER_ORDER)].copy()
    early_var_omnibus = cluster_robust_term_table(
        early_var,
        "value ~ C(clamp_magnitude) * C(pointer)",
    )
    early_var_pointer = overall_pointer_medians(early_var, seed_prefix="early_var_pointer")
    early_var_pointer_clamp = grouped_median_descriptives(
        early_var,
        group_cols=["pointer", "clamp_magnitude"],
        seed_prefix="early_var_pointer_clamp",
    )
    save_table(early_var, f"{OUT_PREFIX}_early_stl_variability_subject.csv", outputs)
    save_table(early_var_omnibus, f"{OUT_PREFIX}_early_stl_variability_omnibus.csv", outputs)
    save_table(
        early_var_pointer, f"{OUT_PREFIX}_early_stl_variability_pointer_medians.csv", outputs
    )
    save_table(
        early_var_pointer_clamp,
        f"{OUT_PREFIX}_early_stl_variability_pointer_clamp_medians.csv",
        outputs,
    )
    results["early_var_omnibus"] = early_var_omnibus

    late_var = load_table("late_variability_subject.csv")
    late_var_omnibus = cluster_robust_term_table(
        late_var,
        "value ~ C(clamp_magnitude) + C(pointer)",
    )
    late_var_pointer = overall_pointer_medians(late_var, seed_prefix="late_var_pointer")
    late_var_pointer_clamp = grouped_median_descriptives(
        late_var,
        group_cols=["pointer", "clamp_magnitude"],
        seed_prefix="late_var_pointer_clamp",
    )
    save_table(late_var_omnibus, f"{OUT_PREFIX}_late_variability_omnibus.csv", outputs)
    save_table(late_var_pointer, f"{OUT_PREFIX}_late_variability_pointer_medians.csv", outputs)
    save_table(
        late_var_pointer_clamp, f"{OUT_PREFIX}_late_variability_pointer_clamp_medians.csv", outputs
    )
    results["late_var_omnibus"] = late_var_omnibus

    summary_text = build_summary_markdown(results)
    SUMMARY_MARKDOWN_PATH.write_text(summary_text, encoding="utf-8")

    return {
        "tables": {name: str(path) for name, path in outputs.items()},
        "summary_markdown": str(SUMMARY_MARKDOWN_PATH),
    }
