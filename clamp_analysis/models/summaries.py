from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from clamp_analysis.models.fitting import prepare_late_adaptation_data

MEAN_TIMESERIES_SEM_FLOOR_DEG = 1.0

DEFAULT_REPRESENTATIVE_CLAMPS = (5, 35, 65, 95, 125, 155)

POST_NO_FEEDBACK_CONDITIONS = ("FeedbackBaseline", "Clamp", "FeedbackWashout")


def prepare_mean_timeseries_data(
    data_path: str | Path = "formattedData.csv",
    subset_groups: list[int] | tuple[int, ...] | None = None,
    participants_per_group: int | None = None,
    seed: int = 42,
):
    return prepare_late_adaptation_data(
        data_path=data_path,
        subset_groups=subset_groups,
        participants_per_group=participants_per_group,
        seed=seed,
    )


def load_pointer_device_map(details_path: str | Path = "detailsV2.csv"):
    details = pd.read_csv(details_path)
    details["ppid"] = (
        details["ppid_session_dataname"]
        .astype(str)
        .str.replace(
            r"_s\d+_participant_details$",
            "",
            regex=True,
        )
    )
    details = details.rename(columns={"pointer": "pointer_device"})
    details = details.loc[:, ["ppid", "pointer_device"]].drop_duplicates()
    return details.reset_index(drop=True)


def attach_pointer_device(raw_df, details_path: str | Path = "detailsV2.csv"):
    working = raw_df.copy()
    working["ppid"] = working["ppid"].astype(str)
    device_map = load_pointer_device_map(details_path=details_path)
    merged = working.merge(device_map, on="ppid", how="left")
    return merged


def summarize_mean_timeseries(
    raw_df,
    value_col="recentred_hand_angle",
    sem_floor_deg=MEAN_TIMESERIES_SEM_FLOOR_DEG,
):
    if value_col != "recentred_hand_angle":
        raise ValueError("Mean-timeseries fits should use recentred_hand_angle.")

    working = raw_df.copy()
    working["ppid"] = working["ppid"].astype(str)
    working["clamp_magnitude"] = pd.to_numeric(working["clamp_magnitude"], errors="coerce")
    working["trial_num"] = pd.to_numeric(working["trial_num"], errors="coerce")
    working["trial_num_in_block"] = pd.to_numeric(working["trial_num_in_block"], errors="coerce")
    working["recentred_hand_angle"] = pd.to_numeric(working[value_col], errors="coerce")
    working["cycle_wrt_clamp"] = pd.to_numeric(working["cycle_wrt_clamp"], errors="coerce")

    sort_cols = ["ppid", "clamp_magnitude", "trial_num"]
    working = working.sort_values(sort_cols).reset_index(drop=True)

    subset = working.loc[working["condition"].isin(POST_NO_FEEDBACK_CONDITIONS)].copy()
    subset["sequence_trial"] = pd.to_numeric(subset["cycle_wrt_clamp"], errors="coerce") + 11.0

    records = []
    summary_group_cols = [
        "clamp_magnitude",
        "condition",
        "clamped",
        "cycle_wrt_clamp",
        "sequence_trial",
    ]
    for key, gdf in subset.groupby(summary_group_cols, dropna=False, observed=False):
        clamp_mag, condition, clamped, cycle_wrt_clamp, sequence_trial = key
        values = pd.to_numeric(gdf[value_col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        sem = float(sd / np.sqrt(len(values))) if len(values) > 1 else float(sem_floor_deg)
        obs_var = max(sem**2, float(sem_floor_deg) ** 2)
        raw_in_block = (
            pd.to_numeric(gdf["trial_num_in_block"], errors="coerce").dropna().to_numpy(dtype=float)
        )
        rotation_model = -float(clamp_mag) if bool(clamped) else 0.0
        if condition == "FeedbackBaseline":
            trial_num_in_block = int(float(cycle_wrt_clamp) + 11.0)
        elif condition == "Clamp":
            trial_num_in_block = int(float(cycle_wrt_clamp) + 1.0)
        elif condition == "FeedbackWashout":
            trial_num_in_block = int(float(cycle_wrt_clamp) - 239.0)
        else:
            trial_num_in_block = int(float(sequence_trial))
        records.append(
            {
                "ppid": "group_mean",
                "group": 0,
                "clamp_magnitude": float(clamp_mag),
                "target_angles_degrees": 0.0,
                "target_clamp_sign": 1.0,
                "clamp_sign": 1.0,
                "rotation_model": float(rotation_model),
                "condition": str(condition),
                "clamped": bool(clamped),
                "sequence_trial": int(sequence_trial),
                "trial_num_in_block": int(trial_num_in_block),
                "raw_trial_num_in_block": int(np.nanmedian(raw_in_block))
                if len(raw_in_block)
                else np.nan,
                "cycle_wrt_clamp": float(cycle_wrt_clamp),
                "late_mean": float(np.mean(values)),
                "mean_hand_angle": float(np.mean(values)),
                "mean_sd": sd,
                "mean_sem": sem,
                "obs_var": float(obs_var),
                "n_obs": int(len(values)),
            }
        )

    summary = pd.DataFrame.from_records(records)
    if summary.empty:
        raise ValueError("No mean-timeseries summary rows were created.")

    return summary.sort_values(["clamp_magnitude", "sequence_trial"]).reset_index(drop=True)


def fit_ci_from_summary(summary_df, value_col="late_mean", ci_level=0.95):
    out = summary_df.copy()
    mean_vals = pd.to_numeric(out[value_col], errors="coerce").to_numpy(dtype=float)
    sem = np.sqrt(np.clip(pd.to_numeric(out["obs_var"], errors="coerce"), 1e-9, None))
    n_obs = pd.to_numeric(out["n_obs"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    dof = np.maximum(n_obs - 1.0, 1.0)
    t_crit = stats.t.ppf((1.0 + ci_level) / 2.0, df=dof)
    out["ci_low"] = mean_vals - t_crit * sem
    out["ci_high"] = mean_vals + t_crit * sem
    return out


def _resolve_representative_clamps(preferred_clamps, available_clamps):
    remaining = sorted({float(clamp) for clamp in available_clamps})
    selected = []
    for preferred in preferred_clamps:
        if not remaining:
            break
        preferred = float(preferred)
        actual = min(remaining, key=lambda clamp: (abs(clamp - preferred), clamp))
        remaining.remove(actual)
        selected.append(int(actual))
    return selected


def plot_mean_timeseries_fit(
    human_summary_df,
    model_prediction_df,
    model_name,
    representative_clamps=DEFAULT_REPRESENTATIVE_CLAMPS,
    ci_level=0.95,
):
    human = fit_ci_from_summary(human_summary_df, value_col="late_mean", ci_level=ci_level)
    model = model_prediction_df.copy()
    model["predicted_late_mean"] = pd.to_numeric(model["predicted_late_mean"], errors="coerce")

    available_clamps = sorted(
        set(
            pd.to_numeric(human["clamp_magnitude"], errors="coerce").dropna().astype(float).tolist()
        )
        & set(
            pd.to_numeric(model["clamp_magnitude"], errors="coerce").dropna().astype(float).tolist()
        )
    )
    rep_clamps = _resolve_representative_clamps(representative_clamps, available_clamps)
    if not rep_clamps:
        raise ValueError("No overlapping clamp magnitudes were available to plot.")

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, clamp_mag in zip(axes, rep_clamps):
        hdf = human.loc[human["clamp_magnitude"].eq(float(clamp_mag))].sort_values("sequence_trial")
        mdf = model.loc[model["clamp_magnitude"].eq(float(clamp_mag))].sort_values("sequence_trial")

        x_h = pd.to_numeric(hdf["cycle_wrt_clamp"], errors="coerce").to_numpy(dtype=float)
        y_h = pd.to_numeric(hdf["late_mean"], errors="coerce").to_numpy(dtype=float)
        y_lo = pd.to_numeric(hdf["ci_low"], errors="coerce").to_numpy(dtype=float)
        y_hi = pd.to_numeric(hdf["ci_high"], errors="coerce").to_numpy(dtype=float)
        x_m = pd.to_numeric(mdf["cycle_wrt_clamp"], errors="coerce").to_numpy(dtype=float)
        y_m = pd.to_numeric(mdf["predicted_late_mean"], errors="coerce").to_numpy(dtype=float)

        ax.fill_between(x_h, y_lo, y_hi, color="#b7b7b7", alpha=0.28, linewidth=0)
        ax.plot(x_h, y_h, color="#222222", linewidth=2.0, label="Human mean")
        ax.plot(x_m, y_m, color="#0b5cad", linewidth=2.0, linestyle="--", label="Model")
        ax.axhline(0.0, color="0.35", linewidth=1.0, linestyle="--")
        ax.axvline(-0.5, color="#3d8b3d", linewidth=1.0, linestyle=":", alpha=0.9)
        ax.axvline(239.5, color="#8b3d3d", linewidth=1.0, linestyle=":", alpha=0.9)
        ax.set_title(f"{int(clamp_mag)} deg clamp")
        ax.grid(alpha=0.18)

    for ax in axes[len(rep_clamps) :]:
        ax.set_visible(False)

    axes[0].legend(loc="upper left", framealpha=0.92)
    fig.suptitle(f"{model_name}: participant-averaged mean time series", fontsize=16)
    fig.text(
        0.5,
        0.03,
        "Cycle relative to clamp onset (-10:-1 = FeedbackBaseline, 0:239 = Clamp, 240:249 = Washout)",
        ha="center",
    )
    fig.text(0.03, 0.5, "Recentred hand angle (deg)", va="center", rotation="vertical")
    fig.tight_layout(rect=[0.04, 0.06, 1.0, 0.96])
    return fig
