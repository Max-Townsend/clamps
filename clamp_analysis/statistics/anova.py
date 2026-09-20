"""Group-by-device ANOVAs and participant-level descriptive statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm

from clamp_analysis import paths

TABLE_DIR = paths.RESULTS_DIR / "tables"

DETAILS_PATH = paths.DETAILS_PATH

CI_LEVEL = 0.95

DEVICE_ORDER = ["Mouse", "Trackpad"]

MEASURES = {
    "Azimuthal adaptation extent": {
        "filename": "late_adaptation_subject_means.csv",
        "add_pointer": False,
    },
    "Early STL": {
        "filename": "stl_early_subject_means.csv",
        "add_pointer": True,
    },
    "Washout STL": {
        "filename": "stl_washout_subject_means.csv",
        "add_pointer": True,
    },
}


def load_pointer_lookup() -> pd.DataFrame:
    """Return one Mouse/Trackpad label per participant ID."""
    details = pd.read_csv(DETAILS_PATH, dtype={"ppid_session_dataname": str})
    details["ppid"] = details["ppid_session_dataname"].str.replace(
        r"_s\d+_participant_details$", "", regex=True
    )
    lookup = details[["ppid", "pointer"]].drop_duplicates()

    # A participant must not map to conflicting device labels.
    labels_per_ppid = lookup.groupby("ppid")["pointer"].nunique(dropna=True)
    if (labels_per_ppid > 1).any():
        conflicting = labels_per_ppid[labels_per_ppid > 1].index.tolist()
        raise ValueError(f"Conflicting pointer labels for participant(s): {conflicting}")

    return lookup.drop_duplicates(subset="ppid", keep="first")


def load_measure(filename: str, *, add_pointer: bool) -> tuple[pd.DataFrame, int]:
    """Load a participant-by-clamp file and apply the analysis device filter."""
    data = pd.read_csv(TABLE_DIR / filename, dtype={"ppid": str})
    source_n = data["ppid"].nunique()

    if add_pointer:
        data = data.merge(load_pointer_lookup(), on="ppid", how="left")

    if "pointer" not in data.columns:
        raise KeyError(f"{filename} has no pointer column after loading")

    data = data.loc[data["pointer"].isin(DEVICE_ORDER)].copy()
    data["group"] = pd.to_numeric(data["group"], errors="raise").astype(int)
    data["clamp_magnitude"] = pd.to_numeric(data["clamp_magnitude"], errors="raise").astype(int)
    data["value"] = pd.to_numeric(data["value"], errors="raise")

    # The reported group-level ANOVA first averages four clamp-level values
    # within each participant.  Verify that those four inputs are present.
    rows_per_participant = data.groupby("ppid").size()
    if not rows_per_participant.eq(4).all():
        bad = rows_per_participant.loc[~rows_per_participant.eq(4)]
        raise ValueError(
            f"Expected four clamp rows per included participant in {filename}; "
            f"found:\n{bad.to_string()}"
        )

    return data.reset_index(drop=True), source_n


def participant_group_means(data: pd.DataFrame) -> pd.DataFrame:
    """Create the one-row-per-participant dataset entered into the ANOVA."""
    means = (
        data.groupby(
            ["ppid", "group", "pointer"],
            observed=True,
            as_index=False,
        )["value"]
        .mean()
        .rename(columns={"value": "value"})
    )

    if len(means) != means["ppid"].nunique():
        raise ValueError("A participant appears in more than one group/device cell")
    if len(means) != 484:
        raise ValueError(f"Expected 484 included participants, found {len(means)}")

    return means


def clamp_labels(data: pd.DataFrame) -> pd.Series:
    """Create a readable list of the four clamp magnitudes in each group."""
    unique_clamps = (
        data[["group", "clamp_magnitude"]]
        .drop_duplicates()
        .sort_values(["group", "clamp_magnitude"])
    )
    counts = unique_clamps.groupby("group")["clamp_magnitude"].size()
    if not counts.eq(4).all():
        raise ValueError(f"Expected four clamp magnitudes per group:\n{counts}")

    return unique_clamps.groupby("group")["clamp_magnitude"].apply(
        lambda values: ", ".join(str(value) for value in values)
    )


def cell_descriptives(
    participant_means: pd.DataFrame,
    *,
    measure: str,
    group_clamps: pd.Series,
) -> pd.DataFrame:
    """Calculate raw cell means and two-sided t-based confidence intervals."""
    result = (
        participant_means.groupby(["group", "pointer"], observed=True)["value"]
        .agg(n="count", mean="mean", sd="std")
        .reset_index()
    )

    result["sem"] = result["sd"] / np.sqrt(result["n"])
    alpha = 1.0 - CI_LEVEL
    result["t_critical"] = stats.t.ppf(
        1.0 - alpha / 2.0,
        df=result["n"] - 1,
    )
    half_width = result["t_critical"] * result["sem"]
    result["ci_low"] = result["mean"] - half_width
    result["ci_high"] = result["mean"] + half_width

    result.insert(0, "measure", measure)
    result.insert(
        2,
        "assigned_clamp_magnitudes",
        result["group"].map(group_clamps),
    )
    result["pointer"] = pd.Categorical(result["pointer"], categories=DEVICE_ORDER, ordered=True)
    return result.sort_values(["group", "pointer"]).reset_index(drop=True)


def rerun_anova(participant_means: pd.DataFrame, *, measure: str) -> pd.DataFrame:
    """Rerun the Type-II 10 x 2 OLS ANOVA used in the thesis."""
    formula = "value ~ C(group) * C(pointer)"
    model = smf.ols(formula, data=participant_means).fit()
    table = (
        anova_lm(model, typ=2)
        .reset_index()
        .rename(
            columns={
                "index": "term",
                "df": "df_num",
                "F": "f_value",
                "PR(>F)": "p_value",
            }
        )
    )
    residual_df = float(table.loc[table["term"].eq("Residual"), "df_num"].iloc[0])
    table["df_den"] = residual_df
    table.insert(0, "measure", measure)

    return table
