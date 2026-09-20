from __future__ import annotations

import csv
import math
import statistics
import urllib.request
from collections import defaultdict
from pathlib import Path

from clamp_analysis import paths

OUTDIR = paths.LITERATURE_DIR

SOURCEDIR = OUTDIR / "source_downloads"

SOURCE_FILES = {
    "Fig1B.csv": "https://ndownloader.figshare.com/files/44164199",
    "Exp2.csv": "https://ndownloader.figshare.com/files/44164190",
    "README.txt": "https://ndownloader.figshare.com/files/44164202",
}

FIGSHARE_DOI = "10.6084/m9.figshare.24503926.v2"

FIGSHARE_URL = "https://doi.org/10.6084/m9.figshare.24503926.v2"

ELIFE_ARTICLE_DOI = "10.7554/eLife.94608.3"

ELIFE_ARTICLE_URL = "https://elifesciences.org/articles/94608"

LEGACY_SERIES = {
    "1": {
        "series_order": 1,
        "series_id": "kim_2018_exp1",
        "series_label": "Kim, 2018 exp1",
        "source_paper": "Kim et al., 2018",
        "source_year": 2018,
        "source_doi": "10.1038/s42003-018-0021-y",
        "source_experiment": "Experiment 1",
    },
    "2": {
        "series_order": 2,
        "series_id": "kim_2018_exp2",
        "series_label": "Kim, 2018 exp2",
        "source_paper": "Kim et al., 2018",
        "source_year": 2018,
        "source_doi": "10.1038/s42003-018-0021-y",
        "source_experiment": "Experiment 2",
    },
    "3": {
        "series_order": 3,
        "series_id": "morehead_2017",
        "series_label": "Morehead, 2017",
        "source_paper": "Morehead et al., 2017",
        "source_year": 2017,
        "source_doi": "10.1162/jocn_a_01108",
        "source_experiment": "Task-irrelevant clamped feedback",
    },
}

EXP2_SERIES = {
    "series_order": 4,
    "series_id": "zhang_2024_exp2",
    "series_label": "Zhang et al., 2024 exp2",
    "source_paper": "Zhang et al., 2024",
    "source_year": 2024,
    "source_doi": ELIFE_ARTICLE_DOI,
    "source_experiment": "Experiment 2",
}

PHASE_LABELS = {
    "0": "baseline",
    "1": "adaptation",
    "2": "washout_no_feedback",
    "3": "washout_no_feedback",
}


def download(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        destination.write_bytes(response.read())


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sem(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def normalize_exp2_trials(exp2_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    normalized = []
    for row in exp2_rows:
        trial_type_code = row["trialType"]
        normalized.append(
            {
                "series_id": EXP2_SERIES["series_id"],
                "series_label": EXP2_SERIES["series_label"],
                "source_paper": EXP2_SERIES["source_paper"],
                "source_year": EXP2_SERIES["source_year"],
                "source_doi": EXP2_SERIES["source_doi"],
                "source_dataset": "Exp2.csv",
                "source_dataset_doi": FIGSHARE_DOI,
                "source_dataset_url": FIGSHARE_URL,
                "source_experiment": EXP2_SERIES["source_experiment"],
                "group_id": int(row["group"]),
                "perturbation_size_deg": float(row["perturbSize"]),
                "subject_id": int(row["sub"]),
                "trial_type_code": int(trial_type_code),
                "phase": PHASE_LABELS.get(trial_type_code, "unknown"),
                "cycle_num": int(row["cycleNum"]),
                "hand_angle_deg": float(row["handangle"]),
            }
        )
    return normalized


def summarize_exp2_for_panel_b(
    exp2_trials: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    subject_cycle_window = defaultdict(list)
    for row in exp2_trials:
        cycle_num = int(row["cycle_num"])
        if row["phase"] != "adaptation":
            continue
        if not (100 <= cycle_num <= 110):
            continue
        key = (float(row["perturbation_size_deg"]), int(row["subject_id"]))
        subject_cycle_window[key].append(float(row["hand_angle_deg"]))

    subject_means = []
    grouped_subject_means = defaultdict(list)
    for (perturbation_size, subject_id), values in sorted(subject_cycle_window.items()):
        subject_mean = mean(values)
        grouped_subject_means[perturbation_size].append(subject_mean)
        subject_means.append(
            {
                "series_id": EXP2_SERIES["series_id"],
                "series_label": EXP2_SERIES["series_label"],
                "source_paper": EXP2_SERIES["source_paper"],
                "source_year": EXP2_SERIES["source_year"],
                "source_doi": EXP2_SERIES["source_doi"],
                "source_experiment": EXP2_SERIES["source_experiment"],
                "subject_id": subject_id,
                "perturbation_size_deg": perturbation_size,
                "extent_window": "cycles 100-110 inclusive",
                "subject_mean_extent_deg": subject_mean,
            }
        )

    summary_rows = []
    for perturbation_size in sorted(grouped_subject_means):
        values = grouped_subject_means[perturbation_size]
        summary_rows.append(
            {
                "series_order": EXP2_SERIES["series_order"],
                "series_id": EXP2_SERIES["series_id"],
                "series_label": EXP2_SERIES["series_label"],
                "source_paper": EXP2_SERIES["source_paper"],
                "source_year": EXP2_SERIES["source_year"],
                "source_doi": EXP2_SERIES["source_doi"],
                "source_dataset": "Exp2.csv",
                "source_dataset_doi": FIGSHARE_DOI,
                "source_dataset_url": FIGSHARE_URL,
                "source_experiment": EXP2_SERIES["source_experiment"],
                "data_origin": "derived_from_raw_trial_data",
                "perturbation_size_deg": perturbation_size,
                "adaptation_extent_deg": mean(values),
                "sem_deg": sem(values),
                "n_subjects": len(values),
                "extent_window": "cycles 100-110 inclusive",
                "panel_target": "eLife 94608 Figure 1B scatter points",
                "notes": "Computed from subject means over adaptation cycles 100-110.",
            }
        )
    return subject_means, summary_rows


def normalize_legacy_fig1b(fig1b_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    normalized = []
    for row in fig1b_rows:
        series = LEGACY_SERIES[row["study"]]
        normalized.append(
            {
                "series_order": series["series_order"],
                "series_id": series["series_id"],
                "series_label": series["series_label"],
                "source_paper": series["source_paper"],
                "source_year": series["source_year"],
                "source_doi": series["source_doi"],
                "source_dataset": "Fig1B.csv",
                "source_dataset_doi": FIGSHARE_DOI,
                "source_dataset_url": FIGSHARE_URL,
                "source_experiment": series["source_experiment"],
                "data_origin": "published_summary_point_from_elife_figshare",
                "perturbation_size_deg": float(row["perturbSize"]),
                "adaptation_extent_deg": float(row["extent"]),
                "sem_deg": "",
                "n_subjects": "",
                "extent_window": "as provided in Fig1B.csv",
                "panel_target": "eLife 94608 Figure 1B scatter points",
                "notes": "Legacy summary point included by Zhang et al. in Fig1B.csv.",
            }
        )
    return normalized


def write_provenance_note() -> None:
    text = f"""Prepared for reproducing the scatter portion of Figure 1B from {ELIFE_ARTICLE_URL}

Primary article:
- Zhang Z, Wang H, Zhang T, Nie Z, Wei K (2024). Perceptual error based on Bayesian cue combination drives implicit motor adaptation.
- eLife DOI: {ELIFE_ARTICLE_DOI}
- Article URL: {ELIFE_ARTICLE_URL}

Downloaded source files:
- Fig1B.csv from figshare dataset {FIGSHARE_DOI}
- Exp2.csv from figshare dataset {FIGSHARE_DOI}
- README.txt from figshare dataset {FIGSHARE_DOI}

How the outputs were built:
- legacy_panelB_points_standardized.csv re-labels the three legacy series that Zhang et al. packaged in Fig1B.csv:
  - Kim, 2018 exp1
  - Kim, 2018 exp2
  - Morehead, 2017
- exp2_trial_level_standardized.csv is a normalized trial-level export of the paper's Experiment 2 raw data.
- exp2_subject_mean_extent_cycles100_110.csv computes each participant's mean hand angle over adaptation cycles 100-110 inclusive.
- panelB_unified_summary_with_exp2.csv combines the three legacy series with a new Zhang et al., 2024 Experiment 2 summary derived from Exp2.csv using the paper's Figure 3B definition of adaptation extent.

Important note:
- The figshare README says washout uses trialType == 3, but the downloaded Exp2.csv encodes washout rows with trialType == 2. The standardized trial-level file preserves the raw code and maps both 2 and 3 to washout_no_feedback.
"""
    (OUTDIR / "README_provenance.txt").write_text(text, encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    SOURCEDIR.mkdir(parents=True, exist_ok=True)

    for filename, url in SOURCE_FILES.items():
        download(url, SOURCEDIR / filename)

    fig1b_rows = read_csv_rows(SOURCEDIR / "Fig1B.csv")
    exp2_rows = read_csv_rows(SOURCEDIR / "Exp2.csv")

    legacy_rows = normalize_legacy_fig1b(fig1b_rows)
    exp2_trials = normalize_exp2_trials(exp2_rows)
    exp2_subject_means, exp2_summary_rows = summarize_exp2_for_panel_b(exp2_trials)
    unified_summary_rows = sorted(
        legacy_rows + exp2_summary_rows,
        key=lambda row: (int(row["series_order"]), float(row["perturbation_size_deg"])),
    )

    write_csv(
        OUTDIR / "legacy_panelB_points_standardized.csv",
        legacy_rows,
        [
            "series_order",
            "series_id",
            "series_label",
            "source_paper",
            "source_year",
            "source_doi",
            "source_dataset",
            "source_dataset_doi",
            "source_dataset_url",
            "source_experiment",
            "data_origin",
            "perturbation_size_deg",
            "adaptation_extent_deg",
            "sem_deg",
            "n_subjects",
            "extent_window",
            "panel_target",
            "notes",
        ],
    )
    write_csv(
        OUTDIR / "exp2_trial_level_standardized.csv",
        exp2_trials,
        [
            "series_id",
            "series_label",
            "source_paper",
            "source_year",
            "source_doi",
            "source_dataset",
            "source_dataset_doi",
            "source_dataset_url",
            "source_experiment",
            "group_id",
            "perturbation_size_deg",
            "subject_id",
            "trial_type_code",
            "phase",
            "cycle_num",
            "hand_angle_deg",
        ],
    )
    write_csv(
        OUTDIR / "exp2_subject_mean_extent_cycles100_110.csv",
        exp2_subject_means,
        [
            "series_id",
            "series_label",
            "source_paper",
            "source_year",
            "source_doi",
            "source_experiment",
            "subject_id",
            "perturbation_size_deg",
            "extent_window",
            "subject_mean_extent_deg",
        ],
    )
    write_csv(
        OUTDIR / "panelB_unified_summary_with_exp2.csv",
        unified_summary_rows,
        [
            "series_order",
            "series_id",
            "series_label",
            "source_paper",
            "source_year",
            "source_doi",
            "source_dataset",
            "source_dataset_doi",
            "source_dataset_url",
            "source_experiment",
            "data_origin",
            "perturbation_size_deg",
            "adaptation_extent_deg",
            "sem_deg",
            "n_subjects",
            "extent_window",
            "panel_target",
            "notes",
        ],
    )
    write_provenance_note()
