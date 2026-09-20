"""All input and output locations, independent of the current working directory.

Set CLAMP_PROJECT_ROOT to use a data checkout elsewhere. Set CLAMP_RESULTS_DIR
to regenerate outputs in a separate directory without overwriting saved results.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(
    os.environ.get("CLAMP_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).resolve()

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"

DATA_PATH = DATA_DIR / "processed" / "formattedData.csv"

DETAILS_PATH = RAW_DIR / "detailsV2.csv"

SETTINGS_PATH = RAW_DIR / "settingsV2.csv"

LITERATURE_DIR = DATA_DIR / "literature"

RESULTS_DIR = Path(os.environ.get("CLAMP_RESULTS_DIR", PROJECT_ROOT / "results")).resolve()

TABLE_DIR = RESULTS_DIR / "tables"

FIGURE_DIR = RESULTS_DIR / "figures"

FITS_DIR = RESULTS_DIR / "fits"

RECOVERY_DIR = RESULTS_DIR / "recovery"

KINEMATICS_DIR = RESULTS_DIR / "kinematics"

FIT_COLLECTION = FITS_DIR / "mean_timeseries_shared_point_joint_device_motorvar"

RECOVERY_COLLECTION = RECOVERY_DIR / "mean_timeseries_shared_point_recovery_joint_device_motorvar"


def ensure_output_dirs():
    for directory in (TABLE_DIR, FIGURE_DIR, FITS_DIR, RECOVERY_DIR, KINEMATICS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
