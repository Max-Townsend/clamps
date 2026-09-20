# Reading and modifying the code

Start with `clamp_analysis/__main__.py`: it connects the named commands to their workflows. Importing a module does not launch fitting or rebuild data.

## Behavioral data

`behavior/preprocess.py` converts raw trials into the formatted dataset. `CLAMP_GROUPS` specifies the experimental assignment. The preparation preserves participant selection, row IDs, target-centered angles and the original sign convention.

`behavior/pipeline.py:run` is the main behavioral workflow. It loads trials, computes same-target differences, aggregates the specified windows, builds baseline-bias tables, and writes the common inputs for statistics and figures. `behavior/summaries.py:AnalysisConfig` contains those windows. `behavior/bias.py` contains the separate spatial-bias and bias-direction calculations.

## Statistics

`statistics/clamp_tests.py:run_all_analyses` runs clamp-level comparisons, robust models, variability and bias tests. `statistics/chapter_tests.py:run_all_analyses` runs the chapter's group/device ANOVAs, phase comparisons, effect sizes and prose reports. `statistics/anova.py` provides the group/device summaries and ANOVAs used for Figure S4.1. Seeds and bootstrap counts are named constants in the statistical modules.

Clamp-level and chapter-level tables use the `paper_requested_seven_row_` and `paper_draft_results_` prefixes, respectively, under `results/tables/`.

## Models

`models/state_space.py` implements the three learning models. `models/joint_device.py` defines the shared dynamics and device-specific uncertainty parameters. `models/fitting.py` contains the likelihood and optimization machinery; `models/parameters.py` defines parameter specifications.

`workflows/fit.py` places the scientific settings at the top, followed by `run` and named helper functions. The workflow prepares device summaries, calibrates motor variance, fits each model and exports joint/device predictions. `workflows/recovery.py` follows the same structure: prepare fitted inputs and parameter ranges, generate synthetic datasets, fit candidates, then call `write_recovery_reports` for recovery and identifiability summaries.

The `fit` and `recover` commands accept explicit output directories. To adopt new fits for the standard figures, select the new collection in `paths.py`; the retained thesis fits remain the default reference.

## Movement trajectories

`kinematics/ballistic.py` streams raw tracker records and extracts the first movement bout. `kinematics/updates.py` summarizes radial and tangential changes; `calibration.py` and `radial.py` compare them with calibrated model predictions. The raw tracker file is large, so normal figure reproduction uses the bundled extraction cache. `trackers --rebuild-cache` rebuilds it from the raw file.

## Figures

`figures/chapter.py:build_chapter_figures` assembles Figures 4.1–4.4 from reusable functions in `figures/panels.py`. Axis variables describe their contents. `figures/anova.py` and `figures/supplement.py` make the supplementary figures. `figures/pipeline.py` exports each figure in all three formats and writes the model parameter table.

## Scientific conventions

Raw preparation uses the participant list in `data/raw/ppidsV2.csv` and requires at least 1,000 main trials per participant. Target-centered data angles wrap to (-180°, 180°]; model predictions wrap to [-180°, 180°).

Main outcomes use task-aligned hand angles without baseline correction. Analysis cycles are `cycle_num - 21`. Late adaptation uses clamp trials 881–960; early STL uses cycles 0–9; washout STL uses cycles 240–249. The distribution reference uses feedback-baseline trials 21–40.

STL is the difference from the previous presentation of the same target, assigned to the current presentation. Baseline spatial bias uses no-feedback reaches with a ±40° restriction, and bias-direction strata use thresholds of ±3°.

Tracker analyses sample five fractions of distance along each movement bout. Seven updates connect the first eight target exposures; missing bouts are not bridged. Motor variance is calibrated per device from no-feedback data, while learning dynamics are shared across devices.
