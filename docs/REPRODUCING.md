# Reproducing the clamp analyses

[Back to the project overview](../README.md)

This guide reproduces the Chapter 4 analyses, figures and statistical outputs using the retained thesis fits. Run all commands from the repository root.

## Data availability

For the full dataset, including the raw movement-tracker data, contact **Max Townsend** at [max.o.b.townsend@gmail.com](mailto:max.o.b.townsend@gmail.com).

The raw tracker file (`data/raw/trackers.csv`, approximately 8.85 GB) is excluded from Git and remains available locally. Standard figure and statistical reproduction uses the formatted dataset and retained tracker-derived caches. Re-extracting movement bouts with `trackers --rebuild-cache` requires the full tracker file; place the supplied file at `data/raw/trackers.csv` before running that command.

### Bundled data

The five other large input files are distributed as lossless gzip archives in `data/archives/`. The CSV archives are about 27 MB and 31 MB. The MATLAB archives use numbered parts no larger than **40 MiB**, below [GitHub's 100 MiB per-file limit](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github). Git LFS is not required for these bundles.

Analysis commands automatically restore any missing inputs they need. To restore all five files explicitly, using only Python's standard library:

```powershell
py -3.13 -m clamp_analysis restore-data
```

Existing local files are preserved. The reconstructed CSV/MAT files and raw trackers are ignored by Git; the compact archives are included instead. See [the archive guide](data_archives.md) for details.

## Start here

From this folder, using Python 3.13:

```powershell
py -3.13 -m pip install -r requirements.txt
py -3.13 -m clamp_analysis reproduce
```

The reproduction command recomputes participant summaries and statistics and draws every numbered figure. It uses the retained thesis fits and recovery simulations; it does not silently launch hours of optimization. Figures are saved as PNG, SVG and PDF in `results/figures/`.

To redraw the figures alone:

```powershell
py -3.13 -m clamp_analysis figures
```

## Where things live

| Folder | Contents |
|---|---|
| `clamp_analysis/behavior/` | Raw-trial preparation, analysis windows, participant summaries and baseline biases |
| `clamp_analysis/statistics/` | ANOVAs, clustered models, planned comparisons and bootstrap intervals |
| `clamp_analysis/models/` | Scalar BCC, causal inference, Vector-BCC, joint-device parameters and fitting machinery |
| `clamp_analysis/kinematics/` | First movement-bout extraction, quality checks, radial/tangential updates and model calibration |
| `clamp_analysis/figures/` | Reusable plotting functions and numbered thesis figure layouts |
| `clamp_analysis/workflows/` | Joint-device model fitting and simulation-based recovery workflows |
| `clamp_analysis/literature/` | Preparation of the literature data for Figure 4.1 |
| `data/raw/` | Original trials, trackers, participant list, device details and settings |
| `data/processed/` | The complete formatted thesis dataset |
| `data/archives/` | Compressed bundles for restoring the five large CSV/MAT inputs |
| `data/literature/` | Included literature source data, provenance and derived summaries |
| `results/` | Regenerated figures/tables and retained fits, recovery results and tracker caches |
| `docs/` | Figure-to-code map, code reading guide and data restoration instructions |

All filesystem locations are defined in `clamp_analysis/paths.py`. For an installed package, `CLAMP_PROJECT_ROOT` can point to another checkout. `CLAMP_RESULTS_DIR` can direct regenerated outputs elsewhere; that directory must also contain the saved fit/recovery inputs when making figures.

## Rebuild the underlying analyses

```powershell
# Rebuild the formatted dataset from raw trials and the original participant list.
py -3.13 -m clamp_analysis preprocess

# Rebuild radial model calibration using the retained first-bout tracker cache.
py -3.13 -m clamp_analysis trackers

# Re-extract the first bouts from the complete 8.85 GB tracker file as well.
py -3.13 -m clamp_analysis trackers --rebuild-cache

# Recreate the literature summaries from the included original CSV/MAT files.
py -3.13 -m clamp_analysis literature

# Refit the three joint-device models. Outputs go to an explicit new location.
py -3.13 -m clamp_analysis fit --output results/refits --cores 4

# Rerun the full 60-dataset, 180-fit model/parameter recovery analysis.
py -3.13 -m clamp_analysis recover --output results/recovery_rerun --cores 4
```

Fitting and recovery defaults use the publication settings. Lower restart/evaluation counts are available for pipeline checks, but their output should not replace the thesis fits. New fits are intentionally written separately; the default figure workflow continues to use the named thesis collection in `paths.py` until you explicitly select a replacement.

## Scientific conventions and provenance

The main dataset contains **486 participants, 524,649 trials and 40 clamp magnitudes**. Two participants lack device metadata; device-specific analyses use 484 participants. Main behavioral analyses use task-aligned hand angles without baseline correction.

Read [the chapter map](chapter4_map.md) for exact figure dependencies, [the code reading guide](reading_guide.md) for the analysis windows, scientific conventions and where to make changes.
