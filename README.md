# Modelling implicit sensorimotor adaptation

**Testing how people adapt their movements to visual errors, using behavioural data and computational models.**

In a clamped-feedback experiment, the cursor follows a prescribed direction regardless of the participant's movement direction. This project investigates whether implicit adaptation is better explained by perceptual error in two dimensions than by angular error alone.

Research code and data from **Max Townsend's PhD in computational cognitive science**, supporting *Implicit sensorimotor adaptation operates over multidimensional perceptual error* (**manuscript in preparation**).

[Request the manuscript](mailto:max.o.b.townsend@gmail.com?subject=Request%3A%20Implicit%20sensorimotor%20adaptation%20operates%20over%20multidimensional%20perceptual%20error) · [Browse the figures](results/figures/) · [Contact](mailto:max.o.b.townsend@gmail.com)

## What is included

- **Behavioural analysis at scale:** 486 participants, 524,649 trials and 40 clamp magnitudes; 484 participants have device metadata for mouse/trackpad comparisons.
- **Competing models:** scalar Bayesian cue combination (BCC), causal inference and Vector-BCC, with shared learning dynamics and device-specific uncertainty parameters.
- **Model evaluation:** fitting, model and parameter recovery, movement-trajectory analysis, statistical summaries and six reproducible figures.

**Stack:** Python · NumPy · pandas · SciPy · statsmodels · Matplotlib · joblib.

## Explore the code

| Start here | What to look for |
|---|---|
| [Behavioural pipeline](clamp_analysis/behavior/pipeline.py) | Trial data to participant summaries and analysis tables |
| [Learning models](clamp_analysis/models/state_space.py) | Three accounts of how perceptual error drives adaptation |
| [Joint-device models](clamp_analysis/models/joint_device.py) | Shared dynamics and mouse/trackpad uncertainty |
| [Recovery workflow](clamp_analysis/workflows/recovery.py) | Simulating and refitting data to assess model distinguishability |

See the [code reading guide](docs/reading_guide.md) and [figure-to-code map](docs/chapter4_map.md) for more detail. Outputs retain the thesis Chapter 4 numbering.

## Reproduce the results

With **Python 3.13**, clone the repository and run from its root in a virtual environment:

```powershell
git clone https://github.com/Max-Townsend/clamps.git
cd clamps
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m clamp_analysis reproduce
```

On macOS/Linux, create the environment with `python3.13 -m venv .venv` and activate it with `source .venv/bin/activate`.

This regenerates summaries, statistics and figures using the saved fits. Required data archives unpack automatically; Git LFS is not needed. Outputs are written to `results/`, with figures in PNG, SVG and PDF. Full refitting is a separate, expensive workflow: see [reproduction instructions](docs/REPRODUCING.md).

## Paper and data access

Townsend, M., Warburton, M., Campagnoli, C., Mon-Williams, M., Mushtaq, F., & Morehead, J. R. **Implicit sensorimotor adaptation operates over multidimensional perceptual error.** Manuscript in preparation; available on request.

The full raw movement-tracker file (approximately **8.85 GB**) is available on request. Standard reproduction uses bundled data and tracker-derived caches. For the manuscript, full data or questions, contact [max.o.b.townsend@gmail.com](mailto:max.o.b.townsend@gmail.com).
