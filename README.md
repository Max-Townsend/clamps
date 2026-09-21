# Implicit sensorimotor adaptation

Python code and data for studying how people adapt their movements to visual feedback. In these experiments, the cursor follows a fixed direction relative to the target, regardless of the participant's movement direction. The question is whether adaptation depends on perceptual error in two dimensions rather than angular error alone.

The dataset contains 486 participants, 524,649 trials and 40 clamp magnitudes. The analyses compare three models: scalar Bayesian cue combination (BCC), causal inference and Vector-BCC. They include mouse/trackpad comparisons, movement trajectories, and model and parameter recovery.

## Run the analyses

Use **Python 3.13**. In PowerShell:

```powershell
git clone https://github.com/Max-Townsend/clamps.git
cd clamps
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m clamp_analysis reproduce
```

On macOS/Linux, create the environment with `python3.13 -m venv .venv` and activate it with `source .venv/bin/activate`.

This uses the saved fits to regenerate summaries, statistics and figures in `results/`. Data archives unpack automatically; Git LFS is not needed. You can also [browse the existing figures](results/figures/) without running anything.

See the [reproduction guide](docs/REPRODUCING.md) for refitting and other commands. The full raw movement-tracker file (8.85 GB) is available on request; normal reproduction uses the included tracker-derived caches.

## Code

- [Behavioural analysis](clamp_analysis/behavior/pipeline.py)
- [Model equations](clamp_analysis/models/state_space.py) and [mouse/trackpad parameters](clamp_analysis/models/joint_device.py)
- [Model and parameter recovery](clamp_analysis/workflows/recovery.py)

The [reading guide](docs/reading_guide.md) explains the analysis choices. The [figure map](docs/chapter4_map.md) connects outputs to code; figure numbers follow Chapter 4 of my PhD thesis.

## Paper

Townsend, M., Warburton, M., Campagnoli, C., Mon-Williams, M., Mushtaq, F., & Morehead, J. R. *Implicit sensorimotor adaptation operates over multidimensional perceptual error.* In preparation. [Request the manuscript](mailto:max.o.b.townsend@gmail.com?subject=Request%3A%20Implicit%20sensorimotor%20adaptation%20operates%20over%20multidimensional%20perceptual%20error).

For the paper, full data or questions: Max Townsend · [max.o.b.townsend@gmail.com](mailto:max.o.b.townsend@gmail.com)
