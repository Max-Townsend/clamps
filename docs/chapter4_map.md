# Chapter 4 output map

The source is `PhDThesisFinal_CORRECTED.pdf`, Chapter 4 (printed pages 113–144; PDF pages 124–155).

| Thesis item | Output | Implementation and inputs |
|---|---|---|
| Figure 4.1 | `results/figures/figure_4_1.*` | `figures/chapter.py`; literature points prepared by `literature/zhang.py` and `literature/morehead.py` |
| Figure 4.2 A–D | `results/figures/figure_4_2.*` | `behavior/pipeline.py`, `figures/panels.py`; participant means, same-target STL, learning curves and saved model predictions |
| Figure 4.2 E–F | Same figure | `kinematics/ballistic.py`, `updates.py`, `calibration.py`, `radial.py`; first-bout radial/tangential changes and calibrated BCC predictions |
| Figure 4.3 A–B | `results/figures/figure_4_3.*` | Participant baseline-versus-late distributions for peak and large-error clamps |
| Figure 4.3 C–D | Same figure | `behavior/bias.py`; signed baseline-bias strata and high-error trajectories |
| Figure 4.4 A–C | `results/figures/figure_4_4.*` | Device-specific adaptation, early/late within-person reach variability, Vector-BCC predictions |
| Figure 4.4 D | Same figure | No-feedback spatial bias and the transformation-bias fit in `models/transformation_bias.py` |
| Figure S4.1 | `results/figures/figure_S4_1.*` | `statistics/anova.py`, `figures/anova.py`; group × device participant means and t confidence intervals |
| Figure S4.2 | `results/figures/figure_S4_2.*` | `figures/supplement.py`; BIC, model recovery confusion matrix and Vector-BCC parameter recovery |
| Model parameter table | `results/tables/model_parameters.csv` | Saved joint-device parameter summaries, with fixed parameters marked |
| Behavioral statistical results | `results/tables/paper_requested_seven_row_*.csv`, `paper_draft_results_*.csv` | `statistics/clamp_tests.py`, `statistics/chapter_tests.py` |
| Statistical prose summaries | `results/tables/*summary.md`, `*manuscript_ready.md` | Same statistical workflows |
| Model fitting | `results/fits/mean_timeseries_shared_point_joint_device_motorvar/` | `workflows/fit.py`; model equations in `models/state_space.py`, shared/device-specific parameters in `models/joint_device.py` |
| Model/parameter recovery | `results/recovery/mean_timeseries_shared_point_recovery_joint_device_motorvar/` | `workflows/recovery.py`; 20 synthetic datasets per generating model, each fit by all three candidate models |

Each numbered figure is exported separately in PNG, SVG and PDF formats.
