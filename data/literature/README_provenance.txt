Prepared for reproducing the scatter portion of Figure 1B from https://elifesciences.org/articles/94608

Primary article:
- Zhang Z, Wang H, Zhang T, Nie Z, Wei K (2024). Perceptual error based on Bayesian cue combination drives implicit motor adaptation.
- eLife DOI: 10.7554/eLife.94608.3
- Article URL: https://elifesciences.org/articles/94608

Downloaded source files:
- Fig1B.csv from figshare dataset 10.6084/m9.figshare.24503926.v2
- Exp2.csv from figshare dataset 10.6084/m9.figshare.24503926.v2
- README.txt from figshare dataset 10.6084/m9.figshare.24503926.v2

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
