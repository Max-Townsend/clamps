Ryan Morehead multi-clamp local dataset provenance

Source directory: data/literature/morehead_raw
MATLAB script reproduced: MultiSizeClamp_allsizes_plotting.m

Reproduction details:
- signflip fixed to -1, matching the script's intended collapsing of CW/CCW onto positive clamp magnitudes.
- Target-specific sequences aligned exactly as in the MATLAB plotting script.
- Baseline correction uses aligned trials 31-40.
- Late learning summary uses mean aligned trials 301-500 ('last 200 cycles' in the MATLAB comments).
- Subject 1 from the first 1.75/2.5/3.5/7.5 block is excluded entirely because the raw block contains a duplicated 2.5 deg assignment and no valid 3.5 deg assignment for that participant.

Outputs:
- ryan_morehead_multiclamp_trial_level_standardized.csv
- ryan_morehead_multiclamp_subject_mean_extent_late200.csv
- ryan_morehead_multiclamp_panel_summary.csv
- panelB_unified_summary_with_exp2_and_morehead_multiclamp.csv

Notes: Recomputed from local MATLAB files using the exact late-learning logic in MultiSizeClamp_allsizes_plotting.m. Includes local multi-size clamp data and additional small-sample conditions not present in the published Morehead 2017 summary. For the 1.75/2.5/3.5/7.5 block, subject 1 is excluded entirely to avoid relying on the manual duplicated-condition repair used in the original MATLAB plotting script.