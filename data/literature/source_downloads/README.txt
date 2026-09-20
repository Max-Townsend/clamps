The README.txt file was generated on 01-21-2024 by Zhaoran Zhang

General information:

1.	Title of dataset: PEA model data

2.	Author information: 

Names: Zhaoran Zhang, Huijun Wang, Tianyang Zhang, Zixuan Nie, and Kunlin Wei
Institution: Peking University
Email: wei.kunlin@pku.edu.cn zhangzhaoran10@gmail.com

3.	Date of data collection: 2021-2023

4.	Location of data collection: Peking University, Beijing, China

5.	Citation: Zhang, Z., Wang, H., Zhang, T., Nie, Z., & Wei, K. (2023). Perceptual error based on Bayesian cue combination drives implicit motor adaptation. bioRxiv, 2023-11. 
 https://doi.org/10.1101/2023.11.23.568442

6.	Information about funding sources that support the collection of the data:
STI2030-Major Projects (2021ZD0202600) and the National Natural Science Foundation of China (62061136001, 32071047, 31871102), awarded to KW, and the National Natural Science Foundation of China (32300868) awarded to ZZ.


Description of dataset:

1.	File list:

Exp1.csv	-Data of Experiment 1
Exp2.csv	-Data of Experiment 2
Exp3.csv	-Data of Experiment 3
Exp4data.csv	-Data of Experiment 4
Tsay_STL.csv	-Data from Tsay, Avraham, et al., 2021, Experiment 2
Fig1B.csv	-Data from Kim et al., 2018 and Morehead et al. 2017
Tsay_report.csv	-Data from Tsay et al., 2020

2.	Method and purpose of data collection: See Zhang Wang et al., 2024 for detailed information.

Data information:

1.	Exp1.csv

Number of variables: 4
Number of rows: 162
Variable list: 
sub: subject number
angle: perturbation angle (for sides==1) or perturbation size (for sides==2), unit in degree
sigma: estimated visual uncertainty, unit in degree
sides: data used to estimate sigma include both sides or only one side (1 = 1 side, Figure 2C; 2 = 2 sides, Figure 2B)

2.	Exp2.csv

Number of variables: 6
Number of rows: 10080
Variable list: 
group: group number
perturbSize: size of perturbation angle for each group, unit in degree
sub: subject number
trialType: different epoch in the experiment (0 = baseline; 1 = adaptation; 3 = washout without visual feedback)
cycleNum: cycle number, hand angle in each cycle is averaged from 4 single trials
handangle: hand angle for each trial, unit in degree

3.	Exp3.csv

Number of variables: 4
Number of rows: 132
Variable list: 
sub: subject number
perturbSize: perturbation angle for each group, unit in degree
trialNum: trial number in the proprioception test block following the reaching block, proprioception bias in each trial number is the averaged from 4 rounds of repetitions. 
propBias: proprioception bias, unit in degree

4.	Exp4data.csv

Number of variables: 5
Number of rows: 10260
Variable list: 
sub: subject number
day: day of test
perturb: perturbation angle, unit in degree
blur: if the cursor is blur or clear (1 = clear; 2 = blur)
stl: single-trial adaptation, unit in degree

5.	Tsay_STL.csv

Number of variables: 3
Number of rows: 216
Variable list: 
sub: subject number
perturb: perturbation angle, unit in degree
stl: single-trial adaptation, unit in degree

6.	Fig1B.csv

Number of variables: 3
Number of rows: 16
Variable list: 
study: index of study (1 = Kim, 2018 exp1; 2 = Kim, 2018 exp2; 3 = Morehead, 2017)
perturbSize: perturbation size, unit in degree
extent: adaptation extent, unit in degree

7.	Tsay_report.csv

Number of variables: 4
Number of rows: 5760
Variable list:
SN: subject number
cond: perturbation angle, unit in degree
hand_theta: hand angle, unit in degree
hand_report: reported hand angle, unit in degree
