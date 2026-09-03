# Categorical CVAE conditional-probability validation

Report format: `categorical-conditional-probability-report-v1`<br>
Experiment protocol: `categorical-conditional-probability-v1`<br>
Protocol SHA-256: `d74e723836611baddf84bc22db769ab77e1028af23bcddec373cfc3bad5fb209`

**Run status: complete canonical grid.**

## The question in plain English

For a person with covariates X, how close is the model's predicted probability for each category to the probability that actually generated the simulation? The known DGP probabilities let us answer this directly, rather than relying only on whether a single realized outcome was right.

Across the 170 base fits shown here, the CVAE's median focal-class absolute probability error was 0.0171, compared with 0.0063 for independent softmax. The CVAE had lower focal MAE in 13/170 matched fits and lower equal-outcome total variation in 2/170. Its worst fit-level focal MAE was 0.1050 in `core_n8000_q050_k5` (seed 80021). These are descriptive comparisons over the declared regimes, not a claim of validity outside them.

The CVAE met its regime-specific engineering envelope in 0/34 evaluated cells. Within this declared experiment, that is a failed validation, not merely an inconclusive result. It does not prove that every possible CVAE workflow will fail, but the present frozen workflow should not be treated as an accurate conditional-probability engine.

**Pipeline decision:** the categorical input/output and serialization contracts can support reversible integration and shadow runs. Do not make the current fitted CVAE a required scientific or production dependency; keep the probability model swappable while the fitting objective and marginal integration are hardened.

## What the experiment says about the requested circumstances

### Baseline probability and calibration

- With a 1% population-average focal probability, the median individual-probability MAE was 0.0056 (0.56 percentage points), the median 95th-percentile absolute error was 0.0172, and the median absolute fit-level bias was 0.0031.
- With a 5% population-average focal probability, the median individual-probability MAE was 0.0161 (1.61 percentage points), the median 95th-percentile absolute error was 0.0501, and the median absolute fit-level bias was 0.0114.
- With a 20% population-average focal probability, the median individual-probability MAE was 0.0396 (3.96 percentage points), the median 95th-percentile absolute error was 0.1140, and the median absolute fit-level bias was 0.0202.

A smaller absolute error at 1% does not mean rare outcomes were easy: an MAE near 0.0056 is about half of a 1% population-average risk. The target prevalence is a population mean; each person's p-true varies with X.

| target prevalence | CVAE focal Brier | oracle focal Brier | CVAE focal ECE | CVAE focal ROC-AUC | CVAE focal AP | observed prevalence (AP null) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 0.00958 | 0.00953 | 0.0038 | 0.725 | 0.026 | 0.010 |
| 0.05 | 0.04631 | 0.04507 | 0.0129 | 0.742 | 0.140 | 0.050 |
| 0.2 | 0.14124 | 0.13853 | 0.0229 | 0.745 | 0.420 | 0.200 |

The raw focal Brier score stays numerically close to the oracle because irreducible outcome noise and prevalence dominate that score. Direct p-hat-versus-p-true errors and Brier regret are more sensitive to the conditional-probability error in this simulation. AUC and AP can look respectable while the probabilities themselves are miscalibrated.

The main calibration pattern is compression toward the population mean: high individualized probabilities were underpredicted. q=0.01, p-true [0.05, 0.1): MAE 0.0344, bias -0.0285; q=0.05, p-true [0.2, 0.5): MAE 0.0735, bias -0.0597; q=0.2, p-true [0.5, 1]: MAE 0.1125, bias -0.1026.

### Training sample size

Pooled descriptively across prevalence and K, median CVAE focal MAE was n=500: 0.0207; n=2,000: 0.0134; n=8,000: 0.0169. The correctly specified softmax baseline improved monotonically over the same sample sizes (0.0139 -> 0.0062 -> 0.0033), whereas the frozen CVAE worsened again at n=8,000. More data did not reliably cure this workflow.

### Outcome cardinality

Focal MAE alone is misleading at K=20. The DGP holds the anchor focal probability function fixed across K and makes its nonfocal levels exchangeable; the context outcomes carry distinct class surfaces. The all-outcome probability vector, measured by equal-outcome total variation and KL regret, generally deteriorated as K increased.

| n | K | CVAE focal MAE | CVAE mean-outcome TV | softmax mean-outcome TV | CVAE KL regret |
|---:|---:|---:|---:|---:|---:|
| 500 | 2 | 0.0251 | 0.0367 | 0.0234 | 0.0076 |
| 500 | 5 | 0.0232 | 0.0887 | 0.0632 | 0.0301 |
| 500 | 20 | 0.0137 | 0.1687 | 0.1542 | 0.0989 |
| 2,000 | 2 | 0.0166 | 0.0305 | 0.0131 | 0.0046 |
| 2,000 | 5 | 0.0134 | 0.0790 | 0.0335 | 0.0218 |
| 2,000 | 20 | 0.0107 | 0.0976 | 0.0761 | 0.0320 |
| 8,000 | 2 | 0.0289 | 0.0292 | 0.0057 | 0.0069 |
| 8,000 | 5 | 0.0137 | 0.1071 | 0.0175 | 0.0374 |
| 8,000 | 20 | 0.0083 | 0.1401 | 0.0364 | 0.0647 |

### Mixed versus uniform cardinality

For homogeneous `[5,5,5,5]` versus heterogeneous `[2,3,5,10]` (reported homogeneous/heterogeneous), n=500: focal MAE 0.0263/0.0265 and TV 0.0927/0.0809; n=2,000: focal MAE 0.0146/0.0134 and TV 0.0548/0.0557; n=8,000: focal MAE 0.0276/0.0342 and TV 0.0952/0.0958. There was no stable additional heterogeneity penalty, but all six cells failed and the designs were not entropy matched. That result is inconclusive, not reassuring.

## How to read the evidence

- `p-true` is the exact simulated conditional probability; `p-hat` is a fitted model probability.
- MAE, RMSE, total variation, KL regret, and expected Brier regret directly compare p-hat with p-true; lower is better.
- Held-out NLL and multiclass Brier use realized outcomes and are proper predictive scores; lower is better.
- ROC-AUC and average precision (PR-AUC/AP) measure ranking, not calibration. AP is especially useful for rare classes, but its baseline changes with prevalence.
- AUC/AP are undefined when a test class has no positive or no negative observation. Coverage is shown explicitly; partial defined-class macros are not complete evidence.
- Trend-plot bars are two-sided 95% Student-t intervals across the fixed data seeds within each exact core cell unless a caption says they are full ranges. Displayed lower whiskers for nonnegative metrics are truncated at zero.

## Descriptive comparison over included base fits

Entries are median [minimum, maximum] across fit records; regimes are intentionally not pooled into one inferential estimate. Raw held-out multiclass Brier scores are comparable only within the same schema/cell because their scale changes with cardinality and outcome entropy; direct regret is the safer cross-regime quantity.

| model | focal MAE | focal RMSE | focal p95 absolute error | mean-outcome TV | oracle KL regret | expected Brier regret | held-out multiclass Brier | defined-class macro ROC-AUC | ROC coverage | defined-class macro AP | AP coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| categorical CVAE | 0.0171 [0.0014, 0.1050] | 0.0254 [0.0024, 0.1125] | 0.0513 [0.0042, 0.2295] | 0.0909 [0.0111, 0.1770] | 0.0291 [0.0009, 0.1106] | 0.0081 [0.0005, 0.0275] | 0.7670 [0.2746, 0.9528] | 0.6320 [0.5661, 0.7667] | 4320/4320 | 0.2880 [0.0639, 0.7198] | 4320/4320 |
| independent softmax | 0.0063 [0.0006, 0.0452] | 0.0094 [0.0009, 0.0615] | 0.0200 [0.0019, 0.1292] | 0.0334 [0.0038, 0.1647] | 0.0039 [0.0001, 0.0989] | 0.0015 [0.0001, 0.0101] | 0.7586 [0.2738, 0.9512] | 0.6399 [0.5796, 0.7671] | 4320/4320 | 0.2939 [0.0663, 0.7232] | 4320/4320 |

## Core factorial results

### Low-prevalence cells in terms of events actually observed during training

The MAE interval is truncated at its natural lower bound of zero. Signed bias can cancel across seeds; the engineering gate instead uses the mean absolute seed-level bias.

| target prevalence | n | K | focal training events, median [range] | CVAE focal MAE, mean [95% t interval] | CVAE focal bias, mean [95% t interval] |
|---:|---:|---:|---:|---:|---:|
| 0.01 | 500 | 2 | 4 [2, 6] | 0.0077 [0.0049, 0.0105] | -0.0016 [-0.0060, 0.0028] |
| 0.01 | 500 | 5 | 4 [2, 6] | 0.0083 [0.0045, 0.0121] | -0.0030 [-0.0085, 0.0024] |
| 0.01 | 500 | 20 | 6 [5, 8] | 0.0073 [0.0051, 0.0096] | 0.0021 [-0.0005, 0.0048] |
| 0.01 | 2000 | 2 | 19 [13, 22] | 0.0050 [0.0033, 0.0068] | 0.0002 [-0.0025, 0.0029] |
| 0.01 | 2000 | 5 | 20 [19, 22] | 0.0039 [0.0034, 0.0044] | 0.0012 [0.0006, 0.0019] |
| 0.01 | 2000 | 20 | 19 [14, 26] | 0.0039 [0.0024, 0.0054] | -0.0004 [-0.0037, 0.0028] |
| 0.01 | 8000 | 2 | 84 [71, 99] | 0.0139 [0.0021, 0.0258] | 0.0133 [0.0006, 0.0260] |
| 0.01 | 8000 | 5 | 80 [73, 95] | 0.0053 [0.0028, 0.0077] | 0.0019 [-0.0046, 0.0084] |
| 0.01 | 8000 | 20 | 80 [62, 89] | 0.0033 [0.0013, 0.0053] | -0.0027 [-0.0053, 0.0000] |
| 0.05 | 500 | 2 | 29 [20, 32] | 0.0265 [0.0159, 0.0370] | 0.0048 [-0.0116, 0.0213] |
| 0.05 | 500 | 5 | 19 [16, 27] | 0.0284 [0.0170, 0.0398] | -0.0087 [-0.0250, 0.0077] |
| 0.05 | 500 | 20 | 21 [19, 30] | 0.0152 [0.0108, 0.0196] | 0.0006 [-0.0175, 0.0187] |
| 0.05 | 2000 | 2 | 97 [96, 121] | 0.0208 [0.0110, 0.0305] | 0.0172 [0.0055, 0.0289] |
| 0.05 | 2000 | 5 | 105 [93, 112] | 0.0157 [0.0102, 0.0213] | 0.0057 [-0.0082, 0.0195] |
| 0.05 | 2000 | 20 | 98 [95, 115] | 0.0105 [0.0082, 0.0127] | -0.0032 [-0.0133, 0.0069] |
| 0.05 | 8000 | 2 | 391 [358, 404] | 0.0424 [0.0277, 0.0571] | 0.0398 [0.0256, 0.0540] |
| 0.05 | 8000 | 5 | 405 [391, 430] | 0.0301 [0.0000, 0.0822] | 0.0161 [-0.0450, 0.0773] |
| 0.05 | 8000 | 20 | 412 [370, 445] | 0.0092 [0.0057, 0.0127] | 0.0046 [-0.0028, 0.0119] |

### Focal-class ROC-AUC and average precision

Average precision must be interpreted against the observed focal test prevalence shown in the same row, not against a universal 0.5 baseline.

| target prevalence | n | K | observed focal prevalence (AP null) | CVAE focal ROC-AUC | CVAE focal AP | softmax focal ROC-AUC | softmax focal AP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 500 | 2 | 0.0096 [0.0093, 0.0100] | 0.5647 [0.4523, 0.6771] | 0.0126 [0.0078, 0.0174] | 0.7150 [0.6947, 0.7354] | 0.0247 [0.0173, 0.0321] |
| 0.01 | 500 | 5 | 0.0099 [0.0088, 0.0110] | 0.5373 [0.3611, 0.7135] | 0.0120 [0.0066, 0.0173] | 0.7123 [0.6687, 0.7559] | 0.0289 [0.0183, 0.0394] |
| 0.01 | 500 | 20 | 0.0098 [0.0090, 0.0105] | 0.6347 [0.5237, 0.7457] | 0.0174 [0.0078, 0.0269] | 0.7091 [0.6666, 0.7516] | 0.0262 [0.0169, 0.0355] |
| 0.01 | 2000 | 2 | 0.0098 [0.0089, 0.0108] | 0.7035 [0.6310, 0.7759] | 0.0234 [0.0153, 0.0315] | 0.7467 [0.7146, 0.7788] | 0.0315 [0.0258, 0.0372] |
| 0.01 | 2000 | 5 | 0.0095 [0.0089, 0.0101] | 0.7444 [0.7162, 0.7727] | 0.0298 [0.0220, 0.0375] | 0.7623 [0.7454, 0.7792] | 0.0338 [0.0274, 0.0401] |
| 0.01 | 2000 | 20 | 0.0095 [0.0090, 0.0101] | 0.7177 [0.6777, 0.7576] | 0.0274 [0.0150, 0.0399] | 0.7267 [0.6977, 0.7558] | 0.0282 [0.0180, 0.0384] |
| 0.01 | 8000 | 2 | 0.0098 [0.0091, 0.0105] | 0.7491 [0.7191, 0.7791] | 0.0329 [0.0242, 0.0415] | 0.7534 [0.7260, 0.7809] | 0.0331 [0.0292, 0.0370] |
| 0.01 | 8000 | 5 | 0.0099 [0.0090, 0.0107] | 0.7327 [0.7160, 0.7493] | 0.0309 [0.0224, 0.0393] | 0.7421 [0.7323, 0.7519] | 0.0348 [0.0273, 0.0422] |
| 0.01 | 8000 | 20 | 0.0099 [0.0097, 0.0101] | 0.7442 [0.7250, 0.7633] | 0.0308 [0.0254, 0.0362] | 0.7475 [0.7282, 0.7668] | 0.0312 [0.0263, 0.0360] |
| 0.05 | 500 | 2 | 0.0505 [0.0498, 0.0512] | 0.6898 [0.6299, 0.7496] | 0.0988 [0.0638, 0.1337] | 0.7412 [0.7284, 0.7541] | 0.1396 [0.1282, 0.1511] |
| 0.05 | 500 | 5 | 0.0503 [0.0480, 0.0525] | 0.6658 [0.5688, 0.7627] | 0.1011 [0.0556, 0.1466] | 0.7514 [0.7312, 0.7716] | 0.1486 [0.1251, 0.1721] |
| 0.05 | 500 | 20 | 0.0498 [0.0481, 0.0515] | 0.7372 [0.7174, 0.7571] | 0.1367 [0.1224, 0.1509] | 0.7444 [0.7369, 0.7518] | 0.1394 [0.1314, 0.1474] |
| 0.05 | 2000 | 2 | 0.0498 [0.0481, 0.0516] | 0.7431 [0.7314, 0.7549] | 0.1371 [0.1258, 0.1485] | 0.7478 [0.7380, 0.7576] | 0.1431 [0.1316, 0.1546] |
| 0.05 | 2000 | 5 | 0.0503 [0.0487, 0.0519] | 0.7438 [0.7231, 0.7646] | 0.1439 [0.1315, 0.1563] | 0.7533 [0.7442, 0.7623] | 0.1478 [0.1427, 0.1529] |
| 0.05 | 2000 | 20 | 0.0512 [0.0488, 0.0535] | 0.7473 [0.7441, 0.7505] | 0.1471 [0.1370, 0.1571] | 0.7505 [0.7479, 0.7531] | 0.1484 [0.1377, 0.1591] |
| 0.05 | 8000 | 2 | 0.0502 [0.0496, 0.0507] | 0.7336 [0.7047, 0.7625] | 0.1302 [0.1008, 0.1596] | 0.7593 [0.7472, 0.7715] | 0.1520 [0.1452, 0.1588] |
| 0.05 | 8000 | 5 | 0.0501 [0.0481, 0.0520] | 0.7277 [0.6618, 0.7936] | 0.1289 [0.0966, 0.1611] | 0.7536 [0.7432, 0.7640] | 0.1445 [0.1334, 0.1556] |
| 0.05 | 8000 | 20 | 0.0492 [0.0466, 0.0517] | 0.7452 [0.7354, 0.7550] | 0.1430 [0.1361, 0.1498] | 0.7479 [0.7402, 0.7556] | 0.1448 [0.1375, 0.1521] |
| 0.2 | 500 | 2 | 0.1993 [0.1966, 0.2020] | 0.7395 [0.7268, 0.7521] | 0.4165 [0.3982, 0.4348] | 0.7462 [0.7356, 0.7569] | 0.4258 [0.4103, 0.4413] |
| 0.2 | 500 | 5 | 0.2019 [0.1976, 0.2063] | 0.7428 [0.7348, 0.7509] | 0.4259 [0.4148, 0.4370] | 0.7482 [0.7395, 0.7570] | 0.4320 [0.4211, 0.4428] |
| 0.2 | 500 | 20 | 0.1994 [0.1966, 0.2022] | 0.7426 [0.7346, 0.7506] | 0.4146 [0.4043, 0.4249] | 0.7449 [0.7406, 0.7492] | 0.4197 [0.4094, 0.4300] |
| 0.2 | 2000 | 2 | 0.1995 [0.1956, 0.2035] | 0.7467 [0.7451, 0.7482] | 0.4259 [0.4217, 0.4301] | 0.7509 [0.7481, 0.7538] | 0.4303 [0.4225, 0.4381] |
| 0.2 | 2000 | 5 | 0.2014 [0.1990, 0.2037] | 0.7423 [0.7376, 0.7471] | 0.4234 [0.4141, 0.4326] | 0.7446 [0.7396, 0.7496] | 0.4259 [0.4161, 0.4357] |
| 0.2 | 2000 | 20 | 0.1990 [0.1969, 0.2011] | 0.7453 [0.7385, 0.7520] | 0.4202 [0.4089, 0.4315] | 0.7467 [0.7410, 0.7524] | 0.4225 [0.4121, 0.4329] |
| 0.2 | 8000 | 2 | 0.1989 [0.1959, 0.2019] | 0.7450 [0.7420, 0.7479] | 0.4190 [0.4081, 0.4299] | 0.7516 [0.7465, 0.7567] | 0.4283 [0.4196, 0.4370] |
| 0.2 | 8000 | 5 | 0.2004 [0.1975, 0.2033] | 0.7430 [0.7411, 0.7449] | 0.4241 [0.4115, 0.4368] | 0.7475 [0.7442, 0.7508] | 0.4295 [0.4167, 0.4422] |
| 0.2 | 8000 | 20 | 0.1996 [0.1974, 0.2017] | 0.7442 [0.7352, 0.7532] | 0.4193 [0.4071, 0.4315] | 0.7457 [0.7367, 0.7548] | 0.4206 [0.4076, 0.4336] |

### Discrimination across sample size, prevalence, and cardinality

![Discrimination across sample size, prevalence, and cardinality](categorical_probability_validation_figures/core_discrimination.png)

### Focal-class discrimination (AP null is observed focal prevalence)

![Focal-class discrimination (AP null is observed focal prevalence)](categorical_probability_validation_figures/core_focal_discrimination.png)

### Held-out proper scores and empirical calibration

![Held-out proper scores and empirical calibration](categorical_probability_validation_figures/core_realized_scores.png)

### Direct error against known conditional probabilities

![Direct error against known conditional probabilities](categorical_probability_validation_figures/core_oracle_regret.png)

### Focal-class probability error across the core grid

![Focal-class probability error across the core grid](categorical_probability_validation_figures/core_focal_error.png)

### Focal-class error by known p-true range

![Focal-class error by known p-true range](categorical_probability_validation_figures/focal_true_probability_bands.png)

### Low-prevalence error versus realized focal training events

![Low-prevalence error versus realized focal training events](categorical_probability_validation_figures/low_prevalence_training_counts.png)

## Regime-specific engineering envelopes

Each row is one exact DGP/training regime. `met` requires both the cell-mean checks and every fixed-seed replicate to meet the declared fit-level checks; it is not permission to extrapolate to another regime.

Only 5/170 base fits met all five predictive tolerances; 85/170 met all numerical/validity prerequisites; and 1/170 met both. At the cell-mean level, 0/34 cells met all five predictive checks even before requiring every seed replicate to pass.

| check | base fits passing | cell aggregates passing |
|---|---:|---:|
| absolute focal bias | 44/170 | 4/34 |
| focal MAE | 24/170 | 3/34 |
| focal p95 absolute error | 32/170 | 6/34 |
| mean-outcome KL regret | 43/170 | 9/34 |
| mean-outcome TV | 45/170 | 9/34 |
| valid probabilities / no floor hit | 170/170 | 34/34 |
| GH mean absolute change | 85/170 | 15/34 |
| GH p99 absolute change | 91/170 | 15/34 |

The cell bias gate averages the absolute seed-level biases. The low-prevalence table below instead shows mean signed bias, so opposite seed biases can cancel there and must not be used to reconstruct the gate.

| scenario | n | target prevalence | cardinalities / K | replicates passing | cell envelope | mean focal MAE | worst focal MAE | mean TV | mean KL regret | worst GH p99 change |
|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|
| `core_n500_q010_k2` | 500 | 0.01 | 2 | 1/5 | not met | 0.0077 | 0.0111 | 0.0274 | 0.0064 | 0.0013 |
| `core_n500_q010_k5` | 500 | 0.01 | 5 | 0/5 | not met | 0.0083 | 0.0133 | 0.0876 | 0.0292 | 0.0005 |
| `core_n500_q010_k20` | 500 | 0.01 | 20 | 0/5 | not met | 0.0073 | 0.0102 | 0.1682 | 0.0980 | 0.0001 |
| `core_n500_q050_k2` | 500 | 0.05 | 2 | 0/5 | not met | 0.0265 | 0.0398 | 0.0390 | 0.0095 | 0.0014 |
| `core_n500_q050_k5` | 500 | 0.05 | 5 | 0/5 | not met | 0.0284 | 0.0409 | 0.0922 | 0.0322 | 0.0005 |
| `core_n500_q050_k20` | 500 | 0.05 | 20 | 0/5 | not met | 0.0152 | 0.0209 | 0.1677 | 0.0979 | 0.0001 |
| `core_n500_q200_k2` | 500 | 0.2 | 2 | 0/5 | not met | 0.0399 | 0.0473 | 0.0377 | 0.0075 | 0.0011 |
| `core_n500_q200_k5` | 500 | 0.2 | 5 | 0/5 | not met | 0.0365 | 0.0469 | 0.0846 | 0.0249 | 0.0004 |
| `core_n500_q200_k20` | 500 | 0.2 | 20 | 0/5 | not met | 0.0285 | 0.0459 | 0.1677 | 0.1026 | 0.0001 |
| `core_n2000_q010_k2` | 2000 | 0.01 | 2 | 0/5 | not met | 0.0050 | 0.0072 | 0.0224 | 0.0032 | 0.0209 |
| `core_n2000_q010_k5` | 2000 | 0.01 | 5 | 0/5 | not met | 0.0039 | 0.0043 | 0.0782 | 0.0223 | 0.0075 |
| `core_n2000_q010_k20` | 2000 | 0.01 | 20 | 0/5 | not met | 0.0039 | 0.0053 | 0.0966 | 0.0324 | 0.0004 |
| `core_n2000_q050_k2` | 2000 | 0.05 | 2 | 0/5 | not met | 0.0208 | 0.0328 | 0.0315 | 0.0052 | 0.0250 |
| `core_n2000_q050_k5` | 2000 | 0.05 | 5 | 0/5 | not met | 0.0157 | 0.0208 | 0.0827 | 0.0240 | 0.0058 |
| `core_n2000_q050_k20` | 2000 | 0.05 | 20 | 0/5 | not met | 0.0105 | 0.0129 | 0.1023 | 0.0349 | 0.0004 |
| `core_n2000_q200_k2` | 2000 | 0.2 | 2 | 0/5 | not met | 0.0499 | 0.0765 | 0.0468 | 0.0091 | 0.0373 |
| `core_n2000_q200_k5` | 2000 | 0.2 | 5 | 0/5 | not met | 0.0276 | 0.0531 | 0.0860 | 0.0271 | 0.0073 |
| `core_n2000_q200_k20` | 2000 | 0.2 | 20 | 0/5 | not met | 0.0282 | 0.0375 | 0.0973 | 0.0323 | 0.0006 |
| `core_n8000_q010_k2` | 8000 | 0.01 | 2 | 0/5 | not met | 0.0139 | 0.0289 | 0.0189 | 0.0035 | 0.0127 |
| `core_n8000_q010_k5` | 8000 | 0.01 | 5 | 0/5 | not met | 0.0053 | 0.0073 | 0.1057 | 0.0383 | 0.0394 |
| `core_n8000_q010_k20` | 8000 | 0.01 | 20 | 0/5 | not met | 0.0033 | 0.0057 | 0.1329 | 0.0599 | 0.0338 |
| `core_n8000_q050_k2` | 8000 | 0.05 | 2 | 0/5 | not met | 0.0424 | 0.0536 | 0.0292 | 0.0072 | 0.0470 |
| `core_n8000_q050_k5` | 8000 | 0.05 | 5 | 0/5 | not met | 0.0301 | 0.1050 | 0.1211 | 0.0528 | 0.1048 |
| `core_n8000_q050_k20` | 8000 | 0.05 | 20 | 0/5 | not met | 0.0092 | 0.0138 | 0.1373 | 0.0626 | 0.0492 |
| `core_n8000_q200_k2` | 8000 | 0.2 | 2 | 0/5 | not met | 0.0421 | 0.0618 | 0.0411 | 0.0076 | 0.0779 |
| `core_n8000_q200_k5` | 8000 | 0.2 | 5 | 0/5 | not met | 0.0607 | 0.0786 | 0.0985 | 0.0329 | 0.0734 |
| `core_n8000_q200_k20` | 8000 | 0.2 | 20 | 0/5 | not met | 0.0546 | 0.0618 | 0.1491 | 0.0732 | 0.0327 |
| `heterogeneity_homogeneous_n500` | 500 | 0.05 | 5 | 0/5 | not met | 0.0263 | 0.0372 | 0.0927 | 0.0309 | 0.0006 |
| `heterogeneity_heterogeneous_n500` | 500 | 0.05 | heterogeneous; sd(K)=3.082, sum(K)=20 | 0/5 | not met | 0.0265 | 0.0416 | 0.0809 | 0.0279 | 0.0007 |
| `heterogeneity_homogeneous_n2000` | 2000 | 0.05 | 5 | 0/5 | not met | 0.0146 | 0.0185 | 0.0548 | 0.0111 | 0.0082 |
| `heterogeneity_heterogeneous_n2000` | 2000 | 0.05 | heterogeneous; sd(K)=3.082, sum(K)=20 | 0/5 | not met | 0.0134 | 0.0173 | 0.0557 | 0.0115 | 0.0044 |
| `heterogeneity_homogeneous_n8000` | 8000 | 0.05 | 5 | 0/5 | not met | 0.0276 | 0.0732 | 0.0952 | 0.0318 | 0.0444 |
| `heterogeneity_heterogeneous_n8000` | 8000 | 0.05 | heterogeneous; sd(K)=3.082, sum(K)=20 | 0/5 | not met | 0.0342 | 0.0814 | 0.0958 | 0.0358 | 0.0626 |
| `rho_control_n2000_q050_k5` | 2000 | 0.05 | 5 | 0/5 | not met | 0.0190 | 0.0261 | 0.0943 | 0.0318 | 0.0125 |

## Cardinality heterogeneity

The homogeneous `[5,5,5,5]` and heterogeneous `[2,3,5,10]` designs have the same number of semantic outcomes and the same summed decoder width. They are **not entropy matched**, so any difference cannot be attributed to cardinality heterogeneity alone. Within n and seed, both variants share X and the anchor's p-true/Y, so the table and plot use paired heterogeneous-minus-homogeneous differences.

| n | model | metric | heterogeneous - homogeneous, mean [95% paired t interval] | paired seeds |
|---:|---|---|---:|---:|
| 500 | categorical CVAE | focal RMSE | 0.0015 [-0.0154, 0.0183] | 5 |
| 500 | categorical CVAE | mean TV | -0.0118 [-0.0288, 0.0051] | 5 |
| 500 | categorical CVAE | oracle KL regret | -0.0030 [-0.0151, 0.0091] | 5 |
| 500 | categorical CVAE | expected Brier regret | -0.0026 [-0.0067, 0.0015] | 5 |
| 500 | independent softmax | focal RMSE | 0.0000 [-0.0000, 0.0000] | 5 |
| 500 | independent softmax | mean TV | -0.0057 [-0.0143, 0.0028] | 5 |
| 500 | independent softmax | oracle KL regret | 0.0001 [-0.0033, 0.0034] | 5 |
| 500 | independent softmax | expected Brier regret | -0.0009 [-0.0025, 0.0007] | 5 |
| 2000 | categorical CVAE | focal RMSE | 0.0005 [-0.0048, 0.0058] | 5 |
| 2000 | categorical CVAE | mean TV | 0.0009 [-0.0117, 0.0135] | 5 |
| 2000 | categorical CVAE | oracle KL regret | 0.0005 [-0.0039, 0.0048] | 5 |
| 2000 | categorical CVAE | expected Brier regret | 0.0006 [-0.0013, 0.0025] | 5 |
| 2000 | independent softmax | focal RMSE | 0.0000 [-0.0000, 0.0000] | 5 |
| 2000 | independent softmax | mean TV | -0.0008 [-0.0036, 0.0019] | 5 |
| 2000 | independent softmax | oracle KL regret | 0.0003 [0.0001, 0.0004] | 5 |
| 2000 | independent softmax | expected Brier regret | -0.0001 [-0.0003, 0.0002] | 5 |
| 8000 | categorical CVAE | focal RMSE | 0.0081 [-0.0529, 0.0691] | 5 |
| 8000 | categorical CVAE | mean TV | 0.0006 [-0.0253, 0.0265] | 5 |
| 8000 | categorical CVAE | oracle KL regret | 0.0040 [-0.0102, 0.0183] | 5 |
| 8000 | categorical CVAE | expected Brier regret | 0.0034 [-0.0051, 0.0118] | 5 |
| 8000 | independent softmax | focal RMSE | 0.0000 [-0.0000, 0.0000] | 5 |
| 8000 | independent softmax | mean TV | -0.0023 [-0.0029, -0.0016] | 5 |
| 8000 | independent softmax | oracle KL regret | -0.0001 [-0.0002, -0.0000] | 5 |
| 8000 | independent softmax | expected Brier regret | -0.0001 [-0.0001, -0.0000] | 5 |

![Decoder-width-matched cardinality heterogeneity comparison](categorical_probability_validation_figures/heterogeneity_comparison.png)

## Residual-dependence control

This comparison holds the marginal conditional probabilities fixed while changing the Gaussian-copula residual dependence.

![Residual-dependence control](categorical_probability_validation_figures/rho_control.png)

## Numerical integration diagnostics

Only 85/170 base fits met all numerical/validity prerequisites. All probability matrices were syntactically valid, so the failures came from GH21-versus-GH31 disagreement. Some large-n error magnitudes are therefore numerically uncertain; predictive checks also failed in stable-integration cells, so this does not explain away the probability-recovery failure.

| GH21-vs-GH31 diagnostic | median | minimum | maximum |
|---|---:|---:|---:|
| mean absolute probability change | 0.000502 | 0.000010 | 0.028259 |
| RMSE probability change | 0.000873 | 0.000015 | 0.039754 |
| p99 absolute probability change | 0.003236 | 0.000051 | 0.104809 |
| maximum absolute probability change | 0.005930 | 0.000155 | 0.130726 |

A post-run audit found that the original undamped Newton solver for the experimental `truth_calibration_intercept`/`slope` fields could diverge for rare classes. The committed artifact is a disclosed correction-only rerun using centered/scaled damped Newton updates. Every nonexcluded result field matched the first complete run exactly. These coefficients remain excluded from the gates and headline evidence.

![Gauss-Hermite integration diagnostics](categorical_probability_validation_figures/quadrature_diagnostics.png)

## Initialization sensitivity

| scenario | data seed | initialization replicate | focal MAE | focal RMSE | focal p95 error | mean TV | mean KL regret | best epoch | active latent units |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `core_n2000_q050_k5` | 5003 | 0 | 0.0134 | 0.0220 | 0.0476 | 0.0775 | 0.0190 | 30 | 2 |
| `core_n2000_q050_k5` | 5003 | 1 | 0.0134 | 0.0159 | 0.0321 | 0.0972 | 0.0330 | 30 | 2 |
| `core_n2000_q050_k5` | 5003 | 2 | 0.0171 | 0.0309 | 0.0678 | 0.1371 | 0.0598 | 30 | 2 |
| `core_n500_q010_k20` | 5003 | 0 | 0.0071 | 0.0105 | 0.0188 | 0.1651 | 0.0989 | 30 | 2 |
| `core_n500_q010_k20` | 5003 | 1 | 0.0101 | 0.0136 | 0.0233 | 0.1797 | 0.1094 | 30 | 2 |
| `core_n500_q010_k20` | 5003 | 2 | 0.0067 | 0.0104 | 0.0213 | 0.1712 | 0.1013 | 30 | 2 |
| `core_n8000_q200_k2` | 5003 | 0 | 0.0618 | 0.0684 | 0.1116 | 0.0373 | 0.0070 | 27 | 2 |
| `core_n8000_q200_k2` | 5003 | 1 | 0.0299 | 0.0397 | 0.0825 | 0.0340 | 0.0053 | 30 | 2 |
| `core_n8000_q200_k2` | 5003 | 2 | 0.0459 | 0.0534 | 0.0966 | 0.0411 | 0.0071 | 26 | 2 |

![Initialization/training replicate sensitivity](categorical_probability_validation_figures/initialization_diagnostics.png)

## Individual probability and calibration plots

Sentinels were fixed before fitting. Agreement panels plot individual p-hat values directly against p-true using the capped plotting rows. Reliability panels use the full held-out test set, bin on each candidate model's p-hat and show both the observed event fraction (95% Wilson interval) and mean p-true in that same bin. Bin counts are summarized in each panel subtitle. Low-probability panels restrict the predicted-probability x axis but retain the full 0-1 y axis, so severe underprediction remains visible rather than being clipped.

### core_n500_q010_k20: individual p-hat versus p-true

![core_n500_q010_k20: individual p-hat versus p-true](categorical_probability_validation_figures/core_n500_q010_k20_probability_agreement.png)

### core_n500_q010_k20: observed and oracle calibration for every anchor level (page 1)

![core_n500_q010_k20: observed and oracle calibration for every anchor level (page 1)](categorical_probability_validation_figures/core_n500_q010_k20_anchor_calibration_01.png)

### core_n500_q010_k20: observed and oracle calibration for every anchor level (page 2)

![core_n500_q010_k20: observed and oracle calibration for every anchor level (page 2)](categorical_probability_validation_figures/core_n500_q010_k20_anchor_calibration_02.png)

### core_n500_q010_k20: observed and oracle calibration for every anchor level (0-0.20 zoom) (page 3)

![core_n500_q010_k20: observed and oracle calibration for every anchor level (0-0.20 zoom) (page 3)](categorical_probability_validation_figures/core_n500_q010_k20_anchor_calibration_low_probability_01.png)

### core_n500_q010_k20: observed and oracle calibration for every anchor level (0-0.20 zoom) (page 4)

![core_n500_q010_k20: observed and oracle calibration for every anchor level (0-0.20 zoom) (page 4)](categorical_probability_validation_figures/core_n500_q010_k20_anchor_calibration_low_probability_02.png)

### core_n2000_q050_k5: individual p-hat versus p-true

![core_n2000_q050_k5: individual p-hat versus p-true](categorical_probability_validation_figures/core_n2000_q050_k5_probability_agreement.png)

### core_n2000_q050_k5: observed and oracle calibration for every anchor level (page 1)

![core_n2000_q050_k5: observed and oracle calibration for every anchor level (page 1)](categorical_probability_validation_figures/core_n2000_q050_k5_anchor_calibration_01.png)

### core_n2000_q050_k5: observed and oracle calibration for every anchor level (0-0.20 zoom) (page 2)

![core_n2000_q050_k5: observed and oracle calibration for every anchor level (0-0.20 zoom) (page 2)](categorical_probability_validation_figures/core_n2000_q050_k5_anchor_calibration_low_probability_01.png)

### core_n8000_q200_k2: individual p-hat versus p-true

![core_n8000_q200_k2: individual p-hat versus p-true](categorical_probability_validation_figures/core_n8000_q200_k2_probability_agreement.png)

### core_n8000_q200_k2: observed and oracle calibration for every anchor level (page 1)

![core_n8000_q200_k2: observed and oracle calibration for every anchor level (page 1)](categorical_probability_validation_figures/core_n8000_q200_k2_anchor_calibration_01.png)

### core_n8000_q200_k2: observed and oracle calibration for every anchor level (0-0.20 zoom) (page 2)

![core_n8000_q200_k2: observed and oracle calibration for every anchor level (0-0.20 zoom) (page 2)](categorical_probability_validation_figures/core_n8000_q200_k2_anchor_calibration_low_probability_01.png)

### heterogeneity_homogeneous_n2000: individual p-hat versus p-true

![heterogeneity_homogeneous_n2000: individual p-hat versus p-true](categorical_probability_validation_figures/heterogeneity_homogeneous_n2000_probability_agreement.png)

### heterogeneity_homogeneous_n2000: observed and oracle calibration for every anchor level (page 1)

![heterogeneity_homogeneous_n2000: observed and oracle calibration for every anchor level (page 1)](categorical_probability_validation_figures/heterogeneity_homogeneous_n2000_anchor_calibration_01.png)

### heterogeneity_homogeneous_n2000: observed and oracle calibration for every anchor level (0-0.20 zoom) (page 2)

![heterogeneity_homogeneous_n2000: observed and oracle calibration for every anchor level (0-0.20 zoom) (page 2)](categorical_probability_validation_figures/heterogeneity_homogeneous_n2000_anchor_calibration_low_probability_01.png)

### heterogeneity_heterogeneous_n2000: individual p-hat versus p-true

![heterogeneity_heterogeneous_n2000: individual p-hat versus p-true](categorical_probability_validation_figures/heterogeneity_heterogeneous_n2000_probability_agreement.png)

### heterogeneity_heterogeneous_n2000: observed and oracle calibration for every anchor level (page 1)

![heterogeneity_heterogeneous_n2000: observed and oracle calibration for every anchor level (page 1)](categorical_probability_validation_figures/heterogeneity_heterogeneous_n2000_anchor_calibration_01.png)

### heterogeneity_heterogeneous_n2000: observed and oracle calibration for every anchor level (0-0.20 zoom) (page 2)

![heterogeneity_heterogeneous_n2000: observed and oracle calibration for every anchor level (0-0.20 zoom) (page 2)](categorical_probability_validation_figures/heterogeneity_heterogeneous_n2000_anchor_calibration_low_probability_01.png)

### rho_control_n2000_q050_k5: individual p-hat versus p-true

![rho_control_n2000_q050_k5: individual p-hat versus p-true](categorical_probability_validation_figures/rho_control_n2000_q050_k5_probability_agreement.png)

### rho_control_n2000_q050_k5: observed and oracle calibration for every anchor level (page 1)

![rho_control_n2000_q050_k5: observed and oracle calibration for every anchor level (page 1)](categorical_probability_validation_figures/rho_control_n2000_q050_k5_anchor_calibration_01.png)

### rho_control_n2000_q050_k5: observed and oracle calibration for every anchor level (0-0.20 zoom) (page 2)

![rho_control_n2000_q050_k5: observed and oracle calibration for every anchor level (0-0.20 zoom) (page 2)](categorical_probability_validation_figures/rho_control_n2000_q050_k5_anchor_calibration_low_probability_01.png)

## Reproducibility and provenance

| item | value |
|---|---|
| selected scenarios | 34 |
| selected data seeds | `[5003, 10007, 20011, 40009, 80021]` |
| base fits | 170 |
| total fits including initialization sensitivity | 176 |
| runtime seconds | 592.5 |
| device | `cpu` |
| git dirty | `False` |
| git head | `44d774bbee083f60e874f00cead485b9888f9e42` |
| numpy | `2.4.6` |
| package path | `python/multioutcome_cvae/__init__.py` |
| package version | `0.3.0` |
| platform | `macOS-26.6.2-arm64-arm-64bit` |
| python | `3.11.15` |
| runner source path | `validation/categorical_probability_validation.py` |
| runner source sha256 | `27a3089811e8afde5e8f4fdbe69f46415a073e8973c50e30e32367376d0c0de7` |
| torch | `2.14.0` |
| trainer source path | `python/multioutcome_cvae/model.py` |
| trainer source sha256 | `709304247799912f97b2415426c36d5ed31601534676b3f1334fb88a1777ccdd` |

### Development-history disclosure

- Retired proposed seed set: `[4243, 8677, 16127, 32353, 65537]`.
- Checks whose results were seen during development:
  - An implementation-only CLI smoke fit was run for the rho=0 control with proposed seed 4243. It completed successfully as software but did not meet the complete engineering envelope (0 of 1 evaluated cells passed).
  - A CVAE smoke fit for retired seed 4243 on core_n500_q010_k2 produced focal MAE about 0.0092, bias about +0.0049, and p95 absolute error about 0.0227.
  - Focal training counts were inspected across the entire retired proposed seed set, and baseline-only software checks were run on several retired cells.
- Disposition: The entire proposed seed set was retired. No DGP coefficient, model setting, diagnostic, or engineering tolerance was changed in response. The replacement canonical seeds were reserved unseen until after source commit and public protocol posting.
- Canonical seed set: `[5003, 10007, 20011, 40009, 80021]`.

### Protocol rerun commands

The machine-readable result is committed losslessly as `docs/categorical_probability_validation_results.json.gz` to avoid adding a 62 MB pretty-printed JSON file to Git history.
The committed correction-only canonical artifact was produced from Git commit `44d774bbee083f60e874f00cead485b9888f9e42` with runner SHA-256 `27a3089811e8afde5e8f4fdbe69f46415a073e8973c50e30e32367376d0c0de7`. The first complete run came from `113c5a6080980076ce753187c3f93ed72ab86e42`; all fields other than runtime/provenance and the corrected, excluded truth-calibration coefficients matched exactly. Checkout the recorded artifact commit to reproduce the exact canonical runner.

```bash
PYTHONPATH=python .venv/bin/python -m validation.categorical_probability_validation --output /tmp/categorical_probability_validation_results.json --verbose
gzip -n -9 /tmp/categorical_probability_validation_results.json
MPLCONFIGDIR=/tmp/multioutcome-cvae-matplotlib PYTHONPATH=python .venv/bin/python -m validation.categorical_probability_report /tmp/categorical_probability_validation_results.json.gz --output docs/categorical_probability_validation.md --figure-dir docs/categorical_probability_validation_figures
```

### Exact protocol manifest

```json
{
  "config": {
    "anchor_x_weights": [
      1.35,
      -0.9,
      0.65
    ],
    "baseline_l2": 0.001,
    "baseline_max_iter": 100,
    "baseline_tolerance_grad": 1e-08,
    "batch_size": 128,
    "beta_kl": 0.2,
    "calibration_bins": 10,
    "context_weight_scale": 0.9,
    "copula_rho": 0.35,
    "covariate_lower": -1.0,
    "covariate_upper": 1.0,
    "data_seeds": [
      5003,
      10007,
      20011,
      40009,
      80021
    ],
    "early_stopping_min_delta": 0.0001,
    "early_stopping_patience": 6,
    "early_stopping_start_epoch": 8,
    "fitted_quadrature_check_order": 31,
    "fitted_quadrature_order": 21,
    "focal_prevalences": [
      0.01,
      0.05,
      0.2
    ],
    "heterogeneity_anchor_index": 2,
    "heterogeneity_focal_prevalence": 0.05,
    "heterogeneity_schemas": [
      [
        5,
        5,
        5,
        5
      ],
      [
        2,
        3,
        5,
        10
      ]
    ],
    "hidden_dim": 48,
    "homogeneous_cardinalities": [
      2,
      5,
      20
    ],
    "include_heterogeneity": true,
    "include_initialization_sensitivity": true,
    "include_rho_control": true,
    "initialization_sensitivity_replicates": 3,
    "intercept_alpha": 0.5,
    "kl_warmup_epochs": 8,
    "latent_dim": 2,
    "learning_rate": 0.001,
    "n_hidden_layers": 2,
    "n_test": 20000,
    "num_epochs": 30,
    "plot_rows": 1000,
    "population_quadrature_order": 15,
    "probability_epsilon": 1e-12,
    "quadrature_check_rows": 2000,
    "quadrature_x_batch_size": 128,
    "rho_control_cardinality": 5,
    "rho_control_n_train": 2000,
    "rho_control_prevalence": 0.05,
    "sample_sizes": [
      500,
      2000,
      8000
    ],
    "semantic_outcomes": 3,
    "validation_sizes": [
      [
        500,
        250
      ],
      [
        2000,
        500
      ],
      [
        8000,
        2000
      ]
    ],
    "x_dim": 3
  },
  "development_history": {
    "canonical_seed_set": [
      5003,
      10007,
      20011,
      40009,
      80021
    ],
    "disposition": "The entire proposed seed set was retired. No DGP coefficient, model setting, diagnostic, or engineering tolerance was changed in response. The replacement canonical seeds were reserved unseen until after source commit and public protocol posting.",
    "retired_proposed_seed_set": [
      4243,
      8677,
      16127,
      32353,
      65537
    ],
    "seen_development_checks": [
      "An implementation-only CLI smoke fit was run for the rho=0 control with proposed seed 4243. It completed successfully as software but did not meet the complete engineering envelope (0 of 1 evaluated cells passed).",
      "A CVAE smoke fit for retired seed 4243 on core_n500_q010_k2 produced focal MAE about 0.0092, bias about +0.0049, and p95 absolute error about 0.0227.",
      "Focal training counts were inspected across the entire retired proposed seed set, and baseline-only software checks were run on several retired cells."
    ]
  },
  "dgp_contract": {
    "X": "iid Uniform[-1,1] in three dimensions",
    "context_prevalence": "exactly balanced in the population by deterministic multinomial intercept calibration",
    "dependence": "equicorrelated Gaussian copula; V_j=sqrt(rho)S+sqrt(1-rho)E_j and U_j=Phi(V_j)",
    "focal_prevalence": "population E_X[P(anchor=focal|X)] solved by deterministic tensor Gauss-Legendre quadrature",
    "marginals": "exact linear softmax"
  },
  "engineering_tolerances": {
    "absolute_focal_bias": "<= max(0.0025, 0.05*q)",
    "focal_mae": "<= max(0.005, 0.10*q)",
    "focal_p95_absolute_error": "<= max(0.02, 0.25*q)",
    "mean_outcome_kl_regret": "<= 0.01 nat/outcome",
    "mean_outcome_total_variation": "<= 0.05",
    "probability_validity": "no invalid values, invalid row sums, or floor hits",
    "quadrature_mean_absolute_change": "GH21 vs GH31 <= 0.0005",
    "quadrature_p99_absolute_change": "GH21 vs GH31 <= 0.005",
    "scope": "per-cell engineering envelope only; ROC AUC and average precision are descriptive and there is no global pass/fail claim"
  },
  "primary_estimands": {
    "all_outcomes": "equal-outcome mean total variation, expected Brier regret, and KL/log-score regret",
    "cvae_probability": "E_Z[softmax(decoder(X,Z))] under the fitted standard-normal prior",
    "focal": "bias, MAE, RMSE, and p95 absolute p_hat-p_true error",
    "realized_secondary": "NLL, multiclass Brier, one-vs-rest ROC AUC, average precision, and adaptive-bin empirical reliability"
  },
  "protocol": "categorical-conditional-probability-v1",
  "question": "How accurately are individual marginal conditional probabilities recovered across sample size, focal prevalence, cardinality, and matched cardinality heterogeneity?",
  "replication": {
    "base": "five fresh fixed data/training seeds in every cell",
    "initialization_sensitivity": "three total initialization/training replicates on the first fixed data split for predeclared worst, central, and favorable core cells",
    "plot_payload": "first plot_rows test rows for predeclared sentinel cells at the first fixed data seed and initialization replicate zero"
  },
  "reproducibility_metadata_fields": "Python, NumPy, Torch, platform, git HEAD/dirty state, imported package path, CVAETrainer source path/hash, and runner source path/hash are captured at execution time outside this environment-independent hash",
  "scenarios": [
    {
      "anchor_index": 0,
      "cardinalities": [
        2,
        2,
        2
      ],
      "copula_rho": 0.35,
      "design": "core",
      "focal_prevalence": 0.01,
      "n_test": 20000,
      "n_train": 500,
      "n_validation": 250,
      "name": "core_n500_q010_k2",
      "outcomes": [
        {
          "focal_class_index": 0,
          "intercepts": [
            -5.071324348887,
            0.0
          ],
          "levels": [
            "focal",
            "other_1"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0
            ],
            [
              -0.9,
              0.0
            ],
            [
              0.65,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            0.0,
            0.0
          ],
          "levels": [
            "level_0",
            "level_1"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0
            ],
            [
              0.839094611045,
              -0.839094611045
            ],
            [
              0.325453888768,
              -0.325453888768
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            0.0,
            0.0
          ],
          "levels": [
            "level_0",
            "level_1"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              -0.606859120465
            ],
            [
              0.0,
              0.0
            ],
            [
              0.664621702857,
              -0.664621702857
            ]
          ]
        }
      ],
      "pair_key": "core_n500_q010_k2",
      "variant": "homogeneous"
    },
    {
      "anchor_index": 0,
      "cardinalities": [
        5,
        5,
        5
      ],
      "copula_rho": 0.35,
      "design": "core",
      "focal_prevalence": 0.01,
      "n_test": 20000,
      "n_train": 500,
      "n_validation": 250,
      "name": "core_n500_q010_k5",
      "outcomes": [
        {
          "focal_class_index": 0,
          "intercepts": [
            -3.685029987767,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "levels": [
            "focal",
            "other_1",
            "other_2",
            "other_3",
            "other_4"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              -0.9,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.65,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            7.1513196e-05,
            0.000770550074,
            0.00040446016,
            -0.000520446354,
            -0.000726077075
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.839094611045,
              -0.050230546965,
              -0.870138796344,
              -0.487544804105,
              0.568819536368
            ],
            [
              0.325453888768,
              0.898597180138,
              0.229909710751,
              -0.75650516455,
              -0.697455615108
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            -0.000776956154,
            -0.000104550295,
            0.000712610457,
            0.000544792259,
            -0.000375896266
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              0.819622582788,
              -0.100304506355,
              -0.881614176941,
              -0.444563019958
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.664621702857,
              -0.371777919979,
              -0.89439309367,
              -0.180987411213,
              0.782536722005
            ]
          ]
        }
      ],
      "pair_key": "core_n500_q010_k5",
      "variant": "homogeneous"
    },
    {
      "anchor_index": 0,
      "cardinalities": [
        20,
        20,
        20
      ],
      "copula_rho": 0.35,
      "design": "core",
      "focal_prevalence": 0.01,
      "n_test": 20000,
      "n_train": 500,
      "n_validation": 250,
      "name": "core_n500_q010_k20",
      "outcomes": [
        {
          "focal_class_index": 0,
          "intercepts": [
            -2.12688536972,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "levels": [
            "focal",
            "other_1",
            "other_2",
            "other_3",
            "other_4",
            "other_5",
            "other_6",
            "other_7",
            "other_8",
            "other_9",
            "other_10",
            "other_11",
            "other_12",
            "other_13",
            "other_14",
            "other_15",
            "other_16",
            "other_17",
            "other_18",
            "other_19"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              -0.9,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.65,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585,
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585,
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585,
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4",
            "level_5",
            "level_6",
            "level_7",
            "level_8",
            "level_9",
            "level_10",
            "level_11",
            "level_12",
            "level_13",
            "level_14",
            "level_15",
            "level_16",
            "level_17",
            "level_18",
            "level_19"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.839094611045,
              0.697455615108,
              0.487544804105,
              0.229909710751,
              -0.050230546965,
              -0.325453888768,
              -0.568819536368,
              -0.75650516455,
              -0.870138796344,
              -0.898597180138,
              -0.839094611045,
              -0.697455615108,
              -0.487544804105,
              -0.229909710751,
              0.050230546965,
              0.325453888768,
              0.568819536368,
              0.75650516455,
              0.870138796344,
              0.898597180138
            ],
            [
              0.325453888768,
              0.568819536368,
              0.75650516455,
              0.870138796344,
              0.898597180138,
              0.839094611045,
              0.697455615108,
              0.487544804105,
              0.229909710751,
              -0.050230546965,
              -0.325453888768,
              -0.568819536368,
              -0.75650516455,
              -0.870138796344,
              -0.898597180138,
              -0.839094611045,
              -0.697455615108,
              -0.487544804105,
              -0.229909710751,
              0.050230546965
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05,
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05,
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05,
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4",
            "level_5",
            "level_6",
            "level_7",
            "level_8",
            "level_9",
            "level_10",
            "level_11",
            "level_12",
            "level_13",
            "level_14",
            "level_15",
            "level_16",
            "level_17",
            "level_18",
            "level_19"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              0.782536722005,
              0.881614176941,
              0.89439309367,
              0.819622582788,
              0.664621702857,
              0.444563019958,
              0.180987411213,
              -0.100304506355,
              -0.371777919979,
              -0.606859120465,
              -0.782536722005,
              -0.881614176941,
              -0.89439309367,
              -0.819622582788,
              -0.664621702857,
              -0.444563019958,
              -0.180987411213,
              0.100304506355,
              0.371777919979
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.664621702857,
              0.444563019958,
              0.180987411213,
              -0.100304506355,
              -0.371777919979,
              -0.606859120465,
              -0.782536722005,
              -0.881614176941,
              -0.89439309367,
              -0.819622582788,
              -0.664621702857,
              -0.444563019958,
              -0.180987411213,
              0.100304506355,
              0.371777919979,
              0.606859120465,
              0.782536722005,
              0.881614176941,
              0.89439309367,
              0.819622582788
            ]
          ]
        }
      ],
      "pair_key": "core_n500_q010_k20",
      "variant": "homogeneous"
    },
    {
      "anchor_index": 0,
      "cardinalities": [
        2,
        2,
        2
      ],
      "copula_rho": 0.35,
      "design": "core",
      "focal_prevalence": 0.05,
      "n_test": 20000,
      "n_train": 500,
      "n_validation": 250,
      "name": "core_n500_q050_k2",
      "outcomes": [
        {
          "focal_class_index": 0,
          "intercepts": [
            -3.374725967262,
            0.0
          ],
          "levels": [
            "focal",
            "other_1"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0
            ],
            [
              -0.9,
              0.0
            ],
            [
              0.65,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            0.0,
            0.0
          ],
          "levels": [
            "level_0",
            "level_1"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0
            ],
            [
              0.839094611045,
              -0.839094611045
            ],
            [
              0.325453888768,
              -0.325453888768
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            0.0,
            0.0
          ],
          "levels": [
            "level_0",
            "level_1"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              -0.606859120465
            ],
            [
              0.0,
              0.0
            ],
            [
              0.664621702857,
              -0.664621702857
            ]
          ]
        }
      ],
      "pair_key": "core_n500_q050_k2",
      "variant": "homogeneous"
    },
    {
      "anchor_index": 0,
      "cardinalities": [
        5,
        5,
        5
      ],
      "copula_rho": 0.35,
      "design": "core",
      "focal_prevalence": 0.05,
      "n_test": 20000,
      "n_train": 500,
      "n_validation": 250,
      "name": "core_n500_q050_k5",
      "outcomes": [
        {
          "focal_class_index": 0,
          "intercepts": [
            -1.988431606142,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "levels": [
            "focal",
            "other_1",
            "other_2",
            "other_3",
            "other_4"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              -0.9,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.65,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            7.1513196e-05,
            0.000770550074,
            0.00040446016,
            -0.000520446354,
            -0.000726077075
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.839094611045,
              -0.050230546965,
              -0.870138796344,
              -0.487544804105,
              0.568819536368
            ],
            [
              0.325453888768,
              0.898597180138,
              0.229909710751,
              -0.75650516455,
              -0.697455615108
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            -0.000776956154,
            -0.000104550295,
            0.000712610457,
            0.000544792259,
            -0.000375896266
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              0.819622582788,
              -0.100304506355,
              -0.881614176941,
              -0.444563019958
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.664621702857,
              -0.371777919979,
              -0.89439309367,
              -0.180987411213,
              0.782536722005
            ]
          ]
        }
      ],
      "pair_key": "core_n500_q050_k5",
      "variant": "homogeneous"
    },
    {
      "anchor_index": 0,
      "cardinalities": [
        20,
        20,
        20
      ],
      "copula_rho": 0.35,
      "design": "core",
      "focal_prevalence": 0.05,
      "n_test": 20000,
      "n_train": 500,
      "n_validation": 250,
      "name": "core_n500_q050_k20",
      "outcomes": [
        {
          "focal_class_index": 0,
          "intercepts": [
            -0.430286988095,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "levels": [
            "focal",
            "other_1",
            "other_2",
            "other_3",
            "other_4",
            "other_5",
            "other_6",
            "other_7",
            "other_8",
            "other_9",
            "other_10",
            "other_11",
            "other_12",
            "other_13",
            "other_14",
            "other_15",
            "other_16",
            "other_17",
            "other_18",
            "other_19"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              -0.9,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.65,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585,
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585,
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585,
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4",
            "level_5",
            "level_6",
            "level_7",
            "level_8",
            "level_9",
            "level_10",
            "level_11",
            "level_12",
            "level_13",
            "level_14",
            "level_15",
            "level_16",
            "level_17",
            "level_18",
            "level_19"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.839094611045,
              0.697455615108,
              0.487544804105,
              0.229909710751,
              -0.050230546965,
              -0.325453888768,
              -0.568819536368,
              -0.75650516455,
              -0.870138796344,
              -0.898597180138,
              -0.839094611045,
              -0.697455615108,
              -0.487544804105,
              -0.229909710751,
              0.050230546965,
              0.325453888768,
              0.568819536368,
              0.75650516455,
              0.870138796344,
              0.898597180138
            ],
            [
              0.325453888768,
              0.568819536368,
              0.75650516455,
              0.870138796344,
              0.898597180138,
              0.839094611045,
              0.697455615108,
              0.487544804105,
              0.229909710751,
              -0.050230546965,
              -0.325453888768,
              -0.568819536368,
              -0.75650516455,
              -0.870138796344,
              -0.898597180138,
              -0.839094611045,
              -0.697455615108,
              -0.487544804105,
              -0.229909710751,
              0.050230546965
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05,
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05,
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05,
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4",
            "level_5",
            "level_6",
            "level_7",
            "level_8",
            "level_9",
            "level_10",
            "level_11",
            "level_12",
            "level_13",
            "level_14",
            "level_15",
            "level_16",
            "level_17",
            "level_18",
            "level_19"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              0.782536722005,
              0.881614176941,
              0.89439309367,
              0.819622582788,
              0.664621702857,
              0.444563019958,
              0.180987411213,
              -0.100304506355,
              -0.371777919979,
              -0.606859120465,
              -0.782536722005,
              -0.881614176941,
              -0.89439309367,
              -0.819622582788,
              -0.664621702857,
              -0.444563019958,
              -0.180987411213,
              0.100304506355,
              0.371777919979
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.664621702857,
              0.444563019958,
              0.180987411213,
              -0.100304506355,
              -0.371777919979,
              -0.606859120465,
              -0.782536722005,
              -0.881614176941,
              -0.89439309367,
              -0.819622582788,
              -0.664621702857,
              -0.444563019958,
              -0.180987411213,
              0.100304506355,
              0.371777919979,
              0.606859120465,
              0.782536722005,
              0.881614176941,
              0.89439309367,
              0.819622582788
            ]
          ]
        }
      ],
      "pair_key": "core_n500_q050_k20",
      "variant": "homogeneous"
    },
    {
      "anchor_index": 0,
      "cardinalities": [
        2,
        2,
        2
      ],
      "copula_rho": 0.35,
      "design": "core",
      "focal_prevalence": 0.2,
      "n_test": 20000,
      "n_train": 500,
      "n_validation": 250,
      "name": "core_n500_q200_k2",
      "outcomes": [
        {
          "focal_class_index": 0,
          "intercepts": [
            -1.661789569696,
            0.0
          ],
          "levels": [
            "focal",
            "other_1"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0
            ],
            [
              -0.9,
              0.0
            ],
            [
              0.65,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            0.0,
            0.0
          ],
          "levels": [
            "level_0",
            "level_1"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0
            ],
            [
              0.839094611045,
              -0.839094611045
            ],
            [
              0.325453888768,
              -0.325453888768
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            0.0,
            0.0
          ],
          "levels": [
            "level_0",
            "level_1"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              -0.606859120465
            ],
            [
              0.0,
              0.0
            ],
            [
              0.664621702857,
              -0.664621702857
            ]
          ]
        }
      ],
      "pair_key": "core_n500_q200_k2",
      "variant": "homogeneous"
    },
    {
      "anchor_index": 0,
      "cardinalities": [
        5,
        5,
        5
      ],
      "copula_rho": 0.35,
      "design": "core",
      "focal_prevalence": 0.2,
      "n_test": 20000,
      "n_train": 500,
      "n_validation": 250,
      "name": "core_n500_q200_k5",
      "outcomes": [
        {
          "focal_class_index": 0,
          "intercepts": [
            -0.275495208576,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "levels": [
            "focal",
            "other_1",
            "other_2",
            "other_3",
            "other_4"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              -0.9,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.65,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            7.1513196e-05,
            0.000770550074,
            0.00040446016,
            -0.000520446354,
            -0.000726077075
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.839094611045,
              -0.050230546965,
              -0.870138796344,
              -0.487544804105,
              0.568819536368
            ],
            [
              0.325453888768,
              0.898597180138,
              0.229909710751,
              -0.75650516455,
              -0.697455615108
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            -0.000776956154,
            -0.000104550295,
            0.000712610457,
            0.000544792259,
            -0.000375896266
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              0.819622582788,
              -0.100304506355,
              -0.881614176941,
              -0.444563019958
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.664621702857,
              -0.371777919979,
              -0.89439309367,
              -0.180987411213,
              0.782536722005
            ]
          ]
        }
      ],
      "pair_key": "core_n500_q200_k5",
      "variant": "homogeneous"
    },
    {
      "anchor_index": 0,
      "cardinalities": [
        20,
        20,
        20
      ],
      "copula_rho": 0.35,
      "design": "core",
      "focal_prevalence": 0.2,
      "n_test": 20000,
      "n_train": 500,
      "n_validation": 250,
      "name": "core_n500_q200_k20",
      "outcomes": [
        {
          "focal_class_index": 0,
          "intercepts": [
            1.282649409471,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "levels": [
            "focal",
            "other_1",
            "other_2",
            "other_3",
            "other_4",
            "other_5",
            "other_6",
            "other_7",
            "other_8",
            "other_9",
            "other_10",
            "other_11",
            "other_12",
            "other_13",
            "other_14",
            "other_15",
            "other_16",
            "other_17",
            "other_18",
            "other_19"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              -0.9,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.65,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585,
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585,
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585,
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4",
            "level_5",
            "level_6",
            "level_7",
            "level_8",
            "level_9",
            "level_10",
            "level_11",
            "level_12",
            "level_13",
            "level_14",
            "level_15",
            "level_16",
            "level_17",
            "level_18",
            "level_19"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.839094611045,
              0.697455615108,
              0.487544804105,
              0.229909710751,
              -0.050230546965,
              -0.325453888768,
              -0.568819536368,
              -0.75650516455,
              -0.870138796344,
              -0.898597180138,
              -0.839094611045,
              -0.697455615108,
              -0.487544804105,
              -0.229909710751,
              0.050230546965,
              0.325453888768,
              0.568819536368,
              0.75650516455,
              0.870138796344,
              0.898597180138
            ],
            [
              0.325453888768,
              0.568819536368,
              0.75650516455,
              0.870138796344,
              0.898597180138,
              0.839094611045,
              0.697455615108,
              0.487544804105,
              0.229909710751,
              -0.050230546965,
              -0.325453888768,
              -0.568819536368,
              -0.75650516455,
              -0.870138796344,
              -0.898597180138,
              -0.839094611045,
              -0.697455615108,
              -0.487544804105,
              -0.229909710751,
              0.050230546965
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05,
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05,
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05,
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4",
            "level_5",
            "level_6",
            "level_7",
            "level_8",
            "level_9",
            "level_10",
            "level_11",
            "level_12",
            "level_13",
            "level_14",
            "level_15",
            "level_16",
            "level_17",
            "level_18",
            "level_19"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              0.782536722005,
              0.881614176941,
              0.89439309367,
              0.819622582788,
              0.664621702857,
              0.444563019958,
              0.180987411213,
              -0.100304506355,
              -0.371777919979,
              -0.606859120465,
              -0.782536722005,
              -0.881614176941,
              -0.89439309367,
              -0.819622582788,
              -0.664621702857,
              -0.444563019958,
              -0.180987411213,
              0.100304506355,
              0.371777919979
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.664621702857,
              0.444563019958,
              0.180987411213,
              -0.100304506355,
              -0.371777919979,
              -0.606859120465,
              -0.782536722005,
              -0.881614176941,
              -0.89439309367,
              -0.819622582788,
              -0.664621702857,
              -0.444563019958,
              -0.180987411213,
              0.100304506355,
              0.371777919979,
              0.606859120465,
              0.782536722005,
              0.881614176941,
              0.89439309367,
              0.819622582788
            ]
          ]
        }
      ],
      "pair_key": "core_n500_q200_k20",
      "variant": "homogeneous"
    },
    {
      "anchor_index": 0,
      "cardinalities": [
        2,
        2,
        2
      ],
      "copula_rho": 0.35,
      "design": "core",
      "focal_prevalence": 0.01,
      "n_test": 20000,
      "n_train": 2000,
      "n_validation": 500,
      "name": "core_n2000_q010_k2",
      "outcomes": [
        {
          "focal_class_index": 0,
          "intercepts": [
            -5.071324348887,
            0.0
          ],
          "levels": [
            "focal",
            "other_1"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0
            ],
            [
              -0.9,
              0.0
            ],
            [
              0.65,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            0.0,
            0.0
          ],
          "levels": [
            "level_0",
            "level_1"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0
            ],
            [
              0.839094611045,
              -0.839094611045
            ],
            [
              0.325453888768,
              -0.325453888768
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            0.0,
            0.0
          ],
          "levels": [
            "level_0",
            "level_1"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              -0.606859120465
            ],
            [
              0.0,
              0.0
            ],
            [
              0.664621702857,
              -0.664621702857
            ]
          ]
        }
      ],
      "pair_key": "core_n2000_q010_k2",
      "variant": "homogeneous"
    },
    {
      "anchor_index": 0,
      "cardinalities": [
        5,
        5,
        5
      ],
      "copula_rho": 0.35,
      "design": "core",
      "focal_prevalence": 0.01,
      "n_test": 20000,
      "n_train": 2000,
      "n_validation": 500,
      "name": "core_n2000_q010_k5",
      "outcomes": [
        {
          "focal_class_index": 0,
          "intercepts": [
            -3.685029987767,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "levels": [
            "focal",
            "other_1",
            "other_2",
            "other_3",
            "other_4"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              -0.9,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.65,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            7.1513196e-05,
            0.000770550074,
            0.00040446016,
            -0.000520446354,
            -0.000726077075
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.839094611045,
              -0.050230546965,
              -0.870138796344,
              -0.487544804105,
              0.568819536368
            ],
            [
              0.325453888768,
              0.898597180138,
              0.229909710751,
              -0.75650516455,
              -0.697455615108
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            -0.000776956154,
            -0.000104550295,
            0.000712610457,
            0.000544792259,
            -0.000375896266
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              0.819622582788,
              -0.100304506355,
              -0.881614176941,
              -0.444563019958
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.664621702857,
              -0.371777919979,
              -0.89439309367,
              -0.180987411213,
              0.782536722005
            ]
          ]
        }
      ],
      "pair_key": "core_n2000_q010_k5",
      "variant": "homogeneous"
    },
    {
      "anchor_index": 0,
      "cardinalities": [
        20,
        20,
        20
      ],
      "copula_rho": 0.35,
      "design": "core",
      "focal_prevalence": 0.01,
      "n_test": 20000,
      "n_train": 2000,
      "n_validation": 500,
      "name": "core_n2000_q010_k20",
      "outcomes": [
        {
          "focal_class_index": 0,
          "intercepts": [
            -2.12688536972,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "levels": [
            "focal",
            "other_1",
            "other_2",
            "other_3",
            "other_4",
            "other_5",
            "other_6",
            "other_7",
            "other_8",
            "other_9",
            "other_10",
            "other_11",
            "other_12",
            "other_13",
            "other_14",
            "other_15",
            "other_16",
            "other_17",
            "other_18",
            "other_19"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              -0.9,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.65,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585,
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585,
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585,
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4",
            "level_5",
            "level_6",
            "level_7",
            "level_8",
            "level_9",
            "level_10",
            "level_11",
            "level_12",
            "level_13",
            "level_14",
            "level_15",
            "level_16",
            "level_17",
            "level_18",
            "level_19"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.839094611045,
              0.697455615108,
              0.487544804105,
              0.229909710751,
              -0.050230546965,
              -0.325453888768,
              -0.568819536368,
              -0.75650516455,
              -0.870138796344,
              -0.898597180138,
              -0.839094611045,
              -0.697455615108,
              -0.487544804105,
              -0.229909710751,
              0.050230546965,
              0.325453888768,
              0.568819536368,
              0.75650516455,
              0.870138796344,
              0.898597180138
            ],
            [
              0.325453888768,
              0.568819536368,
              0.75650516455,
              0.870138796344,
              0.898597180138,
              0.839094611045,
              0.697455615108,
              0.487544804105,
              0.229909710751,
              -0.050230546965,
              -0.325453888768,
              -0.568819536368,
              -0.75650516455,
              -0.870138796344,
              -0.898597180138,
              -0.839094611045,
              -0.697455615108,
              -0.487544804105,
              -0.229909710751,
              0.050230546965
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05,
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05,
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05,
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4",
            "level_5",
            "level_6",
            "level_7",
            "level_8",
            "level_9",
            "level_10",
            "level_11",
            "level_12",
            "level_13",
            "level_14",
            "level_15",
            "level_16",
            "level_17",
            "level_18",
            "level_19"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              0.782536722005,
              0.881614176941,
              0.89439309367,
              0.819622582788,
              0.664621702857,
              0.444563019958,
              0.180987411213,
              -0.100304506355,
              -0.371777919979,
              -0.606859120465,
              -0.782536722005,
              -0.881614176941,
              -0.89439309367,
              -0.819622582788,
              -0.664621702857,
              -0.444563019958,
              -0.180987411213,
              0.100304506355,
              0.371777919979
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.664621702857,
              0.444563019958,
              0.180987411213,
              -0.100304506355,
              -0.371777919979,
              -0.606859120465,
              -0.782536722005,
              -0.881614176941,
              -0.89439309367,
              -0.819622582788,
              -0.664621702857,
              -0.444563019958,
              -0.180987411213,
              0.100304506355,
              0.371777919979,
              0.606859120465,
              0.782536722005,
              0.881614176941,
              0.89439309367,
              0.819622582788
            ]
          ]
        }
      ],
      "pair_key": "core_n2000_q010_k20",
      "variant": "homogeneous"
    },
    {
      "anchor_index": 0,
      "cardinalities": [
        2,
        2,
        2
      ],
      "copula_rho": 0.35,
      "design": "core",
      "focal_prevalence": 0.05,
      "n_test": 20000,
      "n_train": 2000,
      "n_validation": 500,
      "name": "core_n2000_q050_k2",
      "outcomes": [
        {
          "focal_class_index": 0,
          "intercepts": [
            -3.374725967262,
            0.0
          ],
          "levels": [
            "focal",
            "other_1"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0
            ],
            [
              -0.9,
              0.0
            ],
            [
              0.65,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            0.0,
            0.0
          ],
          "levels": [
            "level_0",
            "level_1"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0
            ],
            [
              0.839094611045,
              -0.839094611045
            ],
            [
              0.325453888768,
              -0.325453888768
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            0.0,
            0.0
          ],
          "levels": [
            "level_0",
            "level_1"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              -0.606859120465
            ],
            [
              0.0,
              0.0
            ],
            [
              0.664621702857,
              -0.664621702857
            ]
          ]
        }
      ],
      "pair_key": "core_n2000_q050_k2",
      "variant": "homogeneous"
    },
    {
      "anchor_index": 0,
      "cardinalities": [
        5,
        5,
        5
      ],
      "copula_rho": 0.35,
      "design": "core",
      "focal_prevalence": 0.05,
      "n_test": 20000,
      "n_train": 2000,
      "n_validation": 500,
      "name": "core_n2000_q050_k5",
      "outcomes": [
        {
          "focal_class_index": 0,
          "intercepts": [
            -1.988431606142,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "levels": [
            "focal",
            "other_1",
            "other_2",
            "other_3",
            "other_4"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              -0.9,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.65,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            7.1513196e-05,
            0.000770550074,
            0.00040446016,
            -0.000520446354,
            -0.000726077075
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.839094611045,
              -0.050230546965,
              -0.870138796344,
              -0.487544804105,
              0.568819536368
            ],
            [
              0.325453888768,
              0.898597180138,
              0.229909710751,
              -0.75650516455,
              -0.697455615108
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            -0.000776956154,
            -0.000104550295,
            0.000712610457,
            0.000544792259,
            -0.000375896266
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              0.819622582788,
              -0.100304506355,
              -0.881614176941,
              -0.444563019958
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.664621702857,
              -0.371777919979,
              -0.89439309367,
              -0.180987411213,
              0.782536722005
            ]
          ]
        }
      ],
      "pair_key": "core_n2000_q050_k5",
      "variant": "homogeneous"
    },
    {
      "anchor_index": 0,
      "cardinalities": [
        20,
        20,
        20
      ],
      "copula_rho": 0.35,
      "design": "core",
      "focal_prevalence": 0.05,
      "n_test": 20000,
      "n_train": 2000,
      "n_validation": 500,
      "name": "core_n2000_q050_k20",
      "outcomes": [
        {
          "focal_class_index": 0,
          "intercepts": [
            -0.430286988095,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "levels": [
            "focal",
            "other_1",
            "other_2",
            "other_3",
            "other_4",
            "other_5",
            "other_6",
            "other_7",
            "other_8",
            "other_9",
            "other_10",
            "other_11",
            "other_12",
            "other_13",
            "other_14",
            "other_15",
            "other_16",
            "other_17",
            "other_18",
            "other_19"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              -0.9,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.65,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585,
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585,
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585,
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4",
            "level_5",
            "level_6",
            "level_7",
            "level_8",
            "level_9",
            "level_10",
            "level_11",
            "level_12",
            "level_13",
            "level_14",
            "level_15",
            "level_16",
            "level_17",
            "level_18",
            "level_19"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.839094611045,
              0.697455615108,
              0.487544804105,
              0.229909710751,
              -0.050230546965,
              -0.325453888768,
              -0.568819536368,
              -0.75650516455,
              -0.870138796344,
              -0.898597180138,
              -0.839094611045,
              -0.697455615108,
              -0.487544804105,
              -0.229909710751,
              0.050230546965,
              0.325453888768,
              0.568819536368,
              0.75650516455,
              0.870138796344,
              0.898597180138
            ],
            [
              0.325453888768,
              0.568819536368,
              0.75650516455,
              0.870138796344,
              0.898597180138,
              0.839094611045,
              0.697455615108,
              0.487544804105,
              0.229909710751,
              -0.050230546965,
              -0.325453888768,
              -0.568819536368,
              -0.75650516455,
              -0.870138796344,
              -0.898597180138,
              -0.839094611045,
              -0.697455615108,
              -0.487544804105,
              -0.229909710751,
              0.050230546965
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05,
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05,
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05,
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4",
            "level_5",
            "level_6",
            "level_7",
            "level_8",
            "level_9",
            "level_10",
            "level_11",
            "level_12",
            "level_13",
            "level_14",
            "level_15",
            "level_16",
            "level_17",
            "level_18",
            "level_19"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              0.782536722005,
              0.881614176941,
              0.89439309367,
              0.819622582788,
              0.664621702857,
              0.444563019958,
              0.180987411213,
              -0.100304506355,
              -0.371777919979,
              -0.606859120465,
              -0.782536722005,
              -0.881614176941,
              -0.89439309367,
              -0.819622582788,
              -0.664621702857,
              -0.444563019958,
              -0.180987411213,
              0.100304506355,
              0.371777919979
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.664621702857,
              0.444563019958,
              0.180987411213,
              -0.100304506355,
              -0.371777919979,
              -0.606859120465,
              -0.782536722005,
              -0.881614176941,
              -0.89439309367,
              -0.819622582788,
              -0.664621702857,
              -0.444563019958,
              -0.180987411213,
              0.100304506355,
              0.371777919979,
              0.606859120465,
              0.782536722005,
              0.881614176941,
              0.89439309367,
              0.819622582788
            ]
          ]
        }
      ],
      "pair_key": "core_n2000_q050_k20",
      "variant": "homogeneous"
    },
    {
      "anchor_index": 0,
      "cardinalities": [
        2,
        2,
        2
      ],
      "copula_rho": 0.35,
      "design": "core",
      "focal_prevalence": 0.2,
      "n_test": 20000,
      "n_train": 2000,
      "n_validation": 500,
      "name": "core_n2000_q200_k2",
      "outcomes": [
        {
          "focal_class_index": 0,
          "intercepts": [
            -1.661789569696,
            0.0
          ],
          "levels": [
            "focal",
            "other_1"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0
            ],
            [
              -0.9,
              0.0
            ],
            [
              0.65,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            0.0,
            0.0
          ],
          "levels": [
            "level_0",
            "level_1"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0
            ],
            [
              0.839094611045,
              -0.839094611045
            ],
            [
              0.325453888768,
              -0.325453888768
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            0.0,
            0.0
          ],
          "levels": [
            "level_0",
            "level_1"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              -0.606859120465
            ],
            [
              0.0,
              0.0
            ],
            [
              0.664621702857,
              -0.664621702857
            ]
          ]
        }
      ],
      "pair_key": "core_n2000_q200_k2",
      "variant": "homogeneous"
    },
    {
      "anchor_index": 0,
      "cardinalities": [
        5,
        5,
        5
      ],
      "copula_rho": 0.35,
      "design": "core",
      "focal_prevalence": 0.2,
      "n_test": 20000,
      "n_train": 2000,
      "n_validation": 500,
      "name": "core_n2000_q200_k5",
      "outcomes": [
        {
          "focal_class_index": 0,
          "intercepts": [
            -0.275495208576,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "levels": [
            "focal",
            "other_1",
            "other_2",
            "other_3",
            "other_4"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              -0.9,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.65,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            7.1513196e-05,
            0.000770550074,
            0.00040446016,
            -0.000520446354,
            -0.000726077075
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.839094611045,
              -0.050230546965,
              -0.870138796344,
              -0.487544804105,
              0.568819536368
            ],
            [
              0.325453888768,
              0.898597180138,
              0.229909710751,
              -0.75650516455,
              -0.697455615108
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            -0.000776956154,
            -0.000104550295,
            0.000712610457,
            0.000544792259,
            -0.000375896266
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              0.819622582788,
              -0.100304506355,
              -0.881614176941,
              -0.444563019958
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.664621702857,
              -0.371777919979,
              -0.89439309367,
              -0.180987411213,
              0.782536722005
            ]
          ]
        }
      ],
      "pair_key": "core_n2000_q200_k5",
      "variant": "homogeneous"
    },
    {
      "anchor_index": 0,
      "cardinalities": [
        20,
        20,
        20
      ],
      "copula_rho": 0.35,
      "design": "core",
      "focal_prevalence": 0.2,
      "n_test": 20000,
      "n_train": 2000,
      "n_validation": 500,
      "name": "core_n2000_q200_k20",
      "outcomes": [
        {
          "focal_class_index": 0,
          "intercepts": [
            1.282649409471,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "levels": [
            "focal",
            "other_1",
            "other_2",
            "other_3",
            "other_4",
            "other_5",
            "other_6",
            "other_7",
            "other_8",
            "other_9",
            "other_10",
            "other_11",
            "other_12",
            "other_13",
            "other_14",
            "other_15",
            "other_16",
            "other_17",
            "other_18",
            "other_19"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              -0.9,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.65,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585,
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585,
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585,
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4",
            "level_5",
            "level_6",
            "level_7",
            "level_8",
            "level_9",
            "level_10",
            "level_11",
            "level_12",
            "level_13",
            "level_14",
            "level_15",
            "level_16",
            "level_17",
            "level_18",
            "level_19"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.839094611045,
              0.697455615108,
              0.487544804105,
              0.229909710751,
              -0.050230546965,
              -0.325453888768,
              -0.568819536368,
              -0.75650516455,
              -0.870138796344,
              -0.898597180138,
              -0.839094611045,
              -0.697455615108,
              -0.487544804105,
              -0.229909710751,
              0.050230546965,
              0.325453888768,
              0.568819536368,
              0.75650516455,
              0.870138796344,
              0.898597180138
            ],
            [
              0.325453888768,
              0.568819536368,
              0.75650516455,
              0.870138796344,
              0.898597180138,
              0.839094611045,
              0.697455615108,
              0.487544804105,
              0.229909710751,
              -0.050230546965,
              -0.325453888768,
              -0.568819536368,
              -0.75650516455,
              -0.870138796344,
              -0.898597180138,
              -0.839094611045,
              -0.697455615108,
              -0.487544804105,
              -0.229909710751,
              0.050230546965
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05,
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05,
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05,
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4",
            "level_5",
            "level_6",
            "level_7",
            "level_8",
            "level_9",
            "level_10",
            "level_11",
            "level_12",
            "level_13",
            "level_14",
            "level_15",
            "level_16",
            "level_17",
            "level_18",
            "level_19"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              0.782536722005,
              0.881614176941,
              0.89439309367,
              0.819622582788,
              0.664621702857,
              0.444563019958,
              0.180987411213,
              -0.100304506355,
              -0.371777919979,
              -0.606859120465,
              -0.782536722005,
              -0.881614176941,
              -0.89439309367,
              -0.819622582788,
              -0.664621702857,
              -0.444563019958,
              -0.180987411213,
              0.100304506355,
              0.371777919979
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.664621702857,
              0.444563019958,
              0.180987411213,
              -0.100304506355,
              -0.371777919979,
              -0.606859120465,
              -0.782536722005,
              -0.881614176941,
              -0.89439309367,
              -0.819622582788,
              -0.664621702857,
              -0.444563019958,
              -0.180987411213,
              0.100304506355,
              0.371777919979,
              0.606859120465,
              0.782536722005,
              0.881614176941,
              0.89439309367,
              0.819622582788
            ]
          ]
        }
      ],
      "pair_key": "core_n2000_q200_k20",
      "variant": "homogeneous"
    },
    {
      "anchor_index": 0,
      "cardinalities": [
        2,
        2,
        2
      ],
      "copula_rho": 0.35,
      "design": "core",
      "focal_prevalence": 0.01,
      "n_test": 20000,
      "n_train": 8000,
      "n_validation": 2000,
      "name": "core_n8000_q010_k2",
      "outcomes": [
        {
          "focal_class_index": 0,
          "intercepts": [
            -5.071324348887,
            0.0
          ],
          "levels": [
            "focal",
            "other_1"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0
            ],
            [
              -0.9,
              0.0
            ],
            [
              0.65,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            0.0,
            0.0
          ],
          "levels": [
            "level_0",
            "level_1"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0
            ],
            [
              0.839094611045,
              -0.839094611045
            ],
            [
              0.325453888768,
              -0.325453888768
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            0.0,
            0.0
          ],
          "levels": [
            "level_0",
            "level_1"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              -0.606859120465
            ],
            [
              0.0,
              0.0
            ],
            [
              0.664621702857,
              -0.664621702857
            ]
          ]
        }
      ],
      "pair_key": "core_n8000_q010_k2",
      "variant": "homogeneous"
    },
    {
      "anchor_index": 0,
      "cardinalities": [
        5,
        5,
        5
      ],
      "copula_rho": 0.35,
      "design": "core",
      "focal_prevalence": 0.01,
      "n_test": 20000,
      "n_train": 8000,
      "n_validation": 2000,
      "name": "core_n8000_q010_k5",
      "outcomes": [
        {
          "focal_class_index": 0,
          "intercepts": [
            -3.685029987767,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "levels": [
            "focal",
            "other_1",
            "other_2",
            "other_3",
            "other_4"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              -0.9,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.65,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            7.1513196e-05,
            0.000770550074,
            0.00040446016,
            -0.000520446354,
            -0.000726077075
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.839094611045,
              -0.050230546965,
              -0.870138796344,
              -0.487544804105,
              0.568819536368
            ],
            [
              0.325453888768,
              0.898597180138,
              0.229909710751,
              -0.75650516455,
              -0.697455615108
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            -0.000776956154,
            -0.000104550295,
            0.000712610457,
            0.000544792259,
            -0.000375896266
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              0.819622582788,
              -0.100304506355,
              -0.881614176941,
              -0.444563019958
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.664621702857,
              -0.371777919979,
              -0.89439309367,
              -0.180987411213,
              0.782536722005
            ]
          ]
        }
      ],
      "pair_key": "core_n8000_q010_k5",
      "variant": "homogeneous"
    },
    {
      "anchor_index": 0,
      "cardinalities": [
        20,
        20,
        20
      ],
      "copula_rho": 0.35,
      "design": "core",
      "focal_prevalence": 0.01,
      "n_test": 20000,
      "n_train": 8000,
      "n_validation": 2000,
      "name": "core_n8000_q010_k20",
      "outcomes": [
        {
          "focal_class_index": 0,
          "intercepts": [
            -2.12688536972,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "levels": [
            "focal",
            "other_1",
            "other_2",
            "other_3",
            "other_4",
            "other_5",
            "other_6",
            "other_7",
            "other_8",
            "other_9",
            "other_10",
            "other_11",
            "other_12",
            "other_13",
            "other_14",
            "other_15",
            "other_16",
            "other_17",
            "other_18",
            "other_19"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              -0.9,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.65,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585,
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585,
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585,
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4",
            "level_5",
            "level_6",
            "level_7",
            "level_8",
            "level_9",
            "level_10",
            "level_11",
            "level_12",
            "level_13",
            "level_14",
            "level_15",
            "level_16",
            "level_17",
            "level_18",
            "level_19"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.839094611045,
              0.697455615108,
              0.487544804105,
              0.229909710751,
              -0.050230546965,
              -0.325453888768,
              -0.568819536368,
              -0.75650516455,
              -0.870138796344,
              -0.898597180138,
              -0.839094611045,
              -0.697455615108,
              -0.487544804105,
              -0.229909710751,
              0.050230546965,
              0.325453888768,
              0.568819536368,
              0.75650516455,
              0.870138796344,
              0.898597180138
            ],
            [
              0.325453888768,
              0.568819536368,
              0.75650516455,
              0.870138796344,
              0.898597180138,
              0.839094611045,
              0.697455615108,
              0.487544804105,
              0.229909710751,
              -0.050230546965,
              -0.325453888768,
              -0.568819536368,
              -0.75650516455,
              -0.870138796344,
              -0.898597180138,
              -0.839094611045,
              -0.697455615108,
              -0.487544804105,
              -0.229909710751,
              0.050230546965
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05,
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05,
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05,
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4",
            "level_5",
            "level_6",
            "level_7",
            "level_8",
            "level_9",
            "level_10",
            "level_11",
            "level_12",
            "level_13",
            "level_14",
            "level_15",
            "level_16",
            "level_17",
            "level_18",
            "level_19"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              0.782536722005,
              0.881614176941,
              0.89439309367,
              0.819622582788,
              0.664621702857,
              0.444563019958,
              0.180987411213,
              -0.100304506355,
              -0.371777919979,
              -0.606859120465,
              -0.782536722005,
              -0.881614176941,
              -0.89439309367,
              -0.819622582788,
              -0.664621702857,
              -0.444563019958,
              -0.180987411213,
              0.100304506355,
              0.371777919979
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.664621702857,
              0.444563019958,
              0.180987411213,
              -0.100304506355,
              -0.371777919979,
              -0.606859120465,
              -0.782536722005,
              -0.881614176941,
              -0.89439309367,
              -0.819622582788,
              -0.664621702857,
              -0.444563019958,
              -0.180987411213,
              0.100304506355,
              0.371777919979,
              0.606859120465,
              0.782536722005,
              0.881614176941,
              0.89439309367,
              0.819622582788
            ]
          ]
        }
      ],
      "pair_key": "core_n8000_q010_k20",
      "variant": "homogeneous"
    },
    {
      "anchor_index": 0,
      "cardinalities": [
        2,
        2,
        2
      ],
      "copula_rho": 0.35,
      "design": "core",
      "focal_prevalence": 0.05,
      "n_test": 20000,
      "n_train": 8000,
      "n_validation": 2000,
      "name": "core_n8000_q050_k2",
      "outcomes": [
        {
          "focal_class_index": 0,
          "intercepts": [
            -3.374725967262,
            0.0
          ],
          "levels": [
            "focal",
            "other_1"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0
            ],
            [
              -0.9,
              0.0
            ],
            [
              0.65,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            0.0,
            0.0
          ],
          "levels": [
            "level_0",
            "level_1"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0
            ],
            [
              0.839094611045,
              -0.839094611045
            ],
            [
              0.325453888768,
              -0.325453888768
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            0.0,
            0.0
          ],
          "levels": [
            "level_0",
            "level_1"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              -0.606859120465
            ],
            [
              0.0,
              0.0
            ],
            [
              0.664621702857,
              -0.664621702857
            ]
          ]
        }
      ],
      "pair_key": "core_n8000_q050_k2",
      "variant": "homogeneous"
    },
    {
      "anchor_index": 0,
      "cardinalities": [
        5,
        5,
        5
      ],
      "copula_rho": 0.35,
      "design": "core",
      "focal_prevalence": 0.05,
      "n_test": 20000,
      "n_train": 8000,
      "n_validation": 2000,
      "name": "core_n8000_q050_k5",
      "outcomes": [
        {
          "focal_class_index": 0,
          "intercepts": [
            -1.988431606142,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "levels": [
            "focal",
            "other_1",
            "other_2",
            "other_3",
            "other_4"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              -0.9,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.65,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            7.1513196e-05,
            0.000770550074,
            0.00040446016,
            -0.000520446354,
            -0.000726077075
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.839094611045,
              -0.050230546965,
              -0.870138796344,
              -0.487544804105,
              0.568819536368
            ],
            [
              0.325453888768,
              0.898597180138,
              0.229909710751,
              -0.75650516455,
              -0.697455615108
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            -0.000776956154,
            -0.000104550295,
            0.000712610457,
            0.000544792259,
            -0.000375896266
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              0.819622582788,
              -0.100304506355,
              -0.881614176941,
              -0.444563019958
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.664621702857,
              -0.371777919979,
              -0.89439309367,
              -0.180987411213,
              0.782536722005
            ]
          ]
        }
      ],
      "pair_key": "core_n8000_q050_k5",
      "variant": "homogeneous"
    },
    {
      "anchor_index": 0,
      "cardinalities": [
        20,
        20,
        20
      ],
      "copula_rho": 0.35,
      "design": "core",
      "focal_prevalence": 0.05,
      "n_test": 20000,
      "n_train": 8000,
      "n_validation": 2000,
      "name": "core_n8000_q050_k20",
      "outcomes": [
        {
          "focal_class_index": 0,
          "intercepts": [
            -0.430286988095,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "levels": [
            "focal",
            "other_1",
            "other_2",
            "other_3",
            "other_4",
            "other_5",
            "other_6",
            "other_7",
            "other_8",
            "other_9",
            "other_10",
            "other_11",
            "other_12",
            "other_13",
            "other_14",
            "other_15",
            "other_16",
            "other_17",
            "other_18",
            "other_19"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              -0.9,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.65,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585,
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585,
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585,
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4",
            "level_5",
            "level_6",
            "level_7",
            "level_8",
            "level_9",
            "level_10",
            "level_11",
            "level_12",
            "level_13",
            "level_14",
            "level_15",
            "level_16",
            "level_17",
            "level_18",
            "level_19"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.839094611045,
              0.697455615108,
              0.487544804105,
              0.229909710751,
              -0.050230546965,
              -0.325453888768,
              -0.568819536368,
              -0.75650516455,
              -0.870138796344,
              -0.898597180138,
              -0.839094611045,
              -0.697455615108,
              -0.487544804105,
              -0.229909710751,
              0.050230546965,
              0.325453888768,
              0.568819536368,
              0.75650516455,
              0.870138796344,
              0.898597180138
            ],
            [
              0.325453888768,
              0.568819536368,
              0.75650516455,
              0.870138796344,
              0.898597180138,
              0.839094611045,
              0.697455615108,
              0.487544804105,
              0.229909710751,
              -0.050230546965,
              -0.325453888768,
              -0.568819536368,
              -0.75650516455,
              -0.870138796344,
              -0.898597180138,
              -0.839094611045,
              -0.697455615108,
              -0.487544804105,
              -0.229909710751,
              0.050230546965
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05,
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05,
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05,
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4",
            "level_5",
            "level_6",
            "level_7",
            "level_8",
            "level_9",
            "level_10",
            "level_11",
            "level_12",
            "level_13",
            "level_14",
            "level_15",
            "level_16",
            "level_17",
            "level_18",
            "level_19"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              0.782536722005,
              0.881614176941,
              0.89439309367,
              0.819622582788,
              0.664621702857,
              0.444563019958,
              0.180987411213,
              -0.100304506355,
              -0.371777919979,
              -0.606859120465,
              -0.782536722005,
              -0.881614176941,
              -0.89439309367,
              -0.819622582788,
              -0.664621702857,
              -0.444563019958,
              -0.180987411213,
              0.100304506355,
              0.371777919979
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.664621702857,
              0.444563019958,
              0.180987411213,
              -0.100304506355,
              -0.371777919979,
              -0.606859120465,
              -0.782536722005,
              -0.881614176941,
              -0.89439309367,
              -0.819622582788,
              -0.664621702857,
              -0.444563019958,
              -0.180987411213,
              0.100304506355,
              0.371777919979,
              0.606859120465,
              0.782536722005,
              0.881614176941,
              0.89439309367,
              0.819622582788
            ]
          ]
        }
      ],
      "pair_key": "core_n8000_q050_k20",
      "variant": "homogeneous"
    },
    {
      "anchor_index": 0,
      "cardinalities": [
        2,
        2,
        2
      ],
      "copula_rho": 0.35,
      "design": "core",
      "focal_prevalence": 0.2,
      "n_test": 20000,
      "n_train": 8000,
      "n_validation": 2000,
      "name": "core_n8000_q200_k2",
      "outcomes": [
        {
          "focal_class_index": 0,
          "intercepts": [
            -1.661789569696,
            0.0
          ],
          "levels": [
            "focal",
            "other_1"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0
            ],
            [
              -0.9,
              0.0
            ],
            [
              0.65,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            0.0,
            0.0
          ],
          "levels": [
            "level_0",
            "level_1"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0
            ],
            [
              0.839094611045,
              -0.839094611045
            ],
            [
              0.325453888768,
              -0.325453888768
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            0.0,
            0.0
          ],
          "levels": [
            "level_0",
            "level_1"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              -0.606859120465
            ],
            [
              0.0,
              0.0
            ],
            [
              0.664621702857,
              -0.664621702857
            ]
          ]
        }
      ],
      "pair_key": "core_n8000_q200_k2",
      "variant": "homogeneous"
    },
    {
      "anchor_index": 0,
      "cardinalities": [
        5,
        5,
        5
      ],
      "copula_rho": 0.35,
      "design": "core",
      "focal_prevalence": 0.2,
      "n_test": 20000,
      "n_train": 8000,
      "n_validation": 2000,
      "name": "core_n8000_q200_k5",
      "outcomes": [
        {
          "focal_class_index": 0,
          "intercepts": [
            -0.275495208576,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "levels": [
            "focal",
            "other_1",
            "other_2",
            "other_3",
            "other_4"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              -0.9,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.65,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            7.1513196e-05,
            0.000770550074,
            0.00040446016,
            -0.000520446354,
            -0.000726077075
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.839094611045,
              -0.050230546965,
              -0.870138796344,
              -0.487544804105,
              0.568819536368
            ],
            [
              0.325453888768,
              0.898597180138,
              0.229909710751,
              -0.75650516455,
              -0.697455615108
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            -0.000776956154,
            -0.000104550295,
            0.000712610457,
            0.000544792259,
            -0.000375896266
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              0.819622582788,
              -0.100304506355,
              -0.881614176941,
              -0.444563019958
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.664621702857,
              -0.371777919979,
              -0.89439309367,
              -0.180987411213,
              0.782536722005
            ]
          ]
        }
      ],
      "pair_key": "core_n8000_q200_k5",
      "variant": "homogeneous"
    },
    {
      "anchor_index": 0,
      "cardinalities": [
        20,
        20,
        20
      ],
      "copula_rho": 0.35,
      "design": "core",
      "focal_prevalence": 0.2,
      "n_test": 20000,
      "n_train": 8000,
      "n_validation": 2000,
      "name": "core_n8000_q200_k20",
      "outcomes": [
        {
          "focal_class_index": 0,
          "intercepts": [
            1.282649409471,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "levels": [
            "focal",
            "other_1",
            "other_2",
            "other_3",
            "other_4",
            "other_5",
            "other_6",
            "other_7",
            "other_8",
            "other_9",
            "other_10",
            "other_11",
            "other_12",
            "other_13",
            "other_14",
            "other_15",
            "other_16",
            "other_17",
            "other_18",
            "other_19"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              -0.9,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.65,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585,
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585,
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585,
            6.6713895e-05,
            -0.000677315507,
            -0.000485492362,
            0.000377298125,
            0.00071879585
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4",
            "level_5",
            "level_6",
            "level_7",
            "level_8",
            "level_9",
            "level_10",
            "level_11",
            "level_12",
            "level_13",
            "level_14",
            "level_15",
            "level_16",
            "level_17",
            "level_18",
            "level_19"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.839094611045,
              0.697455615108,
              0.487544804105,
              0.229909710751,
              -0.050230546965,
              -0.325453888768,
              -0.568819536368,
              -0.75650516455,
              -0.870138796344,
              -0.898597180138,
              -0.839094611045,
              -0.697455615108,
              -0.487544804105,
              -0.229909710751,
              0.050230546965,
              0.325453888768,
              0.568819536368,
              0.75650516455,
              0.870138796344,
              0.898597180138
            ],
            [
              0.325453888768,
              0.568819536368,
              0.75650516455,
              0.870138796344,
              0.898597180138,
              0.839094611045,
              0.697455615108,
              0.487544804105,
              0.229909710751,
              -0.050230546965,
              -0.325453888768,
              -0.568819536368,
              -0.75650516455,
              -0.870138796344,
              -0.898597180138,
              -0.839094611045,
              -0.697455615108,
              -0.487544804105,
              -0.229909710751,
              0.050230546965
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05,
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05,
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05,
            -0.000724778302,
            -0.000350648919,
            0.000508203815,
            0.000664748593,
            -9.7525186e-05
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4",
            "level_5",
            "level_6",
            "level_7",
            "level_8",
            "level_9",
            "level_10",
            "level_11",
            "level_12",
            "level_13",
            "level_14",
            "level_15",
            "level_16",
            "level_17",
            "level_18",
            "level_19"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              0.782536722005,
              0.881614176941,
              0.89439309367,
              0.819622582788,
              0.664621702857,
              0.444563019958,
              0.180987411213,
              -0.100304506355,
              -0.371777919979,
              -0.606859120465,
              -0.782536722005,
              -0.881614176941,
              -0.89439309367,
              -0.819622582788,
              -0.664621702857,
              -0.444563019958,
              -0.180987411213,
              0.100304506355,
              0.371777919979
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.664621702857,
              0.444563019958,
              0.180987411213,
              -0.100304506355,
              -0.371777919979,
              -0.606859120465,
              -0.782536722005,
              -0.881614176941,
              -0.89439309367,
              -0.819622582788,
              -0.664621702857,
              -0.444563019958,
              -0.180987411213,
              0.100304506355,
              0.371777919979,
              0.606859120465,
              0.782536722005,
              0.881614176941,
              0.89439309367,
              0.819622582788
            ]
          ]
        }
      ],
      "pair_key": "core_n8000_q200_k20",
      "variant": "homogeneous"
    },
    {
      "anchor_index": 2,
      "cardinalities": [
        5,
        5,
        5,
        5
      ],
      "copula_rho": 0.35,
      "design": "heterogeneity",
      "focal_prevalence": 0.05,
      "n_test": 20000,
      "n_train": 500,
      "n_validation": 250,
      "name": "heterogeneity_homogeneous_n500",
      "outcomes": [
        {
          "focal_class_index": null,
          "intercepts": [
            7.1513196e-05,
            0.000770550074,
            0.00040446016,
            -0.000520446354,
            -0.000726077075
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.839094611045,
              -0.050230546965,
              -0.870138796344,
              -0.487544804105,
              0.568819536368
            ],
            [
              0.325453888768,
              0.898597180138,
              0.229909710751,
              -0.75650516455,
              -0.697455615108
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            -0.000776956154,
            -0.000104550295,
            0.000712610457,
            0.000544792259,
            -0.000375896266
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              0.819622582788,
              -0.100304506355,
              -0.881614176941,
              -0.444563019958
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.664621702857,
              -0.371777919979,
              -0.89439309367,
              -0.180987411213,
              0.782536722005
            ]
          ]
        },
        {
          "focal_class_index": 0,
          "intercepts": [
            -1.988431606142,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "levels": [
            "focal",
            "other_1",
            "other_2",
            "other_3",
            "other_4"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              -0.9,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.65,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            -0.00021266021,
            -0.00078925009,
            -0.000275401641,
            0.000619254041,
            0.0006580579
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4"
          ],
          "name": "context_3",
          "role": "context",
          "x_weights": [
            [
              0.400195365068,
              -0.643006895612,
              -0.797595481556,
              0.150065778737,
              0.890341233364
            ],
            [
              0.806128817112,
              0.629715913882,
              -0.416942979077,
              -0.887400846322,
              -0.131500905596
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        }
      ],
      "pair_key": "heterogeneity_n500",
      "variant": "homogeneous"
    },
    {
      "anchor_index": 2,
      "cardinalities": [
        2,
        3,
        5,
        10
      ],
      "copula_rho": 0.35,
      "design": "heterogeneity",
      "focal_prevalence": 0.05,
      "n_test": 20000,
      "n_train": 500,
      "n_validation": 250,
      "name": "heterogeneity_heterogeneous_n500",
      "outcomes": [
        {
          "focal_class_index": null,
          "intercepts": [
            0.0,
            0.0
          ],
          "levels": [
            "level_0",
            "level_1"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0
            ],
            [
              0.839094611045,
              -0.839094611045
            ],
            [
              0.325453888768,
              -0.325453888768
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            0.002095325626,
            -0.000713272615,
            -0.001382053011
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              0.272149718348,
              -0.879008838813
            ],
            [
              0.0,
              0.0,
              0.0
            ],
            [
              0.664621702857,
              -0.85786626627,
              0.193244563413
            ]
          ]
        },
        {
          "focal_class_index": 0,
          "intercepts": [
            -1.988431606142,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "levels": [
            "focal",
            "other_1",
            "other_2",
            "other_3",
            "other_4"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              -0.9,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.65,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            -0.000198375793,
            0.000577663966,
            -0.000736245575,
            0.000613861186,
            -0.000256903784,
            -0.000198375793,
            0.000577663966,
            -0.000736245575,
            0.000613861186,
            -0.000256903784
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4",
            "level_5",
            "level_6",
            "level_7",
            "level_8",
            "level_9"
          ],
          "name": "context_3",
          "role": "context",
          "x_weights": [
            [
              0.400195365068,
              -0.150065778737,
              -0.643006895612,
              -0.890341233364,
              -0.797595481556,
              -0.400195365068,
              0.150065778737,
              0.643006895612,
              0.890341233364,
              0.797595481556
            ],
            [
              0.806128817112,
              0.887400846322,
              0.629715913882,
              0.131500905596,
              -0.416942979077,
              -0.806128817112,
              -0.887400846322,
              -0.629715913882,
              -0.131500905596,
              0.416942979077
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        }
      ],
      "pair_key": "heterogeneity_n500",
      "variant": "heterogeneous"
    },
    {
      "anchor_index": 2,
      "cardinalities": [
        5,
        5,
        5,
        5
      ],
      "copula_rho": 0.35,
      "design": "heterogeneity",
      "focal_prevalence": 0.05,
      "n_test": 20000,
      "n_train": 2000,
      "n_validation": 500,
      "name": "heterogeneity_homogeneous_n2000",
      "outcomes": [
        {
          "focal_class_index": null,
          "intercepts": [
            7.1513196e-05,
            0.000770550074,
            0.00040446016,
            -0.000520446354,
            -0.000726077075
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.839094611045,
              -0.050230546965,
              -0.870138796344,
              -0.487544804105,
              0.568819536368
            ],
            [
              0.325453888768,
              0.898597180138,
              0.229909710751,
              -0.75650516455,
              -0.697455615108
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            -0.000776956154,
            -0.000104550295,
            0.000712610457,
            0.000544792259,
            -0.000375896266
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              0.819622582788,
              -0.100304506355,
              -0.881614176941,
              -0.444563019958
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.664621702857,
              -0.371777919979,
              -0.89439309367,
              -0.180987411213,
              0.782536722005
            ]
          ]
        },
        {
          "focal_class_index": 0,
          "intercepts": [
            -1.988431606142,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "levels": [
            "focal",
            "other_1",
            "other_2",
            "other_3",
            "other_4"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              -0.9,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.65,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            -0.00021266021,
            -0.00078925009,
            -0.000275401641,
            0.000619254041,
            0.0006580579
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4"
          ],
          "name": "context_3",
          "role": "context",
          "x_weights": [
            [
              0.400195365068,
              -0.643006895612,
              -0.797595481556,
              0.150065778737,
              0.890341233364
            ],
            [
              0.806128817112,
              0.629715913882,
              -0.416942979077,
              -0.887400846322,
              -0.131500905596
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        }
      ],
      "pair_key": "heterogeneity_n2000",
      "variant": "homogeneous"
    },
    {
      "anchor_index": 2,
      "cardinalities": [
        2,
        3,
        5,
        10
      ],
      "copula_rho": 0.35,
      "design": "heterogeneity",
      "focal_prevalence": 0.05,
      "n_test": 20000,
      "n_train": 2000,
      "n_validation": 500,
      "name": "heterogeneity_heterogeneous_n2000",
      "outcomes": [
        {
          "focal_class_index": null,
          "intercepts": [
            0.0,
            0.0
          ],
          "levels": [
            "level_0",
            "level_1"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0
            ],
            [
              0.839094611045,
              -0.839094611045
            ],
            [
              0.325453888768,
              -0.325453888768
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            0.002095325626,
            -0.000713272615,
            -0.001382053011
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              0.272149718348,
              -0.879008838813
            ],
            [
              0.0,
              0.0,
              0.0
            ],
            [
              0.664621702857,
              -0.85786626627,
              0.193244563413
            ]
          ]
        },
        {
          "focal_class_index": 0,
          "intercepts": [
            -1.988431606142,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "levels": [
            "focal",
            "other_1",
            "other_2",
            "other_3",
            "other_4"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              -0.9,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.65,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            -0.000198375793,
            0.000577663966,
            -0.000736245575,
            0.000613861186,
            -0.000256903784,
            -0.000198375793,
            0.000577663966,
            -0.000736245575,
            0.000613861186,
            -0.000256903784
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4",
            "level_5",
            "level_6",
            "level_7",
            "level_8",
            "level_9"
          ],
          "name": "context_3",
          "role": "context",
          "x_weights": [
            [
              0.400195365068,
              -0.150065778737,
              -0.643006895612,
              -0.890341233364,
              -0.797595481556,
              -0.400195365068,
              0.150065778737,
              0.643006895612,
              0.890341233364,
              0.797595481556
            ],
            [
              0.806128817112,
              0.887400846322,
              0.629715913882,
              0.131500905596,
              -0.416942979077,
              -0.806128817112,
              -0.887400846322,
              -0.629715913882,
              -0.131500905596,
              0.416942979077
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        }
      ],
      "pair_key": "heterogeneity_n2000",
      "variant": "heterogeneous"
    },
    {
      "anchor_index": 2,
      "cardinalities": [
        5,
        5,
        5,
        5
      ],
      "copula_rho": 0.35,
      "design": "heterogeneity",
      "focal_prevalence": 0.05,
      "n_test": 20000,
      "n_train": 8000,
      "n_validation": 2000,
      "name": "heterogeneity_homogeneous_n8000",
      "outcomes": [
        {
          "focal_class_index": null,
          "intercepts": [
            7.1513196e-05,
            0.000770550074,
            0.00040446016,
            -0.000520446354,
            -0.000726077075
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.839094611045,
              -0.050230546965,
              -0.870138796344,
              -0.487544804105,
              0.568819536368
            ],
            [
              0.325453888768,
              0.898597180138,
              0.229909710751,
              -0.75650516455,
              -0.697455615108
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            -0.000776956154,
            -0.000104550295,
            0.000712610457,
            0.000544792259,
            -0.000375896266
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              0.819622582788,
              -0.100304506355,
              -0.881614176941,
              -0.444563019958
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.664621702857,
              -0.371777919979,
              -0.89439309367,
              -0.180987411213,
              0.782536722005
            ]
          ]
        },
        {
          "focal_class_index": 0,
          "intercepts": [
            -1.988431606142,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "levels": [
            "focal",
            "other_1",
            "other_2",
            "other_3",
            "other_4"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              -0.9,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.65,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            -0.00021266021,
            -0.00078925009,
            -0.000275401641,
            0.000619254041,
            0.0006580579
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4"
          ],
          "name": "context_3",
          "role": "context",
          "x_weights": [
            [
              0.400195365068,
              -0.643006895612,
              -0.797595481556,
              0.150065778737,
              0.890341233364
            ],
            [
              0.806128817112,
              0.629715913882,
              -0.416942979077,
              -0.887400846322,
              -0.131500905596
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        }
      ],
      "pair_key": "heterogeneity_n8000",
      "variant": "homogeneous"
    },
    {
      "anchor_index": 2,
      "cardinalities": [
        2,
        3,
        5,
        10
      ],
      "copula_rho": 0.35,
      "design": "heterogeneity",
      "focal_prevalence": 0.05,
      "n_test": 20000,
      "n_train": 8000,
      "n_validation": 2000,
      "name": "heterogeneity_heterogeneous_n8000",
      "outcomes": [
        {
          "focal_class_index": null,
          "intercepts": [
            0.0,
            0.0
          ],
          "levels": [
            "level_0",
            "level_1"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0
            ],
            [
              0.839094611045,
              -0.839094611045
            ],
            [
              0.325453888768,
              -0.325453888768
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            0.002095325626,
            -0.000713272615,
            -0.001382053011
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              0.272149718348,
              -0.879008838813
            ],
            [
              0.0,
              0.0,
              0.0
            ],
            [
              0.664621702857,
              -0.85786626627,
              0.193244563413
            ]
          ]
        },
        {
          "focal_class_index": 0,
          "intercepts": [
            -1.988431606142,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "levels": [
            "focal",
            "other_1",
            "other_2",
            "other_3",
            "other_4"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              -0.9,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.65,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            -0.000198375793,
            0.000577663966,
            -0.000736245575,
            0.000613861186,
            -0.000256903784,
            -0.000198375793,
            0.000577663966,
            -0.000736245575,
            0.000613861186,
            -0.000256903784
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4",
            "level_5",
            "level_6",
            "level_7",
            "level_8",
            "level_9"
          ],
          "name": "context_3",
          "role": "context",
          "x_weights": [
            [
              0.400195365068,
              -0.150065778737,
              -0.643006895612,
              -0.890341233364,
              -0.797595481556,
              -0.400195365068,
              0.150065778737,
              0.643006895612,
              0.890341233364,
              0.797595481556
            ],
            [
              0.806128817112,
              0.887400846322,
              0.629715913882,
              0.131500905596,
              -0.416942979077,
              -0.806128817112,
              -0.887400846322,
              -0.629715913882,
              -0.131500905596,
              0.416942979077
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        }
      ],
      "pair_key": "heterogeneity_n8000",
      "variant": "heterogeneous"
    },
    {
      "anchor_index": 0,
      "cardinalities": [
        5,
        5,
        5
      ],
      "copula_rho": 0.0,
      "design": "rho_control",
      "focal_prevalence": 0.05,
      "n_test": 20000,
      "n_train": 2000,
      "n_validation": 500,
      "name": "rho_control_n2000_q050_k5",
      "outcomes": [
        {
          "focal_class_index": 0,
          "intercepts": [
            -1.988431606142,
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "levels": [
            "focal",
            "other_1",
            "other_2",
            "other_3",
            "other_4"
          ],
          "name": "anchor",
          "role": "anchor",
          "x_weights": [
            [
              1.35,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              -0.9,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.65,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            7.1513196e-05,
            0.000770550074,
            0.00040446016,
            -0.000520446354,
            -0.000726077075
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4"
          ],
          "name": "context_1",
          "role": "context",
          "x_weights": [
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.839094611045,
              -0.050230546965,
              -0.870138796344,
              -0.487544804105,
              0.568819536368
            ],
            [
              0.325453888768,
              0.898597180138,
              0.229909710751,
              -0.75650516455,
              -0.697455615108
            ]
          ]
        },
        {
          "focal_class_index": null,
          "intercepts": [
            -0.000776956154,
            -0.000104550295,
            0.000712610457,
            0.000544792259,
            -0.000375896266
          ],
          "levels": [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4"
          ],
          "name": "context_2",
          "role": "context",
          "x_weights": [
            [
              0.606859120465,
              0.819622582788,
              -0.100304506355,
              -0.881614176941,
              -0.444563019958
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.664621702857,
              -0.371777919979,
              -0.89439309367,
              -0.180987411213,
              0.782536722005
            ]
          ]
        }
      ],
      "pair_key": "core_n2000_q050_k5",
      "variant": "conditional_independence"
    }
  ]
}
```

## Limits of this experiment

- The DGP uses bounded covariates and model-aligned linear-softmax marginals. It tests interpolation and controlled recovery, not arbitrary real-world misspecification or extrapolation.
- The sample-size, prevalence, cardinality, architecture, optimization, and dependence settings are the declared grid; results do not automatically transfer beyond it.
- The decoder-width-matched heterogeneity comparison has equal J and sum(K), but it is not entropy matched.
- The anchor focal probability function is held fixed across K and its nonfocal levels are exchangeable; context outcomes have distinct class surfaces. This isolates focal recovery but makes focal MAE alone an incomplete high-K diagnostic.
- Reliability from realized outcomes is noisy for rare classes even with a large test set; direct oracle errors are the primary simulation evidence.
- Initialization replicates reuse a fixed data split and therefore measure optimization sensitivity, not new-sample uncertainty.
- Quadrature comparisons assess GH21 versus GH31 agreement, not mathematical proof that either order is exact.
- Experimental truth-calibration intercept/slope fields were corrected after a disclosed rare-class solver defect and remain excluded from the gates and headline evidence.
- Engineering envelopes are regime-specific safeguards rather than a global certification of predictive validity.
