# Member A+B: measured technical handoff

This file reports the completed local run, not the exploratory scores quoted in the proposal. It is supporting material for Members C-G, not the group's final 1,000-word report.

## Business question and scope

Predict the dataset-defined repayment difficulty outcome and prioritise human review. The approved main model remains Full CatBoost, with L2 Logistic Regression as the benchmark and Lean CatBoost as a data-complexity comparison. The scores are decision support, not automatic loan rejection.

## Data engineering evidence

- Development applicants: 27,000; evaluation applicants: 3,000; development difficulty rate: 8.07%.
- Full feature count: 1,210; Lean feature count: 439. Full uses all seven tables. Lean uses current applications, previous applications and instalments only; it still includes the application's external scores and bureau enquiry counts.
- All features preserve one row per applicant. Bureau history is aggregated to account level before applicant aggregation. No applicant IDs or TARGET are used as predictors.
- 10 personal attributes are kept for outcome monitoring only and are not used by any model: CODE_GENDER, DAYS_BIRTH, AGE_YEARS, NAME_FAMILY_STATUS, CNT_CHILDREN, CNT_FAM_MEMBERS, INCOME_PER_PERSON, NAME_TYPE_SUITE, OCCUPATION_TYPE, ORGANIZATION_TYPE. Sex is excluded because it has no lawful basis as a credit-scoring input under the Equality Act 2010; age, family and occupation fields are excluded under data minimisation because they gave no consistent cross-validated gain. See `audit/AUDIT.md` for the tests behind this policy.
- Split instalment payments are consolidated using exact source-decimal arithmetic before calculating arrears; no currency rounding or arbitrary forgiveness threshold is applied. Undefined ratios remain missing; employment sentinel values become missing with an indicator. Histories without previous-application parent rows are retained via their own validated applicant ID.
- Three development bureau records with post-application update dates are excluded. Dates with uncertain realised-outcome availability are omitted or masked; future scheduled contractual dates are distinguished from realised future events.

| Split | Source | Distinct loans missing previous-application parent |
|---|---|---|
| development | instalment_payments | 2914 |
| development | pos_cash_balance | 2740 |
| development | credit_card_balance | 799 |
| evaluation | instalment_payments | 314 |
| evaluation | pos_cash_balance | 313 |
| evaluation | credit_card_balance | 80 |

The raw dictionary and feature-level definitions are in `feature_dictionary.csv`; audit evidence is in `data_quality.csv`, `raw_column_profile.csv` and `feature_profile.csv`. The dictionary defines TARGET with unspecified X/Y timing parameters, so it must not be renamed as a specific 90-day default or a loss event.

## Validation and selection

A fixed stratified split keeps 16,200 applicants for training/tuning, 5,400 for sigmoid calibration and F2 threshold selection, and 5,400 for final testing. Five-fold stratified CV runs only inside the training partition. Median imputation, scaling and one-hot encoding are fitted inside each Logistic Regression training fold. CatBoost handles numeric missingness and categorical values directly.

Four Logistic Regression candidates vary C and class weighting. Four Full CatBoost candidates vary depth, leaf regularisation and class weighting, under a fixed 400-tree budget. Candidates are selected by mean CV Average Precision. Lean uses the selected Full settings to isolate the incremental value of the extra tables; it is not independently tuned. There is no holdout-based early stopping, threshold adjustment or model refitting after the final test. CV standard deviations below describe variation across five folds, not confidence intervals.

| Model | Candidate (0-based) | Mean AP | AP SD | Folds |
|---|---|---|---|---|
| catboost_full | 0 | 0.2504 | 0.0187 | 5 |
| catboost_full | 1 | 0.2545 | 0.0160 | 5 |
| catboost_full | 2 | 0.2543 | 0.0171 | 5 |
| catboost_full | 3 | 0.2444 | 0.0170 | 5 |
| catboost_lean | 0 | 0.2481 | 0.0153 | 5 |
| logistic_full | 0 | 0.2151 | 0.0148 | 5 |
| logistic_full | 1 | 0.2024 | 0.0180 | 5 |
| logistic_full | 2 | 0.1975 | 0.0139 | 5 |
| logistic_full | 3 | 0.1835 | 0.0171 | 5 |

## Independent holdout performance

| Model | AP (PR-AUC) | ROC-AUC | Precision | Recall | F2 | Brier | Threshold |
|---|---|---|---|---|---|---|---|
| logistic_full | 0.2374 | 0.7426 | 0.1659 | 0.7018 | 0.4263 | 0.0695 | 0.0845 |
| catboost_full | 0.2747 | 0.7695 | 0.1732 | 0.6606 | 0.4227 | 0.0670 | 0.0858 |
| catboost_lean | 0.2716 | 0.7680 | 0.1609 | 0.7408 | 0.4306 | 0.0672 | 0.0746 |

AP is the non-interpolated Average Precision implementation of the PR metric. Random-ranking expected precision is the positive prevalence. Thresholds maximise F2 on the calibration partition only. The same calibration data are used to fit the sigmoid and choose the threshold; metrics on that partition are therefore selection metrics, not independent evidence.

Full-minus-Lean holdout AP difference: **+0.0031**. This is a descriptive difference from one split, not a significance claim or proof that more data always helps. The final model is retained as specified by the approved proposal; report any observed trade-offs honestly.

## Probability calibration

| Model | Raw Brier | Calibrated Brier |
|---|---|---|
| logistic_full | 0.06881 | 0.06948 |
| catboost_full | 0.08018 | 0.06704 |
| catboost_lean | 0.08095 | 0.06722 |

Lower Brier is better. Calibration is assessed rather than assumed to help. See `figures/calibration.png` for reliability curves and `metrics.csv` for log loss and uncalibrated/0.5-threshold comparisons. Calibration uses an independent sigmoid on log-odds and is monotone for this run. The resulting SHAP explanations still refer to the underlying raw model, not the calibrated probability scale.

## Review capacity and exposure analysis

At 20% review capacity, probability-ranked Full CatBoost identifies **222 of 436 difficulty cases (50.92%)** among 1,080 reviewed holdout applicants. Review precision is **20.56%**, or **2.55 times** the holdout prevalence.

| Strategy | Capacity | Reviewed | Cases found | Precision | Recall | Lift | Difficulty credit coverage |
|---|---|---|---|---|---|---|---|
| probability | 10% | 540 | 144 | 26.67% | 33.03% | 3.30x | 30.45% |
| probability | 20% | 1080 | 222 | 20.56% | 50.92% | 2.55x | 49.75% |
| probability | 30% | 1620 | 283 | 17.47% | 64.91% | 2.16x | 61.70% |
| exposure | 10% | 540 | 132 | 24.44% | 30.28% | 3.03x | 42.31% |
| exposure | 20% | 1080 | 188 | 17.41% | 43.12% | 2.16x | 55.50% |
| exposure | 30% | 1620 | 241 | 14.88% | 55.28% | 1.84x | 68.90% |

Exposure ranking uses probability times loan amount. Credit coverage counts loan amounts associated with observed difficulty cases. It is not realised financial loss, expected loss, LGD, savings, or ROI. The table exposes the trade-off between identifying more cases and covering more credit. No actual monetary benefit can be inferred without operating costs, intervention effects and loss data.

## Interpretation and evidence for Members C-F

The largest mean absolute SHAP contributors in the fixed 500-applicant holdout sample are:

1. `EXT_SOURCE_3`: 0.3085 mean absolute log-odds contribution.
2. `EXT_SOURCE_2`: 0.2903 mean absolute log-odds contribution.
3. `EXT_SOURCE_1`: 0.2081 mean absolute log-odds contribution.
4. `DAYS_EMPLOYED`: 0.1340 mean absolute log-odds contribution.
5. `CREDIT_TO_GOODS`: 0.0817 mean absolute log-odds contribution.
6. `BUREAU__debt_to_credit__mean`: 0.0751 mean absolute log-odds contribution.
7. `AMT_ANNUITY`: 0.0715 mean absolute log-odds contribution.
8. `PREV__status_Refused__mean`: 0.0604 mean absolute log-odds contribution.
9. `INST__on_time_paid__min`: 0.0586 mean absolute log-odds contribution.
10. `BUREAU__DAYS_ENDDATE_FACT__max`: 0.0573 mean absolute log-odds contribution.

Use `figures/shap_importance.png` and `figures/shap_distribution.png` with `feature_dictionary.csv` to explain these terms. SHAP additivity is checked against the raw prediction. Contributions are associations conditional on this model, not causal evidence. Coefficients in `logistic_coefficients.csv` use standardised numeric and one-hot features; exponentiating a numeric coefficient describes a one-standard-deviation change, conditional on the other predictors.

`subgroup_diagnostics.csv` supplies sample sizes and outcome counts for gender and income-category diagnostics, and `audit/` adds age, family-status and education results, proxy-leakage tests and a counterfactual gender flip. Small groups are unstable, and these checks do not certify fairness. Proxies such as occupation remain under monitoring; any real deployment needs compliance review and ongoing outcome monitoring by group.

## Predictions and reproducibility

`predictions.csv` contains exactly 3,000 template-ordered applicants and two columns, SK_ID_CURR and binary TARGET. Predicted class 1 count: 920. These binary labels use the calibration-selected F2 threshold (0.085837), not a forced 20% positive rate. `evaluation_risk_scores.csv` separately supplies calibrated probabilities, risk/exposure ranks and exact 10/20/30% review indicators.

The saved production prototype is the same Full CatBoost evaluated above; it is trained on the fit partition, not all development labels. Reproduction instructions are in `../README.md`. `run_manifest.json` records input hashes, seed, partitions and selected settings; `verification.json` records the independent reconstruction check when completed.

## Limitations and remaining group responsibilities

The evaluation set has no labels, so no evaluation-set accuracy is claimed. The holdout is one random stratified split rather than a future time period. Model, hyperparameter and feature choices were made by cross-validation or legal requirement; the holdout was used only to report results. Unique customer identity across applications is unavailable. Source snapshot timing is assumed from the supplied dictionary and relative dates; real deployment needs point-in-time auditing and temporal validation. Missingness may indicate incomplete coverage rather than no debt. Scheduled-payment versions are retained separately because the supplied data do not identify which overlapping contractual schedule supersedes another.

Members C/D should develop the business interpretation and responsible-use analysis from these measured results. Members E/F/G still own the final report, slide deck, recording, group review and submission. The official assessment requires disclosure of AI use; see `../AI_USAGE.md` and confirm intellectual ownership only after reviewing and understanding the work.
