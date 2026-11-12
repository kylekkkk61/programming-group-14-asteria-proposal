# Review correction record

This record covers the fixes for the decimal-precision error and the SHAP legend confirmed by an independent review. `outputs/` is the current version. The earlier outputs, which contain the known bug, were kept locally for comparison and are not part of the repository.

## What changed

1. Instalment amounts are converted from the raw CSV text directly to `Decimal` and summed exactly to decide the completion day, `late` and `shortfall`, before being converted to model floats. There is no rounding to cents or mills and no arbitrary shortfall tolerance; genuinely small differences in the raw data are kept.
2. The real case of applicant 104276, previous loan 2566955, instalment 5 changed from a false shortfall, late=1 and a 1,024-day delay to shortfall=0, late=0 and a 0-day delay.
3. The SHAP distribution plot now separates the numeric high/low colours from the grey used for categorical and missing values, so normal text categories are no longer read as missing. SHAP still explains raw log-odds; it is not causal and not in probability points.
4. The data split, feature list and order, candidate hyperparameter grids, CV seed and model selection rule are unchanged. Three regression cases were added: an on-time decimal split payment, a late decimal completion and a genuine tiny shortfall.

## Verification evidence

- All 17 regression tests pass; the new decimal-payment case failed before the fix.
- An independent `fractions.Fraction` exact-rational check covered every raw due instalment: 970,946 in development and 107,544 in evaluation. Differences in shortfall, late and delay_days were all 0. See `instalment_precision_check.json`.
- All 45 five-fold candidate fits and the three final model fits were re-run; features, calibrators, thresholds, models, figures and predictions were rebuilt.
- Features were rebuilt again from the seven raw evaluation tables and scored with the new saved model; the 3,000 predictions exactly match the delivered CSV. ID, column, label and review-capacity checks pass; see `verification.json`.
- The old and new partitions are identical, and every non-INST feature is identical value by value.
- Development: 4,786 applicants have changed instalment features, across 48 features. This count includes tiny numerical differences from the exact decimal conversion and is not the number of misclassified delays.
- Evaluation: 525 applicants have changed instalment features, across 42 features, with the same caveat.

## Before/after comparison on the same holdout

| Model | AP: old → new | ROC-AUC: old → new | F2: old → new | Brier: old → new |
|---|---|---|---|---|
| logistic_full | 0.2464 → 0.2458 | 0.7512 → 0.7503 | 0.4183 → 0.4234 | 0.0689 → 0.0690 |
| catboost_full | 0.2768 → 0.2799 | 0.7700 → 0.7779 | 0.4311 → 0.4217 | 0.0669 → 0.0666 |
| catboost_lean | 0.2762 → 0.2696 | 0.7754 → 0.7722 | 0.4432 → 0.4335 | 0.0668 → 0.0671 |

Full CatBoost probability Top 20% recall: **51.83% → 53.44%**; cases captured 226 → 233, with 1,080 reviewed in both.

**238 of 3,000** evaluation classifications differ from the old version; predicted positives went from 897 to 783, and the threshold from 0.08875981 to 0.09556862. This is the combined result of re-running feature engineering, CV selection, training and calibration; the changes should not all be attributed to any single instalment.

## Limits of interpretation

The fix was decided by the correctness of the payment arithmetic. Features were not changed repeatedly, and the search was not widened, in response to holdout scores, so not every metric is guaranteed to improve. The holdout had already been inspected in the old version; this is a corrected re-test on the same fixed holdout, not a fresh unseen test. The evaluation set has no TARGET, so it cannot give a true accuracy.

The Full/Lean difference is still a descriptive comparison on one holdout and does not show a statistically significant advantage. Overlapping payment versions, real deployment-time availability and cross-application customer identity remain disclosed data limits; no versions were dropped and no business definitions were changed without evidence.


---

# Second revision: personal-attribute policy

## What changed

The model no longer uses ten personal attributes, which are kept only for monitoring: `CODE_GENDER` (excluded because sex has no lawful basis as a credit-scoring input under the Equality Act 2010), age, family fields and occupation fields (excluded under data minimisation because adding them gave no consistent cross-validated gain). A `Maternity leave` income type is masked if it appears; this extract has none. The evidence, the pre-set retention rule and the checked legal sources are in `audit/AUDIT.md`.

The pipeline re-ran its pre-specified procedure unchanged: the same split, folds, candidate grids and AP selection rule. On the reduced feature set it selected different CatBoost settings (depth 4, l2_leaf_reg 5, SqrtBalanced class weights instead of depth 6, l2_leaf_reg 8, no class weights). The top two candidates differ by 0.0002 CV AP.

## Before and after on the same holdout

| Model | AP: before → after | ROC-AUC: before → after | Brier: before → after |
|---|---|---|---|
| logistic_full | 0.2458 → 0.2374 | 0.7503 → 0.7426 | 0.0690 → 0.0695 |
| catboost_full | 0.2799 → 0.2747 | 0.7779 → 0.7695 | 0.0666 → 0.0670 |
| catboost_lean | 0.2696 → 0.2716 | 0.7722 → 0.7680 | 0.0671 → 0.0672 |

Full CatBoost Top 20% queue: 233 → 222 difficulty cases found (of 436). Gap between men's and women's queue rates: 13.2 → 5.4 points. Evaluation predictions: 293 of 3,000 labels changed; predicted positives went from 783 to 920 as the label threshold moved from 0.0956 to 0.0858.

The before/after model differences mix two effects: removing attributes and the new hyperparameters. `audit/AUDIT.md` separates them by comparing variants under the same settings; under those settings the cost of the policy is AP −0.010 (95% interval −0.025 to +0.005).

## Limits

All feature decisions used cross-validation only. The holdout has now been examined across several revisions, so it is a repeated re-test of one fixed split; a later out-of-time sample is needed for a clean estimate.

