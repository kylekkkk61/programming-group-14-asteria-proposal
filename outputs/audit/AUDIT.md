# Personal-attribute audit and feature policy

This report explains which personal attributes the final model may use, why, and what that choice costs. It is produced by `audit.py` and covers sex, age, family details, occupation and the remaining socio-economic fields. All numbers come from the files in this folder; `stage1_removal/` holds the first audit run that led to the decision.

**Final policy.** No model uses `CODE_GENDER`, age (`DAYS_BIRTH`, `AGE_YEARS`), family fields (`NAME_FAMILY_STATUS`, `CNT_CHILDREN`, `CNT_FAM_MEMBERS`, `INCOME_PER_PERSON`, `NAME_TYPE_SUITE`) or occupation fields (`OCCUPATION_TYPE`, `ORGANIZATION_TYPE`). These ten columns stay in the feature tables so outcomes can be monitored by group. Income type, education and housing stay in the model as financial-circumstance fields and are monitored. The list lives in `EXCLUDED_FROM_MODELS` in `src/features.py`, and `verify_outputs.py` and the unit tests fail if any saved model uses one of them.

---

## 1. Legal and regulatory basis (sources checked on 12 November 2026)

| Point | Source | What it means for this model |
|---|---|---|
| A service provider must not discriminate as to the terms of a service, by not providing it, or by subjecting the person to any other detriment. | [Equality Act 2010, s.29](https://www.legislation.gov.uk/ukpga/2010/15/section/29) | Lending is a service; extra review, delay or worse terms can be a detriment. |
| Direct discrimination is less favourable treatment because of a protected characteristic. Only for age can the provider defend it as "a proportionate means of achieving a legitimate aim". | [s.13(1)–(2)](https://www.legislation.gov.uk/ukpga/2010/15/section/13) | A score that is higher because an applicant is male has no general justification defence. |
| The former exception that allowed sex-based differences in insurance was removed on 21 December 2012. | [Sch. 3 para. 22 (omitted)](https://www.legislation.gov.uk/ukpga/2010/15/schedule/3/paragraph/22) | No financial-services exception for sex remains. |
| Age discrimination is not unlawful in connection with a financial service, including credit, if any risk assessment that considers age uses relevant information from a source it is reasonable to rely on. | [Sch. 3 para. 20A](https://www.legislation.gov.uk/ukpga/2010/15/schedule/3/paragraph/20A) | Age could lawfully be used. We excluded it because it added nothing (Section 3). |
| The services part of the Act does not apply to marriage and civil partnership, or to age for people under 18. | [s.28(1)](https://www.legislation.gov.uk/ukpga/2010/15/section/28) | Marital status is not a protected characteristic in lending. We excluded it because it added nothing. |
| Indirect discrimination: a neutral rule that puts a protected group at a particular disadvantage is unlawful unless it is a proportionate means of achieving a legitimate aim. | [s.19](https://www.legislation.gov.uk/ukpga/2010/15/section/19) | Proxies such as occupation need a justification; a feature that adds no predictive value is hard to justify. |
| Personal data must be "adequate, relevant and limited to what is necessary" (data minimisation). | [UK GDPR art. 5(1)(c)](https://www.legislation.gov.uk/eur/2016/679/article/5) | The basis for our retention rule. |
| Sex and age are not special category data (the list covers race, politics, religion, union membership, genetics, biometrics, health, sex life and sexual orientation). | [ICO](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/lawful-basis/special-category-data/what-is-special-category-data/) | Holding them for monitoring needs no special-category condition. |
| Pseudonymised data that can be linked back to a person with additional information is still personal data. | [UK GDPR recital 26](https://www.legislation.gov.uk/eur/2016/679/introduction) | De-identifying the training data does not take the scoring of a real applicant outside data-protection or equality law. |
| Firms are not expected to collect new protected-characteristic data for monitoring, but where they already hold it they should use it, where possible, to monitor differences in outcomes across groups. | [FCA–ICO joint statement, section 4](https://www.fca.org.uk/publications/corporate-documents/joint-fca-and-ico-statement-regulatory-expectations-regarding-firms) | Keep `CODE_GENDER` for monitoring, not for scoring. |
| Good practice seen by the FCA: testing for unfair or unjustified differences across consumer groups, comparing outcomes and error rates, retesting after changes, and not relying only on a person reviewing model outputs. The publication is illustrative and creates no new rules. | [FCA, AI in financial services: good practice](https://www.fca.org.uk/publications/good-poor-practice/ai-financial-services) | The tests in this audit follow that practice. |
| Before lending, a firm must assess creditworthiness, including affordability. | [FCA consumer credit sales and advice](https://www.fca.org.uk/firms/consumer-credit-sales-and-advice) | The model ranks extra review; it cannot replace affordability checks. |

This is a reading of primary sources for a student project, not legal advice. A real deployment would need a compliance review.

---

## 2. Design

Every variant reuses the locked partitions, the same five folds and the hyperparameters selected by `pipeline.py` for the final model (depth 4, l2_leaf_reg 5, SqrtBalanced class weights, 400 trees). The final ("policy") variant must reproduce the saved submitted model exactly, or the script stops.

**Retention rule, fixed before the results were seen.** Sex is excluded by law. Any other identity group is used only if adding it raises mean cross-validated AP **and** raises AP in at least four of five folds. Decisions use cross-validation only; the holdout is reported but never used to choose.

The audit ran twice. Stage 1 (`stage1_removal/`) started from a model without sex and removed each group in turn; no group met the rule. Stage 2 (this folder) starts from the final model and adds each group back, under the final hyperparameters, to confirm the decision.

---

## 3. Results

### 3.1 Does each group earn its place? (cross-validation, stage 2)

| Group added back | Mean CV AP change | Folds improved | Decision |
|---|---:|---:|---|
| Sex | +0.0053 | 4 of 5 | **Excluded by law** even though it passes the statistical rule |
| Age | −0.0005 | 3 of 5 | Excluded (no consistent gain) |
| Family | −0.0023 | 2 of 5 | Excluded (no consistent gain) |
| Occupation | −0.0045 | 0 of 5 | Excluded (no gain; strong sex proxy) |

Stage 1 gave the same decisions. Removing age changed CV AP by −0.0001, and keeping it was better in only 3 of 5 folds; removing family raised CV AP by 0.0024; removing occupation changed it by −0.0007, with keeping it better in 3 of 5 folds.

Sex is the one attribute with some cross-validated signal. That is why it must be excluded by rule rather than by a performance test: its predictive value does not make its use lawful.

### 3.2 Performance of every variant

| Variant | Features | CV AP | Holdout AP | ROC-AUC | Top 10% cases | Top 20% cases | Top 30% cases |
|---|---:|---:|---:|---:|---:|---:|---:|
| All attributes (original approach) | 1,220 | 0.2569 | 0.2847 | 0.772 | 151 | 232 | 286 |
| **Final policy (submitted)** | **1,210** | **0.2545** | **0.2747** | **0.770** | **144** | **222** | **283** |
| Final + sex | 1,211 | 0.2597 | 0.2734 | 0.770 | 147 | 229 | 289 |
| Final + age | 1,212 | 0.2540 | 0.2741 | 0.768 | 145 | 223 | 289 |
| Final + family | 1,215 | 0.2522 | 0.2693 | 0.764 | 140 | 217 | 281 |
| Final + occupation | 1,212 | 0.2500 | 0.2826 | 0.772 | 145 | 223 | 297 |
| Identity-blind (also no income type, education, housing) | 1,207 | 0.2517 | 0.2649 | 0.763 | 145 | 216 | 284 |
| Lean (three tables) | 439 | 0.2481 | 0.2716 | 0.768 | 143 | 226 | 292 |
| Lean without external scores | 430 | 0.1918 | 0.2079 | 0.712 | 125 | 197 | 255 |

The holdout has 5,400 applicants and 436 difficulty cases; random review of 20% would find about 87. "Final + occupation" has a higher holdout AP but is worse in all five CV folds, which is exactly why the holdout is not used to choose features.

### 3.3 What the policy costs (paired bootstrap, 2,000 resamples)

| Comparison (final minus …) | AP difference | 95% interval | Top 20% recall difference | 95% interval |
|---|---:|---|---:|---|
| All attributes | −0.010 | [−0.025, +0.005] | −2.4 pts | [−5.0, +0.2] |
| Final + sex | +0.001 | [−0.013, +0.015] | −1.7 pts | [−4.3, +0.9] |
| Final + age | +0.000 | [−0.013, +0.013] | −0.5 pts | [−3.4, +2.4] |
| Final + family | +0.005 | [−0.007, +0.018] | +0.9 pts | [−1.7, +3.6] |
| Final + occupation | −0.008 | [−0.020, +0.006] | −0.5 pts | [−3.0, +1.9] |
| Lean | +0.003 | [−0.016, +0.022] | −1.2 pts | [−4.7, +2.1] |
| Lean without external scores | +0.066 | [+0.038, +0.094] | +5.8 pts | [+1.4, +9.9] |

Every personal-attribute comparison has an interval that includes zero. Against the original all-attributes model, the policy costs about 10 cases in a 1,080-applicant queue (222 against 232), which this holdout cannot separate from noise. The only clear loss in the table comes from removing external credit scores, which are not personal attributes.

### 3.4 What the policy buys: gender outcomes

| Model | Women queued | Men queued | Gap | 95% interval | Women recall | Men recall | Gap |
|---|---:|---:|---:|---|---:|---:|---:|
| All attributes | 15.5% | 28.7% | 13.2 pts | [11.2, 15.7] | 44.4% | 64.2% | 19.8 pts |
| Final + sex | — | — | 12.2 pts | [9.8, 14.4] | — | — | 13.6 pts |
| **Final policy** | **18.2%** | **23.6%** | **5.4 pts** | **[3.1, 7.7]** | **46.9%** | **56.0%** | **9.0 pts** (interval −0.4 to 18.1) |
| Identity-blind | — | — | 5.7 pts | [3.1, 7.7] | — | — | 4.1 pts |

Observed difficulty rates are 10.6% for men and 6.8% for women, so some gap in queue rate is expected from any risk ranking. The policy cuts the queue-rate gap by more than half. The remaining gap is not explained by an input the model is forbidden to use.

**Trade-off in calibration.** Without sex, the model over-predicts for women (mean predicted probability 1.11 times the observed rate) and under-predicts for men (0.82 times). With all attributes the ratios were 1.01 and 0.92. Group-level calibration is worse, ranking within each gender is unchanged (ROC-AUC 0.766 for women, 0.767 for men), and the gap in who is queued is smaller. We accept this trade-off because the alternative uses sex directly.

### 3.5 Direct effect of sex (counterfactual flip)

In the model that is allowed to use sex ("final + sex"), we changed only each holdout applicant's recorded sex. On average the raw score moved by 0.038; men's scores fell by 0.041 when recorded as female and women's rose by 0.037 when recorded as male. **164 of the 513 men in the queue would leave it**, and **287 women outside it would enter**, purely because of the recorded sex. This is direct evidence that a model with sex as an input treats otherwise identical applicants differently.

### 3.6 Proxies: how much does the model still "know" about sex?

A separate classifier tried to predict sex from the features each model may use (holdout ROC-AUC; 0.5 means no information):

| Feature set | AUC predicting sex | Strongest predictors |
|---|---:|---|
| Final + occupation | 0.875 | `OCCUPATION_TYPE`, `ORGANIZATION_TYPE`, `EXT_SOURCE_1`, `OWN_CAR_AGE`, income |
| **Final policy** | **0.795** | `EXT_SOURCE_1`, `OWN_CAR_AGE`, income, `FLAG_OWN_CAR`, `DAYS_ID_PUBLISH` |
| Identity-blind | 0.793 | `EXT_SOURCE_1`, `OWN_CAR_AGE`, income, `FLAG_OWN_CAR`, document flag 8 |

Removing occupation removes the strongest proxy. What remains is spread across financial fields (income, car ownership, external score) that are legitimate credit information and cannot be removed without destroying the model. This residual association is why outcome monitoring by group stays necessary.

### 3.7 Age and other groups under the final model

| Age band | n | Difficulty rate | Queued | Recall | Calibration ratio |
|---|---:|---:|---:|---:|---:|
| Under 30 | 796 | 12.4% | 40.7% | 61.6% | 0.95 |
| 30–44 | 2,143 | 8.4% | 18.3% | 50.3% | 0.90 |
| 45–59 | 1,850 | 6.5% | 16.3% | 47.1% | 1.10 |
| 60 and over | 611 | 5.7% | 10.1% | 37.1% | 1.03 |

Younger applicants are queued much more often, in line with a difficulty rate more than twice that of the over-60s, and this pattern is the same with or without age as an input (stage 1). Results by family status, income type and education are in `audit_groups.csv`.

---

## 4. Decision and what remains

1. **Submitted model:** Full CatBoost using all seven tables and no personal attributes from the excluded list.
2. **Cost:** within sampling noise on this holdout; about 10 fewer cases in a 20% queue than the original all-attributes model.
3. **Benefit:** no direct use of sex; the gender queue-rate gap falls from 13.2 to 5.4 points; three groups of personal data that added nothing are no longer processed for scoring.
4. **Still required before any deployment:** compliance review of income type and education as possible proxies; monitoring of queue rates, approval rates after review and calibration by sex and age band; and re-running this audit on new, later data.

## Files

| File | Content |
|---|---|
| `audit_metrics.csv` | Overall, CV and gender-gap metrics for every variant |
| `audit_groups.csv` | Results by gender, age band, family status, income type and education |
| `audit_bootstrap.csv` | Bootstrap intervals for each variant and differences from the final model |
| `audit_proxy_leakage.csv` | Sex-prediction AUC and top predictors per feature set |
| `audit_summary.json` | Settings, retention decisions and the counterfactual flip |
| `holdout_*.csv`, `cv_*.csv` | Per-applicant holdout scores and per-fold CV results |
| `stage1_removal/` | The first audit run (removal direction) |
| `run.log` | Console output |
