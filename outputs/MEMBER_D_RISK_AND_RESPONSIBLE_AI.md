# Member D — Risk interpretation, errors and responsible lending (working draft)

> **Handover note for Member D and Members F/G.** All numbers refer to the final model behind `predictions.csv`: Full CatBoost on all seven tables, with sex, age, family and occupation fields excluded from scoring and kept only for monitoring. Sources are exact Tree SHAP on 500 holdout applicants, the holdout predictions (5,400 applicants, 436 difficulty cases) and the audit in `audit/AUDIT.md`, whose legal sources were checked against legislation.gov.uk, the ICO and the FCA on 12 November 2026. Member D should decide what goes into the slides; the legal points are a reading of primary sources, not legal advice.

**Terminology.** "Repayment difficulty" means `TARGET = 1`: a payment more than X days late on at least one of the first Y instalments (X and Y undisclosed). It is not a default and says nothing about money lost.

---

## 1. What drives the score

### 1.1 Attribution by data source

Mean absolute SHAP summed by source table (`shap_by_source.csv`; total 3.79 log-odds over the 500-applicant sample):

| Source | Share |
|---|---:|
| Current application (includes the three external credit scores) | 40.0% |
| Bureau accounts and monthly history | 17.0% |
| Instalment payments | 16.4% |
| Previous applications | 15.8% |
| POS/cash balances | 5.7% |
| Credit card balances | 5.1% |

This shows how the model spreads attribution in this sample. It is not each table's share of predictive power and it is not causal. The Lean comparison shows the point: bureau, POS and card tables receive 28% of attribution, yet removing them changes holdout AP by an amount the bootstrap cannot separate from zero.

### 1.2 Top features and what a reviewer could check

Medians are shown because several features have extreme values. "Direction" is the association inside this model.

| Rank | Feature | Mean abs. SHAP | Median, no difficulty | Median, difficulty | Direction | Possible review question |
|---:|---|---:|---:|---:|---|---|
| 1 | `EXT_SOURCE_3` (external score) | 0.309 | 0.550 | 0.352 | Lower score, higher risk | Is a low external score consistent with the applicant's own repayment record? (20.5% missing) |
| 2 | `EXT_SOURCE_2` (external score) | 0.290 | 0.573 | 0.430 | Lower score, higher risk | Same; almost always present |
| 3 | `EXT_SOURCE_1` (external score) | 0.208 | 0.519 | 0.344 | Lower score, higher risk | Missing for 55.5%; missing is not evidence of risk by itself |
| 4 | `DAYS_EMPLOYED` | 0.134 | 4.7 years | 3.1 years | Shorter current employment, higher risk | Is income stable and verifiable? |
| 5 | `CREDIT_TO_GOODS` | 0.082 | 1.12 | 1.16 | Credit above goods price, higher risk | What does the amount above the goods price pay for? |
| 6 | `BUREAU__debt_to_credit__mean` | 0.075 | 0.18 | 0.31 | More of external credit still owed, higher risk | What are current external commitments? (means distorted by outliers up to 383) |
| 7 | `AMT_ANNUITY` | 0.072 | 24,689 | 24,869 | Non-linear; works with income and credit amount | Is the repayment affordable on verified income? |
| 8 | `PREV__status_Refused__mean` | 0.060 | 0 | 0 | More past refusals, higher risk | 32% of no-difficulty and 43% of difficulty applicants had a refusal; reasons unknown |
| 9 | `INST__on_time_paid__min` | 0.059 | 343 | 0 | Lower minimum on-time payment, higher risk | Has any past instalment gone unpaid by its due date? |
| 10 | `BUREAU__DAYS_ENDDATE_FACT__max` | 0.057 | −344 days | −330 days | Recency of closed external credit | Has the applicant recently closed external credit? |
| 11 | `NAME_INCOME_TYPE` | 0.053 | — | — | "Working" 9.3% difficulty, "Pensioner" 6.2% | Stability of income source; monitored as a possible age proxy |

Three reminders. A past refusal shows an earlier application was declined for an unknown reason; it is not evidence of a credit defect. `CREDIT_TO_GOODS` compares credit with the goods price and is not a loan-to-value ratio on an asset; no cap value has been tested. Logistic Regression coefficients point the same way for shared features (`EXT_SOURCE_3` −0.35, `CREDIT_TO_GOODS` +0.14), which supports the directions but not causation.

---

## 2. Errors and what they mean for operations

### 2.1 Error counts on the holdout

| Operating point | Flagged | True positives | False positives | False negatives | True negatives | Difficulty rate among unflagged |
|---|---:|---:|---:|---:|---:|---:|
| Top 20% review queue (1,080 by rank) | 1,080 | 222 | 858 | 214 | 4,106 | 4.95% |
| Label threshold (p ≥ 0.0858) | 1,663 | 288 | 1,375 | 148 | 3,589 | 3.96% |

Against an 8.07% base rate, both operating points lower the difficulty rate among unflagged applicants, but about 1 in 20–25 of them still experiences difficulty.

### 2.2 Who the errors are (Top 20% queue, medians)

| Group | n | Predicted probability | `EXT_SOURCE_2` | `EXT_SOURCE_3` | Bureau debt-to-credit | Credit-to-income | Share of past instalments late |
|---|---:|---:|---:|---:|---:|---:|---:|
| True positives | 222 | 0.202 | 0.341 | 0.276 | 0.42 | 3.33 | 0.079 |
| False positives | 858 | 0.159 | 0.367 | 0.322 | 0.39 | 3.22 | 0.065 |
| False negatives | 214 | 0.073 | 0.543 | 0.466 | 0.22 | 3.00 | 0.030 |
| True negatives | 4,106 | 0.045 | 0.599 | 0.580 | 0.16 | 3.27 | 0.006 |

**False positives look like true positives at application time:** similar external scores, external debt and late-payment history. The model is not guessing; the information available does not separate these groups. `TARGET = 0` only means the dataset's definition was not met.

**False negatives look like low-risk applicants:** their external scores and payment history resemble true negatives, and their loan size and credit-to-income ratio are no higher than other groups. The data contain no post-loan events, so we cannot say why they later struggled.

### 2.3 Implications

- A place in the queue should trigger verification of income, commitments and affordability, not an automatic decline. Queue precision is 20.6%, so about four in five queued applicants do not go on to meet the difficulty definition.
- Applicants outside the queue still need Asteria's standard credit and affordability checks; UK lenders must assess creditworthiness, including affordability, before lending ([FCA](https://www.fca.org.uk/firms/consumer-credit-sales-and-advice)).
- False negatives cannot be reduced much with application-time data. Realistic levers are more review capacity (30% finds 283 cases against 222) and monitoring early repayment behaviour after lending; neither has been tested here.

---

## 3. Fairness and responsible lending

### 3.1 The legal position (checked against primary sources)

- **Sex.** Lending is a service, and a service provider must not discriminate as to terms or by "any other detriment" ([Equality Act 2010 s.29](https://www.legislation.gov.uk/ukpga/2010/15/section/29)). Direct discrimination is less favourable treatment because of a protected characteristic, and only for age can it be defended as proportionate ([s.13](https://www.legislation.gov.uk/ukpga/2010/15/section/13)). The old insurance exception for sex was removed in 2012 ([Sch. 3 para. 22](https://www.legislation.gov.uk/ukpga/2010/15/schedule/3/paragraph/22)). Using sex to score applicants therefore has no lawful basis.
- **Age.** Age-related treatment is permitted in financial services, including credit, if any risk assessment that considers age relies on relevant, reliable information ([Sch. 3 para. 20A](https://www.legislation.gov.uk/ukpga/2010/15/schedule/3/paragraph/20A)). Age could be used lawfully.
- **Marital status.** The services part of the Act does not cover marriage and civil partnership ([s.28](https://www.legislation.gov.uk/ukpga/2010/15/section/28)).
- **Proxies.** A neutral feature that disadvantages a protected group is indirect discrimination unless it is a proportionate means to a legitimate aim ([s.19](https://www.legislation.gov.uk/ukpga/2010/15/section/19)).
- **Data protection.** Personal data must be limited to what is necessary ([UK GDPR art. 5(1)(c)](https://www.legislation.gov.uk/eur/2016/679/article/5)). Pseudonymised data that can be linked back to a person is still personal data ([recital 26](https://www.legislation.gov.uk/eur/2016/679/introduction)), so de-identifying training data does not change the rules for scoring a real applicant. Sex and age are not special category data ([ICO](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/lawful-basis/special-category-data/what-is-special-category-data/)).
- **Monitoring.** Firms holding protected-characteristic data should use it, where possible, to monitor differences in outcomes across groups ([FCA–ICO joint statement](https://www.fca.org.uk/publications/corporate-documents/joint-fca-and-ico-statement-regulatory-expectations-regarding-firms)). FCA good practice includes comparing outcomes and error rates across groups and not relying only on a person reviewing model outputs ([FCA](https://www.fca.org.uk/publications/good-poor-practice/ai-financial-services); illustrative, not new rules).

### 3.2 Our policy and the evidence behind it

| Attribute group | Decision | Reason |
|---|---|---|
| Sex | Excluded from scoring; kept for monitoring | No lawful basis. In cross-validation it adds +0.005 AP, which shows the rule must be legal, not statistical |
| Age | Excluded | Lawful, but adding it back gave no consistent CV gain (−0.0005; better in 3 of 5 folds) |
| Family (status, children, household size, companion) | Excluded | No consistent gain (−0.002; 2 of 5 folds) |
| Occupation and employer type | Excluded | No gain (−0.004; 0 of 5 folds) and the strongest sex proxy |
| Income type, education, housing | Kept, monitored | Financial circumstances; flagged for compliance review as possible proxies |

The retention rule (use a group only if it raises mean CV AP and does so in at least four of five folds) was fixed before results were seen, and decisions used cross-validation only.

### 3.3 What it costs and what it buys

| | All attributes | Final model |
|---|---:|---:|
| Holdout AP | 0.285 | 0.275 |
| Top 20% cases found | 232 | 222 |
| Women / men queued | 15.5% / 28.7% | 18.2% / 23.6% |
| Gap in queue rate (95% interval) | 13.2 pts [11.2, 15.7] | 5.4 pts [3.1, 7.7] |
| Women / men recall in the queue | 44.4% / 64.2% | 46.9% / 56.0% |
| Calibration ratio, women / men (predicted ÷ observed) | 1.01 / 0.92 | 1.11 / 0.82 |
| Ranking quality within gender (ROC-AUC, women / men) | 0.762 / 0.770 | 0.766 / 0.767 |

- **Cost:** AP −0.010 (95% interval −0.025 to +0.005) and 10 fewer cases at 20% capacity, within sampling noise.
- **Direct effect removed:** in a model that uses sex, changing only the recorded sex moves 164 of 513 queued men out of the queue and 287 women into it.
- **Gap halved, not closed.** Men have a higher observed difficulty rate (10.6% against 6.8%), and the remaining features still carry some information about sex: a classifier predicts sex from them with AUC 0.80, mostly through the external score, income and car ownership. These are legitimate credit information, so the residual gap must be monitored rather than engineered away.
- **Group calibration gets worse.** Without sex the model over-predicts for women and under-predicts for men. We accept this because the alternative uses sex directly; it should be tracked in monitoring.

### 3.4 Age under the final model

| Age band | Difficulty rate | Queued | Recall in queue |
|---|---:|---:|---:|
| Under 30 | 12.4% | 40.7% | 61.6% |
| 30–44 | 8.4% | 18.3% | 50.3% |
| 45–59 | 6.5% | 16.3% | 47.1% |
| 60 and over | 5.7% | 10.1% | 37.1% |

Younger applicants are queued far more often, in line with a difficulty rate more than twice that of the over-60s. The pattern is the same with or without age as an input, because employment length, income type and credit history carry the same information. Being queued means extra checks, so the burden on younger applicants should be monitored.

### 3.5 Recommendations for deployment

1. Keep sex, age, family and occupation fields out of scoring, and keep them for monitoring.
2. Monitor by sex and age band: queue rate, approval rate after review, and calibration, with agreed tolerances and a named owner.
3. Have compliance review income type and education as possible proxies.
4. Restrict reviewer reason codes to verifiable financial information (external scores, commitments, repayment history, loan structure).
5. Re-run the audit on new data and after any retraining.

---

## 4. Limits the team should state

- **Outcome definition.** X and Y in `TARGET` are unknown; the model predicts this dataset's label, not default or loss.
- **One random split.** Holdout numbers come from one 5,400-applicant random split, not a later period. All model and feature choices were made by cross-validation or legal requirement; the holdout was used only to report results.
- **Data timing.** The extract has relative dates only; deployment needs point-in-time checks of when each source is available.
- **Missing data.** No bureau record may mean no external borrowing or no coverage.
- **Overlapping payment schedules** are kept as separate versions because the data do not say which replaces which.
- **SHAP** comes from 500 applicants and describes associations in this model.
