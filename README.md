# Asteria Finance: Member A+B technical delivery

This project implements **Member A (data engineering)** and **Member B (modelling and validation)** from the confirmed group proposal. It covers all seven tables, an L2 Logistic Regression benchmark, CatBoost, five-fold stratified cross-validation, probability calibration, SHAP, the Full/Lean comparison, review-capacity analysis and `predictions.csv`. The original CSV files are never modified.

## Running the project

A local `.venv` in this folder holds every Python package. Use Python 3.12; exact package versions are pinned in `requirements.txt`.

```bash
# Run inside this project folder; skip installation if .venv already exists.
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# Rebuild every deliverable from the raw data (includes the full five-fold tuning).
.venv/bin/python pipeline.py

# Independent exact-arithmetic check of every raw instalment.
.venv/bin/python -m tests.check_instalment_precision

# Behavioural regression checks.
.venv/bin/python -m unittest discover -s tests -v

# A second full run that does not overwrite the delivered results.
.venv/bin/python pipeline.py --output outputs_reproduced

# Re-create predictions from the raw evaluation CSVs with the saved model and compare.
.venv/bin/python verify_outputs.py

# Personal-attribute and data-scope audit (about 15 minutes; writes outputs/audit/).
.venv/bin/python audit.py
```

With uv, use `uv venv .venv --python 3.12` and `uv pip install --python .venv/bin/python -r requirements.txt`. Do not install into the system Python. The default data path is `asteria_student_data`; `--data` accepts any folder with the same structure. `--threads` sets CPU threads; `--iterations` changes the number of trees per CatBoost candidate and therefore changes results. The delivered run uses the defaults: 400 trees, 4 threads and the fixed seed 2026.

## File guide

| File | Purpose |
|---|---|
| `pipeline.py` | End-to-end entry point from the seven raw CSVs to the final predictions |
| `src/features.py` | Key checks, cleaning, split-payment consolidation and applicant-level features |
| `src/modelling.py` | Models, five-fold tuning, calibration, classification threshold and review metrics |
| `src/reporting.py` | EDA, performance figures, SHAP and review-strategy figures |
| `verify_outputs.py` | Rebuilds evaluation features from raw data, re-scores with the saved model and checks the output contract |
| `audit.py` | Personal-attribute and data-scope audit: add-back tests, group outcomes, bootstrap intervals, proxy leakage and a counterfactual sex flip |
| `tests/test_pipeline.py` | Regression checks that fail if key behaviour degrades |
| `outputs/RESULTS.md` | Measured results and the technical handoff for Members C–G |
| `outputs/MEMBER_C_BUSINESS_STRATEGY.md` | Business strategy working draft (Member C) |
| `outputs/MEMBER_D_RISK_AND_RESPONSIBLE_AI.md` | Risk interpretation and responsible-lending working draft (Member D) |
| `outputs/MEMBER_E_TECHNICAL_REPORT.md` | Technical report working draft (Member E) |
| `outputs/audit/` | Audit report (`AUDIT.md`) with checked legal sources, metrics, group results and per-applicant holdout scores |
| `outputs/predictions.csv` † | The two-column binary predictions required by the assignment |
| `outputs/evaluation_risk_scores.csv` † | Calibrated probabilities, loan amount, both rankings and Top 10/20/30% flags |
| `outputs/development_features.parquet` † | Development features indexed by applicant ID, plus TARGET |
| `outputs/evaluation_features.parquet` † | Evaluation features with the same schema, without TARGET |
| `outputs/feature_dictionary.csv` | Source, definition and Lean membership of every feature |
| `outputs/data_quality.csv` | Row counts, duplicates, missing parent links and date exclusions |
| `outputs/raw_column_profile.csv` | Type, missing rate and distinct count of every raw column |
| `outputs/feature_profile.csv` | Missing rate and distinct count of every feature |
| `outputs/cross_validation.csv` | Score and time per candidate and fold (tuning evidence, not test scores) |
| `outputs/metrics.csv` | Raw/calibrated probabilities, 0.5/F2 thresholds and holdout results |
| `outputs/review_strategies.csv` | Recall, precision, lift and credit coverage at each review capacity |
| `outputs/holdout_*.csv` † | Per-applicant holdout predictions for tracing false positives and negatives |
| `outputs/subgroup_diagnostics.csv` | Descriptive results by gender and income type for Member D |
| `outputs/shap_*.csv`, `outputs/shap_values.parquet` † | SHAP ranking, source summary and per-applicant explanations |
| `outputs/logistic_coefficients.csv` | Logistic Regression coefficients on standardised numeric and one-hot features |
| `outputs/figures/` | English figures ready for the slides and report |
| `outputs/*.joblib` † | Model, calibrator, columns and threshold; only load files you trust |
| `outputs/run_manifest.json` | Parameters, seed, input SHA-256 hashes, row counts, run time and thresholds |
| `outputs/partitions.csv` † | The fit/calibration/holdout partition of each applicant |
| `AI_USAGE.md` | Factual record of AI use and the items the group must confirm before submission |

† Produced by `pipeline.py` but not stored in the Git repository: these files hold per-applicant data derived from the course dataset, or model files that the code rebuilds. Run the pipeline locally to create them. (`outputs/shap_importance.csv` and `outputs/shap_by_source.csv` are stored.)

## Data processing decisions

1. The main table and every aggregate are one row per `SK_ID_CURR`. The main table is left-joined, so applicants without history are kept. Primary keys, foreign keys, duplicate months, ownership conflicts and output IDs are all checked.
2. Bureau monthly records are first aggregated to one row per `SK_ID_BUREAU`, joined to the bureau accounts and then aggregated to the applicant. These means weight each account equally; POS and card means weight each observed row equally, so long-history accounts carry more weight.
3. Instalment, POS and card records are attributed through their own `SK_ID_CURR`, without an inner join to previous applications. The assignment diagram notes that some historical facilities have no matching previous application; these records are kept, and ownership is cross-checked whenever the parent row exists.
4. Exact duplicate export rows are removed and counted (this extract has none). Day placeholders such as `DAYS_EMPLOYED=365243` become missing, and a separate `EMPLOYMENT_SENTINEL` indicator keeps the anomaly visible. `XNA`/`XAP` in the main table become missing. Unknown values are never read as zero risk.
5. Plausible large or extreme amounts are kept; no arbitrary capping is applied. Ratios with a non-positive denominator become missing, and all infinities become missing. CatBoost handles numeric missingness natively; categories use `__MISSING__`. Logistic Regression median imputation, missing indicators, scaling and one-hot encoding are fitted inside each training fold only, and unseen categories are handled at prediction time.
6. Only records available at application time are used: positive or unknown decision dates, bureau opening/update dates and monthly dates are excluded. Realised future drawdown or closure dates are not treated as known history, and `DAYS_LAST_DUE` and `DAYS_TERMINATION` are dropped because their meaning is ambiguous. Contractually known future due dates may be kept, so not every positive date is treated as an error.
7. Instalment amounts are read from the CSV as text and summed and compared with exact `Decimal` arithmetic; there is no rounding to two decimals and no tolerance that would erase a real small shortfall. Completion day, `late` and `shortfall` are decided before converting to model floats. Instalments are first consolidated by applicant, previous loan, instalment version, instalment number and due date. The scheduled amount is taken once per instalment, every payment before the application date is summed, and conflicting scheduled amounts stop the run. `late` is set only when on-time payments are below the scheduled amount. Delay counts to the day cumulative payments reach the amount; unfinished instalments are censored at the application date. Ratios and `late` stay missing when the scheduled amount is missing or non-positive. When a payment amount or date is unknown, `payment_unknown` is set and the ratio, shortfall, `late` and delay stay missing instead of being treated as unpaid. Future instalments are excluded and future payments do not count as paid.
8. The data provide no observation snapshot, cross-application customer identity or absolute dates, so they cannot prove that every source column would be available at decision time in production. Source-timing assumptions and limits are documented; deployment needs fresh point-in-time validation and out-of-time testing.
9. **Personal attributes.** Ten columns stay in the feature tables for outcome monitoring but are never used by a model (`EXCLUDED_FROM_MODELS` in `src/features.py`). `CODE_GENDER` is excluded because sex has no lawful basis as a credit-scoring input under the Equality Act 2010. Age (`DAYS_BIRTH`, `AGE_YEARS`), family fields (`NAME_FAMILY_STATUS`, `CNT_CHILDREN`, `CNT_FAM_MEMBERS`, `INCOME_PER_PERSON`, `NAME_TYPE_SUITE`) and occupation fields (`OCCUPATION_TYPE`, `ORGANIZATION_TYPE`) are excluded under data minimisation, because a rule fixed in advance (use a group only if it raises mean CV AP in at least four of five folds) rejected all three. A `Maternity leave` income type, if it ever appears, is masked because pregnancy and maternity is protected. `verify_outputs.py` and the unit tests fail if a saved model uses an excluded column. See `outputs/audit/AUDIT.md`.

## Feature names and formulas

- Unprefixed columns are raw current-application fields or derived ratios. `AGE_YEARS=-DAYS_BIRTH/365.25`; `EMPLOYMENT_YEARS=-DAYS_EMPLOYED/365.25` (non-positive, non-placeholder values only).
- `CREDIT_TO_INCOME=AMT_CREDIT/AMT_INCOME_TOTAL`, `ANNUITY_TO_INCOME=AMT_ANNUITY/AMT_INCOME_TOTAL`, `CREDIT_TO_GOODS=AMT_CREDIT/AMT_GOODS_PRICE`, `INCOME_PER_PERSON=AMT_INCOME_TOTAL/CNT_FAM_MEMBERS`.
- `EXTERNAL_DEBT_TO_INCOME` is the recorded debt on available bureau accounts divided by income. It is not a complete personal debt ratio, and missing account debt is not treated as zero.
- `PREV`, `INST`, `BUREAU`, `POS` and `CARD` name the source. `__mean`, `__max` and `__min` are the mean, maximum and minimum of that source's numeric fields; `__row_count` counts available rows and `__loan_count` counts distinct previous loans. `__has_history` separates missing history from observed history. Without records, counts are 0 and other statistics stay missing.
- `_6m`/`_12m` windows cover [-183,0]/[-365,0] days for daily data and [-6,0]/[-12,0] months for monthly data, including the application month where provided. `recent_minus_prior6m` is the mean of the recent six-month window minus the mean of the preceding, non-overlapping six months; it is not a regression slope. These change features exist for INST, POS, CARD and bureau monthly history (aggregated to the applicant); PREV has none. Day windows approximate half a year and a year and are not exact calendar months.
- `PREV__status_*__mean` is the share of each application status; `credit_to_requested` is granted credit divided by the requested amount; row counts and date ranges describe application frequency and recency.
- In `INST`, `amount` is the scheduled instalment, `paid` is the amount paid by the application date, `on_time_paid` is the amount paid by the due date, `shortfall=max(amount-paid,0)` and `payment_ratio=paid/amount`; `late` and `delay_days` follow the consolidation above. `observed_payments` counts visible payment rows.
- `BUREAU__active` marks accounts with status Active; `debt_to_credit` is account debt divided by credit amount. Nested `BH` features are computed on each account's monthly history and then aggregated across accounts; for example, `BUREAU__BH__delinquent__mean__mean` is the average, across accounts, of the share of known-status months that were delinquent.
- `BH__delinquent` is 1 for status 1–5 and 0 for status 0. Status C (closed) and X (unknown) are not assumed to be observed non-delinquent months; they get separate closed/unknown indicators.
- In `POS`/`CARD`, `late=SK_DPD>0`. Card `utilisation=AMT_BALANCE/AMT_CREDIT_LIMIT_ACTUAL` and `payment_to_minimum=AMT_PAYMENT_TOTAL_CURRENT/AMT_INST_MIN_REGULARITY`. Utilisation above 1 is kept so that over-limit signals are not erased.
- Full uses every source. Lean uses the current application, previous applications and instalment payments, and drops the bureau tables, POS, card and `EXTERNAL_DEBT_TO_INCOME`. Lean still contains the application's external scores (`EXT_SOURCE_1/2/3`) and bureau enquiry counts (`AMT_REQ_CREDIT_BUREAU_*`), so it is a reduced-table model and not an internal-only model. `audit.py` tests a version without those columns. Neither model uses the excluded personal attributes.
- Every `SK_ID_*` is used only for joining or ordering; `TARGET` is never a feature and there is no target encoding.

## Training and how to read the results

A fixed stratified split gives 60% fit/tune, 20% calibration and 20% holdout: 16,200 applicants for fit and 5,400 for each of the others, leaving enough positives for calibration and testing. Only the fit partition is used for five-fold StratifiedKFold tuning; the holdout plays no part in feature selection, hyperparameters, early stopping, calibration or threshold choice.

- Logistic Regression: L2, C ∈ {0.01, 0.1}, class_weight ∈ {None, balanced}; four candidates.
- Full CatBoost: depth/l2 ∈ {(4,5), (6,8)} × auto_class_weights ∈ {None, SqrtBalanced}; four candidates, 400 trees, learning_rate=0.06. No holdout early stopping. This is a bounded baseline search, not an exhaustive optimisation.
- Candidates are selected by mean five-fold **Average Precision (AP)**. The report column `pr_auc_ap` means AP, not a trapezoidal area under the PR curve. Five-fold scores are tuning evidence; generalisation is judged on the holdout.
- Lean reuses the hyperparameters selected for Full and the same folds, to compare data sources. It was not tuned separately, so these are not Lean's best possible results.
- A sigmoid calibrator is fitted on log-odds in the separate calibration partition, and the threshold that maximises F2 is chosen on the same partition. Scores on that partition may be optimistic and are not an independent test; the threshold is then fixed for the holdout.
- The delivered model is the Full CatBoost and calibrator evaluated on the holdout; the holdout is not added back for retraining. This gives up some training data so that the delivered model is exactly the reported model. The model was not trained on all 27,000 applicants.
- The classification threshold and the human review quota are separate. A Top-K queue always selects ceil(n × capacity) applicants by rank, breaking ties by ascending ID, so its size is fixed and reproducible. It is defined by rank, not by a fixed probability cut-off.
- The exposure ranking score is calibrated risk × requested credit. "Captured amount" is the loan amount of observed difficulty cases; it is not loss, LGD, ROI or savings.
- SHAP uses CatBoost's exact Tree SHAP on a fixed random sample of up to 500 holdout applicants. It explains the uncalibrated model's log-odds, and the code checks that contributions plus the base value equal the raw score. SHAP values are not causal effects or calibrated probability points. In the distribution plot, blue/red show low/high values of numeric features only; grey covers categorical values and missing values and does not mean the whole column is missing.
- The subgroup table reports sample sizes and positive counts; small groups are unstable, and fairness is not claimed to be proven. Small groups without both TARGET classes have no ROC/AP and are omitted.

Technical references: [scikit-learn Average Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html), [CatBoost SHAP API](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_get_feature_importance).

## Submission scope

Give `pipeline.py`, `src/`, `requirements.txt` and `outputs/predictions.csv` to whoever assembles the submission; `src/` must stay with the entry script. Given the raw data path, the code rebuilds every result. The README, tests and verification script can be included to make re-running easier.

Do not package the raw `asteria_student_data/`, the original ZIP, `.venv/`, the integrated features, or per-applicant risk/holdout/SHAP files as submission data. The assignment says not to include datasets, and models can be rebuilt by the code, so the large model files are not needed. The report PDF, slide PDF, video and the final group ZIP are work for Members C–G and the whole group; none of it has been done or submitted on their behalf. Disclose AI use as the assignment requires, and have members confirm that they understand and can explain the code and analysis.

## Revision history

The results were revised twice after independent review: a fix to decimal instalment arithmetic, and the personal-attribute policy described above. `outputs/CORRECTION_REPORT.md` records what changed and the before/after results. Earlier outputs are not kept in the repository.
