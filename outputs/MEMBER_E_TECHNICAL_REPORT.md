# Member E — Technical report working draft

> **Handover note for Member E.** Part A is a complete draft of the report body for the final model, which is also the model behind `predictions.csv`. Part B covers the word count, figures and the decisions already made. Part C is an AI-use declaration template; it is excluded from the word limit but must be completed truthfully by the group. Every number in Part A comes from files in `outputs/` or `outputs/audit/`.

---

## Part A — Report body (767 words including table text)

# Predicting Repayment Difficulty to Prioritise Loan Review at Asteria Finance

## 1. Business understanding

Asteria Finance lends unsecured credit to applicants with varied financial backgrounds, including thin credit files. Only 8.07% of development applicants (2,180 of 27,000) experienced repayment difficulty (TARGET = 1: a payment more than X days late within the first Y instalments; X and Y are undisclosed). Following CRISP-DM, we framed the task as decision support: a calibrated score ranks applicants so that limited extra-review capacity reaches those most likely to struggle, while every applicant still passes standard affordability checks. We therefore judged models on ranking quality (Average Precision, AP) and recall within a fixed review queue, rather than accuracy.

## 2. Data engineering

All seven tables were joined into one row per applicant (27,000 development; 3,000 evaluation). Design choices targeted leakage, correctness and lawful use:

- **Point-in-time filtering.** Only records dated on or before the application are used; post-application bureau updates, realised closure dates and future payments are excluded, and unfinished instalment delays are censored at the application date.
- **Hierarchical aggregation.** Monthly bureau history is summarised per account before aggregation to the applicant. Six- and twelve-month windows and recent-versus-prior changes capture deterioration in instalment, POS, card and bureau behaviour.
- **Exact instalment settlement.** Split payments are consolidated with exact decimal arithmetic; an independent check of all 1,078,490 due instalments found no discrepancies.
- **Feature families.** Affordability ratios, past application outcomes, instalment shortfall, lateness and delay, bureau debt-to-credit and delinquency, and card utilisation.
- **Cleaning.** The 365243 day placeholder becomes missing, with an indicator kept; non-positive denominators give missing ratios; histories lacking a parent application are kept through their own applicant ID.
- **Personal attributes.** Sex is excluded from scoring because UK equality law gives no basis for it in credit decisions. Age, family and occupation fields were tested and excluded under data minimisation because adding them back gave no consistent cross-validated gain. All are kept for monitoring, leaving 1,210 model features.

## 3. Modelling and validation

A locked stratified split gave 16,200 applicants for fitting and five-fold cross-validation (CV), 5,400 for calibration and threshold choice, and 5,400 for a final holdout. Each model had four candidates varying regularisation or tree depth and class weighting, selected by mean CV AP. We compared an L2 Logistic Regression benchmark, CatBoost (gradient-boosted trees handling non-linearity, missing values and categories natively), and Lean CatBoost on three tables (application, previous applications, instalments).

Table 1. Holdout results; the Top 20% queue is the 1,080 highest-scored applicants.

| Model | CV AP | Holdout AP | ROC-AUC | Brier | Top 20% recall |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.215 | 0.237 | 0.743 | 0.0695 | 48.9% |
| Full CatBoost (submitted) | 0.255 | 0.275 | 0.770 | 0.0670 | 50.9% |
| Lean CatBoost | 0.248 | 0.272 | 0.768 | 0.0672 | 51.8% |
| Using all personal attributes* | 0.257 | 0.285 | 0.772 | 0.0666 | 53.2% |
| No external scores* | 0.192 | 0.208 | 0.712 | 0.0700 | 45.2% |

*Audit variants using the submitted model's settings.

Paired bootstrap intervals show that CatBoost beats Logistic Regression (AP +0.036, 95% CI +0.007 to +0.067) and that external scores matter (removing them costs 0.066, CI 0.038 to 0.094). Full and Lean are indistinguishable (+0.003, CI −0.016 to +0.022), so Full, chosen before results were seen, is kept. Sigmoid calibration corrected class-weighted raw probabilities (Brier 0.080 to 0.067); reliability curves track the diagonal (Figure 1). The label threshold for predictions.csv (p ≥ 0.0858) maximises F2 on the calibration partition, weighting missed cases above extra reviews; F2 is flat near this point, so it is not a risk-appetite cut-off.

## 4. Business use, fairness and limits

Ranking by probability, a 20% review queue finds 222 of 436 holdout difficulty cases, against about 87 from random review (lift 2.55; Figure 2). Ranking by probability × loan amount covers more difficulty-case credit (55.5% versus 49.8%) but finds fewer cases (188), so we recommend probability ranking and present exposure as an alternative. The queue is defined by rank (600 evaluation applicants); the label threshold marks 920 as positive. Automatic approval was not tested.

Exact Tree SHAP on 500 holdout applicants ranks the external scores highest, then employment length, credit-to-goods price, bureau debt-to-credit and past refusals; these are associations, not causes. Compared with a model using all personal attributes, the submitted model loses 0.010 AP (CI −0.025 to +0.005) but cuts the gap in queue rates between men and women from 13.2 to 5.4 points. In a model that uses sex, changing only an applicant's recorded sex moves 164 men out of the queue and 287 women into it. The remaining gap partly reflects higher observed difficulty among men (10.6% versus 6.8%) and must be monitored.

Model, hyperparameter and feature choices were made by cross-validation or legal requirement; the holdout was used only to report results. Limits: the holdout is one random split rather than a later period, the outcome definition is undisclosed, and the data have relative dates only, so point-in-time and out-of-time validation are needed before deployment.

---

## Part B — Notes for assembling the final report

**Word count.** The limit is 1,000 words. Part A has 767 words counting prose, headings and table cells. Check the Student Handbook for whether tables, captions and references count, and re-count in Word after editing. The remaining room is for references and Member E's own wording, not new claims.

**Figures.** The body refers to two:

- **Figure 1:** `outputs/figures/evaluation.png` (holdout precision-recall, ROC and calibration curves).
- **Figure 2:** `outputs/figures/review_capacity.png` (cases and credit covered at 10/20/30% capacity, probability versus exposure).

Optional if space allows: `outputs/figures/shap_importance.png`.

**Decisions reflected in Part A:**

1. One model throughout: Full CatBoost without personal attributes produces the predictions and is the model recommended to Asteria.
2. Probability ranking is the main strategy; exposure ranking is an alternative for a different goal.
3. The review queue (by rank) and the label threshold (F2) are separate operating points.
4. Lean is reported as equivalent and as a cheaper option, not as the recommended model, because the proposal fixed Full before results were seen.
5. Sex is excluded by law; age, family and occupation by a pre-set data-minimisation rule. `audit/AUDIT.md` holds the evidence and the checked legal sources.

**What the report must not claim** (see Member C, Section 8): prevented defaults, savings, automatic approval, Lean as a backup without external data, Full being better than Lean, or a fair model rather than a fairer one.

---

## Part C — Generative AI use declaration (template, excluded from word count)

The assignment requires the group to state **why** AI was used, **what** for, **which** AI, and **where** generated content appears, and to confirm intellectual ownership. Fill in the bracketed items from your own records, and do not sign the final statement until every member has reviewed and can explain the work.

- **Why:** To help implement and verify a reproducible data-engineering and modelling pipeline across seven tables, test the use of personal attributes, check legal sources, and draft structured analysis notes for the business, risk and report sections.
- **What:** Code for data preparation, feature engineering, modelling, calibration, SHAP, evaluation and the attribute audit (`pipeline.py`, `src/`, `audit.py`, `verify_outputs.py`, `tests/`); verification checks; and first drafts of the analysis documents in `outputs/`, which members then reviewed and edited. [Add other uses, such as slide drafting or proofreading.]
- **Which:** OpenAI Codex desktop app; ChatGPT for the earlier proposal discussion. [Confirm product and model names from your session records; more than one model was used across sessions.]
- **Where:** The submitted Python code was written with AI assistance and run locally on the supplied data to produce `predictions.csv` and the figures. [List the report sections and slides that keep AI-drafted text, and how members changed them.]
- **Ownership:** [To be confirmed by all members after review:] We have reviewed, tested and understood the submitted code, methods, results and recommendations, and we take intellectual ownership of this work.
