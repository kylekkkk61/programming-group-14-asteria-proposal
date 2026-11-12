# Member C — Business strategy and value analysis (working draft)

> **Handover note for Member C and Members F/G.** Every number here comes from the final model, which is also the model behind `predictions.csv`: Full CatBoost on all seven tables, with no personal attributes from the excluded list (sex, age, family fields, occupation). Holdout results use 5,400 applicants with 436 difficulty cases; evaluation results use the 3,000 unlabelled applicants. Anything the data cannot show, such as savings, automatic approval or reduced losses, is labelled as a proposal to test. Keep the list in Section 8 next to whoever writes the slides.

**Terminology.** "Repayment difficulty" means `TARGET = 1`: a payment more than X days late on at least one of the first Y instalments (X and Y undisclosed). It is not a default definition and carries no loss amount.

---

## 1. The business problem and our offer

Asteria lends unsecured credit to applicants with varied financial backgrounds, including some with limited credit histories. In the development data, 8.07% of applicants (2,180 of 27,000) experienced repayment difficulty. When difficulty is this rare, extra manual review spent evenly across all applicants mostly lands on people who would have repaid, and review capacity is limited.

**Our offer:** a calibrated risk score that ranks applicants so that Asteria's limited extra-review capacity goes to those most likely to struggle. The score supports human decisions; it does not approve or decline anyone, and every applicant still goes through Asteria's standard credit and affordability checks, which UK lenders must carry out before lending ([FCA](https://www.fca.org.uk/firms/consumer-credit-sales-and-advice)).

**Built to be lawful from the start.** The score uses financial and repayment information only. Sex is excluded because UK equality law gives no basis for using it in credit; age, family and occupation are excluded because our tests showed they added nothing. This is a selling point for a lender facing FCA scrutiny of AI (see Member D and `audit/AUDIT.md`).

**Headline evidence (holdout):** reviewing the 20% of applicants with the highest scores finds **222 of 436 difficulty cases (50.9%)**; a random 20% would find about 87. Queue precision is 20.6%, 2.55 times the base rate.

| Review capacity | Applicants reviewed | Cases found | Recall | Precision | Lift over random |
|---|---:|---:|---:|---:|---:|
| 10% | 540 | 144 | 33.0% | 26.7% | 3.30× |
| 20% | 1,080 | 222 | 50.9% | 20.6% | 2.55× |
| 30% | 1,620 | 283 | 64.9% | 17.5% | 2.16× |

The 20% capacity is a planning scenario chosen by the team; Asteria's real capacity is unknown, which is why three levels are shown.

---

## 2. Ranking by probability or by exposure

- **Probability ranking:** order applicants by calibrated probability of difficulty.
- **Exposure ranking:** order by probability × requested credit (`AMT_CREDIT`). This is a risk-weighted loan amount, not an expected loss (no loss-given-default or recovery data). The data do not state a currency.

| Strategy | Capacity | Cases found | Recall | Lift | Share of difficulty-case credit covered |
|---|---|---:|---:|---:|---:|
| Probability | 10% | 144 | 33.0% | 3.30× | 30.5% |
| Probability | 20% | 222 | 50.9% | 2.55× | 49.8% |
| Probability | 30% | 283 | 64.9% | 2.16× | 61.7% |
| Exposure | 10% | 132 | 30.3% | 3.03× | 42.3% |
| Exposure | 20% | 188 | 43.1% | 2.16× | 55.5% |
| Exposure | 30% | 241 | 55.3% | 1.84× | 68.9% |

*Credit covered = loan amounts of the difficulty cases in the queue ÷ loan amounts of all difficulty cases. It is not money lost or saved.*

**This is a choice between two goals.** Probability ranking finds more applicants who will struggle; exposure ranking covers more of the credit they hold. At 20% capacity the two queues share 668 applicants. The 412 applicants only in the probability queue have a mean probability of 16.1% and smaller loans (mean 260,309); 61 experienced difficulty. The 412 only in the exposure queue have a mean probability of 8.1%, about the base rate, and much larger loans (mean 1,169,228); 27 experienced difficulty. Exposure ranking picks them for loan size, not for high risk.

**Recommendation:** use probability ranking for the main queue, because the goal is to find applicants at risk of difficulty with limited capacity. If management cares more about the amount of credit at risk, exposure ranking is a valid alternative and the table shows what it costs in cases found. Large loans can also go to senior review through a separate loan-size rule; whether Asteria has such a rule is unknown, so present it as a suggestion.

---

## 3. Two operating points: the review queue and the classification label

**The review queue is defined by rank.** It always takes the top ceil(n × capacity) applicants in the current batch. In the 3,000-applicant evaluation batch the Top 20% queue is exactly 600 applicants, and the lowest probability that made it is 0.115; in the holdout the lowest is 0.116. This cut-off changes from batch to batch, so the queue must not be described as a fixed probability threshold.

**The classification label uses a fixed threshold.** `predictions.csv` marks an applicant as 1 when the calibrated probability is at least 0.0858, the value that maximised the F2 score on the calibration partition (F2 weights recall four times as much as precision). F2 is flat near its peak: between thresholds of 0.08 and 0.09 it moves only from 0.405 to 0.416 while the share flagged moves from 35% to 30%. So the label threshold is a reasonable choice for the assignment's binary output, not a tested risk appetite.

**How they fit together in these batches.** Everyone in the Top 20% queue is also above the label threshold, giving three groups. This nesting holds here but is not guaranteed for future data.

| Group | Holdout applicants | Holdout difficulty rate | Evaluation applicants | Status |
|---|---:|---:|---:|---|
| In the Top 20% queue (by rank) | 1,080 | 20.6% (222 cases) | 600 | Measured |
| Above the label threshold, outside the queue | 583 | 11.3% (66 cases) | 320 | Measured |
| Below the label threshold | 3,737 | 4.0% (148 cases) | 2,080 | Measured |

The 920 positive labels in `predictions.csv` are the first two groups combined (600 + 320).

**Proposed treatments, to be tested in a pilot:**

- *Queue:* extra review of income, commitments and affordability, guided by each applicant's main risk factors.
- *Above threshold, outside the queue:* lighter checks, such as document verification, or review when capacity allows. Their effect on difficulty rates is unknown.
- *Below threshold:* Asteria's standard underwriting. These applicants still have a 4.0% difficulty rate, about half the base rate, so the score alone cannot justify approving them without the usual checks.

---

## 4. Which data to use

| Model | Data | Features | CV AP | Holdout AP | ROC-AUC | Top 10% cases | Top 20% cases |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression (benchmark) | All seven tables | 1,210 | 0.215 | 0.237 | 0.743 | 138 | 213 |
| **Full CatBoost (submitted)** | All seven tables | **1,210** | **0.255** | **0.275** | **0.770** | **144** | **222** |
| Lean CatBoost | Application, previous applications, instalments | 439 | 0.248 | 0.272 | 0.768 | 143 | 226 |
| Lean without external scores | As Lean, minus external scores and bureau enquiry counts | 430 | 0.192 | 0.208 | 0.712 | 125 | 197 |

Paired bootstrap on the holdout (2,000 resamples):

| Comparison | AP difference | 95% interval | Conclusion |
|---|---:|---|---|
| Full − Logistic Regression | +0.036 | [+0.007, +0.067] | CatBoost is better than the linear benchmark |
| Full − Lean | +0.003 | [−0.016, +0.022] | Not distinguishable |
| Full − Lean without external scores | +0.066 | [+0.038, +0.094] | External scores add clear value |

**What this means for Asteria:**

1. **External credit scores are essential.** Without them the model loses 25 cases in the Top 20% queue. A model without external data still beats random review (197 against about 87 cases) but is a degraded mode, not an equal backup.
2. **The four extra tables have not shown a measurable gain.** Full and Lean are within sampling noise of each other in both cross-validation and the holdout.
3. **We submit and recommend Full CatBoost**, because it was the model chosen in the proposal before any results were seen. Lean looks slightly better on some holdout figures, and switching to it after seeing them would be choosing on the test set.
4. **A cheaper route exists if integration is costly.** If bureau account histories, POS and card feeds are expensive to connect, Asteria could launch with the three Lean tables and add the others later, confirming on a larger sample whether they help. Present this as an option, with the evidence above.

---

## 5. Personal attributes: what fairness costs and buys

Full details are in `audit/AUDIT.md`; the business-level summary is:

| | Using all attributes | Final model |
|---|---:|---:|
| Top 20% cases found (holdout) | 232 | 222 |
| AP (holdout) | 0.285 | 0.275 |
| Women queued / men queued | 15.5% / 28.7% | 18.2% / 23.6% |
| Gap in queue rate | 13.2 pts | 5.4 pts |

The 10-case and 0.010-AP differences are within sampling noise (AP interval −0.025 to +0.005). The fairness gain is not: in a model that uses sex, flipping only the recorded sex moves 164 men out of the queue and 287 women into it. For Asteria the message is that a lawful, fairer model costs little or nothing measurable in risk detection.

---

## 6. An illustrative scenario (assumptions stated)

The evaluation batch has no labels, so its true results are unknown. Applying holdout rates to 3,000 applicants **as an illustration only**, with the same 8.07% difficulty rate (about 242 cases) and the same model behaviour:

| Approach | Applicants reviewed | Expected difficulty cases found | Reviews of applicants without difficulty |
|---|---:|---:|---:|
| Random 20% review | 600 | about 48 | about 552 |
| Top 20% by model score | 600 | about 123 | about 477 |

For the same review effort, the model-ranked queue is expected to hold about 2.5 times as many applicants who later struggle. This is about **where review effort goes**. Finding an applicant who will struggle does not by itself prevent the difficulty, and fewer cases in one queue does not mean proportionally fewer staff hours.

**To estimate financial value Asteria would need to provide:** cost per review, loss given difficulty and recovery rates, how often review changes the decision or terms, and revenue lost when a sound applicant is declined or drops out.

---

## 7. Proposed next step: a controlled pilot

1. Run the model in shadow mode on new applications: score and rank without changing decisions, to check stability and capacity fit.
2. Introduce the review queue for part of the applications, keep a comparison group under the current process, and track difficulty rates, approval rates, review time and outcomes by group (sex, age band).
3. Re-test the extra tables and the personal-attribute audit on the larger, later sample.

---

## 8. Claims to avoid in the pitch

| Avoid | Say instead |
|---|---|
| "The model prevents 51% of defaults." | "Reviewing the top 20% finds 51% of applicants who later had repayment difficulty (holdout)." |
| "Auto-approve the bottom 70%." | "Lower-scoring applicants continue through standard underwriting." |
| "Saves 80% of review staff." | "Focuses the same review effort on higher-risk applicants." |
| "Exposure ranking distorts risk." | "Exposure ranking covers more credit but finds fewer cases; it suits a different goal." |
| "Full is better than Lean." | "Full and Lean perform about the same; the extra tables need more evidence." |
| "Removing gender makes the model fair." | "Removing personal attributes halves the gender gap in who is reviewed; we monitor the rest." |
| "Credit coverage = losses avoided." | "Share of credit held by applicants who later had difficulty." |
