# Customer Churn Prediction — Telco (Analytics + ML)

Predict which telecom subscribers churned, rank them by risk, and backtest how much of
past churn a targeted retention campaign would have caught.

**Stack:** Python · pandas · scikit-learn · matplotlib · **157 tests** (unit + integration)
**Dataset:** IBM Telco Customer Churn — 7,043 customers × 21 features (downloaded by the pipeline)
**Selected model:** Random forest — holdout PR-AUC 0.6421, top-decile lift 2.82x (holdout, n=1,407)

> **Read this first:** `Churn` in this dataset is an **observed historical outcome**, not a
> future event. The ranked list is a **backtest**: it shows which customers a campaign
> aimed at the top 10% *would have caught*, given who actually left. 534 of the 703
> highest-risk customers have already churned. It is not a prospect list, and no claim
> here is a forecast of who will leave next month.

---

## Headline result

| Model | CV PR-AUC (5-fold, train only) | Holdout PR-AUC | Holdout ROC-AUC | Precision @0.5 | Recall @0.5 | Top-decile lift |
|---|---|---|---|---|---|---|
| Baseline (predict the prior) | 0.2658 | 0.2658 | 0.5000 | 0.000 | 0.000 | 0.86x |
| Logistic regression | 0.6593 ± 0.0129 | 0.6188 | 0.8352 | 0.488 | 0.794 | 2.61x |
| **Random forest** | **0.6664 ± 0.0152** | **0.6421** | **0.8389** | 0.508 | 0.797 | **2.82x** |
| Gradient boosting | 0.6631 ± 0.0093 | 0.6513 | 0.8399 | 0.633 | 0.535 | 2.77x |

Holdout PR-AUC 0.6421 against 0.2658 for a random ranking.

**Note on the table:** gradient boosting has the higher *holdout* score (0.6513) and the
higher ROC-AUC, while random forest is selected on the cross-validated score. That is not
a contradiction — the two columns answer different questions. CV touches every training
row five times and is the more stable estimator; the holdout is a single 1,407-row sample.
The two are within one standard deviation of each other (0.6664 ± 0.0152 vs 0.6631 ±
0.0093), so the honest summary is that these two models are effectively tied and the
ranking between them is not meaningful.

**Backtest at the top 10%:** 703 customers targeted → **534 of 1,869 churners caught
(28.6%)**, hit rate **76.0%** vs a 26.6% baseline, **2.86x** lift. All out-of-fold, on the full
7,032-row base. The 2.82x in the table above is the holdout decile-1 figure; the two are
different populations.

---

## The problem in numbers

- 1,869 of 7,032 customers churned (**26.6%**)
- **$139,131/month** of recurring revenue sat on churned accounts
- Churn is not spread evenly — it concentrates hard in one segment

| Segment | Churn rate | MRR at risk |
|---|---|---|
| Month-to-month contract | **42.7%** | $120,847 (87% of total) |
| One year | 11.3% | $14,118 |
| Two year | 2.8% | $4,165 |
| Electronic check | **45.3%** | $84,289 |
| Fiber optic | **41.9%** | $114,300 |
| InternetService = "No" | 7.4% | $2,302 |

Churn by tenure: **53.3%** in the first 6 months, falling to **9.5%** past 49 months.
Worst combination — month-to-month + fiber optic — churns at **54.6%**.

---

## Data quality decisions

| Issue | Rows | Decision | Why |
|---|---|---|---|
| `TotalCharges` blank | 11 | **Dropped** | All have `tenure = 0`: signed up, cancelled before any invoice. Imputing 0 would invent a free customer. |
| `SeniorCitizen` as 0/1 | — | Cast to "Yes"/"No" | It is a category, not a quantity. Keeps one-hot treatment consistent. |
| `TotalCharges` whitespace-only | 11 | Stripped, then coerced to numeric | The blanks are `" "`, **not** NaN — `isna()` returns 0. Only visible after `.str.strip()`. |
| Duplicate `customerID` | 0 | Verified, none | Duplicate ids mean train/test leakage. Asserted on every run. |

**Honest caveat:** all 11 dropped rows are non-churners, so the headline churn rate moves
from 26.5370% (raw) to 26.5785% (clean) — **+0.0415pp**. Immaterial, but stated rather
than hidden.

---

## Layout

```
data/raw/          Telco-Customer-Churn.csv   (gitignored, fetched automatically)
data/processed/    churn_scored.csv           <- the ranked deliverable, committed
src/
  config.py        paths, constants, categorical schema
  data_loader.py   load + clean, with the reasoning in the docstring
  features.py      engineered features, preprocessing, collinearity analysis
  metrics.py       imbalanced-class metrics, decile lift/gain, thresholds
  eda.py           exploratory analysis + 7 figures
  train.py         4-model comparison, 5-fold CV, model selection
  evaluate.py      curves, calibration, threshold sweep, out-of-fold ranking
  run_all.py       one command, all four stages
tests/             unit tests + doc-vs-artefact guards
reports/
  figures/         13 PNGs
  metrics.json     full model comparison results
  evaluation.json  curves, threshold sweep, backtest summary
  RETENTION_PLAYBOOK.md
```

---

## Run it

```bash
pip install -r requirements.txt
python -m src.run_all                        # ~2 min, downloads data, rebuilds everything
python -m unittest discover -s tests -t .   # full suite
```

`run_all` **downloads the dataset** if it is absent (it is gitignored), clears every
generated artefact so nothing stale survives, then runs clean → EDA → train → evaluate.
The suite includes `tests/test_docs.py`, which fails if any figure quoted in this file or
in the playbook stops matching the generated JSON.

---

## Engineering choices worth defending

**Preprocessing lives inside the pipeline.** The imputer, scaler and one-hot encoder are
steps of the same `Pipeline` as the classifier, so `cross_val_score` refits them inside
each fold. Had they sat outside, the test fold's distribution would leak into training
through the scaler's mean and standard deviation.

**Selection CV runs on the train split only.** Running it over the full matrix would let
the 1,407 holdout labels influence which model wins, which makes the reported holdout
score an overestimate.

Worth being precise here, because the honest finding is less dramatic than the folklore.
In the current configuration, the scope does **not** change the winner: full-matrix CV also
ranks random forest first (0.6662 vs gradient boosting 0.6636). But an earlier
configuration *did* flip it, and that is the real point — whether a leak changes the
argmax is not something you can know in advance, which is exactly why you close it rather
than check it. A leak that happens to be harmless this time is still a leak.

`tests/test_docs.py::test_selection_used_train_only_cv` pins the fix.

**Thresholds come from out-of-fold train scores, and are measured on the holdout.** Two
distinct mistakes are designed out:

1. Deriving a cut from holdout labels and then reporting recall at that same cut.
   Circular — recall is pinned to whatever you targeted. The fingerprint was every model
   reporting an identical recall at the same target, which three different models cannot
   do honestly.
2. Deriving it from *in-sample* train scores, where the model has partly memorised the
   labels. Measured cost: for the selected random forest the in-sample-derived cut
   (threshold 0.5238) reaches only **0.7727** holdout recall against its 0.80 target,
   missing by 2.7 points. With out-of-fold train scores the same model measures
   **0.8048**.

**The design matrix is rank-deficient, and the cause is measured not asserted.**
`src/features.py::collinearity_report` computes it: **32 design columns, rank 24**, so 8
dependent directions, from two unrelated sources.

- **2 from the engineered columns.** `contract_months` (1/12/24 by contract) and
  `services_count` (a sum of the seven service one-hots) are exact linear combinations of
  columns already present. An earlier version also built `is_month_to_month`, `is_echeck`
  and `has_internet`, which were exact restatements of the `Contract`, `PaymentMethod` and
  `InternetService` one-hots — zero residual under least-squares reconstruction. They were
  removed.
- **6 from structure inside the data.** A customer with no internet service cannot have
  Online Security, so the "No internet service" level of all six add-on columns is the same
  indicator row as `InternetService == "No"`. This is real information and cannot be removed
  without discarding it.

`OneHotEncoder(drop="first")` matters here: with `drop=None` every categorical block's
levels sum to 1, which is collinear with the intercept the classifier fits itself. On this
feature set `drop=None` gives **48 design columns with 24 dependent directions**; switching
to `drop="first"` gives **32 columns with 8 dependent directions** — same data, same
engineered features, only the encoder changed.

**Why that matters for model choice, not accuracy.** Trees split on one feature at a time
, and both encodings expose the same cut points, so gradient boosting and the forests are
unaffected. Logistic regression is not: under exact collinearity L2 has no unique minimum
and spreads weight arbitrarily across duplicate columns, so no individual coefficient is
readable. That is why the logistic-regression row is a reference point, not the
recommendation.

**Selection metric is PR-AUC, not accuracy and not ROC-AUC.** Accuracy is useless here:
"predict nobody churns" scores 73.4%. ROC-AUC is better, but still the wrong yardstick, and
the reason is worth stating precisely because the usual one is wrong — it is *not* that a
zero-recall model scores well. It does not: a model that misses every churner has ROC-AUC
exactly 0.5, and this project's own dummy baseline is recorded at 0.5000 with recall
0.0000. The actual argument is about the two baselines. A random ranking scores 0.2658 on
PR-AUC (the base rate) but 0.5000 on ROC-AUC, so on ROC-AUC a model that learned nothing
looks exactly like one that learned a little, while on PR-AUC the floor sits where the
class balance puts it. Precision is also not diluted by the 73% majority class, which is
what makes the threshold sweep interpretable as "what fraction of my contacts are worth
it".

**Ranked scores are out-of-fold.** `cross_val_predict` gives all 7,032 customers a score
from a fold that never saw them. Scoring everyone with the fitted model inflates the
top-10% lift from an honest 2.86x to an unearned 3.08x, because 80% of rows were in
training.

**`handle_unknown="ignore"`** on the one-hot encoder: scoring runs against future data
that can contain payment methods absent from today's training set, and it must not crash.

**`customerID` is dropped before fitting.** It is a label, not a signal — keeping it lets
the model memorise rows.

**What was deliberately not built:** no `TotalCharges / tenure` feature (it is
`MonthlyCharges` by construction — the same collinearity argument), and no hyperparameter
grid search (the top two models are within one standard deviation; tuning would cost
minutes and buy noise on a 7k-row dataset).

---

## Known limitations

1. **The deliverable is a backtest, not a live scoring pipeline.** `Churn` is historical.
   To prospect, you need periodic features and a forward-looking label.
2. **The top two models are statistically tied.** Random forest is selected on CV, but
   gradient boosting wins on the holdout. Do not present the selection as a clear win.
3. **No temporal split.** All splits are random, so this cannot detect drift. A real
   deployment holds back the most recent month.
4. **No causal claims.** The model finds who churned, not why. Contract type and payment
   method are both symptoms and causes. "Move customers to annual plans" needs an
   experiment.
5. **Calibration is approximate.** Good enough to rank; not good enough to quote "this
   customer has a 76% chance of leaving" in a contract. See
   `reports/figures/calibration_curve.png`.
6. **No cost model.** A false positive costs a discount, a false negative costs a
   customer. The optimal threshold depends on that ratio, which the business has not
   supplied — so the sweep exposes the trade-off rather than hard-coding an answer.
7. **Out-of-fold is not out-of-time.** Every fold shares one distribution.

---

## Interview questions this project answers

- *"Walk me through an analysis you did"* → the 87%-of-revenue-at-risk concentration
- *"How do you handle class imbalance?"* → PR-AUC over accuracy, dummy baseline, decile lift
- *"How do you avoid fooling yourself?"* → train-only CV selection, out-of-fold scores and
  out-of-fold thresholds, measured rank deficiency
- *"How do you pick a threshold?"* → campaign size is a business constraint, not an F1 optimum
- *"How would you deploy this?"* → weekly batch scoring, monitor decile-1 lift, retrain monthly
- *"What would break this?"* → no temporal split, no cost asymmetry, historical labels,
  two models tied within one SD
