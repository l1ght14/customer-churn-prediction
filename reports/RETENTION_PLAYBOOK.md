# Retention Playbook — Telco Churn Model

**Audience:** Head of Customer Success + Retention budget owner
**Model:** Random forest, holdout PR-AUC 0.6421, CV 0.6664 ± 0.0152, selected on
train-only cross-validation
**Ranked list:** `data/processed/churn_scored.csv` — 7,032 customers by out-of-fold risk

> **This is a backtest, not a live prospect list.** `Churn` is an observed historical
> outcome. Every "churner caught" below is someone who already left — the numbers show
> what a campaign aimed at this slice *would have caught*, not who to call tomorrow. To
> run this live you need periodic features and a forward-looking label.

---

## 1. What we found

1,869 customers (26.6%) churned, carrying **$139,131/month** of lost recurring revenue.

The churn is not spread evenly. **87% of that revenue sits in one segment: month-to-month
contract holders.** Two-year contracts churn at 2.8%; month-to-month at 42.7% — a 15x
difference. Every other finding is downstream of that:

- New customers churn hardest: 53.3% within 6 months, 9.5% past 49 months.
- Fiber-optic customers churn at 41.9% vs 7.4% for customers with no internet service.
- Customers paying by electronic check churn at 45.3% vs 15.3% on automatic card.
- Churned customers pay **$74.44/month** on average vs $61.31 for retained customers —
  we are losing the higher-paying half.

## 2. What the model is worth

Two different populations appear below, so they are labelled.

**Holdout set (1,407 rows, 374 of them churners):**

| | |
|---|---|
| Ranking quality (holdout PR-AUC) | 0.6421 vs 0.2658 for random |
| Top-decile lift | 2.82x |

**Full base (all 7,032 rows, scored out-of-fold):**

| | |
|---|---|
| Targeting the top 10% (703 customers) | would have found **534 of 1,869 churners (28.6%)** |
| Hit rate in that group | **76.0%** vs 26.6% baseline |
| Lift | **2.86x** better than picking customers at random |
| MRR in the targeted slice | **$57,899/month** ($43,980 of it already lost) |

Out-of-fold scores, so no customer is graded by a model that saw them in training.

Gradient boosting scores slightly higher on the holdout (0.6513) and is statistically tied
with the selected model (0.6631 ± 0.0093 vs 0.6664 ± 0.0152). Either is defensible; the
difference is not a finding.

## 3. Recommended operating point

**Recommendation: contact the top 20% of the scoring set.** The basis is budget, not
lift: rows below it drop precision under 64% and the hit:miss ratio below 1.8:1. The model
keeps beating random well past this point (deciles 1-4 and the top 30% row), so if you have
budget to go further, take it — this is the point where a retention offer stops being
marginal, not where the model stops working.

The table below is measured on the **1,407-row holdout set**. Each "top N%" is a cut on
the 5,625-row training score set, so the realised holdout share drifts a little above the
nominal (20% is realised as 286 rows = 20.3%).

| Target share of train (realised on holdout) | Customers | Churners found | Recall | Precision | Hit:miss |
|---|---|---|---|---|---|
| Top 5% (78 = 5.5%) | 78 | 63 | 16.8% | 80.8% | 4.2:1 |
| Top 10% (145 = 10.3%) | 145 | 107 | 28.6% | 73.8% | 2.8:1 |
| Top 15% (218 = 15.5%) | 218 | 150 | 40.1% | 68.8% | 2.2:1 |
| **Top 20% (285 = 20.3%)** | **285** | **182** | **48.7%** | **63.9%** | **1.8:1** |
| Top 30% (441 = 31.3%) | 441 | 258 | 69.0% | 58.5% | 1.4:1 |

| Decile (holdout) | Lift vs random | Cum. churners found |
|---|---|---|
| 1 | 2.82x | 28.1% |
| 2 | 2.00x | 48.1% |
| 3 | 1.87x | 66.8% |
| 4 | 1.24x | 79.1% |
| 5–10 | 0.03–0.69x | 100% |

Deciles 1-4 beat random targeting. The bottom half of the holdout ranking (deciles 6-10)
holds only **14.4% of the 374 holdout churners** — against the 50% a random half of the
holdout would turn up.

## 4. The playbook

**Segment 1 — Month-to-month + fiber optic (54.6% churn).** Highest-value target.
Hypothesis to test: offer a 12-month contract at a 15% discount, paired with a free
month of Tech Support. Anchor against the fiber-specific alternative service rather than
a generic discount.

**Segment 2 — Electronic check payers (45.3% churn).** The second-worst rate in the
table, and the cheapest to act on: moving a customer to autopay is a process change, not a
discount, so margin cost is near zero. Hypothesis to test: a one-time $10 credit on
conversion. (This is the *easiest* segment to act on, not the highest-value one —
month-to-month holds more revenue at risk.)

**Segment 3 — First 6 months (53.3% churn).** The highest rate in the table and the one
the model scores well — the highest-risk rows in the ranked list are almost all short
tenure, so this is a rule layered on the score rather than a replacement for it.
Hypothesis to test: trigger an onboarding check-in at day 45, on the reasoning that churn
concentrates in the first months. The dataset has no churn date, so that 45-day timing is
a judgement, not a measurement — validate the window before operationalising it.

**Do not** run retention offers to two-year contract customers (2.8% churn). The discount
cost exceeds the retention value.

## 5. Replaying the backtest

`churn_scored.csv` ranks all 7,032 customers, and it ranks people who have **already
churned**, so the workflow is a replay, not a call list. The top 20% of the ranked file is
1,406 customers, of whom **940 (66.9%) have already churned** — that 67.1% is the hit rate
a real campaign aimed at this slice would have had.

To rehearse the process:

1. Re-score after each data refresh with `python -m src.run_all` (it refits from source;
   the model binary is not committed).
2. Take the top 20% from `churn_scored.csv` (1,406 of 7,032 rows).
3. Check the `already_churned` column against that slice's share: expect ~66.9% true. The
   §3 table shows 64.0% because it is measured on the 1,407-row holdout, a different and
   smaller population — the two numbers are not interchangeable.
4. Route by segment using the `Contract` / `PaymentMethod` / `InternetService` columns.

To go live, retrain against a forward-looking label on a rolling window, then drop the
`Churn` and `already_churned` columns, which only exist because this is a backtest.

## 6. Monitoring

Baselines below are on the full 7,032-row base, matching §2.

| Metric | Baseline | Alert if |
|---|---|---|
| Top-decile lift (out-of-fold, full base) | 2.86x | drops below 2.0x |
| Top-decile churn rate (full base) | 76.0% | drops below 60% |
| PR-AUC on latest month (holdout) | 0.6421 | drops below 0.58 |
| Population churn rate | 26.6% | moves ±3pp |

A fall in top-decile lift with a flat population churn rate means the *model* has decayed,
not the business — retrain before changing the offer.

## 7. What we cannot promise

- **Every offer above is a hypothesis, not a validated intervention.** Contract type may
  be a symptom of customer intent rather than a lever. Validate with an A/B test before
  scaling spend — hold out 10% of the target group as a control and measure *incremental*
  save, not raw retention.
- **These are observed outcomes, not predicted ones.** The 28.6% capture figure is a
  backtest. A live figure would be lower, because some targeted customers would have
  stayed regardless.
- **The two candidate models are tied.** Do not present the model choice as a decisive
  finding; on a different seed the ranking could reverse.
- **The model ranks, it does not price.** It does not know that a $10 discount costs less
  than a $70 customer. Once you supply that cost ratio the optimal threshold moves, and
  the sweep should be re-run rather than reused.
- **No temporal validation.** All splits are random. If churn drivers drift seasonally the
  holdout estimate is optimistic. Hold back the most recent month going forward.
- **The dataset has no reactivation or cancellation-reason field**, so nothing here speaks
  to win-back or to *why* customers left. That needs a second data source.
