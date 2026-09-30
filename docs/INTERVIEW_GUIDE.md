# Interview Guide — Customer Churn Prediction (Telco)

**Read this top to bottom once. Then drill section 7 (the walkthrough script) and section 8 (Q&A).**
Every line reference here is verified against the code. If a line number ever disagrees with
your editor, the code moved — the argument is still valid, the anchor is not.

---

## 1. The 60-second version

> "I built a churn prediction model on 7,000 telecom subscribers where only 26% churn, which
> makes the problem deceptive — most models look good on accuracy while being useless.
>
> The business finding came first: **87% of the revenue at risk sits in month-to-month
> contract holders**, who churn at 42.7% against 2.8% for two-year. Two-year contracts
> churn 15x less.
>
> I compared four models on PR-AUC, not accuracy, and included a dummy baseline so the floor
> was visible. Random forest won on cross-validated score: **0.6421 holdout PR-AUC against
> 0.2658 for a random ranking**.
>
> The part I'm most confident defending is the evaluation. I found and fixed three separate
> ways this project was lying to me — model selection reading holdout labels, thresholds
> fitted on the labels they were scored against, and call-list scores graded by models that
> had memorised them. Each fix is pinned by a test that fails if the bug returns.
>
> Operationally, targeting the top 20% finds 48.7% of churners at 63.9% precision."

**If they ask the one hardest follow-up:** *"how do you know your evaluation isn't fooling
you?"* — go to section 6. That is the strongest part of this project.

---

## 2. What problem am I actually solving?

Frame it as a business decision, not a model.

A telecom company wants to stop losing subscribers. Two questions follow, and they are
different questions:

| Question | Answered by | Not answered by |
|---|---|---|
| Who churned, and what does it tell us? | EDA / segmentation | A model |
| Who is most likely to churn, so we can act? | A ranked score + a threshold | A single accuracy number |

The dataset has one row per customer, 21 columns, and a `Churn` column. It is a **static
snapshot** — there is no date. That single fact shapes every limitation in section 9.

### The headline business finding

| Segment | Churn rate | MRR at risk |
|---|---|---|
| Month-to-month contract | **42.7%** | $120,847 — **87% of all revenue at risk** |
| One year | 11.3% | $14,118 |
| Two year | 2.8% | $4,165 |
| Electronic check | 45.3% | $84,289 |
| Fiber optic | 41.9% | $114,300 |
| InternetService = "No" | 7.4% | $2,302 |

Churn falls from **53.3% in the first 6 months to 9.5% past 49 months**. Churned customers
paid **$74.44/month** vs **$61.31** for retained — we lose the higher-paying half. Worst
combination, month-to-month + fiber optic, churns at **54.6%**.

**What this buys you in an interview:** you can talk about segmentation before you talk about
modelling, which is what a business interviewer's actual question is.

---

## 3. Architecture — what each file is for

Run order is bottom-up. `python -m src.run_all` walks these four stages.

| File | Lines | Responsibility | One-line why it exists |
|---|---|---|---|
| `config.py` | 39 | Paths, seed, target name, categorical schema | One place to change anything global |
| `data_loader.py` | 136 | Load + clean, with the reasoning in the docstring | The cleaning decisions are the analysis |
| `features.py` | 228 | Engineered features, preprocessing, collinearity analysis | Where encoding decisions live |
| `metrics.py` | 238 | Imbalanced metrics, decile lift/gain, thresholds | Every metric defined in code, not imported |
| `eda.py` | 233 | Segment analysis + 7 figures | Answers business questions |
| `train.py` | 210 | 4-model comparison, CV, selection | Where evaluation validity is enforced |
| `evaluate.py` | 275 | Curves, calibration, thresholds, ranked export | Turns a model into a deliverable |
| `run_all.py` | 98 | Orchestration, data bootstrap, stale cleanup | One command, reproducible |

```
data/raw/          Telco-Customer-Churn.csv   (gitignored, auto-downloaded)
data/processed/    churn_scored.csv           ← the deliverable, committed
src/               8 modules
tests/             5 files, 157 tests
reports/           figures/ (13 PNGs), metrics.json, evaluation.json, playbook
```

---

## 4. Line-by-line walkthrough, in execution order

### Stage 0 — `config.py`

```
4   ROOT = Path(__file__).resolve().parents[1]
6   RAW_CSV    = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
7   SCORED_CSV = ROOT / "data" / "processed" / "churn_scored.csv"
12  TARGET = "Churn"
14  RANDOM_STATE = 42
15  TEST_SIZE = 0.2
22  RAW_TEXT_FEATURES = [ ... 16 categorical columns ... ]
```

**Say this:** *"One seed, 42, used for the split, the folds and every estimator. That's why
two runs produce byte-identical output."*

`RAW_TEXT_FEATURES` (line 22) is worth flagging: the categorical schema is *declared here*,
not inferred. `ColumnTransformer` needs a column list, and inferring it means the encoding
silently changes when someone adds a column.

---

### Stage 1 — `data_loader.py` — the cleaning decisions

This is the highest-signal 17 lines in the project. `clean()` is lines 26–42.

```python
26  def clean(df: pd.DataFrame) -> pd.DataFrame:
28      out = df.copy()                                    # never mutate the input
30      text_cols = out.select_dtypes(include=["string", "object"]).columns
31      out[text_cols] = out[text_cols].apply(lambda col: col.str.strip())
33      out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
35      out = out[out["TotalCharges"].notna()].copy()
36      out = out.reset_index(drop=True)
38      out["SeniorCitizen"] = _as_yes_no(out["SeniorCitizen"])
39      out[config.TARGET] = out[config.TARGET].map({"No": 0, "Yes": 1})
```

**Line 33 — the trap everyone falls into.** `TotalCharges` arrives as text because 11 rows
are blank. If you read the file naively, `df.isna().sum()` on that column returns **0** —
the blanks are a single space `" "`, not null. You only see them after the strip on line 31.
`errors="coerce"` turns unparseable values into NaN, which is what finally surfaces them.

**Line 35 — why drop rather than impute.** All 11 have `tenure == 0`: they signed up and
cancelled before any invoice was raised, so they have no revenue history. Imputing 0 would
invent a "free" customer and bias the model. Verified by
`test_data_loader.py:124` and `:72`.

**Say this if asked about the bias:** *"All 11 dropped rows are non-churners, so the
headline churn rate moves from 26.5370% to 26.5785% — plus 0.0415pp. Immaterial, but I'd
rather state it than have you find it."* That honesty is worth more than the number.

**Line 28** — `df.copy()` is deliberate and tested (`test_data_loader.py:44`). A cleaning
function that mutates its argument is a bug waiting to surface two stages later.

---

### Stage 2 — `features.py` — encoding and the collinearity finding

`build_features()` (83–90) adds exactly **two** columns:

```python
86   out["contract_months"] = contract_months(out["Contract"])   # 1 / 12 / 24
89   out["services_count"]   = (out[SERVICE_COLUMNS]=="Yes").sum(axis=1)
```

**Be ready for this question:** *"why only two features? I thought feature engineering was
about adding lots of columns."*

The honest answer is the strongest thing on your CV here. An earlier version built **five**:
`contract_months`, `services_count`, `is_month_to_month`, `is_echeck`, `has_internet`. The
last three are *exact restatements* of columns already in the matrix — `is_month_to_month`
IS the `Contract == "Month-to-month"` one-hot. Least-squares reconstruction of each from
the other design columns had **zero residual**.

`collinearity_report()` (93–148) measures this rather than asserting it:

```
32 design columns, rank 24  →  8 dependent directions
   2 from the engineered columns
   6 from structure inside the data
```

The 6 structural ones are not a bug. A customer with no internet service cannot hold Online
Security, so the "No internet service" level of all six add-on columns is *the same indicator
row* as `InternetService == "No"`. That's real information and cannot be removed without
discarding it.

**`OneHotEncoder(drop="first")`** (line ~196) matters: with `drop=None` every categorical
block's levels sum to 1, which is collinear with the intercept the classifier fits itself. On
this feature set `drop=None` gives **48 columns with 24 dependent directions**;
`drop="first"` gives **32 columns with 8**.

**Why it matters for model choice, not accuracy:** trees split one feature at a time and
both encodings expose the same cut points, so the forest and gradient boosting are
unaffected. Logistic regression is not — under exact collinearity L2 has no unique minimum
and spreads weight across duplicates, so no individual coefficient is readable. That is why
the logistic-regression row is a reference point, not the recommendation.

`make_pipeline()` (205–228) — three things to point at:
- **preprocessor and classifier are one `Pipeline`**, so `cross_val_score` refits the imputer
  and scaler *inside each fold*. Outside the loop, the test fold's distribution leaks into
  training through the scaler's mean and standard deviation.
- **`clone(model)`** — without it the caller's estimator instance becomes the fitted step,
  so two pipelines built from one model overwrite each other's state. Covered by
  `test_pipeline.py:180`.
- **`handle_unknown="ignore"`** — scoring runs against future data that can contain payment
  methods absent from today's training set, and it must not crash.

---

### Stage 3 — `metrics.py` — every metric written out, not imported

You implemented these rather than calling sklearn directly. That is a feature, not
reinvention: the decile lift definition is unusual, and owning it means you can defend it.

**`lift_and_gain()` (75–132).** Sort by predicted risk, split into 10 equal groups:
- `churn_rate_in_decile` = share of that decile that actually churned
- `lift` = that ÷ **overall** base rate — "how many times more churners per contact than random"
- `cum_gain` = % of *all* churners captured through this decile

**Three traps in here, each of which bit me:**
1. **The lift denominator is the overall rate, not the mean of the decile rates.** With
   equal-size bins those are algebraically identical, so an equal-bin test can never tell
   them apart. `test_metrics.py:99` uses 103 rows over 10 bins specifically to make them
   diverge.
2. **`cum_gain` is inclusive of the current decile** (`y_sorted[:hi]`). The off-by-one
   corrupts three of the ten published values. Pinned by `test_metrics.py:83` with a fixture
   that puts a churner exactly on the boundary.
3. **Per-decile lift is not monotone on real data.** Decile 5 is 0.64x and decile 6 is 0.69x
   in this project's own holdout. I originally wrote "decays monotonically" in the docstring;
   it was false and a reviewer caught it. Say this — it shows you test claims, not just code.

**`threshold_for_target_size()` (177–200).** Label-free on purpose. A contact budget is a
business decision, knowable before you know any outcome — which is exactly what a live
scoring pipeline needs. One guard: `max(1, int(round(...)))`, because 1% of 10 rows rounds
to 0 and `top[-1]` is the *minimum* score, which would flag everyone.

**`classification_report_at()` (22–73).** Manual confusion matrix so F2 and the zero-denominator
guards are visible. F2 (β=4) weights recall ~4x precision — right when a missed churn is a lost
customer and a false alarm costs one discount email. It emits `roc_auc`/`pr_auc` as `None` on
a single-class slice rather than NaN, because a bare NaN token is invalid strict JSON.

---

### Stage 4 — `train.py` — where evaluation validity is won or lost

**`out_of_fold_scores()` (61–73).** Returns probabilities from a model that never saw the row
it is scoring, via `cross_val_predict`. This one function is load-bearing for the threshold
logic.

**`evaluate_model()` (76–109).** The critical lines:

```python
89   pipe = make_pipeline(model)
90   pipe.fit(Xtr, ytr)
91   prob_tr = out_of_fold_scores(model, Xtr, ytr)   # ← train scores, out-of-fold
92   prob     = pipe.predict_proba(Xte)[:, 1]       # ← holdout scores
94   thr_size   = metrics.threshold_for_target_size(prob_tr, target_share=0.10)
95   thr_recall = metrics.threshold_for_target_recall(ytr, prob_tr, target_recall=0.80)
```

**Read lines 91 and 94–95 aloud.** The cut comes from the *train* scores; the holdout is only
ever *scored* at it. That one distinction is the difference between a measurement and a
tautology.

**`cross_validate()` (112–120) + the sort (158).** Selection CV runs on `Xtr, ytr` **only**. The
sort key is `cv_pr_auc_mean`, never a holdout metric.

**`candidate_models()` (36–59).** Four models spanning interpretability vs accuracy. Note
`n_jobs=1` on the forest with the comment explaining why: a 300-tree forest over 5 CV folds
was the peak memory cost, and the parallel version was slower in wall clock *and*
non-deterministic in its last float bits, which is exactly what churns a committed artefact's
hash.

**The model comparison table** — know these numbers cold:

| Model | CV PR-AUC (train only) | Holdout PR-AUC | Holdout ROC-AUC | P@0.5 | R@0.5 | Lift |
|---|---|---|---|---|---|---|
| Baseline (prior) | 0.2658 | 0.2658 | 0.5000 | 0.000 | 0.000 | 0.86x |
| Logistic regression | 0.6593 ± 0.0129 | 0.6188 | 0.8352 | 0.488 | 0.794 | 2.61x |
| **Random forest** | **0.6664 ± 0.0152** | **0.6421** | **0.8389** | 0.508 | 0.797 | **2.82x** |
| Gradient boosting | 0.6631 ± 0.0093 | 0.6513 | 0.8399 | 0.633 | 0.535 | 2.77x |

**If asked why random forest and not gradient boosting** (GB has the *higher holdout*
score): *"They're statistically tied — 0.6664 ± 0.0152 vs 0.6631 ± 0.0093. I select on CV
because it's the more stable estimator; the holdout is a single 1,407-row sample. I don't
present that as a decisive win, and section 2 of my limitations says so."*

---

### Stage 5 — `evaluate.py` — model to deliverable

**`score_all_customers()` (163–193).** `cross_val_predict` over all 7,032 rows, so every
customer gets a score from a fold that never saw them. Line 190 rounds to 10 dp before
writing — the forest's float accumulation order varies between runs, and unrounded that made
the committed CSV change hash on every rebuild.

**`threshold_sweep()` (137–161).** `y_prob_train` is a **required positional argument**. It
used to default to `None` and fall back to the scored array, which silently reintroduced the
exact circularity the function exists to prevent. A missing argument should be a `TypeError`,
not a quiet leak.

**`plot_calibration()` (91–106).** The honest question this answers: *can I quote "this
customer has a 76% chance of leaving"?* Good enough to rank, not good enough for a contract.

**The deliverable, `churn_scored.csv`:**

```
customerID, Churn, MonthlyCharges, tenure, Contract, PaymentMethod,
InternetService, churn_score, already_churned
```

`already_churned` exists so nobody misreads this as a prospect list. See section 10.

---

### Stage 6 — `run_all.py` — reproducibility

`fetch_raw_data()` (23–38) downloads the dataset if absent — the raw CSV is gitignored, so
without a bootstrap a fresh clone cannot run. `clean_outputs()` (40–63) deletes generated
artefacts **by glob, not by allowlist**; an allowlist silently rots, because a newly written
report isn't on it and survives the rebuild that should have deleted it. That already happened
once with a stale figure pushing the documented PNG count out of date.

---

## 5. Every headline number, and where it comes from

| Number | Value | Produced by | Pinned by a test? |
|---|---|---|---|
| Rows after cleaning | 7,032 | `data_loader.clean` | Yes — `test_data_loader.py:95` |
| Churners | 1,869 | `config.TARGET` sum | Yes — `test_data_loader.py:95` |
| Churn rate | 26.58% | `data_loader.churn_rate` | Yes — `test_data_loader.py:109` |
| MRR at risk | $139,130.85 | `data_loader.revenue_at_risk` | **No** — see gap note below |
| Segment table | 6 rows | `eda.churn_rate_by` | Partly — `test_pipeline.py:487` covers the 3 contract rows only |
| Tenure bands | 53.3% → 9.5% | `eda.churn_rate_by_tenure_band` | Yes — `test_pipeline.py:499` |
| Model table | 4 rows | `train.run` | Selected model's row only — `test_docs.py:86` |
| Collinearity | 32 / 24 / 8 | `features.collinearity_report` | Yes — `test_features.py:292` |
| Backtest top 10% | 534 of 1,869 | `metrics.score_summary` | Yes — `test_docs.py:150` |
| Campaign table | 5 rows | `evaluate.threshold_sweep` | Yes — `test_docs.py:178` |
| Ranked CSV bytes | SHA-stable | `evaluate.score_all_customers` | Yes — `test_pipeline.py:560` |

**Known coverage gap, stated rather than hidden.** Three of these are only partly pinned.
`revenue_at_risk` is tested on a 4-row toy fixture (`test_data_loader.py:85`) but nothing
asserts the real $139,130.85; the fiber-optic, electronic-check and no-internet segment rows
are computed by the same function as the contract rows, so the code path is covered even
though those three specific values are not pinned; and `test_docs.py:86` quotes only the
selected model's scores, so a future change to the *other three* rows would not fail the
build. Closing these means adding four assertions — a real gap, and the first thing I'd fix
next.

**`tests/test_docs.py` is the unusual one.** 34 tests assert that every number quoted in the
README and the playbook still matches the generated JSON. It exists because docs drift
silently: a model change moves PR-AUC and the prose keeps claiming the old figure.

The sharpest version is `test_no_stale_figures` (line 311). It carries a blocklist of 21
figures that were once correct and are now wrong — `"541 of"`, `"540 of"`, `"28.8%"`,
`"76.5%"`, `"$57,668"`, `"11 PNGs"`, `"69 unit tests"`, `"about 35 seconds"`, `"Segment 4"`,
`"0.5260"` among them — and fails if any of them reappears in either document. A stale figure
is invisible in review because it reads exactly like a fresh one. Blacklisting the specific
old values is the only thing that catches it.

Note what that list implies: the backtest number, the hit rate, the revenue figures, the
figure count, the test count and a threshold value have *all* changed during this project.
Each change followed a model or pipeline change, and each one would have silently left
several wrong numbers in the docs.

**And the blocklist is itself incomplete — that is the lesson.** It listed `"541 of"` while
the README said `"540 of"`, so the stale value walked straight past a guard that was
specifically looking for it. Two nearby wrong numbers, one of which the test was watching for,
and the test still passed. I found it only by re-deriving every figure from the raw data. That
is why "verify the docs against the artefacts from scratch" belongs in the process — a
blocklist only catches the mistakes you already know you made.

---

## 6. The evaluation-validity story — your strongest asset

This is the answer to *"how do you know your numbers are real?"*, and it's a story, not a
defence. Three bugs, each found by adversarial review, each pinned by a test.

### Bug 1 — selection bias, twice

**What I did wrong:** picked the winning model by holdout score and reported that same
holdout score. It's a maximum of four numbers, so it's biased upward.

**First fix — still wrong.** I moved the sort key to the cross-validated score, but left the
CV running on the *full* matrix including the holdout rows. The holdout labels were still
influencing the choice, so the reported holdout score was still not unbiased.

**Honest detail worth volunteering:** with the current feature set the leak does *not* change
the winner — full-matrix CV also ranks random forest first (0.6660 vs 0.6636). An earlier
configuration did flip it. *That is the real point: whether a leak changes the argmax is not
something you can know in advance, which is why you close it rather than check it.*
Pinned by `test_pipeline.py:337` (AST-parses the sort key) and `:144`.

### Bug 2 — threshold circularity

**What I did wrong:** derived the campaign cut from holdout labels, then reported recall at
that cut. Recall is pinned to whatever you targeted, so it measures nothing.

**How I found it:** three different models all reported recall of *exactly* 0.8021 at the 80%
target. Independent models cannot land on the same number by chance. That's the fingerprint.

**Second layer:** the cut was derived from *in-sample* train scores — scores from a model that
had partly memorised those labels. Measured cost: the in-sample-derived cut (0.5238) reaches
only **0.7727** holdout recall, missing its own 0.80 constraint by 2.7 points. Out-of-fold
train scores give **0.8048**.

### Bug 3 — the call list was graded by a model that memorised it

**What I did wrong:** scored all 7,032 customers with a model fitted on 80% of them, then
reported a 3.08x lift. `cross_val_predict` gives the honest **2.86x**.

### And the one that stayed hidden

A reviewer mutation-tested my own suite and found real defects passing, most importantly that
**no test executed the pipeline at all** — every test checked an individual function in
isolation, so a wrong ordering inside the actual stage went undetected. `tests/test_pipeline.py`
(577 lines) now runs the real stage.

Then a second reviewer found the fix itself was bypassable - inserting `ytr = yte` leaves the
call reading `threshold_for_target_recall(ytr, prob_tr, ...)`, so every name-based check stayed
green while the leak went live. Two guards now close it:
- a **data-flow** guard rejecting any second assignment to a split variable inside a function
  (`test_pipeline.py:439`)
- a **behavioural** guard recomputing the cut from the model and comparing to the recorded
  value (`test_pipeline.py:415`). This was needed because the honest and leaked cuts land
  almost on top of each other - 0.4922 vs 0.4939 - and reach recall of 0.8048 vs 0.8021, a
  0.8pp gap. Recall cannot separate them, so the guard is written against the **threshold
  value**, which separates cleanly.

That is the general lesson worth stating: **when a reported metric is nearly identical under
the honest and the cheated path, the metric is not evidence. Test the underlying quantity.**
Deliberately breaking code in known ways and checking the suite catches it - that is mutation
testing, and it found more real problems here than code review did.

---

## 7. Live code walkthrough script

If they say *"walk me through the code"*, do this. Ten minutes, this order.

| # | Open | Say |
|---|---|---|
| 1 | `README.md` headline | The 60-second version. Establish scope and the one number that matters. |
| 2 | `src/data_loader.py:26-42` | The cleaning trap. `isna()` returns 0 on the blanks. Show it. |
| 3 | `src/config.py:22` | Schema is declared, not inferred. |
| 4 | `src/features.py:83-90` | Two engineered columns, and why the other three were removed. |
| 5 | `src/features.py:93-148` | `collinearity_report` — 32 columns, rank 24. Run it live if you can. |
| 6 | `src/features.py:205-228` | `make_pipeline` — clone, fused preprocessor, unknown categories. |
| 7 | `src/metrics.py:75-132` | Decile lift. Three traps. |
| 8 | **`src/train.py:89-95`** | **The centrepiece.** Train scores → cut → holdout scoring. |
| 9 | `src/evaluate.py:163-193` | Out-of-fold ranking + the rounding for reproducibility. |
| 10 | `tests/test_pipeline.py:275` | The test that runs the real stage. Proves the claim isn't just prose. |

**If they ask you to run something, run this** (the full rebuild takes roughly two minutes,
the test suite about one, so budget three):

```bash
python -m src.run_all
python -m unittest discover -s tests -t .
```

Then say: *"157 tests, 34 of which assert that the numbers in my README still match the
generated JSON. If a model change moves a figure, the build breaks."*

---

## 8. Question bank

### On the business problem
**"What's the single most useful thing you found?"**
87% of revenue at risk sits in month-to-month contracts, churning at 42.7% vs 2.8% for
two-year. That reframes retention from "reduce churn" to "move customers off monthly billing."

**"Would you actually run this?"**
Not as written, and I say so in the README. `Churn` is a historical label, so this is a
backtest — 534 of the 703 highest-risk customers have already left. To prospect I need
periodic features and a forward-looking label. What I would ship first is the *segmentation*,
which needs no model.

**"What's the retention play?"**
Three segments in `reports/RETENTION_PLAYBOOK.md`, each labelled a hypothesis. Month-to-month
+ fiber optic (54.6%) gets a contract-migration offer; electronic-check payers (45.3%) get an
autopay nudge, which is a process fix not a discount; first-6-months (53.3%) get a day-45
onboarding check-in. I would not spend on two-year contracts at all. **Every offer needs an
A/B test** — contract type may be a symptom of intent rather than a lever.

### On metrics
**"Why PR-AUC and not ROC-AUC?"**
Be precise — the usual answer is wrong. It is **not** that a zero-recall model looks good on
ROC-AUC: it doesn't, that scores exactly 0.5, and my own dummy baseline is recorded at 0.5000
with recall 0.0000. The real argument is the *baseline*. Random scores 0.2658 on PR-AUC — the
base rate — but 0.5000 on ROC-AUC, so ROC-AUC can't distinguish "learned nothing" from
"learned a little". Precision also isn't diluted by the 73% majority class, which is what
makes my threshold sweep readable as "what fraction of my contacts are worth it".

**"Why is accuracy useless here?"**
At 26% positives, "predict nobody churns" scores 73.4%. I include the dummy baseline in the
model table specifically so the floor is visible rather than theoretical.

**"What is decile lift?"**
Sort by predicted risk, split into ten groups, divide each group's churn rate by the overall
base rate. Top decile 2.82x means contacting those customers finds 2.8x more churners per
contact than picking at random. The bottom half of the holdout ranking holds 14.4% of churners
against the 50% a random half would find.

### On method
**"How did you avoid fooling yourself?"** → Section 6. This is your best answer.

**"Why a random split and not a time-based one?"**
Honest answer: no temporal split is a real limitation, listed first among them. A snapshot
with no date column can't support a time-based split. In production I'd hold back the most
recent month. Out-of-fold is not out-of-time — every fold shares one distribution.

**"Why only two engineered features?"** → Section 4, Stage 2. The collinearity finding.

**"How would you deploy this?"**
Weekly batch scoring, not real-time. Monitor top-decile lift (2.86x baseline, alert below
2.0x) and top-decile churn rate (76.0%, alert below 60%). A fall in lift with a flat
population churn rate means the *model* decayed, not the business — retrain before changing
the offer. Refit from source on a schedule; nothing here needs a saved pickle.

**"How would you improve it?"** → Section 11.

---

## 9. Limitations — say these before they find them

1. **It's a backtest, not a live pipeline.** `Churn` is observed history. 534 of the 703
   targeted customers have already churned.
2. **The top two models are statistically tied.** Don't present the selection as decisive.
3. **No temporal split.** Cannot detect drift.
4. **No causal claims.** The model finds *who*, not *why*. Contract type is both symptom and
   cause; recommending annual plans needs an experiment.
5. **Calibration is approximate.** Good for ranking, not for quoting a probability in a contract.
6. **No cost model.** A false positive costs a discount, a false negative costs a customer.
   The optimal threshold depends on that ratio, which the business hasn't given me — so the
   sweep exposes the trade-off instead of hard-coding an answer.
7. **Out-of-fold is not out-of-time.**

---

## 10. Traps — what NOT to say

Each of these is a claim I made and had to retract. Saying them now would be caught.

| ❌ Don't say | ✅ Say instead |
|---|---|
| "ROC-AUC flatters imbalance because the TN axis is 2.8x larger, so a zero-recall model still looks respectable" | A zero-recall model has ROC-AUC exactly 0.5. The real argument is the baseline difference. |
| "Per-decile lift decays monotonically" | It decays *on average*. Decile 5 is 0.64x and decile 6 is 0.69x in my own holdout. |
| "Contact the top 20% — the last slice that clearly beats random" | Top 20% is a *budget* decision. Deciles 1–4 beat random, so go further if you have budget. |
| "The model found 28.6% of customers will churn" | It found 28.6% of *past* churners in the top 10%. Observed, not predicted. |
| "Model selection uses the full dataset for stability" | Selection CV uses the train split only. The holdout is untouched until one final report. |
| "Feature importance shows tenure is the key driver" | I don't compute feature importance. The *lift table* shows tenure-band behaviour descriptively. |
| "Random forest is clearly the best model" | It wins on CV by less than one standard deviation. GB wins on holdout. They're tied. |

---

## 11. If they ask you to extend it

Have an answer ready for each. These are the obvious follow-ups.

**Add a temporal split.** Can't with this data — no date column. The honest version: with a
dated dataset, hold back the most recent month, train on everything before, and report
whether lift decays. That's the first thing I'd do with real data.

**Add a cost model.** Take `cost_of_false_positive` (a discount) and
`cost_of_false_negative` (a lost customer), then pick the threshold maximising
`recall × saved_value − precision × offer_cost`. `threshold_sweep` already gives the
frontier; it needs one more column.

**Make the segmentation causal.** Run an A/B test on the month-to-month migration offer with
a 10% holdout, and measure *incremental* save, not raw retention. That's the difference
between a ranked list and a business result.

**Add interpretability properly.** Permutation importance on the holdout — out-of-sample, so
it isn't the in-sample impurity shortcut. Honest caveat: the matrix is rank-deficient by 8
directions for structural reasons, so treat correlated features as a group.

**Explain a single prediction.** SHAP values. Caveat again: correlated features split credit
arbitrarily, so explain the *group*, not the column.

---

## 12. Rehearsal checklist

- [ ] Say the 60-second version without looking. Timed at 60s.
- [ ] Recall all four model rows and know which column selected the winner.
- [ ] Explain `TotalCharges` blanks without notes.
- [ ] State the collinearity finding as 32 / 24 / 8 with the two causes.
- [ ] Tell the three-bug evaluation story end to end, including the one that stayed hidden.
- [ ] Know the top-20% operating point: 285 customers, 48.7% recall, 63.9% precision.
- [ ] Name three limitations before you're asked.
- [ ] Run `python -m src.run_all` and `python -m unittest` from memory.
- [ ] Be able to explain why you're **not** presenting it as a live model.
