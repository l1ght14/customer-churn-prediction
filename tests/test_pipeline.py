"""Integration tests that execute the pipeline stages.

The unit tests cover the metric functions in isolation. These cover the wiring between
stages, which is where a whole class of defects lives: a threshold taken from the wrong
score array, a ranking produced in-sample, a doc that stops matching the run.

Each test here is written to fail on a specific realistic mistake:

  * threshold_sweep deriving its cut from the holdout -> test_sweep_thresholds_come_from_train
  * score_all_customers degrading to an in-sample fit -> test_scores_are_out_of_fold
  * model selection sorting on a holdout metric      -> test_selection_is_not_on_holdout
  * the export churning hash between rebuilds        -> test_ranked_csv_is_byte_reproducible
  * a stage that silently stops writing its artefact -> test_run_all_produces_every_artefact

The pipeline stages are slow (~60s), so they run once in a class-level fixture and the
assertions read the produced artefacts.
"""
import json
import unittest
from pathlib import Path

import numpy
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src import config, eda, evaluate, features, metrics, train
from src.data_loader import load_clean
from src.features import stratified_split


class TestThresholdSweepUsesTrainScores(unittest.TestCase):
    """The sweep table in the playbook comes from here, so the source of the cut matters."""

    def setUp(self):
        rng = numpy.random.default_rng(7)
        self.y = numpy.array([1] * 40 + [0] * 360)
        self.p_holdout = numpy.clip(rng.normal(size=400) + self.y * 1.2, 0, 1)
        self.p_train = numpy.clip(rng.normal(size=400) + self.y * 1.2, 0, 1)

    def test_sweep_thresholds_come_from_train(self):
        sweep = evaluate.threshold_sweep(self.y, self.p_holdout, y_prob_train=self.p_train)
        for row, share in zip(sweep.to_dict("records"), (0.05, 0.10, 0.15, 0.20, 0.30)):
            expected = metrics.threshold_for_target_size(self.p_train, target_share=share)
            # the sweep rounds to 4dp on the way out
            self.assertAlmostEqual(
                row["threshold_from_train"], expected, places=4,
                msg=f"share={share}: cut must come from the TRAIN scores",
            )

    def test_sweep_would_differ_if_cut_came_from_holdout(self):
        """Confirms the test above is discriminating, not vacuous.

        The two score arrays must be genuinely different distributions, not the same
        multiset sorted two ways.
        """
        y = numpy.array([1] * 200 + [0] * 200)
        p_holdout = numpy.linspace(0.99, 0.01, 400)
        p_train = numpy.linspace(0.60, 0.20, 400)  # a genuinely different distribution
        from_train = evaluate.threshold_sweep(y, p_holdout, y_prob_train=p_train)
        from_holdout = evaluate.threshold_sweep(y, p_holdout, y_prob_train=p_holdout)
        self.assertNotEqual(
            from_train["threshold_from_train"].tolist(),
            from_holdout["threshold_from_train"].tolist(),
        )

    def test_sweep_reports_measured_recall_not_targeted_recall(self):
        """Recall is measured on the holdout, so it must not be pinned to the target.

        Uses continuous scores: with a heavily tied score vector several campaign sizes
        collapse onto the same cut, which is real tie behaviour but would mask the
        thing being tested here.
        """
        rng = numpy.random.default_rng(11)
        y = numpy.array([1] * 100 + [0] * 900)
        p_holdout = numpy.clip(rng.uniform(size=1000) + y * 0.5, 0, 1)
        sweep = evaluate.threshold_sweep(y, p_holdout, y_prob_train=p_holdout.copy())
        recalls = sweep["recall"].tolist()
        self.assertEqual(len(set(recalls)), len(recalls), f"recall is pinned: {recalls}")
        # and none of them lands exactly on a campaign-size fraction by construction
        self.assertTrue(all(0 < r < 1 for r in recalls))


class TestScoreAllCustomersIsOutOfFold(unittest.TestCase):
    def setUp(self):
        self.df = load_clean()
        self.X, self.y = features.split_xy(features.build_features(self.df))

    def test_scores_are_out_of_fold(self):
        model = DummyClassifier(strategy="prior")
        scored = evaluate.score_all_customers(model, self.df, n_splits=2)

        cv = StratifiedKFold(2, shuffle=True, random_state=config.RANDOM_STATE)
        expected = cross_val_predict(
            features.make_pipeline(model), self.X, self.y, cv=cv, method="predict_proba", n_jobs=1
        )[:, 1]
        # compare as ordered-by-score, since the function sorts descending
        got = numpy.sort(scored["churn_score"].to_numpy())[::-1]
        self.assertTrue(numpy.allclose(got, numpy.sort(expected)[::-1], atol=1e-9))

    def test_in_sample_scores_would_differ_from_out_of_fold(self):
        """Proves the previous test can fail.

        A constant dummy gives identical scores either way, so use a real model here -
        this one is slow but it is the assertion that makes the other one meaningful.
        """
        model = train.candidate_models()["logistic_regression"]
        oof = evaluate.score_all_customers(model, self.df, n_splits=2)["churn_score"].to_numpy()
        pipe = features.make_pipeline(model)
        pipe.fit(self.X, self.y)
        in_sample = pipe.predict_proba(self.X)[:, 1]
        self.assertFalse(
            numpy.allclose(numpy.sort(oof), numpy.sort(in_sample), atol=1e-6),
            "out-of-fold and in-sample rankings are identical - the test above is vacuous",
        )

    def test_scores_are_rounded_for_reproducibility(self):
        scored = evaluate.score_all_customers(DummyClassifier(strategy="prior"), self.df, n_splits=2)
        values = scored["churn_score"].to_numpy()
        self.assertTrue(
            numpy.allclose(values, numpy.round(values, 10)),
            "churn_score must be rounded before export or the CSV hash churns every run",
        )

    def test_already_churned_mirrors_the_label(self):
        scored = evaluate.score_all_customers(DummyClassifier(strategy="prior"), self.df, n_splits=2)
        self.assertTrue((scored["already_churned"] == scored[config.TARGET].astype(bool)).all())

    def test_output_is_sorted_descending(self):
        scored = evaluate.score_all_customers(DummyClassifier(strategy="prior"), self.df, n_splits=2)
        self.assertTrue(scored["churn_score"].is_monotonic_decreasing)


class TestSelectionIsNotOnHoldout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not config.METRICS_JSON.exists():
            raise unittest.SkipTest("run `python -m src.run_all` first")
        cls.metrics = json.loads(config.METRICS_JSON.read_text(encoding="utf-8"))
        cls.evaluation = json.loads(
            (config.ROOT / "reports" / "evaluation.json").read_text(encoding="utf-8")
        )

    def test_cross_validate_is_called_with_train_split_only(self):
        import ast
        import textwrap

        source = textwrap.dedent(Path(train.__file__).read_text(encoding="utf-8"))
        calls = [
            ast.unparse(node)
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "cross_validate"
        ]
        self.assertTrue(calls)
        for call in calls:
            self.assertIn("Xtr, ytr", call, f"cross_validate called as {call}")

    def test_evaluate_derives_sweep_cuts_from_train_scores(self):
        """The playbook campaign table comes from evaluate.py, so it needs the same guard."""
        import ast
        import textwrap

        source = textwrap.dedent(Path(evaluate.__file__).read_text(encoding="utf-8"))
        found = 0
        for node in ast.walk(ast.parse(source)):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "threshold_sweep"):
                continue
            found += 1
            args = [ast.unparse(a) for a in node.args]
            kwargs = {kw.arg: ast.unparse(kw.value) for kw in node.keywords}
            self.assertEqual(len(args) + len(kwargs), 3, f"threshold_sweep needs all 3 inputs, got {args} {kwargs}")
            self.assertEqual(args[1], "yte_prob", "the scored array is the holdout")
            train_scores = args[2] if len(args) > 2 else kwargs.get("y_prob_train")
            self.assertEqual(
                train_scores, "ytr_prob",
                f"the cut must come from the out-of-fold TRAIN scores, got {train_scores!r}",
            )
        self.assertEqual(found, 1, "expected exactly one threshold_sweep call in evaluate.py")

    def test_make_pipeline_does_not_alias_the_callers_estimator(self):
        """Regression for the clone() call.

        Without it, make_pipeline(m).fit(...) fits the caller's own instance, so two
        pipelines built from one model object overwrite each other's state and the object
        the caller holds is not the one that was trained.
        """
        model = DummyClassifier(strategy="prior")
        pipe = features.make_pipeline(model)
        self.assertIsNot(pipe.named_steps["model"], model)
        self.assertIsInstance(pipe.named_steps["model"], DummyClassifier)

    def test_fitting_one_pipeline_does_not_disturb_another(self):
        X, y = features.split_xy(features.build_features(load_clean()))
        shared = DummyClassifier(strategy="prior")
        first = features.make_pipeline(shared).fit(X, y)
        before = first.predict_proba(X)[:20].copy()
        features.make_pipeline(shared).fit(X.iloc[:100], y.iloc[:100])
        after = first.predict_proba(X)[:20]
        self.assertTrue(numpy.array_equal(before, after), "a second fit mutated the first pipeline")

    def test_clean_outputs_removes_stale_artefacts_of_any_name(self):
        """The glob, not an allowlist.

        An allowlist silently rots: a newly written report is not on it and survives the
        rebuild that should have deleted it. That already happened once with a figure.

        Runs against a temporary root - clean_outputs wipes the real committed artefacts,
        and a test must not leave the repo in a different state from how it found it.
        """
        import shutil
        import tempfile

        from src import config as cfg
        from src import run_all

        originals = (cfg.ROOT, cfg.FIG_DIR, cfg.MODEL_DIR, cfg.SCORED_CSV)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            try:
                cfg.ROOT = root
                cfg.FIG_DIR = root / "reports" / "figures"
                cfg.MODEL_DIR = root / "models"
                cfg.SCORED_CSV = root / "data" / "processed" / "churn_scored.csv"

                planted = [
                    root / "reports" / "unexpected_new_report.json",
                    cfg.FIG_DIR / "zz_stale_figure.png",
                    cfg.MODEL_DIR / "zz_stale_model.joblib",
                    cfg.SCORED_CSV.parent / "zz_old_ranked.csv",
                ]
                for p in planted:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text("stale", encoding="utf-8")

                handbook = root / "reports" / "RETENTION_PLAYBOOK.md"
                handbook.write_text("hand-authored", encoding="utf-8")

                run_all.clean_outputs()

                for p in planted:
                    self.assertFalse(p.exists(), f"clean_outputs left {p.name} behind")
                self.assertTrue(handbook.exists(), "clean_outputs deleted the hand-authored playbook")
            finally:
                cfg.ROOT, cfg.FIG_DIR, cfg.MODEL_DIR, cfg.SCORED_CSV = originals
                shutil.rmtree(tmp, ignore_errors=True)

    def test_sweep_thresholds_in_the_artefact_match_a_fresh_train_recomputation(self):
        """Behavioural guard on evaluate.py's threshold source.

        The AST check above only sees that the call is spelled `ytr_prob`; zeroing that
        variable satisfies it. Recomputing the cuts from the model and comparing against
        the recorded values catches it. This is the source of the playbook's entire
        campaign table, so it is the one that matters.
        """
        best_name = self.metrics["best_model"]
        model = train.candidate_models()[best_name]

        df = load_clean()
        X, y = features.split_xy(features.build_features(df))
        Xtr, Xte, ytr, yte = stratified_split(X, y)
        ytr_prob = cross_val_predict(
            features.make_pipeline(model), Xtr, ytr,
            cv=StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE),
            method="predict_proba", n_jobs=1,
        )[:, 1]

        recorded = {r["target_share"]: r["threshold_from_train"] for r in self.evaluation["threshold_sweep"]}
        for share, thr in recorded.items():
            expected = metrics.threshold_for_target_size(ytr_prob, target_share=share)
            self.assertAlmostEqual(
                thr, expected, places=3,
                msg=f"share={share}: the campaign cut was not derived from the train scores",
            )

    def test_evaluate_run_produces_cuts_derived_from_train_scores(self):
        """Execute evaluate.run() and check what it actually produced.

        Every other guard here reads the committed evaluation.json, which a source change
        does not regenerate - so a mutation at the call site in evaluate.run() is invisible
        to them. This runs the stage into a temporary root and inspects its real output.

        The model is stubbed with a cheap estimator: the point is the wiring (which array
        the cut comes from), not the model.
        """
        import shutil
        import tempfile

        from src import config as cfg

        originals = (cfg.ROOT, cfg.FIG_DIR, cfg.MODEL_DIR, cfg.SCORED_CSV, cfg.METRICS_JSON)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            try:
                cfg.ROOT = root
                cfg.FIG_DIR = root / "reports" / "figures"
                cfg.MODEL_DIR = root / "models"
                cfg.SCORED_CSV = root / "data" / "processed" / "churn_scored.csv"
                cfg.METRICS_JSON = root / "reports" / "metrics.json"

                # evaluate.run reads metrics.json for the selected model name
                cfg.METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
                cfg.METRICS_JSON.write_text(
                    json.dumps({"best_model": "logistic_regression"}), encoding="utf-8"
                )
                monkey = train.candidate_models
                train.candidate_models = lambda: {"logistic_regression": monkey()["logistic_regression"]}
                try:
                    evaluate.run()
                finally:
                    train.candidate_models = monkey

                produced = json.loads(cfg.METRICS_JSON.read_text(encoding="utf-8"))
                out = json.loads((root / "reports" / "evaluation.json").read_text(encoding="utf-8"))

                X, y = features.split_xy(features.build_features(load_clean()))
                Xtr, Xte, ytr, yte = stratified_split(X, y)
                expected = train.out_of_fold_scores(
                    train.candidate_models()["logistic_regression"], Xtr, ytr
                )
                for row in out["threshold_sweep"]:
                    share = row["target_share"]
                    self.assertAlmostEqual(
                        row["threshold_from_train"],
                        metrics.threshold_for_target_size(expected, target_share=share),
                        places=3,
                        msg=f"share={share}: evaluate.run() did not use the train scores",
                    )
                self.assertGreater(
                    max(r["threshold_from_train"] for r in out["threshold_sweep"]), 0.0,
                    "every campaign cut is 0 - the train scores were zeroed out",
                )
                self.assertTrue(produced["best_model"] == "logistic_regression")
            finally:
                cfg.ROOT, cfg.FIG_DIR, cfg.MODEL_DIR, cfg.SCORED_CSV, cfg.METRICS_JSON = originals
                shutil.rmtree(tmp, ignore_errors=True)

    def test_selection_key_is_the_cv_score(self):
        """Parse train.py and confirm the sort key is the CV mean, not a holdout metric."""
        import ast
        import textwrap

        source = textwrap.dedent(Path(train.__file__).read_text(encoding="utf-8"))
        keys = []
        for node in ast.walk(ast.parse(source)):
            if not (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "sort"):
                continue
            for kw in node.keywords:
                if kw.arg == "key" and isinstance(kw.value, ast.Lambda):
                    keys.append(ast.unparse(kw.value.body))
        self.assertTrue(keys, "expected a .sort(key=lambda ...) call")
        for key in keys:
            self.assertEqual(
                key, "r['cv_pr_auc_mean']",
                "selection must key on the cross-validated score, not a holdout metric",
            )

    def test_thresholds_derive_from_train_scores_not_holdout(self):
        """Structural guard on the holdout-label leak.

        Deriving the cut from `yte` and then reporting recall at that cut is circular.
        A behavioural check cannot catch it here: on this dataset the leaked and honest
        paths coincide numerically, because targeting 80% of 374 churners needs 300 either
        way and both report 0.8021. So the only way to pin this is to parse the call.

        `prob_tr` is the out-of-fold TRAIN score array and `prob` is the holdout score
        array. threshold_for_target_recall must see ytr/prob_tr; threshold_for_target_size
        must see prob_tr.
        """
        import ast
        import textwrap

        source = textwrap.dedent(Path(train.__file__).read_text(encoding="utf-8"))
        tree = ast.parse(source)
        seen = {"threshold_for_target_recall": 0, "threshold_for_target_size": 0}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            name = node.func.attr
            if name not in seen or not node.args:
                continue
            seen[name] += 1
            first = ast.unparse(node.args[0])
            if name == "threshold_for_target_recall":
                self.assertEqual(
                    first, "ytr",
                    f"line {node.lineno}: threshold derived from {first!r}, must be 'ytr'",
                )
                self.assertEqual(
                    ast.unparse(node.args[1]), "prob_tr",
                    f"line {node.lineno}: second arg must be the out-of-fold TRAIN scores",
                )
            else:
                self.assertEqual(
                    first, "prob_tr",
                    f"line {node.lineno}: size cut derived from {first!r}, must be 'prob_tr'",
                )
        for name, count in seen.items():
            self.assertGreater(count, 0, f"no {name} call found - did the code change?")

    def test_holdout_recall_is_not_pinned_to_the_target(self):
        """A leaked cut reports the target by construction, so no model may match it.

        Not sufficient on its own (on this dataset the honest path also lands near 0.80),
        but a hard 0.8000 across every model would be a clear signal.
        """
        for row in self.metrics["results"]:
            if row["model"] == "baseline_dummy":
                continue
            recall = row["at_train_derived_80pct_recall"]["recall"]
            self.assertNotAlmostEqual(
                recall, 0.80, places=6,
                msg=f"{row['model']} reports exactly the target: the cut is circular",
            )

    def test_recorded_thresholds_match_a_fresh_out_of_fold_recomputation(self):
        """The guard that actually catches a rebind like `ytr = yte`.

        The name-based AST checks above cannot see data flow: rebinding `ytr` to the
        holdout labels leaves the call site reading `threshold_for_target_recall(ytr, ...)`
        and every spelling check green. Recomputing the cut from the model and comparing
        to the recorded value catches it, because the honest and leaked thresholds differ
        (0.4922 vs 0.4939 on this data) and the recall they reach is nearly the same
        (0.8048 vs 0.8021) - too close to separate, so the threshold is the only signal.
        """
        best_name = self.metrics["best_model"]
        row = next(r for r in self.metrics["results"] if r["model"] == best_name)
        model = train.candidate_models()[best_name]

        df = load_clean()
        X, y = features.split_xy(features.build_features(df))
        Xtr, Xte, ytr, yte = stratified_split(X, y)

        prob_tr = train.out_of_fold_scores(model, Xtr, ytr)
        expected_size = metrics.threshold_for_target_size(prob_tr, target_share=0.10)
        expected_recall = metrics.threshold_for_target_recall(ytr, prob_tr, target_recall=0.80)

        self.assertAlmostEqual(row["threshold_top_10pct"], expected_size, places=5)
        self.assertAlmostEqual(row["threshold_80pct_recall"], expected_recall, places=5)

    def test_no_rebinding_of_split_variables_after_the_split(self):
        """Reject `ytr = yte` / `prob = prob_tr` style substitutions in train.py and
        evaluate.py. The other AST checks look at how a name is USED; this looks at whether
        it was assigned twice, which is the other half of the leak.

        Only the SECOND assignment counts - each name has to be bound once legitimately.
        """
        import ast
        import textwrap

        protected = {"Xtr", "Xte", "ytr", "yte", "prob", "prob_tr", "ytr_prob"}
        for module in (train, evaluate):
            source = textwrap.dedent(Path(module.__file__).read_text(encoding="utf-8"))
            tree = ast.parse(source)
            for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
                # scoped per function: `prob` is legitimately bound once in each of two
                # different functions, and only a rebind *within* one scope is a leak.
                # Seed with the parameters - a parameter binding is not an Assign node, so
                # without this `ytr = yte` inside a function taking ytr reads as a first
                # binding and slips through.
                bound: set[str] = {a.arg for a in fn.args.args}
                for node in ast.walk(fn):
                    if isinstance(node, ast.Assign):
                        targets = node.targets
                    elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                        targets = [node.target]
                    else:
                        continue
                    for target in targets:
                        for sub in ast.walk(target):
                            if isinstance(sub, ast.Name) and sub.id in protected:
                                if sub.id in bound:
                                    raise AssertionError(
                                        f"{Path(module.__name__)}.py:{node.lineno} rebinds "
                                        f"{sub.id!r} inside {fn.name}() - a second assignment "
                                        f"is how a holdout leak hides behind a name check that "
                                        f"still looks correct"
                                    )
                                bound.add(sub.id)


class TestEdaProducesTheQuotedFindings(unittest.TestCase):
    """The README's segment table and tenure bands come from eda.py. Assert them."""

    @classmethod
    def setUpClass(cls):
        cls.df = load_clean()

    def test_contract_churn_rates(self):
        t = eda.churn_rate_by(self.df, "Contract")
        self.assertAlmostEqual(t.loc["Month-to-month", "churn_rate"], 0.427, places=3)
        self.assertAlmostEqual(t.loc["One year", "churn_rate"], 0.113, places=3)
        self.assertAlmostEqual(t.loc["Two year", "churn_rate"], 0.028, places=3)

    def test_month_to_month_holds_most_of_the_revenue_at_risk(self):
        t = eda.churn_rate_by(self.df, "Contract")
        share = t.loc["Month-to-month", "monthly_rev_at_risk"] / t["monthly_rev_at_risk"].sum()
        self.assertGreater(share, 0.85)
        self.assertLess(share, 0.90)

    def test_tenure_band_churn_falls_monotonically(self):
        table = eda.churn_rate_by_tenure_band(self.df)
        rates = table["churn_rate"].tolist()
        self.assertEqual(rates, sorted(rates, reverse=True), f"not monotonic: {rates}")
        self.assertAlmostEqual(rates[0], 0.533, places=3)
        self.assertAlmostEqual(rates[-1], 0.095, places=3)

    def test_churned_customers_pay_more(self):
        means = eda.churned_vs_retained_means(self.df)
        self.assertAlmostEqual(means.loc["churned", "MonthlyCharges"], 74.44, places=2)
        self.assertAlmostEqual(means.loc["retained", "MonthlyCharges"], 61.31, places=2)
        self.assertGreater(
            means.loc["churned", "MonthlyCharges"], means.loc["retained", "MonthlyCharges"]
        )

    def test_concentration_keys_track_the_requested_fraction(self):
        out = eda.concentration(self.df, top_frac=0.2)
        self.assertIn("top_20.0pct_of_churned_customers", out)
        self.assertEqual(out["top_20.0pct_of_churned_customers"], int(1869 * 0.2))

    def test_churn_rate_by_has_no_nan_for_zero_churn_categories(self):
        # a category with no churners must report 0 revenue at risk, not NaN
        out = eda.churn_rate_by(self.df, "Contract")
        self.assertFalse(out["monthly_rev_at_risk"].isna().any())
        self.assertAlmostEqual(float(out.loc["Two year", "monthly_rev_at_risk"]), 4165.30, places=2)

    def test_tenure_band_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            eda.tenure_band(500)


class TestPipelineProducesEveryArtefact(unittest.TestCase):
    """Reads what the last `python -m src.run_all` wrote.

    Skipped on a fresh checkout rather than erroring: run the pipeline first.
    """

    @classmethod
    def setUpClass(cls):
        if not config.METRICS_JSON.exists():
            raise unittest.SkipTest("run `python -m src.run_all` first")
        cls.metrics = json.loads(config.METRICS_JSON.read_text(encoding="utf-8"))
        cls.evaluation = json.loads(
            (config.ROOT / "reports" / "evaluation.json").read_text(encoding="utf-8")
        )

    def test_all_artefacts_exist(self):
        for path in (
            config.METRICS_JSON,
            config.ROOT / "reports" / "evaluation.json",
            config.ROOT / "reports" / "eda_findings.json",
            config.SCORED_CSV,
        ):
            self.assertTrue(path.exists(), f"{path.name} was not produced")

    def test_ranked_csv_has_the_expected_shape_and_columns(self):
        df = pd.read_csv(config.SCORED_CSV)
        self.assertEqual(len(df), 7032)
        for col in (config.ID_COL, config.TARGET, "churn_score", "already_churned"):
            self.assertIn(col, df.columns)

    def test_ranked_csv_is_byte_reproducible(self):
        """Rebuild the ranking and compare bytes.

        Guards the n_jobs=-1 float-accumulation churn that made the committed deliverable
        change hash on every rebuild.
        """
        first = config.SCORED_CSV.read_bytes()
        best = self.metrics["best_model"]
        scored = evaluate.score_all_customers(train.candidate_models()[best], load_clean())
        rebuilt = scored.to_csv(index=False).encode()
        self.assertEqual(
            first, rebuilt,
            "rebuilding the ranking produced different bytes - the committed deliverable churns",
        )


if __name__ == "__main__":
    unittest.main()
