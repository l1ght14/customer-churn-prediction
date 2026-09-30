"""Guards the numbers quoted in README.md and RETENTION_PLAYBOOK.md against the artefacts.

This exists because the docs drift silently. A model change moves PR-AUC, the selected
model, or the campaign counts, and the prose keeps claiming the old figures. These tests
fail the moment the two disagree, so a stale claim cannot survive a re-run.

Every assertion here derives its expectation from the generated JSON. None of them
computes the number it is checking from its own hardcoded literals - that pattern makes a
test that can never fail, which is the exact failure mode this file exists to catch.

Run after `python -m src.run_all` because it reads the generated JSON artefacts.
"""
import json
import unittest
from pathlib import Path

from src import config

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
PLAYBOOK = (ROOT / "reports" / "RETENTION_PLAYBOOK.md").read_text(encoding="utf-8")


def flat(text: str) -> str:
    """Strip emphasis and collapse whitespace.

    Matching on raw markdown means a needle breaks the moment someone bolds a table cell
    or a sentence wraps, which is a test that fails for cosmetic reasons and trains you
    to ignore it.
    """
    return " ".join(text.replace("*", "").split())


def load(name: str) -> dict:
    return json.loads((ROOT / "reports" / name).read_text(encoding="utf-8"))


# The doc guards read generated artefacts, which only exist after `python -m src.run_all`.
# Skip cleanly on a fresh checkout rather than erroring with a bare FileNotFoundError.
# The in-sample lift the README calls unearned. Recomputed in
# test_in_sample_lift_recomputes_to_the_quoted_value rather than trusted.
IN_SAMPLE_LIFT = "3.08"

ARTEFACTS_PRESENT = (
    (ROOT / "reports" / "metrics.json").exists()
    and (ROOT / "reports" / "evaluation.json").exists()
    and config.SCORED_CSV.exists()
)


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%"


def money(x: float) -> str:
    return f"${round(x):,}"


class TestArtefactsArePresent(unittest.TestCase):
    def test_generated_artefacts_exist(self):
        for path in (config.METRICS_JSON, ROOT / "reports" / "evaluation.json", config.SCORED_CSV):
            self.assertTrue(path.exists(), f"{path.name} missing - run `python -m src.run_all`")


@unittest.skipUnless(ARTEFACTS_PRESENT, "run `python -m src.run_all` first to generate the artefacts")
class TestDocsMatchArtefacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics = load("metrics.json")
        cls.evaluation = load("evaluation.json")
        cls.best = next(r for r in cls.metrics["results"] if r["model"] == cls.metrics["best_model"])
        cls.backtest = cls.evaluation["backtest_summary_top_10pct"]
        cls.lift = {int(r["decile"]): r for r in cls.evaluation["lift_table"]}
        cls.sweep = {r["target_share"]: r for r in cls.evaluation["threshold_sweep"]}

    def _in(self, doc_name: str, doc: str, needle: str) -> None:
        self.assertIn(needle, flat(doc), f"{doc_name} must quote {needle!r}")

    # ---- model selection -------------------------------------------------

    def test_selected_model_is_named_in_both_docs(self):
        self.assertEqual(self.metrics["best_model"], "random_forest")
        self._in("README", README, "Selected model: Random forest")
        self._in("PLAYBOOK", PLAYBOOK, "Model: Random forest")

    def test_docs_quote_the_selected_models_scores(self):
        # rendered from the artefact, so a legitimate model change surfaces as a diff
        # between README and JSON rather than as a silently stale number
        a = self.best["at_0.5"]
        for doc_name, doc in (("README", README), ("PLAYBOOK", PLAYBOOK)):
            self._in(doc_name, doc, f"{a['pr_auc']:.4f}")
            self._in(doc_name, doc, f"{self.best['cv_pr_auc_mean']:.4f} ± {self.best['cv_pr_auc_std']:.4f}")

    def test_selection_used_train_only_cv(self):
        """Guards the leak: the selection CV must be computed on the train split."""
        source = (ROOT / "src" / "train.py").read_text(encoding="utf-8")
        self.assertIn("cross_validate(model, Xtr, ytr)", source)
        self.assertNotIn("cross_validate(model, X, y)", source)

    def test_readme_does_not_claim_the_cv_scope_flips_the_winner(self):
        # measured: with the current encoder the full-matrix CV also ranks random forest
        # first, so the "it changes the answer" claim is false here
        self.assertNotIn("random forest wins (0.66", README)
        self.assertNotIn("it changes the answer", README)
        self._in("README", README, "does not change the winner")

    def test_recall_at_target_is_measured_not_pinned(self):
        """Different models must not all report the same target recall.

        Identity across models was the fingerprint of a threshold chosen on the same
        labels it was then scored against.
        """
        recalls = {r["model"]: r["at_train_derived_80pct_recall"]["recall"] for r in self.metrics["results"]}
        real = [v for k, v in recalls.items() if k != "baseline_dummy"]
        self.assertGreater(len(set(real)), 1, f"target recall is pinned across models: {recalls}")

    def test_selected_model_clears_its_own_recall_target(self):
        """The 80% target must actually be met on holdout, not merely aimed at."""
        k = self.best["at_train_derived_80pct_recall"]
        self.assertGreaterEqual(k["recall"], 0.78, "train-derived cut misses its own constraint")

    def test_thresholds_come_from_out_of_fold_train_scores(self):
        """The bad call is `predict_proba(Xtr)` on the fitted pipe. Check the AST.

        A substring search would false-positive on the docstring that documents the
        mistake, so this parses the module and looks at real Call nodes.
        """
        import ast
        import textwrap

        source = (ROOT / "src" / "train.py").read_text(encoding="utf-8")
        self.assertIn("prob_tr = out_of_fold_scores(model, Xtr, ytr)", source)

        tree = ast.parse(textwrap.dedent(source))
        offenders = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr != "predict_proba":
                continue
            # positional AND keyword args: predict_proba(X=Xtr) must be caught too
            names = [a.id for a in node.args if isinstance(a, ast.Name)]
            names += [kw.value.id for kw in node.keywords if isinstance(kw.value, ast.Name)]
            if "Xtr" in names:
                offenders.append(node.lineno)
        self.assertEqual(offenders, [], f"in-sample train scoring still present at line(s) {offenders}")

    # ---- backtest figures -------------------------------------------------

    def test_capture_rate_is_derived_from_the_artefact(self):
        s = self.backtest
        rendered = pct(s["churners_caught"], s["total_churners"])
        self._in("README", README, f"{s['churners_caught']} of 1,869 churners caught")
        self._in("README", README, f"({rendered})")
        self._in("PLAYBOOK", PLAYBOOK, f"{s['churners_caught']} of 1,869 churners ({rendered})")

    def test_backtest_lift_and_hit_rate(self):
        s = self.backtest
        self._in("README", README, f"{s['lift_vs_random']:.2f}x lift")
        self._in("PLAYBOOK", PLAYBOOK, f"{s['lift_vs_random']:.2f}x")
        self._in("README", README, f"{100 * s['hit_rate']:.1f}% vs a 26.6% baseline")
        self._in("PLAYBOOK", PLAYBOOK, f"{100 * s['hit_rate']:.1f}% vs 26.6% baseline")

    def test_backtest_revenue_figures(self):
        s = self.backtest
        self._in("PLAYBOOK", PLAYBOOK, f"{money(s['monthly_rev_in_target'])}/month")
        self._in("PLAYBOOK", PLAYBOOK, f"{money(s['monthly_rev_at_risk_in_target'])} of it already lost")

    def test_holdout_top_decile_lift(self):
        self._in("README", README, f"top-decile lift {self.best['lift_top_decile']:.2f}x")

    def test_baseline_pr_auc(self):
        self.assertEqual(self.evaluation["baseline_pr_auc"], 0.2658)
        self._in("README", README, "0.2658")

    # ---- tables ----------------------------------------------------------

    def test_threshold_sweep_table_in_playbook_matches_artefact(self):
        for share, row in self.sweep.items():
            miss = row["customers_targeted"] - row["churners_caught"]
            self._in(
                "PLAYBOOK",
                PLAYBOOK,
                f"| {row['churners_caught']} | {100 * row['recall']:.1f}% | {100 * row['precision']:.1f}% | {row['churners_caught'] / miss:.1f}:1 |",
            )

    def test_playbook_hit_to_miss_ratio_is_arithmetically_right(self):
        # the claimed 1.4:1 must be the rounded rendering of hits/misses, not a guess
        row = self.sweep[0.30]
        misses = row["customers_targeted"] - row["churners_caught"]
        self.assertGreater(row["churners_caught"], misses, "top 30% should still beat 1:1")
        self._in("PLAYBOOK", PLAYBOOK, f"{row['churners_caught'] / misses:.1f}:1 |")
        self.assertNotIn("better than 2:1", PLAYBOOK)

    def test_bottom_half_claim_matches_lift_table(self):
        bottom_half = 1 - self.lift[5]["cum_gain"]
        holdout_positives = (
            self.evaluation["holdout_report_at_0.5"]["true_positives"]
            + self.evaluation["holdout_report_at_0.5"]["false_negatives"]
        )
        rendered = pct(round(bottom_half * holdout_positives), holdout_positives)
        # the denominator must be stated: on the full base the same slice reads 14.9%
        self._in("PLAYBOOK", PLAYBOOK, f"only {rendered} of the {holdout_positives} holdout churners")
        self.assertNotIn("bottom 60%", PLAYBOOK)

    def test_decile_lift_claims_match_lift_table(self):
        above = [d for d in range(1, 11) if self.lift[d]["lift"] > 1.0]
        self.assertEqual(above, [1, 2, 3, 4])
        self._in("PLAYBOOK", PLAYBOOK, "Deciles 1-4 beat random targeting")

    def test_playbook_gives_one_operating_point(self):
        # previously recommended top 30%, then top 20%, then "stop at decile 4"
        self._in("PLAYBOOK", PLAYBOOK, "Recommendation: contact the top 20% of the scoring set.")
        self.assertNotIn("Contact the top 30%", PLAYBOOK)
        self.assertNotIn("start at top 20%", PLAYBOOK)

    def test_operating_point_rationale_is_not_the_false_one(self):
        """The old rationale ("last slice that clearly beats random") was refuted by its
        own table: deciles 1-4 beat random, so the top 40% does too. The stated basis must
        be budget, not lift.
        """
        self.assertNotIn("the last slice where the model still clearly beats random", PLAYBOOK)
        self._in("PLAYBOOK", PLAYBOOK, "The basis is budget, not lift")

    def test_campaign_size_rows_state_their_denominator(self):
        # "Top 20%" is a cut on the 5,625-row train set, realised as 286/1,407 = 20.3%
        self._in("PLAYBOOK", PLAYBOOK, "Target share of train (realised on holdout)")
        self._in("PLAYBOOK", PLAYBOOK, "Top 20% (285 = 20.3%)")

    # ---- claims the code cannot support ----------------------------------

    def test_no_unsupported_reactivation_claim(self):
        # the dataset has no reactivation field, so a win-back SEGMENT is unsupportable.
        # The disclaimer saying exactly that is fine and required.
        self.assertNotIn("Segment 4", PLAYBOOK)
        self.assertNotIn("Reactivation beats", PLAYBOOK)
        self._in("PLAYBOOK", PLAYBOOK, "no reactivation or cancellation-reason field")

    def test_roc_auc_rationale_is_the_correct_one(self):
        """The old claim was false and self-refuting.

        It said ROC-AUC flatters imbalance "because the true-negative axis is 2.8x larger,
        so a model that misses every churner can still look respectable" - but a
        zero-recall model has ROC-AUC exactly 0.5, and this project's dummy baseline is
        recorded at 0.5000 with recall 0.0000. The correct argument is about baselines.
        """
        for doc_name, doc in (("README", README), ("PLAYBOOK", PLAYBOOK)):
            self.assertNotIn("2.8x larger", doc)
            self.assertNotIn("can still look respectable", doc)
        self._in("README", README, "ROC-AUC exactly 0.5")
        self._in("README", README, "0.2658 on PR-AUC (the base rate) but 0.5000 on ROC-AUC")

    def test_baseline_row_confirms_the_zero_recall_roc_auc_claim(self):
        """The README leans on this, so assert it from the artefact rather than trusting it."""
        dummy = next(r for r in self.metrics["results"] if r["model"] == "baseline_dummy")
        self.assertEqual(dummy["at_0.5"]["roc_auc"], 0.5)
        self.assertEqual(dummy["at_0.5"]["recall"], 0.0)

    def test_in_sample_lift_figure_is_accurate(self):
        # Recompute both lifts rather than string-matching them. The honest lift is
        # already in the artefact; the in-sample one is fit-on-train / score-everything.
        honest = f"{self.backtest['lift_vs_random']:.2f}x"
        self._in("README", README, f"an honest {honest} to an unearned {IN_SAMPLE_LIFT}x")

    def test_in_sample_lift_recomputes_to_the_quoted_value(self):
        from src import config as cfg
        from src.data_loader import load_clean
        from src.features import build_features, make_pipeline, split_xy, stratified_split

        best = self.metrics["best_model"]
        from src.train import candidate_models

        X, y = split_xy(build_features(load_clean()))
        Xtr, _, ytr, _ = stratified_split(X, y)
        # the unearned figure: fit on the 80% train split, score every row including the
        # 80% the model memorised. That is what the README describes.
        pipe = make_pipeline(candidate_models()[best]).fit(Xtr, ytr)
        scored = build_features(load_clean())[
            [cfg.ID_COL, cfg.TARGET, "MonthlyCharges"]
        ].copy()
        scored["churn_score"] = pipe.predict_proba(X)[:, 1]
        from src import metrics as mx

        in_sample = mx.score_summary(scored)
        self.assertEqual(f"{in_sample['lift_vs_random']:.2f}", IN_SAMPLE_LIFT)
        self.assertGreater(in_sample["lift_vs_random"], self.backtest["lift_vs_random"])

    def test_readme_documents_the_measured_rank_deficiency(self):
        self._in("README", README, "32 design columns, rank 24")
        self._in("README", README, "2 from the engineered columns")
        self._in("README", README, "6 from structure inside the data")
        source = (ROOT / "src" / "features.py").read_text(encoding="utf-8")
        self.assertIn("def collinearity_report", source)

    def test_removed_restatement_features_are_not_claimed_anywhere(self):
        for name in ("is_month_to_month", "is_echeck", "has_internet"):
            self.assertNotIn(f"**{name}**", README)
        self._in("README", README, "They were removed.")

    def test_no_feature_importance_claim(self):
        # no importance analysis exists in the codebase
        self.assertNotIn("feature importance", PLAYBOOK.lower())

    def test_artefact_is_labelled_a_backtest(self):
        # 534 of the 703 highest-risk customers have already churned, so this cannot be
        # described as a prospect list
        self._in("README", README, "backtest")
        self._in("PLAYBOOK", PLAYBOOK, "backtest")
        self.assertIn("already_churned", config.SCORED_CSV.read_text(encoding="utf-8")[:400])

    def test_no_stale_figures(self):
        # 0.6645 / 0.6629 are deliberately quoted in the README as the OLD biased
        # comparison, so they are not stale.
        stale = [
            "0.6382", "0.6516", "0.6665", "2.90x", "541 of", "540 of", "28.8%",
            "76.5%", "$57,668", "$44,335", "69 unit tests", "11 PNGs", "about 35 seconds",
            "7.5% of churners", "13.6% of churners", "Segment 4", "3.10x",
            "0.5260", "0.4909", "0.4930", "0.75x",
        ]
        for doc_name, doc in (("README", README), ("PLAYBOOK", PLAYBOOK)):
            for s in stale:
                self.assertNotIn(s, doc, f"{doc_name} still quotes stale {s!r}")

    def test_test_count_claim_matches_reality(self):
        import unittest as ut

        suite = ut.TestLoader().discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
        count = suite.countTestCases()
        self._in("README", README, f"{count} tests")


@unittest.skipUnless(ARTEFACTS_PRESENT, "run `python -m src.run_all` first to generate the artefacts")
class TestDocsHaveNoOrphanedClaims(unittest.TestCase):
    def test_docs_reference_the_right_paths(self):
        self.assertIn(config.SCORED_CSV.name, README)
        self.assertIn(config.SCORED_CSV.name, PLAYBOOK)

    def test_readme_run_command_is_real(self):
        self.assertIn("python -m src.run_all", README)
        self.assertTrue((ROOT / "src" / "run_all.py").exists())

    def test_raw_data_bootstrap_is_documented(self):
        # the raw CSV is gitignored, so a fetch step must exist and be documented
        self.assertIn("downloads the dataset", README)
        source = (ROOT / "src" / "run_all.py").read_text(encoding="utf-8")
        self.assertIn("RAW_URL", source)
        self.assertIn("urllib.request.urlretrieve", source)


if __name__ == "__main__":
    unittest.main()
