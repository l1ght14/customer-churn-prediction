import unittest

import numpy
import pandas as pd

from src import config, metrics


class TestClassificationReport(unittest.TestCase):
    def test_confusion_matrix_counts(self):
        # y_true = [1,1,1,0,0], predictions flag the first two only
        report = metrics.classification_report_at([1, 1, 1, 0, 0], [0.9, 0.8, 0.1, 0.4, 0.2], threshold=0.5)
        self.assertEqual(report["true_positives"], 2)
        self.assertEqual(report["false_positives"], 0)
        self.assertEqual(report["false_negatives"], 1)
        self.assertEqual(report["true_negatives"], 2)

    def test_precision_recall_values(self):
        report = metrics.classification_report_at([1, 1, 1, 0, 0], [0.9, 0.8, 0.1, 0.4, 0.2], threshold=0.5)
        self.assertAlmostEqual(report["precision"], 1.0)
        self.assertAlmostEqual(report["recall"], 2 / 3, places=3)

    def test_f2_heavier_than_f1_when_recall_matters(self):
        # 1 of 2 churners caught, but 3 false alarms: precision 0.25, recall 0.50
        y = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        prob = [0.9, 0.2, 0.8, 0.7, 0.6, 0.2, 0.2, 0.2, 0.2, 0.2]
        report = metrics.classification_report_at(y, prob, threshold=0.5)
        self.assertAlmostEqual(report["precision"], 0.25, places=3)
        self.assertAlmostEqual(report["recall"], 0.50, places=3)
        self.assertLess(report["precision"], report["recall"])
        self.assertGreater(report["f2"], report["f1"])

    def test_perfect_predictor_scores_one(self):
        report = metrics.classification_report_at([1, 0, 1, 0], [0.99, 0.01, 0.99, 0.01], threshold=0.5)
        self.assertAlmostEqual(report["pr_auc"], 1.0)
        self.assertAlmostEqual(report["roc_auc"], 1.0)

    def test_accuracy_alone_is_misleading_on_imbalanced_data(self):
        # 100 customers, 10 churners, model flags none -> 90% accuracy, zero recall
        y = [1] * 10 + [0] * 90
        prob = [0.1] * 100
        report = metrics.classification_report_at(y, prob, threshold=0.5)
        self.assertAlmostEqual(report["accuracy"], 0.9)
        self.assertEqual(report["recall"], 0.0)

    def test_no_false_positives_case_does_not_divide_by_zero(self):
        report = metrics.classification_report_at([0, 0, 0], [0.9, 0.9, 0.9], threshold=0.5)
        self.assertEqual(report["precision"], 0.0)
        self.assertEqual(report["f1"], 0.0)

    def test_no_true_positives_case_does_not_divide_by_zero(self):
        report = metrics.classification_report_at([1, 1, 1], [0.1, 0.1, 0.1], threshold=0.5)
        self.assertEqual(report["recall"], 0.0)
        self.assertEqual(report["f2"], 0.0)


class TestLiftAndGain(unittest.TestCase):
    def setUp(self):
        # 100 customers, the 10 churners are exactly the 10 highest-risk
        self.y = [1] * 10 + [0] * 90
        self.p = numpy.concatenate([numpy.linspace(0.99, 0.90, 10), numpy.full(90, 0.05)])

    def test_ten_deciles(self):
        table = metrics.lift_and_gain(self.y, self.p)
        self.assertEqual(len(table), 10)
        self.assertEqual(table["n_customers"].sum(), 100)
        self.assertEqual(table["n_customers"].tolist(), [10] * 10)

    def test_top_decile_captures_all_churners(self):
        table = metrics.lift_and_gain(self.y, self.p)
        self.assertEqual(table.loc[0, "churners"], 10)
        self.assertAlmostEqual(table.loc[0, "cum_gain"], 1.0)

    def test_top_decile_lift_equals_capture_rate(self):
        table = metrics.lift_and_gain(self.y, self.p)
        # 100% of churners in 10% of the base -> lift 10
        self.assertAlmostEqual(table.loc[0, "lift"], 10.0)

    def test_cum_gain_is_monotonic(self):
        table = metrics.lift_and_gain(self.y, self.p)
        self.assertTrue(table["cum_gain"].is_monotonic_increasing)

    def test_cum_gain_boundary_is_inclusive_of_the_current_decile(self):
        """Guards an off-by-one that silently corrupts the cum_gain column.

        Scores descend, so rank order equals index order. Positives sit at indices
        88/89/90, which straddles the decile-9 boundary at hi=90: an inclusive-vs-
        exclusive slice reports 2 churners or 3. Ordinary fixtures happen to put a
        non-churner exactly on the boundary.
        """
        y = [0] * 88 + [1, 1, 1] + [0] * 9
        p = numpy.linspace(0.99, 0.01, 100)
        table = metrics.lift_and_gain(y, p, n_bins=10)
        self.assertEqual(table.loc[8, "churners"], 2, "decile 9 spans ranks 80-89")
        self.assertAlmostEqual(table.loc[8, "cum_gain"], 2 / 3, places=3)
        self.assertEqual(table.loc[9, "churners"], 1, "decile 10 spans ranks 90-99")
        self.assertAlmostEqual(table.loc[9, "cum_gain"], 1.0)

    def test_base_rate_denominator_is_the_overall_rate_not_a_mean_of_bins(self):
        """Needs UNEQUAL bins.

        With equal-size bins the mean of the decile rates is algebraically identical to
        the overall rate, so an equal-bin fixture can never tell the two denominators
        apart - which is why a 100-row/10-bin test proves nothing. 103 rows over 10 bins
        gives bin sizes 10,10,10,10,10,11,11,10,10,11 and the two values diverge.
        """
        y = [1] * 4 + [0] * 6 + [0] * 93  # 4 positives, overall rate 4/103
        p = numpy.linspace(0.99, 0.01, 103)
        table = metrics.lift_and_gain(y, p, n_bins=10)
        overall = sum(y) / len(y)
        mean_of_bins = table["churn_rate_in_decile"].mean()
        self.assertNotAlmostEqual(overall, mean_of_bins, places=3)
        self.assertAlmostEqual(
            table.loc[0, "lift"], table.loc[0, "churn_rate_in_decile"] / overall, places=6
        )

    def test_raises_when_fewer_rows_than_bins(self):
        # without the guard an empty bin yields NaN churn_rate and NaN lift, and a bare
        # NaN token is not valid strict JSON
        with self.assertRaises(ValueError):
            metrics.lift_and_gain([1, 0, 1, 0, 0], numpy.linspace(0.9, 0.1, 5), n_bins=10)

    def test_final_decile_cum_gain_is_one(self):
        table = metrics.lift_and_gain(self.y, self.p)
        self.assertAlmostEqual(table.loc[9, "cum_gain"], 1.0)

    def test_lift_declines_as_you_move_down_the_ranking(self):
        # GRADED density: 30 churners spread as 20 / 10 / 0 / 0 / 0 across five buckets.
        # A ranking where every churner sits in decile 1 gives [10,0,0,0,0], which is
        # sorted descending and would let a constant lift column pass.
        y = [1] * 30 + [0] * 70
        p = numpy.linspace(0.99, 0.01, 100)
        table = metrics.lift_and_gain(y, p, n_bins=5)
        lifts = table["lift"].tolist()
        self.assertEqual(lifts, sorted(lifts, reverse=True))
        self.assertGreater(len(set(round(v, 6) for v in lifts)), 2)

    def test_random_scoring_gives_lift_around_one(self):
        # NB: the customer-share-weighted average of per-decile lift is exactly 1 for
        # ANY ranking (it is just total churners / total customers divided back out),
        # so it cannot measure discrimination. Only the shape across deciles can.
        numpy.random.seed(0)
        y = numpy.array([1] * 20 + [0] * 80)
        p = numpy.random.RandomState(0).uniform(0, 1, 100)
        table = metrics.lift_and_gain(y, p)
        self.assertLess(float(table["lift"].head(3).mean()), 1.8)

    def test_random_scoring_top_decile_not_strongly_above_one(self):
        # with pure noise the top decile should not reach 2.5x. Seed 0 gives 1.5.
        numpy.random.seed(0)
        y = numpy.array([1] * 20 + [0] * 80)
        p = numpy.random.RandomState(0).uniform(0, 1, 100)
        table = metrics.lift_and_gain(y, p)
        self.assertLess(table.loc[0, "lift"], 2.5)

    def test_a_real_signal_lifts_the_top_deciles(self):
        y = numpy.array([1] * 10 + [0] * 90)
        p = numpy.concatenate([numpy.linspace(0.99, 0.90, 10), numpy.full(90, 0.05)])
        table = metrics.lift_and_gain(y, p)
        self.assertGreater(float(table["lift"].head(3).mean()), 1.8)


class TestCumulativeGain(unittest.TestCase):
    def test_starts_at_zero_and_ends_at_one(self):
        y = [1, 0, 1, 0]
        p = [0.9, 0.8, 0.7, 0.1]
        gains, lift = metrics.cumulative_gain(y, p)
        self.assertAlmostEqual(gains[0], 0.0)
        self.assertAlmostEqual(gains[-1], 1.0)

    def test_lengths_match(self):
        y = [1, 0, 1, 0, 1]
        p = [0.9, 0.8, 0.7, 0.6, 0.5]
        gains, lift = metrics.cumulative_gain(y, p)
        self.assertEqual(len(gains), len(lift))
        self.assertEqual(len(gains), len(y) + 1)


class TestThresholds(unittest.TestCase):
    def setUp(self):
        self.y = [1] * 10 + [0] * 90
        # churners sit at the top of the risk ranking
        self.p = numpy.concatenate([numpy.linspace(0.95, 0.60, 10), numpy.full(90, 0.05)])

    def test_target_recall_threshold_catches_that_share(self):
        thr = metrics.threshold_for_target_recall(self.y, self.p, target_recall=0.8)
        caught = sum(1 for pi, yi in zip(self.p, self.y) if pi >= thr and yi == 1)
        self.assertGreaterEqual(caught, 8)

    def test_target_recall_threshold_is_not_so_low_it_flags_everyone(self):
        thr = metrics.threshold_for_target_recall(self.y, self.p, target_recall=0.8)
        flagged = sum(1 for pi in self.p if pi >= thr)
        self.assertLess(flagged, 100)

    def test_target_size_threshold_flags_about_ten_percent(self):
        # Two-sided. assertGreaterEqual alone passes if the function returns 0.0 and
        # flags all 100 customers.
        thr = metrics.threshold_for_target_size(self.p, target_share=0.10)
        flagged = sum(1 for pi in self.p if pi >= thr)
        self.assertEqual(flagged, 10)

    def test_target_size_threshold_scales_with_requested_share(self):
        # needs DISTINCT scores. A fixture with 90 tied values cannot isolate 20 rows at
        # any threshold, so the exact-count assertion would be testing the fixture.
        p = numpy.linspace(0.99, 0.01, 100)
        for share in (0.05, 0.10, 0.20, 0.30):
            thr = metrics.threshold_for_target_size(p, target_share=share)
            flagged = sum(1 for pi in p if pi >= thr)
            self.assertEqual(flagged, int(round(len(p) * share)), f"share={share}")

    def test_target_size_threshold_is_mono_tonic_in_share(self):
        p = numpy.linspace(0.99, 0.01, 100)
        flags = [
            sum(1 for pi in p if pi >= metrics.threshold_for_target_size(p, target_share=s))
            for s in (0.05, 0.10, 0.20, 0.30)
        ]
        self.assertEqual(flags, sorted(flags))

    def test_full_target_size_returns_finite_threshold(self):
        thr = metrics.threshold_for_target_size(self.p, target_share=1.0)
        self.assertGreater(thr, 0.0)

    def test_target_size_takes_no_labels(self):
        # signature is label-free on purpose: a contact budget is a business decision,
        # knowable before any outcome. Guard against y_true creeping back in.
        import inspect

        params = list(inspect.signature(metrics.threshold_for_target_size).parameters)
        self.assertEqual(params, ["y_prob", "target_share"])

    def test_target_size_rejects_out_of_range_share(self):
        with self.assertRaises(ValueError):
            metrics.threshold_for_target_size(self.p, target_share=0.0)
        with self.assertRaises(ValueError):
            metrics.threshold_for_target_size(self.p, target_share=1.5)

    def test_target_recall_rejects_unreachable_target(self):
        # a target above 1.0 must raise, not silently return a threshold that flags everyone
        with self.assertRaises(ValueError):
            metrics.threshold_for_target_recall(self.y, self.p, target_recall=1.5)

    def test_target_recall_threshold_is_the_tightest_qualifying_cut(self):
        """`>=` not `>`: a looser cut catches one extra churner and one extra contact."""
        y = [1] * 10 + [0] * 90
        p = numpy.concatenate([numpy.linspace(0.99, 0.90, 10), numpy.full(90, 0.05)])
        thr = metrics.threshold_for_target_recall(y, p, target_recall=0.8)
        caught = sum(1 for pi, yi in zip(p, y) if pi >= thr and yi == 1)
        flagged = sum(1 for pi in p if pi >= thr)
        self.assertEqual(caught, 8, "must stop at exactly the target, not one past it")
        self.assertEqual(flagged, 8)
        # the cut is the 8th score, so a `>` instead of `>=` in the argmax would land one
        # row lower and report 9 -- which the two exact assertions above already catch
        self.assertEqual(sorted(p, reverse=True)[7], thr)

    def test_classification_threshold_is_inclusive(self):
        # `>=` not `>`: a customer sitting exactly on the cut must be flagged
        report = metrics.classification_report_at([1, 0], [0.5, 0.5], threshold=0.5)
        self.assertEqual(report["true_positives"], 1)
        self.assertEqual(report["true_negatives"], 0)

    def test_target_recall_cannot_silently_return_zero_threshold(self):
        # Regression: the old guard returned 0.0 (flag EVERYONE) for an infeasible target.
        y = [1] * 10 + [0] * 90
        p = numpy.concatenate([numpy.linspace(0.99, 0.90, 10), numpy.full(90, 0.05)])
        thr = metrics.threshold_for_target_recall(y, p, target_recall=0.8)
        self.assertGreater(thr, 0.0)
        flagged = sum(1 for pi in p if pi >= thr)
        self.assertLess(flagged, 100)


class TestScoreSummary(unittest.TestCase):
    def setUp(self):
        # 100 customers, 10 churners sitting at the very top of the risk ranking
        self.df = pd.DataFrame(
            {
                config.TARGET: [1] * 10 + [0] * 90,
                "MonthlyCharges": [100.0] * 100,
                "churn_score": numpy.linspace(0.99, 0.01, 100),
            }
        )

    def test_targeting_top_decile_catches_all_churners(self):
        s = metrics.score_summary(self.df)
        self.assertEqual(s["customers_targeted"], 10)
        self.assertEqual(s["churners_caught"], 10)
        self.assertAlmostEqual(s["capture_rate"], 1.0)
        self.assertAlmostEqual(s["lift_vs_random"], 10.0, places=3)

    def test_hit_rate_beats_baseline(self):
        s = metrics.score_summary(self.df)
        self.assertGreater(s["hit_rate"], s["baseline_hit_rate"])

    def test_rev_in_target_is_summed(self):
        s = metrics.score_summary(self.df)
        self.assertAlmostEqual(s["monthly_rev_in_target"], 1000.0)
        self.assertAlmostEqual(s["monthly_rev_at_risk_in_target"], 1000.0)

    def test_counts_cover_the_whole_base(self):
        s = metrics.score_summary(self.df)
        self.assertEqual(s["customers_scored"], 100)
        self.assertEqual(s["total_churners"], 10)
        self.assertEqual(s["churners_caught"] + s["churners_caught_not_targeted"], s["total_churners"])

    def test_scores_are_ranked_by_risk_not_by_label(self):
        """Guards the core contract: ranking is by churn_score.

        Every other fixture here puts the churners at the top of the score ranking, so a
        version that ranked by the LABEL instead would be indistinguishable. Here the
        churners sit at the BOTTOM, so it cannot.
        """
        df = pd.DataFrame(
            {
                config.TARGET: [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                "MonthlyCharges": [10.0] * 10,
                "churn_score": numpy.linspace(0.99, 0.01, 10),
            }
        )
        s = metrics.score_summary(df)
        self.assertEqual(s["churners_caught"], 0, "must rank by score, not by the label")
        self.assertEqual(s["lift_vs_random"], 0.0)

    def test_lift_vs_random_is_consistent_with_its_inputs(self):
        s = metrics.score_summary(self.df)
        self.assertAlmostEqual(
            s["capture_rate"], s["churners_caught"] / s["total_churners"], places=3
        )
        self.assertAlmostEqual(
            s["lift_vs_random"],
            s["capture_rate"] / s["targeted_share_of_base"],
            places=3,
        )

    def test_rejects_out_of_range_target_share(self):
        with self.assertRaises(ValueError):
            metrics.score_summary(self.df, target_share=0.0)
        with self.assertRaises(ValueError):
            metrics.score_summary(self.df, target_share=1.5)

    def test_tiny_budget_still_targets_at_least_one_customer(self):
        # 1% of 10 rows rounds to 0; top[-1] would be the MINIMUM score and would flag
        # everyone, which is the opposite of the intent and used to happen silently
        p = numpy.linspace(0.9, 0.1, 10)
        thr = metrics.threshold_for_target_size(p, target_share=0.01)
        self.assertEqual(thr, 0.9, "must cut at the top score, not the bottom")
        self.assertEqual(sum(1 for x in p if x >= thr), 1)

    def test_rejects_missing_score_column(self):
        with self.assertRaises(KeyError):
            metrics.score_summary(self.df, score_col="nope")


if __name__ == "__main__":
    unittest.main()
