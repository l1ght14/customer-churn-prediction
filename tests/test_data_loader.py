import unittest

import pandas as pd

from src import config, data_loader


def tiny_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["A", "B", "C", "D", "E", "F"],
            "gender": ["Female ", "Male", "Female", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 0, 1, 0],
            "tenure": [0, 12, 24, 5, 0, 60],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "Month-to-month", "Two year"],
            "MonthlyCharges": [29.85, 70.0, 80.0, 65.0, 20.0, 95.0],
            "TotalCharges": [" ", "840.0", "1920.0", "325.0", " ", "5700.0"],
            "Churn": ["No", "Yes", "No", "Yes", "Yes", "No"],
        }
    )


class TestLoadRaw(unittest.TestCase):
    def test_raw_file_has_expected_shape(self):
        raw = data_loader.load_raw()
        self.assertEqual(raw.shape, (7043, 21))

    def test_raw_is_not_cleaned(self):
        # TotalCharges stays a string until clean() forces the cast
        self.assertNotIn("float64", str(data_loader.load_raw()["TotalCharges"].dtype))

    def test_raw_blanks_are_whitespace_not_null(self):
        # pandas does NOT read " " as NaN, so a naive .isna() check finds nothing.
        # This is the trap: the nulls are only visible after stripping.
        raw = data_loader.load_raw()
        self.assertEqual(int(raw["TotalCharges"].isna().sum()), 0)
        self.assertEqual(int((raw["TotalCharges"].str.strip() == "").sum()), config.BLANK_CHARGES_DROP_COUNT)


class TestClean(unittest.TestCase):
    def setUp(self):
        self.cleaned = data_loader.clean(tiny_raw())

    def test_does_not_mutate_input(self):
        raw = tiny_raw()
        before = raw.copy()
        data_loader.clean(raw)
        pd.testing.assert_frame_equal(raw, before)

    def test_drops_blank_totalcharges_rows(self):
        self.assertEqual(len(self.cleaned), 4)

    def test_totalcharges_becomes_numeric(self):
        self.assertTrue(pd.api.types.is_numeric_dtype(self.cleaned["TotalCharges"]))

    def test_text_whitespace_is_stripped(self):
        self.assertNotIn("Female ", set(self.cleaned["gender"]))

    def test_seniorcitizen_becomes_readable_categorical(self):
        self.assertEqual(set(self.cleaned["SeniorCitizen"]), {"Yes", "No"})

    def test_target_becomes_binary_int(self):
        self.assertTrue(pd.api.types.is_integer_dtype(self.cleaned[config.TARGET]))
        self.assertEqual(set(self.cleaned[config.TARGET]), {0, 1})

    def test_index_is_reset_after_drop(self):
        self.assertEqual(list(self.cleaned.index), list(range(len(self.cleaned))))

    def test_no_nulls_remain(self):
        self.assertEqual(int(self.cleaned.isna().sum().sum()), 0)

    def test_dropped_rows_are_all_tenure_zero(self):
        raw = tiny_raw()
        dropped = raw[raw["TotalCharges"] == " "]
        self.assertTrue((dropped["tenure"] == 0).all())


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.cleaned = data_loader.clean(tiny_raw())

    def test_churn_rate(self):
        self.assertAlmostEqual(data_loader.churn_rate(self.cleaned), 2 / 4)

    def test_revenue_at_risk_only_counts_churned(self):
        expected = 70.0 + 65.0
        self.assertAlmostEqual(data_loader.revenue_at_risk(self.cleaned), expected)

    def test_revenue_at_risk_zero_when_nobody_churns(self):
        no_churn = self.cleaned.assign(**{config.TARGET: 0})
        self.assertEqual(data_loader.revenue_at_risk(no_churn), 0.0)


class TestQualityReport(unittest.TestCase):
    def test_report_counts_match_real_dataset(self):
        raw = data_loader.load_raw()
        cld = data_loader.clean(raw)
        report = data_loader.data_quality_report(raw, cld)

        self.assertEqual(report["blank_totalcharges_rows"], 11)
        self.assertTrue(report["dropped_rows_all_non_churned"])
        self.assertEqual(report["raw_rows"], 7043)
        self.assertEqual(report["clean_rows"], 7032)
        self.assertEqual(report["rows_dropped"], 11)
        self.assertEqual(report["duplicate_customer_ids"], 0)
        self.assertEqual(report["null_cells"], 0)
        self.assertTrue(report["dropped_rows_all_tenure_zero"])
        self.assertEqual(report["churned_customers"], 1869)
        self.assertAlmostEqual(report["churn_rate"], 1869 / 7032, places=6)

    def test_assert_no_leakage_passes_on_unique_ids(self):
        data_loader.assert_no_leakage(data_loader.clean(tiny_raw()))

    def test_assert_no_leakage_catches_duplicates(self):
        # The inner assertTrue must sit OUTSIDE the assertRaises block. Inside it, the
        # AssertionError it raises is swallowed by assertRaises and the test goes green
        # on a broken implementation - a false green.
        dup = data_loader.clean(tiny_raw())
        dup.loc[1, config.ID_COL] = dup.loc[0, config.ID_COL]
        self.assertTrue(dup[config.ID_COL].duplicated().any())
        with self.assertRaises(AssertionError):
            data_loader.assert_no_leakage(dup)

    def test_dropped_blank_charge_rows_all_had_zero_tenure(self):
        raw = data_loader.load_raw()
        dropped = raw[raw["TotalCharges"].str.strip() == ""]
        self.assertTrue((dropped["tenure"] == 0).all())

    def test_missing_columns_empty_for_real_data(self):
        self.assertEqual(data_loader.missing_columns(data_loader.load_clean()), [])


if __name__ == "__main__":
    unittest.main()
