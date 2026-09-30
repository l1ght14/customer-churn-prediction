import math
import unittest
from pathlib import Path

import numpy
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

import src.features as features_module
from src import config, data_loader
from src.features import (
    ADDON_COLUMNS,
    CONTRACT_MONTHS,
    DERIVED_COLUMNS,
    SERVICE_COLUMNS,
    build_features,
    collinearity_report,
    contract_months,
    count_services,
    make_pipeline,
    split_xy,
    stratified_split,
)


def sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": list("ABCDEF"),
            "gender": ["Female", "Male", "Female", "Male", "Female", "Male"],
            "SeniorCitizen": ["No", "Yes", "No", "No", "Yes", "No"],
            "Partner": ["No", "Yes", "No", "No", "Yes", "No"],
            "Dependents": ["No", "Yes", "No", "No", "Yes", "No"],
            "tenure": [1, 60, 24, 3, 48, 12],
            "PhoneService": ["Yes", "Yes", "No", "Yes", "Yes", "Yes"],
            "MultipleLines": ["No", "No", "No phone service", "Yes", "Yes", "No"],
            "InternetService": ["Fiber optic", "DSL", "No", "Fiber optic", "DSL", "Fiber optic"],
            "OnlineSecurity": ["No", "Yes", "No", "No", "Yes", "No"],
            "OnlineBackup": ["No", "Yes", "No", "Yes", "Yes", "No"],
            "DeviceProtection": ["No", "Yes", "No", "No", "Yes", "No"],
            "TechSupport": ["No", "Yes", "No", "Yes", "Yes", "No"],
            "StreamingTV": ["Yes", "Yes", "No", "No", "Yes", "Yes"],
            "StreamingMovies": ["Yes", "No", "No", "No", "No", "Yes"],
            "Contract": ["Month-to-month", "Two year", "One year", "Month-to-month", "Two year", "One year"],
            "PaperlessBilling": ["Yes", "No", "No", "Yes", "No", "Yes"],
            "PaymentMethod": ["Electronic check", "Credit card (automatic)", "Mailed check", "Electronic check", "Bank transfer (automatic)", "Credit card (automatic)"],
            "MonthlyCharges": [70.0, 25.0, 65.0, 85.0, 30.0, 55.0],
            "TotalCharges": [70.0, 1500.0, 1560.0, 255.0, 1440.0, 660.0],
            config.TARGET: [1, 0, 0, 1, 0, 0],
        }
    )


class TestDerivedFeatures(unittest.TestCase):
    def test_contract_months_mapping(self):
        self.assertEqual(
            contract_months(pd.Series(["Month-to-month", "One year", "Two year"])).tolist(), [1, 12, 24]
        )

    def test_contract_months_rejects_unknown(self):
        with self.assertRaises(ValueError):
            contract_months(pd.Series(["Weekly"]))

    def test_contract_months_has_entry_for_every_real_contract(self):
        observed = set(data_loader.load_clean()["Contract"].unique())
        self.assertTrue(observed <= set(CONTRACT_MONTHS))

    def test_count_services_counts_only_yes(self):
        row = pd.Series(
            {
                "PhoneService": "Yes",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "Yes",
            }
        )
        self.assertEqual(count_services(row), 4)

    def test_build_features_adds_expected_columns(self):
        out = build_features(sample())
        for col in DERIVED_COLUMNS:
            self.assertIn(col, out.columns)

    def test_removed_restatement_features_are_gone(self):
        # is_month_to_month / is_echeck / has_internet were exact one-hot restatements.
        # Re-adding any of them re-creates the rank deficiency for nothing.
        out = build_features(sample())
        for gone in ("is_month_to_month", "is_echeck", "has_internet"):
            self.assertNotIn(gone, out.columns)

    def test_build_features_does_not_mutate_input(self):
        df = sample()
        before = df.copy()
        build_features(df)
        pd.testing.assert_frame_equal(df, before)

    def test_count_services_ignores_the_no_internet_level(self):
        """Regression: `!= "No"` would score all 1,520 no-internet customers at 7, since PhoneService is 'Yes' for them.

        'No internet service' is a real third level, so the count must be == 'Yes'.
        """
        row = pd.Series({c: "No internet service" for c in SERVICE_COLUMNS})
        self.assertEqual(count_services(row), 0)

    def test_no_internet_customers_hold_no_addons(self):
        # PhoneService is NOT an internet add-on, so an offline customer with a phone
        # line legitimately scores 1. The six add-ons must all be inactive.
        df = build_features(data_loader.load_clean())
        offline = df[df["InternetService"] == "No"]
        self.assertGreater(len(offline), 1000)
        for col in ADDON_COLUMNS:
            self.assertEqual(set(offline[col].unique()), {"No internet service"}, col)
        self.assertLessEqual(int(offline["services_count"].max()), 1)

    def test_services_count_distribution_is_pinned(self):
        df = build_features(data_loader.load_clean())
        s = df["services_count"]
        self.assertEqual(int(s.min()), 0)
        self.assertEqual(int(s.max()), 7)
        self.assertAlmostEqual(float(s.mean()), 2.94, places=2)

    def test_build_features_preserves_row_count(self):
        self.assertEqual(len(build_features(sample())), 6)


class TestSplitXY(unittest.TestCase):
    def test_split_xy_drops_id_and_target(self):
        X, y = split_xy(build_features(sample()))
        self.assertNotIn(config.ID_COL, X.columns)
        self.assertNotIn(config.TARGET, X.columns)
        self.assertEqual(len(X), len(y))

    def test_split_xy_returns_binary_target(self):
        _, y = split_xy(build_features(sample()))
        self.assertEqual(set(y), {0, 1})

    def test_split_xy_on_real_data_has_expected_width(self):
        X, y = split_xy(build_features(data_loader.load_clean()))
        # 21 raw columns - customerID and Churn - and 2 engineered columns, so 21 wide
        self.assertEqual(X.shape, (7032, 21))
        self.assertEqual(int(y.sum()), 1869)


class TestStratifiedSplit(unittest.TestCase):
    def test_split_preserves_class_balance(self):
        df = data_loader.load_clean()
        feat = build_features(df)
        X, y = split_xy(feat)
        Xtr, Xte, ytr, yte = stratified_split(X, y)
        full = y.mean()
        self.assertAlmostEqual(ytr.mean(), full, places=2)
        self.assertAlmostEqual(yte.mean(), full, places=2)

    def test_split_positive_counts_are_pinned(self):
        """Stratification on this dataset happens to leave the RATE identical either way
        at random_state=42, so a 2-dp rate comparison cannot detect losing `stratify`.
        Assert the exact counts instead.
        """
        X, y = split_xy(build_features(data_loader.load_clean()))
        _, _, ytr, yte = stratified_split(X, y)
        self.assertEqual(int(ytr.sum()), 1495)
        self.assertEqual(int(yte.sum()), 374)
        self.assertEqual(int(ytr.sum()) + int(yte.sum()), int(y.sum()))

    def test_stratify_is_passed_to_train_test_split(self):
        import ast
        import textwrap

        source = textwrap.dedent(Path(features_module.__file__).read_text(encoding="utf-8"))
        tree = ast.parse(source)
        found = False
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "train_test_split"):
                continue
            if any(kw.arg == "stratify" for kw in node.keywords):
                found = True
        self.assertTrue(found, "stratify= must be passed to train_test_split")

    def test_split_is_deterministic(self):
        # Comparing .shape proves nothing - shapes are identical for any random_state.
        # Compare the actual row identities instead.
        X, y = split_xy(build_features(data_loader.load_clean()))
        a_tr, a_te, _, _ = stratified_split(X, y)
        b_tr, b_te, _, _ = stratified_split(X, y)
        self.assertEqual(a_tr.index.tolist(), b_tr.index.tolist())
        self.assertEqual(a_te.index.tolist(), b_te.index.tolist())

    def test_split_actually_partitions_the_rows(self):
        X, y = split_xy(build_features(data_loader.load_clean()))
        Xtr, Xte, _, _ = stratified_split(X, y)
        overlap = set(Xtr.index) & set(Xte.index)
        self.assertEqual(overlap, set(), "a row appears in both train and test - leakage")
        self.assertEqual(len(set(Xtr.index) | set(Xte.index)), len(X))

    def test_test_set_size(self):
        X, y = split_xy(build_features(data_loader.load_clean()))
        Xtr, Xte, _, _ = stratified_split(X, y)
        # sklearn sizes the test fold with ceil, not round
        self.assertEqual(len(Xte), math.ceil(len(X) * config.TEST_SIZE))
        self.assertEqual(len(Xtr) + len(Xte), len(X))


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.feat = build_features(sample())
        self.X = self.feat.drop(columns=[config.ID_COL, config.TARGET])
        self.y = self.feat[config.TARGET]

    def test_pipeline_handles_unseen_category(self):
        # handle_unknown='ignore' is the whole point: at scoring time a new payment
        # method must not crash the model.
        pipe = make_pipeline(DummyClassifier(strategy="prior"))
        pipe.fit(self.X, self.y)
        unseen = self.X.iloc[[0]].copy()
        unseen["PaymentMethod"] = "Cryptocurrency"
        preds = pipe.predict_proba(unseen)
        self.assertEqual(preds.shape, (1, 2))

    def test_pipeline_outputs_probabilities_that_sum_to_one(self):
        pipe = make_pipeline(DummyClassifier(strategy="prior"))
        pipe.fit(self.X, self.y)
        proba = pipe.predict_proba(self.X)
        # each row's [P(no churn), P(churn)] must total 1
        self.assertTrue((numpy.abs(proba.sum(axis=1) - 1.0) < 1e-9).all())
        self.assertTrue(((proba[:, 1] >= 0).all()))
        self.assertTrue(((proba[:, 1] <= 1).all()))

    def test_pipeline_imputes_missing_numerics(self):
        # A bare fit/predict asserts nothing beyond "did not raise". Assert the imputer's
        # output actually equals the training median.
        from src.features import NUMERIC_FEATURES

        df = build_features(sample())
        X = df.drop(columns=[config.ID_COL, config.TARGET]).copy()
        X.loc[0, "tenure"] = None
        # median must be taken AFTER the hole is punched, because that is what the
        # imputer sees: dropping row 0 moves the tenure median from 18 to 24
        median_row = X[NUMERIC_FEATURES].median().tolist()

        pipe = make_pipeline(DummyClassifier(strategy="prior"))
        pipe.fit(X, df[config.TARGET])

        transformed = pipe.named_steps["preprocess"].transform(X)
        self.assertFalse(numpy.isnan(transformed).any(), "imputation left NaNs in the matrix")

        num_pipe = pipe.named_steps["preprocess"].named_transformers_["num"]
        expected = num_pipe.named_steps["scale"].transform([median_row])[0][0]
        self.assertAlmostEqual(transformed[0, 0], expected, places=9)

    def test_imputation_uses_the_training_median(self):
        # tenure after nulling row 0 is [_, 60, 24, 3, 48, 12] -> median 24
        from src.features import NUMERIC_FEATURES

        df = build_features(sample())
        X = df.drop(columns=[config.ID_COL, config.TARGET]).copy()
        X.loc[0, "tenure"] = None
        self.assertEqual(X["tenure"].median(), 24.0)
        pipe = make_pipeline(DummyClassifier(strategy="prior"))
        pipe.fit(X, df[config.TARGET])
        imputed = pipe.named_steps["preprocess"].named_transformers_["num"].named_steps["impute"]
        block = imputed.transform(X[NUMERIC_FEATURES])
        self.assertEqual(block[0, 0], 24.0)
    def test_every_engineered_feature_reaches_the_model(self):
        # Regression guard: three derived features were built and then silently dropped
        # because NUMERIC_FEATURES omitted them, with no warning.
        from src.features import DERIVED_COLUMNS, NUMERIC_FEATURES

        self.assertEqual(sorted(DERIVED_COLUMNS), sorted(set(DERIVED_COLUMNS) & set(NUMERIC_FEATURES)))

        df = data_loader.load_clean()
        X, y = split_xy(build_features(df))
        pipe = make_pipeline(DummyClassifier(strategy="prior"))
        pipe.fit(X, y)
        names = list(pipe.named_steps["preprocess"].get_feature_names_out())
        for col in DERIVED_COLUMNS:
            self.assertTrue(
                any(col in n for n in names), f"{col} is engineered but never reaches the model"
            )

    def test_model_sees_more_columns_than_the_raw_predictors(self):
        # 27 categorical levels (drop="first") + 5 numeric = 32 design columns
        df = data_loader.load_clean()
        X, y = split_xy(build_features(df))
        pipe = make_pipeline(DummyClassifier(strategy="prior"))
        pipe.fit(X, y)
        self.assertEqual(len(pipe.named_steps["preprocess"].get_feature_names_out()), 32)

    def test_design_matrix_collinearity_breakdown_is_pinned(self):
        """Rank deficiency is real and has two distinct causes with different fixes.

        Measured, not asserted, because the README makes a claim about it.
        """
        df = data_loader.load_clean()
        X, y = split_xy(build_features(df))
        r = collinearity_report(X, y)
        self.assertEqual(r["n_design_columns"], 32)
        self.assertEqual(r["rank"], 24)
        self.assertTrue(r["rank_deficient"])
        self.assertEqual(r["dependent_directions"], 8)
        # both engineered columns are exact linear combinations of existing ones
        self.assertEqual(r["dependent_from_engineered_columns"], 2)
        # the rest is structure in the telco data: no internet service means no add-ons
        self.assertEqual(r["dependent_from_dataset_structure"], 6)
        self.assertEqual(r["rank_if_engineered_columns_dropped"], 24)

    def test_dropping_engineered_columns_removes_only_their_own_dependency(self):
        df = data_loader.load_clean()
        X, y = split_xy(build_features(df))
        with_derived = collinearity_report(X, y)
        without = collinearity_report(X.drop(columns=list(DERIVED_COLUMNS)), y)
        # 2 fewer columns, rank unchanged -> the deficiency does not shrink
        self.assertEqual(with_derived["n_design_columns"] - without["n_design_columns"], 2)
        self.assertEqual(with_derived["rank"], without["rank"])
        # the structural part survives, which is why this is not a fixable bug
        self.assertTrue(without["rank_deficient"])

    def test_drop_first_encoder_actually_drops_a_level(self):
        # regression guard for the dummy-variable trap: with drop=None the matrix had
        # 48 design columns for 24 independent directions. drop="first" is the fix.
        df = data_loader.load_clean()
        X, y = split_xy(build_features(df))
        pipe = make_pipeline(DummyClassifier(strategy="prior"))
        pipe.fit(X, y)
        names = list(pipe.named_steps["preprocess"].get_feature_names_out())
        gender = [n for n in names if "gender" in n]
        self.assertEqual(len(gender), 1, f"expected 1 gender level, got {gender}")

    def test_pipeline_exposes_preprocessor_and_model_steps(self):
        pipe = make_pipeline(LogisticRegression())
        self.assertEqual(list(pipe.named_steps), ["preprocess", "model"])

    def test_pipeline_keeps_categorical_columns_as_separate_block(self):
        pipe = make_pipeline(LogisticRegression())
        blocks = pipe.named_steps["preprocess"].transformers
        names = [name for name, _, _ in blocks]
        self.assertIn("num", names)
        self.assertIn("cat", names)


if __name__ == "__main__":
    unittest.main()
