"""Feature engineering and the preprocessing pipeline.

Two engineered features survive, and the honest reason is compression, not information:

  contract_months  1 / 12 / 24 for the three contract levels. Exact linear encoding of
                   what the Contract one-hot already says, but a single ordered column
                   instead of three, which is a cheaper search space for the trees.
  services_count   how many add-ons the customer pays for, 0-7. A sum of seven one-hots.

Deliberately NOT engineered, and this is the part worth defending: an earlier version
also built is_month_to_month, is_echeck and has_internet. Those are *exact* linear
combinations of columns already present - is_month_to_month is the Month-to-month
one-hot, is_echeck is the Electronic check one-hot, has_internet is DSL + Fiber. Least
squares reconstruction of each from the other design columns has zero residual. Adding
them made the matrix rank-deficient for nothing, and under exact collinearity L2 splits
weight arbitrarily across the duplicates, which destroys the logistic-regression
coefficients a retention manager is supposed to be able to read.

Same logic rejects TotalCharges / tenure, which is MonthlyCharges by construction.

So the rule applied throughout: no feature that a linear combination of the existing
design matrix can reconstruct.
"""
from __future__ import annotations

import pandas as pd
import numpy
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

CONTRACT_MONTHS = {"Month-to-month": 1, "One year": 12, "Two year": 24}

# The seven add-on / access columns that count as "engagement" breadth.
SERVICE_COLUMNS = [
    "PhoneService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# The six that share a "No internet service" level, which is why the matrix is
# rank-deficient for reasons that live in the data rather than in the encoding.
ADDON_COLUMNS = [c for c in SERVICE_COLUMNS if c != "PhoneService"]

DERIVED_COLUMNS = ["contract_months", "services_count"]

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
] + DERIVED_COLUMNS


def contract_months(contract: pd.Series) -> pd.Series:
    """Map contract label to months. Unknown labels raise rather than silently becoming NaN."""
    unknown = set(contract.unique()) - set(CONTRACT_MONTHS)
    if unknown:
        raise ValueError(f"unmapped contract labels: {sorted(unknown)}")
    return contract.map(CONTRACT_MONTHS).astype(int)


def count_services(row: pd.Series) -> int:
    """How many of the seven service columns say 'Yes'.

    Deliberately == 'Yes' and not != 'No'. 'No internet service' is a real third level in
    this dataset: a customer without internet cannot hold any add-on, and counting
    `!= 'No'` would score those 1,520 customers as holding all seven: PhoneService is 'Yes' for
    every one of them, so the six 'No internet service' levels all read as held.
    """
    return int(sum(1 for col in SERVICE_COLUMNS if row[col] == "Yes"))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new frame with engineered columns added. Input is not mutated."""
    out = df.copy()
    out["contract_months"] = contract_months(out["Contract"])
    # vectorised equivalent of count_services applied row-wise; df.apply(axis=1) walks
    # 7,032 Python-level rows and runs three times per pipeline invocation
    out["services_count"] = (out[SERVICE_COLUMNS] == "Yes").sum(axis=1).astype(int)
    return out


def collinearity_report(X, y) -> dict:
    """How many design columns are independent? Measured, not asserted.

    The design matrix is rank-deficient, and it is worth being precise about why,
    because the two causes have different fixes.

    Cause 1 - the engineered columns. contract_months is 1/12/24 by contract level and
    services_count is a sum of the seven service one-hots, so each is an EXACT linear
    combination of columns already present. That is 2 dependent directions, and it is
    removable: drop the two columns.

    Cause 2 - structure inside the data. A customer with no internet service cannot have
    Online Security, so the "No internet service" level of all six add-on columns is the
    same indicator row as InternetService == "No". Those 6 dependent directions are not a
    bug and cannot be removed without discarding real information.

    Consequence for model choice, not for accuracy:

      * Gradient boosting and the forests are unaffected. Trees split on one feature at a
        time, and both encodings expose the same cut points.
      * Logistic regression IS affected. Under exact collinearity L2 has no unique
        minimum, so it spreads weight arbitrarily across the duplicate columns and no
        individual coefficient is interpretable. That is why the logistic-regression row
        is a reference point rather than the recommendation.
    """
    numeric_cols = [c for c in NUMERIC_FEATURES if c in X.columns]
    categorical_cols = [c for c in config.RAW_TEXT_FEATURES if c in X.columns]

    def rank_of(numeric, categorical) -> tuple[int, int]:
        pipe = make_pipeline(DummyClassifier(strategy="prior"), numeric, categorical)
        pipe.fit(X, y)
        matrix = pipe.named_steps["preprocess"].transform(X)
        return int(matrix.shape[1]), int(numpy.linalg.matrix_rank(matrix))

    cols, rank = rank_of(numeric_cols, categorical_cols)

    without_derived = [c for c in numeric_cols if c not in DERIVED_COLUMNS]
    cols_nd, rank_nd = rank_of(without_derived, categorical_cols)

    addons = [c for c in ADDON_COLUMNS if c in categorical_cols]
    cols_nda, rank_nda = rank_of(
        without_derived, [c for c in categorical_cols if c not in addons]
    )

    return {
        "n_design_columns": cols,
        "rank": rank,
        "dependent_directions": cols - rank,
        "rank_deficient": bool(rank < cols),
        # how much of the deficiency the engineered columns account for
        "dependent_from_engineered_columns": (cols - rank) - (cols_nd - rank_nd),
        # what is left is structure in the telco data itself
        "dependent_from_dataset_structure": (cols_nd - rank_nd) - (cols_nda - rank_nda),
        "rank_if_engineered_columns_dropped": rank_nd,
    }


def split_xy(feat: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Features and target, with the customer id dropped.

    customerID is a label, not a signal: keeping it lets the model memorise rows
    instead of learning the pattern.
    """
    missing = [c for c in NUMERIC_FEATURES + config.RAW_TEXT_FEATURES if c not in feat.columns]
    if missing:
        raise KeyError(f"expected feature columns missing from the frame: {missing}")
    X = feat.drop(columns=[config.ID_COL, config.TARGET])
    y = feat[config.TARGET]
    return X, y


def stratified_split(X: pd.DataFrame, y: pd.Series):
    """80/20 split that preserves the churn ratio on both sides.

    Stratifying matters here: with only 26% positives, an unlucky split can move the
    test churn rate enough to make PR-AUC non-comparable between runs.
    """
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )


def make_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    """Numeric: median-impute + standardise. Categorical: mode-impute + one-hot.

    handle_unknown='ignore' because scoring runs against future data that can contain
    payment methods or contract types absent from today's training set.
    """
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            # drop="first" avoids the dummy-variable trap. With drop=None every
            # categorical block's levels sum to 1, which is collinear with the constant
            # the classifier fits itself, so the raw matrix is rank-deficient before any
            # engineered column is added. Trees do not care; L2 regression splits weight
            # across the redundant level columns and no single coefficient is readable.
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, list(numeric_cols)), ("cat", categorical, list(categorical_cols))],
        remainder="drop",
    )


def make_pipeline(model, numeric_cols=None, categorical_cols=None) -> Pipeline:
    """Preprocessor + classifier as one object.

    Keeping them fused means cross_val_score and GridSearchCV refit the imputer and
    the scaler inside each fold. If preprocessing sat outside the CV loop, the test
    fold's distribution would leak into training through the scaler's statistics.

    The estimator is CLONED. Without that the caller's instance becomes the fitted step
    in place, so calling `make_pipeline(m).fit(...)` twice with the same `m` silently
    rewrites the first pipeline's predictions - and two callers sharing one model object
    fight over its state. It also means the model the caller holds is never the model
    that was trained, which is a nasty thing to debug.

    Column lists default to the project schema but are overridable, so a caller can ask
    what the matrix looks like with a feature removed.
    """
    numeric_cols = NUMERIC_FEATURES if numeric_cols is None else numeric_cols
    categorical_cols = config.RAW_TEXT_FEATURES if categorical_cols is None else categorical_cols
    return Pipeline(
        [
            ("preprocess", make_preprocessor(numeric_cols, categorical_cols)),
            ("model", clone(model)),
        ]
    )
