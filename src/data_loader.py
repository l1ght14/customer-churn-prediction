"""Load and clean the IBM Telco churn CSV.

Cleaning decisions (each one is defensible in an interview):
1. TotalCharges is read as a string in the raw file because 11 rows are blank.
   Force to numeric, which turns those blanks into NaN instead of 0.
2. Those 11 NaN rows all have tenure == 0: the customer signed up and left before
   any invoice was raised. They carry no revenue signal, so we drop them.
   Imputing 0 would silently invent a "free" customer and bias the churn model.
3. TotalCharges has a trailing whitespace variant in the raw file -> strip.
4. SeniorCitizen is 0/1 ints but is really a categorical -> cast to string "No"/"Yes"
   so the one-hot encoder treats it the same as every other Yes/No column.
"""
from __future__ import annotations

import pandas as pd

from . import config


def load_raw(path=None) -> pd.DataFrame:
    """Read the raw CSV exactly as-is. No cleaning, no imputation."""
    path = path or config.RAW_CSV
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy. Input is never mutated."""
    out = df.copy()

    text_cols = out.select_dtypes(include=["string", "object"]).columns
    out[text_cols] = out[text_cols].apply(lambda col: col.str.strip())

    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")

    out = out[out["TotalCharges"].notna()].copy()
    out = out.reset_index(drop=True)

    out["SeniorCitizen"] = _as_yes_no(out["SeniorCitizen"])
    out[config.TARGET] = out[config.TARGET].map({"No": 0, "Yes": 1})
    out[config.TARGET] = out[config.TARGET].astype(int)

    return out


def _as_yes_no(series: pd.Series) -> pd.Series:
    """Normalise SeniorCitizen to Yes/No from either encoding.

    clean() is written for a raw frame, where this is 0/1. Re-running it on an already
    cleaned frame would map {"Yes","No"} through a {0:..,1:..} dict and produce an
    all-NaN column with no warning - so handle that spelling too rather than trust the
    caller to know which frame it has.
    """
    if series.dtype == object or str(series.dtype).startswith("string"):
        return series
    return series.map({0: "No", 1: "Yes"})


def load_clean(path=None) -> pd.DataFrame:
    """Convenience: raw -> clean in one call."""
    return clean(load_raw(path))


def churn_rate(df: pd.DataFrame) -> float:
    """Share of customers who churned. The one number that frames the problem."""
    return float(df[config.TARGET].mean())


def revenue_at_risk(df: pd.DataFrame) -> float:
    """Monthly recurring revenue currently sitting on customers who already churned.

    This is the number the retention team is arguing about, so it goes in the README.
    """
    return float(df.loc[df[config.TARGET] == 1, "MonthlyCharges"].sum())


def data_quality_report(raw: pd.DataFrame, clean: pd.DataFrame) -> dict:
    """Before/after evidence that the cleaning did something and why."""
    return {
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(clean)),
        "rows_dropped": int(len(raw) - len(clean)),
        "blank_totalcharges_rows": int((raw["TotalCharges"].str.strip() == "").sum()),
        "dropped_rows_all_tenure_zero": bool(
            (raw.loc[raw["TotalCharges"].str.strip() == "", "tenure"] == 0).all()
        ),
        "dropped_rows_all_non_churned": bool(
            (raw.loc[raw["TotalCharges"].str.strip() == "", config.TARGET] == "No").all()
        ),
        "duplicate_customer_ids": int(clean[config.ID_COL].duplicated().sum()),
        "null_cells": int(clean.isna().sum().sum()),
        "churn_rate": churn_rate(clean),
        "churned_customers": int(clean[config.TARGET].sum()),
    }


def assert_no_leakage(df: pd.DataFrame) -> None:
    """Guard: a customer must not appear in both train and test."""
    if df[config.ID_COL].duplicated().any():
        raise AssertionError("duplicate customerID detected - train/test split would leak")


def missing_columns(df: pd.DataFrame) -> list[str]:
    """Columns absent from a frame vs the expected raw schema (diagnostic helper)."""
    expected = {
        config.ID_COL,
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        config.TARGET,
    }
    return sorted(expected - set(df.columns))


if __name__ == "__main__":
    import json

    raw = load_raw()
    cld = clean(raw)
    print(json.dumps(data_quality_report(raw, cld), indent=2))
    print("monthly revenue at risk: $%.2f" % revenue_at_risk(cld))
