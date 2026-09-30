"""Exploratory analysis. Answers business questions, not "what is the shape of the data".

Every function returns a small tidy table or scalar so the findings can be asserted in
tests and quoted in the report. Figures are a side effect.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from . import config, data_loader

TENURE_BANDS = [(0, 6, "0-6 mo"), (7, 12, "7-12 mo"), (13, 24, "13-24 mo"), (25, 48, "25-48 mo"), (49, 100, "49+ mo")]


def churn_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Churn rate per category, sorted worst-first. The core segmentation table."""
    grouped = df.groupby(column)[config.TARGET].agg(["sum", "count"])
    grouped["churn_rate"] = grouped["sum"] / grouped["count"]
    # reindex onto the full category index: groupby on the churned subset omits a
    # category with zero churners, which would leave NaN revenue rather than 0.0
    revenue = (
        df[df[config.TARGET] == 1].groupby(column)["MonthlyCharges"].sum()
        .reindex(grouped.index)
        .fillna(0.0)
    )
    grouped["monthly_rev_at_risk"] = revenue
    return grouped.sort_values("churn_rate", ascending=False)


def tenure_band(tenure: int) -> str:
    for low, high, label in TENURE_BANDS:
        if low <= tenure <= high:
            return label
    raise ValueError(f"tenure out of range: {tenure}")


def add_tenure_band(df: pd.DataFrame) -> pd.DataFrame:
    """New frame with a tenure_band column. Input is not mutated."""
    return df.assign(tenure_band=df["tenure"].map(tenure_band))


def churn_rate_by_tenure_band(df: pd.DataFrame) -> pd.DataFrame:
    banded = add_tenure_band(df)
    grouped = banded.groupby("tenure_band")[config.TARGET].agg(["sum", "count"])
    grouped["churn_rate"] = grouped["sum"] / grouped["count"]
    order = [label for _, _, label in TENURE_BANDS]
    return grouped.reindex(order).dropna(subset=["churn_rate"])


def churn_rate_matrix(df: pd.DataFrame, row: str, col: str) -> pd.DataFrame:
    """Pivot churn rate across two categoricals, e.g. contract x internet service."""
    return pd.pivot_table(
        df, index=row, columns=col, values=config.TARGET, aggfunc="mean"
    ).round(3)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    summary = df[cols].describe().T
    summary["skew"] = df[cols].skew()
    return summary.round(3)


def churned_vs_retained_means(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    return df.groupby(config.TARGET)[cols].mean().round(2).rename(index={0: "retained", 1: "churned"})


def concentration(df: pd.DataFrame, top_frac: float = 0.1) -> dict:
    """How much of the churned revenue sits in the highest-earning churned decile.

    Answers "is this a broad problem or a concentrated one?" - it changes who you
    target and how much budget you need.

    Note the slice is the top `top_frac` of CHURNED customers by monthly charge, not
    the top decile of the whole base, so the keys carry the fraction rather than
    hard-coding "10pct".
    """
    churned = df[df[config.TARGET] == 1]
    n = int(len(churned) * top_frac)
    worst = churned.nlargest(n, "MonthlyCharges")
    total_rev_at_risk = float(churned["MonthlyCharges"].sum())
    frac_pct = round(100 * top_frac, 1)
    return {
        "churned_customers": int(len(churned)),
        "total_monthly_rev_at_risk": total_rev_at_risk,
        f"top_{frac_pct}pct_of_churned_customers": n,
        f"top_{frac_pct}pct_rev": float(worst["MonthlyCharges"].sum()),
        f"top_{frac_pct}pct_rev_share": float(worst["MonthlyCharges"].sum() / total_rev_at_risk),
        "avg_monthly_charge_churned": float(churned["MonthlyCharges"].mean()),
        "avg_monthly_charge_retained": float(df.loc[df[config.TARGET] == 0, "MonthlyCharges"].mean()),
    }


def _save(fig, name: str) -> None:
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / name, dpi=140)
    plt.close(fig)


def plot_churn_rate_by_category(df: pd.DataFrame, column: str, name: str) -> None:
    table = churn_rate_by(df, column)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(table.index[::-1], table["churn_rate"][::-1], color="#c44e52")
    ax.set_xlabel("Churn rate")
    ax.set_title(f"Churn rate by {column}")
    ax.axvline(df[config.TARGET].mean(), ls="--", color="grey", lw=1)
    for y, v in enumerate(table["churn_rate"][::-1]):
        ax.text(v + 0.005, y, f"{v:.1%}", va="center", fontsize=9)
    _save(fig, name)


def plot_churn_by_tenure_band(df: pd.DataFrame) -> None:
    table = churn_rate_by_tenure_band(df)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(table.index, table["churn_rate"], color="#4c72b0")
    ax.set_xlabel("Tenure band")
    ax.set_ylabel("Churn rate")
    ax.set_title("Churn collapses as tenure grows")
    for i, v in enumerate(table["churn_rate"]):
        ax.text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=9)
    _save(fig, "churn_by_tenure_band.png")


def plot_tenure_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bins = range(0, 75, 5)
    ax.hist(df.loc[df[config.TARGET] == 0, "tenure"], bins=bins, alpha=0.6, label="retained", color="#55a868")
    ax.hist(df.loc[df[config.TARGET] == 1, "tenure"], bins=bins, alpha=0.6, label="churned", color="#c44e52")
    ax.set_xlabel("Tenure (months)")
    ax.set_title("Tenure distribution by churn status")
    ax.legend()
    _save(fig, "tenure_distribution_by_churn.png")


def plot_monthly_charges(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    data = [df.loc[df[config.TARGET] == 0, "MonthlyCharges"], df.loc[df[config.TARGET] == 1, "MonthlyCharges"]]
    # matplotlib >=3.11 renamed the boxplot `labels` kwarg to `tick_labels`
    ax.boxplot(data, tick_labels=["retained", "churned"], patch_artist=True, boxprops={"facecolor": "#a8c4e0"})
    ax.set_ylabel("Monthly charges ($)")
    ax.set_title("Churned customers pay more per month")
    _save(fig, "monthly_charges_by_churn.png")


def plot_heatmap(df: pd.DataFrame) -> None:
    matrix = churn_rate_matrix(df, "Contract", "InternetService")
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.imshow(matrix.values, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.values[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=10)
    ax.set_title("Churn rate: contract x internet service")
    _save(fig, "churn_heatmap_contract_internet.png")


def run() -> dict:
    """Full EDA. Returns the headline numbers used in the README and report."""
    raw = data_loader.load_raw()
    df = data_loader.clean(raw)

    print("=" * 62)
    print("DATA QUALITY")
    print("=" * 62)
    report = data_loader.data_quality_report(raw, df)
    print(json.dumps(report, indent=2))

    print("\n" + "=" * 62)
    print("OVERALL")
    print("=" * 62)
    print(f"churn rate            : {data_loader.churn_rate(df):.2%}")
    print(f"monthly rev at risk   : ${data_loader.revenue_at_risk(df):,.2f}")

    print("\nnumeric summary (raw):")
    print(numeric_summary(df).to_string())
    print("\nmeans by churn status:")
    print(churned_vs_retained_means(df).to_string())

    print("\n" + "=" * 62)
    print("CHURN RATE BY CATEGORY")
    print("=" * 62)
    findings = {}
    for col in ["Contract", "PaymentMethod", "InternetService", "PaperlessBilling", "Dependents", "SeniorCitizen"]:
        table = churn_rate_by(df, col)
        findings[col] = {k: round(float(v), 4) for k, v in table["churn_rate"].items()}
        print(f"\n-- {col}")
        print(table[["sum", "count", "churn_rate", "monthly_rev_at_risk"]].round(3).to_string())

    print("\n" + "=" * 62)
    print("CHURN BY TENURE BAND")
    print("=" * 62)
    tenure_table = churn_rate_by_tenure_band(df)
    findings["tenure_band"] = {k: round(float(v), 4) for k, v in tenure_table["churn_rate"].items()}
    print(tenure_table.round(3).to_string())

    print("\n" + "=" * 62)
    print("CONCENTRATION")
    print("=" * 62)
    conc = concentration(df)
    findings["concentration"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in conc.items()}
    print(json.dumps(findings["concentration"], indent=2))

    print("\ncontract x internet churn rate:")
    print(churn_rate_matrix(df, "Contract", "InternetService").to_string())

    plot_churn_rate_by_category(df, "Contract", "churn_by_contract.png")
    plot_churn_rate_by_category(df, "PaymentMethod", "churn_by_payment_method.png")
    plot_churn_rate_by_category(df, "InternetService", "churn_by_internet_service.png")
    plot_churn_by_tenure_band(df)
    plot_tenure_distribution(df)
    plot_monthly_charges(df)
    plot_heatmap(df)
    print(f"\nfigures written to {config.FIG_DIR}")

    config.ROOT.joinpath("reports").mkdir(parents=True, exist_ok=True)
    (config.ROOT / "reports" / "eda_findings.json").write_text(json.dumps(findings, indent=2))
    return findings


if __name__ == "__main__":
    run()
