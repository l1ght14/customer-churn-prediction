"""Evaluation: curves, calibration, threshold choice, and the ranked customer export.

Everything here answers a question a retention manager actually asks:
  - Can I trust the score?        -> calibration plot
  - Who is highest risk?          -> decile lift table + ranked CSV export
  - What if I can only afford N?  -> threshold sweep at fixed campaign sizes
  - What does it buy me?          -> score_summary (a backtest, not a forecast)
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config, data_loader, metrics, train
from .data_loader import load_clean
from .features import build_features, make_pipeline, split_xy, stratified_split


def _save(fig, name: str) -> None:
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / name, dpi=140)
    plt.close(fig)


def plot_precision_recall(y_true, y_prob, name="precision_recall_curve.png") -> None:
    """PR curve with the base-rate reference line.

    The dashed line is the score a random ranking gets. Distance above that line is
    the entire value of the model, which is why PR-AUC is the selection metric here.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    base = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.plot(recall, precision, color="#c44e52", lw=2, label=f"model (PR-AUC {average_precision_score(y_true, y_prob):.3f})")
    ax.axhline(base, ls="--", color="grey", lw=1, label=f"random baseline ({base:.3f})")
    ax.set_xlabel("Recall (share of churners caught)")
    ax.set_ylabel("Precision (share of flagged who churn)")
    ax.set_title("Precision-Recall curve")
    ax.legend(loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    _save(fig, name)


def plot_roc(y_true, y_prob, name="roc_curve.png") -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(fpr, tpr, color="#4c72b0", lw=2, label=f"model (ROC-AUC {roc_auc_score(y_true, y_prob):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    _save(fig, name)


def plot_gain_and_lift(y_true, y_prob, name="gain_lift_curve.png") -> None:
    """The retention team's chart: how deep down the risk list do you have to go."""
    gains, lift = metrics.cumulative_gain(y_true, y_prob)
    depth = np.linspace(0, 1, len(gains))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(depth, gains, color="#c44e52", lw=2)
    ax1.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    ax1.set_xlabel("Share of customers contacted")
    ax1.set_ylabel("Cumulative share of churners found")
    ax1.set_title("Gain curve")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.02)

    ax2.plot(depth, lift, color="#4c72b0", lw=2)
    ax2.axhline(1.0, ls="--", color="grey", lw=1)
    ax2.set_xlabel("Share of customers contacted")
    ax2.set_ylabel("Lift vs random targeting")
    ax2.set_title("Lift curve")
    ax2.set_xlim(0, 1)
    _save(fig, name)


def plot_calibration(y_true, y_prob, name="calibration_curve.png", n_bins: int = 10) -> None:
    """Are the probabilities usable as-is, or only their ordering?

    Ordering-only is enough for ranking customers into a call list. It is NOT enough if
    finance wants "this customer has an 80% chance of leaving" in a contract.
    """
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", color="#55a868", lw=2, label="model")
    ax.set_xlabel("Predicted probability (mean per bin)")
    ax.set_ylabel("Observed churn rate")
    ax.set_title("Calibration - is 0.5 really 50%?")
    ax.legend(loc="upper left")
    _save(fig, name)


def plot_decile_lift(table: pd.DataFrame, name="decile_lift.png") -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    colors = ["#c44e52" if v > 1 else "#bbbbbb" for v in table["lift"]]
    ax.bar(table["decile"], table["lift"], color=colors)
    ax.axhline(1.0, ls="--", color="grey", lw=1, label="random targeting")
    above = int((table["lift"] > 1).sum())
    ax.set_xlabel("Decile (1 = highest predicted risk)")
    ax.set_ylabel("Lift vs random")
    # title states what the bars actually show rather than a rounder claim
    ax.set_title(f"Decile lift vs random - {above} of {len(table)} deciles beat random")
    ax.legend()
    _save(fig, name)


def plot_threshold_tradeoff(y_true, y_prob, name="threshold_tradeoff.png") -> None:
    """Precision and recall as the threshold moves. The "where do we operate" chart."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # thresholds has one fewer entry than precision/recall
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(thresholds, precision[:-1], label="precision", color="#4c72b0")
    ax.plot(thresholds, recall[:-1], label="recall", color="#c44e52")
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Rate")
    ax.set_title("Precision-recall trade-off across thresholds")
    ax.legend()
    ax.set_xlim(0, 1)
    _save(fig, name)


def threshold_sweep(y_true, y_prob, y_prob_train) -> pd.DataFrame:
    """Campaign sizes a retention team would actually consider, with the trade-off exposed.

    y_prob_train is REQUIRED and positional. Defaulting it to y_prob would silently
    re-derive every campaign cut from the very set it is then scored against, which is
    the circularity this function exists to avoid. An omission should be a TypeError,
    not a quiet leak.
    """
    rows = []
    for share in (0.05, 0.10, 0.15, 0.20, 0.30):
        thr = metrics.threshold_for_target_size(y_prob_train, target_share=share)
        rep = metrics.classification_report_at(y_true, y_prob, threshold=thr)
        rows.append(
            {
                "target_share": share,
                "threshold_from_train": round(thr, 4),
                "customers_targeted": rep["true_positives"] + rep["false_positives"],
                "churners_caught": rep["true_positives"],
                "recall": rep["recall"],
                "precision": rep["precision"],
                "f2": rep["f2"],
            }
        )
    return pd.DataFrame(rows)


def score_all_customers(model, df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Rank every customer by risk using OUT-OF-FOLD predictions.

    Scoring all rows with the fitted model would be dishonest: 80% of them were in its
    training set, so their scores are optimistic and the campaign summary would quote a
    lift the model has never actually earned on unseen data. cross_val_predict gives
    every customer a score from a fold that never saw them, so the reported lift is a
    real out-of-sample number.

    The Churn column is HISTORICAL. Customers flagged Churn=1 have already left; this
    file is a backtest ranking, not a live prospect list. See the README caveat.
    """
    feat = build_features(df)
    data_loader.assert_no_leakage(feat)
    X, y = split_xy(feat)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    oof_prob = cross_val_predict(
        make_pipeline(model), X, y, cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]

    keep = [config.ID_COL, config.TARGET, "MonthlyCharges", "tenure", "Contract", "PaymentMethod", "InternetService"]
    scored = feat[keep].copy()
    # Round before writing. The forest runs n_jobs=1 inside cross_val_predict, so float
    # accumulation order varies between runs and the last bits differ; unrounded that
    # makes the committed deliverable change hash on every rebuild, which is a permanent
    # spurious diff. 10dp is far below any decision boundary.
    scored["churn_score"] = np.round(oof_prob, 10)
    scored["already_churned"] = scored[config.TARGET].eq(1)
    return scored.sort_values("churn_score", ascending=False).reset_index(drop=True)


def run() -> dict:
    df = load_clean()
    feat = build_features(df)
    X, y = split_xy(feat)
    data_loader.assert_no_leakage(feat)
    Xtr, Xte, ytr, yte = stratified_split(X, y)

    best_name = json.loads(config.METRICS_JSON.read_text())["best_model"]
    model = train.candidate_models()[best_name]

    # Refit from source rather than loading a pickled model: nothing in this project
    # reads a saved binary, it is 5 MB, and it cannot be committed. A reviewer cloning
    # the repo gets the same numbers from the same code path.
    pipe = make_pipeline(model)
    pipe.fit(Xtr, ytr)
    yte_prob = pipe.predict_proba(Xte)[:, 1]
    # shared with train.py on purpose: out_of_fold_scores is the one place that knows how
    # train scores must be produced, and it is covered by a test that recomputes the
    # thresholds from it. Duplicating the cross_val_predict call here would create a
    # second, untested path to the campaign cuts in the playbook.
    ytr_prob = train.out_of_fold_scores(model, Xtr, ytr)

    print("=" * 100)
    print("CURVES  (holdout, model selected on train-only CV)")
    print("=" * 100)
    rep = metrics.classification_report_at(yte, yte_prob, threshold=0.5)
    print(f"model              : {best_name}")
    prauc = f"{rep['pr_auc']:.4f}" if rep["pr_auc"] is not None else "n/a (single-class slice)"
    roc = f"{rep['roc_auc']:.4f}" if rep["roc_auc"] is not None else "n/a (single-class slice)"
    print(f"holdout PR-AUC     : {prauc}")
    print(f"holdout ROC-AUC    : {roc}")
    print(f"holdout baseline   : {yte.mean():.4f}  (PR-AUC of random ranking)")

    plot_precision_recall(yte, yte_prob)
    plot_roc(yte, yte_prob)
    plot_gain_and_lift(yte, yte_prob)
    plot_calibration(yte, yte_prob)
    plot_threshold_tradeoff(yte, yte_prob)

    lift_table = metrics.lift_and_gain(yte, yte_prob)
    plot_decile_lift(lift_table)

    print("\n" + "=" * 100)
    print("THRESHOLD SWEEP - thresholds from TRAIN scores, measured on holdout")
    print("=" * 100)
    sweep = threshold_sweep(yte, yte_prob, y_prob_train=ytr_prob)
    print(sweep.to_string(index=False))

    print("\n" + "=" * 100)
    print("BACKTEST: top 10% of customers by out-of-fold score")
    print("  OBSERVED churn outcomes, not a forecast. Churn=1 customers already left.")
    print("=" * 100)
    scored = score_all_customers(model, df)
    summary = metrics.score_summary(scored)
    for k, v in summary.items():
        print(f"{k:<34} {v}")

    config.SCORED_CSV.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(config.SCORED_CSV, index=False)
    print(f"\nscored customers written to {config.SCORED_CSV}")

    top = scored.head(20)
    already = int(top[config.TARGET].sum())
    print(f"\n20 highest-risk customers ({already} of 20 already churned - postmortem ranking, not a prospect list):")
    print(top[[config.ID_COL, "churn_score", config.TARGET, "MonthlyCharges", "Contract", "tenure"]].to_string(index=False))

    out = {
        "model": best_name,
        "holdout_report_at_0.5": rep,
        "baseline_pr_auc": round(float(yte.mean()), 4),
        "threshold_sweep": sweep.to_dict(orient="records"),
        "lift_table": lift_table.to_dict(orient="records"),
        "backtest_summary_top_10pct": summary,
    }
    config.ROOT.joinpath("reports").mkdir(parents=True, exist_ok=True)
    (config.ROOT / "reports" / "evaluation.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()
