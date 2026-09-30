"""Metric helpers for imbalanced binary classification.

Why not accuracy: at 26% churn, a model that predicts "nobody churns" scores 73.4%
accuracy and is worthless.

Why not ROC-AUC, stated carefully: the usual justification is wrong. A model that misses
every churner has ROC-AUC exactly 0.5, so ROC-AUC does not flatter a useless model. The
real argument is the baseline - random scores 0.2658 on PR-AUC (the base rate) but 0.5000
on ROC-AUC, so ROC-AUC cannot distinguish "learned nothing" from "learned a little".
Precision is also not diluted by the majority class, which is what makes a precision-based
threshold sweep readable as "what fraction of my contacts are worth it".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config


def classification_report_at(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Precision / recall / F1 / F2 at one threshold, plus threshold-free PR and ROC AUC.

    F2 is included deliberately: it weights recall ~4x more than precision, which is
    what you want when a missed churn is a lost customer and a false alarm costs
    one discount email.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.size == 0:
        raise ValueError("classification_report_at received an empty y_true")
    if not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    beta2 = 4.0
    f2 = ((1 + beta2) * precision * recall / (beta2 * precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true)

    report = {
        "threshold": round(float(threshold), 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "specificity": round(float(specificity), 4),
        "f1": round(float(f1), 4),
        "f2": round(float(f2), 4),
        "accuracy": round(float(accuracy), 4),
    }
    # ROC-AUC and PR-AUC are undefined when only one class is present. Always emit the
    # keys, as None, so callers can index them unconditionally without a KeyError.
    if y_true.min() != y_true.max():
        report["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
        report["pr_auc"] = round(float(average_precision_score(y_true, y_prob)), 4)
    else:
        report["roc_auc"] = None
        report["pr_auc"] = None
    return report


def lift_and_gain(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """Decile lift/gain table.

    Sort by predicted risk, split into 10 equal groups, then:
      churn_rate_in_decile = share of that decile which actually churned
      lift  = churn_rate_in_decile / base churn rate. "If we contact this decile, how
              many times more churners do we find than by contacting random customers?"
              Lift of 2.5 in the top decile is the number retention teams quote.
      cum_gain = % of ALL churners captured once you include this decile and everything
              above it.

    lift is per-decile, not cumulative: cumulative gain saturates at 1.0 in the top
    decile when the ranking is good, which flattens the rest of the column and hides
    where the model stops being useful. Per-decile lift decays on average and shows you
    roughly how deep down the list you can go before targeting stops paying. It is not
    strictly monotone: real deciles of unequal size and score distribution can invert
    (in this project's own holdout, decile 5 is 0.64x and decile 6 is 0.69x), so read the
    column, do not assume it.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.size == 0:
        raise ValueError("lift_and_gain received an empty y_true")
    if not 0 < y_true.sum():
        raise ValueError("lift_and_gain needs at least one positive; base rate is undefined")
    if y_true.size < n_bins:
        # otherwise some bins are empty, chunk.mean() is NaN, and json.dumps writes a
        # bare NaN token that strict parsers reject
        raise ValueError(
            f"lift_and_gain needs at least n_bins={n_bins} rows, got {y_true.size}; "
            "fewer bins than rows would produce empty deciles"
        )

    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    total_pos = y_sorted.sum()

    rows = []
    for i in range(n_bins):
        lo = int(len(y_sorted) * i / n_bins)
        hi = int(len(y_sorted) * (i + 1) / n_bins)
        chunk = y_sorted[lo:hi]
        rows.append(
            {
                "decile": i + 1,
                "n_customers": len(chunk),
                "share_of_customers": len(chunk) / len(y_sorted),
                "churners": int(chunk.sum()),
                "churn_rate_in_decile": float(chunk.mean()),
                "cum_gain": float(y_sorted[:hi].sum() / total_pos),
            }
        )

    table = pd.DataFrame(rows)
    base_rate = float(y_true.mean())
    table["lift"] = table["churn_rate_in_decile"] / base_rate
    return table.round(4)


def cumulative_gain(y_true, y_prob) -> tuple[np.ndarray, np.ndarray]:
    """Gain and lift across all thresholds. Used for the gain chart."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.size == 0 or not y_true.sum():
        raise ValueError("cumulative_gain needs a non-empty y_true with positives")
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]

    gains = np.cumsum(y_sorted) / y_sorted.sum()
    depth = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    lift = gains / depth
    return np.concatenate([[0.0], gains]), np.concatenate([[0.0], lift])


def threshold_for_target_recall(y_true, y_prob, target_recall: float = 0.80) -> float:
    """Tightest threshold that still captures at least target_recall of the churners.

    Retention teams get a fixed contact budget, so the question is never "what is the
    best threshold" but "how deep down the risk list can we go and still catch 80%".
    This returns the TIGHTEST qualifying cut (the highest score that still meets the
    constraint), so it contacts as few customers as the constraint allows. Ties can push
    the achieved recall above the target - that overshoot is inherent to thresholding.

    Pass the labels of the set the threshold is chosen ON. Applying that threshold to a
    different set is how you end up reporting a recall you pinned yourself.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.size == 0 or not y_true.sum():
        raise ValueError("threshold_for_target_recall needs a non-empty y_true with positives")
    if not 0 < target_recall <= 1:
        raise ValueError(f"target_recall must be in (0, 1], got {target_recall}")

    order = np.argsort(-y_prob)
    cum = np.cumsum(y_true[order]) / y_true.sum()
    # cum[-1] is exactly 1.0 by construction, so any target in (0, 1] is reachable and
    # no unreachable-target branch is needed. The range check above is the guard.
    reached = int(np.argmax(cum >= target_recall))
    # inclusive threshold: the customer at `reached` is contacted
    return float(y_prob[order][reached])


def threshold_for_target_size(y_prob, target_share: float = 0.10) -> float:
    """Threshold whose top-target_share cut selects roughly target_share of customers.

    Takes no labels on purpose. A size-based cut is a business decision about contact
    budget, not an inference, so it must be available before you know any outcomes -
    which is exactly what a live scoring pipeline needs.

    Ties are the caveat: this returns a score, and every customer holding that exact
    score gets flagged. With a heavily tied score distribution the flagged count can
    exceed target_share, and score_summary (which counts with nlargest) will disagree.
    Real risk scores are continuous, so this is a property of the method rather than a
    bug - but it is asserted in the tests so nobody is surprised by it.
    """
    y_prob = np.asarray(y_prob)
    if y_prob.size == 0:
        raise ValueError("threshold_for_target_size received an empty y_prob")
    if not 0 < target_share <= 1:
        raise ValueError(f"target_share must be in (0, 1], got {target_share}")
    # max(1, ...): a tiny budget like 1% of 10 rows rounds to 0, and top[-1] would be the
    # MINIMUM score - flagging everyone, the opposite of the intent, with no error.
    cutoff = max(1, int(round(len(y_prob) * target_share)))
    top = np.sort(y_prob)[::-1]
    return float(top[cutoff - 1])


def score_summary(df: pd.DataFrame, score_col: str = "churn_score", target_share: float = 0.10) -> dict:
    """What a campaign targeting the top slice would look like. Pure business numbers.

    Note these are OBSERVED outcomes. `churners_caught` counts customers who already
    churned, so this is a backtest of a campaign, not a forecast of one.
    """
    n = len(df)
    if n == 0:
        raise ValueError("score_summary received an empty frame")
    if not 0 < target_share <= 1:
        raise ValueError(f"target_share must be in (0, 1], got {target_share}")
    if score_col not in df.columns:
        raise KeyError(f"score column {score_col!r} not in frame")
    target = int(round(n * target_share))
    if target < 1:
        # otherwise nlargest(0) gives an empty top and the means below become NaN
        target = 1
    top = df.nlargest(target, score_col)
    actual_churners = int(top[config.TARGET].sum())
    total_churners = int(df[config.TARGET].sum())
    capture_rate = actual_churners / total_churners if total_churners else 0.0
    return {
        "customers_scored": n,
        "customers_targeted": target,
        "targeted_share_of_base": round(target / n, 4),
        "total_churners": total_churners,
        "churners_caught": actual_churners,
        "churners_caught_not_targeted": total_churners - actual_churners,
        "capture_rate": round(capture_rate, 4),
        "lift_vs_random": round(capture_rate / (target / n), 4),
        "monthly_rev_in_target": round(float(top["MonthlyCharges"].sum()), 2),
        "monthly_rev_at_risk_in_target": round(
            float(top.loc[top[config.TARGET] == 1, "MonthlyCharges"].sum()), 2
        ),
        "hit_rate": round(float(top[config.TARGET].mean()), 4),
        "baseline_hit_rate": round(float(df[config.TARGET].mean()), 4),
    }
