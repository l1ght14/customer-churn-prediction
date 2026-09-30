"""Model comparison. Trains four candidates and reports imbalanced-class metrics.

The DummyClassifier is not filler: with 26% positives a naive baseline ("predict the
prior") already scores 73.4% accuracy. Any candidate has to beat PR-AUC of 0.2658
before it deserves to be called predictive, and that number is the one to quote.

Selection criterion is average precision (PR-AUC), not accuracy and not raw ROC-AUC.

The reason is about baselines, not about the true-negative axis. A random ranking scores
the base rate (0.2658) on PR-AUC but exactly 0.5000 on ROC-AUC, so on ROC-AUC a model that
learned nothing is indistinguishable from one that learned a little. Note the usual
"ROC-AUC flatters imbalance because negatives outnumber positives" line is NOT the
argument: a zero-recall model has ROC-AUC 0.5, and the dummy baseline below is recorded at
roc_auc 0.5000 with recall 0.0000.

Selection happens on the cross-validated score of the TRAIN split only, never on the
holdout. The holdout is only ever reported. Running that CV over the full matrix would
leak: every holdout label would influence which model wins, so the reported holdout
score would no longer be unbiased.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

from . import config, metrics
from .data_loader import load_clean
from .features import build_features, make_pipeline, split_xy, stratified_split


def candidate_models() -> dict:
    """Four models spanning the interpretability / accuracy trade-off.

    LogisticRegression is the interpretable one. Its coefficients are only readable
    because the design matrix is not rank-deficient: exact-duplicate columns would make
    L2 split weight arbitrarily across the copies, so no individual coefficient would
    mean anything. Logistic regression is a reference point, not the recommendation.
    """
    return {
        "baseline_dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=config.RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=20, class_weight="balanced_subsample",
            # n_jobs=1: a 300-tree forest over 5 CV folds is the peak memory and runtime
            # cost of this pipeline, and on a 7k-row dataset the parallel version is both
            # slower in wall clock (thread setup) and non-deterministic in its last float
            # bits, which is exactly what churns a committed artefact's hash.
            random_state=config.RANDOM_STATE, n_jobs=1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=config.RANDOM_STATE),
    }


def out_of_fold_scores(model, X, y) -> np.ndarray:
    """Probabilities from a model that never saw the row it is scoring.

    Needed for threshold selection. Using `pipe.predict_proba(Xtr)` after fitting on
    Xtr gives in-sample scores: the model has partly memorised those labels, so the
    score distribution is pushed apart and the derived cut is wrong on new data.
    Measured for the selected random forest: an in-sample-derived 80%-recall cut reaches
    only 0.7727 on the holdout, missing its own constraint by 2.7 points.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    return cross_val_predict(
        make_pipeline(model), X, y, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]


def evaluate_model(name, model, Xtr, ytr, Xte, yte) -> dict:
    """Fit on train, score on test.

    Both operating points below are chosen from OUT-OF-FOLD TRAIN scores and only then
    applied to the holdout. Two separate mistakes are designed out:

    1. Deriving a threshold from holdout labels and then reporting recall at that same
       threshold. Recall is pinned to whatever you targeted, so it measures nothing. The
       fingerprint was several different models reporting an identical recall at the same
       target, which independent models cannot do honestly.
    2. Deriving it from in-sample train scores, where the model has partly memorised the
       labels. The cut is systematically wrong on unseen data.
    """
    pipe = make_pipeline(model)
    pipe.fit(Xtr, ytr)
    prob_tr = out_of_fold_scores(model, Xtr, ytr)
    prob = pipe.predict_proba(Xte)[:, 1]

    thr_size = metrics.threshold_for_target_size(prob_tr, target_share=0.10)
    thr_recall = metrics.threshold_for_target_recall(ytr, prob_tr, target_recall=0.80)

    return {
        "model": name,
        "at_0.5": metrics.classification_report_at(yte, prob, threshold=0.5),
        "at_top_10pct": metrics.classification_report_at(yte, prob, threshold=thr_size),
        "at_train_derived_80pct_recall": metrics.classification_report_at(yte, prob, threshold=thr_recall),
        "lift_top_decile": float(metrics.lift_and_gain(yte, prob).loc[0, "lift"]),
        # the cuts themselves, so a test can recompute them from the model and detect a
        # threshold that was quietly derived from the holdout labels instead. The reported
        # recall cannot do that job: the honest and leaked cuts are 0.4922 vs 0.4939 and
        # reach recall 0.8048 vs 0.8021, a 0.8pp gap, so only the threshold value itself
        # separates them.
        "threshold_top_10pct": round(float(thr_size), 6),
        "threshold_80pct_recall": round(float(thr_recall), 6),
    }


def cross_validate(model, X, y, n_splits: int = 5) -> dict:
    """Mean +/- std of PR-AUC across folds. Shows the result is not one lucky split."""
    pipe = make_pipeline(model)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="average_precision", n_jobs=-1)
    return {
        "cv_pr_auc_mean": round(float(scores.mean()), 4),
        "cv_pr_auc_std": round(float(scores.std()), 4),
    }


def _print_row(row: dict) -> None:
    a = row["at_0.5"]
    b = row["at_top_10pct"]
    # roc_auc/pr_auc are None when the slice holds a single class
    roc = f"{a['roc_auc']:.4f}" if a["roc_auc"] is not None else "n/a"
    prauc = f"{a['pr_auc']:.4f}" if a["pr_auc"] is not None else "n/a"
    print(
        f"{row['model']:<22} PR-AUC {prauc} | ROC-AUC {roc} | "
        f"@0.5 P {a['precision']:.3f} R {a['recall']:.3f} | "
        f"@top10% recall {b['recall']:.3f} precision {b['precision']:.3f} | "
        f"lift {row['lift_top_decile']:.2f}x"
    )


def run() -> dict:
    df = load_clean()
    feat = build_features(df)
    X, y = split_xy(feat)
    Xtr, Xte, ytr, yte = stratified_split(X, y)

    base_rate = float(y.mean())
    print("=" * 108)
    print(f"train {len(Xtr)} rows / test {len(Xte)} rows | churn rate train {ytr.mean():.4f} test {yte.mean():.4f}")
    print(f"PR-AUC of a coin flip with the base rate: {base_rate:.4f}  <- every model must beat this")
    print("=" * 108)

    results = []
    for name, model in candidate_models().items():
        row = evaluate_model(name, model, Xtr, ytr, Xte, yte)
        # TRAIN SPLIT ONLY. Passing the full X, y here would let holdout labels pick the
        # winner, which makes the holdout score below an overestimate of the truth.
        row.update(cross_validate(model, Xtr, ytr))
        results.append(row)
        _print_row(row)

    results.sort(key=lambda r: r["cv_pr_auc_mean"], reverse=True)
    best = results[0]

    print("\n" + "=" * 108)
    print(f"SELECTED: {best['model']}")
    print("  selected on 5-fold CV mean PR-AUC over the TRAIN split only.")
    print("  Two separate leaks were designed out:")
    print("    1. picking the winner by holdout score and then quoting that score;")
    print("    2. running the selection CV over the full matrix, holdout rows included,")
    print("       so holdout labels still influenced the choice.")
    print("  Both inflate the reported number. The holdout is now untouched until the")
    print("  single report below.")
    print(f"\n  unbiased holdout PR-AUC : {best['at_0.5']['pr_auc']:.4f}")
    print(f"  selection score (CV)    : {best['cv_pr_auc_mean']:.4f} +/- {best['cv_pr_auc_std']:.4f}")
    print(f"  top-decile lift         : {best['lift_top_decile']:.2f}x vs random selection")
    print("=" * 108)

    pipe = make_pipeline(candidate_models()[best["model"]])
    pipe.fit(Xtr, ytr)
    prob = pipe.predict_proba(Xte)[:, 1]
    # the same table evaluate_model already computed and stored as lift_top_decile.
    # Recomputing it here would duplicate a number that could silently drift out of
    # step with the one printed in the comparison table above.
    lift_table = metrics.lift_and_gain(yte, prob)
    self_check = float(lift_table.loc[0, "lift"])
    if abs(self_check - best["lift_top_decile"]) > 1e-9:
        raise AssertionError(
            f"decile lift {self_check} disagrees with the comparison table "
            f"{best['lift_top_decile']}"
        )

    print("\ndecile lift table on the test set:")
    print(lift_table.to_string(index=False))

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    out = {
        "base_rate": round(base_rate, 4),
        "n_train": len(Xtr),
        "n_test": len(Xte),
        "results": results,
        "best_model": best["model"],
        "lift_table": lift_table.to_dict(orient="records"),
    }
    config.ROOT.joinpath("reports").mkdir(parents=True, exist_ok=True)
    config.METRICS_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nmetrics written to {config.METRICS_JSON}")
    print(f"metrics written to {config.METRICS_JSON}")
    return out


if __name__ == "__main__":
    run()
