"""First experiment (Member 2 - Models).

1. Trip-grouped 5-fold cross-validation on the 80 training trips.
2. Baseline A: nearest candidate (no learning) - rank by distance only.
3. Baseline B: Logistic Regression on the candidate features.
4. Both are refit on all training trips and scored on the 10 student-testing trips.

Requires:
    python -m src.data.build_labels     --split train / student_test
    python -m src.features.build_features --split train / student_test

Usage:
    python -m src.models.baseline_experiment

Outputs (results/models/):
    baseline_cv_folds.csv     per-fold metrics
    baseline_summary.csv      CV mean/std + student-test metrics + timings
    baseline_config.json      seeds, features, CV setup
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.io import PROCESSED_DIR, ROOT
from src.features.candidates import FEATURES
from src.models.evaluate import evaluate

SEED = 42
N_FOLDS = 5
OUT = ROOT / "results" / "models"


def load(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = PROCESSED_DIR / "features"
    return (pd.read_parquet(d / f"{split}_candidates.parquet"),
            pd.read_parquet(d / f"{split}_observations.parquet"))


def nearest_scores(c: pd.DataFrame) -> np.ndarray:
    return -c.dist_m.to_numpy()


def make_logreg():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, C=1.0, random_state=SEED))


def run_model(name, fit_predict, proba, tr_c, tr_o, te_c, te_o, folds):
    rows = []
    for k, (tr_trips, va_trips) in enumerate(folds):
        c_tr, c_va = tr_c[tr_c.trip_id.isin(tr_trips)], tr_c[tr_c.trip_id.isin(va_trips)].copy()
        o_va = tr_o[tr_o.trip_id.isin(va_trips)]
        t0 = time.perf_counter()
        c_va["score"] = fit_predict(c_tr, c_va)
        m = evaluate(c_va, o_va, "score", proba=proba)
        m.update(model=name, fold=k, seconds=time.perf_counter() - t0)
        rows.append(m)
        print(f"  {name:<16} fold {k}: top1={m['top1_accuracy']:.4f}  macroF1={m['macro_f1']:.4f}")

    t0 = time.perf_counter()
    te = te_c.copy()
    te["score"] = fit_predict(tr_c, te)
    t_total = time.perf_counter() - t0
    test = evaluate(te, te_o, "score", proba=proba)
    return rows, test, t_total


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tr_c, tr_o = load("train")
    te_c, te_o = load("student_test")
    print(f"train: {tr_o.trip_id.nunique()} trips / {len(tr_o):,} obs / {len(tr_c):,} candidate rows")
    print(f"student_test: {te_o.trip_id.nunique()} trips / {len(te_o):,} obs / {len(te_c):,} candidate rows")

    trips = np.array(sorted(tr_o.trip_id.unique()))
    gkf = GroupKFold(n_splits=N_FOLDS)
    folds = [(trips[a], trips[b]) for a, b in gkf.split(trips, groups=trips)]
    for k, (_, va) in enumerate(folds):
        assert not set(va) & set(folds[k][0]), "trip leakage between folds"

    def fit_nearest(c_tr, c_ev):
        return nearest_scores(c_ev)

    def fit_logreg(c_tr, c_ev):
        model = make_logreg().fit(c_tr[FEATURES].to_numpy(), c_tr.label.to_numpy())
        t0 = time.perf_counter()
        p = model.predict_proba(c_ev[FEATURES].to_numpy())[:, 1]
        fit_logreg.infer_s = time.perf_counter() - t0
        return p

    all_rows, summary = [], []
    for name, fn, proba in [("nearest_candidate", fit_nearest, False),
                            ("logistic_regression", fit_logreg, True)]:
        print(f"\n== {name} ==")
        rows, test, total_s = run_model(name, fn, proba, tr_c, tr_o, te_c, te_o, folds)
        all_rows += rows
        cv = pd.DataFrame(rows)
        summary.append({
            "model": name,
            "cv_top1_mean": cv.top1_accuracy.mean(), "cv_top1_std": cv.top1_accuracy.std(),
            "cv_macro_f1_mean": cv.macro_f1.mean(), "cv_macro_f1_std": cv.macro_f1.std(),
            "cv_log_loss_mean": cv.log_loss.mean(),
            "test_top1": test["top1_accuracy"], "test_top1_unambiguous": test["top1_accuracy_unambiguous"],
            "test_macro_f1": test["macro_f1"], "test_precision": test["precision"],
            "test_recall": test["recall"], "test_log_loss": test["log_loss"],
            "test_route_continuity": test["route_continuity"],
            "test_candidate_miss_rate": test["candidate_miss_rate"],
            "fit_plus_infer_s": total_s,
            "test_infer_s": getattr(fn, "infer_s", 0.0),
        })

    pd.DataFrame(all_rows).to_csv(OUT / "baseline_cv_folds.csv", index=False)
    summ = pd.DataFrame(summary)
    summ.to_csv(OUT / "baseline_summary.csv", index=False)
    (OUT / "baseline_config.json").write_text(json.dumps({
        "seed": SEED, "cv": f"GroupKFold(n_splits={N_FOLDS}) grouped by trip_id",
        "folds": [list(v) for _, v in folds], "features": FEATURES,
        "logistic_regression": {"C": 1.0, "max_iter": 2000, "scaler": "StandardScaler"},
    }, indent=2))

    pd.set_option("display.width", 200)
    print("\n== Summary ==")
    print(summ[["model", "cv_top1_mean", "cv_macro_f1_mean", "test_top1", "test_macro_f1",
                "test_log_loss", "test_route_continuity", "test_candidate_miss_rate"]].round(4).to_string(index=False))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
