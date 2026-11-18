"""Train and compare five candidate classifiers (model experimentation stage).

Models (five algorithm families):
    1. logistic_regression   linear / probabilistic
    2. knn                   instance-based (distance-weighted)
    3. random_forest         bagging ensemble
    4. lightgbm              gradient boosting
    5. mlp                   neural network (epoch-wise loss history)

Protocol (identical for every model):
    * 5-fold GroupKFold on the 80 training trips, grouped by trip_id
    * refit on all 80 training trips, evaluate once on the 10 student-testing trips
    * same features, same seed, same metrics (src.models.evaluate)

Requires features:  python -m src.features.build_features --split train / student_test

Usage:
    python -m src.models.compare_models                 # all five
    python -m src.models.compare_models --models lightgbm mlp

Outputs:
    results/models/compare_cv_folds.csv        per-fold metrics
    results/models/model_comparison.csv        report table (CV + student-test + timings)
    results/models/compare_config.json         hyperparameters, seed, folds, features
    results/models/plots/*.png                 comparison plots, confusion matrices, loss curves
    artifacts/models/<model>.joblib            models refit on all training trips
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.io import PROCESSED_DIR, ROOT
from src.features.candidates import FEATURES
from src.models.evaluate import evaluate

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 42
N_FOLDS = 5
KNN_MAX_TRAIN = 150_000      # KNN stores the training set; cap it to keep inference practical
MLP_EPOCHS = 60
MLP_PATIENCE = 6
OUT = ROOT / "results" / "models"
PLOTS = OUT / "plots"
ART = ROOT / "artifacts" / "models"

CONFIGS = {
    "logistic_regression": {"family": "Linear", "C": 1.0, "max_iter": 2000},
    "knn": {"family": "Instance-based", "n_neighbors": 25, "weights": "distance",
            "max_train_rows": KNN_MAX_TRAIN},
    "random_forest": {"family": "Bagging ensemble", "n_estimators": 300, "max_depth": None,
                      "min_samples_leaf": 5, "max_features": "sqrt"},
    "lightgbm": {"family": "Gradient boosting", "n_estimators": 600, "learning_rate": 0.05,
                 "num_leaves": 63, "min_child_samples": 40, "subsample": 0.8,
                 "subsample_freq": 1, "colsample_bytree": 0.8},
    "mlp": {"family": "Neural network", "hidden_layer_sizes": [64, 32], "alpha": 1e-4,
            "learning_rate_init": 1e-3, "batch_size": 1024, "epochs": MLP_EPOCHS,
            "patience": MLP_PATIENCE},
}


# ---------------------------------------------------------------- data
def load(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = PROCESSED_DIR / "features"
    return (pd.read_parquet(d / f"{split}_candidates.parquet"),
            pd.read_parquet(d / f"{split}_observations.parquet"))


def xy(c: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return c[FEATURES].to_numpy(dtype=np.float32), c.label.to_numpy()


def split_trips(trips: np.ndarray, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = rng.permutation(trips)
    n_val = max(1, int(round(len(t) * frac)))
    return t[n_val:], t[:n_val]


# ---------------------------------------------------------------- models
class MLPWithHistory:
    """sklearn MLP trained epoch by epoch, with trip-grouped validation loss and early stopping."""

    def __init__(self, cfg: dict, seed: int):
        self.cfg, self.seed = cfg, seed
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    def fit(self, c: pd.DataFrame):
        tr_t, va_t = split_trips(np.array(sorted(c.trip_id.unique())), 0.1, self.seed)
        ctr, cva = c[c.trip_id.isin(tr_t)], c[c.trip_id.isin(va_t)]
        Xtr, ytr = xy(ctr)
        Xva, yva = xy(cva)
        self.scaler = StandardScaler().fit(Xtr)
        Xtr, Xva = self.scaler.transform(Xtr), self.scaler.transform(Xva)
        net = MLPClassifier(hidden_layer_sizes=tuple(self.cfg["hidden_layer_sizes"]),
                            alpha=self.cfg["alpha"], learning_rate_init=self.cfg["learning_rate_init"],
                            batch_size=self.cfg["batch_size"], random_state=self.seed)
        rng = np.random.default_rng(self.seed)
        mon = rng.choice(len(Xtr), size=min(len(Xtr), 100_000), replace=False)  # train-loss monitor sample
        best, best_state, wait = np.inf, None, 0
        for _ in range(self.cfg["epochs"]):
            idx = rng.permutation(len(Xtr))
            for s in range(0, len(idx), self.cfg["batch_size"]):
                b = idx[s:s + self.cfg["batch_size"]]
                net.partial_fit(Xtr[b], ytr[b], classes=[0, 1])
            tl = log_loss(ytr[mon], net.predict_proba(Xtr[mon])[:, 1], labels=[0, 1])
            vl = log_loss(yva, net.predict_proba(Xva)[:, 1], labels=[0, 1])
            self.history["train_loss"].append(tl)
            self.history["val_loss"].append(vl)
            if vl < best - 1e-5:
                best, wait = vl, 0
                best_state = ([w.copy() for w in net.coefs_], [b.copy() for b in net.intercepts_])
            else:
                wait += 1
                if wait >= self.cfg["patience"]:
                    break
        net.coefs_, net.intercepts_ = best_state
        self.net = net
        self.best_epoch = int(np.argmin(self.history["val_loss"])) + 1
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.net.predict_proba(self.scaler.transform(X))


class LGBMWithHistory:
    """LightGBM with a trip-grouped validation set to record boosting-round loss curves."""

    def __init__(self, cfg: dict, seed: int):
        self.cfg, self.seed = cfg, seed

    def fit(self, c: pd.DataFrame):
        tr_t, va_t = split_trips(np.array(sorted(c.trip_id.unique())), 0.1, self.seed)
        ctr, cva = c[c.trip_id.isin(tr_t)], c[c.trip_id.isin(va_t)]
        params = {k: v for k, v in self.cfg.items() if k != "family"}
        self.model = lgb.LGBMClassifier(**params, random_state=self.seed, n_jobs=-1, verbose=-1)
        self.model.fit(*xy(ctr), eval_set=[xy(ctr), xy(cva)], eval_names=["train", "val"],
                       eval_metric="binary_logloss")
        ev = self.model.evals_result_
        self.history = {"train_loss": ev["train"]["binary_logloss"], "val_loss": ev["val"]["binary_logloss"]}
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class SklearnWrapper:
    def __init__(self, est, max_rows: int | None = None, seed: int = SEED):
        self.est, self.max_rows, self.seed = est, max_rows, seed

    def fit(self, c: pd.DataFrame):
        if self.max_rows and len(c) > self.max_rows:
            # Stratified row sample: keep the positive/negative balance of the full set.
            c = c.groupby("label", group_keys=False).sample(frac=self.max_rows / len(c), random_state=self.seed)
        self.est.fit(*xy(c))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.est.predict_proba(X)


def build(name: str):
    cfg = CONFIGS[name]
    if name == "logistic_regression":
        return SklearnWrapper(make_pipeline(StandardScaler(), LogisticRegression(
            C=cfg["C"], max_iter=cfg["max_iter"], random_state=SEED)))
    if name == "knn":
        return SklearnWrapper(make_pipeline(StandardScaler(), KNeighborsClassifier(
            n_neighbors=cfg["n_neighbors"], weights=cfg["weights"], n_jobs=-1)), max_rows=cfg["max_train_rows"])
    if name == "random_forest":
        return SklearnWrapper(RandomForestClassifier(
            n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
            min_samples_leaf=cfg["min_samples_leaf"], max_features=cfg["max_features"],
            n_jobs=-1, random_state=SEED))
    if name == "lightgbm":
        return LGBMWithHistory(cfg, SEED)
    if name == "mlp":
        return MLPWithHistory(cfg, SEED)
    raise ValueError(name)


# ---------------------------------------------------------------- experiment
def score(model, c: pd.DataFrame) -> tuple[np.ndarray, float]:
    X, _ = xy(c)
    t0 = time.perf_counter()
    p = model.predict_proba(X)[:, 1]
    return p, time.perf_counter() - t0


def run(name: str, tr_c, tr_o, te_c, te_o, folds) -> tuple[list[dict], dict, object, pd.DataFrame]:
    rows = []
    for k, (tr_trips, va_trips) in enumerate(folds):
        c_tr = tr_c[tr_c.trip_id.isin(tr_trips)]
        c_va = tr_c[tr_c.trip_id.isin(va_trips)].copy()
        t0 = time.perf_counter()
        model = build(name).fit(c_tr)
        fit_s = time.perf_counter() - t0
        c_va["score"], inf_s = score(model, c_va)
        m = evaluate(c_va, tr_o[tr_o.trip_id.isin(va_trips)], "score", proba=True)
        m.update(model=name, fold=k, fit_s=fit_s, infer_s=inf_s)
        rows.append(m)
        print(f"  fold {k}: top1={m['top1_accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
              f"logloss={m['log_loss']:.4f} fit={fit_s:.1f}s")

    t0 = time.perf_counter()
    final = build(name).fit(tr_c)
    fit_s = time.perf_counter() - t0
    te = te_c.copy()
    te["score"], inf_s = score(final, te)
    test = evaluate(te, te_o, "score", proba=True)
    test.update(fit_s=fit_s, infer_s=inf_s, infer_ms_per_obs=1000 * inf_s / len(te_o))
    tr_eval = tr_c.copy()
    tr_eval["score"], _ = score(final, tr_eval)
    test["train_top1"] = evaluate(tr_eval, tr_o, "score", proba=True)["top1_accuracy"]
    print(f"  student_test: top1={test['top1_accuracy']:.4f} macroF1={test['macro_f1']:.4f} "
          f"logloss={test['log_loss']:.4f} (train top1 {test['train_top1']:.4f})")
    return rows, test, final, te


# ---------------------------------------------------------------- plots
def bar(df, cols, labels, title, fname, ylabel, err=None):
    x = np.arange(len(df))
    w = 0.8 / len(cols)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (c, lab) in enumerate(zip(cols, labels)):
        yerr = df[err[i]] if err else None
        b = ax.bar(x + (i - (len(cols) - 1) / 2) * w, df[c], w, yerr=yerr, capsize=3, label=lab)
        ax.bar_label(b, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x, df.model, rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    if len(cols) > 1:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PLOTS / fname, dpi=150)
    plt.close(fig)


def plots(summary: pd.DataFrame, tests: dict[str, pd.DataFrame], finals: dict) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    bar(summary, ["cv_macro_f1_mean"], ["CV macro-F1"], "Cross-validation macro-F1 (5 trip-grouped folds, mean ± std)",
        "cv_macro_f1.png", "macro-F1", err=["cv_macro_f1_std"])
    bar(summary, ["test_top1", "test_macro_f1"], ["Top-1 accuracy", "macro-F1"],
        "Student-testing performance", "student_test_top1_f1.png", "score")
    bar(summary, ["cv_log_loss_mean", "test_log_loss"], ["CV", "Student-test"], "Log loss (lower is better)",
        "log_loss.png", "log loss")
    bar(summary, ["train_top1", "cv_top1_mean", "test_top1"], ["Train (refit)", "CV", "Student-test"],
        "Top-1 accuracy: train vs CV vs test (overfitting check)", "top1_train_cv_test.png", "Top-1 accuracy")
    bar(summary, ["infer_ms_per_obs"], ["inference"], "Inference time per observation (student-test)",
        "inference_time.png", "ms / observation")

    for name, te in tests.items():
        cm = confusion_matrix(te.label, (te.score >= 0.5).astype(int), labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4, 3.6))
        ConfusionMatrixDisplay(cm, display_labels=["not ref", "ref arc"]).plot(ax=ax, colorbar=False, values_format="d")
        ax.set_title(f"{name} (student-test candidates)")
        fig.tight_layout()
        fig.savefig(PLOTS / f"confusion_{name}.png", dpi=150)
        plt.close(fig)

    for name, model in finals.items():
        h = getattr(model, "history", None)
        if not h:
            continue                                  # no epoch history: no curve (never fabricated)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(1, len(h["train_loss"]) + 1), h["train_loss"], label="train")
        ax.plot(range(1, len(h["val_loss"]) + 1), h["val_loss"], label="validation (held-out trips)")
        ax.set_xlabel("epoch" if name == "mlp" else "boosting round")
        ax.set_ylabel("log loss")
        ax.set_title(f"{name}: training history")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(PLOTS / f"loss_curve_{name}.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(CONFIGS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    tr_c, tr_o = load("train")
    te_c, te_o = load("student_test")
    trips = np.array(sorted(tr_o.trip_id.unique()))
    folds = [(trips[a], trips[b]) for a, b in GroupKFold(n_splits=N_FOLDS).split(trips, groups=trips)]
    print(f"train {len(trips)} trips / {len(tr_c):,} rows | student_test {te_o.trip_id.nunique()} trips / {len(te_c):,} rows")

    fold_rows, summary, tests, finals = [], [], {}, {}
    for name in args.models:
        print(f"\n== {name} ==")
        rows, test, final, te = run(name, tr_c, tr_o, te_c, te_o, folds)
        fold_rows += rows
        tests[name], finals[name] = te, final
        joblib.dump(final, ART / f"{name}.joblib")
        cv = pd.DataFrame(rows)
        summary.append({
            "model": name, "family": CONFIGS[name]["family"],
            "key_config": json.dumps({k: v for k, v in CONFIGS[name].items() if k != "family"}),
            "cv_macro_f1_mean": cv.macro_f1.mean(), "cv_macro_f1_std": cv.macro_f1.std(),
            "cv_top1_mean": cv.top1_accuracy.mean(), "cv_top1_std": cv.top1_accuracy.std(),
            "cv_log_loss_mean": cv.log_loss.mean(),
            "train_top1": test["train_top1"],
            "test_top1": test["top1_accuracy"], "test_top1_unambiguous": test["top1_accuracy_unambiguous"],
            "test_macro_f1": test["macro_f1"], "test_precision": test["precision"], "test_recall": test["recall"],
            "test_log_loss": test["log_loss"], "test_route_continuity": test["route_continuity"],
            "test_candidate_miss_rate": test["candidate_miss_rate"],
            "fit_s_full": test["fit_s"], "infer_s_test": test["infer_s"], "infer_ms_per_obs": test["infer_ms_per_obs"],
            "best_epoch": getattr(final, "best_epoch", None),
        })

    summ = pd.DataFrame(summary)
    pd.DataFrame(fold_rows).to_csv(OUT / "compare_cv_folds.csv", index=False)
    summ.to_csv(OUT / "model_comparison.csv", index=False)
    (OUT / "compare_config.json").write_text(json.dumps(
        {"seed": SEED, "cv": f"GroupKFold({N_FOLDS}) by trip_id", "folds": [list(v) for _, v in folds],
         "features": FEATURES, "models": CONFIGS}, indent=2))
    plots(summ, tests, finals)

    pd.set_option("display.width", 220)
    print("\n== Model comparison ==")
    print(summ[["model", "cv_macro_f1_mean", "cv_top1_mean", "test_top1", "test_macro_f1",
                "test_log_loss", "test_route_continuity", "infer_ms_per_obs"]].round(4).to_string(index=False))
    best = summ.sort_values(["cv_macro_f1_mean", "cv_log_loss_mean"], ascending=[False, True]).iloc[0]
    print(f"\nHighest CV macro-F1: {best.model}  (selection uses CV only; student-test is for comparison)")
    print(f"Saved results to {OUT} and models to {ART}")


if __name__ == "__main__":
    main()
