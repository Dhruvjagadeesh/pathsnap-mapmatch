"""Evaluation metrics shared by every model.

Observation-level metrics are computed over ALL observations of a split, so an
observation whose reference arc is missing from its candidate set always counts
as an error (it is also reported as the candidate miss rate).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score


def pick_top1(cands: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """Highest-scoring candidate per observation (ties broken by smaller distance)."""
    order = cands.sort_values(["trip_id", "obs_idx", score_col, "dist_m"],
                              ascending=[True, True, False, True], kind="stable")
    return order.drop_duplicates(["trip_id", "obs_idx"]).reset_index(drop=True)


def route_continuity(top1: pd.DataFrame) -> float:
    """Share of consecutive observation pairs whose predicted arcs are the same arc
    or directly connected (end node of the previous arc = start node of the next)."""
    df = top1.sort_values(["trip_id", "obs_idx"])
    same_trip = df.trip_id.to_numpy()[1:] == df.trip_id.to_numpy()[:-1]
    a_prev, a_next = df.arc_id.to_numpy()[:-1], df.arc_id.to_numpy()[1:]
    v_prev, u_next = df.arc_v.to_numpy()[:-1], df.arc_u.to_numpy()[1:]
    ok = (a_prev == a_next) | (v_prev == u_next)
    return float(ok[same_trip].mean()) if same_trip.any() else float("nan")


def evaluate(cands: pd.DataFrame, obs: pd.DataFrame, score_col: str = "score",
             proba: bool = True) -> dict:
    """cands must contain trip_id, obs_idx, arc_id, arc_u, arc_v, dist_m, label and score_col."""
    top1 = pick_top1(cands, score_col)
    n_obs = len(obs)
    correct = int(top1.label.sum())
    out = {
        "top1_accuracy": correct / n_obs,
        "candidate_miss_rate": float(1 - obs.candidate_hit.mean()),
        "route_continuity": route_continuity(top1),
        "n_obs": n_obs,
    }
    clean = obs.loc[~obs.ambiguous, ["trip_id", "obs_idx"]]
    t_clean = top1.merge(clean, on=["trip_id", "obs_idx"])
    out["top1_accuracy_unambiguous"] = float(t_clean.label.sum() / max(len(clean), 1))

    y = cands.label.to_numpy()
    if proba:
        p = np.clip(cands[score_col].to_numpy(dtype=float), 1e-7, 1 - 1e-7)
        yhat = (p >= 0.5).astype(int)
        out["log_loss"] = float(log_loss(y, p, labels=[0, 1]))
    else:
        # Rule-based scorer: predict positive only for the chosen top-1 candidate.
        chosen = set(zip(top1.trip_id, top1.obs_idx, top1.arc_id))
        yhat = np.fromiter(((t, o, a) in chosen for t, o, a in
                            zip(cands.trip_id, cands.obs_idx, cands.arc_id)), dtype=int, count=len(cands))
        out["log_loss"] = float("nan")
    out["macro_f1"] = float(f1_score(y, yhat, average="macro", zero_division=0))
    out["precision"] = float(precision_score(y, yhat, zero_division=0))
    out["recall"] = float(recall_score(y, yhat, zero_division=0))
    return out
