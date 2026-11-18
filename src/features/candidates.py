"""Candidate generation and per-candidate features (shared contract, Members 1 & 2).

For every GPS observation we take up to MAX_CANDIDATES arcs within RADIUS_M
metres (the nearest arc is always kept, however far, so no observation is
left without candidates). Each (observation, candidate arc) pair is one row.

The features use only GPS data and road geometry. The reference route is never
used here; it only produces the `label` column, which is added afterwards.

See docs/feature_label_contract.md for the meaning of each column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shapely

from src.data.io import Trip

RADIUS_M = 50.0
MAX_CANDIDATES = 8
MIN_MOVE_M = 2.0     # heading is unreliable below this displacement

FEATURES = [
    "dist_m", "dist_rank", "dist_minus_min", "dist_ratio_min", "n_candidates",
    "heading_diff", "heading_cos", "heading_valid", "speed_mps",
    "arc_len_m", "proj_frac", "is_twoway",
    "dist_prev_m", "dist_next_m", "dist_ctx_mean_m", "dt_prev_s",
]


def _bearing(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def gps_kinematics(trip: Trip) -> pd.DataFrame:
    """Heading (central difference), speed and time gap for every observation."""
    xy = trip.track[["x", "y"]].to_numpy()
    t = trip.track["t"].to_numpy(dtype=float)
    n = len(xy)
    i_prev = np.clip(np.arange(n) - 1, 0, n - 1)
    i_next = np.clip(np.arange(n) + 1, 0, n - 1)
    d = xy[i_next] - xy[i_prev]
    move = np.hypot(d[:, 0], d[:, 1])
    dt_span = np.maximum(t[i_next] - t[i_prev], 1e-6)
    dt_prev = np.r_[0.0, np.diff(t)]
    return pd.DataFrame({
        "gps_heading": _bearing(d[:, 0], d[:, 1]),
        "heading_valid": (move >= MIN_MOVE_M).astype(np.int8),
        "speed_mps": move / dt_span,
        "dt_prev_s": dt_prev,
    })


def generate_candidates(trip: Trip) -> pd.DataFrame:
    xy = trip.track[["x", "y"]].to_numpy()
    pts = shapely.points(xy)
    n = len(pts)

    q_pt, q_arc = trip.tree.query(pts, predicate="dwithin", distance=RADIUS_M)
    # Guarantee at least one candidate per observation (nearest arc, any distance).
    missing = np.setdiff1d(np.arange(n), q_pt)
    if len(missing):
        m_pt, m_arc = trip.tree.query_nearest(pts[missing], all_matches=False)
        q_pt = np.r_[q_pt, missing[m_pt]]
        q_arc = np.r_[q_arc, m_arc]

    lines = trip.lines[q_arc]
    df = pd.DataFrame({"obs_idx": q_pt, "arc_id": q_arc,
                       "dist_m": shapely.distance(pts[q_pt], lines)})
    df = df.sort_values(["obs_idx", "dist_m", "arc_id"], kind="stable")
    df["dist_rank"] = df.groupby("obs_idx").cumcount()
    df = df[df.dist_rank < MAX_CANDIDATES].reset_index(drop=True)

    q_pt, q_arc = df.obs_idx.to_numpy(), df.arc_id.to_numpy()
    lines = trip.lines[q_arc]
    g = df.groupby("obs_idx")["dist_m"]
    dmin = g.transform("min").to_numpy()
    df["dist_minus_min"] = df.dist_m - dmin
    df["dist_ratio_min"] = (dmin + 1.0) / (df.dist_m + 1.0)
    df["n_candidates"] = g.transform("size")

    # Direction agreement between GPS movement and the directed arc.
    kin = gps_kinematics(trip)
    a = trip.arcs[q_arc]
    vec = trip.nodes_xy[a[:, 1]] - trip.nodes_xy[a[:, 0]]
    arc_bearing = _bearing(vec[:, 0], vec[:, 1])
    diff = np.abs(arc_bearing - kin.gps_heading.to_numpy()[q_pt]) % 360.0
    diff = np.minimum(diff, 360.0 - diff)
    valid = kin.heading_valid.to_numpy()[q_pt]
    df["heading_diff"] = np.where(valid == 1, diff, 90.0)          # neutral value when unknown
    df["heading_cos"] = np.where(valid == 1, np.cos(np.radians(diff)), 0.0)
    df["heading_valid"] = valid
    df["speed_mps"] = kin.speed_mps.to_numpy()[q_pt]
    df["dt_prev_s"] = kin.dt_prev_s.to_numpy()[q_pt]

    df["arc_len_m"] = shapely.length(lines)
    df["proj_frac"] = shapely.line_locate_point(lines, pts[q_pt], normalized=True)
    df["is_twoway"] = (trip.twin[q_arc] >= 0).astype(np.int8)
    df["arc_u"], df["arc_v"] = a[:, 0], a[:, 1]      # endpoints: used for continuity metrics, not as features

    # Temporal context: how close the previous / next observations are to the same arc.
    prev_i = np.clip(q_pt - 1, 0, n - 1)
    next_i = np.clip(q_pt + 1, 0, n - 1)
    df["dist_prev_m"] = shapely.distance(pts[prev_i], lines)
    df["dist_next_m"] = shapely.distance(pts[next_i], lines)
    df["dist_ctx_mean_m"] = (df.dist_prev_m + df.dist_m + df.dist_next_m) / 3.0

    df.insert(0, "trip_id", trip.trip_id)
    return df
