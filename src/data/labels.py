"""Assign a reference arc to every GPS observation (Member 1 - Data).

Problem
-------
`.route` lists the arcs the vehicle travelled, in order, but it does not say
which observation belongs to which arc (e.g. 1240 observations vs 149 arcs).

Method: monotone alignment by dynamic programming
-------------------------------------------------
Let d[i, k] be the distance in metres from observation i to route arc k.
We choose route positions k_0 <= k_1 <= ... <= k_{N-1} (the vehicle never moves
backwards along its own route) that minimise  sum_i d[i, k_i].

    D[0, k] = d[0, k]
    D[i, k] = d[i, k] + min_{j <= k} D[i-1, j]

The inner minimum is a running (prefix) minimum, so each step is O(K) and the
whole alignment is O(N*K). Backtracking gives the route position, and therefore
the directed arc id, of every observation.

Why not simply the nearest route arc? A route can pass the same road twice, in
opposite directions, or along a reverse-twin arc with identical geometry. The
monotone constraint uses the order of the route to choose the right one.

Ambiguity flags (stored per observation, never used as model input)
------------------------------------------------------------------
* flag_far        assigned arc is more than FAR_M metres away (GPS error / route gap)
* flag_competing  some other arc (not the assigned arc or its reverse twin) lies
                  within COMPETE_M metres of the assigned distance (parallel roads,
                  junctions) - the label is plausible but not certain
* flag_near_node  observation is within NODE_M metres of an end of the assigned
                  arc, where the boundary between consecutive arcs is uncertain
* ambiguous       flag_far OR flag_competing
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shapely

from src.data.io import Trip

FAR_M = 30.0
COMPETE_M = 3.0
NODE_M = 10.0
SEARCH_M = 60.0


def align_to_route(d: np.ndarray) -> np.ndarray:
    """Monotone DP alignment. d: (N, K) distances. Returns route position per observation."""
    n, k = d.shape
    idx = np.arange(k)
    back = np.empty((n, k), dtype=np.int32)
    prev = d[0].astype(np.float64)
    back[0] = idx
    for i in range(1, n):
        pmin = np.minimum.accumulate(prev)
        shifted = np.concatenate(([np.inf], pmin[:-1]))
        new_min = prev <= shifted                      # positions that set a new running minimum
        argmin = np.maximum.accumulate(np.where(new_min, idx, 0))
        back[i] = argmin
        prev = d[i] + pmin
    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    return pos


def assign_reference_arcs(trip: Trip) -> pd.DataFrame:
    if trip.route is None or len(trip.route) == 0:
        raise ValueError(f"Trip {trip.trip_id} has no reference route")

    tr = trip.track
    pts = shapely.points(tr[["x", "y"]].to_numpy())
    route_lines = trip.lines[trip.route]

    d = shapely.distance(pts[:, None], route_lines[None, :]).astype(np.float32)
    pos = align_to_route(d)
    ref_arc = trip.route[pos]
    ref_dist = d[np.arange(len(pts)), pos].astype(float)

    # Snapped position on the assigned arc (used for maps and for the near-node flag).
    lines = trip.lines[ref_arc]
    frac = shapely.line_locate_point(lines, pts, normalized=True)
    snap = shapely.line_interpolate_point(lines, frac, normalized=True)
    snap_xy = shapely.get_coordinates(snap)
    arc_len = shapely.length(lines)
    dist_to_end = np.minimum(frac, 1 - frac) * arc_len
    slon, slat = trip.to_lonlat.transform(snap_xy[:, 0], snap_xy[:, 1])

    # Competing arcs: any other arc within (ref_dist + COMPETE_M), excluding the reverse twin.
    q_pt, q_arc = trip.tree.query(pts, predicate="dwithin", distance=SEARCH_M)
    pair_d = shapely.distance(pts[q_pt], trip.lines[q_arc])
    own = (q_arc == ref_arc[q_pt]) | (q_arc == trip.twin[ref_arc[q_pt]])
    competing = (~own) & (pair_d <= ref_dist[q_pt] + COMPETE_M)
    n_comp = np.bincount(q_pt[competing], minlength=len(pts))

    out = pd.DataFrame({
        "trip_id": trip.trip_id,
        "obs_idx": np.arange(len(pts)),
        "t": tr.t.to_numpy(), "lon": tr.lon.to_numpy(), "lat": tr.lat.to_numpy(),
        "x": tr.x.to_numpy(), "y": tr.y.to_numpy(),
        "ref_pos": pos, "ref_arc": ref_arc, "ref_dist_m": ref_dist,
        "ref_frac": frac, "snap_lon": slon, "snap_lat": slat,
        "n_competing": n_comp,
    })
    out["flag_far"] = out.ref_dist_m > FAR_M
    out["flag_competing"] = out.n_competing > 0
    out["flag_near_node"] = dist_to_end < NODE_M
    out["ambiguous"] = out.flag_far | out.flag_competing
    return out


def summarise(lab: pd.DataFrame, trip: Trip) -> dict:
    used = lab.ref_pos.nunique()
    return {
        "trip_id": trip.trip_id,
        "n_obs": len(lab),
        "n_dropped_rows": int(trip.track.attrs.get("n_dropped", 0)),
        "route_arcs": len(trip.route),
        "route_arcs_used": int(used),
        "route_arcs_skipped": int(len(trip.route) - used),
        "ref_dist_median_m": float(lab.ref_dist_m.median()),
        "ref_dist_p95_m": float(lab.ref_dist_m.quantile(0.95)),
        "pct_far": float(lab.flag_far.mean() * 100),
        "pct_competing": float(lab.flag_competing.mean() * 100),
        "pct_near_node": float(lab.flag_near_node.mean() * 100),
        "pct_ambiguous": float(lab.ambiguous.mean() * 100),
        "utm_epsg": trip.epsg,
    }
