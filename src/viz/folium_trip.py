"""Folium map of one trip: road network, GPS trace, reference route, label links.

Usage (after build_labels has run for the split):
    python -m src.viz.folium_trip --split train --trip 00000048

Output: results/maps/<split>_<trip>_reference.html
"""
from __future__ import annotations

import argparse

import folium
import numpy as np
import pandas as pd
import shapely

from src.data.io import DEV_SPLITS, PROCESSED_DIR, ROOT, load_trip, route_lonlat

NET_BUFFER_M = 300
MAX_NET_ARCS = 8000

LEGEND = """
<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:#fff;padding:10px 14px;
border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.25);font:13px Arial;line-height:1.7">
<b>Trip {tid}</b><br>
<span style="color:#9aa0a6">&#9473;&#9473;</span> Road network<br>
<span style="color:#1a9e5c">&#9473;&#9473;</span> Reference route<br>
<span style="color:#2f6fed">&#9679;</span> GPS observation<br>
<span style="color:#f39c12">&#9679;</span> Ambiguous label<br>
<span style="color:#d62728">&#9679;</span> Far (&gt;30 m) label<br>
<span style="color:#8e44ad">&#9473;</span> Observation &#8594; assigned arc
</div>"""


def build_map(split: str, trip_id: str) -> folium.Map:
    trip = load_trip(split, trip_id)
    lab = pd.read_parquet(PROCESSED_DIR / "labels" / split / f"{trip_id}.parquet")

    m = folium.Map(location=[lab.lat.mean(), lab.lon.mean()], zoom_start=14, tiles="cartodbpositron")

    # Road network near the trajectory (one line per undirected road).
    hull = shapely.buffer(shapely.multipoints(trip.track[["x", "y"]].to_numpy()).envelope, NET_BUFFER_M)
    near = trip.tree.query(hull)
    near = near[(trip.twin[near] < 0) | (near < trip.twin[near])][:MAX_NET_ARCS]
    ll = trip.nodes_ll
    segs = [[[ll[u, 1], ll[u, 0]], [ll[v, 1], ll[v, 0]]] for u, v in trip.arcs[near]]
    net = folium.FeatureGroup(name=f"Road network ({len(segs)} roads)")
    folium.PolyLine(segs, color="#9aa0a6", weight=2, opacity=0.7).add_to(net)
    net.add_to(m)

    ref = folium.FeatureGroup(name="Reference route")
    folium.PolyLine(route_lonlat(trip), color="#1a9e5c", weight=6, opacity=0.8,
                    tooltip=f"Reference route: {len(trip.route)} arcs").add_to(ref)
    ref.add_to(m)

    links = folium.FeatureGroup(name="Observation → assigned arc", show=False)
    folium.PolyLine(lab[["lat", "lon", "snap_lat", "snap_lon"]].to_numpy().reshape(-1, 2, 2).tolist(),
                    color="#8e44ad", weight=1, opacity=0.8).add_to(links)
    links.add_to(m)

    gps = folium.FeatureGroup(name=f"GPS observations ({len(lab)})")
    colour = np.where(lab.flag_far, "#d62728", np.where(lab.ambiguous, "#f39c12", "#2f6fed"))
    for r, c in zip(lab.itertuples(index=False), colour):
        folium.CircleMarker(
            [r.lat, r.lon], radius=3, color=c, fill=True, fill_opacity=0.9, weight=0,
            tooltip=f"obs {r.obs_idx} | t={r.t:.0f}s | arc {r.ref_arc} | {r.ref_dist_m:.1f} m"
                    + (" | competing" if r.flag_competing else "") + (" | far" if r.flag_far else ""),
        ).add_to(gps)
    gps.add_to(m)

    folium.Marker([lab.lat.iloc[0], lab.lon.iloc[0]], tooltip="Start",
                  icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([lab.lat.iloc[-1], lab.lon.iloc[-1]], tooltip="End",
                  icon=folium.Icon(color="red", icon="stop")).add_to(m)

    m.fit_bounds([[lab.lat.min(), lab.lon.min()], [lab.lat.max(), lab.lon.max()]])
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(LEGEND.format(tid=trip_id)))
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=DEV_SPLITS)
    ap.add_argument("--trip", required=True)
    args = ap.parse_args()
    out = ROOT / "results" / "maps" / f"{args.split}_{args.trip}_reference.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    build_map(args.split, args.trip).save(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
