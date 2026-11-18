"""Build per-observation reference-arc labels for a development split.

Usage (from the project root):
    python -m src.data.build_labels --split train
    python -m src.data.build_labels --split student_test

Outputs:
    data/processed/labels/<split>/<trip_id>.parquet
    results/eda/label_summary_<split>.csv
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
from tqdm import tqdm

from src.data.io import DEV_SPLITS, PROCESSED_DIR, ROOT, list_trips, load_trip
from src.data.labels import assign_reference_arcs, summarise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=DEV_SPLITS)
    ap.add_argument("--trips", nargs="*", help="optional subset of trip ids")
    args = ap.parse_args()

    out_dir = PROCESSED_DIR / "labels" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    trips = args.trips or list_trips(args.split)

    rows, t0 = [], time.perf_counter()
    for tid in tqdm(trips, desc=f"labels[{args.split}]"):
        trip = load_trip(args.split, tid)
        lab = assign_reference_arcs(trip)
        lab.to_parquet(out_dir / f"{tid}.parquet", index=False)
        rows.append(summarise(lab, trip))

    summary = pd.DataFrame(rows)
    res = ROOT / "results" / "eda"
    res.mkdir(parents=True, exist_ok=True)
    summary.to_csv(res / f"label_summary_{args.split}.csv", index=False)

    n = summary.n_obs.sum()
    w = lambda c: (summary[c] * summary.n_obs).sum() / n  # noqa: E731  observation-weighted mean
    print(f"\n{args.split}: {len(summary)} trips, {n:,} observations, {time.perf_counter() - t0:.1f} s")
    print(f"  median distance to reference arc : {summary.ref_dist_median_m.median():.2f} m")
    print(f"  far (> 30 m)                     : {w('pct_far'):.2f} %")
    print(f"  competing arc within +3 m        : {w('pct_competing'):.2f} %")
    print(f"  near arc end (< 10 m)            : {w('pct_near_node'):.2f} %")
    print(f"  ambiguous (far or competing)     : {w('pct_ambiguous'):.2f} %")
    print(f"  route arcs with no observation   : {summary.route_arcs_skipped.sum()} of {summary.route_arcs.sum()}")
    print(f"Saved labels to {out_dir}")


if __name__ == "__main__":
    main()
