"""Build the candidate feature table for a development split.

Requires labels first:  python -m src.data.build_labels --split <split>

Usage:
    python -m src.features.build_features --split train
    python -m src.features.build_features --split student_test

Outputs:
    data/processed/features/<split>_candidates.parquet   one row per (observation, candidate arc)
    data/processed/features/<split>_observations.parquet one row per observation
    results/eda/candidate_summary_<split>.csv
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
from tqdm import tqdm

from src.data.io import DEV_SPLITS, PROCESSED_DIR, ROOT, list_trips, load_trip
from src.features.candidates import MAX_CANDIDATES, RADIUS_M, generate_candidates

OBS_COLS = ["trip_id", "obs_idx", "t", "lon", "lat", "ref_arc", "ref_dist_m",
            "flag_far", "flag_competing", "flag_near_node", "ambiguous"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=DEV_SPLITS)
    args = ap.parse_args()

    lab_dir = PROCESSED_DIR / "labels" / args.split
    out_dir = PROCESSED_DIR / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    cands, obs, summ = [], [], []
    t0 = time.perf_counter()
    for tid in tqdm(list_trips(args.split), desc=f"features[{args.split}]"):
        lab_path = lab_dir / f"{tid}.parquet"
        if not lab_path.is_file():
            raise FileNotFoundError(f"{lab_path} missing - run build_labels for this split first")
        lab = pd.read_parquet(lab_path)[OBS_COLS]
        trip = load_trip(args.split, tid)

        c = generate_candidates(trip)
        c = c.merge(lab[["obs_idx", "ref_arc", "ambiguous"]], on="obs_idx", how="left")
        c["label"] = (c.arc_id == c.ref_arc).astype("int8")
        hit = c.groupby("obs_idx")["label"].max()
        lab["candidate_hit"] = lab.obs_idx.map(hit).fillna(0).astype("int8")
        c = c.drop(columns=["ref_arc"])

        cands.append(c)
        obs.append(lab)
        summ.append({"trip_id": tid, "n_obs": len(lab), "n_candidates": len(c),
                     "mean_cands_per_obs": len(c) / max(len(lab), 1),
                     "candidate_miss_rate": 1 - lab.candidate_hit.mean()})

    cand_df = pd.concat(cands, ignore_index=True)
    obs_df = pd.concat(obs, ignore_index=True)
    cand_df.to_parquet(out_dir / f"{args.split}_candidates.parquet", index=False)
    obs_df.to_parquet(out_dir / f"{args.split}_observations.parquet", index=False)

    res = ROOT / "results" / "eda"
    res.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summ).to_csv(res / f"candidate_summary_{args.split}.csv", index=False)

    print(f"\n{args.split}: {obs_df.trip_id.nunique()} trips, {len(obs_df):,} observations, "
          f"{len(cand_df):,} candidate rows ({time.perf_counter() - t0:.1f} s)")
    print(f"  radius {RADIUS_M:.0f} m, max {MAX_CANDIDATES} candidates")
    print(f"  mean candidates / observation : {len(cand_df) / len(obs_df):.2f}")
    print(f"  positive rate                 : {cand_df.label.mean() * 100:.2f} %")
    print(f"  candidate miss rate           : {(1 - obs_df.candidate_hit.mean()) * 100:.2f} %")
    print(f"  ambiguous observations        : {obs_df.ambiguous.mean() * 100:.2f} %")


if __name__ == "__main__":
    main()
