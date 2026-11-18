"""Loading raw PathSnap trips and projecting them to metres.

Each trip folder contains four header-less, whitespace-separated files:
    <id>.track  lon lat t        GPS observations (~1 Hz)
    <id>.nodes  lon lat          road nodes   (row index = node_id)
    <id>.arcs   from to          directed arcs (row index = arc_id)
    <id>.route  arc_id           reference route, chronological arc sequence

Trips are spread over several countries, so every trip is projected into its
own UTM zone (chosen from the trip's mean GPS position). Distances are then in
metres and Euclidean geometry is valid at trip scale.

Only the `train` and `student_test` partitions may be loaded by development
code; the evaluator partition is reserved for the API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
DEV_SPLITS = ("train", "student_test")


def check_split(split: str) -> None:
    if split not in DEV_SPLITS:
        raise ValueError(f"Split '{split}' is not allowed in development code. Use one of {DEV_SPLITS}.")


def list_trips(split: str, raw_dir: Path = RAW_DIR) -> list[str]:
    check_split(split)
    return sorted(p.name for p in (raw_dir / split).iterdir() if p.is_dir())


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def _read(path: Path, names: list[str], dtype=None) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype, engine="c")


@dataclass
class Trip:
    trip_id: str
    track: pd.DataFrame          # lon, lat, t, x, y   (one row per GPS observation)
    nodes_xy: np.ndarray         # (n_nodes, 2) metres
    nodes_ll: np.ndarray         # (n_nodes, 2) lon, lat
    arcs: np.ndarray             # (n_arcs, 2) from_node, to_node
    route: np.ndarray | None     # reference arc ids (None if absent)
    epsg: int
    lines: np.ndarray = field(repr=False, default=None)   # shapely LineStrings, one per arc (metres)
    tree: shapely.STRtree = field(repr=False, default=None)
    twin: np.ndarray = field(repr=False, default=None)    # reverse-direction arc id or -1

    @property
    def to_lonlat(self) -> Transformer:
        return Transformer.from_crs(self.epsg, 4326, always_xy=True)


def reverse_twins(arcs: np.ndarray) -> np.ndarray:
    """For every arc (u, v) return the id of arc (v, u), or -1 if the road is one-way."""
    df = pd.DataFrame({"u": arcs[:, 0], "v": arcs[:, 1], "arc": np.arange(len(arcs))})
    rev = df.rename(columns={"u": "v", "v": "u", "arc": "twin"})
    merged = df.merge(rev, on=["u", "v"], how="left").drop_duplicates("arc").sort_values("arc")
    return merged["twin"].fillna(-1).astype(np.int64).to_numpy()


def load_trip(split: str, trip_id: str, raw_dir: Path = RAW_DIR, with_route: bool = True) -> Trip:
    check_split(split)
    folder = raw_dir / split / trip_id

    track = _read(folder / f"{trip_id}.track", ["lon", "lat", "t"])
    nodes = _read(folder / f"{trip_id}.nodes", ["lon", "lat"]).to_numpy(dtype=float)
    arcs = _read(folder / f"{trip_id}.arcs", ["u", "v"], dtype=np.int64).to_numpy()
    route = None
    route_path = folder / f"{trip_id}.route"
    if with_route and route_path.is_file():
        route = _read(route_path, ["arc"], dtype=np.int64)["arc"].to_numpy()

    # Basic cleaning: drop invalid rows, sort by time, remove duplicate timestamps.
    n_raw = len(track)
    track = track.dropna()
    track = track[(track.lon.between(-180, 180)) & (track.lat.between(-90, 90))]
    track = track.sort_values("t", kind="stable").drop_duplicates("t").reset_index(drop=True)
    track.attrs["n_dropped"] = n_raw - len(track)

    epsg = utm_epsg(float(track.lon.mean()), float(track.lat.mean()))
    fwd = Transformer.from_crs(4326, epsg, always_xy=True)
    tx, ty = fwd.transform(track.lon.to_numpy(), track.lat.to_numpy())
    track["x"], track["y"] = tx, ty
    nx, ny = fwd.transform(nodes[:, 0], nodes[:, 1])
    nodes_xy = np.column_stack([nx, ny])

    lines = shapely.linestrings(np.stack([nodes_xy[arcs[:, 0]], nodes_xy[arcs[:, 1]]], axis=1))
    return Trip(
        trip_id=trip_id, track=track, nodes_xy=nodes_xy, nodes_ll=nodes, arcs=arcs,
        route=route, epsg=epsg, lines=lines, tree=shapely.STRtree(lines), twin=reverse_twins(arcs),
    )


def route_lonlat(trip: Trip) -> list[list[float]]:
    """Reference route as a [lat, lon] polyline (for Folium)."""
    if trip.route is None or len(trip.route) == 0:
        return []
    node_seq = [trip.arcs[trip.route[0], 0]] + [trip.arcs[a, 1] for a in trip.route]
    return [[trip.nodes_ll[n, 1], trip.nodes_ll[n, 0]] for n in node_seq]
