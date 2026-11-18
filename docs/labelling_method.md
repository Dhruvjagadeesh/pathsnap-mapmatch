# Assigning a reference arc to each GPS observation

## The problem
Each trip's `.route` file lists the directed arcs the vehicle travelled, in order. It does not say which GPS observation belongs to which arc. For example, trip 00000048 has 1240 observations but only 149 route arcs. To train a classifier we need one reference arc per observation.

## Data facts that shape the method
Checked on the training data:
- **Arcs are directed.** Most roads appear as a pair of opposite arcs `(u, v)` and `(v, u)` with identical geometry. Some roads are one-way and have no reverse arc.
- **The route is a connected directed path.** The end node of each route arc is the start node of the next one.
- **Trips come from several countries** (UK, Germany, Russia, Japan, …). Each trip is therefore projected to its own UTM zone, so distances are measured in metres.

## Method: monotone alignment by dynamic programming
Let `d[i, k]` be the point-to-segment distance in metres between observation `i` and route arc `k`.

We choose route positions `k_0 ≤ k_1 ≤ … ≤ k_{N-1}` that minimise `Σ d[i, k_i]`.

```
D[0, k] = d[0, k]
D[i, k] = d[i, k] + min_{j ≤ k} D[i-1, j]
```

The inner minimum is a running prefix minimum, so the cost is O(N·K). Backtracking then gives each observation its route position, and from that its arc id.

### Why this and not "nearest route arc"
- **Opposite-direction twins.** These are geometrically identical, so the nearest-arc rule cannot choose between them. The route order can.
- **Revisited roads.** When the route passes the same road twice, the monotone constraint assigns each pass to the correct part of the route.
- **Noisy fixes.** A noisy GPS fix cannot jump back to an earlier part of the route.

Route arcs that receive no observation, typically short arcs passed between two 1 Hz samples, are allowed. They are counted in `route_arcs_skipped`.

## Ambiguity flags
These are stored per observation and are **never used as model features**.

| Flag | Rule | Meaning |
|---|---|---|
| `flag_far` | assigned arc > 30 m away | GPS error, or the route doesn't explain this fix |
| `flag_competing` | another arc (other than the assigned arc and its reverse twin) within assigned distance + 3 m | parallel road or junction: the label is plausible but not certain |
| `flag_near_node` | within 10 m of either end of the assigned arc | boundary between consecutive arcs is uncertain |
| `ambiguous` | `flag_far OR flag_competing` | used to report accuracy on unambiguous observations separately |

All observations are kept for training and evaluation. Accuracy is also reported on the unambiguous subset, so label noise is visible rather than hidden.

## Limitations
- The reference route is treated as the reference, not as verified physical truth.
- Alignment errors are most likely at junctions and on short arcs, which is what `flag_near_node` marks.
- The 30 m, 3 m and 10 m thresholds are engineering choices, and are reported as such.

## Reproduce
```
python -m src.data.build_labels --split train
python -m src.data.build_labels --split student_test
python -m src.viz.folium_trip --split train --trip 00000048
```
