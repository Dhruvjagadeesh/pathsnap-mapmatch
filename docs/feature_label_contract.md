# Candidate, feature and label contract (Members 1 & 2)

## Unit of prediction
There is one row per **(GPS observation, candidate arc)** pair. The model scores every candidate of an observation. The highest-scoring candidate is the observation-level prediction, and the candidate probabilities become the HMM emission scores.

## Candidate generation
- **Search:** all arcs within **50 m** of the observation (UTM metres, using an STRtree index).
- **Cap:** at most **8** per observation, keeping the closest.
- **Fallback:** if no arc lies within 50 m, the single nearest arc is kept, so no observation is ever left without candidates.
- **Candidate miss:** an observation's reference arc is not among its candidates.

## Label
`label = 1` if `arc_id == ref_arc`, where `ref_arc` comes from `docs/labelling_method.md`. Otherwise `label = 0`.

The label is attached after the features are built. Route information is never used to compute any feature.

## Identifier columns (not features)
`trip_id`, `obs_idx`, `arc_id`, `arc_u`, `arc_v`, `ambiguous`, `label`

## Features
| Column | Meaning |
|---|---|
| `dist_m` | distance from GPS point to arc (m) |
| `dist_rank` | 0 = closest candidate of this observation |
| `dist_minus_min` | `dist_m` minus the closest candidate's distance |
| `dist_ratio_min` | (closest + 1) / (`dist_m` + 1) |
| `n_candidates` | number of candidates for this observation |
| `heading_diff` | angle between GPS heading and arc direction, 0–180°; 90 when heading is unknown |
| `heading_cos` | cos(`heading_diff`); 0 when heading is unknown |
| `heading_valid` | 1 if the GPS moved at least 2 m between neighbours |
| `speed_mps` | GPS speed from neighbouring fixes |
| `arc_len_m` | arc length (m) |
| `proj_frac` | position of the projected point along the arc (0–1) |
| `is_twoway` | 1 if the reverse arc exists |
| `dist_prev_m`, `dist_next_m` | distance from the previous / next observation to the same arc |
| `dist_ctx_mean_m` | mean of previous, current and next distances |
| `dt_prev_s` | time since the previous observation (s) |

## Cross-validation
- `GroupKFold(5)` grouped by `trip_id` on the 80 training trips. No trip appears in both the training and validation parts of a fold.
- The 10 student-testing trips are used only for development comparison.
- The evaluator trips are never loaded by this code: `check_split` rejects them.

## Metrics
- **Observation-level Top-1:** over all observations, where a candidate miss counts as an error. Also reported on the unambiguous observations only.
- **Candidate-level:** macro-F1 (threshold 0.5), precision, recall and log loss.
- **Route continuity:** share of consecutive predicted arcs that are the same arc or directly connected.
- **Candidate miss rate.**
