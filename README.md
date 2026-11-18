# PathSnap — Map-Matching & Route Reconstruction

PathSnap reconstructs the road route a vehicle actually drove from noisy GPS traces. It works in four stages:

1. **Candidate generation:** nearby road arcs for every GPS fix.
2. **ML classifier:** scores each candidate.
3. **HMM + Viterbi decoding:** turns the per-fix scores into one connected route.
4. **FastAPI service + web UI:** a two-laptop evaluation setup.

UE24CS352A Machine Learning — PES University.

## Project structure
```
src/
  data/       io.py (loading, per-trip UTM projection), labels.py + build_labels.py (reference-arc labels)
  features/   candidates.py (candidate generation, 16 features), build_features.py
  models/     evaluate.py (metrics), baseline_experiment.py, compare_models.py (5 models)
  hmm/        viterbi.py (decoder), toy_example.py (hand-checkable test)
  viz/        folium_trip.py (interactive maps)
api/          main.py — FastAPI server (Laptop 1)
ui/           index.html — evaluator interface (Laptop 2)
docs/         labelling_method.md, feature_label_contract.md
results/      generated metrics, plots and maps
artifacts/    trained models
data/         raw/train, raw/student_test (not committed), evaluator/ (server only, never committed)
```

## Setup (Windows, PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Place the dataset as `data\raw\train\<trip_id>\` (80 trips) and `data\raw\student_test\<trip_id>\` (10 trips).

## Run the full pipeline
```powershell
.\run_all.ps1
```
Or run each step on its own:

| Step | Command |
|---|---|
| Viterbi toy check | `python -m src.hmm.toy_example` |
| Labels | `python -m src.data.build_labels --split train` (then `student_test`) |
| Folium map | `python -m src.viz.folium_trip --split train --trip 00000048` |
| Features | `python -m src.features.build_features --split train` (then `student_test`) |
| Baselines | `python -m src.models.baseline_experiment` |
| Five models | `python -m src.models.compare_models` |

## Method summary
- **Projection:** trips come from several countries, so each trip is projected into its own UTM zone and all distances are in metres.
- **Labels:** GPS fixes are aligned to the ordered reference route with a monotone dynamic-programming alignment. Ambiguous fixes are flagged. See `docs/labelling_method.md`.
- **Candidates:** up to 8 arcs within 50 m of each fix, with 16 geometric and kinematic features. See `docs/feature_label_contract.md`.
- **Models:** Logistic Regression, KNN, Random Forest, LightGBM and MLP. They are compared with 5-fold cross-validation grouped by trip on the 80 training trips, then evaluated on the 10 student-testing trips.
- **Decoder:** log-space Viterbi over the candidate arcs of each fix. It handles missing candidates and disconnected transitions by restarting the route segment.

## Two-laptop deployment
**Laptop 1 (API):**
1. Allow the port through the firewall. Run this once, in an Administrator PowerShell:
   ```powershell
   New-NetFirewallRule -DisplayName "PathSnap API 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
   ```
2. Start the server:
   ```powershell
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

**Laptop 2 (UI):**
1. In the `ui` folder, run `python -m http.server 5500`.
2. Open `http://localhost:5500`.
3. Enter `http://<Laptop1-IPv4>:8000` and click **Connect**.

## Data integrity
- Development code can only load the `train` and `student_test` splits.
- Evaluator data stays on Laptop 1 and is never committed.
- Ground-truth routes are used only on the server, to score a prediction after it has been made.
