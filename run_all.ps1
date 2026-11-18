# Runs the complete development pipeline on the train and student_test splits.
# Usage (from the project root, venv active):  .\run_all.ps1
$ErrorActionPreference = "Stop"

function Step($title, $cmd) {
    Write-Host "`n=== $title ===" -ForegroundColor Cyan
    $t = Get-Date
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $title" }
    Write-Host ("    done in {0:n1} s" -f ((Get-Date) - $t).TotalSeconds) -ForegroundColor DarkGray
}

Step "Decoder sanity check (Viterbi toy example)" "python -m src.hmm.toy_example"
Step "Labels: train"                 "python -m src.data.build_labels --split train"
Step "Labels: student_test"          "python -m src.data.build_labels --split student_test"
Step "Folium map: train 00000048"    "python -m src.viz.folium_trip --split train --trip 00000048"
Step "Features: train"               "python -m src.features.build_features --split train"
Step "Features: student_test"        "python -m src.features.build_features --split student_test"
Step "Baselines"                     "python -m src.models.baseline_experiment"
Step "Five-model comparison"         "python -m src.models.compare_models"

Write-Host "`nPipeline complete. Results in .\results, models in .\artifacts\models" -ForegroundColor Green
