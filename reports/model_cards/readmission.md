# Model card: readmission

Selected: **xgboost**, selected exclusively by development CV. Feature set: full. Threshold 0.090540; calibration selected: True.

Fit/calibration/threshold/test roles: data/splits/readmission.json. No test tuning. Numerical/full-feature comparisons labelled in metrics. Test metrics: artifacts/metrics/readmission_test_metrics.csv; confidence intervals fixed model only, clustered by patient where needed. Predictions and artifact hashes saved.

Intended use: course prototype, encounter-level risk ranking at discharge. Not clinically validated. Historical data, label noise, population shift, missing external outcomes, limited Heart sample. Do not infer causality from coefficients/importance.
