# Model card: heart

Selected: **logistic**, selected exclusively by development CV. Feature set: full. Threshold 0.269344; calibration selected: True.

Fit/calibration/threshold/test roles: data/splits/heart.json. No test tuning. Numerical/full-feature comparisons labelled in metrics. Test metrics: artifacts/metrics/heart_test_metrics.csv; confidence intervals fixed model only, clustered by patient where needed. Predictions and artifact hashes saved.

Intended use: course prototype, diagnostic classification after tests available. Not clinically validated. Historical data, label noise, population shift, missing external outcomes, limited Heart sample. Do not infer causality from coefficients/importance.
