#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
"$PYTHON" -m healthcare_ml.cli download --all
"$PYTHON" -m healthcare_ml.cli validate-data --all
"$PYTHON" -m healthcare_ml.cli prepare --all
"$PYTHON" -m healthcare_ml.cli smoke --all
if [[ "${1:-}" == "--smoke" ]]; then "$PYTHON" -m pytest -q; exit; fi
"$PYTHON" -m healthcare_ml.cli train --config configs/heart.yaml
"$PYTHON" -m healthcare_ml.cli train --config configs/readmission.yaml
"$PYTHON" -m healthcare_ml.cli eda --all
"$PYTHON" -m healthcare_ml.cli ablations --all
"$PYTHON" -m healthcare_ml.cli optimization
"$PYTHON" -m healthcare_ml.cli freeze --all
"$PYTHON" -m healthcare_ml.cli evaluate --all
"$PYTHON" -m healthcare_ml.cli explain --all
"$PYTHON" -m pytest -q
"$PYTHON" -m healthcare_ml.cli report --all

