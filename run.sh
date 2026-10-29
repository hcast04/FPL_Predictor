#!/usr/bin/env bash
# Entry point for the v2.6 fix tooling.
#
#   ./run.sh              full result set (evidence, unit tests, OOS validation)
#   ./run.sh --quick      skip the model fits
#   ./run.sh patch        regenerate FPL_Predictor_v2.6.ipynb from the original
#   ./run.sh notebook     launch Jupyter with the right interpreter
#
# Creates .venv on first use. Your system python3 has no scientific stack and
# no pip, so everything runs through this virtualenv.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "Creating .venv ..."
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install numpy pandas scipy scikit-learn requests soccerdata jupyter
fi
PY=.venv/bin/python

case "${1:-}" in
  patch)    exec "$PY" fixes/apply_v26_patch.py ;;
  notebook) exec .venv/bin/jupyter notebook FPL_Predictor_v2.6.ipynb ;;
  *)        cd fixes && exec "../$PY" run_all.py "$@" ;;
esac
