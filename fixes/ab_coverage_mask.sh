#!/usr/bin/env bash
# Resolve F2: does masking 2022-23's fabricated zero xG/xA help or hurt?
#
# You already have the F2-ON arm cached. This preserves it, so after one run
# with the mask off both arms can be compared on identical evaluation code.
#
#   ./ab_coverage_mask.sh save-on     # before changing anything
#   ...set V26_APPLY_COVERAGE_MASK = False in the notebook, Run All...
#   ./ab_coverage_mask.sh save-off
#   ./ab_coverage_mask.sh compare
set -euo pipefail
cd "$(dirname "$0")/.."
CACHE=data/walk_forward_fold_cache_base_v2_4_0
PY=.venv/bin/python

case "${1:-}" in
  save-on)  rm -rf data/fold_cache_F2on;  cp -r "$CACHE" data/fold_cache_F2on
            echo "saved F2-ON arm  -> data/fold_cache_F2on" ;;
  save-off) rm -rf data/fold_cache_F2off; cp -r "$CACHE" data/fold_cache_F2off
            echo "saved F2-OFF arm -> data/fold_cache_F2off" ;;
  compare)  exec "$PY" fixes/compare_coverage_arms.py ;;
  *) echo "usage: $0 {save-on|save-off|compare}"; exit 1 ;;
esac
