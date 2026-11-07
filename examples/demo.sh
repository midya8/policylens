#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m policylens demo
python - <<'PY'
from policylens.demo import run_demo
r=run_demo()
assert abs(r['balanced_design']['doubly_robust']-.78)<1e-12
assert abs(r['balanced_design']['ips']-.78)<1e-12
assert r['clipped_balanced_design']['weights']['effective_sample_size'] > r['balanced_design']['weights']['effective_sample_size']
try:
    assert abs(r['clipped_balanced_design']['doubly_robust']-.78)<1e-12
except AssertionError:
    print('clipped truth-equality gate fails as expected')
else:
    raise SystemExit('ERROR: clipped truth gate unexpectedly passed')
PY
