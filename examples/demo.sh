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

python -m policylens crossfit --input examples/categorical.json --folds 3 --seed 17 > /tmp/policylens-crossfit-example.json
python - <<'PY2'
import json
from policylens import evaluate
with open('/tmp/policylens-crossfit-example.json') as source:
    result=json.load(source)
assert evaluate(result['prepared_input'])==result['evaluation']
assert result['cross_fit']['fold_sizes']==[4,4,4]
print('cross-fit CLI prepared-input round trip passes')
PY2
