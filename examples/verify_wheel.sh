#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
python -m pip wheel --no-deps "$root" --wheel-dir "$tmp/wheels" >/dev/null
python -m venv "$tmp/venv"
"$tmp/venv/bin/python" -m pip install --no-index "$tmp"/wheels/*.whl >/dev/null
cd "$tmp"
"$tmp/venv/bin/python" - <<'PY'
import importlib
for name in ['core','crossfit','demo','cli','ledger','ledger_fixtures','ledger_export']:importlib.import_module('policylens.'+name)
from policylens.ledger_export import main
main()
print('7 modules import outside checkout')
PY
"$tmp/venv/bin/python" -m policylens demo > demo.json
"$tmp/venv/bin/python" -m policylens ledger-demo > ledger.json
"$tmp/venv/bin/python" -m policylens evaluate --input "$root/examples/log.json" > evaluate.json
"$tmp/venv/bin/python" -m policylens crossfit --input "$root/examples/categorical.json" --folds 3 --seed 17 > crossfit.json
"$tmp/venv/bin/python" -m policylens audit --input "$root/examples/log.json" --clip 1 > audit.json
"$tmp/venv/bin/python" - <<'PY'
import json
x=json.load(open('ledger.json'));assert x['original_demo']==json.load(open('demo.json'));assert len(x['witnesses'])==95
assert json.load(open('evaluate.json'))['n']>0
assert json.load(open('crossfit.json'))['evaluation']['n']>0
assert json.load(open('audit.json'))['selected']['clip']==1.
print('all five commands, original reports and95edits reproduced')
PY
