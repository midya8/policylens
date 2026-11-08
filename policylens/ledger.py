"""Bounded row witnesses for the existing estimators, not a new estimator."""
import math
from .core import evaluate


def audit(payload, clip=None):
    """Return original reports plus inspectable additive and normalized terms.

    Audits are limited to 256 rows/16 actions; evaluate retains its own limits.
    LOO values delete one supplied row and renormalize, never refit q or policy.
    They are sensitivity diagnostics, not cross-validation or confidence bounds.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get('rows'), list) or not 1 <= len(payload['rows']) <= 256:
        raise ValueError('audit requires 1-256 rows')
    if not isinstance(payload.get('action_names'), list) or len(payload['action_names']) > 16:
        raise ValueError('audit requires at most 16 actions')
    raw = evaluate(payload)
    selected = evaluate(payload, clip)
    rows = payload['rows']; n = len(rows)
    weights = [r['target'][r['action']] / r['behavior'][r['action']] for r in rows]
    used = [min(w, clip) if clip is not None else w for w in weights]
    maximum = max(used)
    scaled = [w / maximum if maximum else 0. for w in used]
    mass = math.fsum(scaled)
    witnesses = []
    for i, (r, w, u, s) in enumerate(zip(rows, weights, used, scaled)):
        a = r['action']; dm = math.fsum(p*q for p,q in zip(r['target'],r['reward_model']))
        residual = r['reward'] - r['reward_model'][a]
        normalized = s / mass if mass else None
        witness = {'row': i, 'action': a, 'reward': r['reward'],
                   'behavior_observed': r['behavior'][a], 'target_observed': r['target'][a],
                   'raw_weight': w, 'used_weight': u, 'clipped': u < w,
                   'direct_row': dm, 'residual': residual,
                   'ips_contribution': u*r['reward']/n, 'direct_contribution': dm/n,
                   'dr_contribution': (dm+u*residual)/n,
                   'normalized_weight': normalized,
                   'snips_contribution': normalized*r['reward'] if normalized is not None else None,
                   'ips_removed_by_cap': (w-u)*r['reward']/n,
                   'dr_removed_by_cap': (w-u)*residual/n,
                   'unsupported_actions': [j for j,(b,t) in enumerate(zip(r['behavior'],r['target'])) if b == 0 and t > 0]}
        witnesses.append(witness)
    loo = []
    if n > 1:
        for i in range(n):
            out = evaluate({'action_names':payload['action_names'], 'rows':rows[:i]+rows[i+1:]}, clip)
            loo.append({'removed_row':i, 'n':n-1, **{k:out[k] for k in ['ips','snips','direct_method','doubly_robust']}})
    result = {'raw': raw, 'selected': selected, 'rows': witnesses, 'leave_one_row_out':loo,
              'cap_breakpoints': sorted(set(w for w in weights if w > 0)),
              'scope':'Fixed supplied policy, reward predictions and observed rows. Deleting a row does not refit. '
                      'Breakpoints are positive observed raw weights, not recommended caps. Between breakpoints '
                      'IPS/DR are affine in cap; SNIPS and ESS are generally not. No variance, confidence, '
                      'identification, safe deployment, policy selection or causal validity certificate.'}
    def finite(x):
        if isinstance(x,float) and not math.isfinite(x): raise ValueError('audit witness exceeds finite range')
        if isinstance(x,dict):
            for v in x.values(): finite(v)
        if isinstance(x,list):
            for v in x: finite(v)
    finite(result)
    return result
