"""Standard logged contextual-bandit estimators, no policy training."""
import math


def _number(v):
    return type(v) in (int, float) and math.isfinite(v)


def _probabilities(row, width):
    return isinstance(row, list) and len(row) == width and all(
        _number(v) and 0 <= v <= 1 for v in row) and math.isclose(math.fsum(row),1,abs_tol=1e-9,rel_tol=0)


def evaluate(payload, clip=None):
    """Estimate one fixed target policy using caller-supplied logged rows.

    Exact fields: action_names, rows. Each row: action, reward, behavior,
    target, reward_model. Behavior/target contain full action distributions.
    Reward/reward_model must be in [0,1]. No silent normalization or clipping.
    """
    if clip is not None and (not _number(clip) or clip <= 0):
        raise ValueError('clip must be a finite positive number or None')
    if not isinstance(payload,dict) or set(payload) != {'action_names','rows'}:
        raise ValueError('input must have exactly action_names and rows')
    names=payload['action_names'];rows=payload['rows']
    if not isinstance(names,list) or len(names)<2 or any(
        not isinstance(v,str) or not v.strip() for v in names) or len(set(names))!=len(names):
        raise ValueError('action_names must be distinct nonempty strings')
    if not isinstance(rows,list) or not rows or len(rows)>100000:
        raise ValueError('rows must be a nonempty list, maximum 100000')
    width=len(names); weights=[];ips=[];direct=[];dr=[];unsupported=0
    for row in rows:
        if not isinstance(row,dict) or set(row)!={'action','reward','behavior','target','reward_model'}:
            raise ValueError('row requires action/reward/behavior/target/reward_model only')
        a=row['action'];r=row['reward'];b=row['behavior'];t=row['target'];q=row['reward_model']
        if type(a) is not int or not 0<=a<width or not _number(r) or not 0<=r<=1:
            raise ValueError('action index/reward invalid')
        if not _probabilities(b,width) or not _probabilities(t,width):
            raise ValueError('behavior/target must be finite action probability distributions')
        if not isinstance(q,list) or len(q)!=width or any(not _number(v) or not 0<=v<=1 for v in q):
            raise ValueError('reward_model must contain one finite [0,1] prediction per action')
        if b[a]<=0:
            raise ValueError('logged action must have positive behavior propensity')
        unsupported += any(tp>0 and bp==0 for tp,bp in zip(t,b))
        w=t[a]/b[a]
        if not math.isfinite(w):
            raise ValueError('importance weight exceeds finite float range')
        if clip is not None:w=min(w,clip)
        dm=math.fsum(tp*value for tp,value in zip(t,q))
        weights.append(w);ips.append(w*r);direct.append(dm);dr.append(dm+w*(r-q[a]))
    n=len(rows)
    def mean(values):
        try:value=math.fsum(v/n for v in values)
        except OverflowError as exc:raise ValueError('mean exceeds finite range') from exc
        if not math.isfinite(value):raise ValueError('mean exceeds finite range')
        return value
    max_w=max(weights)
    scaled=[w/max_w for w in weights] if max_w else [0.]*n
    sum_scaled=math.fsum(scaled)
    snips=math.fsum(w*row['reward'] for w,row in zip(scaled,rows))/sum_scaled if sum_scaled else None
    ess=sum_scaled**2/math.fsum(w*w for w in scaled) if sum_scaled else 0.
    identifiable=unsupported==0
    return {'n':n,'ips':mean(ips),'snips':snips,'direct_method':mean(direct),'doubly_robust':mean(dr),
            'clip':clip,'weights':{'mean':mean(weights),'max':max_w,'effective_sample_size':ess,
                                 'zero_fraction':weights.count(0.)/n},
            'support':{'rows_with_unsupported_target_actions':unsupported,
                       'observed_context_support_complete':identifiable},
            'interpretation': 'support_violation_no_nonparametric_identification' if not identifiable else
                              ('clipped_estimands_may_be_biased' if clip is not None else 'assumptions_still_required'),
            'scope':'Point estimates for one fixed target policy, not a confidence bound or safe deployment '
                    'decision. Requires correct logging propensities, action/reward consistency, relevant '
                    'context, target support and appropriate sampling assumptions. Reward predictions '
                    'must be trained independently or cross-fitted for usual DR theory; schema cannot '
                    'verify this. SNIPS is finite-sample biased. Clipping can bias IPS/DR and changes '
                    'the estimator. ESS is descriptive, not a validity certificate. Support checks apply '
                    'only to supplied observed contexts; no hidden-confounding or unseen-context check.'}
