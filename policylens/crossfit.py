"""Held-out categorical-context reward means, no propensity fitting."""
import math
import random
from .core import evaluate, _number


def cross_fit(payload, folds=5, seed=0, prior_mean=0.5, prior_strength=1.0):
    """Build out-of-fold predictions, preserving row and action order.

    Input rows require context/action/reward/behavior/target, with no existing
    reward_model. Context is an explicit categorical string, never inferred.
    Row-iid fold splitting is not group/time splitting. Hyperparameters must
    be fixed independently of evaluation results; no tuning occurs here.
    """
    if not isinstance(payload, dict) or set(payload) != {'action_names', 'rows'}:
        raise ValueError('cross-fit input requires exactly action_names and rows')
    rows = payload['rows']
    if not isinstance(rows, list) or not rows or len(rows) > 100000:
        raise ValueError('rows must be a nonempty list, maximum 100000')
    if type(folds) is not int or not 2 <= folds <= min(20, len(rows)):
        raise ValueError('folds must be an integer from 2 to min(20, n)')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    if not _number(prior_mean) or not 0 <= prior_mean <= 1:
        raise ValueError('prior_mean must be finite and in [0,1]')
    if not _number(prior_strength) or not 0 < prior_strength <= 1e6:
        raise ValueError('prior_strength must be finite in (0, 1000000]')
    names = payload['action_names']
    if not isinstance(names, list):
        raise ValueError('action_names must be a list')
    prepared = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'context', 'action', 'reward', 'behavior', 'target'}:
            raise ValueError('cross-fit rows require context/action/reward/behavior/target only')
        if not isinstance(row['context'], str) or not row['context'].strip() or len(row['context']) > 1024:
            raise ValueError('context must be a nonempty categorical string, maximum 1024 characters')
        prepared.append({k: (list(v) if isinstance(v, list) else v)
                         for k, v in row.items() if k != 'context'})
        prepared[-1]['reward_model'] = [prior_mean] * len(names)
    output = {'action_names': list(names), 'rows': prepared}
    # Validate all actions/rewards/propensities before using them as keys/indices.
    evaluate(output)
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    assignment = [0] * len(rows)
    for position, index in enumerate(order):
        assignment[index] = position % folds
    cells = {}
    for i, row in enumerate(rows):
        cell = cells.setdefault((row['context'], row['action']), [[0, 0.] for _ in range(folds)])
        cell[assignment[i]][0] += 1
        cell[assignment[i]][1] += row['reward']
    cache = {}
    counts = []
    fallback = target_fallback = 0
    for i, row in enumerate(rows):
        train_counts = []
        for action in range(len(names)):
            key = (row['context'], action, assignment[i])
            if key not in cache:
                pieces = cells.get((row['context'], action), [])
                count = sum(v[0] for f, v in enumerate(pieces) if f != assignment[i])
                total = math.fsum(v[1] for f, v in enumerate(pieces) if f != assignment[i])
                prediction = ((total + prior_strength * prior_mean) / (count + prior_strength)
                              if count else prior_mean)
                cache[key] = count, prediction
            count, prediction = cache[key]
            train_counts.append(count)
            prepared[i]['reward_model'][action] = prediction
            fallback += count == 0
            target_fallback += count == 0 and row['target'][action] > 0
        counts.append(train_counts)
    return {'prepared_input': output,
            'cross_fit': {'folds': folds, 'seed': seed, 'fold_assignment': assignment,
                          'fold_sizes': [assignment.count(f) for f in range(folds)],
                          'training_counts': counts,
                          'prior_mean': prior_mean, 'prior_strength': prior_strength,
                          'prior_only_predictions': fallback,
                          'target_positive_prior_only_predictions': target_fallback,
                          'scope': 'Categorical cell means with a fixed pseudo-count prior, not a Bayesian interval. '
                                   'Each prediction excludes its entire row fold. Row-iid splitting only: no group/time '
                                   'leakage protection. Independent rows, fixed hyperparameters, correct propensities, '
                                   'relevant contexts and target support remain assumptions. No tuning, propensity fitting, '
                                   'confidence interval or guarantee that the reward model is correct.'}}
