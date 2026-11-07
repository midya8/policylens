# PolicyLens

Offline contextual-bandit value estimation from logged actions, rewards and
propensities. IPS, self-normalized IPS, direct method and doubly robust
estimates, with support/weight diagnostics and explicit clipping. Dependency-
free Python 3.10+, JSON CLI/API, no online actions; optional held-out tabular reward fitting.

## Known truth, then a deliberately biased shortcut

```bash
python -m policylens demo
bash examples/demo.sh
```

| Development arm | IPS | SNIPS | Direct | DR | Weight ESS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exactly balanced design, wrong reward model | 0.7800 | 0.7800 | 0.5000 | 0.7800 | 4.9231/20 |
| Same design, weights capped at 1 | 0.1850 | 0.6167 | 0.5000 | 0.5350 | 8.4706/20 |
| Seeded 2000-row log, oracle reward model | 0.7286 | 0.7533 | 0.7796 | 0.7532 | 478.4011/2000 |
| Same noisy log, five-fold tabular reward model | 0.7286 | 0.7533 | 0.7496 | 0.7509 | 478.4011/2000 |

Known population target value 0.78, behavior value 0.29. Two equally likely
contexts, two actions. The balanced design matches behavior frequencies
exactly and uses deterministic mean rewards; it is an arithmetic identity
fixture, not a random trial. It recovers target value despite a wrong
constant 0.5 reward model. Clipping improves descriptive ESS but biases the
IPS/DR estimates on this design. Bigger ESS does not certify correctness.

The seeded log is a separate noisy realization with Bernoulli rewards.
Its reward model is the true conditional expectation, an oracle, not a learned
or cross-fitted predictor. Even then DR 0.7532 does not equal population 0.78.
These selected development fixtures are not a variance, accuracy or causal
identification benchmark. No estimator is claimed universally best.

## Evaluate your own fixed policy

```bash
python -m policylens evaluate --input examples/log.json
python -m policylens evaluate --input examples/log.json --clip 1
```

```python
from policylens import evaluate
report = evaluate(payload)  # optional clip=1
```

Input has exactly `action_names` and `rows`. Each row contains action index,
observed reward, full behavior probabilities, full target probabilities and
one reward prediction per action. Action order is shared across every row.
Rewards and reward predictions must be in[0,1]; distributions must sum to1
within1e-9 and are not silently normalized.

```json
{"action_names":["a","b"],"rows":[
 {"action":0,"reward":1,"behavior":[0.5,0.5],
  "target":[0.25,0.75],"reward_model":[0.6,0.4]}]}
```

`--input -` reads stdin. Errors exit2 with stderr and no partial JSON.
Maximum100000 rows. The library does not mutate inputs, fit a policy/reward
model, upload data, execute actions or write files.

## Learn reward predictions without using held-out rewards

`evaluate` still accepts supplied predictions unchanged. The optional `crossfit`
command instead takes exact row fields `context`, `action`, `reward`, `behavior`,
`target`. Contexts are explicit categorical strings. It shuffles row indices with
an isolated seeded RNG, assigns balanced folds, and trains each prediction on all
folds except that row's entire fold. It fits only observed context-action cell
reward means, shrunk toward a fixed prior:

`q = (training_reward_sum + prior_strength * prior_mean) / (training_count + prior_strength)`

Unseen cells return `prior_mean` exactly, even for tiny positive prior strengths.
Defaults: five folds, seed 0, prior mean 0.5, prior strength 1. No tuning occurs.
Folds must be 2 through min(20, n); prior strength is positive and at most 1e6.

```sh
python -m policylens crossfit --input examples/categorical.json --folds 3 --seed 17
```

The JSON result contains `prepared_input` (usable by `evaluate`), `cross_fit`
(split seed, per-row fold assignment, per-action training counts and prior-only
prediction counts), and `evaluation`. Source row/action order is preserved;
input is not mutated. Library `cross_fit` returns preparation and diagnostics;
call `evaluate(result['prepared_input'])` separately.

On the same 2000-row noisy development log, five folds/seed 17 with the fixed
default prior give direct 0.7496455 and DR 0.7509088, versus oracle DR 0.7532475
and known population value 0.78. All folds have 400 rows and no cell needs the
prior-only fallback. **This is not an improvement over the oracle arm**, an
independent test set, or an estimator ranking. Hyperparameters were not selected
by a sweep; the fixture remains development data.

Whole-fold reward perturbation tests confirm held-out predictions cannot use
those rewards. This is row-iid cross-fitting, not group/time splitting: repeated
people, clustered logs and adaptive trajectories can still leak across folds.
Context encoding, target policy and hyperparameters must be fixed independently
of evaluation outcomes. Sparse/unseen cells can be mostly prior, and held-out
prediction does not make the categorical reward model correct. Behavior
probabilities are not learned or authenticated. Unsupported target actions remain
flagged after fitting. No confidence intervals or deployment decision are added.

Cross-fitting background: [Kallus et al. (2022), introduction and algorithm discussion](https://proceedings.mlr.press/v162/kallus22a/kallus22a.pdf).
PolicyLens implements only ordinary contextual-bandit estimators and a tabular
reward-model helper, not that paper's distributionally robust algorithms or its
efficiency theorem. No new statistical method.

## Estimators and diagnostics

For each row, `w=target[action]/behavior[action]` and
`dm=sum_a target[a]*reward_model[a]`:

- IPS: `mean(w*reward)`.
- SNIPS: `sum(w*reward)/sum(w)`, or null if all observed weights are zero.
- Direct method: `mean(dm)`.
- DR: `mean(dm+w*(reward-reward_model[action]))`.

No output estimate is silently clamped to[0,1]. Importance estimators can
leave the reward range in finite samples. Weight ESS uses normalized weights
to avoid squaring extreme raw weights: `(sum(w))^2/sum(w^2)`. It is a
concentration diagnostic, not a statistical coverage/validity guarantee.

Full behavior vectors let the tool flag target probability on an action with
zero logging support at each supplied context. Reports keep numeric estimates
but label this `support_violation_no_nonparametric_identification`, rather
than laundering it through the reward model. A logged action with zero
behavior propensity is invalid and rejected. Observed-context checks cannot
prove support on unseen contexts or authenticate supplied propensities.

Optional clipping replaces w by min(w,clip) for IPS, SNIPS and DR, reports the
cap explicitly and labels possible bias. It is not an automatic repair for
lack of support, poor overlap, hidden confounding or a bad reward model.

## Assumptions and limits

Usual off-policy identification needs correct logging propensities conditional
on relevant context, consistency of actions/rewards, target support and the
sampling setup assumed by the estimator. Reward predictions must be trained
independently or cross-fitted for usual DR theory. No fitting/cross-fitting
is performed or verified here. Rows lack explicit raw contexts; the caller
owns how probabilities and predictions were conditioned on them.

The standard double-robustness result is not a promise that this finite report
is unbiased regardless of input quality. SNIPS is finite-sample biased.
Clipped estimates are generally biased. No confidence intervals, sequential
validity, learning guarantees, safety/deployment approval or policy selection
from the same log are supplied. Comparing many policies or selecting after
seeing estimates needs a separate analysis plan and data discipline.

## Verification and attribution

28 tests cover hand arithmetic, on-policy identities, DR reductions, support
violations, zero weights, invalid rows, extreme weights/ESS, explicit clipping,
no mutation, CLI failures and demo regressions. CI runs Python 3.10-3.13 and
the offline demo, including a passing exact-truth gate and a failing clipped
truth-equality gate.

Standard estimators, original small implementation, not new theory:
- [Dudik et al., Doubly Robust Policy Evaluation and Optimization](https://arxiv.org/pdf/1503.02834v1.pdf)
- [Karampatziakis et al., Empirical Likelihood for Contextual Bandits](https://www.arxiv.org/pdf/1906.03323v2), IPS/SNIPS comparison and reward-predictor discussion. This project does not implement their empirical-likelihood estimator.

See [DESIGN.md](DESIGN.md). MIT licensed.
