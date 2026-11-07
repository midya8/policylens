# PolicyLens

Offline contextual-bandit value estimation from logged actions, rewards and
propensities. IPS, self-normalized IPS, direct method and doubly robust
estimates, with support/weight diagnostics and explicit clipping. Dependency-
free Python3.10+, JSON CLI/API, no online actions or model training.

## Known truth, then a deliberately biased shortcut

```bash
python -m policylens demo
bash examples/demo.sh
```

| Development arm | IPS | SNIPS | Direct | DR | Weight ESS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exactly balanced design, wrong reward model | 0.7800 | 0.7800 | 0.5000 | 0.7800 | 4.9231/20 |
| Same design, weights capped at1 | 0.1850 | 0.6167 | 0.5000 | 0.5350 | 8.4706/20 |
| Seeded2000-row log, oracle reward model | 0.7286 | 0.7533 | 0.7796 | 0.7532 | 478.4011/2000 |

Known population target value0.78, behavior value0.29. Two equally likely
contexts, two actions. The balanced design matches behavior frequencies
exactly and uses deterministic mean rewards; it is an arithmetic identity
fixture, not a random trial. It recovers target value despite a wrong
constant0.5 reward model. Clipping improves descriptive ESS but biases the
IPS/DR estimates on this design. Bigger ESS does not certify correctness.

The seeded log is a separate noisy realization with Bernoulli rewards.
Its reward model is the true conditional expectation, an oracle, not a learned
or cross-fitted predictor. Even then DR0.7532 does not equal population0.78.
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

16 tests cover hand arithmetic, on-policy identities, DR reductions, support
violations, zero weights, invalid rows, extreme weights/ESS, explicit clipping,
no mutation, CLI failures and demo regressions. CI runs Python3.10-3.13 and
the offline demo, including a passing exact-truth gate and a failing clipped
truth-equality gate.

Standard estimators, original small implementation, not new theory:
- [Dudik et al., Doubly Robust Policy Evaluation and Optimization](https://arxiv.org/pdf/1503.02834v1.pdf)
- [Karampatziakis et al., Empirical Likelihood for Contextual Bandits](https://www.arxiv.org/pdf/1906.03323v2), IPS/SNIPS comparison and reward-predictor discussion. This project does not implement their empirical-likelihood estimator.

See [DESIGN.md](DESIGN.md). MIT licensed.
