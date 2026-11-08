# Row contributions, cap paths and delete-one-row sensitivity

PolicyLens v0.2 adds a bounded witness around the existing contextual-bandit
estimators. It is not a new estimator, confidence method, policy optimizer,
causal-identification result or deployment decision. The original `evaluate`,
`cross_fit`, random seed, data-generating setup, reward model fits and demo
reports stay unchanged. Original tests are retained without weaker tolerances.

## What is held fixed

The inspector has five public synthetic cases: the original 20-row exactly
balanced design, unsupported target actions, zero observed target mass, a
rare-action large weight with negative residual and an on-policy control.
The live browser accepts only a mixture of the original target with behavior,
one row's observed reward, and a positive weight cap. It does not fit a policy,
refit reward predictions, authenticate propensities or collect user data.

The target is `pi_lambda = (1-lambda)*mu + lambda*pi_original`, with lambda in
[0,1]. A selected reward edit lies in [0,1]. All behavior vectors, logged actions
and reward predictions remain fixed. Changing the selected reward row restores
other rows to their original values. This is a single-row edit, not a persistent
multi-row editor. The original population value 0.78 is shown only when the
balanced input and target are unchanged. The original four demo reports always
remain visible separately, including both noisy-log reward-prediction arms.
Those arms use the same development log; neither is an independent test.

## Row algebra

For row i, action a_i, reward r_i, full behavior vector mu_i, target pi_i and
fixed reward prediction vector q_i, define:

```
w_i = pi_i[a_i] / mu_i[a_i]
u_i(c) = min(w_i, c), c > 0, or u_i = w_i without a cap
d_i = sum_a pi_i[a] q_i[a]
e_i = r_i - q_i[a_i]

IPS = sum_i u_i r_i / n
DM  = sum_i d_i / n
DR  = sum_i (d_i + u_i e_i) / n
SNIPS = sum_i u_i r_i / sum_i u_i  (undefined if denominator is zero)
```

Each ledger row exposes these quantities, its additive IPS/DM/DR contribution,
its normalized weight and SNIPS contribution, and all target actions with zero
supplied behavior probability. A logged action with zero propensity is rejected,
not divided by zero. Full behavior/target vectors retain the original 1e-9
absolute sum tolerance and are not silently normalized.

With nonnegative weights and rewards in [0,1], defined SNIPS is a convex
combination of observed rewards. IPS and DR need not be in [0,1]. In particular,
a large weight multiplying a negative residual can make DR negative. Estimates
are never silently truncated to a plausible-looking reward range.

The signed cap removals are `(w_i-u_i)*r_i/n` for IPS and
`(w_i-u_i)*e_i/n` for DR. Their sums equal raw minus capped estimates up to
floating-point error. DR removal can be negative. Do not infer that a smaller
weight necessarily makes the estimate smaller or closer to truth.

## Cap path, not cap selection

Positive observed raw weights are the exact real-arithmetic breakpoints of the
min-weight operation. On an interval between breakpoints, the set C of capped
rows is fixed. Define `R = sum_{i not in C} w_i*r_i`,
`R_C = sum_{i in C} r_i`, `W = sum_{i not in C} w_i`,
and `W2 = sum_{i not in C} w_i**2`. Then

```
IPS(c) = (R + c*R_C)/n
DR(c) = mean(d_i) + [sum_{i not in C} w_i*e_i + c*sum_{i in C} e_i]/n
SNIPS(c) = (R + c*R_C) / (W + c*|C|)
ESS(c) = (W + c*|C|)**2 / (W2 + c**2*|C|)
```

IPS and DR are affine on that interval. SNIPS and ESS generally are not. These
identities describe this fixed log; they do not identify an optimal cap. All-zero
weights make SNIPS undefined and descriptive ESS zero, rather than inventing a
normalization. The browser curve samples 41 evenly spaced positive caps from
0.01 to the maximum raw weight (at least 0.01), plus all in-range breakpoints.
Polylines connect these evaluated points. SNIPS between points is not exactly
linear. The curve table exposes every sampled value. The entered cap may lie
outside this plotted range and is always reported separately. The chart includes
zero on its shared value axis; it does not clip negative DR values.

ESS is computed as `(sum s_i)**2 / sum(s_i**2)` with `s_i = u_i/max(u)` to avoid
squaring extreme raw weights. Likewise SNIPS uses scaled weights. This changes
neither real-arithmetic ratio. ESS is descriptive weight concentration, not
sample independence, statistical validity, uncertainty coverage or proof of
representative contexts. Clipping may increase ESS while biasing the estimator,
as the original balanced fixture demonstrates.

## Delete one supplied row

For each row, `audit` calls the unchanged evaluator on all other rows. Remaining
estimator denominators are n-1; SNIPS re-normalizes its remaining weights. With
only one row there is no delete-one-row report. Reward predictions and the
policy stay fixed throughout. This is conditional deletion sensitivity, not
retraining, cross-validation, a jackknife variance estimator, a standard error,
a confidence interval or an iid guarantee. The browser independently computes
these reports; the chart uses a shared signed DR axis and the table preserves
all four estimators, including undefined SNIPS where deletion removes all mass.

## Identification and provenance boundaries

The native audit calls `evaluate` to obtain both unchanged raw and selected
reports, then exposes row algebra. It accepts at most 256 rows and 16 actions;
the original evaluator retains its original 100,000-row limit. The independent
browser reimplements the estimators for its fixed fixtures; it is not executing
the user's callback or certifying that externally supplied propensities are real.
Nothing is sent, uploaded, trained or deployed by this browser.

A positive target probability with zero behavior support in a supplied observed
context is reported even when the reward model has a plausible prediction.
Numeric DM/DR estimates remain visible, but missing outcomes are not identified
nonparametrically. Conversely, a clean observed-context check does not establish
support on unseen contexts, correct conditioning, authenticated propensities,
consistency, no hidden confounding or an appropriate sampling process.

Standard double robustness is a statement under assumptions, not finite-sample
truth equality. Fixed/or independently trained reward predictions are different
from fitting to the evaluated outcomes. The original optional row-fold tabular
crossfit helper remains unchanged. Its held-out exclusion tests do not prove
valid group/time splitting or correctness of its model. This release does not
implement distributionally robust OPE, optimized shrinkage, empirical likelihood,
policy selection or high-confidence lower bounds from the papers cited below.

## Numerical and reproduction contract

Both implementations use binary64 arithmetic and compensated summation (Python
`math.fsum`, independent JavaScript compensated sums). Native means divide terms
before summing; browser means do the same. Reports must remain finite; nonfinite
inputs and results refuse. A finite residual or small native-browser discrepancy
is not a certified forward-error bound. Probability vectors at a tolerance
boundary and extreme subnormal inputs remain floating-point decisions, not
exact real arithmetic. UI controls are intentionally bounded: mixture/reward
[0,1], cap [0.01,100]. Invalid input clears old reports, charts and export data.

```
python -m unittest discover -s tests -v
node --test tests/test_ledger.cjs tests/test_ledger_browser.cjs
python -m policylens.ledger_export
python -m policylens audit --input examples/log.json --clip 1
python -m policylens ledger-demo
bash examples/verify_wheel.sh
```

36 Python tests include 160 independent exact Fraction games and piecewise-cap
identities. Four Node/actual-Chrome tests include 80 separate integer-grid oracle
games, 95 native/browser edit witnesses, 589 row ledgers and 589 delete-one-row
reports. Chrome checks rendered terms, signed SVG geometry, keyboard edits,
invalid/export refusal, clipboard failure/success and Blob download. Widths
320/390/1280 check global overflow, touch targets and readable SVG text; actual
screenshots are inspected separately. Archive comparison checks exact keys,
list lengths and nonfloat values, rejects nonfinite numbers and permits 1e-12
absolute/relative error only for floating-point leaves. Original estimator test
tolerances and reports are unchanged. This is implementation verification on
selected synthetic fixtures, not a statistical performance benchmark.

## Sources and what this release uses

1. [Dudik et al., Doubly Robust Policy Evaluation and Optimization (2014/2015 version)](https://arxiv.org/pdf/1503.02834v1.pdf): IPS/DM/DR equations, support and bias/variance assumptions. Standard estimators, not a new algorithm.
2. [Su et al., Doubly robust off-policy evaluation with shrinkage (ICML 2020)](https://proceedings.mlr.press/v119/su20a.html): clipping/shrinkage has a bias-variance tradeoff. This release does not implement optimized shrinkage or claim the paper's empirical gains.
3. [Wang et al., Optimal and Adaptive Off-policy Evaluation in Contextual Bandits (ICML 2017)](https://proceedings.mlr.press/v70/wang17a.html): finite-sample OPE tradeoffs; no universal estimator ranking inferred from these selected fixtures.
4. [Kuzborskij et al., Confident Off-Policy Evaluation and Selection through Self-Normalized Importance Weighting (AISTATS 2021)](https://proceedings.mlr.press/v130/kuzborskij21a/kuzborskij21a.pdf): SN normalization, finite-sample bias and ESS discussion. Its confidence lower bound is not implemented here.
5. [Vowpal Wabbit, Offline Policy Evaluation Using the Command Line](https://vowpalwabbit.org/docs/vowpal_wabbit/python/latest/tutorials/off_policy_evaluation.html): practical logged-feedback estimator setup. The documentation's shorthand unbiasedness statements require statistical assumptions; this interface does not turn them into input-validity guarantees. Living documentation, not version-pinned.
6. [Nan Jiang, Importance Sampling lecture](https://nanjiang.cs.illinois.edu/files/cs498/9_is.pdf): change-of-measure and importance-sampling background. Educational source, not a performance guarantee.
7. [Kallus et al., Doubly Robust Distributionally Robust Off-Policy Evaluation and Learning (ICML 2022)](https://proceedings.mlr.press/v162/kallus22a/kallus22a.pdf): out-of-fold fitting context. The original tabular helper is ordinary row-fold cross-fitting, not this paper's localized/distributionally robust method or efficiency result.

Independent educational implementation. Attribution is not endorsement.
