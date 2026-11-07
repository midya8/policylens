# PolicyLens design

## Input and authority boundary

This is local point-estimate arithmetic, not an online policy engine. No
network requests, actions or automated deployment. Optional crossfit fits local
categorical reward means; evaluate itself never fits a model. Callers
provide complete behavior/target action probabilities and reward predictions
aligned by explicit action names. Raw contexts are not carried; provenance,
conditioning and independent/cross-fitted predictions are caller obligations.
Strict schema validation does not prove causal identifiability.

## Estimators

`w = pi(a|x) / b(a|x)`; `DM = sum_a pi(a|x) * q(x,a)`.
IPS averages `w*r`, SNIPS divides weighted reward by total weight, DR averages
`DM + w*(r - q(x,a))`. The direct method averages DM. Clipping, when requested,
replaces weights in all weighted expressions and is explicitly labeled.
There is no self-normalized DR variant hidden under the DR name.

## Support and overlap

The full behavior vector reveals target-positive/behavior-zero actions at
observed contexts. Those rows trigger a no-nonparametric-identification label,
not a silent zero-weight success. Numeric estimates remain diagnostic; they
must not be read as identified policy values. A logged action with zero
behavior probability is rejected as inconsistent input. Tiny positive
probabilities can create huge variance even without a support violation.
ESS/max/mean/zero-weight fraction are descriptive only. Support outside the
observed contexts is unknown.

## Numerical decisions

Finite numeric values, bounded rewards/reward predictions and normalized
probability vectors. No automatic normalization or final clipping. SNIPS
uses weights divided by their maximum, preserving its ratio while avoiding
large raw sums. ESS uses those same scaled weights, avoiding raw squares.
Means divide contributions before fsum; unrepresentable weights/results fail.
If every observed target weight is zero, SNIPS=null and ESS=0. No made-up
zero-denominator result or Infinity JSON.

## Demonstration design

Two equally likely contexts. Behavior (.8,.2)/(.2,.8), target (.1,.9)/(.9,.1),
reward means (.2,.8)/(.9,.1). Known target value 0.78 and behavior value 0.29.
Twenty deterministic mean-reward rows match logging frequencies exactly;
with wrong q=.5, IPS/SNIPS/DR=.78 but direct=.5. Clip 1 changes IPS=.185,
SNIPS=.6166667 and DR=.535 while ESS rises 4.9231 to 8.4706. This isolates
arithmetic on a constructed design, not general estimator performance.
The 2000-row seed 13 Bernoulli log uses oracle q and gives DR=.7532475,
not exact population truth. Selected development fixtures, no power/variance
study, training benchmark or causal proof.

## Statistical scope

Unclipped IPS relies on correct propensities and support. Usual DR theory
requires correct nuisance components in its stated sampling/identification
setup, with independent or cross-fitted reward predictions as appropriate.
It does not fix hidden confounding, misspecified context or zero support.
SNIPS/clipping have finite-sample bias. No confidence bounds, multiple-policy
selection correction, sequential monitoring or deploy/no-deploy decision.

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

## Sources

- https://arxiv.org/pdf/1503.02834v1.pdf
- https://www.arxiv.org/pdf/1906.03323v2

Standard IPS/SNIPS/DM/DR only; not an empirical-likelihood implementation.
