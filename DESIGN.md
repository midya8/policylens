# PolicyLens design

## Input and authority boundary

This is local point-estimate arithmetic, not an online policy engine. No
network requests, actions, reward fitting or automated deployment. Callers
provide complete behavior/target action probabilities and reward predictions
aligned by explicit action names. Raw contexts are not carried; provenance,
conditioning and independent/cross-fitted predictions are caller obligations.
Strict schema validation does not prove causal identifiability.

## Estimators

w=pi(a|x)/b(a|x); DM=sum_a pi(a|x)*q(x,a).
IPS averages w*r, SNIPS divides weighted reward by total weight, DR averages
DM+w*(r-q(x,a)). The direct method averages DM. Clipping, when requested,
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

## Sources

- https://arxiv.org/pdf/1503.02834v1.pdf
- https://www.arxiv.org/pdf/1906.03323v2

Standard IPS/SNIPS/DM/DR only; not an empirical-likelihood implementation.
