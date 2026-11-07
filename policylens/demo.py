"""Exact balanced design and seeded logged fixture; no real performance claim."""
import random
from .core import evaluate


def run_demo():
    # Exactly weighted design with deterministic rewards, not a random log.
    names=['a','b'];rows=[]
    for kind in [0,1]:
        behavior=[.8,.2] if kind==0 else [.2,.8]
        target=[.1,.9] if kind==0 else [.9,.1]
        truth=[.2,.8] if kind==0 else [.9,.1]
        for a,count in enumerate([8,2] if kind==0 else [2,8]):
            for _ in range(count):
                rows.append({'action':a,'reward':truth[a],'behavior':behavior,'target':target,
                             'reward_model':[.5,.5]})
    exact={'action_names':names,'rows':rows}
    rng=random.Random(13);log=[]
    for _ in range(2000):
        kind=rng.randrange(2);b=[.8,.2] if kind==0 else [.2,.8]
        t=[.1,.9] if kind==0 else [.9,.1];truth=[.2,.8] if kind==0 else [.9,.1]
        a=0 if rng.random()<b[0] else 1
        log.append({'action':a,'reward':float(rng.random()<truth[a]),'behavior':b,'target':t,
                    'reward_model':truth})
    return {'known_target_value':.78,'known_behavior_value':.29,
            'balanced_design':evaluate(exact),
            'clipped_balanced_design':evaluate(exact,clip=1),
            'seeded_log':evaluate({'action_names':names,'rows':log}),
            'scope':'Selected two-context/two-action development fixtures. Balanced design exactly '
                    'matches behavior frequencies and uses deterministic mean rewards; not a random '
                    'trial. Seeded log uses true reward expectations as an oracle reward model, not '
                    'a learned or cross-fitted model. No estimator accuracy benchmark or causal proof.'}
