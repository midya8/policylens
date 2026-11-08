"""Public synthetic fixtures, retaining the original balanced design exactly."""
from copy import deepcopy
from .ledger import audit
from .demo import run_demo


def fixtures():
    rows=[]
    for kind in [0,1]:
        b=[.8,.2] if kind==0 else [.2,.8]
        t=[.1,.9] if kind==0 else [.9,.1]
        truth=[.2,.8] if kind==0 else [.9,.1]
        for a,count in enumerate([8,2] if kind==0 else [2,8]):
            for _ in range(count):rows.append(dict(action=a,reward=truth[a],behavior=b[:],target=t[:],reward_model=[.5,.5]))
    def row(a,r,b,t,q):return dict(action=a,reward=r,behavior=b,target=t,reward_model=q)
    return [
        {'id':'balanced','title':'Original balanced design','note':'20 deterministic mean-reward rows, exactly matching behavior frequencies. Known target 0.78 only for the unedited design; no random-trial claim.','truth':.78,'input':{'action_names':['a','b'],'rows':rows}},
        {'id':'unsupported','title':'Unobserved supported-by-model action','note':'Target puts positive mass on an action with zero behavior support. Numeric direct/DR output does not identify its reward nonparametrically.','truth':None,'input':{'action_names':['a','b'],'rows':[row(0,.2,[1.,0.],[.25,.75],[.2,.9]),row(0,.4,[1.,0.],[.25,.75],[.4,.9])]}},
        {'id':'zero_mass','title':'Zero observed target mass','note':'All observed target weights are zero. SNIPS is undefined; direct and DR still return numbers. The full behavior vectors have support.','truth':None,'input':{'action_names':['a','b'],'rows':[row(0,1.,[.5,.5],[0.,1.],[.3,.7]),row(0,0.,[.5,.5],[0.,1.],[.6,.2])]}},
        {'id':'rare_action','title':'One large weight, opposite residuals','note':'A selected rare-action row dominates weight mass; negative DR correction is allowed. Higher ESS after clipping need not improve value accuracy. No population truth specified.','truth':None,'input':{'action_names':['a','b'],'rows':[row(1,0.,[.99,.01],[.1,.9],[.4,.9]),row(0,1.,[.99,.01],[.1,.9],[.3,.8]),row(0,.5,[.99,.01],[.1,.9],[.4,.7]),row(0,0.,[.99,.01],[.1,.9],[.2,.6])]}},
        {'id':'on_policy','title':'On-policy arithmetic control','note':'Original target equals behavior, so raw weights are one. Clipping below one changes IPS/DR, although SNIPS retains the reward mean.','truth':None,'input':{'action_names':['a','b'],'rows':[row(0,1.,[.3,.7],[.3,.7],[.2,.9]),row(1,0.,[.3,.7],[.3,.7],[.8,.4]),row(1,.5,[.3,.7],[.3,.7],[.6,.1])]}}
    ]


def edited(case, mix=1., row_index=0, reward=None):
    if type(mix) not in (int,float) or not 0 <= mix <= 1:raise ValueError('mix must be in [0,1]')
    p=deepcopy(case['input'])
    for r in p['rows']:r['target']=[(1-mix)*b+mix*t for b,t in zip(r['behavior'],r['target'])]
    if type(row_index) is not int or not 0 <= row_index < len(p['rows']):raise ValueError('invalid row index')
    if reward is not None:
        if type(reward) not in (int,float) or not 0 <= reward <= 1:raise ValueError('reward must be in [0,1]')
        p['rows'][row_index]['reward']=reward
    return p


def build():
    cases=fixtures(); witnesses=[]
    for c in cases:
        for mix in [0.,.25,.5,1.]:
            for cap in [None,.25,1.,4.]:
                witnesses.append({'case':c['id'],'mix':mix,'cap':cap,'row':0,'reward':None,'audit':audit(edited(c,mix),cap)})
        for r in [0.,.5,1.]:
            witnesses.append({'case':c['id'],'mix':1.,'cap':1.,'row':len(c['input']['rows'])-1,'reward':r,'audit':audit(edited(c,1.,len(c['input']['rows'])-1,r),1.)})
    return {'schema':'policylens-ledger-v1','cases':cases,'witnesses':witnesses,'original_demo':run_demo()}
