import argparse
import json
import sys
from .core import evaluate
from .crossfit import cross_fit
from .demo import run_demo
from .ledger import audit
from .ledger_fixtures import build


def unique_pairs(pairs):
    out={}
    for k,v in pairs:
        if k in out:raise ValueError('duplicate JSON key: '+k)
        out[k]=v
    return out


def main(argv=None):
    parser=argparse.ArgumentParser(prog='policylens')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('demo');sub.add_parser('ledger-demo');a=sub.add_parser('audit');a.add_argument('--input',required=True);a.add_argument('--clip',type=float);p=sub.add_parser('evaluate')
    p.add_argument('--input',required=True);p.add_argument('--clip',type=float)
    c=sub.add_parser('crossfit')
    c.add_argument('--input',required=True);c.add_argument('--clip',type=float)
    c.add_argument('--folds',type=int,default=5);c.add_argument('--seed',type=int,default=0)
    c.add_argument('--prior-mean',type=float,default=.5)
    c.add_argument('--prior-strength',type=float,default=1.)
    args=parser.parse_args(argv)
    try:
        if args.command=='demo':result=run_demo()
        elif args.command=='ledger-demo':result=build()
        else:
            if args.input=='-':payload=json.load(sys.stdin,object_pairs_hook=unique_pairs)
            else:
                with open(args.input,encoding='utf-8') as source:payload=json.load(source,object_pairs_hook=unique_pairs)
            if args.command=='crossfit':
                result=cross_fit(payload,args.folds,args.seed,args.prior_mean,args.prior_strength)
                result['evaluation']=evaluate(result['prepared_input'],args.clip)
            elif args.command=='audit':result=audit(payload,args.clip)
            else:result=evaluate(payload,args.clip)
        body=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)
    except (ValueError,TypeError,OSError,OverflowError) as exc:
        print('policylens input error: '+str(exc),file=sys.stderr);return 2
    print(body);return 0
