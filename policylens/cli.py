import argparse
import json
import sys
from .core import evaluate
from .demo import run_demo


def main(argv=None):
    parser=argparse.ArgumentParser(prog='policylens')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('demo');p=sub.add_parser('evaluate')
    p.add_argument('--input',required=True);p.add_argument('--clip',type=float)
    args=parser.parse_args(argv)
    try:
        if args.command=='demo':result=run_demo()
        else:
            if args.input=='-':payload=json.load(sys.stdin)
            else:
                with open(args.input,encoding='utf-8') as source:payload=json.load(source)
            result=evaluate(payload,args.clip)
        body=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)
    except (ValueError,TypeError,OSError,OverflowError) as exc:
        print('policylens input error: '+str(exc),file=sys.stderr);return 2
    print(body);return 0
