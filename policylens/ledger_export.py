"""Strict finite archive reproduction."""
import json
import math
from pathlib import Path
from .ledger_fixtures import build


def compare(a,b,path='root'):
    if isinstance(a,bool) or isinstance(b,bool) or a is None or b is None or isinstance(a,str) or isinstance(b,str):
        if type(a)!=type(b) or a!=b:raise ValueError(path)
    elif type(a) is int or type(b) is int:
        if type(a) is not int or type(b) is not int or a!=b:raise ValueError(path)
    elif isinstance(a,float) and isinstance(b,float):
        if not math.isfinite(a) or not math.isfinite(b):raise ValueError(path)
        if type(a)==int and type(b)==int:
            if a!=b:raise ValueError(path)
        elif not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12):raise ValueError(path)
    elif isinstance(a,dict) and isinstance(b,dict):
        if set(a)!=set(b):raise ValueError(path)
        for k in a:compare(a[k],b[k],path+'.'+k)
    elif isinstance(a,list) and isinstance(b,list):
        if len(a)!=len(b):raise ValueError(path)
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'.'+str(i))
    else:raise ValueError(path)


def load():
    root=Path(__file__).parent/'artifacts'
    result=json.loads((root/'ledger-meta.json').read_text())
    result['witnesses']=[]
    for c in result['cases']:
        for p in sorted(root.glob('ledger-'+c['id']+'-*.json')):result['witnesses'].extend(json.loads(p.read_text()))
    return result


def main():
    compare(build(),load())
    print('95 edit witnesses and original demo reproduce')


if __name__=='__main__':main()
