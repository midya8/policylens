import copy
from fractions import Fraction as F
import json
import math
import random
import subprocess
import sys
import unittest
from policylens.ledger import audit
from policylens.ledger_fixtures import fixtures, edited, build
from policylens.ledger_export import compare, load
from policylens.demo import run_demo


class TestLedger(unittest.TestCase):
    def test_archives_and_original_reports(self):
        b=build();compare(b,load());self.assertEqual(len(b['witnesses']),95)
        self.assertEqual(b['original_demo'],run_demo())
        self.assertEqual(audit(fixtures()[0]['input'])['raw'],run_demo()['balanced_design'])
        self.assertEqual(audit(fixtures()[0]['input'],1)['selected'],run_demo()['clipped_balanced_design'])

    def test_row_sums_and_loo(self):
        for case in fixtures():
            p=case['input'];before=copy.deepcopy(p)
            for cap in [None,.25,1.,4.]:
                x=audit(p,cap)
                for field,key in [('ips_contribution','ips'),('direct_contribution','direct_method'),('dr_contribution','doubly_robust')]:
                    self.assertAlmostEqual(math.fsum(r[field] for r in x['rows']),x['selected'][key])
                if x['selected']['snips'] is None:self.assertTrue(all(r['snips_contribution'] is None for r in x['rows']))
                else:self.assertAlmostEqual(math.fsum(r['snips_contribution'] for r in x['rows']),x['selected']['snips'])
                for field,key in [('ips_removed_by_cap','ips'),('dr_removed_by_cap','doubly_robust')]:
                    self.assertAlmostEqual(math.fsum(r[field] for r in x['rows']),x['raw'][key]-x['selected'][key])
                for v in x['leave_one_row_out']:
                    other=audit({'action_names':p['action_names'],'rows':[r for i,r in enumerate(p['rows']) if i!=v['removed_row']]},cap)['selected']
                    for k in ['ips','snips','direct_method','doubly_robust']:self.assertEqual(v[k],other[k])
            self.assertEqual(p,before)

    def test_independent_fraction_oracle_160_games(self):
        rng=random.Random(71)
        for k in range(160):
            rows=[];num=[];dm=[];dr=[];ws=[]
            cap=None if k%3==0 else F((k%5)+1,4)
            for i in range(1+k%7):
                b=F(rng.randrange(1,8),8);t=F(rng.randrange(9),8);a=rng.randrange(2)
                r=F(rng.randrange(9),8);q=[F(rng.randrange(9),8),F(rng.randrange(9),8)]
                bvec=[b,1-b];tv=[t,1-t];w=tv[a]/bvec[a];w=min(w,cap) if cap is not None else w
                d=sum(tv[j]*q[j] for j in range(2));ws.append(w);num.append(w*r);dm.append(d);dr.append(d+w*(r-q[a]))
                rows.append(dict(action=a,reward=float(r),behavior=list(map(float,bvec)),target=list(map(float,tv)),reward_model=list(map(float,q))))
            x=audit(dict(action_names=['a','b'],rows=rows),float(cap) if cap is not None else None)['selected'];n=len(rows)
            for key,value in [('ips',sum(num)/n),('direct_method',sum(dm)/n),('doubly_robust',sum(dr)/n)]:self.assertAlmostEqual(x[key],float(value),places=12)
            if sum(ws):
                self.assertAlmostEqual(x['snips'],float(sum(num)/sum(ws)),places=12)
                self.assertAlmostEqual(x['weights']['effective_sample_size'],float(sum(ws)**2/sum(w*w for w in ws)),places=12)
            else:self.assertIsNone(x['snips'])

    def test_cap_piecewise_affine_not_snips_or_ess(self):
        p=fixtures()[0]['input']
        a,b,m=[audit(p,c)['selected'] for c in [.5,1.,.75]]
        for k in ['ips','doubly_robust']:self.assertAlmostEqual(m[k],(a[k]+b[k])/2)
        self.assertNotAlmostEqual(m['snips'],(a['snips']+b['snips'])/2)
        self.assertNotAlmostEqual(m['weights']['effective_sample_size'],(a['weights']['effective_sample_size']+b['weights']['effective_sample_size'])/2)

    def test_support_zero_and_negative_controls(self):
        cases=fixtures();x=audit(cases[1]['input']);self.assertEqual([r['unsupported_actions'] for r in x['rows']],[[1],[1]])
        self.assertFalse(x['selected']['support']['observed_context_support_complete'])
        z=audit(cases[2]['input']);self.assertIsNone(z['selected']['snips']);self.assertEqual(z['selected']['weights']['effective_sample_size'],0)
        self.assertLess(audit(cases[3]['input'])['selected']['doubly_robust'],0)
        self.assertTrue(audit(edited(cases[1],0.))['selected']['support']['observed_context_support_complete'])

    def test_bounds_invalid_and_single_row(self):
        p=fixtures()[0]['input']
        for cap in [0,-1,True,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):audit(p,cap)
        for value in [[],[p['rows'][0]]*257]:
            with self.assertRaises(ValueError):audit(dict(action_names=['a','b'],rows=value))
        one=audit(dict(action_names=['a','b'],rows=p['rows'][:1]));self.assertEqual(one['leave_one_row_out'],[])
        for mix in [-1,1.1,True,float('nan')]:
            with self.assertRaises(ValueError):edited(fixtures()[0],mix)

    def test_strict_archive_comparison(self):
        for x,y in [(True,1),(1,1.),(None,0),({'a':1},{'a':1,'b':2}),([1],[1,2]),(float('nan'),0.),(1.,1.001)]:
            with self.assertRaises(ValueError):compare(x,y)
        compare({'x':[1.,None,True]}, {'x':[1.+1e-13,None,True]})

    def test_cli_and_duplicate_rejection(self):
        p=subprocess.run([sys.executable,'-m','policylens','audit','--input','examples/log.json','--clip','1'],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr);self.assertIn('rows',json.loads(p.stdout))
        p=subprocess.run([sys.executable,'-m','policylens','audit','--input','-'],input='{"rows":[],"rows":[]}',capture_output=True,text=True)
        self.assertEqual(p.returncode,2);self.assertEqual(p.stdout,'');self.assertIn('duplicate',p.stderr)
