import copy
import io
import json
import math
import unittest
from contextlib import redirect_stdout,redirect_stderr
from unittest.mock import patch
from policylens import cross_fit,evaluate
from policylens.cli import main


def fixture():
    return {'action_names':['a','b'],'rows':[
        {'context':'x' if i<8 else 'y','action':i%2,'reward':float(i%3==0),
         'behavior':[.5,.5],'target':[.2,.8]} for i in range(12)]}


class TestCrossFit(unittest.TestCase):
    def test_manual_excluded_fold_means_counts(self):
        p=fixture();out=cross_fit(p,folds=3,seed=19,prior_mean=.3,prior_strength=2)
        ids=out['cross_fit']['fold_assignment']
        self.assertEqual(sorted(out['cross_fit']['fold_sizes']),[4,4,4])
        for i,row in enumerate(p['rows']):
            for a in range(2):
                rewards=[r['reward'] for j,r in enumerate(p['rows'])
                         if ids[j]!=ids[i] and r['context']==row['context'] and r['action']==a]
                self.assertEqual(out['cross_fit']['training_counts'][i][a],len(rewards))
                self.assertAlmostEqual(out['prepared_input']['rows'][i]['reward_model'][a],
                                       (sum(rewards)+.6)/(len(rewards)+2))
        evaluate(out['prepared_input'])

    def test_entire_heldout_fold_rewards_cannot_change_its_predictions(self):
        p=fixture();base=cross_fit(p,folds=3,seed=4)
        ids=base['cross_fit']['fold_assignment']
        for fold in range(3):
            changed=copy.deepcopy(p)
            for i,r in enumerate(changed['rows']):
                if ids[i]==fold:r['reward']=1-r['reward']
            out=cross_fit(changed,folds=3,seed=4)
            for i in range(len(ids)):
                if ids[i]==fold:self.assertEqual(out['prepared_input']['rows'][i]['reward_model'],
                                                base['prepared_input']['rows'][i]['reward_model'])

    def test_unique_contexts_prior_fallback_is_explicit(self):
        p=fixture()
        for i,r in enumerate(p['rows']):r['context']=str(i)
        out=cross_fit(p,prior_mean=.7)
        self.assertEqual(out['cross_fit']['prior_only_predictions'],24)
        self.assertEqual(out['cross_fit']['target_positive_prior_only_predictions'],24)
        for r,c in zip(out['prepared_input']['rows'],out['cross_fit']['training_counts']):
            self.assertEqual(r['reward_model'],[.7,.7]);self.assertEqual(c,[0,0])

    def test_determinism_order_and_no_mutation_aliasing(self):
        p=fixture();before=copy.deepcopy(p);out=cross_fit(p,seed=22)
        self.assertEqual(out,cross_fit(p,seed=22));self.assertEqual(p,before)
        for i,r in enumerate(out['prepared_input']['rows']):
            self.assertEqual(r['action'],p['rows'][i]['action'])
        out['prepared_input']['rows'][0]['target'][0]=0
        out['prepared_input']['action_names'][0]='changed'
        self.assertEqual(p,before)

    def test_nondivisible_folds_and_two_rows(self):
        for n,k in [(2,2),(11,3),(12,5)]:
            p=fixture();p['rows']=p['rows'][:n];out=cross_fit(p,folds=k)
            sizes=out['cross_fit']['fold_sizes'];self.assertLessEqual(max(sizes)-min(sizes),1)
            self.assertEqual(sum(sizes),n)

    def test_invalid_parameters(self):
        for kw in [{'folds':True},{'folds':1},{'folds':13},{'folds':2.5},
                   {'seed':True},{'seed':-1},{'seed':2**32},
                   {'prior_mean':False},{'prior_mean':math.nan},{'prior_mean':1.1},
                   {'prior_strength':0},{'prior_strength':math.inf},{'prior_strength':1e7}]:
            with self.subTest(kw=kw),self.assertRaises(ValueError):cross_fit(fixture(),**kw)

    def test_invalid_rows_no_model_or_extra_fields(self):
        for field,value in [('context',''),('context',1),('context','x'*1025),
                            ('action',True),('reward',2),('behavior',[0,1])]:
            p=fixture();p['rows'][0][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):cross_fit(p)
        p=fixture();p['rows'][0]['reward_model']=[.5,.5]
        with self.assertRaises(ValueError):cross_fit(p)

    def test_support_violation_remains_visible_after_model_fit(self):
        p=fixture()
        for r in p['rows']:r['behavior']=[1,0] if r['action']==0 else [0,1]
        out=cross_fit(p);report=evaluate(out['prepared_input'])
        self.assertEqual(report['support']['rows_with_unsupported_target_actions'],12)
        self.assertEqual(report['interpretation'],'support_violation_no_nonparametric_identification')

    def test_cli_crossfit_roundtrip_and_clipping(self):
        out=io.StringIO()
        with patch('sys.stdin',io.StringIO(json.dumps(fixture()))),redirect_stdout(out):
            code=main(['crossfit','--input','-','--folds','3','--seed','4','--clip','1'])
        self.assertEqual(code,0);result=json.loads(out.getvalue())
        self.assertEqual(result['evaluation'],evaluate(result['prepared_input'],clip=1))
        self.assertEqual(result['cross_fit']['folds'],3)

    def test_cli_invalid_no_partial_output(self):
        out,err=io.StringIO(),io.StringIO()
        with patch('sys.stdin',io.StringIO('{}')),redirect_stdout(out),redirect_stderr(err):
            code=main(['crossfit','--input','-'])
        self.assertEqual(code,2);self.assertEqual(out.getvalue(),'');self.assertIn('input error',err.getvalue())

    def test_tiny_prior_strength_preserves_prior_only_prediction(self):
        p=fixture()
        for i,r in enumerate(p['rows']):r['context']=str(i)
        out=cross_fit(p,prior_mean=.3,prior_strength=5e-324)
        for row in out['prepared_input']['rows']:
            self.assertEqual(row['reward_model'],[.3,.3])

    def test_demo_crossfit_is_not_oracle_or_population_truth(self):
        from policylens.demo import run_demo
        d=run_demo();cf=d['cross_fitted_seeded_log']
        self.assertAlmostEqual(cf['doubly_robust'],.7509088189421467)
        self.assertNotAlmostEqual(cf['doubly_robust'],d['known_target_value'])
        self.assertNotAlmostEqual(cf['doubly_robust'],d['seeded_log']['doubly_robust'])
        self.assertEqual(d['cross_fit_diagnostics']['prior_only_predictions'],0)
