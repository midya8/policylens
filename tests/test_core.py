import copy
import math
import unittest
from policylens import evaluate
from policylens.demo import run_demo


def data():
    return {'action_names':['a','b'],'rows':[
        {'action':0,'reward':1.,'behavior':[.5,.5],'target':[.25,.75],'reward_model':[.6,.4]},
        {'action':1,'reward':0.,'behavior':[.5,.5],'target':[.25,.75],'reward_model':[.6,.4]}]}


class TestEstimators(unittest.TestCase):
    def test_hand_calculated_estimates(self):
        r=evaluate(data())
        self.assertEqual(r['ips'],.25)
        self.assertEqual(r['snips'],.25)
        self.assertAlmostEqual(r['direct_method'],.45)
        self.assertAlmostEqual(r['doubly_robust'],.25)
        self.assertAlmostEqual(r['weights']['effective_sample_size'],1.6)

    def test_on_policy_ips_snips_equal_reward_mean(self):
        d=data()
        for row in d['rows']:row['target']=list(row['behavior'])
        r=evaluate(d)
        self.assertEqual(r['ips'],.5);self.assertEqual(r['snips'],.5)
        self.assertEqual(r['weights']['effective_sample_size'],2)

    def test_zero_reward_model_dr_is_ips(self):
        d=data()
        for row in d['rows']:row['reward_model']=[0.,0.]
        r=evaluate(d);self.assertEqual(r['ips'],r['doubly_robust'])

    def test_exact_observed_reward_model_dr_is_direct(self):
        d=data();d['rows'][0]['reward_model'][0]=1;d['rows'][1]['reward_model'][1]=0
        r=evaluate(d);self.assertEqual(r['doubly_robust'],r['direct_method'])

    def test_support_violation_not_laundered_by_reward_model(self):
        d=data()
        for row in d['rows']:row.update(action=0,behavior=[1.,0.],target=[0.,1.])
        r=evaluate(d)
        self.assertEqual(r['support']['rows_with_unsupported_target_actions'],2)
        self.assertIsNone(r['snips']);self.assertEqual(r['weights']['effective_sample_size'],0)
        self.assertIn('no_nonparametric_identification',r['interpretation'])

    def test_observed_action_zero_behavior_rejected(self):
        d=data();d['rows'][0]['behavior']=[0,1]
        with self.assertRaises(ValueError):evaluate(d)

    def test_reward_and_model_validation(self):
        for value in [-1,1.1,True,float('inf'),float('nan'),'1']:
            for key in ['reward','reward_model']:
                d=data()
                if key=='reward':d['rows'][0][key]=value
                else:d['rows'][0][key][0]=value
                with self.subTest(value=value,key=key),self.assertRaises(ValueError):evaluate(d)

    def test_probability_and_action_validation(self):
        for value in [[.4,.4],[True,0],[float('nan'),.5],[1],[-1,2]]:
            for key in ['behavior','target']:
                d=data();d['rows'][0][key]=value
                with self.subTest(value=value,key=key),self.assertRaises(ValueError):evaluate(d)
        for value in [True,-1,2,'0']:
            d=data();d['rows'][0]['action']=value
            with self.assertRaises(ValueError):evaluate(d)

    def test_schema_and_names(self):
        for value in [{},{'rows':[]},data()]:
            if 'action_names' in value:value['action_names']=[[],[]]
            with self.assertRaises(ValueError):evaluate(value)
        d=data();d['rows'][0]['extra']=1
        with self.assertRaises(ValueError):evaluate(d)

    def test_clipping_explicit_and_no_mutation(self):
        d=data();saved=copy.deepcopy(d);r=evaluate(d,clip=1)
        self.assertEqual(d,saved);self.assertEqual(r['clip'],1)
        self.assertIn('clipped',r['interpretation'])
        self.assertNotEqual(r['doubly_robust'],evaluate(d)['doubly_robust'])
        for value in [0,-1,True,float('inf')]:
            with self.assertRaises(ValueError):evaluate(d,clip=value)

    def test_extreme_weights_ess_does_not_square_raw_weights(self):
        d=data();d['rows'][0]['behavior']=[1e-300,1.]
        r=evaluate(d)
        self.assertTrue(math.isfinite(r['weights']['effective_sample_size']))
        self.assertAlmostEqual(r['weights']['effective_sample_size'],1.)
        self.assertGreater(r['ips'],1.) # Estimates are not silently clipped to reward range.

    def test_infinite_importance_weight_fails(self):
        d=data();d['rows'][0]['behavior']=[5e-324,1.]
        with self.assertRaises(ValueError):evaluate(d)

    def test_balanced_design_and_clipping_bias_example(self):
        r=run_demo();a=r['balanced_design'];b=r['clipped_balanced_design']
        for key in ['ips','snips','doubly_robust']:self.assertAlmostEqual(a[key],.78)
        self.assertEqual(a['direct_method'],.5)
        self.assertEqual(b['ips'],.185);self.assertEqual(b['doubly_robust'],.535)
        self.assertLess(a['weights']['effective_sample_size'],b['weights']['effective_sample_size'])

    def test_seeded_fixture_is_not_exact_truth(self):
        r=run_demo()['seeded_log'];self.assertAlmostEqual(r['doubly_robust'],.7532475)
        self.assertNotEqual(r['doubly_robust'],.78)
