"""Finite reward-coordinate identities, not statistical identification checks."""
import copy
import unittest

from policylens import evaluate
from policylens.ledger_fixtures import edited, fixtures


class RewardCoordinateTests(unittest.TestCase):
    def close(self, actual, expected):
        self.assertAlmostEqual(actual, expected, delta=2e-12 * max(1, abs(expected)))

    def controls(self):
        for case in fixtures():
            for mix in (0., .25, .5, 1.):
                for cap in (None, .25, 1., 4.):
                    yield case['id'], mix, cap, edited(case, mix)

    def test_joint_reward_and_model_affine_coordinates(self):
        for case, mix, cap, payload in self.controls():
            original = copy.deepcopy(payload)
            report = evaluate(payload, cap)
            for scale, offset in ((.5, .25), (-.5, .75), (0., .5)):
                changed = copy.deepcopy(payload)
                for row in changed['rows']:
                    row['reward'] = scale * row['reward'] + offset
                    row['reward_model'] = [scale * q + offset for q in row['reward_model']]
                frozen = copy.deepcopy(changed)
                with self.subTest(case=case, mix=mix, cap=cap, scale=scale):
                    result = evaluate(changed, cap)
                    self.close(result['ips'], scale * report['ips'] + offset * report['weights']['mean'])
                    if report['snips'] is None:
                        self.assertIsNone(result['snips'])
                    else:
                        self.close(result['snips'], scale * report['snips'] + offset)
                    for key in ('direct_method', 'doubly_robust'):
                        self.close(result[key], scale * report[key] + offset)
                    for key in ('n', 'clip', 'weights', 'support', 'interpretation', 'scope'):
                        self.assertEqual(result[key], report[key])
                    self.assertEqual(changed, frozen)
            self.assertEqual(payload, original)

    def test_reward_only_affine_coordinates(self):
        for case, mix, cap, payload in self.controls():
            report = evaluate(payload, cap)
            zero = copy.deepcopy(payload)
            for row in zero['rows']:
                row['reward'] = 0.
            baseline = evaluate(zero, cap)
            for scale, offset in ((.5, .25), (-.5, .75), (0., .5)):
                changed = copy.deepcopy(payload)
                for row in changed['rows']:
                    row['reward'] = scale * row['reward'] + offset
                with self.subTest(case=case, mix=mix, cap=cap, scale=scale):
                    result = evaluate(changed, cap)
                    self.close(result['ips'], scale * report['ips'] + offset * report['weights']['mean'])
                    self.close(result['doubly_robust'], baseline['doubly_robust'] +
                               scale * report['ips'] + offset * report['weights']['mean'])
                    self.assertEqual(result['direct_method'], report['direct_method'])
                    self.assertEqual(result['weights'], report['weights'])
                    self.assertEqual(result['support'], report['support'])

    def test_reward_complement_with_weights_not_forced_to_one(self):
        saw_zero = saw_nonunit = saw_clipped = saw_unsupported = False
        for case, mix, cap, payload in self.controls():
            report = evaluate(payload, cap)
            for row in payload['rows']:
                row['reward'] = 1 - row['reward']
                row['reward_model'] = [1 - q for q in row['reward_model']]
            result = evaluate(payload, cap)
            with self.subTest(case=case, mix=mix, cap=cap):
                self.close(result['ips'] + report['ips'], report['weights']['mean'])
                self.close(result['direct_method'] + report['direct_method'], 1.)
                self.close(result['doubly_robust'] + report['doubly_robust'], 1.)
                if report['snips'] is None:
                    self.assertIsNone(result['snips'])
                    saw_zero = True
                else:
                    self.close(result['snips'] + report['snips'], 1.)
                self.assertEqual(result['support'], report['support'])
                self.assertEqual(result['interpretation'], report['interpretation'])
                saw_nonunit |= abs(report['weights']['mean'] - 1) > .01
                saw_clipped |= cap is not None
                saw_unsupported |= not report['support']['observed_context_support_complete']
        self.assertTrue(saw_zero and saw_nonunit and saw_clipped and saw_unsupported)


if __name__ == '__main__':
    unittest.main()
