import io
import json
import unittest
from contextlib import redirect_stdout,redirect_stderr
from unittest.mock import patch
from policylens.cli import main


class TestCLI(unittest.TestCase):
    def test_stdin_json(self):
        d={'action_names':['a','b'],'rows':[{'action':0,'reward':1,'behavior':[.5,.5],
                                           'target':[.5,.5],'reward_model':[.5,.5]}]}
        out=io.StringIO()
        with patch('sys.stdin',io.StringIO(json.dumps(d))),redirect_stdout(out):
            self.assertEqual(main(['evaluate','--input','-']),0)
        self.assertEqual(json.loads(out.getvalue())['ips'],1)

    def test_invalid_no_partial_report(self):
        for body in ['{','{}']:
            out,err=io.StringIO(),io.StringIO()
            with patch('sys.stdin',io.StringIO(body)),redirect_stdout(out),redirect_stderr(err):
                self.assertEqual(main(['evaluate','--input','-']),2)
            self.assertEqual(out.getvalue(),'');self.assertIn('input error',err.getvalue())
