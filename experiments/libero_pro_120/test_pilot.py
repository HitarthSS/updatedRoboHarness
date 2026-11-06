"""Failure classification and paired random-number invariants."""
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import pilot_common as c
sys.path.insert(0,str(c.ROOT/'runtime'))
from run_episode import AuditedCompletions
from meter_proxy import seed_for


class PilotTests(unittest.TestCase):
    def response(self,text):
        return SimpleNamespace(model='qwen3-vl-32b-instruct',
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    def test_malformed_model_reply_uses_native_fallback(self):
        response=self.response('not valid JSON')
        original=Mock();original.create.return_value=response
        audit=AuditedCompletions(original)
        self.assertIs(audit.create(model='qwen3-vl-32b-instruct'),response)
        self.assertEqual((audit.calls,audit.errors,audit.parse_failures),(1,0,1))
        from soma_vlm import _extract_json
        self.assertIsNone(_extract_json(response.choices[0].message.content))
        original.create.assert_called_once()

    def test_transport_error_remains_fatal(self):
        original=Mock();original.create.side_effect=TimeoutError('transport timeout')
        audit=AuditedCompletions(original)
        with self.assertRaises(TimeoutError):audit.create()
        self.assertEqual((audit.calls,audit.errors,audit.parse_failures),(1,1,0))

    def test_wrong_model_is_rejected(self):
        response=self.response('{}');response.model='some-other-model'
        original=Mock();original.create.return_value=response
        audit=AuditedCompletions(original)
        with self.assertRaises(RuntimeError):audit.create()
        self.assertEqual(audit.errors,1)

    def test_case_seeds_match_across_arms(self):
        a='test/baseline/libero_spatial_swap-t0-s100'
        b='test/improved/libero_spatial_swap-t0-s100'
        self.assertEqual(seed_for(a,1),seed_for(b,1))
        self.assertNotEqual(seed_for(a,1),seed_for(a,2))
        self.assertEqual(len(c.CONFIG['state_indices'])*c.CONFIG['conditions'],120)


if __name__=='__main__':unittest.main()
