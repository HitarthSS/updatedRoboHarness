"""Regression check for the released no-op loop; no network or model call."""
from pathlib import Path
import sys
import unittest
import numpy as np
from repair_release import prepare
sys.path.insert(0,str(prepare()))
from soma_perception import PerceptionModule


class FakeVLM:
    def __init__(self): self.calls=0
    def orchestrate_perception(self, **kwargs):
        self.calls+=1
        return dict(tool_chain=[],refined_task=kwargs['task_desc'])


class ReleaseLoopTest(unittest.TestCase):
    def test_one_real_planning_attempt_is_enabled(self):
        vlm=FakeVLM()
        perception=PerceptionModule(vlm_client=vlm)
        original=np.zeros((32,32,3),dtype=np.uint8)
        frame,task,flags=perception.process_frame(original,'put the bowl on the plate',0,{})
        self.assertEqual(vlm.calls,1)
        np.testing.assert_array_equal(frame,original)
        self.assertEqual(task,'put the bowl on the plate')
        self.assertFalse(flags['encore'])


if __name__=='__main__': unittest.main()
