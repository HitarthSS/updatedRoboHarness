import unittest
import numpy as np
from protocol import Budget, AdaptiveGovernor, InterventionState, libero_policy_image


class ProtocolTests(unittest.TestCase):
    def test_camera_rotation_matches_official_environment_processor(self):
        try:
            import torch
            from lerobot.processor.env_processor import LiberoProcessorStep
        except ImportError:
            self.skipTest('Run this contract check in the isolated LeRobot environment')
        image = np.arange(4*5*3, dtype=np.uint8).reshape(4,5,3)
        batch = torch.from_numpy(image).permute(2,0,1)[None]
        processed = LiberoProcessorStep().observation({'observation.images.image': batch})
        expected = processed['observation.images.image'][0].permute(1,2,0).numpy()
        np.testing.assert_array_equal(libero_policy_image(image), expected)

    def test_highlight_instruction_expires_with_its_modified_image_chunk(self):
        state = InterventionState('pick the black bowl')
        state.update(refined='pick the highlighted green bowl', image_modified=True)
        state.commit(state.task, queued_actions=9)
        self.assertEqual(state.task, 'pick the highlighted green bowl')
        state.commit(state.task, queued_actions=0)
        self.assertEqual(state.task, 'pick the black bowl')
        self.assertFalse(state.needs_replan(state.task))

    def test_intervention_invalidates_cached_actions_and_retains_task(self):
        state = InterventionState('pick the bowl')
        self.assertFalse(state.needs_replan(state.task))
        state.commit(state.task)
        state.update(refined='pick the left bowl')
        self.assertTrue(state.needs_replan(state.task))
        state.commit(state.task)
        self.assertEqual(state.task, 'pick the left bowl')
        self.assertFalse(state.needs_replan(state.task))
        state.update(image_modified=True)
        self.assertTrue(state.needs_replan(state.task))

    def test_recovery_requires_fresh_chunk_without_a_task_change(self):
        state = InterventionState('pick the bowl')
        state.commit(state.task)
        state.update(recovering=True)
        state.update()
        self.assertTrue(state.needs_replan(state.task))
        state.commit(state.task)
        self.assertFalse(state.needs_replan(state.task))

    def test_recovery_cannot_extend_total_horizon(self):
        b = Budget(220)
        for i in range(220):
            b.charge(recovery=i>=200)
        self.assertEqual((b.steps, b.recovery_steps, b.remaining), (220,20,0))
        with self.assertRaises(RuntimeError):
            b.charge()

    def test_recovery_uses_recent_actions_preserves_gripper_and_reserves_policy_budget(self):
        g = AdaptiveGovernor()
        for _ in range(30):
            g.record([1,0,0,0,0,0,1])
        actions = g.recovery(50, Budget(220, steps=50))
        self.assertEqual(len(actions), 15)
        self.assertTrue(all(a[-1] == 1 for a in actions))
        self.assertTrue(all(np.max(np.abs(a)) <= 1 for a in actions))
        self.assertEqual(g.recovery(55, Budget(220, steps=55)), [])
        for _ in range(5):
            g.record([.1,0,0,0,0,0,-1])
        self.assertEqual(g.recovery(210, Budget(220, steps=210)), [])

    def test_changing_scene_calls_less_often_but_stalled_scene_still_monitored(self):
        g = AdaptiveGovernor()
        moving = [i for i in range(100) if g.should_perceive(i,np.full((16,16,3), (i%2)*255))]
        self.assertEqual(moving, [0,30,60,90])
        g = AdaptiveGovernor()
        stalled = [i for i in range(41) if g.should_perceive(i,np.zeros((16,16,3)))]
        self.assertEqual(stalled, [0,10,20,30,40])


if __name__ == '__main__':
    unittest.main()
