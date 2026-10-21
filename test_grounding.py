"""Geometry recovery rejects invalid boxes and masks inconsistent with grounding."""
from unittest import TestCase
import numpy as np
from grounding import grounded_box_mask
from run_episode import AuditedPost

class Predictor:
    def __init__(self,masks,scores):
        self.masks,self.scores=masks,scores
        self.calls=0
    def reset_all_prompts(self,state):
        state.pop('stale_text',None)
    def add_geometric_prompt(self,box,label,state):
        self.calls+=1
        assert label and all(0<=v<=1 for v in box)
        assert 'stale_text' not in state
        return dict(masks=self.masks,scores=self.scores)

class GroundingTests(TestCase):
    def test_uses_existing_box_and_rejects_wrong_and_scene_filling_masks(self):
        target=np.zeros((20,20));target[2:6,2:6]=1
        wrong=np.zeros((20,20));wrong[12:16,12:16]=1
        predictor=Predictor(np.stack([wrong,np.ones((20,20)),target]),[.99,.99,.8])
        trace={}
        mask,score=grounded_box_mask(predictor,{'stale_text':True},[2,2,6,6],(20,20),.25,trace)
        np.testing.assert_array_equal(mask,target)
        self.assertEqual(predictor.calls,1)
        self.assertTrue(trace['accepted'])
        self.assertEqual(score,.8)

    def test_invalid_or_empty_boxes_do_not_query_model(self):
        for box in [[1,1,1,4],[30,30,40,40],[float('nan'),0,2,3]]:
            predictor=Predictor([],[])
            mask,_=grounded_box_mask(predictor,{},box,(20,20),.25,{})
            self.assertFalse(mask.any())
            self.assertEqual(predictor.calls,0)

    def test_grounding_switch_is_candidate_only_and_does_not_mutate_input(self):
        seen=[]
        def post(path,payload):
            seen.append(payload)
            return {'success':True,'grounding_fallback':{'attempted':True,'accepted':True}}
        payload={'image':'development-frame'}
        baseline=AuditedPost(post)
        baseline('/visual_overlay',payload)
        candidate=AuditedPost(post,adaptive_grounding=True)
        candidate('/visual_overlay',payload)
        self.assertNotIn('adaptive_grounding',seen[0])
        self.assertTrue(seen[1]['adaptive_grounding'])
        self.assertNotIn('adaptive_grounding',payload)
        self.assertEqual(candidate.grounding_fallback_successes,1)
