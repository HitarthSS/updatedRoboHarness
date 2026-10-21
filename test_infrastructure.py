"""Ensure actual tool failures cannot become apparently valid harness runs."""
from unittest import TestCase
from run_episode import AuditedPost

class ToolAuditTests(TestCase):
    def test_swallowed_server_error_remains_detectable(self):
        def broken(*args, **kwargs):
            raise RuntimeError('HTTP 500 from the segmentation service')
        audit=AuditedPost(broken)
        try:
            audit('/visual_overlay', {})
        except RuntimeError:
            pass  # Released perception catches this exception too.
        self.assertEqual((audit.calls,audit.errors), (1,1))

    def test_missing_mask_is_a_semantic_rejection_not_an_infrastructure_error(self):
        reply={'success':False,'message':'mask not found'}
        audit=AuditedPost(lambda *args:reply)
        self.assertIs(audit('/visual_overlay',{}),reply)
        self.assertEqual((audit.errors,audit.rejections), (0,1))
