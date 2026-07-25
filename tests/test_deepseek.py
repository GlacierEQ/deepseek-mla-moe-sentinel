"""Test suite for DeepSeek MLA MoE Sentinel solution."""
import unittest
from deepseek_mla_moe_sentinel import DeepSeekMLAMoESentinel

class TestDeepSeekMLAMoESentinel(unittest.TestCase):

    def test_mla_moe_optimization(self):
        sentinel = DeepSeekMLAMoESentinel()
        res = sentinel.optimize_mla_moe(tokens_count=16384, active_experts=8)
        
        self.assertEqual(res["status"], "MLA_MOE_OPTIMAL")
        self.assertTrue(res["memory_saved_percent"] > 90.0)

if __name__ == "__main__":
    unittest.main()
