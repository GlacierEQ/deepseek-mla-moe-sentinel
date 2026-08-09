"""Regression tests for the deterministic MLA/MoE architecture model."""
import unittest

from deepseek_mla_moe_sentinel import DeepSeekMLAMoESentinel


class TestMLAMoEArchitectureModel(unittest.TestCase):
    def test_modeled_storage_and_expert_ratios(self):
        sentinel = DeepSeekMLAMoESentinel()
        result = sentinel.model_mla_moe(tokens_count=16_384, active_experts=8)

        self.assertGreater(result["modeled_storage_reduction_percent"], 90.0)
        self.assertEqual(result["expert_activation_percent"], 3.12)
        self.assertEqual(
            result["evidence_state"],
            "MODELED_MLA_MOE_SCENARIO_NOT_MODEL_EXECUTION",
        )
        self.assertNotIn("answer", result)
        self.assertNotIn("status", result)

    def test_historical_api_is_bounded_alias(self):
        sentinel = DeepSeekMLAMoESentinel()
        self.assertEqual(
            sentinel.optimize_mla_moe(tokens_count=32, active_experts=2),
            sentinel.model_mla_moe(tokens_count=32, active_experts=2),
        )

    def test_invalid_dimensions_and_routing_fail_closed(self):
        with self.assertRaises(ValueError):
            DeepSeekMLAMoESentinel(latent_dim=0)
        with self.assertRaises(ValueError):
            DeepSeekMLAMoESentinel(latent_dim=8, hidden_dim=4)
        sentinel = DeepSeekMLAMoESentinel(total_experts=4)
        with self.assertRaises(ValueError):
            sentinel.model_mla_moe(tokens_count=0)
        with self.assertRaises(ValueError):
            sentinel.model_mla_moe(tokens_count=1, active_experts=5)


if __name__ == "__main__":
    unittest.main()
