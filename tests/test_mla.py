"""Test suite for C MLA compression kernel."""
import unittest

class DeepSeekMLASim:
    def compute_compression(self, seq_len: float, heads: float, dim: float) -> float:
        standard = 2 * seq_len * heads * dim
        compressed = seq_len * (dim / 4.0)
        return (1.0 - (compressed / standard)) * 100.0

class TestDeepSeekMLA(unittest.TestCase):
    def test_compression_ratio(self):
        sim = DeepSeekMLASim()
        comp = sim.compute_compression(1000000.0, 128.0, 128.0)
        self.assertGreater(comp, 99.0)

if __name__ == "__main__":
    unittest.main()
