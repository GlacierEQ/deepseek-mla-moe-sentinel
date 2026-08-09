"""Independent storage-ratio arithmetic check; this does not execute the C sources."""
import unittest


class LatentStorageRatioSim:
    def compute_reduction(self, seq_len: float, heads: float, dim: float) -> float:
        if seq_len <= 0 or heads <= 0 or dim <= 0:
            raise ValueError("dimensions must be positive")
        standard = 2 * seq_len * heads * dim
        compressed = seq_len * (dim / 4.0)
        return (1.0 - (compressed / standard)) * 100.0


class TestLatentStorageRatio(unittest.TestCase):
    def test_storage_ratio_arithmetic(self):
        sim = LatentStorageRatioSim()
        reduction = sim.compute_reduction(1_000_000.0, 128.0, 128.0)
        self.assertGreater(reduction, 99.0)

    def test_invalid_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            LatentStorageRatioSim().compute_reduction(0, 128, 128)


if __name__ == "__main__":
    unittest.main()
