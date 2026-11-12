"""Tests for the Gaussian covariance-based vector HCI surrogate."""
import unittest

import numpy as np

from gaussian_vector_hci import gaussian_vector_hci


class TestGaussianVectorHCI(unittest.TestCase):
    def test_equicorrelated_scalar_and_vector(self):
        rng = np.random.default_rng(32)
        rho = 0.7
        covariance = np.full((3, 3), rho)
        np.fill_diagonal(covariance, 1.0)
        for dimension in (1, 2, 4):
            data = rng.multivariate_normal(
                np.zeros(3), covariance, size=(30000, dimension)
            )
            sources = [data[:, :, j] for j in range(3)]
            r2, r3 = gaussian_vector_hci(sources)
            theory = -dimension * 0.5 * np.log2(1 - rho ** 4)
            self.assertAlmostEqual(r3, theory, delta=0.025)
            self.assertGreaterEqual(r2 + 1e-10, r3)

    def test_markov_chain(self):
        rng = np.random.default_rng(4)
        rho = 0.8
        covariance = np.array([
            [1, rho, rho ** 2],
            [rho, 1, rho],
            [rho ** 2, rho, 1],
        ])
        data = rng.multivariate_normal(
            np.zeros(3), covariance, 50000
        )
        _, r3 = gaussian_vector_hci([data[:, i] for i in range(3)])
        theory = -0.5 * np.log2(1 - rho ** 6)
        self.assertAlmostEqual(r3, theory, delta=0.018)

    def test_two_sources(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=500)
        y = x + rng.normal(size=500)
        r2, rn = gaussian_vector_hci([x, y])
        self.assertAlmostEqual(r2, rn)

    def test_non_gaussian_accepted(self):
        rng = np.random.default_rng(6)
        x = rng.exponential(size=(1000, 2))
        y = x[:, :1] + rng.normal(size=(1000, 1))
        z = x[:, 1:] + rng.normal(size=(1000, 1))
        r2, rn = gaussian_vector_hci([x, y, z])
        self.assertTrue(np.isfinite([r2, rn]).all())
        self.assertLessEqual(rn, r2 + 1e-10)

    def test_four_sources(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(500, 4))
        r2, rn = gaussian_vector_hci(
            [x[:, j] for j in range(4)]
        )
        self.assertTrue(np.isfinite([r2, rn]).all())
        self.assertGreaterEqual(rn, 0)

    def test_invalid(self):
        x = np.ones(100)
        with self.assertRaises(ValueError):
            gaussian_vector_hci([x, x, x], ridge=0)
        with self.assertRaises(ValueError):
            gaussian_vector_hci([np.ones(100), np.ones(99)])


if __name__ == "__main__":
    unittest.main()
