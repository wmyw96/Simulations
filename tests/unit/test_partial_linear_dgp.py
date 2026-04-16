"""Unit tests for the partial linear model DGP."""

from __future__ import annotations

import unittest

import numpy as np

from simlab.dgp.partial_linear import PartialLinearModelUniformNoiseDGP


def linear_mu(x: np.ndarray) -> np.ndarray:
    return x[:, [0]] + 0.5 * x[:, [1]]


def linear_pi(x: np.ndarray) -> np.ndarray:
    return 0.25 * x[:, [0]] - 0.75 * x[:, [1]]


class PartialLinearModelUniformNoiseDGPTests(unittest.TestCase):
    def test_sample_shapes_without_oracle(self) -> None:
        dgp = PartialLinearModelUniformNoiseDGP(
            beta=1.5,
            func_mu=linear_mu,
            func_pi=linear_pi,
            d=2,
            sigma_u=0.2,
            sigma_eps=0.3,
        )

        sample = dgp.sample(n=25, seed=123, oracle=False)

        self.assertEqual(sample.x.shape, (25, 2))
        self.assertEqual(sample.t.shape, (25, 1))
        self.assertEqual(sample.y.shape, (25, 1))
        self.assertIsNone(sample.pi_x)
        self.assertIsNone(sample.mu_x)
        self.assertFalse(sample.metadata["oracle"])

    def test_sample_includes_oracle_outputs(self) -> None:
        dgp = PartialLinearModelUniformNoiseDGP(
            beta=2.0,
            func_mu=linear_mu,
            func_pi=linear_pi,
            d=2,
            sigma_u=0.0,
            sigma_eps=0.0,
        )

        sample = dgp.sample(n=10, seed=7, oracle=True)

        self.assertEqual(sample.pi_x.shape, (10, 1))
        self.assertEqual(sample.mu_x.shape, (10, 1))
        np.testing.assert_allclose(sample.t, sample.pi_x)
        np.testing.assert_allclose(sample.y, dgp.beta * sample.t + sample.mu_x)
        self.assertTrue(sample.metadata["oracle"])

    def test_sampling_is_reproducible_for_fixed_seed(self) -> None:
        dgp = PartialLinearModelUniformNoiseDGP(
            beta=0.5,
            func_mu=linear_mu,
            func_pi=linear_pi,
            d=2,
            sigma_u=0.4,
            sigma_eps=0.1,
        )

        sample_one = dgp.sample(n=12, seed=999, oracle=True)
        sample_two = dgp.sample(n=12, seed=999, oracle=True)

        np.testing.assert_allclose(sample_one.x, sample_two.x)
        np.testing.assert_allclose(sample_one.t, sample_two.t)
        np.testing.assert_allclose(sample_one.y, sample_two.y)
        np.testing.assert_allclose(sample_one.pi_x, sample_two.pi_x)
        np.testing.assert_allclose(sample_one.mu_x, sample_two.mu_x)

    def test_true_parameter_returns_beta(self) -> None:
        dgp = PartialLinearModelUniformNoiseDGP(
            beta=3.25,
            func_mu=linear_mu,
            func_pi=linear_pi,
            d=2,
            sigma_u=0.4,
            sigma_eps=0.1,
        )

        self.assertEqual(dgp.true_parameter(), 3.25)


if __name__ == "__main__":
    unittest.main()
