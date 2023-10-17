import numpy as np
import unittest

import jax.numpy as jnp

from model.afns import adjustment


class TestAdjustment(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.m = 1.5
        
    def test_fn_1(self):
        """Term I4 in Christensen et al. (2011)"""
        rate = 0.5
        self.assertAlmostEqual(
            2 * adjustment.fn_1(self.m, rate),
            self.m / (2 * rate) + \
                jnp.exp(-rate * self.m) / rate**2 - \
                (1 - jnp.exp(-rate * self.m)) / rate**3 / self.m
        )
            
    def test_fn_2(self):
        """Term I5 in Christensen et al. (2011)"""
        rate = 0.6
        self.assertAlmostEqual(
            2 * adjustment.fn_2(self.m, rate),
            3 * jnp.exp(-rate * self.m) / rate**2 + \
                self.m / 2 / rate + \
                jnp.exp(-rate * self.m) * self.m / rate - \
                3 * (1 - jnp.exp(-rate * self.m)) / self.m / rate**3,
            6
        )

    def test_fn_3_same_rate(self):
        """Term I2 in Christensen et al. (2011)"""
        rate = 0.7
        self.assertAlmostEqual(
            adjustment.fn_3(self.m, np.array([rate, rate])),
            1 / (2 * rate**2) - \
                (1 - jnp.exp(-rate * self.m)) / rate**3 / self.m + \
                (1 - jnp.exp(-2 * rate * self.m)) / (4 * rate**3 * self.m),
            6
        )

    def test_fn_4_same_rate(self):
        """Term I3 in Christensen et al. (2011)"""
        rate = 0.75
        self.assertAlmostEqual(
            adjustment.fn_4(self.m, np.array([rate, rate])),
            1 / (2 * rate**2) + \
                jnp.exp(-rate * self.m) / rate**2 - \
                self.m * jnp.exp(- 2 * self.m * rate) / 4 / rate - \
                3 * jnp.exp(- 2 * self.m * rate) / 4 / rate**2 - \
                2 * (1 - jnp.exp(-rate * self.m)) / rate**3 / self.m + \
                5 * (1 - jnp.exp(-2 * rate * self.m)) / (8 * rate**3 * self.m),
            6
        )
            
    def test_fn_5_same_rate(self):
        """Term I6 in Christensen et al. (2011)"""
        rate = 0.8
        self.assertAlmostEqual(
            2 * adjustment.fn_5(self.m, np.array([rate, rate])),
            (1 + jnp.exp(-rate * self.m)) / rate**2 - \
                jnp.exp(-2 * self.m * rate) / 2 / rate**2 - \
                3 * (1 - jnp.exp(-self.m * rate)) / self.m / rate**3 + \
                3 * (1 - jnp.exp(-2 * rate * self.m)) / (4 * rate**3 * self.m),
            6
        )

    def test_fn_3_diff_rate(self):
        """Term J in Christensen et al. (2009)"""
        rates = np.array([0.5, 0.75])
        self.assertAlmostEqual(
            2 * adjustment.fn_3(self.m, rates),
            1 / rates[0] / rates[1] - \
                (1 - jnp.exp(-self.m * rates[0])) / \
                    (rates[0]**2 * rates[1] * self.m) - \
                (1 - jnp.exp(-self.m * rates[1])) / \
                    (rates[0] * rates[1]**2 * self.m) + \
                (1 - jnp.exp(-self.m * jnp.sum(rates))) / \
                    (jnp.prod(rates) * jnp.sum(rates) * self.m),
            6
        )
            
    def test_fn_4_diff_rate(self):
        """Term O in Christensen et al. (2009)"""
        rates = np.array([0.5, 0.75])
        self.assertAlmostEqual(
            2 * adjustment.fn_4(self.m, rates),
            (1 + jnp.sum(jnp.exp(-self.m * rates)))/ (rates[0] * rates[1]) - \
                jnp.sum(1 / rates) / jnp.sum(rates) * \
                    jnp.exp(-self.m * jnp.sum(rates)) - \
                2 * jnp.exp(-self.m * jnp.sum(rates)) / jnp.sum(rates)**2 - \
                self.m * jnp.exp(-self.m * jnp.sum(rates)) / jnp.sum(rates) - \
                2 * (1 - jnp.exp(-self.m * rates[0])) / \
                    (rates[0]**2 * rates[1] * self.m) - \
                2 * (1 - jnp.exp(-self.m * rates[1])) / \
                    (rates[0] * rates[1]**2 * self.m) + \
                2 * (1 - jnp.exp(-self.m * jnp.sum(rates))) / \
                    jnp.sum(rates)**3 / self.m + \
                jnp.sum(1 / rates) / jnp.sum(rates)**2 * \
                    (1 - jnp.exp(-self.m * jnp.sum(rates))) / self.m + \
                (1 - jnp.exp(-self.m * jnp.sum(rates))) / \
                    (jnp.prod(rates) * jnp.sum(rates) * self.m),       
            6
        )
            
    def test_fn_5_diff_rate(self):
        """Term L in Christensen et al. (2009)
        
        There is a typo in the denominator of the final term in term L. It is
        supposed to have a tau but missing in Christensen et al. (2009).
        """
        rates = np.array([0.5, 0.75])
        self.assertAlmostEqual(
            2 * adjustment.fn_5(self.m, rates),
            (1 + jnp.exp(-rates[1] * self.m)) / jnp.prod(rates) - \
                jnp.exp(-jnp.sum(rates) * self.m) / jnp.sum(rates) / rates[0] - \
                (1 - jnp.exp(-rates[0] * self.m)) / \
                    (rates[0]**2 * rates[1] * self.m) - \
                2 * (1 - jnp.exp(-rates[1] * self.m)) / \
                    (rates[0] * rates[1]**2 * self.m) + \
                (rates[0] + 2 * rates[1]) * (1 - jnp.exp(-jnp.sum(rates) * self.m)) / \
                    (jnp.sum(rates)**2 * jnp.prod(rates) * self.m),
            6
        )
        
if __name__ == '__main__':
    unittest.main()